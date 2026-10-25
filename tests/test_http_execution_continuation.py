"""Offline HTTP continuation and ordered, bounded tool execution checks."""

import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from Cozter.backends_agent import _openai_agent as oa
from Cozter import execution_state, skills
from Cozter.tests.helpers import OfflineHttpBackend, OfflineRecord


class _ReasoningBackend(OfflineHttpBackend):
    def _preserve_reasoning_content(self, _model):
        return True


def make_tool_call(call_id, name="read_file", path="file.txt"):
    return {
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps({"path": path})},
    }


def _process():
    events = []
    return SimpleNamespace(emit=events.append, events=events)


class HttpExecutionContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, backend=None, prompt="do the work", **kwargs):
        proc = _process()
        await (backend or OfflineHttpBackend())._run_agent(
            proc, kwargs.get("workspace", "/workspace"), prompt, kwargs.get("model", "model-a"),
            kwargs.get("approval", "full"), kwargs.get("compaction", False),
            kwargs.get("effort", 50),
        )
        return proc

    async def test_completed_write_survives_cancel_without_reexecution(self):
        record = OfflineRecord()
        entered_second = asyncio.Event()
        writes = []
        requests = []

        async def first_stream(*args, **kwargs):
            return "", "", [make_tool_call("write", "write_file"), make_tool_call("build", "bash")]

        async def execute(name, args, *unused):
            if name == "write_file":
                # The call intent must already exist durably before mutation.
                self.assertEqual(
                    record.values["http_messages"][-1]["tool_calls"][0]["id"],
                    "write",
                )
                writes.append(args["path"])
                return "wrote file once"
            entered_second.set()
            await asyncio.Event().wait()

        with (
            mock.patch.object(oa, "current_run", return_value=record),
            mock.patch.object(oa, "_stream_completion", side_effect=first_stream),
            mock.patch.object(oa.tools, "execute_tool", side_effect=execute),
        ):
            task = asyncio.create_task(self._run())
            await asyncio.wait_for(entered_second.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertEqual(record.values["http_pending_results"], {"0": "wrote file once"})

        async def resume_stream(*args, **kwargs):
            requests.append(copy.deepcopy(args[1]))
            return "verified the saved work", "", []

        with (
            mock.patch.object(oa, "current_run", return_value=record),
            mock.patch.object(oa, "_stream_completion", side_effect=resume_stream),
            mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as tool,
        ):
            await self._run(prompt="Continue; also inspect the new requirement.")
        tool.assert_not_awaited()
        self.assertEqual(writes, ["file.txt"])
        results = [message for message in requests[0]["messages"] if message["role"] == "tool"]
        self.assertEqual(results[0]["content"], "wrote file once")
        self.assertIn("outcome is unknown", results[1]["content"])
        self.assertIn("Do not automatically replay", results[1]["content"])
        self.assertEqual(requests[0]["messages"][-1]["content"],
                         "Continue; also inspect the new requirement.")

    async def test_completed_batch_transcript_survives_stream_interruption(self):
        record = OfflineRecord()
        count = 0

        async def stream(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                return "Inspecting", "", [make_tool_call("read")]
            raise asyncio.CancelledError

        with (
            mock.patch.object(oa, "current_run", return_value=record),
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
            mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock,
                              return_value="saved read evidence"),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await self._run()
        self.assertEqual(record.values["http_pending_results"], {})
        self.assertEqual(record.values["http_messages"][-1]["content"], "saved read evidence")
        captured = []

        async def resume(*args, **kwargs):
            captured.extend(copy.deepcopy(args[1]["messages"]))
            return "done", "", []

        with (
            mock.patch.object(oa, "current_run", return_value=record),
            mock.patch.object(oa, "_stream_completion", side_effect=resume),
        ):
            await self._run()
        self.assertEqual([message["content"] for message in captured if message["role"] == "tool"],
                         ["saved read evidence"])
        self.assertEqual(sum(message["role"] == "user" for message in captured), 1)

    async def test_real_disk_record_survives_new_execution_object(self):
        stream_calls = 0

        async def first_stream(*args, **kwargs):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls == 1:
                return "", "", [make_tool_call("write", "write_file")]
            raise asyncio.CancelledError

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output.txt"

            async def write_once(*args):
                self.assertFalse(output.exists())
                output.write_text("saved once", encoding="utf-8")
                return "wrote output.txt"

            with (
                mock.patch.object(oa, "_stream_completion", side_effect=first_stream),
                mock.patch.object(oa.tools, "execute_tool", side_effect=write_once),
            ):
                with self.assertRaises(asyncio.CancelledError):
                    async with execution_state.execution_scope(temp, "queued-job", "request"):
                        async with execution_state.backend_scope(OfflineHttpBackend.name, "model-a", "full", 50):
                            await self._run(workspace=temp)

            self.assertEqual(output.read_text(encoding="utf-8"), "saved once")
            self.assertEqual(len(list((Path(temp) / ".cozter" / "executions").rglob("state.json"))), 1)
            requests = []

            async def resume_stream(*args, **kwargs):
                requests.append(copy.deepcopy(args[1]))
                return "saved output verified", "", []

            with (
                mock.patch.object(oa, "_stream_completion", side_effect=resume_stream),
                mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as tool,
            ):
                async with execution_state.execution_scope(temp, "queued-job", "request plus accepted guidance"):
                    async with execution_state.backend_scope(OfflineHttpBackend.name, "model-a", "full", 50):
                        await self._run(workspace=temp, prompt="Continue and verify the existing output.")
            tool.assert_not_awaited()
            self.assertTrue(any(message.get("content") == "wrote output.txt"
                                for message in requests[0]["messages"]))

    async def test_reasoning_state_is_retained_but_current_system_is_rebuilt(self):
        record = OfflineRecord()
        calls = 0

        async def first(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return "", "opaque provider state", [make_tool_call("read")]
            raise asyncio.CancelledError

        with (
            mock.patch.object(oa, "current_run", return_value=record),
            mock.patch.object(oa, "_stream_completion", side_effect=first),
            mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock,
                              return_value="evidence"),
            mock.patch("Cozter.skills.catalog_prompt", return_value=""),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await self._run(_ReasoningBackend())
        captured = []

        async def resumed(*args, **kwargs):
            captured.extend(copy.deepcopy(args[1]["messages"]))
            return "done", "", []

        with (
            mock.patch.object(oa, "current_run", return_value=record),
            mock.patch.object(oa, "_system_prompt", return_value="fresh policy"),
            mock.patch.object(oa, "_stream_completion", side_effect=resumed),
            mock.patch("Cozter.skills.catalog_prompt", return_value=""),
        ):
            proc = await self._run(_ReasoningBackend())
        self.assertEqual(captured[0]["role"], "system")
        self.assertTrue(captured[0]["content"].startswith("fresh policy\n\n"))
        self.assertIn(skills.refresh_instructions(), captured[0]["content"])
        self.assertIn("No skills are currently discoverable", captured[0]["content"])
        self.assertTrue(any(message.get("reasoning_content") == "opaque provider state" for message in captured))
        self.assertNotIn("opaque provider state", str(proc.events))

    async def test_model_or_permission_change_does_not_reuse_transcript(self):
        for changed in ({"model": "model-b"}, {"approval": "confirm"}):
            with self.subTest(changed=changed):
                record = OfflineRecord()
                seen = []

                async def stream(*args, **kwargs):
                    seen.append(copy.deepcopy(args[1]))
                    return "completed private phase", "", []

                with (
                    mock.patch.object(oa, "current_run", return_value=record),
                    mock.patch.object(oa, "_stream_completion", side_effect=stream),
                ):
                    await self._run()
                    await self._run(**changed)
                self.assertEqual(len(seen[1]["messages"]), 2)
                self.assertNotIn("completed private phase", str(seen[1]))

    async def test_compaction_never_reads_or_writes_execution_record(self):
        record = OfflineRecord()
        with (
            mock.patch.object(oa, "current_run", return_value=record) as current,
            mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock,
                              return_value=("compacted", "", [])) as stream,
        ):
            await self._run(compaction=True)
        current.assert_not_called()
        self.assertEqual(record.values, {})
        self.assertNotIn("tools", stream.call_args.args[1])

    async def test_read_batches_are_bounded_and_writes_are_ordered_barriers(self):
        calls = [make_tool_call(str(index), path=f"before-{index}") for index in range(10)]
        calls += [make_tool_call("write", "write_file", "mutation")]
        calls += [make_tool_call(f"after-{index}", path=f"after-{index}") for index in range(3)]
        requests = []
        active = peak = 0
        written = False

        async def stream(*args, **kwargs):
            requests.append(copy.deepcopy(args[1]))
            return ("", "", calls) if len(requests) == 1 else ("done", "", [])

        async def execute(name, args, *unused):
            nonlocal active, peak, written
            if name == "write_file":
                self.assertEqual(active, 0)
                written = True
                return "written"
            self.assertEqual(written, args["path"].startswith("after-"))
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
                return args["path"]
            finally:
                active -= 1

        with (
            mock.patch.object(oa, "current_run", return_value=None),
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
            mock.patch.object(oa.tools, "execute_tool", side_effect=execute),
        ):
            await self._run()
        self.assertEqual(peak, oa._PARALLEL_READ_TOOLS)
        self.assertEqual([message["tool_call_id"] for message in requests[1]["messages"] if message["role"] == "tool"],
                         [call["id"] for call in calls])

    async def test_read_wave_uses_one_complete_checkpoint_instead_of_one_per_read(self):
        record = OfflineRecord()
        calls = [make_tool_call(str(index), path=str(index)) for index in range(4)]
        requests = 0

        async def stream(*args, **kwargs):
            nonlocal requests
            requests += 1
            return ("", "", calls) if requests == 1 else ("done", "", [])

        async def read(*args):
            self.assertEqual(len(record.values["http_messages"][-1]["tool_calls"]), 4)
            await asyncio.sleep(0)
            return "read evidence"

        with (
            mock.patch.object(oa, "current_run", return_value=record),
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
            mock.patch.object(oa.tools, "execute_tool", side_effect=read),
        ):
            await self._run()
        # Initial request, durable call intent, complete ordered results, and
        # final answer. A read's completion no longer rewrites the full history.
        self.assertEqual(len(record.snapshots), 4)
        self.assertEqual(sum(message["role"] == "tool" for message in record.values["http_messages"]), 4)

    async def test_interrupted_read_wave_preserves_finished_read_and_settles_sibling(self):
        record = OfflineRecord()
        waiting = asyncio.Event()
        stopped = asyncio.Event()

        async def read(name, args, *unused):
            if args["path"] == "finished":
                return "completed read evidence"
            waiting.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with (
            mock.patch.object(oa, "current_run", return_value=record),
            mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock,
                              return_value=("", "", [make_tool_call("one", path="finished"),
                                                       make_tool_call("two", path="waiting")])),
            mock.patch.object(oa.tools, "execute_tool", side_effect=read),
        ):
            task = asyncio.create_task(self._run())
            await asyncio.wait_for(waiting.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(stopped.is_set())
        self.assertEqual(record.values["http_pending_results"], {"0": "completed read evidence"})
        restored, _ = oa._restore_owned_http_messages(record.values["http_messages"],
                                                      record.values["http_pending_results"])
        self.assertEqual(restored[-2]["content"], "completed read evidence")
        self.assertIn("outcome is unknown", restored[-1]["content"])

    async def test_read_wave_and_each_mutation_are_durable_before_later_mutations(self):
        record = OfflineRecord()
        calls = [make_tool_call(str(index), path=str(index)) for index in range(4)]
        calls.extend([make_tool_call("write", "write_file"), make_tool_call("build", "bash")])
        requests = 0

        async def stream(*args, **kwargs):
            nonlocal requests
            requests += 1
            return ("", "", calls) if requests == 1 else ("done", "", [])

        async def execute(name, args, *unused):
            if name == "write_file":
                self.assertEqual(record.values["http_pending_results"],
                                 {str(index): "read evidence" for index in range(4)})
                return "mutation completed"
            if name == "bash":
                self.assertEqual(record.values["http_pending_results"]["4"],
                                 "mutation completed")
                return "build completed"
            return "read evidence"

        with (
            mock.patch.object(oa, "current_run", return_value=record),
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
            mock.patch.object(oa.tools, "execute_tool", side_effect=execute),
        ):
            await self._run()

    def test_plugin_shadowing_read_name_never_enables_parallel_execution(self):
        with mock.patch.dict(oa.tools.TOOLS_BY_NAME, {"read_file": SimpleNamespace(is_plugin=True)}):
            self.assertFalse(oa._parallel_read_tool("read_file"))
        self.assertFalse(oa._parallel_read_tool("unknown_plugin"))

    async def test_failed_read_batch_settles_siblings_before_worker_exit(self):
        sibling_started = asyncio.Event()
        sibling_stopped = asyncio.Event()

        async def execute(name, args, *unused):
            if args["path"] == "fails":
                await sibling_started.wait()
                raise OSError("checkpoint storage failed")
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                sibling_stopped.set()

        with (
            mock.patch.object(oa, "current_run", return_value=None),
            mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock,
                              return_value=("", "", [make_tool_call("one", path="fails"),
                                                       make_tool_call("two", path="waits")])),
            mock.patch.object(oa.tools, "execute_tool", side_effect=execute),
        ):
            with self.assertRaisesRegex(OSError, "checkpoint storage failed"):
                await self._run()
        self.assertTrue(sibling_stopped.is_set())

    def test_invalid_saved_transcript_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "invalid message"):
            oa._restore_owned_http_messages([{"role": "system", "content": "old policy"}], {})
        with mock.patch.object(oa, "_MAX_AGENT_MESSAGE_BYTES", 10):
            with self.assertRaisesRegex(RuntimeError, "safety limit"):
                oa._restore_owned_http_messages([{"role": "user", "content": "too long"}], {})
