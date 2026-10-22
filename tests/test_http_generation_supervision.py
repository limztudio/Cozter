"""Generation recovery retries retained requests without replaying completed tools."""

import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent, config, execution_state, supervision
from Cozter.backends_agent import _openai_agent as oa
from Cozter.tests.helpers import OfflineHttpBackend
from Cozter.tests.test_http_execution_continuation import _call
from Cozter.tests.test_openai_agent import _SSESession, _SSEResponse


def _confirmed():
    return {"verdict": "confirmed_hang", "evidence": "synthetic stalled request",
            "diagnosis": "offline fixture cannot progress", "repair": "retry retained request",
            "progress_paths": []}


class HttpGenerationSupervisionTests(unittest.IsolatedAsyncioTestCase):
    def fast_checks(self, *, threshold=.08):
        self.enterContext(mock.patch.object(config, "get_hang_check_interval_seconds", return_value=.01))
        self.enterContext(mock.patch.object(config, "get_hang_diagnose_after_seconds", return_value=threshold))
        self.enterContext(mock.patch.object(config, "get_hang_review_interval_seconds", return_value=.01))
        self.enterContext(mock.patch.object(config, "get_hang_review_timeout_seconds", return_value=.2))

    async def test_slow_progressing_stream_survives_and_keepalives_are_not_progress(self):
        self.fast_checks(threshold=.15)
        text = "".join(f"part-{index}" for index in range(8))
        chunks = []
        for index in range(8):
            chunks.extend([
                b": keepalive\n\n",
                b'data: {"choices":[{"delta":{}}]}\n\n',
                ("data: " + json.dumps({"choices": [{"delta": {"content": f"part-{index}"}}]}) + "\n\n").encode(),
            ])
        chunks.append(b"data: [DONE]\n\n")

        class DelayedContent:
            async def iter_any(self):
                for chunk in chunks:
                    await asyncio.sleep(.01)
                    yield chunk

        response = _SSEResponse([])
        response.content = DelayedContent()
        events = []
        note = supervision.note_output
        with (
            tempfile.TemporaryDirectory() as workspace,
            supervision.worker_scope(workspace, summary_backend_name="offline", summary_model="model"),
            mock.patch.object(oa.aiohttp, "ClientSession", return_value=_SSESession(response)),
            mock.patch.object(supervision, "_diagnose", new=mock.AsyncMock()) as doctor,
            mock.patch.object(supervision, "note_output", wraps=note) as output,
        ):
            await asyncio.wait_for(OfflineHttpBackend()._run_agent(
                SimpleNamespace(emit=events.append), workspace, "synthetic request", "model", "full", False, 50,
            ), 3)
        doctor.assert_not_awaited()
        self.assertEqual([call.args[0] for call in output.call_args_list], [f"part-{index}" for index in range(8)])
        self.assertEqual(events[-1], {"type": "assistant_text", "text": text})

    async def test_hung_generation_retries_same_transcript_without_repeating_completed_write(self):
        self.fast_checks(threshold=.02)
        requests = []
        mutations = []
        request_settled = asyncio.Event()
        events = []

        async def stream(_endpoint, payload, *_args, **_kwargs):
            requests.append(copy.deepcopy(payload))
            if len(requests) == 1:
                return "", "", [_call("write-once", "write_file")]
            if len(requests) == 2:
                try:
                    await asyncio.Event().wait()
                finally:
                    request_settled.set()
            self.assertTrue(request_settled.is_set())
            return "verified retained write", "", []

        async def mutate(_name, _args, workspace, *_unused):
            mutations.append("write")
            Path(workspace, "retained.txt").write_text("written once", encoding="utf-8")
            return "write completed"

        with (
            tempfile.TemporaryDirectory() as workspace,
            supervision.worker_scope(workspace, summary_backend_name="offline", summary_model="model"),
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
            mock.patch.object(oa.tools, "execute_tool", side_effect=mutate),
            mock.patch.object(supervision, "_diagnose", new=mock.AsyncMock(return_value=_confirmed())) as doctor,
        ):
            async with execution_state.execution_scope(workspace, "retained-job", "synthetic request"):
                async with execution_state.backend_scope("offline-http", "model", "full", 50) as record:
                    with mock.patch.object(oa, "current_run", return_value=record):
                        await asyncio.wait_for(OfflineHttpBackend()._run_agent(
                            SimpleNamespace(emit=events.append), workspace, "synthetic request", "model", "full", False, 50,
                        ), 3)
                    self.assertEqual(record.get("http_pending_results"), {})
                    self.assertEqual(Path(workspace, "retained.txt").read_text(encoding="utf-8"), "written once")
        doctor.assert_awaited_once()
        self.assertEqual(mutations, ["write"])
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[1], requests[2])
        self.assertTrue(any(message.get("content") == "write completed" for message in requests[2]["messages"]))
        self.assertEqual(events[-1]["text"], "verified retained write")

    async def test_user_cancel_settles_connection_cleanup_without_retry_or_recovery(self):
        self.fast_checks(threshold=1)
        started, cleaning, release, settled = (asyncio.Event() for _ in range(4))
        calls = []
        events = []

        async def stream(*_args, **_kwargs):
            calls.append("request")
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaning.set()
                await release.wait()
                settled.set()
                raise

        with (
            tempfile.TemporaryDirectory() as workspace,
            supervision.worker_scope(workspace, summary_backend_name="offline", summary_model="model"),
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
            mock.patch.object(supervision, "_diagnose", new=mock.AsyncMock()) as doctor,
        ):
            owner = asyncio.create_task(OfflineHttpBackend()._run_agent(
                SimpleNamespace(emit=events.append), workspace, "synthetic request", "model", "full", False, 50,
            ))
            try:
                await asyncio.wait_for(started.wait(), 1)
                owner.cancel("operator stop")
                await asyncio.wait_for(cleaning.wait(), 1)
                owner.cancel("repeated operator stop")
                await asyncio.sleep(0)
                self.assertFalse(owner.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(owner, 1)
                self.assertTrue(settled.is_set())
            finally:
                release.set()
                if not owner.done():
                    owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
        doctor.assert_not_awaited()
        self.assertEqual(calls, ["request"])
        self.assertEqual(events, [])

    async def test_repeated_generation_hangs_return_automatic_prerequisite_continuation(self):
        requests = []

        async def stream(*_args, **_kwargs):
            requests.append("request")
            await asyncio.Event().wait()

        async def recover(task, ticket):
            if ticket is None:
                await asyncio.wait((task,))
                return None
            self.assertEqual(ticket.kind, "generation")
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return "confirmed generation hang; retained request"

        with (
            tempfile.TemporaryDirectory() as workspace,
            supervision.worker_scope(workspace, summary_backend_name="offline", summary_model="model"),
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
            mock.patch.object(supervision, "wait_owned", side_effect=recover),
        ):
            result, restarting = await asyncio.wait_for(agent._drive_backend(
                OfflineHttpBackend(), workspace, "synthetic request", "model", "full", effort=50,
            ), 3)
        self.assertFalse(restarting)
        self.assertEqual(len(requests), oa._GENERATION_HANG_RECOVERY_LIMIT + 1)
        self.assertEqual(result.continuation_reason, "prerequisite")
        self.assertIn("automatically", result.continue_instruction)
        self.assertIn("unfinished", result.error)

    async def test_reasoning_advances_progress_without_exposing_opaque_provider_state(self):
        private = "opaque provider reasoning that must not enter recovery output"
        events = [
            {"choices": [{"delta": {"reasoning_content": private}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call", "function": {
                "name": "read_file", "arguments": "{}",
            }}]}}]},
        ]
        response = _SSEResponse([
            ("data: " + json.dumps(event) + "\n\n").encode() for event in events
        ] + [b"data: [DONE]\n\n"])
        with (
            mock.patch.object(supervision, "note_progress") as progress,
            mock.patch.object(supervision, "note_output") as output,
        ):
            _, reasoning, calls = await oa._stream_once(
                "http://offline.invalid", {}, {}, "offline", session=_SSESession(response),
            )
        self.assertEqual(reasoning, private)
        self.assertEqual(len(calls), 1)
        progress.assert_called_once_with()
        self.assertEqual([call.args[0] for call in output.call_args_list], ["{}"])


if __name__ == "__main__":
    unittest.main()
