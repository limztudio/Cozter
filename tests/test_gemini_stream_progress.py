"""Delta-only internal replies preserve protocol blocks and native work stays observable."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent, flexible, supervision, utils
from Cozter.backends_agent.base import Backend, append_text_result
from Cozter.backends_agent._http_process import HttpAgentProcess


class _TextBackend(Backend):
    async def launch(self, *_args, **_kwargs):
        raise AssertionError("No provider process is launched by this fixture")

    def parse_event(self, _event, _result):
        pass

    def extract_agent_text(self, event):
        if event.get("type") == "message" and event.get("role") == "assistant":
            return event.get("content")
        return None


class _DeltaBackend(_TextBackend):
    agent_text_is_delta = True


class _ExitBackend(_TextBackend):
    name = "offline-exit-observer"

    def __init__(self, proc):
        self.proc = proc
        self.result = None
        self.cleaned = False

    async def launch(self, *_args, **_kwargs):
        return self.proc

    def parse_event(self, event, result):
        self.result = result
        text = self.extract_agent_text(event)
        if text:
            append_text_result(result, text)

    async def cleanup_process(self, _proc):
        self.cleaned = True


class GeminiStreamProgressTests(unittest.IsolatedAsyncioTestCase):
    async def drain(self, backend, events, *, failure=False):
        proc = HttpAgentProcess("Offline native delta stream")

        async def produce():
            for event in events:
                proc.emit(event)
            if failure:
                raise RuntimeError("Provider failed after streaming a partial reply")

        proc.start(produce())
        return await utils.drain_llm_subprocess(proc, backend, "Offline internal delta call"), proc

    async def test_fragmented_owner_protocol_survives_metadata_repeated_chunks_and_terminal_stats(self):
        chunks = ["[OWNER]\ntier: low\nreason: ", "Keep ", "all ", "all ", "scope\ntask: Implement and verify it\n[/OWNER]"]
        events = [{"type": "init", "session_id": "irrelevant-internal-id"},
                  {"type": "message", "role": "user", "content": "Prompt echo is not an answer"}]
        events.extend({"type": "message", "role": "assistant", "content": text, "delta": True} for text in chunks)
        events.append({"type": "result", "status": "success", "stats": {"output_tokens": 50}})
        text, proc = await self.drain(_DeltaBackend(), events)
        self.assertEqual(text, "".join(chunks))
        owner = flexible.parse_plan(text, "Original task")
        self.assertTrue(owner.owner_mode)
        self.assertEqual(owner.subtasks[0].tier, "low")
        self.assertEqual(owner.subtasks[0].instruction, "Implement and verify it")
        self.assertEqual(proc.returncode, 0)

    async def test_default_replacement_backends_keep_last_answer_instead_of_concatenating_drafts(self):
        events = [{"type": "message", "role": "assistant", "content": "Partial draft"},
                  {"type": "message", "role": "assistant", "content": "Complete replacement reply"},
                  {"type": "result", "status": "success"}]
        self.assertFalse(_TextBackend().agent_text_is_delta)
        text, _proc = await self.drain(_TextBackend(), events)
        self.assertEqual(text, "Complete replacement reply")

    async def test_failed_delta_stream_discards_every_partial_chunk_and_empty_terminal_stays_empty(self):
        events = [{"type": "message", "role": "assistant", "content": "Partial "},
                  {"type": "message", "role": "assistant", "content": "reply"}]
        with self.assertLogs(level="WARNING"):
            text, proc = await self.drain(_DeltaBackend(), events, failure=True)
        self.assertEqual(text, "")
        self.assertEqual(proc.returncode, 1)
        with self.assertLogs(utils.logger, level="WARNING"):
            empty, proc = await self.drain(_DeltaBackend(), [{"type": "result", "status": "success"}])
        self.assertEqual(empty, "")
        self.assertEqual(proc.returncode, 0)

    async def test_cancelling_partial_delta_reaps_owned_stream_and_runs_backend_cleanup(self):
        proc = HttpAgentProcess("Offline cancelled native stream")
        seen = asyncio.Event()
        settled = asyncio.Event()
        backend = _DeltaBackend()
        cleanup = mock.AsyncMock()
        backend.cleanup_process = cleanup
        original_extract = backend.extract_agent_text

        def extract(event):
            text = original_extract(event)
            if text:
                seen.set()
            return text

        async def produce():
            try:
                proc.emit({"type": "message", "role": "assistant", "content": "Incomplete owner block"})
                await asyncio.Event().wait()
            finally:
                settled.set()

        proc.start(produce())
        with mock.patch.object(backend, "extract_agent_text", side_effect=extract):
            reader = asyncio.create_task(utils.drain_llm_subprocess(proc, backend, "Cancelled native stream"))
            await asyncio.wait_for(seen.wait(), 2)
            reader.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await reader
        self.assertTrue(settled.is_set())
        self.assertEqual(proc.returncode, 130)
        cleanup.assert_awaited_once_with(proc)

    def test_assistant_deltas_are_fresh_progress_while_echoes_heartbeats_and_stats_are_ignored(self):
        proc = SimpleNamespace(pid=-1)
        with (mock.patch.object(supervision, "note_output") as note,
              mock.patch.object(supervision, "attach_process") as attach):
            for event in (
                {"type": "init", "session_id": "id"}, {"type": "heartbeat"},
                {"type": "message", "role": "user", "content": "Prompt echo"},
                {"type": "message", "role": "assistant", "content": " "},
                {"type": "result", "status": "success", "stats": {"tokens": 2}},
            ):
                agent._note_native_progress(proc, event)
            note.assert_not_called()
            for content in ("same", "same", " done"):
                agent._note_native_progress(proc, {"type": "message", "role": "assistant", "content": content})
            self.assertEqual(note.call_args_list, [mock.call("same"), mock.call("same"), mock.call(" done")])
            attach.assert_not_called()

    def test_gemini_tool_ids_keep_repeated_output_distinct_and_track_actual_shell_parameters(self):
        proc = SimpleNamespace(pid=-1)
        with (mock.patch.object(supervision, "note_output") as note,
              mock.patch.object(supervision, "attach_process") as attach):
            for tool_id in ("first", "second"):
                agent._note_native_progress(proc, {
                    "type": "tool_use", "tool_id": tool_id, "tool_name": "run_shell_command",
                    "parameters": {"command": "cmake --build build --parallel 2"},
                })
                attach.assert_called_with(proc, command="cmake --build build --parallel 2")
                event = {"type": "tool_result", "tool_id": tool_id, "status": "success", "output": "OK"}
                agent._note_native_progress(proc, event)
                count = note.call_count
                agent._note_native_progress(proc, event)
                self.assertEqual(note.call_count, count)
            self.assertEqual(note.call_args_list,
                             [mock.call("OK"), mock.call("tool completed"), mock.call("OK"), mock.call("tool completed")])

    async def test_failed_exit_observer_receives_fully_drained_stderr_before_generic_error_and_default_is_noop(self):
        stdout = asyncio.StreamReader()
        stdout.feed_data(b'{"type":"message","role":"assistant","content":"Partial work retained"}\n')
        stdout.feed_eof()
        stderr = asyncio.StreamReader()
        exact_stderr = "Startup warning\nError resuming session: No previous sessions found for this project."
        stderr.feed_data((exact_stderr + "\n").encode())
        stderr.feed_eof()
        backend = _ExitBackend(SimpleNamespace(stdout=stdout, stderr=stderr, pid=-1, returncode=42))
        observed = []

        def observe(returncode, text):
            self.assertTrue(backend.cleaned)
            self.assertIsNone(backend.result.error)
            observed.append((returncode, text))
            return Backend.observe_process_exit(backend, returncode, text)

        with mock.patch.object(backend, "observe_process_exit", side_effect=observe):
            result, restarting = await agent._drive_backend_impl(backend, ".", "task", None, "full", effort=0)
        self.assertFalse(restarting)
        self.assertEqual(observed, [(42, exact_stderr)])
        self.assertEqual(result.text, "Partial work retained")
        self.assertIn("exited with code 42", result.error)
        self.assertIn(exact_stderr, result.error)

    async def test_success_cancel_and_inject_kills_never_invoke_failed_exit_observer(self):
        for mode in ("success", "cancel", "inject"):
            with self.subTest(mode=mode):
                proc = HttpAgentProcess("Offline exit lifecycle")
                backend = _ExitBackend(proc)
                seen = asyncio.Event()

                async def produce():
                    proc.emit({"type": "message", "role": "assistant", "content": "Received work"})
                    if mode != "success":
                        await asyncio.Event().wait()

                async def event(_event):
                    seen.set()

                proc.start(produce())
                queue = asyncio.Queue() if mode == "inject" else None
                with mock.patch.object(backend, "observe_process_exit") as observe:
                    reader = asyncio.create_task(agent._drive_backend_impl(
                        backend, ".", "task", None, "full", effort=0, on_event=event,
                        inject_queue=queue, injected=[] if queue is not None else None,
                    ))
                    await asyncio.wait_for(seen.wait(), 2)
                    if mode == "cancel":
                        reader.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await reader
                    else:
                        if queue is not None:
                            queue.put_nowait("New user instruction")
                        _result, restarting = await asyncio.wait_for(reader, 2)
                        self.assertEqual(restarting, mode == "inject")
                    observe.assert_not_called()
                self.assertTrue(backend.cleaned)
                self.assertEqual(proc.returncode, 0 if mode == "success" else 130)
