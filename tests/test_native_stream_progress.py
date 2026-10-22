"""Complete internal replies survive teardown and native work stays observable."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent, supervision, utils
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


class _ExitBackend(_TextBackend):
    name = "offline-exit-lifecycle"

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


class NativeStreamProgressTests(unittest.IsolatedAsyncioTestCase):
    async def drain(self, backend, events, *, failure=False):
        proc = HttpAgentProcess("Offline native response stream")

        async def produce():
            for event in events:
                proc.emit(event)
            if failure:
                raise RuntimeError("Provider failed after streaming a partial reply")

        proc.start(produce())
        return await utils.drain_llm_subprocess(proc, backend, "Offline internal response call"), proc

    async def test_default_replacement_backends_keep_last_answer_instead_of_concatenating_drafts(self):
        events = [{"type": "message", "role": "assistant", "content": "Partial draft"},
                  {"type": "message", "role": "assistant", "content": "Complete replacement reply"},
                  {"type": "result", "status": "success"}]
        text, _proc = await self.drain(_TextBackend(), events)
        self.assertEqual(text, "Complete replacement reply")

    async def test_failed_stream_discards_partial_reply_and_empty_terminal_stays_empty(self):
        events = [{"type": "message", "role": "assistant", "content": "Partial "},
                  {"type": "message", "role": "assistant", "content": "reply"}]
        with self.assertLogs(level="WARNING"):
            text, proc = await self.drain(_TextBackend(), events, failure=True)
        self.assertEqual(text, "")
        self.assertEqual(proc.returncode, 1)
        with self.assertLogs(utils.logger, level="WARNING"):
            empty, proc = await self.drain(_TextBackend(), [{"type": "result", "status": "success"}])
        self.assertEqual(empty, "")
        self.assertEqual(proc.returncode, 0)

    async def test_cancelling_partial_reply_reaps_owned_stream_and_runs_backend_cleanup(self):
        proc = HttpAgentProcess("Offline cancelled native stream")
        seen = asyncio.Event()
        settled = asyncio.Event()
        backend = _TextBackend()
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

    def test_copilot_tool_ids_keep_repeated_output_distinct_and_track_actual_shell_parameters(self):
        proc = SimpleNamespace(pid=-1)
        with (mock.patch.object(supervision, "note_output") as note,
              mock.patch.object(supervision, "attach_process") as attach):
            for tool_id in ("first", "second"):
                agent._note_native_progress(proc, {
                    "type": "tool.execution_start", "data": {"toolCallId": tool_id, "toolName": "shell",
                    "arguments": {"command": "cmake --build build --parallel 2"}},
                })
                attach.assert_called_with(proc, command="cmake --build build --parallel 2")
                event = {"type": "tool.execution_complete", "data": {"toolCallId": tool_id, "result": {"content": "OK"}}}
                agent._note_native_progress(proc, event)
                count = note.call_count
                agent._note_native_progress(proc, event)
                self.assertEqual(note.call_count, count)
            self.assertEqual(note.call_args_list,
                             [mock.call("OK"), mock.call("tool completed"), mock.call("OK"), mock.call("tool completed")])

    def test_copilot_progress_rejects_retired_wrappers_and_tracks_distinct_shell_chunks(self):
        proc = SimpleNamespace(pid=-1)
        with (mock.patch.object(supervision, "note_output") as note,
              mock.patch.object(supervision, "attach_process") as attach):
            for event in (
                {"type": "tool_use", "input": {"command": "retired command"}},
                {"type": "tool.execution_start", "toolName": "retired tool"},
                {"type": "assistant.message", "content": "retired text"},
                {"type": "assistant.message", "data": {"text": "retired text"}},
            ):
                agent._note_native_progress(proc, event)
            note.assert_not_called()
            attach.assert_not_called()

            agent._note_native_progress(proc, {
                "type": "tool.execution_start", "data": {
                    "toolCallId": "shell", "toolName": "shell",
                    "arguments": {"fullCommandText": "build --verbose"},
                },
            })
            attach.assert_called_once_with(proc, command="build --verbose")
            for sequence in (0, 0, 1):
                agent._note_native_progress(proc, {
                    "type": "tool.shell_output", "data": {
                        "toolCallId": "shell", "sequence": sequence, "text": "still building",
                    },
                })
            agent._note_native_progress(proc, {
                "type": "tool.execution_progress", "data": {
                    "toolCallId": "shell", "progressMessage": "checking output",
                },
            })
            self.assertEqual(note.call_args_list, [
                mock.call("still building"), mock.call("still building"), mock.call("checking output"),
            ])

    async def test_failed_exit_keeps_partial_reply_and_drained_stderr_after_cleanup(self):
        stdout = asyncio.StreamReader()
        stdout.feed_data(b'{"type":"message","role":"assistant","content":"Partial work retained"}\n')
        stdout.feed_eof()
        stderr = asyncio.StreamReader()
        exact_stderr = "Startup warning\nProvider process failed while checking the result."
        stderr.feed_data((exact_stderr + "\n").encode())
        stderr.feed_eof()
        backend = _ExitBackend(SimpleNamespace(stdout=stdout, stderr=stderr, pid=-1, returncode=42))
        result, restarting = await agent._drive_backend_impl(backend, ".", "task", None, "full", effort=0)
        self.assertFalse(restarting)
        self.assertTrue(backend.cleaned)
        self.assertEqual(result.text, "Partial work retained")
        self.assertIn("exited with code 42", result.error)
        self.assertIn(exact_stderr, result.error)

    async def test_success_cancel_and_injection_always_clean_up_the_owned_process(self):
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
                self.assertTrue(backend.cleaned)
                self.assertEqual(proc.returncode, 0 if mode == "success" else 130)
