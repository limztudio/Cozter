"""Offline regressions for stalled loops and streaming request deadlines."""

import asyncio
import copy
import threading
import unittest
from unittest import mock

from Cozter.backends_agent import _openai_agent as oa
from Cozter.backends_agent.base import AgentResult
from Cozter.tests.test_llama_retry import _CaptureProc, _SSESession, _SSEResponse, _tool_call


class _LongWorkBackend(oa.OpenAIChatBackend):
    name = "latency-test"

    def _chat_endpoint(self):
        return "http://offline.invalid/chat/completions"

    def _auto_continue_after_tool_limit(self):
        return True


class ToolLoopLatencyTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, response, tool_result):
        payloads = []
        executed = []

        async def stream(_endpoint, payload, *_args, **_kwargs):
            payloads.append(copy.deepcopy(payload))
            return response(payload, len(payloads))

        async def execute(name, args, *_args):
            executed.append((name, args))
            return tool_result(args)

        proc = _CaptureProc()
        with (
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
            mock.patch.object(oa.tools, "execute_tool", side_effect=execute),
        ):
            await _LongWorkBackend()._run_agent(
                proc, ".", "work", "model", "full", False, 100,
            )
        return payloads, executed, proc.events

    async def test_repeated_only_round_forces_answer_after_five_requests(self):
        def response(payload, number):
            if "tools" not in payload:
                return "Finished from available evidence.", "", []
            return "", "", [_tool_call(f"call-{number}", "same.txt")]

        payloads, executed, events = await self._run(response, lambda _args: "read ok")
        self.assertEqual(len(payloads), 5)  # three reads, one skipped round, final
        self.assertEqual(len(executed), 3)
        self.assertNotIn("tools", payloads[-1])
        self.assertIn("no further progress", payloads[-1]["messages"][-1]["content"])
        self.assertEqual(events[-1]["text"], "Finished from available evidence.")

    async def test_changing_blocked_calls_stop_after_three_rounds(self):
        def response(payload, number):
            if "tools" not in payload:
                return "Cannot proceed: permission blocks these calls.", "", []
            return "", "", [_tool_call(f"call-{number}", f"../outside-{number}")]

        payloads, executed, events = await self._run(
            response, lambda _args: "Error: path escapes workspace: x",
        )
        self.assertEqual(len(payloads), 4)
        self.assertEqual(len(executed), 3)
        self.assertNotIn("tools", payloads[-1])
        result = AgentResult()
        backend = _LongWorkBackend()
        for event in events:
            backend.parse_event(event, result)
        self.assertIn("every tool call was refused", result.error)
        self.assertIn("Cannot proceed", result.text)

    async def test_actual_dispatcher_refusals_with_varying_paths_stop(self):
        calls = []

        async def stream(_endpoint, payload, *_args, **_kwargs):
            calls.append(copy.deepcopy(payload))
            if "tools" not in payload:
                return "Outside paths require full permission.", "", []
            return "", "", [_tool_call(str(len(calls)), f"../outside-{len(calls)}")]

        with mock.patch.object(oa, "_stream_completion", side_effect=stream):
            await _LongWorkBackend()._run_agent(
                _CaptureProc(), ".", "work", "model", "auto", False, 0,
            )
        self.assertEqual(len(calls), 4)
        self.assertNotIn("tools", calls[-1])

    def test_mixed_patch_results_are_useful_progress(self):
        self.assertTrue(oa._is_blocked_tool_result(
            "../a: skipped (path escapes workspace: ../a)\n"
            "../b: skipped (path escapes workspace: ../b)",
        ))
        self.assertFalse(oa._is_blocked_tool_result(
            "../a: skipped (path escapes workspace: ../a)\na.txt: patched",
        ))
        self.assertFalse(oa._is_blocked_tool_result("Error: old text was not found"))

    async def test_progress_resets_blocked_round_counter(self):
        results = iter(["Blocked: unavailable", "Blocked: unavailable", "read ok",
                        "Blocked: unavailable", "Blocked: unavailable", "read ok"])

        def response(_payload, number):
            if number > 6:
                return "done", "", []
            return "", "", [_tool_call(f"call-{number}", f"{number}.txt")]

        payloads, executed, _events = await self._run(response, lambda _args: next(results))
        self.assertEqual(len(executed), 6)
        self.assertTrue(all("tools" in payload for payload in payloads))

    async def test_useful_long_work_retains_auto_continuation_and_effort(self):
        def response(_payload, number):
            if number > 45:
                return "done", "", []
            return "", "", [_tool_call(f"call-{number}", f"{number}.txt")]

        payloads, executed, _events = await self._run(response, lambda _args: "new evidence")
        self.assertEqual(len(executed), 45)
        self.assertEqual(len(payloads), 46)
        self.assertTrue(all("tools" in payload for payload in payloads))
        self.assertTrue(all(payload["reasoning_effort"] == "high" for payload in payloads))

    async def test_ordinary_errors_remain_repairable(self):
        def response(_payload, number):
            if number > 5:
                return "repaired", "", []
            return "", "", [_tool_call(f"call-{number}", f"missing-{number}.txt")]

        payloads, executed, _events = await self._run(
            response, lambda _args: "Tool read_file failed: No such file or directory",
        )
        self.assertEqual(len(executed), 5)
        self.assertIn("tools", payloads[-1])


class RequestDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_share_one_total_budget(self):
        attempts = 0
        cancelled = asyncio.Event()

        async def once(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                await asyncio.sleep(0.08)
                raise oa._RetryableError("disconnected")
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with (
            mock.patch.object(oa, "_stream_once", side_effect=once),
            mock.patch.object(oa, "_backoff_delay", return_value=0),
            self.assertRaisesRegex(RuntimeError, "timed out"),
        ):
            await oa._stream_completion("http://offline", {}, {}, 0.16, 3, "test")
        self.assertEqual(attempts, 2)
        self.assertTrue(cancelled.is_set())

    async def test_heartbeat_stream_cannot_extend_total_deadline(self):
        closed = asyncio.Event()

        class Content:
            async def iter_any(self):
                while True:
                    yield b": heartbeat\n\n"
                    await asyncio.sleep(0.01)

        class Response(_SSEResponse):
            async def __aexit__(self, *_args):
                closed.set()

        response = Response([])
        response.content = Content()
        with (
            mock.patch.object(oa.aiohttp, "ClientSession", return_value=_SSESession(response)),
            self.assertRaisesRegex(RuntimeError, "timed out"),
        ):
            await oa._stream_completion("http://offline", {}, {}, 0.05, 3, "test")
        self.assertTrue(closed.is_set())

    async def test_cancel_without_total_deadline_interrupts_request(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def once(*_args, **_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with mock.patch.object(oa, "_stream_once", side_effect=once):
            task = asyncio.create_task(oa._stream_completion(
                "http://offline", {}, {}, None, 3, "test",
            ))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(cancelled.is_set())

    async def test_timings_record_first_delta_without_prompt_or_response(self):
        response = _SSEResponse([
            b'data: {"choices":[{"delta":{"reasoning_content":"private reasoning"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"private answer"}}]}\n\n',
            b'data: [DONE]\n\n',
        ])
        with (
            mock.patch.object(oa.aiohttp, "ClientSession", return_value=_SSESession(response)),
            self.assertLogs("Cozter.performance", "INFO") as captured,
        ):
            result = await oa._stream_completion(
                "http://offline", {"messages": [{"content": "private prompt"}], "model": "m"},
                {"Authorization": "private key"}, 30, 0, "test",
            )
        self.assertEqual(result[:2], ("private answer", "private reasoning"))
        logs = "\n".join(captured.output)
        self.assertIn("first_delta_seconds=", logs)
        self.assertIn("stage=http_completion", logs)
        self.assertNotIn("private", logs)


class VisionPreparationTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_preparation_runs_outside_event_loop_thread(self):
        owner_thread = threading.get_ident()
        reader_threads = []

        def prepare(*_args):
            reader_threads.append(threading.get_ident())
            return None

        with (
            mock.patch.object(oa, "_vision_parts_for_prompt", side_effect=prepare),
            mock.patch.object(oa, "_stream_completion", return_value=("done", "", [])),
        ):
            await _LongWorkBackend()._run_agent(
                _CaptureProc(), ".", "image request", "model", "full", False, 0,
            )
        self.assertEqual(len(reader_threads), 1)
        self.assertNotEqual(reader_threads[0], owner_thread)

    def test_prompt_names_actual_host_shell_when_exposed(self):
        with mock.patch.object(oa, "shell_context", return_value="Host shell: PowerShell."):
            full = oa._system_prompt(".", ("bash",), "full")
            bounded = oa._system_prompt(".", ("read_file",), "auto")
        self.assertIn("Host shell: PowerShell.", full)
        self.assertNotIn("Host shell:", bounded)


if __name__ == "__main__":
    unittest.main()
