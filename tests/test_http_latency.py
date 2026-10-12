"""Offline regressions for stalled loops and streaming request deadlines."""

import asyncio
import copy
import json
import threading
import unittest
from unittest import mock

import aiohttp
from aiohttp import web

from Cozter.backends_agent import _openai_agent as oa
from Cozter.backends_agent.base import AgentResult
from Cozter.tests.test_openai_agent import _EventCaptureProcess, _SSESession, _SSEResponse, _tool_call


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

        proc = _EventCaptureProcess()
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
                _EventCaptureProcess(), ".", "work", "model", "auto", False, 0,
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
    async def test_compaction_stays_text_only_even_with_attachment_markers(self):
        prompt = "Summarize history: [Photo attachment saved to: image.png]"
        with (
            mock.patch.object(
                oa, "_vision_parts_for_prompt", return_value=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
            ) as prepare,
            mock.patch.object(
                oa, "_stream_completion", return_value=("summary", "", []),
            ) as stream,
        ):
            await _LongWorkBackend()._run_agent(
                _EventCaptureProcess(), ".", prompt, "model", "full", True, 0,
            )
        prepare.assert_not_called()
        payload = stream.await_args.args[1]
        self.assertEqual(payload["messages"][1]["content"], prompt)
        self.assertNotIn("tools", payload)

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
                _EventCaptureProcess(), ".", "[image attachment saved to: image.png]",
                "model", "full", False, 0,
            )
        self.assertEqual(len(reader_threads), 1)
        self.assertNotEqual(reader_threads[0], owner_thread)

    async def test_plain_text_bypasses_image_preparation(self):
        with (
            mock.patch.object(oa, "_vision_parts_for_prompt") as prepare,
            mock.patch.object(oa, "_stream_completion", return_value=("done", "", [])),
        ):
            await _LongWorkBackend()._run_agent(
                _EventCaptureProcess(), ".", "plain text", "model", "full", False, 0,
            )
        prepare.assert_not_called()

    def test_prompt_names_actual_host_shell_when_exposed(self):
        with mock.patch.object(oa, "shell_context", return_value="Host shell: PowerShell."):
            full = oa._system_prompt(".", ("bash",), "full")
            bounded = oa._system_prompt(".", ("read_file",), "auto")
        self.assertIn("Host shell: PowerShell.", full)
        self.assertNotIn("Host shell:", bounded)


class TerminalStreamLatencyTests(unittest.IsolatedAsyncioTestCase):
    async def _stream_with_stalled_tail(self, events):
        tail_read = False
        response_closed = False

        class Content:
            async def iter_any(self):
                nonlocal tail_read
                yield b": heartbeat\n\n"
                for event in events:
                    yield f"data: {json.dumps(event)}\n\n".encode()
                tail_read = True
                await asyncio.Event().wait()

            async def read(self, _size):
                await asyncio.Event().wait()

        class Response(_SSEResponse):
            async def __aexit__(self, *_args):
                nonlocal response_closed
                response_closed = True

        response = Response([])
        response.content = Content()
        with mock.patch.object(
            oa.aiohttp, "ClientSession", return_value=_SSESession(response),
        ):
            result = await oa._stream_completion(
                "http://offline", {}, {}, 0.2, 0, "test",
            )
        self.assertFalse(tail_read)
        self.assertTrue(response_closed)
        return result

    async def test_terminal_event_preserves_final_text_and_reasoning_without_done(self):
        result = await self._stream_with_stalled_tail([{
            "choices": [{"finish_reason": "stop", "delta": {
                "content": "final answer", "reasoning_content": "final reasoning",
            }}],
        }])
        self.assertEqual(result, ("final answer", "final reasoning", []))

    async def test_terminal_tool_event_preserves_last_argument_fragment(self):
        result = await self._stream_with_stalled_tail([
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call-1", "function": {
                    "name": "read_file", "arguments": '{"path":',
                },
            }]}}]},
            {"choices": [{"finish_reason": "tool_calls", "delta": {
                "content": "last text", "tool_calls": [{
                    "index": 0, "function": {"arguments": '"file.txt"}'},
                }],
            }}]},
        ])
        self.assertEqual(result[0], "last text")
        self.assertEqual(result[2][0]["function"]["arguments"], '{"path":"file.txt"}')

    async def test_terminal_event_without_delta_does_not_wait_for_tail(self):
        result = await self._stream_with_stalled_tail([
            {"choices": [{"delta": {"content": "answer"}}]},
            {"choices": [{"finish_reason": "stop"}]},
        ])
        self.assertEqual(result[0], "answer")

    async def test_nonterminal_choices_and_heartbeats_do_not_end_completion(self):
        result = await self._stream_with_stalled_tail([
            {"choices": [{"finish_reason": None, "delta": {"content": "a"}}]},
            {"choices": [{"finish_reason": "", "delta": {"content": "b"}}]},
            {"choices": []},
            {"choices": [{"delta": {"content": "c"}}, {"finish_reason": "stop"}]},
            {"choices": [{"finish_reason": "stop", "delta": {"content": "d"}}]},
        ])
        self.assertEqual(result[0], "abcd")

    async def test_provider_error_in_terminal_event_takes_priority(self):
        with self.assertRaisesRegex(RuntimeError, "quota exhausted"):
            await self._stream_with_stalled_tail([{
                "error": {"message": "quota exhausted"},
                "choices": [{"finish_reason": "stop", "delta": {"content": "partial"}}],
            }])

    async def test_pending_tool_length_terminal_remains_retryable(self):
        with self.assertRaisesRegex(oa._RetryableError, "length while a tool call was pending"):
            await self._stream_with_stalled_tail([{
                "choices": [{"finish_reason": "length", "delta": {"tool_calls": [{
                    "index": 0, "function": {
                        "name": "write_file", "arguments": '{"path":',
                    },
                }]}}],
            }])

    async def test_terminal_event_near_request_deadline_skips_trailer_grace(self):
        response = _SSEResponse([
            b'data: {"choices":[{"finish_reason":"stop","delta":{"content":"done"}}]}\n\n',
        ])
        with mock.patch.object(oa, "_COMPLETION_TAIL_DRAIN_SECONDS", 1):
            result = await oa._stream_completion(
                "http://offline", {}, {}, 0.1, 0, "test", session=_SSESession(response),
            )
        self.assertEqual(result[0], "done")

    async def test_trailer_drain_is_bounded_in_bytes(self):
        response = _SSEResponse([])
        reads = []

        class Content:
            async def read(self, size):
                reads.append(size)
                return b"x" * size

        response.content = Content()
        await oa._drain_completion_tail(response, None)
        self.assertEqual(sum(reads), oa._COMPLETION_TAIL_DRAIN_BYTES)

    async def test_cancellation_during_trailer_drain_still_cancels_request(self):
        started = asyncio.Event()
        response = _SSEResponse([
            b'data: {"choices":[{"finish_reason":"stop","delta":{"content":"done"}}]}\n\n',
        ])
        original_content = response.content

        class Content:
            def iter_any(self):
                return original_content.iter_any()

            async def read(self, _size):
                started.set()
                await asyncio.Event().wait()

        response.content = Content()
        task = asyncio.create_task(oa._stream_completion(
            "http://offline", {}, {}, None, 0, "test", session=_SSESession(response),
        ))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task


class CompletedConnectionReuseTests(unittest.IsolatedAsyncioTestCase):
    async def test_small_normal_trailer_preserves_connection_reuse(self):
        transports = set()

        async def handler(request):
            transports.add(request.transport)
            await request.read()
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(
                b'data: {"choices":[{"finish_reason":"stop","delta":{"content":"ok"}}]}\n\n',
            )
            await asyncio.sleep(0.01)
            await response.write(b'data: [DONE]\n\n')
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/chat", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            port = site._server.sockets[0].getsockname()[1]
            async with aiohttp.ClientSession() as session:
                for _ in range(2):
                    result = await oa._stream_completion(
                        f"http://127.0.0.1:{port}/chat", {}, {}, 5, 0,
                        "offline", session=session,
                    )
                    self.assertEqual(result[0], "ok")
            self.assertEqual(len(transports), 1)
        finally:
            await runner.cleanup()

    async def test_stalled_trailer_connection_is_not_reused_for_next_completion(self):
        transports = set()
        release_tail = asyncio.Event()
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            number = calls
            transports.add(request.transport)
            await request.read()
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(("data: " + json.dumps({
                "choices": [{"finish_reason": "stop", "delta": {
                    "content": f"answer-{number}",
                }}],
            }) + "\n\n").encode())
            if number == 1:
                await release_tail.wait()
            try:
                await response.write(b'data: [DONE]\n\n')
                await response.write_eof()
            except ConnectionResetError:
                pass
            return response

        app = web.Application()
        app.router.add_post("/chat", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            port = site._server.sockets[0].getsockname()[1]
            async with aiohttp.ClientSession() as session:
                for number in range(1, 3):
                    result = await oa._stream_completion(
                        f"http://127.0.0.1:{port}/chat", {}, {}, 5, 0,
                        "offline", session=session,
                    )
                    self.assertEqual(result[0], f"answer-{number}")
            self.assertEqual(len(transports), 2)
        finally:
            release_tail.set()
            await runner.cleanup()


class CompactRequestTests(unittest.TestCase):
    def test_compact_json_preserves_payload_and_escaping_with_fewer_bytes(self):
        payload = oa._completion_payload([
            {"role": "user", "content": "Unicode: 한글 \ud800\n"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c", "type": "function", "function": {
                    "name": "read_file", "arguments": '{"path": "x.txt"}',
                },
            }]},
        ], "model", {"reasoning_effort": "high"}, oa.tools.TOOL_SCHEMA)
        compact = oa._compact_json(payload)
        self.assertEqual(json.loads(compact), payload)
        self.assertLess(len(compact.encode()), len(json.dumps(payload).encode()))
        self.assertIn("\\ud800", compact)
        self.assertIn("\\ud55c", compact)

    def test_message_size_bound_matches_request_serialization(self):
        message = {"role": "tool", "name": "read_file", "content": "한글\n" * 20}
        self.assertEqual(
            oa._message_payload_bytes(message), len(oa._compact_json(message).encode()),
        )

    def test_agent_session_uses_compact_serializer_once_for_all_rounds(self):
        payloads = []

        async def stream(_endpoint, payload, *_args, **_kwargs):
            payloads.append(copy.deepcopy(payload))
            if len(payloads) == 1:
                return "", "", [_tool_call("c", "x.txt")]
            return "done", "", []

        with (
            mock.patch.object(oa.aiohttp, "ClientSession", return_value=_SSESession(
                _SSEResponse([]),
            )) as factory,
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
            mock.patch.object(oa.tools, "execute_tool", return_value="read ok"),
        ):
            asyncio.run(_LongWorkBackend()._run_agent(
                _EventCaptureProcess(), ".", "plain text", "model", "full", False, 100,
            ))
        self.assertEqual(len(payloads), 2)
        factory.assert_called_once_with(json_serialize=oa._compact_json)


if __name__ == "__main__":
    unittest.main()
