"""Tests for the shared OpenAI-compatible loop's transient-failure retry.

The streaming/retry client lives in backends_agent._openai_agent and is
used by both the llama and zai backends. A completion is safe to retry
because tool side effects only run after _stream_completion returns, so
dropped connections / read timeouts / HTTP 429 / 5xx are retried with
backoff. These tests stub the single-attempt request so no real server is
contacted.
"""

import asyncio
import copy
import json
import unittest
from unittest import mock

from Cozter.backends_agent import _openai_agent as oa


class OpenAIBackoffTests(unittest.TestCase):
    def test_parse_retry_after(self) -> None:
        self.assertEqual(oa._parse_retry_after("5"), 5.0)
        self.assertIsNone(oa._parse_retry_after(None))
        self.assertIsNone(oa._parse_retry_after("soon"))  # HTTP-date form

    def test_backoff_honors_retry_after_and_caps(self) -> None:
        self.assertEqual(oa._backoff_delay(1, 3.0), 3.0)
        self.assertEqual(oa._backoff_delay(9, 100.0), 10.0)  # capped
        self.assertGreaterEqual(oa._backoff_delay(3), oa._backoff_delay(1))
        self.assertLessEqual(oa._backoff_delay(20), 10.0 * 1.25)


class OpenAIRetryLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_once = oa._stream_once
        self._orig_delay = oa._backoff_delay

        def _no_delay(*args, **kwargs) -> float:
            return 0.0

        oa._backoff_delay = _no_delay

    def tearDown(self) -> None:
        oa._stream_once = self._orig_once
        oa._backoff_delay = self._orig_delay

    def _run(self, once) -> tuple:
        oa._stream_once = once
        # max_retries=2 -> initial attempt + 2 retries.
        return asyncio.run(oa._stream_completion(
            "http://x/chat/completions", {}, {}, 300, 2, "test",
        ))

    def test_retries_then_succeeds(self) -> None:
        calls = {"n": 0}

        async def once(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise oa._RetryableError("boom")
            return ("ok", [])

        self.assertEqual(self._run(once), ("ok", []))
        self.assertEqual(calls["n"], 3)

    def test_gives_up_after_max_retries(self) -> None:
        calls = {"n": 0}

        async def once(*args, **kwargs):
            calls["n"] += 1
            raise oa._RetryableError("always fails")

        with self.assertRaises(RuntimeError):
            self._run(once)
        self.assertEqual(calls["n"], 3)  # initial + 2 retries


class _SSEContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _SSEResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.status = 200
        self.headers: dict[str, str] = {}
        self.content = _SSEContent(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def text(self) -> str:
        return ""


class _SSESession:
    def __init__(self, response: _SSEResponse) -> None:
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, *args, **kwargs) -> _SSEResponse:
        return self._response


class OpenAIStreamShapeTests(unittest.TestCase):
    def test_sse_line_cap_discards_bad_line_and_keeps_following_event(self) -> None:
        async def collect() -> list[str]:
            content = _SSEContent([b"x" * 2048, b"\nvalid\n"])
            with (
                mock.patch.object(oa, "_MAX_SSE_LINE_BYTES", 1024),
                self.assertLogs(oa.logger, level="WARNING") as captured,
            ):
                lines = [line async for line in oa._iter_sse_lines(content)]
            self.assertIn("Discarding SSE line", captured.output[0])
            return lines

        self.assertEqual(asyncio.run(collect()), ["valid"])

    def _stream(self, events: list[object]) -> tuple[str, list[dict]]:
        lines = [
            f"data: {json.dumps(event)}\n".encode()
            for event in events
        ]
        lines.append(b"data: [DONE]\n")
        response = _SSEResponse([b"".join(lines)])
        session = _SSESession(response)
        with mock.patch.object(oa.aiohttp, "ClientSession", return_value=session):
            return asyncio.run(oa._stream_once(
                "http://x/chat/completions", {}, {}, 30, "test",
            ))

    def test_malformed_sse_shapes_are_ignored_without_losing_valid_deltas(
        self,
    ) -> None:
        text, tool_calls = self._stream([
            [],
            "not a completion object",
            {"choices": None},
            {"choices": {}},
            {"choices": [None]},
            {"choices": [{"delta": None}]},
            {"choices": [{"delta": "not an object"}]},
            {
                "choices": [{
                    "delta": {
                        "content": "Hello",
                        "tool_calls": {"not": "a list"},
                    },
                }],
            },
            {
                "choices": [{
                    "delta": {
                        "content": " world",
                        "tool_calls": [
                            None,
                            "not an object",
                            {"index": "not an integer"},
                            {"index": -1},
                            {"index": 0, "function": "not an object"},
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"x.txt"}',
                                },
                            },
                        ],
                    },
                }],
            },
        ])

        self.assertEqual(text, "Hello world")
        self.assertEqual(tool_calls, [{
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path":"x.txt"}',
            },
        }])

    def test_malformed_tool_call_delta_is_a_no_op(self) -> None:
        buffers: dict[int, dict[str, object]] = {}
        malformed_deltas: tuple[object, ...] = (
            None, "bad", [], {"index": True}, {"index": "0"},
        )
        for delta in malformed_deltas:
            oa._merge_tool_call(buffers, delta)
        self.assertEqual(buffers, {})


class _ToolLimitBackend(oa.OpenAIChatBackend):
    name = "limit-test"

    def __init__(self, *, auto_continue: bool) -> None:
        self.auto_continue = auto_continue

    def _chat_endpoint(self) -> str:
        return "http://x/chat/completions"

    def _auth_headers(self) -> dict[str, str]:
        return {}

    def _request_model(self, model: str | None) -> str:
        return model or "model"

    def _max_agent_turns(self) -> int:
        return 1

    def _auto_continue_after_tool_limit(self) -> bool:
        return self.auto_continue


class _CaptureProc:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: dict) -> None:
        self.events.append(event)


def _tool_call(call_id: str, path: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": json.dumps({"path": path}),
        },
    }


class OpenAIToolLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_stream = oa._stream_completion
        self._orig_execute = oa.tools.execute_tool

    def tearDown(self) -> None:
        oa._stream_completion = self._orig_stream
        oa.tools.execute_tool = self._orig_execute

    def test_auto_continue_keeps_tools_enabled_after_limit(self) -> None:
        calls: list[dict] = []

        async def stream(*args, **kwargs):
            payload = args[1]
            calls.append(copy.deepcopy(payload))
            if len(calls) <= 2:
                return "", [_tool_call(f"call-{len(calls)}", "x.txt")]
            return "done", []

        async def execute_tool(name, args, workspace_path, approval, emit):
            emit({"type": "tool_use", "name": name, "input": args})
            return f"{name} ok"

        oa._stream_completion = stream
        oa.tools.execute_tool = execute_tool

        proc = _CaptureProc()
        asyncio.run(_ToolLimitBackend(auto_continue=True)._run_agent(
            proc, "/tmp", "work", None, "auto", False, 0,
        ))

        self.assertEqual(len(calls), 3)
        self.assertTrue(all("tools" in payload for payload in calls))
        self.assertIn(
            "internal tool-call segment limit",
            calls[1]["messages"][-1]["content"],
        )
        self.assertEqual(
            [e for e in proc.events if e.get("type") == "assistant_text"],
            [{"type": "assistant_text", "text": "done"}],
        )

    def test_non_continuing_backend_still_uses_no_tools_fallback(self) -> None:
        calls: list[dict] = []

        async def stream(*args, **kwargs):
            payload = args[1]
            calls.append(copy.deepcopy(payload))
            if len(calls) == 1:
                return "", [_tool_call("call-1", "x.txt")]
            return "", []

        async def execute_tool(name, args, workspace_path, approval, emit):
            return f"{name} ok"

        oa._stream_completion = stream
        oa.tools.execute_tool = execute_tool

        proc = _CaptureProc()
        asyncio.run(_ToolLimitBackend(auto_continue=False)._run_agent(
            proc, "/tmp", "work", None, "auto", False, 0,
        ))

        self.assertEqual(len(calls), 2)
        self.assertIn("tools", calls[0])
        self.assertNotIn("tools", calls[1])
        self.assertTrue(any(
            e.get("type") == "error"
            and "exceeded 1 tool-call turns" in e.get("message", "")
            for e in proc.events
        ))


if __name__ == "__main__":
    unittest.main()
