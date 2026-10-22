"""Context recovery keeps durable outcomes and never replays a tool batch."""

import asyncio
import copy
import json
import tempfile
import unittest
from unittest import mock

from Cozter import execution_state
from Cozter.backends_agent import _http_context as hc, _openai_agent as oa
from Cozter.tests.helpers import OfflineHttpBackend
from Cozter.tests.test_http_execution_continuation import _call
from Cozter.tests.test_openai_agent import _EventCaptureProcess, _SSEResponse, _SSESession


def _history(waves=8, body_chars=12_000):
    messages = [{"role": "user", "content": "ORIGINAL obligations must remain exact"}]
    for index in range(waves):
        messages.extend([
            {"role": "assistant", "content": f"wave {index}", "tool_calls": [_call(str(index))]},
            {"role": "tool", "tool_call_id": str(index), "name": "read_file",
             "content": f"begin {index}:" + "evidence" * (body_chars // 8) + f":end {index}"},
        ])
    return messages


def _assert_pairs(test, messages):
    for index, message in enumerate(messages):
        calls = message.get("tool_calls")
        if calls:
            test.assertEqual([call["id"] for call in calls],
                             [row["tool_call_id"] for row in messages[index + 1:index + 1 + len(calls)]])
        if message.get("role") == "tool":
            test.assertTrue(any(call["id"] == message["tool_call_id"]
                                for row in messages[:index] for call in row.get("tool_calls", [])))


class ContextProjectionEdgeTests(unittest.TestCase):
    def test_large_unicode_tool_history_preserves_users_calls_and_exact_archive_indexes(self):
        history = _history(body_chars=2000)
        history.insert(5, {"role": "user", "content": "new guidance 漢字🙂"})
        history[2]["content"] += "漢字🙂" * 2000
        payload = {"messages": [{"role": "system", "content": "current skills"}, *history],
                   "tools": [{"function": {"parameters": {"type": "object"}}}]}
        before = copy.deepcopy(payload)
        projected = hc.project_payload(payload, 12_000, "C:/workspace/.cozter/executions/job/state.json")
        self.assertLessEqual(hc.payload_bytes(projected), 12_000)
        self.assertEqual(payload, before)
        self.assertEqual([row for row in projected["messages"] if row["role"] == "user"],
                         [row for row in payload["messages"] if row["role"] == "user"])
        self.assertEqual(projected["messages"][0], payload["messages"][0])
        _assert_pairs(self, projected["messages"])
        for row in projected["messages"]:
            if row["role"] == "tool" and "Tool result preview" in row["content"]:
                source_index = next(i for i, original in enumerate(history)
                                    if original.get("tool_call_id") == row["tool_call_id"])
                self.assertIn(f"indexes {source_index};", row["content"])

    def test_old_wave_omission_is_whole_and_reasoning_or_fixed_input_is_never_cut(self):
        history = _history(waves=12, body_chars=1200)
        history[1]["reasoning_content"] = "opaque state" * 80
        payload = {"messages": [{"role": "system", "content": "skills"}, *history]}
        projected = hc.project_payload(payload, 6500, "state.json")
        self.assertLessEqual(hc.payload_bytes(projected), 6500)
        self.assertTrue(any("Earlier HTTP evidence omitted" in (row.get("content") or "")
                            for row in projected["messages"]))
        self.assertIn(history[1], projected["messages"])
        _assert_pairs(self, projected["messages"])
        history[1]["reasoning_content"] = "opaque state" * 20_000
        with self.assertRaisesRegex(RuntimeError, "preserving all user instructions"):
            hc.project_payload(payload, 6500, "state.json")
        for fixed in (
            {"messages": [{"role": "system", "content": "x" * 10_000}]},
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "x" * 10_000}]}]},
            {"messages": [{"role": "user", "content": "exact user"}], "tools": ["x" * 10_000]},
        ):
            snapshot = copy.deepcopy(fixed)
            with self.assertRaises(RuntimeError):
                hc.project_payload(fixed, 5000, "state.json")
            self.assertEqual(fixed, snapshot)

    def test_pending_or_orphan_tool_groups_are_rejected_without_mutation(self):
        for messages in (
            [{"role": "assistant", "content": None, "tool_calls": [_call("missing")]}],
            [{"role": "tool", "tool_call_id": "orphan", "content": "x" * 5000}],
        ):
            payload = {"messages": [{"role": "system", "content": "s"}, *messages]}
            snapshot = copy.deepcopy(payload)
            with self.assertRaises(RuntimeError):
                hc.project_payload(payload, 50, "state.json")
            self.assertEqual(snapshot, payload)


class ContextRecoveryEdgeTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, workspace, prompt="ORIGINAL obligations must remain exact"):
        proc = _EventCaptureProcess()
        with (
            mock.patch.object(oa, "_system_prompt", return_value="current system and skills"),
            mock.patch.object(oa, "_tools_for_approval", return_value=[]),
        ):
            await OfflineHttpBackend()._run_agent(proc, workspace, prompt, "model", "full", False, 50)
        return proc

    async def test_disk_pending_write_and_unknown_outcome_survive_preflight_and_context_retry(self):
        with tempfile.TemporaryDirectory() as workspace:
            async with execution_state.execution_scope(workspace, "context-job", "request"):
                record = await execution_state.record("context-backend")
                with (
                    mock.patch.object(oa, "current_run", return_value=record),
                    mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock,
                                      return_value=("seed", "", [])),
                ):
                    await self._run(workspace)
                saved = _history(waves=10, body_chars=8000)
                saved.append({"role": "assistant", "content": None,
                              "tool_calls": [_call("write", "write_file"), _call("unknown", "bash")]})
                await record.save(http_messages=saved, http_pending_results={"0": "WROTE ONCE"})
                requests = []

                async def stream(_endpoint, payload, *_args, **_kwargs):
                    requests.append(copy.deepcopy(payload))
                    if len(requests) == 1:
                        raise hc.ContextLengthError("provider context rejection", 18_000)
                    return "saved write verified", "", []

                with (
                    mock.patch.object(oa, "current_run", return_value=record),
                    mock.patch.object(OfflineHttpBackend, "context_window_tokens", return_value=30_000),
                    mock.patch.object(oa, "_stream_completion", side_effect=stream),
                    mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as execute,
                ):
                    await self._run(workspace, prompt="FRESH guidance must remain exact")
                execute.assert_not_awaited()
                self.assertEqual(len(requests), 2)
                self.assertLess(hc.payload_bytes(requests[1]), hc.payload_bytes(requests[0]))
                for payload in requests:
                    _assert_pairs(self, payload["messages"])
                    self.assertIn({"role": "user", "content": "FRESH guidance must remain exact"},
                                  payload["messages"])
                    self.assertTrue(any(row.get("content") == "WROTE ONCE" for row in payload["messages"]))
                    self.assertTrue(any("outcome is unknown" in (row.get("content") or "")
                                        for row in payload["messages"]))
                disk = json.loads(record.path.read_bytes())["values"]
                self.assertEqual(disk["http_messages"][:len(saved)], saved)
                self.assertEqual(disk["http_pending_results"], {})
                self.assertTrue(any(row.get("content") == "WROTE ONCE" for row in disk["http_messages"]))

    async def test_growth_after_mutation_preflights_next_generation_without_reexecution(self):
        with tempfile.TemporaryDirectory() as workspace:
            async with execution_state.execution_scope(workspace, "growth-job", "request"):
                record = await execution_state.record("context-backend")
                with (
                    mock.patch.object(oa, "current_run", return_value=record),
                    mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock,
                                      return_value=("seed", "", [])),
                ):
                    await self._run(workspace)
                await record.save(http_messages=_history(waves=8, body_chars=3000))
                requests = []
                writes = []

                async def stream(_endpoint, payload, *_args, **_kwargs):
                    requests.append(copy.deepcopy(payload))
                    return ("", "", [_call("write", "write_file")]) if len(requests) == 1 else ("done", "", [])

                async def execute(*_args):
                    writes.append("once")
                    return "mutation durable evidence " * 60

                with (
                    mock.patch.object(oa, "current_run", return_value=record),
                    mock.patch.object(OfflineHttpBackend, "context_window_tokens", return_value=20_000),
                    mock.patch.object(oa, "_stream_completion", side_effect=stream),
                    mock.patch.object(oa.tools, "execute_tool", side_effect=execute),
                    self.assertLogs("Cozter.performance", "INFO") as captured,
                ):
                    await self._run(workspace)
                self.assertEqual(writes, ["once"])
                self.assertLessEqual(hc.payload_bytes(requests[1]), hc.context_budget(20_000))
                self.assertEqual(requests[1]["messages"][-1]["content"], "mutation durable evidence " * 60)
                self.assertEqual(record.get("http_messages")[-2]["content"], "mutation durable evidence " * 60)
                context_logs = [line for line in captured.output if "stage=http_context" in line]
                self.assertTrue(context_logs)
                self.assertNotIn("mutation durable evidence", "\n".join(context_logs))

    async def test_cancelled_reduced_generation_keeps_full_disk_history_and_no_tools(self):
        with tempfile.TemporaryDirectory() as workspace:
            async with execution_state.execution_scope(workspace, "cancel-job", "request"):
                record = await execution_state.record("context-backend")
                with (
                    mock.patch.object(oa, "current_run", return_value=record),
                    mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock, return_value=("seed", "", [])),
                ):
                    await self._run(workspace)
                saved = _history(body_chars=3000)
                await record.save(http_messages=saved)
                entered = asyncio.Event()

                async def stream(*_args, **_kwargs):
                    entered.set()
                    await asyncio.Event().wait()

                with (
                    mock.patch.object(oa, "current_run", return_value=record),
                    mock.patch.object(OfflineHttpBackend, "context_window_tokens", return_value=20_000),
                    mock.patch.object(oa, "_stream_completion", side_effect=stream),
                    mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as execute,
                ):
                    task = asyncio.create_task(self._run(workspace))
                    await asyncio.wait_for(entered.wait(), 3)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                execute.assert_not_awaited()
                self.assertEqual(json.loads(record.path.read_bytes())["values"]["http_messages"], saved)

    async def test_structured_http_and_partial_sse_context_errors_are_typed_without_tools(self):
        error = {"code": "context_length_exceeded", "message":
                 "The input (1049221 tokens) exceeds the model's context window of 1048576 tokens."}
        for status, chunks in (
            (400, [json.dumps({"error": error}).encode()]),
            (503, [json.dumps({"error": error}).encode()]),
            (200, [b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"write_file","arguments":"{"}}]}}]}\n\n',
                   b"data: " + json.dumps({"error": error}).encode() + b"\n\n"]),
            (200, [b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"write_file","arguments":"{"}}]}}]}\n\n',
                   b'data: {"choices":[{"delta":{},"finish_reason":"model_context_window_exceeded"}]}\n\n']),
        ):
            expected_window = None if b"finish_reason" in chunks[-1] else 1_048_576
            response = _SSEResponse(chunks)
            response.status = status
            with self.subTest(status=status), self.assertRaises(hc.ContextLengthError) as caught:
                await oa._post_completion_stream(_SSESession(response), "http://offline", {}, {},
                                                 oa.aiohttp.ClientTimeout(), "meta")
            self.assertEqual(caught.exception.window, expected_window)
        response = _SSEResponse([json.dumps({"error": {"code": "invalid_request_error",
                                                        "message": "arguments must be valid JSON"}}).encode()])
        response.status = 400
        with self.assertRaises(RuntimeError) as caught:
            await oa._post_completion_stream(_SSESession(response), "http://offline", {}, {},
                                             oa.aiohttp.ClientTimeout(), "meta")
        self.assertNotIsInstance(caught.exception, hc.ContextLengthError)

    async def test_configured_window_wins_and_irreducible_input_never_reaches_provider(self):
        with (
            mock.patch.object(oa.cfg, "get_model_context_window", return_value=5000),
            mock.patch.object(OfflineHttpBackend, "context_window_tokens", return_value=1_048_576) as adapter,
            mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock) as stream,
            self.assertRaisesRegex(RuntimeError, "user instructions have not been discarded"),
        ):
            await self._run(".", prompt="x" * 10_000)
        adapter.assert_not_called()
        stream.assert_not_awaited()

    async def test_unknown_window_context_retries_are_strictly_smaller_and_bounded(self):
        with tempfile.TemporaryDirectory() as workspace:
            async with execution_state.execution_scope(workspace, "retry-job", "request"):
                record = await execution_state.record("context-backend")
                with (
                    mock.patch.object(oa, "current_run", return_value=record),
                    mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock,
                                      return_value=("seed", "", [])),
                ):
                    await self._run(workspace)
                saved = _history(waves=100, body_chars=1000)
                await record.save(http_messages=saved)
                sizes = []

                async def stream(_endpoint, payload, *_args, **_kwargs):
                    sizes.append(hc.payload_bytes(payload))
                    raise hc.ContextLengthError("context window rejected without numeric capacity")

                with (
                    mock.patch.object(oa, "current_run", return_value=record),
                    mock.patch.object(OfflineHttpBackend, "context_window_tokens", return_value=None),
                    mock.patch.object(oa, "_stream_completion", side_effect=stream),
                    mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as execute,
                    self.assertRaisesRegex(RuntimeError, "3 generation-only retries"),
                ):
                    await self._run(workspace)
                self.assertEqual(len(sizes), 4)
                self.assertTrue(all(after < before for before, after in zip(sizes, sizes[1:])))
                self.assertEqual(json.loads(record.path.read_bytes())["values"]["http_messages"], saved)
                execute.assert_not_awaited()
