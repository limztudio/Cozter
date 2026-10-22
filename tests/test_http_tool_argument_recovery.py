"""Malformed generated calls retry before tools; old poison is retained as audited history."""

import asyncio
import copy
import json
import math
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import execution_state
from Cozter.backends_agent import _openai_agent as oa
from Cozter.tests import test_openai_agent as sse_fixture
from Cozter.tests.helpers import OfflineHttpBackend


_MISSING = object()


def _call(call_id, name="read_skill", arguments="{}"):
    function = {"name": name}
    if arguments is not _MISSING:
        function["arguments"] = arguments
    return {"id": call_id, "type": "function", "function": function}


def _completion(calls=(), text=""):
    delta = {"content": text}
    if calls:
        delta["tool_calls"] = [{"index": index, **call} for index, call in enumerate(calls)]
    return [{"choices": [{"delta": delta, "finish_reason": "tool_calls" if calls else "stop"}]}]


def _response(events):
    wire = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)
    wire += b"data: [DONE]\n\n"
    return sse_fixture._SSEResponse([wire[:13], wire[13:31], wire[31:]])


def _strict_object(raw):
    def constant(value):
        raise ValueError(value)

    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(value)
        return parsed

    parsed = json.loads(raw, parse_constant=constant, parse_float=number)
    if not isinstance(parsed, dict):
        raise ValueError("arguments must be an object")
    return parsed


class _StrictMetaSession:
    def __init__(self, responses, *, before_request=None):
        self.responses = list(responses)
        self.requests = []
        self.rejected = 0
        self.before_request = before_request

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def post(self, _endpoint, *, json, **_kwargs):
        self.requests.append(copy.deepcopy(json))
        if self.before_request is not None:
            self.before_request()
        try:
            for message in json.get("messages", []):
                for call in message.get("tool_calls", []):
                    _strict_object(call["function"]["arguments"])
        except (KeyError, TypeError, ValueError):
            self.rejected += 1
            error = sse_fixture._SSEResponse([b'{"error":{"message":"Tool arguments must be valid JSON objects"}}'])
            error.status = 400
            return error
        if not self.responses:
            raise AssertionError("Unexpected provider request")
        return self.responses.pop(0)


class HttpToolArgumentRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        await self.enterAsyncContext(execution_state.execution_scope(self.ws, "tool-json-job", "Repair requested work"))
        self.record = await execution_state.record("http-tool-json")
        self.events = []
        self.enterContext(mock.patch.object(oa, "_backoff_delay", return_value=0))
        self.enterContext(mock.patch.object(oa, "_system_prompt", return_value="Fresh execution policy"))
        self.enterContext(mock.patch.object(oa, "_tools_for_approval", return_value=[
            {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}},
        ]))

    async def run_backend(self, provider, *, prompt="Repair requested work"):
        with (mock.patch.object(oa, "current_run", return_value=self.record),
              mock.patch.object(oa.aiohttp, "ClientSession", return_value=provider)):
            await OfflineHttpBackend()._run_agent(
                SimpleNamespace(emit=self.events.append), self.ws, prompt, "model-a", "full", False, 50,
            )

    async def seed_record(self, history, pending=None):
        with mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock,
                               return_value=("Initial saved response", "", [])):
            await self.run_backend(_StrictMetaSession([]))
        await self.record.save(http_messages=history, http_pending_results=pending or {})
        self.events.clear()

    async def test_malformed_sibling_discards_whole_generation_before_any_mutation_or_checkpoint(self):
        malformed = [_call("discarded-write", "write_file", '{"path":"must-not-write.txt"}'),
                     _call("discarded-skill", arguments="")]
        accepted = [_call("accepted-write", "write_file", '{"path":"written-once.txt"}')]
        provider = _StrictMetaSession([
            _response(_completion(malformed, "Discard this partial answer")),
            _response(_completion(accepted)),
            _response(_completion(text="Verified the completed mutation")),
        ])
        mutations = []

        async def execute(name, arguments, *_args):
            disk = json.loads(self.record.path.read_bytes())["values"]
            self.assertEqual(disk["http_messages"][-1]["tool_calls"], accepted)
            mutations.append((name, arguments))
            return "Mutation saved once"

        with mock.patch.object(oa.tools, "execute_tool", side_effect=execute):
            await self.run_backend(provider)

        self.assertEqual(provider.requests[0], provider.requests[1])
        self.assertEqual(mutations, [("write_file", {"path": "written-once.txt"})])
        self.assertEqual(provider.rejected, 0)
        self.assertNotIn("discarded-write", self.record.path.read_text(encoding="utf-8"))
        self.assertNotIn("Discard this partial answer", str(self.events))
        self.assertEqual([message["tool_call_id"] for message in provider.requests[-1]["messages"]
                          if message["role"] == "tool"], ["accepted-write"])

    async def test_malformed_generations_stop_after_bounded_retries_with_unchanged_safe_history(self):
        provider = _StrictMetaSession([
            _response(_completion([_call(f"bad-{index}", "write_file", "[]")])) for index in range(3)
        ])
        with mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as execute:
            with self.assertRaises(RuntimeError):
                await self.run_backend(provider)
        execute.assert_not_awaited()
        self.assertEqual(len(provider.requests), 3)
        self.assertTrue(all(request == provider.requests[0] for request in provider.requests))
        self.assertFalse(any(message.get("tool_calls") for message in self.record.get("http_messages")))

    async def test_completed_tool_batches_reject_missing_blank_nonobject_and_nonfinite_arguments(self):
        invalid = [_MISSING, "", " \t\n", "null", "[]", "true", "42", "{", '{"x":NaN}',
                   '{"x":Infinity}', '{"x":1e999}', '{"x":{"nested":[-Infinity]}}', None, 7,
                   {"x": float("inf")}]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                provider = _StrictMetaSession([_response(_completion([
                    _call("valid-sibling", "write_file", '{"path":"never.txt"}'),
                    _call("invalid-sibling", arguments=arguments),
                ]))])
                with self.assertRaises(oa._RetryableError):
                    await oa._stream_completion("http://offline.invalid", {}, {}, 0, "Meta", session=provider)
                self.assertEqual(len(provider.requests), 1)

    async def test_interleaved_fragmented_json_validates_only_after_both_calls_are_complete(self):
        events = [
            {"choices": [{"delta": {"tool_calls": [
                {"index": 1, "id": "second", "function": {"name": "read_file", "arguments": '{"path":"'}},
                {"index": 0, "id": "first", "function": {"name": "write_file", "arguments": '{"path":'}},
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '"unicode-한.txt","content":{"value":1.5}}'}},
                {"index": 1, "function": {"arguments": 'unicode-한.txt"}'}},
            ]}, "finish_reason": "tool_calls"}]},
        ]
        provider = _StrictMetaSession([_response(events)])
        _text, _reasoning, calls = await oa._stream_completion(
            "http://offline.invalid", {}, {}, 0, "Meta", session=provider,
        )
        self.assertEqual([call["id"] for call in calls], ["first", "second"])
        self.assertEqual([_strict_object(call["function"]["arguments"]) for call in calls], [
            {"path": "unicode-한.txt", "content": {"value": 1.5}}, {"path": "unicode-한.txt"},
        ])

    async def test_agent_boundary_rejects_malformed_injected_completion_before_saving_or_executing(self):
        malformed = [_call("valid", "write_file", '{"path":"never.txt"}'), _call("bad", arguments="")]
        with (mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock,
                                return_value=("Partial answer", "", malformed)),
              mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as execute):
            with self.assertRaises(RuntimeError):
                await self.run_backend(_StrictMetaSession([]))
        execute.assert_not_awaited()
        self.assertFalse(any(message.get("tool_calls") for message in self.record.get("http_messages")))
        self.assertNotIn("Partial answer", str(self.events))

    async def test_completed_skill_poison_is_repaired_before_strict_provider_and_later_mutation_runs_once(self):
        old_calls = [_call(f"old-skill-{index}", arguments="") for index in range(3)]
        old_write = _call("old-write", "write_file", '{"path":"already-written.txt"}')
        history = [{"role": "user", "content": "Original requested work"},
                   {"role": "assistant", "content": None, "tool_calls": old_calls}]
        old_outcomes = [{"role": "tool", "tool_call_id": call["id"], "name": "read_skill",
                         "content": f"Error: missing skill name {index}"} for index, call in enumerate(old_calls)]
        history.extend(old_outcomes)
        history.extend([
            {"role": "assistant", "content": None, "tool_calls": [old_write]},
            {"role": "tool", "tool_call_id": "old-write", "name": "write_file", "content": "Old mutation completed"},
            {"role": "assistant", "content": "Earlier work retained"},
        ])
        rejected = _StrictMetaSession([])
        with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
            await oa._stream_completion("http://offline.invalid", {"messages": history}, {}, 0,
                                        "Meta", session=rejected)
        self.assertEqual(rejected.rejected, 1)
        Path(self.ws, "already-written.txt").write_text("Previous completed mutation", encoding="utf-8")
        await self.seed_record(history)
        new_call = _call("new-write", "write_file", '{"path":"new-obligation.txt"}')
        durable_requests = []

        def before_request():
            disk = json.loads(self.record.path.read_bytes())["values"]
            durable_requests.append(disk)
            self.assertEqual([call["function"]["arguments"] for call in disk["http_messages"][1]["tool_calls"]],
                             ["{}", "{}", "{}"])
            self.assertEqual(len(disk["http_argument_repairs"]), 3)

        provider = _StrictMetaSession([
            _response(_completion([new_call])), _response(_completion(text="New obligation verified")),
        ], before_request=before_request)
        mutations = []

        async def execute(name, arguments, *_args):
            mutations.append((name, arguments))
            Path(self.ws, arguments["path"]).write_text("New obligation completed", encoding="utf-8")
            return "New mutation completed once"

        with mock.patch.object(oa.tools, "execute_tool", side_effect=execute):
            await self.run_backend(provider)

        self.assertEqual(provider.rejected, 0)
        self.assertEqual(mutations, [("write_file", {"path": "new-obligation.txt"})])
        self.assertEqual(Path(self.ws, "already-written.txt").read_text(encoding="utf-8"), "Previous completed mutation")
        sent = provider.requests[0]["messages"]
        self.assertEqual([message for message in sent if message["role"] == "tool"],
                         old_outcomes + [history[-2]])
        self.assertEqual(sent[6]["tool_calls"], [old_write])
        audit = self.record.get("http_argument_repairs")
        self.assertEqual([row["tool_call_id"] for row in audit], [call["id"] for call in old_calls])
        self.assertEqual([row["original_arguments"] for row in audit], ["", "", ""])
        self.assertTrue(all(row["arguments_present"] and row["arguments_were_string"] for row in audit))
        self.assertTrue(all(row["wire_arguments"] == "{}" and row["dispatcher_arguments"] == "{}" for row in audit))
        self.assertEqual(len({row["id"] for row in audit}), 3)
        self.assertTrue(all(row["http_identity"] == self.record.get("http_identity") for row in audit))
        self.assertEqual([row["message_index"] for row in audit], [1, 1, 1])
        self.assertEqual([row["call_index"] for row in audit], [0, 1, 2])
        resumed = _StrictMetaSession([_response(_completion(text="Retained work verified again"))],
                                     before_request=before_request)
        with mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as execute:
            await self.run_backend(resumed)
        execute.assert_not_awaited()
        self.assertEqual(self.record.get("http_argument_repairs"), audit)
        self.assertEqual(len([message for message in resumed.requests[0]["messages"] if message["role"] == "user"]), 2)
        self.assertEqual(len(durable_requests), 3)

    async def test_pending_bad_arguments_preserve_known_and_unknown_outcomes_without_replay(self):
        calls = [_call("known-skill", arguments=_MISSING), _call("unknown-command", "bash", "[]"),
                 _call("known-read", "read_file", '{"path":"proof.log"}')]
        history = [{"role": "user", "content": "Original pending work"},
                   {"role": "assistant", "content": None, "tool_calls": calls}]
        await self.seed_record(history, {"0": "Saved skill failure", "2": "Saved full log evidence"})
        provider = _StrictMetaSession([_response(_completion(text="Saved outcomes inspected"))])
        with mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as execute:
            await self.run_backend(provider)
        execute.assert_not_awaited()
        self.assertEqual(provider.rejected, 0)
        outcomes = [message for message in provider.requests[0]["messages"] if message["role"] == "tool"]
        self.assertEqual([message["tool_call_id"] for message in outcomes], [call["id"] for call in calls])
        self.assertEqual(outcomes[0]["content"], "Saved skill failure")
        self.assertEqual(outcomes[2]["content"], "Saved full log evidence")
        self.assertIn("outcome is unknown", outcomes[1]["content"])
        self.assertIn("Do not automatically replay", outcomes[1]["content"])
        audit = self.record.get("http_argument_repairs")
        self.assertEqual(len(audit), 2)
        self.assertFalse(audit[0]["arguments_present"])
        self.assertFalse(audit[0]["arguments_were_string"])
        self.assertEqual(audit[1]["original_arguments"], "[]")
        self.assertEqual(self.record.get("http_pending_results"), {})

    async def test_all_old_invalid_argument_shapes_have_exact_audit_and_no_historical_execution(self):
        arguments = ["", " \t", "{broken", "[]", "null", '{"nested":{"x":NaN}}', '{"x":1e999}', None]
        calls = [_call(f"old-{index}", arguments=value) for index, value in enumerate(arguments)]
        history = [{"role": "assistant", "content": None, "tool_calls": calls}]
        outcomes = [{"role": "tool", "tool_call_id": call["id"], "name": "read_skill", "content": "Old result"}
                    for call in calls]
        history.extend(outcomes)
        await self.seed_record(history)
        provider = _StrictMetaSession([_response(_completion(text="All old outcomes retained"))])
        with mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as execute:
            await self.run_backend(provider)
        execute.assert_not_awaited()
        self.assertEqual(provider.rejected, 0)
        repaired = provider.requests[0]["messages"][1]["tool_calls"]
        self.assertEqual([call["function"]["arguments"] for call in repaired], ["{}"] * len(calls))
        self.assertEqual([message for message in provider.requests[0]["messages"] if message["role"] == "tool"], outcomes)
        audit = self.record.get("http_argument_repairs")
        self.assertEqual(len(audit), len(arguments))
        self.assertEqual([row["original_arguments"] for row in audit[:7]], arguments[:7])
        self.assertFalse(audit[-1]["arguments_were_string"])

    async def test_valid_historical_object_is_canonicalized_without_losing_values_or_claiming_repair(self):
        arguments = {"path": "old-file-한.txt", "options": {"offset": 2, "flags": [True, None, 1.5]}}
        call = _call("old-dict-arguments", "read_file", arguments)
        history = [{"role": "assistant", "content": None, "tool_calls": [call]},
                   {"role": "tool", "tool_call_id": call["id"], "name": "read_file", "content": "Saved file evidence"}]
        await self.seed_record(history)

        def before_request():
            disk = json.loads(self.record.path.read_bytes())["values"]
            raw = disk["http_messages"][0]["tool_calls"][0]["function"]["arguments"]
            self.assertIsInstance(raw, str)
            self.assertEqual(_strict_object(raw), arguments)

        provider = _StrictMetaSession([_response(_completion(text="Saved object arguments retained"))],
                                      before_request=before_request)
        with mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as execute:
            await self.run_backend(provider)
        execute.assert_not_awaited()
        self.assertEqual(provider.rejected, 0)
        sent = provider.requests[0]["messages"]
        self.assertEqual(_strict_object(sent[1]["tool_calls"][0]["function"]["arguments"]), arguments)
        self.assertEqual([message for message in sent if message["role"] == "tool"], [history[1]])
        self.assertEqual(self.record.get("http_argument_repairs"), [])
        self.assertFalse(any(message["role"] == "user" for message in sent))

    async def test_cancelled_repair_save_settles_atomic_audit_before_later_resume(self):
        old_call = _call("old-empty-skill", arguments="")
        history = [{"role": "assistant", "content": None, "tool_calls": [old_call]},
                   {"role": "tool", "tool_call_id": old_call["id"], "name": "read_skill", "content": "Old result"}]
        await self.seed_record(history)
        entered = threading.Event()
        release = threading.Event()
        settled = threading.Event()
        original_write = execution_state._write_snapshot

        def block_repair(path, snapshot):
            if snapshot["values"].get("http_argument_repairs"):
                entered.set()
                if not release.wait(5):
                    raise AssertionError("Fixture did not release atomic repair")
            try:
                original_write(path, snapshot)
            finally:
                if snapshot["values"].get("http_argument_repairs"):
                    settled.set()

        provider = _StrictMetaSession([_response(_completion(text="Should not be requested before save settles"))])
        with (mock.patch.object(execution_state, "_write_snapshot", side_effect=block_repair),
              mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as execute):
            owner = asyncio.create_task(self.run_backend(provider))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                owner.cancel("first")
                await asyncio.sleep(0)
                owner.cancel("second")
                for _ in range(10):
                    await asyncio.sleep(0)
                self.assertFalse(owner.done())
                self.assertEqual(provider.requests, [])
                self.assertFalse(settled.is_set())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(owner, 3)
            finally:
                release.set()
                await asyncio.gather(owner, return_exceptions=True)
        execute.assert_not_awaited()
        self.assertTrue(settled.is_set())
        disk = json.loads(self.record.path.read_bytes())["values"]
        self.assertEqual(disk["http_messages"][0]["tool_calls"][0]["function"]["arguments"], "{}")
        self.assertEqual(disk["http_argument_repairs"][0]["original_arguments"], "")
        audit = copy.deepcopy(disk["http_argument_repairs"])
        resumed = _StrictMetaSession([_response(_completion(text="Repaired saved outcomes verified"))])
        with mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as execute:
            await self.run_backend(resumed)
        execute.assert_not_awaited()
        self.assertEqual(resumed.rejected, 0)
        self.assertEqual(self.record.get("http_argument_repairs"), audit)
