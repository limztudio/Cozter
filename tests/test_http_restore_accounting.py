"""Owned HTTP restoration retains exact recovery while counting history once."""

from collections import Counter
import copy
import json
import random
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter.backends_agent import _openai_agent as oa
from Cozter.tests.helpers import OfflineHttpBackend, OfflineRecord
from Cozter.tests.test_http_execution_continuation import make_tool_call


def _legacy_restore(saved, pending):
    """The previous validation/copy/recovery algorithm, as a small oracle."""
    if saved is None:
        return []
    if not isinstance(saved, list):
        raise RuntimeError("Saved HTTP transcript is invalid; preserve it for recovery.")
    messages = []
    byte_count = 0
    for message in saved:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant", "tool"}:
            raise RuntimeError("Saved HTTP transcript contains an invalid message.")
        try:
            byte_count += oa._message_payload_bytes(message)
        except (TypeError, ValueError):
            raise RuntimeError("Saved HTTP transcript is not serializable.") from None
        if byte_count > oa._MAX_AGENT_MESSAGE_BYTES:
            raise RuntimeError("Saved HTTP transcript exceeds its safety limit.")
        messages.append(copy.deepcopy(message))
    if not messages or messages[-1].get("role") != "assistant":
        return messages
    calls = messages[-1].get("tool_calls")
    if not isinstance(calls, list):
        return messages
    results = pending if isinstance(pending, dict) else {}
    for position, call in enumerate(calls):
        if not isinstance(call, dict):
            raise RuntimeError("Saved HTTP transcript contains an invalid tool call.")
        name, _ = oa.tools.parse_openai_call(call)
        result = results.get(str(position))
        if not isinstance(result, str):
            result = (
                "Interrupted: this tool batch ended before a durable result"
                " was recorded for this call. Its outcome is unknown; it may"
                " already have changed files or started a process. Inspect"
                " the actual files, process state, and existing logs before"
                " deciding what remains. Do not automatically replay this"
                " call or repeat completed side effects."
            )
        messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "name": name, "content": result})
    return messages


class HttpRestoreAccountingTests(unittest.TestCase):
    def test_owned_and_copying_restore_match_legacy_messages_and_json_bytes(self):
        generator = random.Random(120375)
        cases = [(None, None), ([], {}), ([{"role": "assistant", "content": "done"}], {})]
        contents = ["ASCII", "Unicode🙂漢字", "\x00\t\n\r", "lone\ud800surrogate", "", None]
        for _ in range(200):
            saved = [{"role": "user", "content": generator.choice(contents)}]
            for index in range(generator.randrange(4)):
                saved.append({"role": "assistant", "content": generator.choice(contents),
                              "reasoning_content": "opaque\udfff\nprovider-state"})
                saved.append({"role": "tool", "tool_call_id": f"earlier-{index}",
                              "name": "read_file", "content": generator.choice(contents)})
            calls = [make_tool_call(f"call-{index}") for index in range(generator.randrange(6))]
            saved.append({"role": "assistant", "content": None, "tool_calls": calls})
            pending = {str(index): generator.choice(contents + [False, 4, {}])
                       for index in range(len(calls)) if generator.choice([True, False])}
            cases.append((saved, pending))
        for saved, pending in cases:
            expected = _legacy_restore(saved, pending)
            owned = copy.deepcopy(saved)
            actual, byte_count = oa._restore_owned_http_messages(owned, pending)
            self.assertEqual(actual, expected)
            self.assertEqual(byte_count, sum(oa._message_payload_bytes(message) for message in expected))
            if owned:
                self.assertIs(actual[0], owned[0])

    def test_malformed_history_and_saved_history_cap_errors_match_legacy(self):
        cases = [4, {}, [None], [{"role": "system", "content": "old policy"}],
                 [{"role": "user", "content": {"non_json": object()}}],
                 [{"role": "assistant", "tool_calls": [None]}],
                 [{"role": "user", "content": "x" * 200}]]
        with mock.patch.object(oa, "_MAX_AGENT_MESSAGE_BYTES", 150):
            for saved in cases:
                with self.subTest(saved=saved):
                    with self.assertRaises(RuntimeError) as expected:
                        _legacy_restore(saved, {})
                    with self.assertRaisesRegex(RuntimeError, "^" + str(expected.exception) + "$"):
                        oa._restore_owned_http_messages(saved, {})


class HttpRestoreIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def run_backend(self, record, *, prompt="synthetic request"):
        events = []
        with mock.patch.object(oa, "current_run", return_value=record):
            await OfflineHttpBackend()._run_agent(SimpleNamespace(emit=events.append), ".", prompt,
                                       "model-a", "full", False, 50)
        return events

    async def seed_record(self):
        record = OfflineRecord()
        with mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock,
                               return_value=("seeded", "", [])):
            await self.run_backend(record)
        return record

    async def test_worker_reuses_owned_history_size_and_counts_fresh_messages_once(self):
        for prompt in ("synthetic request", "synthetic request with new guidance"):
            with self.subTest(prompt=prompt):
                record = await self.seed_record()
                saved = [{"role": "user", "content": "original evidence"},
                         {"role": "assistant", "content": None,
                          "tool_calls": [make_tool_call("known"), make_tool_call("unknown")],
                          "reasoning_content": "opaque provider state"}]
                record.values["http_messages"] = copy.deepcopy(saved)
                record.values["http_pending_results"] = {"0": "saved evidence"}
                original_history = record.values["http_messages"]
                measured = []
                payloads = []
                original_size = oa._message_payload_bytes

                def measure(message):
                    measured.append(copy.deepcopy(message))
                    return original_size(message)

                async def stream(_endpoint, payload, *_args, **_kwargs):
                    payloads.append(copy.deepcopy(payload))
                    # Mutating the provider's private request cannot reach the
                    # original record object through restored nested values.
                    payload["messages"][2]["tool_calls"][0]["id"] = "private-change"
                    self.assertEqual(original_history[1]["tool_calls"][0]["id"], "known")
                    return "verified", "", []

                with (
                    mock.patch.object(oa, "_message_payload_bytes", side_effect=measure),
                    mock.patch.object(oa, "_stream_completion", side_effect=stream),
                    mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as execute,
                ):
                    await self.run_backend(record, prompt=prompt)
                execute.assert_not_awaited()
                expected = payloads[0]["messages"] + [{"role": "assistant", "content": "verified"}]
                def key(message):
                    return json.dumps(message, sort_keys=True)
                self.assertEqual(Counter(map(key, measured)), Counter(map(key, expected)))
                self.assertEqual(payloads[0]["messages"][1:5], _legacy_restore(saved, {"0": "saved evidence"}))
                self.assertEqual(record.values["http_pending_results"], {})

    async def test_recovery_outcomes_still_trigger_full_conversation_limit_before_requests(self):
        record = await self.seed_record()
        saved = [{"role": "assistant", "content": None, "tool_calls": [make_tool_call("unknown")]}]
        record.values["http_messages"] = saved
        record.values["http_pending_results"] = {}
        system = {"role": "system", "content": "fresh policy"}
        # The saved intent fits; its generated unknown-outcome message does
        # not. This remains the worker's total-limit error, not a saved-state
        # validation error, and no model request or tool may start.
        limit = sum(oa._message_payload_bytes(message) for message in saved + [system]) + 1
        before = copy.deepcopy(record.values)
        with (
            mock.patch.object(oa, "_MAX_AGENT_MESSAGE_BYTES", limit),
            mock.patch.object(oa, "_system_prompt", return_value="fresh policy"),
            mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock) as stream,
            mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as execute,
        ):
            events = await self.run_backend(record)
        stream.assert_not_awaited()
        execute.assert_not_awaited()
        self.assertEqual(record.values, before)
        self.assertEqual([event["type"] for event in events], ["error"])
        self.assertIn("conversation exceeded", events[0]["message"])
