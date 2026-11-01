"""Protected context may exceed a byte estimate without exceeding token capacity."""

import copy
import unittest
from unittest import mock

from Cozter.backends_agent import _http_context as hc
from Cozter.tests.test_http_context_recovery import assert_pairs
from Cozter.tests.test_http_execution_continuation import make_tool_call


class ContextFloorEdgeTests(unittest.TestCase):
    def test_large_user_floor_retains_every_instruction_attachment_recent_pair_and_opaque_state(self):
        archive = "C:/workspace/.cozter/executions/job/state.json"
        messages = [{"role": "system", "content": "Current system and selected skills 漢字"}]
        waves = []
        for index in range(47):
            messages.append({"role": "user", "content": f"Instruction {index}: " + "exact obligation " * 1100})
            if index < 12:
                assistant = {"role": "assistant", "content": None,
                             "tool_calls": [make_tool_call(f"call-{index}")]}
                if index == 0:
                    assistant["reasoning_content"] = "opaque provider state\udfff" * 30
                tool = {"role": "tool", "tool_call_id": f"call-{index}", "name": "read_file",
                        "content": f"Outcome {index}: " + "exact tool evidence " * (10 if index == 0 else 700)}
                waves.append((len(messages), assistant, tool))
                messages.extend((assistant, tool))
        messages.append({"role": "user", "content": [
            {"type": "text", "text": "Keep this attachment instruction exact"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,EXACT-IMAGE"}},
        ]})
        payload = {"model": "offline", "messages": messages,
                   "tools": [{"type": "function", "function": {
                       "name": "read_file", "parameters": {"type": "object"},
                   }}]}
        before = copy.deepcopy(payload)
        budget = hc.context_budget(1_048_576)
        with self.assertRaises(hc.ContextProjectionError) as caught:
            hc.project_payload(payload, budget, archive)
        error = caught.exception
        minimum = error.minimum_payload
        self.assertGreater(error.minimum_bytes, budget)
        self.assertLess(error.minimum_bytes, error.original_bytes)
        self.assertEqual(error.minimum_bytes, hc.payload_bytes(minimum))
        self.assertEqual(error.original_bytes, hc.payload_bytes(payload))
        self.assertEqual(error.budget, budget)
        self.assertEqual(minimum["tools"], payload["tools"])
        self.assertEqual(minimum["model"], payload["model"])
        for role in ("system", "user"):
            self.assertEqual([row for row in minimum["messages"] if row["role"] == role],
                             [row for row in messages if row["role"] == role])
        self.assertIn(waves[0][1], minimum["messages"])
        self.assertIn(waves[0][2], minimum["messages"])
        for _, assistant, tool in waves[-2:]:
            self.assertIn(assistant, minimum["messages"])
            self.assertIn(tool, minimum["messages"])
        for _, assistant, tool in waves[1:-2]:
            self.assertNotIn(assistant, minimum["messages"])
            self.assertNotIn(tool, minimum["messages"])
        notice = next(row["content"] for row in minimum["messages"]
                      if "Earlier HTTP evidence omitted" in (row.get("content") or ""))
        ranges = ", ".join(f"{start - 1}:{start}" for start, _, _ in waves[1:-2])
        self.assertIn(archive, notice)
        self.assertIn("inclusive index ranges " + ranges + ".", notice)
        self.assertIn("Completed tools must not be replayed", notice)
        assert_pairs(self, minimum["messages"])
        self.assertEqual(payload, before)

    def test_archive_notice_cost_cannot_make_the_reported_minimum_larger(self):
        payload = {"messages": [
            {"role": "system", "content": "Current system"},
            {"role": "user", "content": "exact instructions " * 500},
            *[{"role": "assistant", "content": "tiny"} for _ in range(8)],
        ]}
        before = copy.deepcopy(payload)
        with self.assertRaises(hc.ContextProjectionError) as caught:
            hc.project_payload(payload, 100, "C:/" + "long-archive-path/" * 100)
        self.assertEqual(caught.exception.minimum_payload, payload)
        self.assertEqual(caught.exception.minimum_bytes, caught.exception.original_bytes)
        self.assertEqual(payload, before)

    def test_no_archive_and_invalid_pairs_do_not_offer_a_provider_floor(self):
        cases = [
            ({"messages": [{"role": "user", "content": "x" * 3000}]}, None),
            ({"messages": "not a message list"}, "state.json"),
            ({"messages": [{"role": "tool", "tool_call_id": "orphan", "content": "x"}]}, "state.json"),
            ({"messages": [{"role": "assistant", "tool_calls": [make_tool_call("missing")]}]}, "state.json"),
            ({"messages": [{"role": "assistant", "tool_calls": [make_tool_call("invalid")]},
                           "not a tool result"]}, "state.json"),
        ]
        for payload, archive in cases:
            before = copy.deepcopy(payload)
            with self.subTest(archive=archive), self.assertRaises(RuntimeError) as caught:
                hc.project_payload(payload, 1, archive)
            self.assertNotIsInstance(caught.exception, hc.ContextProjectionError)
            self.assertEqual(payload, before)

    def test_large_failed_projection_serializes_the_floor_and_range_notice_only_once(self):
        messages = [{"role": "system", "content": "current skills"},
                    {"role": "user", "content": "exact instructions " * 1000}]
        for index in range(1500):
            messages.extend([
                {"role": "assistant", "content": None, "tool_calls": [make_tool_call(str(index))]},
                {"role": "tool", "tool_call_id": str(index), "content": "durable outcome " * 10},
            ])
        payload = {"messages": messages}
        whole_lists = []
        notices = []
        measure = hc.payload_bytes

        def measured(value):
            if isinstance(value.get("messages"), list):
                whole_lists.append(len(value["messages"]))
            if (value.get("role") == "assistant"
                    and str(value.get("content", "")).startswith("[Earlier HTTP evidence omitted")):
                notices.append(len(value["content"]))
            return measure(value)

        with mock.patch.object(hc, "payload_bytes", side_effect=measured):
            with self.assertRaises(hc.ContextProjectionError) as caught:
                hc.project_payload(payload, 100, "state.json")
        self.assertEqual(len(whole_lists), 2)
        self.assertLessEqual(sum(whole_lists), 2 * len(messages))
        self.assertEqual(len(notices), 1)
        self.assertLess(caught.exception.minimum_bytes, caught.exception.original_bytes)
        assert_pairs(self, caught.exception.minimum_payload["messages"])

