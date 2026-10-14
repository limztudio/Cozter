"""Planner history clipping must not duplicate or truncate durable checkpoints."""

import json
import unittest

from Cozter import agent


class PlannerContextLatencyTests(unittest.TestCase):
    def test_large_durable_request_is_forwarded_once_with_all_saved_evidence(self):
        checkpoint = {
            "reports": [
                {"instruction": f"verified scope {index}",
                 "report": f"evidence-{index}: " + "passed " * 900}
                for index in range(8)
            ],
            "pending": ["Verify final repository state and push completed changes"],
        }
        request = (
            "Original request: repair the entire workspace and push main.\n"
            "[Resume the SAME unfinished job; later queued jobs remain held.]\n"
            "[Saved progress checkpoint]:\n" + json.dumps(checkpoint)
        )
        for history in ("", "Historical memory\n" * 1000):
            with self.subTest(history=bool(history)):
                result = agent._planner_context(history + request, request)
                self.assertEqual(result, request)
                self.assertEqual(result.count("Original request:"), 1)
                recovered = json.loads(result.split("[Saved progress checkpoint]:\n", 1)[1])
                self.assertEqual(recovered, checkpoint)

    def test_medium_request_and_history_fit_the_history_cap_without_duplication(self):
        request = "CURRENT SCOPE START\n" + "requested obligation\n" * 400 + "CURRENT SCOPE END"
        history = "WORKSPACE MEMORY\n" + "old context\n" * 2500 + "LATEST PRIOR EVIDENCE\n"
        result = agent._planner_context(history + request, request)
        self.assertLessEqual(len(result), agent._PLANNER_CONTEXT_CAP)
        self.assertTrue(result.startswith("WORKSPACE MEMORY"))
        self.assertIn("LATEST PRIOR EVIDENCE", result)
        self.assertIn(agent._PLANNER_OMISSION_MARKER, result)
        self.assertTrue(result.endswith(request))
        self.assertEqual(result.count("CURRENT SCOPE START"), 1)

    def test_request_at_cap_boundaries_remains_complete(self):
        for size in (agent._PLANNER_CONTEXT_CAP - 1,
                     agent._PLANNER_CONTEXT_CAP,
                     agent._PLANNER_CONTEXT_CAP + 1):
            with self.subTest(size=size):
                request = "S" + "x" * (size - 2) + "E"
                result = agent._planner_context("history\n" * 4000 + request, request)
                self.assertEqual(result, request)

    def test_short_context_is_unchanged_and_separate_request_is_retained(self):
        self.assertEqual(agent._planner_context("history\nrequest", "request"), "history\nrequest")
        self.assertEqual(agent._planner_context("request", "request"), "request")
        self.assertEqual(agent._planner_context("history", "request"), "history\n\nrequest")
        self.assertEqual(agent._planner_context("", "request"), "request")

    def test_empty_request_still_bounds_history_with_both_ends_preserved(self):
        history = "IMPORTANT PREFIX\n" + "history\n" * 4000 + "LATEST CONTEXT"
        result = agent._planner_context(history, "")
        self.assertLessEqual(len(result), agent._PLANNER_CONTEXT_CAP)
        self.assertTrue(result.startswith("IMPORTANT PREFIX"))
        self.assertTrue(result.endswith("LATEST CONTEXT"))
        self.assertIn(agent._PLANNER_OMISSION_MARKER, result)

