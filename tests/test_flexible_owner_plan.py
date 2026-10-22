"""Whole-job ownership survives malformed optional advice and quoted controls."""

import json
import unittest

from Cozter import flexible


class FlexibleOwnerPlanTests(unittest.TestCase):
    request = "Inspect all cache targets, repair the invariant, verify them, and preserve configured permissions."
    task = "Complete every cache target, including repair, verification, and discovered related obligations."

    def owner(self, tier="low", *, prefix="", task=None):
        return (
            prefix + f"[OWNER]\ntier: {tier}\nreason: Choose the approach for coupled invariants.\n"
            f"task: {task or self.task}\n[/OWNER]"
        )

    def advice(self, rows):
        return "[SPECIALISTS]\n" + json.dumps(rows) + "\n[/SPECIALISTS]\n"

    def assert_fallback(self, raw):
        plan = flexible.parse_plan(raw, self.request)
        self.assertTrue(plan.owner_mode)
        self.assertEqual(plan.subtasks, (flexible.Subtask("low", self.request),))
        self.assertEqual(plan.specialists, ())

    def test_upfront_high_owner_keeps_full_scope_with_two_distinct_advisors(self):
        rows = [
            {"tier": "low", "reason": "Inventory existing evidence", "task": "Read the cache target list and report its paths."},
            {"tier": "mid", "reason": "Review the established migration", "task": "Read migration interfaces and identify existing constraints."},
        ]
        plan = flexible.parse_plan(self.owner("high", prefix=self.advice(rows)), self.request)
        self.assertTrue(plan.owner_mode)
        self.assertEqual(plan.subtasks, (flexible.Subtask("high", self.task),))
        self.assertEqual(plan.owner_reason, "Choose the approach for coupled invariants.")
        self.assertEqual(plan.specialists, tuple(flexible.Lease(**row) for row in rows))
        prompt = "\n".join(flexible.build_subtask_sections(self.request, plan, 0, ["Inventory evidence retained."]).values())
        self.assertIn(self.request, prompt)
        self.assertIn(self.task, prompt)
        self.assertIn("completion owner of the whole user request", prompt)
        self.assertIn("tier label does not create a replacement writer", prompt)
        self.assertNotIn("others have their own workers", prompt)
        self.assertNotIn("Related leftovers of one consecutive tier group", prompt)
        self.assertEqual(prompt, "\n".join(flexible.build_subtask_sections(self.request, plan, 0, ["Inventory evidence retained."]).values()))

    def test_invalid_advice_is_omitted_without_replacing_or_narrowing_owner(self):
        valid = {"tier": "high", "reason": "Choose invariant", "task": "Read the invariants and compare approaches."}
        invalid_rows = [
            [valid, valid],
            [{**valid, "tier": "low"}],
            [valid, {**valid, "tier": "mid"}, {**valid, "tier": "low"}],
            [{**valid, "task": "first\nsecond"}],
            [{**valid, "task": "x" * 6000}],
            [{**valid, "reason": 3}],
            [{**valid, "extra": "field"}],
            {"high": valid},
        ]
        for rows in invalid_rows:
            with self.subTest(rows=str(rows)[:100]):
                plan = flexible.parse_plan(self.owner(prefix=self.advice(rows)), self.request)
                self.assertTrue(plan.owner_mode)
                self.assertEqual(plan.subtasks, (flexible.Subtask("low", self.task),))
                self.assertEqual(plan.specialists, ())
                self.assertIn("advice was omitted", plan.understanding)
                self.assertIn(self.request, "\n".join(flexible.build_subtask_sections(self.request, plan, 0, []).values()))

    def test_bad_json_duplicate_keys_and_fenced_advice_cannot_launch_helpers(self):
        for prefix in (
            "[SPECIALISTS]\nnot json\n[/SPECIALISTS]\n",
            '[SPECIALISTS]\n[{"tier":"low","tier":"high","reason":"review","task":"read"}]\n[/SPECIALISTS]\n',
            "```json\n" + self.advice([{"tier": "high", "reason": "review", "task": "read"}]) + "```\n",
            self.advice([]) + self.advice([]),
            "[SPECIALISTS]\n[" + " " * 12300 + "]\n[/SPECIALISTS]\n",
        ):
            with self.subTest(prefix=prefix[:100]):
                plan = flexible.parse_plan(self.owner(prefix=prefix), self.request)
                self.assertEqual(plan.specialists, ())
                self.assertIn("advice was omitted", plan.understanding)
                self.assertEqual(plan.subtasks[0].instruction, self.task)

    def test_absent_or_empty_advice_preserves_owner_and_normalizes_tiers(self):
        for prefix in ("", self.advice([])):
            plan = flexible.parse_plan(self.owner("HIGH", prefix=prefix), self.request)
            self.assertEqual(plan.subtasks, (flexible.Subtask("high", self.task),))
            self.assertEqual(plan.understanding, "")
            self.assertEqual(plan.specialists, ())

    def test_invalid_or_nonterminal_owner_falls_back_to_entire_user_request(self):
        block = self.owner("high")
        for raw in (
            block + "\nextra prose",
            "```text\n" + block + "\n```",
            block + "\n" + block,
            block.replace("tier: high", "tier: unknown"),
            block.replace("reason: Choose the approach for coupled invariants.", "reason: "),
            block.replace("task: ", "unexpected: "),
            block.replace(self.task, "x" * 6000),
            self.advice([]),
        ):
            with self.subTest(raw=raw[:100]):
                self.assert_fallback(raw)

    def test_quoted_owner_example_does_not_hide_real_terminal_owner_or_create_question(self):
        prefix = (
            "[QUESTION]\nA quoted earlier question must not replace the owner.\n[/QUESTION]\n"
            "~~~text\n" + self.owner("low") + "\n~~~\n"
        )
        plan = flexible.parse_plan(self.owner("high", prefix=prefix), self.request)
        self.assertIsNone(plan.question)
        self.assertEqual(plan.subtasks, (flexible.Subtask("high", self.task),))
        self.assertTrue(plan.owner_mode)

    def test_legacy_queue_is_not_normalized_into_one_owner(self):
        plan = flexible.parse_plan(
            "[PLAN]\n1. [low] [parallel-read] inspect cache logs\n2. [high] implement the repair\n[/PLAN]",
            self.request,
        )
        self.assertFalse(plan.owner_mode)
        self.assertEqual(len(plan.subtasks), 2)
        self.assertTrue(plan.subtasks[0].read_only)
        self.assertTrue(plan.subtasks[0].independent)
        self.assertIn("others have their own workers", "\n".join(flexible.build_subtask_sections(self.request, plan, 1, []).values()))
        self.assertEqual(flexible.Plan("", (flexible.Subtask("mid", "old saved task"),)).specialists, ())

