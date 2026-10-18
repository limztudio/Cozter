"""Reduce duplicate prompt payloads without dropping distinct scope or evidence."""

import unittest

from Cozter import agent, flexible


class PromptPayloadDedupTests(unittest.TestCase):
    def test_fallback_worker_retains_large_request_once(self):
        request = (
            "Repair every requested target, preserving the configured permissions.\n"
            + "Saved evidence: case passed; exit 0; warnings 0.\n" * 1000
            + "Remaining: commit and push; preserve this final constraint."
        )
        for read_only in (False, True):
            with self.subTest(read_only=read_only):
                task = flexible.Subtask("high", request, read_only=read_only)
                prompt = flexible.build_subtask_prompt(
                    request, flexible.Plan("", (task,)), 0, [],
                )
                self.assertEqual(prompt.count(request), 1)
                self.assertIn("Your task: the complete user request above", prompt)
                self.assertIn("including every saved obligation and constraint", prompt)
                self.assertIn("Remaining: commit and push; preserve this final constraint.", prompt)
                self.assertIn("Inspection only" if read_only else "Build/verify", prompt)

    def test_revised_task_keeps_its_distinct_scope(self):
        request = "Original exact scope and constraints.\n" * 100
        instruction = request + "Additionally check the distinct target X."
        prompt = flexible.build_subtask_prompt(
            request, flexible.Plan("", (flexible.Subtask("high", instruction),)), 0, [],
        )
        self.assertIn(f"Your task: {instruction}\n", prompt)
        self.assertEqual(prompt.count(instruction), 1)

    def test_identical_reports_keep_every_plan_mapping_with_one_evidence_copy(self):
        report = "Verified every target; command test.exe; exit 0; warnings 0.\n" * 40
        plan = flexible.Plan("", tuple(flexible.Subtask("mid", f"scope {i}") for i in range(3)))
        worker = flexible.build_subtask_prompt("all scopes", plan, 2, [report, report])
        merge = flexible.build_merge_prompt("all scopes", plan, [report, report, "final evidence"],
                                            collaborative=False, blocked=(1,))
        for prompt in (worker, merge):
            self.assertEqual(prompt.count(report), 1)
            self.assertIn("Same evidence as report 1 above (identical report).", prompt)
        self.assertIn("--- report 2 ---", worker)
        self.assertIn("plan item 2 [BLOCKED]", merge)
        self.assertIn("final evidence", merge)

    def test_reports_with_equal_bounded_ends_are_still_distinct(self):
        opening = "OPENING:" + "head " * 800
        ending = "tail " * 800 + "FINAL evidence: all requested checks passed."
        first, second = opening + "UNIQUE FAILURE A" + ending, opening + "UNIQUE FAILURE B" + ending
        self.assertEqual(flexible.truncate_report(first), flexible.truncate_report(second))
        plan = flexible.Plan("", (flexible.Subtask("mid", "one"), flexible.Subtask("high", "two")))
        for prompt in (
            flexible.build_subtask_prompt("request", plan, 1, [first, second]),
            flexible.build_merge_prompt("request", plan, [first, second], collaborative=False),
        ):
            self.assertNotIn("identical report", prompt)
            self.assertEqual(prompt.count("FINAL evidence: all requested checks passed."), 2)

    def test_previous_draft_reuses_only_evidence_actually_retained_in_prompt(self):
        draft = "Saved full verification evidence: exit 0; warning count 0.\n" * 50
        plan = flexible.Plan("", (flexible.Subtask("high", "finish remaining publication"),))
        in_request = flexible.build_subtask_prompt("Request\n" + draft, plan, 0, [], previous_draft=draft)
        in_report = flexible.build_subtask_prompt("Request", plan, 0, [draft], previous_draft=draft)
        for prompt in (in_request, in_report):
            self.assertEqual(prompt.count(draft), 1)
            self.assertIn("The same retained evidence already appears", prompt)
        distinct = flexible.build_subtask_prompt("Request", plan, 0, [draft + "NEW REMAINDER"],
                                                 previous_draft=draft + "OTHER REMAINDER")
        self.assertIn("NEW REMAINDER", distinct)
        self.assertIn("OTHER REMAINDER", distinct)

    def test_judge_continuation_request_is_exact_suffix_without_repeating_it(self):
        request = "ORIGINAL REQUEST START\n" + "requested scope\n" * 150 + "ORIGINAL REQUEST END"
        followup = "\n\n[Judge follow-up 1/3: verify the remaining final target]"
        draft = "DRAFT OPENING\n" + "retained valid verification\n" * 150 + "DRAFT FINAL EVIDENCE"
        history = "WORKSPACE HISTORY\n"
        contextual = history + "New message:\n" + request
        continuation = agent._build_continuation_prompt(
            contextual, request, draft, followup, draft_limit=12000,
        )
        self.assertTrue(continuation.endswith(request + followup))
        self.assertEqual(continuation.count(request), 1)
        self.assertEqual(continuation.count(followup), 1)
        self.assertEqual(continuation.count(draft), 1)
        self.assertIn(history, continuation)
        planner = agent._planner_context(continuation, request + followup)
        self.assertEqual(planner.count(request), 1)
        self.assertEqual(planner.count(followup), 1)
        self.assertIn("DRAFT FINAL EVIDENCE", planner)

    def test_large_judge_request_remains_complete_and_present_once(self):
        request = "SCOPE START\n" + "Saved checkpoint evidence.\n" * 3000 + "SCOPE END"
        followup = "\n\n[Judge follow-up 1/3: push the verified change]"
        continuation = agent._build_continuation_prompt(
            "memory\n" + request, request, "completed work evidence", followup, draft_limit=12000,
        )
        planner = agent._planner_context(continuation, request + followup)
        self.assertEqual(planner, request + followup)
        self.assertEqual(continuation.count(request), 1)


if __name__ == "__main__":
    unittest.main()
