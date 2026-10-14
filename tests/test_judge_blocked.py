"""Unavailable required prerequisites pause a job without completing or retrying it."""

import unittest

from Cozter import flexible


class BlockedJudgeVerdictTests(unittest.TestCase):
    def test_required_desktop_prerequisite_is_blocked_and_unverified(self):
        missing = (
            "Required interactive desktop checks remain: 301 tests skipped because "
            "the S4U scheduled task runs in Session 0 without a visible desktop."
        )
        next_instruction = "Run the task in the logged-on user's interactive desktop session."
        verdict = flexible.parse_judge(
            f"[VERDICT]\nBLOCKED\n[/VERDICT]\n[MISSING]\n{missing}\n[/MISSING]\n"
            f"[NEXT]\n{next_instruction}\n[/NEXT]\n"
        )
        self.assertTrue(verdict.blocked)
        self.assertFalse(verdict.should_continue)
        self.assertFalse(verdict.verified)
        self.assertEqual(verdict.missing, missing)
        self.assertEqual(verdict.next_instruction, next_instruction)

    def test_missing_authentication_prerequisite_can_omit_next_block(self):
        missing = "The explicitly requested provider deployment requires credentials absent from this environment."
        verdict = flexible.parse_judge(
            f"[VERDICT] blocked [/VERDICT]\n[MISSING] {missing} [/MISSING]"
        )
        self.assertTrue(verdict.blocked)
        self.assertFalse(verdict.should_continue)
        self.assertFalse(verdict.verified)
        self.assertEqual(verdict.missing, missing)
        self.assertEqual(verdict.next_instruction, "")

    def test_blocked_requires_nonempty_missing_prerequisite(self):
        for body in ("", "[MISSING][/MISSING]", "[MISSING] \n\t [/MISSING]",
                     "[NEXT]Log on to the desktop[/NEXT]"):
            with self.subTest(body=body):
                verdict = flexible.parse_judge("[VERDICT]BLOCKED[/VERDICT]\n" + body)
                self.assertFalse(verdict.blocked)
                self.assertFalse(verdict.should_continue)
                self.assertFalse(verdict.verified)

    def test_existing_done_and_continue_semantics_remain_distinct(self):
        done = flexible.parse_judge("[VERDICT]DONE[/VERDICT]")
        self.assertTrue(done.verified)
        self.assertFalse(done.blocked)
        self.assertFalse(done.should_continue)
        continued = flexible.parse_judge(
            "[VERDICT]CONTINUE[/VERDICT][MISSING]Local tests[/MISSING]"
            "[NEXT]Run the requested tests[/NEXT]"
        )
        self.assertTrue(continued.should_continue)
        self.assertTrue(continued.verified)
        self.assertFalse(continued.blocked)
        self.assertEqual(continued.next_instruction, "Run the requested tests")

    def test_malformed_decisions_do_not_infer_an_external_blocker(self):
        for raw in (None, "", "BLOCKED: no desktop", "[VERDICT]UNKNOWN[/VERDICT]",
                    "[MISSING]No desktop[/MISSING]", "[VERDICT]CONTINUE[/VERDICT]"):
            with self.subTest(raw=raw):
                verdict = flexible.parse_judge(raw)
                self.assertFalse(verdict.blocked)
                self.assertFalse(verdict.should_continue)
                self.assertFalse(verdict.verified)

    def test_existing_positional_verified_parameter_remains_compatible(self):
        verdict = flexible.JudgeVerdict(False, "missing", "next", False)
        self.assertFalse(verdict.verified)
        self.assertFalse(verdict.blocked)


class BlockedJudgePromptTests(unittest.TestCase):
    def test_prompt_distinguishes_incidental_limits_from_required_external_scope(self):
        request = "Exercise the application in a visible Windows desktop and verify its UI."
        draft = "PARTIAL: local checks passed; required UI checks cannot run in Session 0."
        prompt = flexible.build_judge_prompt(request, draft)
        self.assertIn(request, prompt)
        self.assertIn(draft, prompt)
        self.assertIn("Incidental expected skips", prompt)
        self.assertIn("outside the requested scope", prompt)
        self.assertIn("necessary part of the explicit original scope", prompt)
        self.assertIn("Session 0", prompt)
        self.assertIn("exact missing prerequisite", prompt)
        self.assertIn("BLOCKED leaves the job unfinished", prompt)
        self.assertIn("Finish every remaining independently actionable item", prompt)
        self.assertIn("Do not repeatedly recheck an unchanged unavailable prerequisite", prompt)
        self.assertIn("[VERDICT]\nBLOCKED\n[/VERDICT]", prompt)
        self.assertIn("this block is optional", prompt)

