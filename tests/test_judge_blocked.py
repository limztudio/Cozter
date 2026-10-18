"""Unavailable required prerequisites remain unfinished for automatic recovery."""

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
        self.assertIn("automatic recovery and availability rechecks with backoff", prompt)
        self.assertIn("Do not repeat completed audits, builds, or expensive work", prompt)
        self.assertIn("[VERDICT]\nBLOCKED\n[/VERDICT]", prompt)
        self.assertIn("this block is optional", prompt)

    def test_recovery_uses_existing_authorization_without_routine_approval_questions(self):
        prompt = flexible.build_judge_prompt("Finish the authorized work", "Desktop unavailable")
        self.assertIn("When full permission already authorizes the necessary action, continue without another approval", prompt)
        self.assertIn("Never ask 'can I continue' merely to resume authorized work", prompt)
        self.assertIn("autonomous recovery step or a precise availability condition to recheck", prompt)
        self.assertIn("Try authorized remediation", prompt)

    def test_missing_credentials_stay_honest_and_explicit_await_is_for_information(self):
        prompt = flexible.build_judge_prompt("Complete the deployment", "Required credentials missing")
        self.assertIn("never invent credentials or claim access that is absent", prompt)
        self.assertIn("Finish all possible work and recheck availability", prompt)
        self.assertIn("An explicit [[await]] asks for necessary information, not permission to simply continue", prompt)

    def test_merge_does_not_turn_every_blocked_report_into_a_permission_question(self):
        plan = flexible.Plan("Finish the requested checks", (flexible.Subtask("high", "verify UI"),))
        prompt = flexible.build_merge_prompt(
            "Verify UI", plan, ["BLOCKED: no interactive desktop"], collaborative=True, blocked={0},
        )
        self.assertIn("an autonomous recovery step or availability check", prompt)
        self.assertIn("Do not ask permission merely to continue routine work", prompt)
        self.assertIn('Use "[[await]]" only for specific information the user must supply', prompt)
        self.assertNotIn("A BLOCKED report needs a user answer", prompt)

