"""Structured worker context retains rendering and role boundaries."""

import hashlib
import unittest

from Cozter import flexible


class FlexiblePromptSectionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = "Request with [image attachment saved to: images/a.png]"
        self.plan = flexible.Plan("Inspect then repair", (
            flexible.Subtask("low", "inspect existing logs", read_only=True, independent=True),
            flexible.Subtask("high", "repair every confirmed finding"),
        ))

    def test_sections_join_to_the_canonical_complete_prompt(self) -> None:
        # Snapshots include the tier policy and terminal control messages.
        # Both roles still join exactly, including repeated evidence.
        cases = (
            (0, [], None, "7b5fa63658dce85e68539a776527f2768148aedae4a584d27d34424fcc870351"),
            (1, ["report A", "report B"], "Previous verified result",
             "d5a0a6c81c44efb007f482da1f519a11f6217adb6be2051ab48ef6d5de6aa187"),
            (0, ["a" * 180, "a" * 180], "a" * 180,
             "795e6b88f0942658443320edae419ae00bf12c2254e270211dac5650e062db1f"),
        )
        for index, reports, draft, expected_hash in cases:
            with self.subTest(index=index, reports=len(reports), draft=draft is not None):
                sections = flexible.build_subtask_sections(
                    self.request, self.plan, index, reports, previous_draft=draft,
                )
                rendered = "\n".join(sections.values())
                self.assertEqual(rendered, flexible.build_subtask_prompt(
                    self.request, self.plan, index, reports, previous_draft=draft,
                ))
                self.assertEqual(sections["request"], self.request)
                self.assertIn("\n[LEASE]\ntier:", sections["worker_instructions"])
                self.assertIn("\n[ESCALATE]\nreason:", sections["worker_instructions"])
                self.assertEqual(hashlib.sha256(rendered.encode()).hexdigest(), expected_hash)

    def test_planner_legacy_workers_and_owner_share_broad_low_and_early_high_boundaries(self) -> None:
        prompts = [flexible.build_plan_prompt(self.request, collaborative=False)]
        prompts.extend(flexible.build_subtask_sections(self.request, self.plan, index, [])["worker_instructions"]
                       for index in range(len(self.plan.subtasks)))
        owner = flexible.fallback_plan(self.request)
        prompts.append(flexible.build_subtask_sections(self.request, owner, 0, [])["worker_instructions"])
        for prompt in prompts:
            self.assertIn("low  - default for bounded implementation, debugging, refactoring, test repairs", prompt)
            self.assertIn("low is not limited to mechanical edits", prompt)
            self.assertIn("mid  - involved non-routine implementation or diagnosis with an understood approach", prompt)
            self.assertIn("Unfamiliar files alone can start with low inspection", prompt)
            self.assertIn("high - substantive new design or approach selection, ambiguous multi-cause diagnosis", prompt)
            self.assertIn("A failed lower-tier attempt or a known root cause is not required", prompt)
            self.assertIn("A low/mid owner can consult high early for a focused decision", prompt)
            self.assertIn("alone do not justify mid/high", prompt)
            self.assertIn("Do not force a mix of tiers", prompt)

    def test_section_identity_is_derived_while_building_not_from_human_markers(self) -> None:
        request = "User text: Earlier reports:\nPlan:\nPrevious draft and evidence from this turn:"
        report = "External report has its own Plan:\nYour task: retain this evidence"
        draft = "Prior draft says Earlier reports:\nretain its exact file: output/proof.log"
        sections = flexible.build_subtask_sections(
            request, self.plan, 1, [report], previous_draft=draft,
        )

        self.assertEqual(list(sections), [
            "request", "plan", "earlier_reports", "previous_draft", "worker_instructions",
        ])
        self.assertEqual(sections["request"], request)
        self.assertIn(report, sections["earlier_reports"])
        self.assertIn(draft, sections["previous_draft"])
        self.assertIn("Your task: repair every confirmed finding", sections["worker_instructions"])
        native_sections = {name: value for name, value in sections.items() if name != "request"}
        self.assertNotIn(request, "\n".join(native_sections.values()))

    def test_optional_sections_are_absent_without_evidence(self) -> None:
        sections = flexible.build_subtask_sections(self.request, self.plan, 0, [])

        self.assertEqual(list(sections), ["request", "plan", "worker_instructions"])
        self.assertIn("[Sub-task 1/2: low]", sections["plan"])
        self.assertIn("[parallel-read] inspect existing logs", sections["plan"])

    def test_changed_reports_drafts_and_plan_have_separate_section_values(self) -> None:
        original = flexible.build_subtask_sections(
            self.request, self.plan, 1, ["Initial proof: logs/first.log"], previous_draft="Initial draft",
        )
        changed_reports = flexible.build_subtask_sections(
            self.request, self.plan, 1, ["Updated proof: logs/final.log"], previous_draft="Initial draft",
        )
        changed_draft = flexible.build_subtask_sections(
            self.request, self.plan, 1, ["Initial proof: logs/first.log"], previous_draft="Verified final draft",
        )
        changed_plan = flexible.build_subtask_sections(
            self.request, flexible.Plan("New verified goal", self.plan.subtasks), 1,
            ["Initial proof: logs/first.log"], previous_draft="Initial draft",
        )

        for updated, changed_key in (
            (changed_reports, "earlier_reports"),
            (changed_draft, "previous_draft"),
            (changed_plan, "plan"),
        ):
            with self.subTest(changed_key=changed_key):
                self.assertNotEqual(updated[changed_key], original[changed_key])
                self.assertEqual(
                    {key: value for key, value in updated.items() if key != changed_key},
                    {key: value for key, value in original.items() if key != changed_key},
                )

    def test_reader_sections_keep_constraints_and_external_evidence(self) -> None:
        sections = flexible.build_subtask_sections(
            self.request, self.plan, 0, ["Retained external evidence: /workspace/logs/proof.log"],
            previous_draft="Verified startup exit code 0; shutdown remains unverified.",
        )
        instructions = sections["worker_instructions"]

        self.assertIn("Inspection only:", instructions)
        self.assertIn("Do not write, edit, delete, apply fixes, create files", instructions)
        self.assertIn("build, run tests, launch or stop processes, commit, or push", instructions)
        self.assertIn("cannot rely on another concurrent task's findings or file changes", instructions)
        self.assertNotIn("Build/verify", instructions)
        self.assertIn("/workspace/logs/proof.log", sections["earlier_reports"])
        self.assertIn("shutdown remains unverified", sections["previous_draft"])

    def test_writer_sections_keep_execution_rules_and_complete_request_reference(self) -> None:
        request = "Finish every target and obligation. " * 10
        plan = flexible.Plan("Finish original request", (flexible.Subtask("high", request.strip()),))
        sections = flexible.build_subtask_sections(request, plan, 0, ["Retained build proof"])
        instructions = sections["worker_instructions"]

        self.assertEqual(sections["request"], request)
        self.assertIn("the complete user request above, including every saved obligation and constraint", instructions)
        self.assertIn("Build/verify, when needed", instructions)
        self.assertIn("never report done when work remains", instructions)
        self.assertNotIn("Inspection only:", instructions)
        self.assertEqual("\n".join(sections.values()).count(request.strip()), 1)


if __name__ == "__main__":
    unittest.main()
