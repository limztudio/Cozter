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
            (0, [], None, "44c74fd10a207326ea53783a6e26e0205acb54a297af42314f1af49fb6c50ed8"),
            (1, ["report A", "report B"], "Previous verified result",
             "7a332872c0a880d51d590bebf4d92a09c51d695b9849c7b6bd5bc3dae12ad918"),
            (0, ["a" * 180, "a" * 180], "a" * 180,
             "a0672a0d04d310cf8c21a8386f929d797a3a75e1955f7fe69823e66361f29e28"),
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
