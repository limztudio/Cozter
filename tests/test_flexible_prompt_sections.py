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
            (0, [], None, "50b3bc5f26a49b951cf5eb698a9591c949e0052658d8077746933fb0314491d1"),
            (1, ["report A", "report B"], "Previous verified result",
             "d5ed9dfdb9814bdd94e2b744053b4dfbdc70c8c765c59a591af8c8df12e0d964"),
            (0, ["a" * 180, "a" * 180], "a" * 180,
             "2d370240c5045d82b7488e088009e9e1a2971acb919b0f47dfdbabf18aadb872"),
        )
        for index, reports, draft, expected_hash in cases:
            with self.subTest(index=index, reports=len(reports), draft=draft is not None):
                sections = flexible.build_subtask_sections(
                    self.request, self.plan, index, reports, previous_draft=draft,
                )
                rendered = "\n".join(sections.values())
                self.assertEqual(sections["request"], self.request)
                self.assertIn("\n[LEASE]\ntier:", sections["worker_instructions"])
                self.assertIn("\n[ESCALATE]\nreason:", sections["worker_instructions"])
                self.assertEqual(hashlib.sha256(rendered.encode()).hexdigest(), expected_hash)

    def test_planner_legacy_workers_and_owner_share_soft_targets_without_retiering(self) -> None:
        prompts = [flexible.build_plan_prompt(self.request, collaborative=False)]
        prompts.extend(flexible.build_subtask_sections(self.request, self.plan, index, [])["worker_instructions"]
                       for index in range(len(self.plan.subtasks)))
        owner = flexible.fallback_plan(self.request)
        prompts.append(flexible.build_subtask_sections(self.request, owner, 0, [])["worker_instructions"])
        for prompt in prompts:
            self.assertIn("Target distribution across new owner and specialist choices: low 60%, mid 30%, high 10%", prompt)
            self.assertIn("soft workload targets, not quotas or random selection", prompt)
            self.assertIn("If adjacent tiers both look suitable, prefer the lower tier", prompt)
            self.assertIn("Never add helpers or reroute an existing owner just to meet the target", prompt)
            self.assertIn("low  - default for bounded implementation, debugging, refactoring, test repairs", prompt)
            self.assertIn("low is not limited to mechanical edits", prompt)
            self.assertIn("mid  - involved non-routine implementation, diagnosis, or design with several constraints", prompt)
            self.assertIn("Unfamiliar files alone can start with low inspection", prompt)
            self.assertIn("high - reserved for difficult reasoning with deeply coupled constraints", prompt)
            self.assertIn("uncertainty or a design decision alone does not justify high", prompt)
            self.assertIn("Direct high ownership is allowed when the request already establishes this difficulty", prompt)
            self.assertIn("a failed lower-tier attempt is not required", prompt)
            self.assertIn("Escalation remains available when evidence reveals a reasoning limit", prompt)
            self.assertIn("alone do not justify mid/high", prompt)
            self.assertIn("Do not force a mix of tiers", prompt)
            for obsolete in ("Give high a wider role", "choose high", "prefer a focused high consultation"):
                self.assertNotIn(obsolete, prompt)
        self.assertIn("Choose the lowest capable specialist", prompts[0])
        for prompt in prompts[1:]:
            self.assertIn("Choose the lowest capable helper using the same 60% low, 30% mid, 10% high", prompt)
            self.assertIn("Reserve high consultations for concrete difficult reasoning", prompt)

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
