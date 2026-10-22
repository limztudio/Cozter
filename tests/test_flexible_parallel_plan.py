"""Explicit independent inspections preserve legacy serial plans and prompts."""

import json
from pathlib import Path
import unittest

from Cozter import config, flexible
from Cozter.tests.helpers import temporary_config


class ParallelReadPlanTests(unittest.TestCase):
    def test_only_explicit_leading_marker_enables_both_execution_flags(self) -> None:
        plan = flexible.parse_plan(
            "[PLAN]\n"
            "1. [low] [parallel-read] inspect transport logs\n"
            "2. [mid] [PARALLEL-READ] inspect parser boundaries\n"
            "3. [high] repair related findings and verify\n"
            "[/PLAN]", "inspect and repair",
        )
        self.assertEqual(
            [(task.tier, task.instruction, task.read_only, task.independent)
             for task in plan.subtasks],
            [("low", "inspect transport logs", True, True),
             ("mid", "inspect parser boundaries", True, True),
             ("high", "repair related findings and verify", False, False)],
        )

    def test_legacy_plan_and_programmatic_subtasks_remain_serial(self) -> None:
        plan = flexible.parse_plan("1. [low] inspect logs\n2. [high] repair", "request")
        for task in (*plan.subtasks, flexible.Subtask("low", "legacy call site")):
            self.assertFalse(task.read_only)
            self.assertFalse(task.independent)
        self.assertEqual(plan.subtasks[0].instruction, "inspect logs")

    def test_execution_marker_does_not_duplicate_scope_and_first_seen_role_wins(self) -> None:
        for first, second, marked in (
            ("[parallel-read] inspect logs", "inspect logs", True),
            ("inspect logs", "[parallel-read] inspect logs", False),
        ):
            with self.subTest(first=first):
                plan = flexible.parse_plan(
                    f"1. [mid] {first}\n2. [mid] {second}\n3. [high] apply the fix",
                    "inspect logs and repair",
                )
                self.assertEqual(len(plan.subtasks), 2)
                self.assertEqual(plan.subtasks[0], flexible.Subtask(
                    "mid", "inspect logs", read_only=marked, independent=marked,
                ))
                self.assertEqual(plan.subtasks[1], flexible.Subtask("high", "apply the fix"))

    def test_marker_tolerates_bullets_and_spaces_without_relaxing_its_meaning(self) -> None:
        plan = flexible.parse_plan(
            "- [low] [ parallel-read ] inspect logs\n"
            "* [mid] [parallel] inspect settings\n"
            "3) [high] explain the [parallel-read] marker", "request",
        )
        self.assertTrue(plan.subtasks[0].read_only)
        self.assertTrue(plan.subtasks[0].independent)
        self.assertEqual(plan.subtasks[0].instruction, "inspect logs")
        self.assertFalse(plan.subtasks[1].read_only)
        self.assertFalse(plan.subtasks[1].independent)
        self.assertEqual(plan.subtasks[1].instruction, "[parallel] inspect settings")
        self.assertFalse(plan.subtasks[2].read_only)
        self.assertFalse(plan.subtasks[2].independent)

    def test_empty_marked_task_is_skipped_and_cannot_become_an_empty_worker(self) -> None:
        plan = flexible.parse_plan(
            "1. [low] [parallel-read]\n2. [mid] complete the request", "request",
        )
        self.assertEqual(plan.subtasks, (flexible.Subtask("mid", "complete the request"),))
        fallback = flexible.parse_plan("1. [low] [parallel-read]", "original request")
        self.assertEqual(fallback.subtasks, (flexible.Subtask("low", "original request"),))

    def test_followups_cannot_opt_themselves_into_concurrent_inspection(self) -> None:
        followups = flexible.parse_followups(
            "[followup:low] [parallel-read] inspect missing transport evidence\n"
            "[followup:low] inspect missing transport evidence\n"
            "[followup:high] apply the remaining fix",
        )
        self.assertEqual(followups, [
            flexible.Subtask("low", "inspect missing transport evidence"),
            flexible.Subtask("high", "apply the remaining fix"),
        ])

    def test_fallback_and_question_do_not_create_parallel_workers(self) -> None:
        fallback = flexible.parse_plan("unparseable", "original request")
        self.assertFalse(fallback.subtasks[0].read_only)
        self.assertFalse(fallback.subtasks[0].independent)
        question = flexible.parse_plan("[QUESTION]Which workspace?[/QUESTION]", "request")
        self.assertEqual(question.subtasks, ())


class ParallelReadPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = flexible.Plan("Inspect related components, then repair", (
            flexible.Subtask("low", "inspect existing logs", read_only=True, independent=True),
            flexible.Subtask("mid", "inspect parser code", read_only=True, independent=True),
            flexible.Subtask("high", "repair and verify related findings"),
        ))

    def test_inspection_prompt_has_scope_and_mutation_restrictions_without_repair_directives(self) -> None:
        prompt = "\n".join(flexible.build_subtask_sections("inspect and repair the project", self.plan, 0, []).values())
        self.assertIn("Inspection only:", prompt)
        self.assertIn("Do not write, edit, delete, apply fixes, create files", prompt)
        self.assertIn("build, run tests, launch or stop processes, commit, or push", prompt)
        self.assertIn("Do not expand into the full original request", prompt)
        self.assertIn("report exact findings", prompt)
        self.assertIn("later writer", prompt)
        self.assertNotIn("Build/verify", prompt)
        self.assertNotIn("run canonical build/test", prompt)
        self.assertNotIn("line as [followup:", prompt)

    def test_readers_use_preceding_reports_but_not_their_concurrent_peers(self) -> None:
        prompt = "\n".join(flexible.build_subtask_sections(
            "inspect and repair", self.plan, 1, ["Previously confirmed workspace and scope"],
            previous_draft="Checkpoint: transport ownership was already verified.",
        ).values())
        self.assertIn("Previously confirmed workspace and scope", prompt)
        self.assertIn("Checkpoint: transport ownership was already verified.", prompt)
        self.assertIn("cannot rely on another concurrent task's findings or file changes", prompt)
        self.assertIn("Do not request redundant workflow stages or repeat completed work", prompt)

    def test_writer_keeps_implementation_and_verification_instructions_and_all_earlier_reports(self) -> None:
        prompt = "\n".join(flexible.build_subtask_sections(
            "inspect and repair", self.plan, 2,
            ["Transport finding: duplicate sync message", "Parser finding: missing source filter"],
        ).values())
        self.assertIn("Transport finding: duplicate sync message", prompt)
        self.assertIn("Parser finding: missing source filter", prompt)
        self.assertIn("Build/verify, when needed", prompt)
        self.assertIn("repair actual failures and verify", prompt)
        self.assertIn("line as [followup:", prompt)
        self.assertNotIn("Inspection only:", prompt)

    def test_read_only_restrictions_do_not_depend_on_independence_flag(self) -> None:
        serial_reader = flexible.Plan("inspect", (
            flexible.Subtask("low", "inspect a dependency", read_only=True),
        ))
        prompt = "\n".join(flexible.build_subtask_sections("request", serial_reader, 0, []).values())
        self.assertIn("Inspection only:", prompt)
        self.assertNotIn("run canonical build/test", prompt)

    def test_rendered_plan_retains_the_parallel_read_markers(self) -> None:
        rendered = flexible._render_plan(self.plan)
        self.assertIn("1. [low] [parallel-read] inspect existing logs", rendered)
        self.assertIn("2. [mid] [parallel-read] inspect parser code", rendered)
        self.assertIn("3. [high] repair and verify related findings", rendered)

    def test_planner_keeps_one_owner_with_optional_independent_advice_and_saved_evidence(self) -> None:
        prompt = flexible.build_plan_prompt("request and saved checkpoint", collaborative=False)
        self.assertIn("Choose ONE persistent owner", prompt)
        self.assertIn("Never split workflow stages into different writers", prompt)
        self.assertIn("read-only, and independent of each other", prompt)
        self.assertIn("They run before the owner's first turn", prompt)
        self.assertIn("Exclude completed actions from the new plan", prompt)
        self.assertIn("They cannot edit, build, test, launch/stop processes, commit, push", prompt)
        self.assertIn("[SPECIALISTS]", prompt)
        self.assertIn("[OWNER]", prompt)

    def test_serial_build_worker_uses_bounded_native_parallelism_without_shared_tree_overlap(self) -> None:
        prompt = "\n".join(flexible.build_subtask_sections("repair and verify", self.plan, 2, []).values())
        self.assertIn("project's supported jobs/parallel flag", prompt)
        self.assertIn("resource bound appropriate to available CPU and memory", prompt)
        self.assertIn("Never overlap configure/build commands that share a generated tree or dependency metadata", prompt)
        self.assertIn("different configurations that use the same generated state", prompt)
        self.assertIn("Final verification must still cover the original scope", prompt)

    def test_resumed_worker_reuses_valid_build_evidence_and_targets_only_invalidated_checks(self) -> None:
        remaining = flexible.Plan("Finish the verified change", (
            flexible.Subtask("low", "commit and push the verified changes"),
        ))
        prompt = "\n".join(flexible.build_subtask_sections(
            "finish the original repair and verification", remaining, 0,
            ["Checkpoint: builds/tests passed; relevant inputs unchanged; commit/push pending."],
        ).values())
        self.assertIn("builds/tests passed; relevant inputs unchanged; commit/push pending", prompt)
        self.assertIn("Reuse verified completed build/test evidence when relevant inputs and environment are unchanged", prompt)
        self.assertIn("rerun only invalidated checks", prompt)
        self.assertIn("resume alone is not a reason to repeat whole audits or builds", prompt)
        self.assertIn("only the checkpoint's remaining obligations, such as commit/push after verified builds", prompt)
        self.assertIn("distinguish valid retained evidence from anything unverified", prompt)

    def test_resumed_plan_does_not_regenerate_verified_builds_for_pending_commit_push(self) -> None:
        prompt = flexible.build_plan_prompt(
            "Checkpoint: build/test verified, only commit/push remains", collaborative=False,
        )
        self.assertIn("Plan only the checkpoint's unfinished obligations", prompt)
        self.assertIn("remaining commit/push and necessary targeted checks, not another audit or rebuild", prompt)
        self.assertIn("relevant inputs and environment are unchanged", prompt)
        self.assertIn("final verification still covers the full original scope", prompt)


class FlexibleParallelConfigTests(unittest.TestCase):
    def test_missing_config_uses_three_and_one_explicitly_disables_concurrency(self) -> None:
        with temporary_config({}):
            self.assertEqual(config.get_flexible_max_parallel_workers(), 3)
        for workers in (1, 2, 3, 4):
            with self.subTest(workers=workers), temporary_config({"flexible_max_parallel_workers": workers}):
                self.assertEqual(config.get_flexible_max_parallel_workers(), workers)

    def test_invalid_values_cannot_expand_concurrency_beyond_four(self) -> None:
        for value in (0, -1, 5, 100, True, False, "4", 2.0, None, [], {}):
            with self.subTest(value=value), temporary_config({"flexible_max_parallel_workers": value}):
                self.assertEqual(config.get_flexible_max_parallel_workers(), 3)

    def test_runtime_config_edit_changes_the_next_read(self) -> None:
        with temporary_config({"flexible_max_parallel_workers": 3}):
            self.assertEqual(config.get_flexible_max_parallel_workers(), 3)
            Path(config.CONFIG_PATH).write_text(json.dumps({
                "flexible_max_parallel_workers": 1, "updated": True,
            }), encoding="utf-8")
            self.assertEqual(config.get_flexible_max_parallel_workers(), 1)
