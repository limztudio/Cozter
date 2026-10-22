"""Tests for the flexible meta-agent.

Flexible understands a request with the summary model, splits it into
difficulty-graded sub-tasks, runs each on the agent+model bound to its
tier, and merges the workers' reports into one reply. These cover the
plan parser (a cheap model writes it, so it must be forgiving), the
per-tier settings, and the routing/merge loop itself.
"""

import asyncio
import tempfile
import unittest
from unittest import mock

from Cozter import agent, flexible, workspace
from Cozter.agent_tools.builtin.bash import BashTool
from Cozter.backends_agent.base import AgentResult, ChatEvent, append_detached_task


class PlanParsingTests(unittest.TestCase):
    def test_tier_names_use_current_schema(self) -> None:
        for tier in flexible.TIERS:
            self.assertEqual(flexible.normalize_tier(f" {tier.upper()} "), tier)
        for obsolete in ("medium", "med", "middle"):
            with self.subTest(tier=obsolete):
                self.assertIsNone(flexible.normalize_tier(obsolete))

    def test_parses_understanding_and_tiers(self) -> None:
        raw = (
            "[UNDERSTANDING]\nUser wants the retry bug fixed.\n"
            "[/UNDERSTANDING]\n"
            "[PLAN]\n"
            "1. [low] add a validation check\n"
            "2. [mid] write unit tests for the helper\n"
            "3. [high] debug the non-obvious retry race\n"
            "[/PLAN]\n"
        )
        plan = flexible.parse_plan(raw, "fix retries")

        self.assertEqual(plan.understanding, "User wants the retry bug fixed.")
        self.assertIsNone(plan.question)
        self.assertEqual(
            [(subtask.tier, subtask.instruction) for subtask in plan.subtasks],
            [
                ("low", "add a validation check"),
                ("mid", "write unit tests for the helper"),
                ("high", "debug the non-obvious retry race"),
            ],
        )

    def test_bullets_without_the_wrapper_still_parse(self) -> None:
        plan = flexible.parse_plan(
            "- [high] do the hard thing\n* [low] do the easy thing\n", "req",
        )
        self.assertEqual([subtask.tier for subtask in plan.subtasks], ["high", "low"])

    def test_exact_duplicate_tasks_are_removed_in_first_seen_order(self) -> None:
        plan = flexible.parse_plan(
            "[PLAN]\n1. [low] inspect status\n2. [mid] repair code\n"
            "3. [low] inspect status\n4. [mid] repair code\n"
            "5. [high] inspect status\n6. [low] inspect logs\n[/PLAN]",
            "inspect and repair",
        )
        self.assertEqual(
            [(task.tier, task.instruction) for task in plan.subtasks],
            [("low", "inspect status"), ("mid", "repair code"),
             ("high", "inspect status"), ("low", "inspect logs")],
        )

    def test_duplicates_do_not_consume_the_unique_task_cap(self) -> None:
        raw = "[PLAN]\n" + "".join(
            f"{index}. [low] task {index // 2}\n"
            for index in range(flexible.MAX_SUBTASKS * 2)
        ) + "[/PLAN]"
        plan = flexible.parse_plan(raw, "inspect all tasks")
        self.assertEqual(
            [task.instruction for task in plan.subtasks],
            [f"task {index}" for index in range(flexible.MAX_SUBTASKS)],
        )
        self.assertNotIn("PARTIAL", plan.understanding)

    def test_subtask_count_is_capped(self) -> None:
        raw = "[PLAN]\n" + "".join(
            f"{index}. [low] task {index}\n"
            for index in range(1, flexible.MAX_SUBTASKS + 8)
        ) + "[/PLAN]"
        plan = flexible.parse_plan(raw, "req")
        self.assertEqual(len(plan.subtasks), flexible.MAX_SUBTASKS)
        self.assertEqual(
            plan.subtasks[-1].instruction,
            f"task {flexible.MAX_SUBTASKS}",
        )

    def test_explicit_remainder_survives_inside_or_after_the_plan(self) -> None:
        note = "PARTIAL: remainder is the release deployment."
        for raw in (
            f"[PLAN]\n1. [low] verify code\n{note}\n[/PLAN]",
            f"[PLAN]\n1. [low] verify code\n[/PLAN]\n{note}",
        ):
            with self.subTest(raw=raw):
                plan = flexible.parse_plan(raw, "verify and deploy")
                self.assertIn(note, plan.understanding)
                self.assertIn(note, "\n".join(flexible.build_subtask_sections("request", plan, 0, []).values()))

    def test_lines_with_an_unknown_tier_are_skipped(self) -> None:
        plan = flexible.parse_plan(
            "[PLAN]\n1. [urgent] nope\n2. [low] yep\n[/PLAN]", "req",
        )
        self.assertEqual(
            [(subtask.tier, subtask.instruction) for subtask in plan.subtasks], [("low", "yep")],
        )

    def test_unparseable_output_keeps_scope_without_buying_the_high_tier(self) -> None:
        for raw in ("", "   ", "no idea, sorry", "[PLAN]\n[/PLAN]", "[medium] inspect it"):
            plan = flexible.parse_plan(raw, "the original request")
            self.assertEqual(len(plan.subtasks), 1)
            self.assertEqual(plan.subtasks[0].tier, "low")
            self.assertEqual(
                plan.subtasks[0].instruction, "the original request",
            )

    def test_question_short_circuits_the_plan(self) -> None:
        plan = flexible.parse_plan(
            "[QUESTION]\nWhich retry path do you mean?\n[/QUESTION]", "req",
        )
        self.assertEqual(plan.question, "Which retry path do you mean?")
        self.assertEqual(plan.subtasks, ())

    def test_planner_offers_the_question_route_only_when_collaborative(
        self,
    ) -> None:
        collaborative = flexible.build_plan_prompt("ctx", collaborative=True)
        autonomous = flexible.build_plan_prompt("ctx", collaborative=False)
        self.assertIn("[QUESTION]", collaborative)
        self.assertNotIn("[QUESTION]", autonomous)


class EscalationParsingTests(unittest.TestCase):
    _FIELDS = {
        "reason": "The lock ordering remains unresolved.",
        "evidence": "Inspected both locks; the targeted race check still fails.",
        "remaining": "Resolve the conflicting lock order and verify the existing regression.",
    }

    @classmethod
    def _block(cls, fields: dict[str, str] | None = None) -> str:
        return "[ESCALATE]\n" + "\n".join(
            f"{key}: {value}" for key, value in (fields if fields is not None else cls._FIELDS).items()
        ) + "\n[/ESCALATE]"

    def test_embedded_block_retains_evidence_and_round_trips_the_checkpoint(self) -> None:
        expected = flexible.Escalation(**self._FIELDS)
        report = "Completed independent checks.\r\n"
        report += "[followup:low] inspect a newly found documentation link\r\n"
        report += self._block().replace("\n", "\r\n") + "\r\n  \t"
        self.assertEqual(flexible.parse_escalation(report), expected)
        self.assertEqual(flexible.restore_escalation(self._FIELDS), expected)
        self.assertEqual(
            flexible.parse_followups(report),
            [flexible.Subtask("low", "inspect a newly found documentation link")],
        )

    def test_partial_duplicate_and_non_field_payloads_never_trigger_escalation(self) -> None:
        block = self._block()
        invalid = (
            "", "[ESCALATE]", "[/ESCALATE]", block.replace("[/ESCALATE]", ""),
            block.replace("evidence:", "unknown:"), block.replace("evidence:", "reason:"),
            block.replace("evidence: " + self._FIELDS["evidence"] + "\n", ""),
            block.replace("evidence: " + self._FIELDS["evidence"], "evidence:   "),
            block.replace("[/ESCALATE]", "an unlabeled line\n[/ESCALATE]"),
            block + "\n" + block,
            "[ESCALATE]\n" + block,
            block + "\n[/ESCALATE]",
            "[/ESCALATE]\n" + block,
            "prefix " + block + " suffix",
        )
        for report in invalid:
            with self.subTest(report=report):
                self.assertIsNone(flexible.parse_escalation(report))

    def test_oversized_block_is_rejected_without_discarding_normal_reports(self) -> None:
        oversized = {**self._FIELDS, "evidence": "x" * flexible._CONTROL_MAX_CHARS}
        self.assertIsNone(flexible.parse_escalation(self._block(oversized)))
        self.assertIsNone(flexible.restore_escalation(oversized))
        self.assertIsNone(flexible.restore_escalation({
            **self._FIELDS, "evidence": " " * flexible._CONTROL_MAX_CHARS + "x",
        }))
        surrounding = "ordinary evidence\n" * flexible._CONTROL_MAX_CHARS
        self.assertEqual(
            flexible.parse_escalation(surrounding + self._block()),
            flexible.Escalation(**self._FIELDS),
        )

    def test_retained_fields_reject_ambiguous_or_corrupt_values(self) -> None:
        for value in (
            None, [], {}, {**self._FIELDS, "tier": "high"},
            {**self._FIELDS, "remaining": None}, {**self._FIELDS, "remaining": "   "},
            {**self._FIELDS, "remaining": "one\ntwo"},
            {**self._FIELDS, "remaining": "one\r"},
            {**self._FIELDS, "remaining": "one\u2028two"},
        ):
            with self.subTest(value=value):
                self.assertIsNone(flexible.restore_escalation(value))


class LeaseParsingTests(unittest.TestCase):
    _FIELDS = {
        "tier": "low",
        "reason": "A bounded inventory leaves the design work with the parent.",
        "task": "Read the existing retry tests and report their documented lock order; do not edit.",
    }

    @classmethod
    def _block(cls, fields: dict[str, str] | None = None) -> str:
        return "[LEASE]\n" + "\n".join(
            f"{key}: {value}" for key, value in (fields if fields is not None else cls._FIELDS).items()
        ) + "\n[/LEASE]"

    def test_any_tier_can_be_retained_without_converting_the_parent_obligation(self) -> None:
        for tier in flexible.TIERS:
            fields = {**self._FIELDS, "tier": f" {tier.upper()} "}
            expected = flexible.Lease(tier, fields["reason"], fields["task"])
            report = "Parent edits are retained.\r\n" + self._block(fields).replace("\n", "\r\n")
            with self.subTest(tier=tier):
                self.assertEqual(flexible.parse_lease(report), expected)
                self.assertEqual(flexible.restore_lease(fields), expected)
                self.assertIsNone(flexible.parse_escalation(report))

    def test_incomplete_duplicate_and_unknown_fields_do_not_start_a_helper(self) -> None:
        block = self._block()
        for report in (
            "", "[LEASE]", "[/LEASE]", block.replace("[/LEASE]", ""),
            block.replace("tier: low", "tier: medium"),
            block.replace("tier: low", "tier: "),
            block.replace("task:", "reason:"), block.replace("reason:", "unknown:"),
            block.replace("task: " + self._FIELDS["task"] + "\n", ""),
            block.replace("[/LEASE]", "unlabeled question\n[/LEASE]"),
            block + "\n" + block, "[LEASE]\n" + block,
            block + "\n[/LEASE]", "[/LEASE]\n" + block,
            "prefix " + block + " suffix",
        ):
            with self.subTest(report=report):
                self.assertIsNone(flexible.parse_lease(report))

    def test_control_parsers_do_not_choose_between_lease_and_escalation(self) -> None:
        escalation = EscalationParsingTests._block()
        report = escalation + "\n" + self._block()
        self.assertEqual(flexible.parse_lease(report), flexible.Lease(**self._FIELDS))
        self.assertIsNone(flexible.parse_escalation(report))

    def test_research_examples_and_nonterminal_blocks_do_not_trigger_control(self) -> None:
        for parser, block, expected in (
            (flexible.parse_lease, self._block(), flexible.Lease(**self._FIELDS)),
            (flexible.parse_escalation, EscalationParsingTests._block(),
             flexible.Escalation(**EscalationParsingTests._FIELDS)),
        ):
            for report in (
                block + "\nThis is just an example.",
                block + "\n[followup:low] unrelated scope",
                "```text\n" + block,
                "   ```json\r\n" + block,
                "~~~~text\n" + block,
                "````text\nexample\n```\n" + block,
                "~~~\nexample\n```\n" + block,
                "\n".join("    " + line for line in block.splitlines()),
            ):
                with self.subTest(parser=parser.__name__, report=report):
                    self.assertIsNone(parser(report))
            for fence in ("```", "````", "~~~"):
                report = f"Example:\n{fence}text\n{block}\n{fence}\nActual request:\n{block}\n"
                with self.subTest(parser=parser.__name__, fence=fence):
                    self.assertEqual(parser(report), expected)

    def test_oversized_or_corrupt_retained_lease_is_not_resumed(self) -> None:
        for fields in (
            None, {}, {**self._FIELDS, "parent": "high"},
            {**self._FIELDS, "tier": "urgent"}, {**self._FIELDS, "tier": 1},
            {**self._FIELDS, "reason": " "}, {**self._FIELDS, "task": "one\ntwo"},
            {**self._FIELDS, "task": "one\r"}, {**self._FIELDS, "task": "one\u2028two"},
            {**self._FIELDS, "task": "x" * flexible._CONTROL_MAX_CHARS},
            {**self._FIELDS, "task": " " * flexible._CONTROL_MAX_CHARS + "x"},
        ):
            with self.subTest(fields=fields):
                self.assertIsNone(flexible.restore_lease(fields))
        self.assertIsNone(flexible.parse_lease(self._block({
            **self._FIELDS, "task": "x" * flexible._CONTROL_MAX_CHARS,
        })))


class MergeFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = flexible.Plan(
            understanding="Fix and verify",
            subtasks=(
                flexible.Subtask("low", "fix the bug"),
                flexible.Subtask("mid", "verify the fix"),
            ),
        )

    def test_missing_or_empty_report_keeps_remainder_visible(self) -> None:
        for reports in (["Fixed"], ["Fixed", ""], ["", "Verified"]):
            with self.subTest(reports=reports):
                reply = flexible.merge_fallback(self.plan, reports)
                self.assertIn("PARTIAL", reply)
                missing = "fix the bug" if not reports[0] else "verify the fix"
                self.assertIn(missing, reply)
                self.assertIn("unverified", reply)

    def test_no_reports_remains_a_failure(self) -> None:
        for reports in ([], [""], ["", ""]):
            with self.subTest(reports=reports):
                self.assertEqual(flexible.merge_fallback(self.plan, reports), "")

    def test_complete_reports_have_no_missing_report_notice(self) -> None:
        reply = flexible.merge_fallback(self.plan, ["Fixed", "Verified"])
        self.assertIn("Fixed", reply)
        self.assertIn("Verified", reply)
        self.assertNotIn("PARTIAL", reply)


class FlexibleSettingsTests(unittest.TestCase):
    def test_tier_model_defaults_follow_the_tier_agent(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            workspace.set_flexible_backend_name(ws, "high", "claude_code")

            self.assertEqual(workspace.get_flexible_model(ws, "high"), "opus")
            # Rebinding one tier leaves the others.
            self.assertEqual(
                workspace.get_flexible_model(ws, "low"), "gpt-6-luna",
            )

    def test_copilot_tier_and_summary_defaults_use_policy_aware_auto(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            workspace.set_flexible_backend_name(ws, "high", "copilot")
            workspace.set_summary_backend_name(ws, "copilot")

            self.assertEqual(workspace.get_flexible_model(ws, "high"), "auto")
            self.assertEqual(workspace.get_summary_model(ws), "auto")

    def test_each_tier_agent_remembers_its_own_model(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            workspace.set_flexible_model(ws, "mid", "gpt-5.6-sol")
            workspace.set_flexible_backend_name(ws, "mid", "claude_code")
            workspace.set_flexible_model(ws, "mid", "sonnet")
            self.assertEqual(workspace.get_flexible_model(ws, "mid"), "sonnet")

            # Switching back restores the last model.
            workspace.set_flexible_backend_name(ws, "mid", "codex")
            self.assertEqual(workspace.get_flexible_model(ws, "mid"), "gpt-5.6-sol")

    def test_flexible_may_not_serve_as_its_own_tier(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            with self.assertRaises(ValueError):
                workspace.set_flexible_backend_name(ws, "low", "flexible")

    def test_flexible_may_not_serve_as_the_summary_agent(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            with self.assertRaises(ValueError):
                workspace.set_summary_backend_name(ws, "flexible")

    def test_unknown_tier_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            with self.assertRaises(ValueError):
                workspace.get_flexible_backend_name(ws, "extreme")


def _worker(text: str, *, usage: dict | None = None) -> AgentResult:
    result = AgentResult()
    result.events.append(ChatEvent(kind="tool", content="ran a tool"))
    result.events.append(ChatEvent(kind="text", content=text))
    result.text = text
    result.usage = usage
    return result


class FlexibleRunTests(unittest.IsolatedAsyncioTestCase):
    """The routing loop, with the planner/merge and workers stubbed out."""

    def setUp(self) -> None:
        self.driven: list[tuple[str, str]] = []
        self.internal: list[str] = []
        self.plan_output = (
            "[UNDERSTANDING]\nfix it\n[/UNDERSTANDING]\n"
            "[PLAN]\n"
            "1. [low] tweak the check\n"
            "2. [high] debug the race\n"
            "[/PLAN]"
        )
        self.worker_texts = ["low report", "high report"]
        self.merge_output = "merged answer"
        self.merge_prompt = ""

    async def _fake_internal(
        self, backend, workspace_path, prompt, model, **kwargs,
    ):
        self.internal.append(kwargs["label"])
        if kwargs["label"] == "Flexible planner":
            return self.plan_output
        self.merge_prompt = prompt
        return self.merge_output

    async def _fake_drive(
        self, backend, workspace_path, prompt, model, approval, **kwargs,
    ):
        self.driven.append((backend.name, model))
        return _worker(
            self.worker_texts[len(self.driven) - 1],
            usage={"input_tokens": 10, "output_tokens": 5},
        ), False

    async def _run(self, ws: str, *, collaborative: bool = True):
        with mock.patch.object(
            agent, "run_internal_backend", self._fake_internal,
        ), mock.patch.object(agent, "_drive_backend", self._fake_drive):
            return await agent._run_flexible(
                "context + user message",
                "user message",
                ws,
                approval="auto",
                effort=0,
                collaborative=collaborative,
                summary_backend_name="codex",
                summary_model="gpt-5.6-luna",
                on_event=None,
                inject_queue=None,
                injected=[],
            )

    async def test_each_subtask_runs_on_its_own_tier_agent(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            await asyncio.to_thread(
                workspace.set_flexible_backend_name, ws, "high", "claude_code",
            )

            _, restarting = await self._run(ws)

        self.assertFalse(restarting)
        self.assertEqual(
            self.driven, [("codex", "gpt-6-luna"), ("claude_code", "opus")],
        )
        self.assertEqual(
            self.internal, ["Flexible planner", "Flexible merge"],
        )

    async def test_pipeline_timings_pair_stage_starts_with_completion(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            with self.assertLogs("Cozter.performance", level="INFO") as captured:
                result, _ = await self._run(ws)
                with mock.patch.object(
                    agent, "run_internal_backend", mock.AsyncMock(return_value="[DONE]"),
                ):
                    await agent._judge_draft(
                        "user message", result.text, ws,
                        summary_backend_name="codex", summary_model="gpt-5.6-luna",
                        inject_queue=None, injected=[],
                    )
        starts = [record for record in captured.output if "event=start" in record]
        ends = [record for record in captured.output if "duration_seconds=" in record]
        self.assertEqual(len(starts), 5)
        self.assertEqual(len(ends), 5)
        expected_stages = (
            "flexible_planner", "flexible_worker", "flexible_worker",
            "flexible_merge", "judge",
        )
        for index, stage in enumerate(expected_stages):
            with self.subTest(stage=stage, index=index):
                self.assertIn(f"stage={stage} ", starts[index])
                self.assertEqual(
                    starts[index].split(" event=start")[0],
                    ends[index].split(" outcome=")[0],
                )
                self.assertLess(
                    captured.output.index(starts[index]),
                    captured.output.index(ends[index]),
                )
        for index, worker_index in ((1, 1), (2, 2)):
            for record in (starts[index], ends[index]):
                self.assertIn(f"worker_index={worker_index} worker_count=2", record)
        private_values = (
            "context + user message", "user message", "low report",
            "high report", "merged answer", "tweak the check", "debug the race",
        )
        for value in private_values:
            self.assertNotIn(value, "\n".join(captured.output))

    async def test_only_the_merged_answer_is_sent_to_the_user(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, _ = await self._run(ws)

        # Worker text is internal.
        texts = [event.content for event in result.events if event.kind == "text"]
        self.assertEqual(texts, ["merged answer"])
        self.assertEqual(result.text, "merged answer")
        # Their tool events still stream through.
        self.assertEqual(
            len([event for event in result.events if event.kind == "tool"]), 2,
        )

    async def test_worker_usage_is_summed_into_one_total(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, _ = await self._run(ws)

        self.assertEqual(
            result.usage, {"input_tokens": 20, "output_tokens": 10},
        )

    async def test_discovered_worker_tasks_survive_merge_and_single_worker(self) -> None:
        async def detached_worker(*_args, **_kwargs):
            result = _worker("Provider task started.")
            append_detached_task(result, "claude_code", "task-123")
            return result, False

        with tempfile.TemporaryDirectory() as ws:
            for plan_output in (
                self.plan_output,
                "[PLAN]\n1. [low] perform the work\n[/PLAN]",
            ):
                with self.subTest(plan_output=plan_output):
                    self.plan_output = plan_output
                    with (
                        mock.patch.object(agent, "run_internal_backend", self._fake_internal),
                        mock.patch.object(agent, "_drive_backend", detached_worker),
                    ):
                        result, restarting = await agent._run_flexible(
                            "context", "request", ws, approval="auto", effort=0,
                            collaborative=False, summary_backend_name="codex",
                            summary_model="summary", on_event=None,
                            inject_queue=None, injected=[],
                        )
                    self.assertFalse(restarting)
                    self.assertEqual(len(result.detached_tasks), 1)
                    self.assertEqual(result.detached_tasks[0].task_id, "task-123")


    async def test_worker_attachments_survive_the_merge(self) -> None:
        self.worker_texts = ["made a chart [[attach: chart.png]]", "done"]
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, _ = await self._run(ws)

        self.assertIn("[[attach: chart.png]]", result.text)

    async def test_an_ambiguous_request_asks_instead_of_guessing(
        self,
    ) -> None:
        self.plan_output = "[QUESTION]\nWhich retry path?\n[/QUESTION]"
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, restarting = await self._run(ws, collaborative=True)

        self.assertFalse(restarting)
        self.assertIn("Which retry path?", result.text)
        self.assertIn("[[await]]", result.text)
        self.assertEqual(self.driven, [])
        self.assertEqual(self.internal, ["Flexible planner"])

    async def test_an_autonomous_turn_never_stops_to_ask(self) -> None:
        self.plan_output = "[QUESTION]\nWhich retry path?\n[/QUESTION]"
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, _ = await self._run(ws, collaborative=False)

        # A scheduled turn degrades into the fallback plan.
        self.assertNotIn("[[await]]", result.text)
        self.assertEqual(self.driven, [("codex", "gpt-6-luna")])

    async def test_a_merged_answer_may_end_on_a_blocking_question(
        self,
    ) -> None:
        # Only the merge can stop the turn.
        self.merge_output = "Fixed the check.\n\nWhich retry path?\n\n[[await]]"
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, _ = await self._run(ws, collaborative=True)

        self.assertIn("[[await]]", result.text)
        self.assertIn("Which retry path?", result.text)
        _, awaiting = agent.extract_await(result.text)
        self.assertTrue(awaiting)

    async def test_the_merge_learns_the_marker_only_when_collaborative(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            await self._run(ws, collaborative=True)
            self.assertIn("[[await]]", self.merge_prompt)

            self.driven.clear()  # worker texts are indexed per run
            await self._run(ws, collaborative=False)
            self.assertNotIn("[[await]]", self.merge_prompt)

    async def test_an_unattended_merge_cannot_strand_the_turn_on_a_question(
        self,
    ) -> None:
        # Scheduled runs drop stray merge markers.
        self.merge_output = "Done, but which retry path?\n\n[[await]]"
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, _ = await self._run(ws, collaborative=False)

        self.assertNotIn("[[await]]", result.text)
        self.assertIn("which retry path?", result.text)

    async def test_a_blocked_worker_pauses_the_turn_without_the_marker(
        self,
    ) -> None:
        # A stuck worker's pause must not depend on the merge relay.
        self.worker_texts = [
            "Blocked. Which retry path?\n\n[[await]]", "high report",
        ]
        self.merge_output = "I need to know: which retry path should I use?"
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, _ = await self._run(ws, collaborative=True)

        _, awaiting = agent.extract_await(result.text)
        self.assertTrue(awaiting)
        # The merge is told which sub-task blocked.
        self.assertIn("BLOCKED", self.merge_prompt)
        # Worker markers stay internal; only the merged reply acts.
        self.assertEqual(result.text.count("[[await]]"), 1)

    async def test_a_blocked_worker_cannot_pause_an_unattended_turn(
        self,
    ) -> None:
        self.worker_texts = [
            "Blocked. Which retry path?\n\n[[await]]", "high report",
        ]
        self.merge_output = "Which retry path?\n\n[[await]]"
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, _ = await self._run(ws, collaborative=False)

        self.assertNotIn("[[await]]", result.text)

    async def test_an_optional_offer_does_not_pause_the_turn(self) -> None:
        # No marker = suggestion, not blocker; pausing would wedge the queue.
        self.merge_output = "Fixed the check. Want me to add tests too?"
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, _ = await self._run(ws, collaborative=True)

        _, awaiting = agent.extract_await(result.text)
        self.assertFalse(awaiting)

    async def test_reports_are_concatenated_when_the_merge_agent_is_gone(
        self,
    ) -> None:
        async def missing_merge(
            _backend, _ws_path, _prompt, _model, **kwargs,
        ):
            self.internal.append(kwargs["label"])
            if kwargs["label"] == "Flexible planner":
                return self.plan_output
            return None  # CLI not found

        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            with mock.patch.object(
                agent, "run_internal_backend", missing_merge,
            ), mock.patch.object(agent, "_drive_backend", self._fake_drive):
                result, _ = await agent._run_flexible(
                    "context", "user message", ws,
                    approval="auto", effort=0, collaborative=False,
                    summary_backend_name="codex",
                    summary_model="gpt-5.6-luna",
                    on_event=None, inject_queue=None, injected=[],
                )

        self.assertIn("low report", result.text)
        self.assertIn("high report", result.text)

    async def test_single_subtask_skips_the_merge_call(self) -> None:
        """One sub-task returns the worker report with no merge model call."""
        self.plan_output = "[PLAN]\n1. [low] tweak the check\n[/PLAN]"
        self.worker_texts = ["solo report"]
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, restarting = await self._run(ws)

        self.assertFalse(restarting)
        self.assertEqual(result.text, "solo report")
        self.assertEqual(self.internal, ["Flexible planner"])

    def test_report_excerpt_keeps_final_evidence_and_real_remainder(self) -> None:
        plan = flexible.Plan(
            understanding="",
            subtasks=(flexible.Subtask(tier="mid", instruction="verify remaining targets"),),
        )
        for final_evidence in (
            "FINAL EVIDENCE: all 24 targets checked; exit code 0; warnings 0.",
            "PARTIAL: target 24 is unverified; remainder: exercise its runtime path.",
        ):
            report = "OPENING: changed retry.py.\n" + "detail\n" * 2_000 + final_evidence
            with self.subTest(final_evidence=final_evidence):
                excerpt = flexible.truncate_report(report)
                self.assertEqual(len(excerpt), flexible._REPORT_MAX_CHARS)
                self.assertTrue(excerpt.startswith("OPENING: changed retry.py."))
                self.assertTrue(excerpt.endswith(final_evidence))
                self.assertIn(flexible.REPORT_TRUNCATION_MARKER, excerpt)
                for downstream in (
                    "\n".join(flexible.build_subtask_sections("request", plan, 0, [report]).values()),
                    flexible.build_merge_prompt("request", plan, [report], collaborative=False),
                ):
                    self.assertIn(final_evidence, downstream)
                    self.assertNotIn(report, downstream)
        completed = flexible.truncate_report(
            "OPENING\n" + "detail\n" * 2_000 + "FINAL: complete; exit code 0."
        )
        self.assertNotIn("PARTIAL", completed)

    def test_current_task_instruction_is_not_repeated_in_plan_overview(self) -> None:
        instruction = "Verify these exact targets and report each result: " + "target/" * 600
        earlier_instruction = "prepare fixtures without changing the production configuration"
        plan = flexible.Plan(
            understanding="",
            subtasks=(
                flexible.Subtask(tier="low", instruction=earlier_instruction),
                flexible.Subtask(tier="high", instruction=instruction),
            ),
        )
        prompt = "\n".join(flexible.build_subtask_sections(
            "original user request", plan, 1, ["Fixture evidence: exit code 0."],
        ).values())
        self.assertEqual(prompt.count(instruction), 1)
        self.assertIn(f"Your task: {instruction}\n", prompt)
        self.assertIn(earlier_instruction, prompt)
        self.assertIn("Fixture evidence: exit code 0.", prompt)
        overview = flexible._render_plan(plan, current=1)
        full_overview = flexible._render_plan(plan)
        self.assertEqual(
            len(full_overview) + len("  <- your task") - len(overview),
            len(instruction) - len("(full instruction below)"),
        )
        # The final merge still sees every plan instruction in full.
        self.assertIn(instruction, flexible.build_merge_prompt(
            "request", plan, ["earlier report", "verified targets"], collaborative=False,
        ))

    def test_merge_keeps_full_plan_once_and_maps_all_reports(self) -> None:
        instructions = (
            "Verify every exact target in these directories: " + "target/" * 600,
            "Exercise the runtime path and report warnings for each requested target",
        )
        plan = flexible.Plan(
            understanding="",
            subtasks=tuple(
                flexible.Subtask(tier=tier, instruction=instruction)
                for tier, instruction in zip(("mid", "high"), instructions)
            ),
        )
        prompt = flexible.build_merge_prompt(
            "original request", plan, ["verified first targets"],
            collaborative=True, blocked=(1,),
        )
        for instruction in instructions:
            self.assertEqual(prompt.count(instruction), 1)
        self.assertIn("--- 1 [mid]: plan item 1 ---", prompt)
        self.assertIn("verified first targets", prompt)
        self.assertIn("--- 2 [high]: plan item 2 [BLOCKED] ---", prompt)
        self.assertIn("(no report)", prompt)
        self.assertIn("missing evidence = PARTIAL + remainder", prompt)
        extra_report = flexible.build_merge_prompt(
            "original request", plan, ["first", "second", "extra evidence"],
            collaborative=False,
        )
        self.assertIn("--- 3 [unknown]: (no sub-task) ---", extra_report)
        self.assertIn("extra evidence", extra_report)

    async def test_workers_get_the_request_not_the_full_history(self) -> None:
        """Workers are driven with the bare user request as context."""
        seen: list[str] = []

        async def capture_drive(
            backend, _ws, prompt, _model, _approval, **kwargs,
        ):
            seen.append(prompt)
            return _worker("done", usage={}), False

        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            with mock.patch.object(
                agent, "run_internal_backend", self._fake_internal,
            ), mock.patch.object(agent, "_drive_backend", capture_drive):
                await agent._run_flexible(
                    "FULL HISTORY + user message", "user message", ws,
                    approval="auto", effort=0, collaborative=False,
                    summary_backend_name="codex",
                    summary_model="gpt-5.6-luna",
                    on_event=None, inject_queue=None, injected=[],
                )

        self.assertEqual(len(seen), 2)
        for prompt in seen:
            self.assertNotIn("FULL HISTORY", prompt)
            self.assertIn("user message", prompt)

    async def test_continuation_workers_receive_previous_draft_evidence(self) -> None:
        seen: list[str] = []
        draft = "Fixed retry.py; exit code 0.\n" + "detail\n" * 3_000
        draft += "FINAL EVIDENCE: integration test passed."

        async def capture_drive(*args, **kwargs):
            seen.append(args[2])
            return _worker("remaining evidence added", usage={}), False

        with tempfile.TemporaryDirectory() as ws:
            with mock.patch.object(
                agent, "run_internal_backend", self._fake_internal,
            ), mock.patch.object(agent, "_drive_backend", capture_drive):
                await agent._run_flexible(
                    "context", "add remaining evidence", ws,
                    approval="auto", effort=100, collaborative=False,
                    summary_backend_name="codex", summary_model="gpt-5.6-luna",
                    on_event=None, inject_queue=None, injected=[],
                    previous_draft=draft,
                )
        self.assertEqual(len(seen), 2)
        for prompt in seen:
            self.assertIn("Fixed retry.py; exit code 0.", prompt)
            self.assertIn("FINAL EVIDENCE: integration test passed.", prompt)
            self.assertIn("without repeating completed actions", prompt)
            self.assertIn(flexible.DRAFT_OMISSION_MARKER, prompt)

    async def test_textless_workers_never_reply_with_a_placeholder(
        self,
    ) -> None:
        """A worker that says nothing must not be quoted back as an answer.

        ``AgentResult.text`` starts empty rather than holding the
        "(no response)" placeholder, precisely so the merge step cannot
        mistake it for a report and relay it to the user as the reply.

        A misconfigured agent breaks every step at once, so the merge is
        stubbed out too: the reply the user is left with has to name the
        tier that went quiet, or there is nothing to act on.
        """
        async def silent_worker(*_args, **_kwargs):
            result = AgentResult()
            # Tool events but no text: a backend dying mid-turn.
            result.events.append(ChatEvent(kind="tool", content="ran a tool"))
            return result, False

        async def silent_merge(*_args, **kwargs):
            if kwargs["label"] == "Flexible planner":
                return self.plan_output
            return ""  # the merge model had nothing to add either

        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            low_agent, low_model = workspace.get_flexible_run_config(ws)["low"]
            with mock.patch.object(
                agent, "run_internal_backend", silent_merge,
            ), mock.patch.object(agent, "_drive_backend", silent_worker):
                result, _ = await agent._run_flexible(
                    "context", "user message", ws,
                    approval="auto", effort=0, collaborative=False,
                    summary_backend_name="codex",
                    summary_model="gpt-5.6-luna",
                    on_event=None, inject_queue=None, injected=[],
                )

        self.assertNotIn("(no response)", result.text)
        # The reply names the quiet tier.
        self.assertIn(f"{low_agent}/{low_model}", result.text)

    async def test_all_failed_workers_keep_errors_and_skip_merge(self) -> None:
        async def failing_worker(*_args, **_kwargs):
            return AgentResult(error="429 usage limit reached"), False

        with tempfile.TemporaryDirectory() as ws:
            with mock.patch.object(
                agent, "run_internal_backend", self._fake_internal,
            ), mock.patch.object(agent, "_drive_backend", failing_worker):
                result, _ = await agent._run_flexible(
                    "context", "user message", ws,
                    approval="auto", effort=0, collaborative=False,
                    summary_backend_name="codex", summary_model="gpt-5.6-luna",
                    on_event=None, inject_queue=None, injected=[],
                )
        self.assertEqual(result.error, "429 usage limit reached")
        self.assertEqual(self.internal, ["Flexible planner"])
        self.assertIn("tweak the check", result.text)
        self.assertIn("debug the race", result.text)
        self.assertIn("429 usage limit reached", result.text)

    async def test_mixed_worker_results_keep_success_and_failed_scope(self) -> None:
        worker_results = [
            _worker("Checked retry.py; exit code 0.", usage={}),
            AgentResult(error="provider unavailable"),
        ]

        async def mixed_workers(*_args, **_kwargs):
            return worker_results.pop(0), False

        async def fallback_merge(*args, **kwargs):
            return self.plan_output if kwargs["label"] == "Flexible planner" else ""

        with tempfile.TemporaryDirectory() as ws:
            with mock.patch.object(
                agent, "run_internal_backend", fallback_merge,
            ), mock.patch.object(agent, "_drive_backend", mixed_workers):
                result, _ = await agent._run_flexible(
                    "context", "user message", ws,
                    approval="auto", effort=0, collaborative=False,
                    summary_backend_name="codex", summary_model="gpt-5.6-luna",
                    on_event=None, inject_queue=None, injected=[],
                )
        self.assertIsNone(result.error)
        self.assertIn("Checked retry.py; exit code 0.", result.text)
        self.assertIn("provider unavailable", result.text)
        self.assertIn("debug the race", result.text)

    async def test_exhausted_account_skips_only_exact_target_and_retains_scope(self) -> None:
        self.plan_output = (
            "[PLAN]\n1. [high] inspect status\n2. [low] repair retries\n"
            "3. [mid] verify retry runtime\n4. [high] preserve other component\n[/PLAN]"
        )
        for other_target in (("meta", "other-model"), ("codex", "other-model")):
            for error in ("meta returned HTTP 401: unauthorized", "quota exhausted", "invalid API key"):
                with self.subTest(other_target=other_target, error=error):
                    calls = []

                    async def drive(backend, _ws, _prompt, model, _approval, **kwargs):
                        self.assertEqual(kwargs["effort"], 100)
                        calls.append((backend.name, model))
                        if len(calls) == 2:
                            return AgentResult(error=error), False
                        return _worker(
                            "Status inspected; exit code 0." if len(calls) == 1
                            else "Other component preserved; exit code 0.",
                            usage={"input_tokens": 10},
                        ), False

                    tiers = {"low": ("meta", "blocked-model"), "mid": ("meta", "blocked-model"),
                             "high": other_target}
                    with tempfile.TemporaryDirectory() as ws:
                        with (
                            mock.patch.object(workspace, "get_flexible_run_config", return_value=tiers),
                            mock.patch.object(agent, "run_internal_backend", self._fake_internal),
                            mock.patch.object(agent, "_drive_backend", side_effect=drive),
                        ):
                            result, restarting = await agent._run_flexible(
                                "context", "inspect and repair", ws, approval="full", effort=100,
                                collaborative=False, summary_backend_name="codex", summary_model="summary",
                                on_event=None, inject_queue=None, injected=[],
                            )
                    self.assertFalse(restarting)
                    self.assertEqual(calls, [other_target, ("meta", "blocked-model"), other_target])
                    self.assertEqual(result.error, error)
                    self.assertEqual(result.usage, {"input_tokens": 20})
                    self.assertIn("Status inspected; exit code 0.", result.text)
                    self.assertIn("Other component preserved; exit code 0.", result.text)
                    self.assertIn("sub-task 3 was not run: verify retry runtime", result.text)
                    self.assertIn("This plan item remains unfinished", result.text)
                    self.assertIn(error, result.text)
                    self.assertFalse(result.continue_instruction)
        self.assertTrue(all(label == "Flexible planner" for label in self.internal))

    async def test_transient_throttling_and_task_refusal_do_not_skip_later_worker(self) -> None:
        self.plan_output = "[PLAN]\n1. [low] first\n2. [mid] second\n[/PLAN]"
        for error in (
            "meta returned HTTP 429: too many requests; retry later",
            "meta stopped after 3 consecutive rounds where every tool call was refused",
            "meta returned HTTP 403: forbidden",
        ):
            with self.subTest(error=error):
                calls = []

                async def drive(*_args, **_kwargs):
                    calls.append(1)
                    return (AgentResult(error=error) if len(calls) == 1 else _worker("second completed")), False

                with tempfile.TemporaryDirectory() as ws:
                    with (
                        mock.patch.object(workspace, "get_flexible_run_config", return_value={
                            "low": ("meta", "same-model"), "mid": ("meta", "same-model"),
                        }),
                        mock.patch.object(agent, "run_internal_backend", self._fake_internal),
                        mock.patch.object(agent, "_drive_backend", side_effect=drive),
                    ):
                        result, _ = await agent._run_flexible(
                            "context", "request", ws, approval="full", effort=100,
                            collaborative=False, summary_backend_name="codex", summary_model="summary",
                            on_event=None, inject_queue=None, injected=[],
                        )
                self.assertEqual(len(calls), 2)
                self.assertIn("second completed", result.text)
                self.assertNotIn("was not run", result.text)

    async def test_injection_at_skipped_worker_restarts_with_pending_scope(self) -> None:
        self.plan_output = "[PLAN]\n1. [low] first\n2. [mid] second\n[/PLAN]"
        queue = asyncio.Queue()
        injected = []

        async def on_event(event):
            if "skipping unavailable" in event.content:
                queue.put_nowait("use another authorized model")

        with tempfile.TemporaryDirectory() as ws:
            with (
                mock.patch.object(workspace, "get_flexible_run_config", return_value={
                    "low": ("meta", "same-model"), "mid": ("meta", "same-model"),
                }),
                mock.patch.object(agent, "run_internal_backend", self._fake_internal),
                mock.patch.object(agent, "_drive_backend", return_value=(AgentResult(error="quota exhausted"), False)) as drive,
            ):
                _, restarting = await agent._run_flexible(
                    "context", "request", ws, approval="full", effort=100,
                    collaborative=False, summary_backend_name="codex", summary_model="summary",
                    on_event=on_event, inject_queue=queue, injected=injected,
                )
        self.assertTrue(restarting)
        self.assertEqual(drive.await_count, 1)
        self.assertEqual(injected, ["use another authorized model"])

    async def test_cancellation_at_skipped_worker_propagates(self) -> None:
        self.plan_output = "[PLAN]\n1. [low] first\n2. [mid] second\n[/PLAN]"

        async def on_event(event):
            if "skipping unavailable" in event.content:
                asyncio.current_task().cancel()

        with tempfile.TemporaryDirectory() as ws:
            with (
                mock.patch.object(workspace, "get_flexible_run_config", return_value={
                    "low": ("meta", "same-model"), "mid": ("meta", "same-model"),
                }),
                mock.patch.object(agent, "run_internal_backend", self._fake_internal),
                mock.patch.object(agent, "_drive_backend", return_value=(AgentResult(error="quota exhausted"), False)) as drive,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await agent._run_flexible(
                        "context", "request", ws, approval="full", effort=100,
                        collaborative=False, summary_backend_name="codex", summary_model="summary",
                        on_event=on_event, inject_queue=None, injected=[],
                    )
        self.assertEqual(drive.await_count, 1)

    async def test_followup_lines_grow_the_total_mid_run(self) -> None:
        """Workers discovering leftovers grow done/total live."""
        self.plan_output = "[PLAN]\n1. [low] first\n[/PLAN]"
        calls: list[str] = []
        statuses: list[str] = []

        async def followup_worker(
            _backend, _ws, _prompt, _model, _approval, **kwargs,
        ):
            call_index = len(calls)
            calls.append(_model)
            if call_index == 0:
                return _worker(
                    "first done\n[followup:mid] handle leftover",
                    usage={},
                ), False
            return _worker("leftover done", usage={}), False

        async def on_event(event: ChatEvent) -> None:
            if event.kind == "tool":
                statuses.append(event.content)

        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            with mock.patch.object(
                agent, "run_internal_backend", self._fake_internal,
            ), mock.patch.object(agent, "_drive_backend", followup_worker):
                result, restarting = await agent._run_flexible(
                    "context", "user message", ws,
                    approval="auto", effort=0, collaborative=False,
                    summary_backend_name="codex",
                    summary_model="gpt-5.6-luna", on_event=on_event,
                    inject_queue=None, injected=[],
                )
        self.assertFalse(restarting)
        # Both the task and follow-up ran.
        self.assertEqual(len(calls), 2)
        # Progress shows the growing total, never a stuck denominator.
        self.assertTrue(any("[1/1]" in status for status in statuses))
        self.assertTrue(any("[2/2]" in status for status in statuses))
        self.assertIn("plan grew to 2 sub-tasks", " ".join(statuses))
        self.assertIn("merged answer", result.text)

    def test_followup_parser_accepts_tiers_and_skips_noise(self) -> None:
        found = flexible.parse_followups(
            "done\n[followup:low] fix leftover\nnoise\n"
            "[followup:high] handle edge",
        )
        self.assertEqual(
            [(subtask.tier, subtask.instruction) for subtask in found],
            [("low", "fix leftover"), ("high", "handle edge")],
        )
        self.assertEqual(flexible.parse_followups("nothing new"), [])
        self.assertEqual(
            flexible.parse_followups("[followup:urgent] nope"), [],
        )

    async def test_inject_during_planning_restarts_the_whole_turn(self) -> None:
        """Planner calls are watched just like worker subprocesses are."""
        planner_started = asyncio.Event()

        async def blocked_planner(*_args, **_kwargs):
            planner_started.set()
            await asyncio.Event().wait()

        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            inject_q: asyncio.Queue[str] = asyncio.Queue()
            injected: list[str] = []
            with mock.patch.object(
                agent, "run_internal_backend", blocked_planner,
            ):
                turn = asyncio.create_task(agent._run_flexible(
                    "context", "original request", ws,
                    approval="auto", effort=0, collaborative=False,
                    summary_backend_name="codex",
                    summary_model="gpt-5.6-luna", on_event=None,
                    inject_queue=inject_q, injected=injected,
                ))
                await asyncio.wait_for(planner_started.wait(), timeout=1)
                inject_q.put_nowait("include the new requirement")
                _, restarting = await asyncio.wait_for(turn, timeout=1)

        self.assertTrue(restarting)
        self.assertEqual(injected, ["include the new requirement"])

    async def test_inject_during_merging_restarts_the_whole_turn(self) -> None:
        """An inject must not be stranded while the summary merge runs."""
        merge_started = asyncio.Event()

        async def planner_then_blocked_merge(
            _backend, _workspace_path, _prompt, _model, **kwargs,
        ):
            if kwargs["label"] == "Flexible planner":
                return (
                    "[PLAN]\n1. [low] inspect the request\n"
                    "2. [low] confirm the fix\n[/PLAN]"
                )
            merge_started.set()
            await asyncio.Event().wait()

        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            inject_q: asyncio.Queue[str] = asyncio.Queue()
            injected: list[str] = []
            with mock.patch.object(
                agent, "run_internal_backend", planner_then_blocked_merge,
            ), mock.patch.object(agent, "_drive_backend", self._fake_drive):
                turn = asyncio.create_task(agent._run_flexible(
                    "context", "original request", ws,
                    approval="auto", effort=0, collaborative=False,
                    summary_backend_name="codex",
                    summary_model="gpt-5.6-luna", on_event=None,
                    inject_queue=inject_q, injected=injected,
                ))
                await asyncio.wait_for(merge_started.wait(), timeout=1)
                inject_q.put_nowait("also include the new requirement")
                _, restarting = await asyncio.wait_for(turn, timeout=1)

        self.assertTrue(restarting)
        self.assertEqual(injected, ["also include the new requirement"])

    async def test_inject_wins_over_a_simultaneous_internal_failure(self) -> None:
        """A failed abandoned planner/merge must not suppress an inject."""
        inject_q: asyncio.Queue[str] = asyncio.Queue()
        inject_q.put_nowait("include the new requirement")
        injected: list[str] = []

        async def failing_call() -> str:
            raise RuntimeError("planner connection dropped")

        text, restarting = await agent._run_with_inject_watch(
            failing_call(), inject_q, injected,
        )

        self.assertIsNone(text)
        self.assertTrue(restarting)
        self.assertEqual(injected, ["include the new requirement"])

    async def test_late_inject_after_internal_failure_still_restarts(
        self,
    ) -> None:
        """An inject that arrives after a failed call must not be dropped."""
        inject_q: asyncio.Queue[str] = asyncio.Queue()
        injected: list[str] = []
        original_take = agent._take_pending_injections

        def take_after_putting(inject_queue, collected):
            inject_queue.put_nowait("include the new requirement")
            return original_take(inject_queue, collected)

        async def failing_call() -> str:
            raise RuntimeError("planner connection dropped")

        with mock.patch.object(
            agent, "_take_pending_injections", take_after_putting,
        ):
            text, restarting = await agent._run_with_inject_watch(
                failing_call(), inject_q, injected,
            )

        self.assertIsNone(text)
        self.assertTrue(restarting)
        self.assertEqual(injected, ["include the new requirement"])

    async def test_final_merge_closes_a_bot_inject_window(self) -> None:
        class ClosingQueue(asyncio.Queue[str]):
            def __init__(self) -> None:
                super().__init__()
                self.closed = False

            def close(self) -> None:
                self.closed = True

        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            inject_q = ClosingQueue()
            with mock.patch.object(
                agent, "run_internal_backend", self._fake_internal,
            ), mock.patch.object(agent, "_drive_backend", self._fake_drive):
                _, restarting = await agent._run_flexible(
                    "context", "user message", ws,
                    approval="auto", effort=0, collaborative=False,
                    summary_backend_name="codex",
                    summary_model="gpt-5.6-luna", on_event=None,
                    inject_queue=inject_q, injected=[],
                )

        self.assertFalse(restarting)
        self.assertTrue(inject_q.closed)


class CoveragePromptTests(unittest.TestCase):
    def test_plan_cap_surfaces_partial_remainder(self) -> None:
        lines = [
            f"{index + 1}. [{['low', 'mid', 'high'][index % 3]}] task {index + 1}"
            for index in range(1, flexible.MAX_SUBTASKS + 5)
        ]
        plan = flexible.parse_plan(
            "[UNDERSTANDING]\nbig\n[/UNDERSTANDING]\n"
            "[PLAN]\n" + "\n".join(lines) + "\n[/PLAN]",
            "big request",
        )
        self.assertEqual(len(plan.subtasks), flexible.MAX_SUBTASKS)
        self.assertIn("PARTIAL", plan.understanding)
        self.assertIn("remainder", plan.understanding)

    def test_build_verify_requires_evidence_across_prompts(self) -> None:
        """Build checks must demand full-log evidence, never eyeball-only."""
        # A cohesive workflow still needs every verification stage and evidence.
        self.assertIn("Build/verify", flexible._PLANNER_RULES)
        self.assertIn("exit codes", flexible._PLANNER_RULES)
        self.assertIn("one cohesive task", flexible._PLANNER_RULES)
        self.assertIn("Never split workflow stages into different writers", flexible._PLANNER_RULES)
        self.assertIn("same owner for every target or related target group", flexible._PLANNER_RULES)
        self.assertIn("runtime launch/exercise/log-check", flexible._PLANNER_RULES)
        # Merge must refuse clean claims without per-report evidence.
        self.assertIn("exit codes", flexible._MERGE_RULES)
        self.assertIn("PARTIAL + remainder", flexible._MERGE_RULES)
        # Worker prompt must demand full-log grep + runtime exercise.
        plan = flexible.Plan(
            understanding="",
            subtasks=(flexible.Subtask(tier="high", instruction="verify"),),
        )
        worker = "\n".join(flexible.build_subtask_sections("verify build", plan, 0, []).values())
        self.assertIn("FULL logs", worker)
        self.assertIn("exit codes", worker)
        # Per-turn preamble must carry the same rule to direct turns.
        for collaborative in (True, False):
            hint = agent._capability_hint(
                collaborative, allow_detached_requests=False,
            )
            self.assertIn("Verify-with-evidence", hint)
            self.assertIn("exit codes", hint)
        # Bash hint must steer builds to capture + grep full logs.
        self.assertIn("FULL", BashTool.description)
        self.assertIn("grep", BashTool.description.lower())


class FlexibleUserReplyPolicyTests(unittest.TestCase):
    def test_worker_prompt_bans_handoff_notes(self) -> None:
        plan = flexible.Plan(
            understanding="",
            subtasks=(flexible.Subtask(tier="low", instruction="do it"),),
        )
        prompt = "\n".join(flexible.build_subtask_sections("req", plan, 0, []).values())
        self.assertIn("[followup:", prompt)
        self.assertIn("Never write 'Next turn should:'", prompt)


class JudgeVerdictTests(unittest.TestCase):
    def test_done_ships_the_draft(self) -> None:
        verdict = flexible.parse_judge(
            "[VERDICT]\nDONE\n[/VERDICT]\n",
        )
        self.assertFalse(verdict.should_continue)
        self.assertTrue(verdict.verified)

    def test_continue_names_missing_and_next(self) -> None:
        verdict = flexible.parse_judge(
            "[VERDICT]\nCONTINUE\n[/VERDICT]\n"
            "[MISSING]\nerror counts\n[/MISSING]\n"
            "[NEXT]\ngrep the full log\n[/NEXT]\n",
        )
        self.assertTrue(verdict.should_continue)
        self.assertEqual(verdict.missing, "error counts")
        self.assertEqual(verdict.next_instruction, "grep the full log")
        self.assertTrue(verdict.verified)

    def test_continue_without_next_does_not_verify_completion(self) -> None:
        verdict = flexible.parse_judge(
            "[VERDICT]\nCONTINUE\n[/VERDICT]\n",
        )
        self.assertFalse(verdict.should_continue)
        self.assertFalse(verdict.verified)

    def test_empty_or_unrecognized_judge_reply_does_not_verify_completion(self) -> None:
        for raw in ("", "looks fine", "[VERDICT]\nUNKNOWN\n[/VERDICT]", None):
            with self.subTest(raw=raw):
                verdict = flexible.parse_judge(raw)
                self.assertFalse(verdict.should_continue)
                self.assertFalse(verdict.verified)

    def test_judge_prompt_covers_request_and_draft(self) -> None:
        prompt = flexible.build_judge_prompt("fix it", "partial fix")
        self.assertIn("fix it", prompt)
        self.assertIn("partial fix", prompt)
        self.assertIn("[VERDICT]", prompt)
        self.assertIn("No tools", prompt)

    def test_long_judge_excerpt_retains_opening_and_final_evidence(self) -> None:
        draft = "REQUEST COVERAGE: all targets checked.\n" + "detail\n" * 3_000
        draft += "FINAL: every target passed; no remaining work."
        excerpt = flexible.bounded_draft_excerpt(draft)
        self.assertLessEqual(len(excerpt), flexible._DRAFT_MAX_CHARS)
        self.assertTrue(excerpt.startswith("REQUEST COVERAGE: all targets checked."))
        self.assertTrue(excerpt.endswith("FINAL: every target passed; no remaining work."))
        prompt = flexible.build_judge_prompt("check all targets", draft)
        self.assertIn(excerpt, prompt)
        self.assertNotIn(flexible.REPORT_TRUNCATION_MARKER, prompt)
        self.assertNotIn("say PARTIAL + remainder", excerpt)
        self.assertIn("Omission alone is not evidence of unfinished work", prompt)

    def test_judge_keeps_explicit_unfinished_work_in_draft(self) -> None:
        draft = "PARTIAL: retry.py is fixed; runtime verification remains."
        prompt = flexible.build_judge_prompt("fix and verify retries", draft)
        self.assertIn(draft, prompt)
        self.assertIn("CONTINUE", prompt)

    def test_draft_excerpt_obeys_explicit_and_tight_budgets(self) -> None:
        draft = "OPENING: completed work.\n" + "detail\n" * 1_000 + "FINAL: verified."
        for limit in (-1, 0, 1, 2, len(flexible.DRAFT_OMISSION_MARKER), 300):
            with self.subTest(limit=limit):
                excerpt = flexible.bounded_draft_excerpt(draft, limit=limit)
                self.assertEqual(len(excerpt), max(0, limit))
                self.assertNotIn("PARTIAL", excerpt)
        excerpt = flexible.bounded_draft_excerpt(draft, limit=300)
        self.assertTrue(excerpt.startswith("OPENING: completed work."))
        self.assertTrue(excerpt.endswith("FINAL: verified."))
        self.assertIn(flexible.DRAFT_OMISSION_MARKER, excerpt)
        self.assertEqual(flexible.bounded_draft_excerpt("short", limit=10), "short")


class FlexibleTurnCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_flexible_continuation_passes_evidence_to_the_actual_worker(self) -> None:
        prompts: list[str] = []
        original_draft = "Fixed retry.py. Evidence: pytest exited 0."

        async def planner(*_args, **kwargs):
            self.assertEqual(kwargs["label"], "Flexible planner")
            return "[PLAN]\n1. [low] finish retry verification\n[/PLAN]"

        async def worker(*args, **kwargs):
            prompts.append(args[2])
            self.assertEqual(kwargs["effort"], 100)
            return AgentResult(
                text=original_draft if len(prompts) == 1 else "Runtime verified."
            ), False

        verdicts = [
            (flexible.JudgeVerdict(True, "runtime evidence", "verify runtime"), False),
            (flexible.JudgeVerdict(False), False),
        ]
        with tempfile.TemporaryDirectory() as ws:
            from Cozter import session
            data = await asyncio.to_thread(session.create_session, ws, name="Retry work")
            await asyncio.to_thread(workspace.set_reasoning_effort, ws, 100)
            for tier in flexible.TIERS:
                await asyncio.to_thread(
                    workspace.set_flexible_backend_name, ws, tier, "meta",
                )
            with (
                mock.patch.object(agent, "run_internal_backend", planner),
                mock.patch.object(agent, "_drive_backend", worker),
                mock.patch.object(agent, "_judge_draft", side_effect=verdicts),
                mock.patch.object(agent, "_run_post_turn_maintenance", mock.AsyncMock()),
            ):
                result = await agent.run(
                    "fix and verify retries", ws, 1, backend_name="flexible",
                    summary_backend_name="meta", session_id=data["id"],
                )
                await asyncio.sleep(0)
        self.assertEqual(result.text, "Runtime verified.")
        self.assertEqual(len(prompts), 2)
        self.assertNotIn("Previous draft and evidence from this turn:", prompts[0])
        self.assertIn(original_draft, prompts[1])
        self.assertIn("verify runtime", prompts[1])
        self.assertIn("without repeating completed actions", prompts[1])

    async def test_failed_flexible_turn_skips_judge_and_auto_chain(self) -> None:
        async def planner(*_args, **kwargs):
            return "[PLAN]\n1. [low] inspect status\n[/PLAN]"

        with tempfile.TemporaryDirectory() as ws:
            from Cozter import session
            data = await asyncio.to_thread(session.create_session, ws, name="Status work")
            with (
                mock.patch.object(agent, "run_internal_backend", planner),
                mock.patch.object(
                    agent, "_drive_backend",
                    return_value=(AgentResult(error="429 usage limit reached"), False),
                ),
                mock.patch.object(
                    agent, "_judge_draft",
                    side_effect=AssertionError("failed provider must not be judged"),
                ),
                mock.patch.object(agent, "_run_post_turn_maintenance", mock.AsyncMock()),
            ):
                result = await agent.run(
                    "inspect status", ws, 1, backend_name="flexible",
                    summary_backend_name="meta", session_id=data["id"],
                )
                await asyncio.sleep(0)
        self.assertEqual(result.error, "429 usage limit reached")
        self.assertIn("429 usage limit reached", result.text)
        self.assertFalse(result.continue_instruction)

    async def test_mixed_success_and_terminal_error_skips_merge_judge_and_chain(self) -> None:
        for error in (
            "meta returned HTTP 429: quota exceeded",
            "meta stopped after 3 consecutive rounds where every tool call was refused",
        ):
            with self.subTest(error=error):
                calls = 0

                async def planner(*_args, **kwargs):
                    self.assertEqual(kwargs["label"], "Flexible planner")
                    return "[PLAN]\n1. [low] inspect status\n2. [high] repair retries\n[/PLAN]"

                async def worker(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    return (
                        AgentResult(text="Status inspected: retry.py needs repair.")
                        if calls == 1 else AgentResult(error=error)
                    ), False

                with tempfile.TemporaryDirectory() as ws:
                    from Cozter import session
                    data = await asyncio.to_thread(session.create_session, ws, name="Retry work")
                    with (
                        mock.patch.object(agent, "run_internal_backend", planner),
                        mock.patch.object(agent, "_drive_backend", worker),
                        mock.patch.object(
                            agent, "_judge_draft",
                            side_effect=AssertionError("terminal failures must not be judged"),
                        ),
                        mock.patch.object(agent, "_run_post_turn_maintenance", mock.AsyncMock()),
                    ):
                        result = await agent.run(
                            "inspect and repair retries", ws, 1, backend_name="flexible",
                            summary_backend_name="meta", session_id=data["id"],
                        )
                        await asyncio.sleep(0)
                self.assertEqual(calls, 2)
                self.assertEqual(error, result.error)
                self.assertIn("Status inspected: retry.py needs repair.", result.text)
                self.assertIn("repair retries", result.text)
                self.assertIn(error, result.text)
                self.assertIn("PARTIAL", result.text)
                self.assertFalse(result.continue_instruction)

    def test_terminal_provider_classification_is_narrow(self) -> None:
        for error in (
            "meta returned HTTP 429: too many requests",
            "zai returned HTTP401: unauthorized",
            "provider returned HTTP status 403: forbidden",
            "You have reached your usage limit",
            "insufficient_quota", "quota exhausted", "insufficient credits",
            "out of credits", "invalid API key", "authentication failed",
        ):
            with self.subTest(error=error):
                self.assertTrue(agent._terminal_provider_error(error))
        for error in (
            "provider unavailable", "HTTP 503", "connection reset", "request timed out",
            "expected status 429 in test response", "quota service temporarily unavailable",
        ):
            with self.subTest(error=error):
                self.assertFalse(agent._terminal_provider_error(error))

    def test_account_suppression_requires_explicit_quota_or_auth_failure(self) -> None:
        for error in (
            "meta returned HTTP401: unauthorized", "HTTP status 401: invalid credentials",
            "You have reached your usage limit", "usage limit reached", "insufficient_quota",
            "quota exhausted", "quota exceeded", "insufficient credits", "invalid API key",
            "authentication failed",
        ):
            with self.subTest(error=error):
                self.assertTrue(agent._account_provider_unavailable(error))
        for error in (
            "HTTP429: too many requests", "HTTP status 403: forbidden",
            "quota service temporarily unavailable", "expected status 401 in a test response",
            "consecutive rounds where every tool call was refused", "path escapes workspace",
        ):
            with self.subTest(error=error):
                self.assertFalse(agent._account_provider_unavailable(error))

    async def test_stage_timing_keeps_output_private_and_preserves_cancellation(self) -> None:
        async def successful_stage():
            return "private worker output"

        async def cancelled_stage():
            raise asyncio.CancelledError

        async def failed_stage():
            raise RuntimeError("private provider error")

        async def failed_worker():
            return AgentResult(error="private provider error"), False

        with self.assertLogs("Cozter.performance", level="INFO") as captured:
            result = await agent._run_with_stage_timing(
                successful_stage(), "flexible_worker", "meta", "muse-spark-1.3",
            )
            with self.assertRaises(asyncio.CancelledError):
                await agent._run_with_stage_timing(
                    cancelled_stage(), "flexible_planner", "meta", "muse-spark-1.2",
                )
            with self.assertRaises(RuntimeError):
                await agent._run_with_stage_timing(
                    failed_stage(), "judge", "meta", "muse-spark-1.2",
                )
            await agent._run_with_stage_timing(
                failed_worker(), "flexible_worker", "meta", "muse-spark-1.3",
            )
        self.assertEqual(result, "private worker output")
        self.assertEqual(len(captured.output), 8)
        starts = [record for record in captured.output if "event=start" in record]
        ends = [record for record in captured.output if "duration_seconds=" in record]
        self.assertEqual(len(starts), 4)
        self.assertEqual(len(ends), 4)
        self.assertIn("stage=flexible_worker backend=meta model=muse-spark-1.3", ends[0])
        self.assertIn("outcome=ok", ends[0])
        self.assertIn("outcome=cancelled", ends[1])
        self.assertIn("outcome=error", ends[2])
        self.assertIn("outcome=error", ends[3])
        self.assertNotIn("private worker output", "\n".join(captured.output))
        self.assertNotIn("private provider error", "\n".join(captured.output))

    async def test_stage_start_is_visible_while_the_call_is_suspended(self) -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        async def suspended_stage():
            entered.set()
            await release.wait()
            return "private worker output"

        with self.assertLogs("Cozter.performance", level="INFO") as captured:
            task = asyncio.create_task(agent._run_with_stage_timing(
                suspended_stage(), "flexible_worker", "codex", "gpt-6.1-sol",
                worker_index=2, worker_count=3,
            ))
            try:
                await asyncio.wait_for(entered.wait(), timeout=1)
                self.assertFalse(task.done())
                self.assertEqual(len(captured.output), 1)
                self.assertIn(
                    "stage=flexible_worker backend=codex model=gpt-6.1-sol"
                    " event=start worker_index=2 worker_count=3",
                    captured.output[0],
                )
                self.assertNotIn("duration_seconds=", captured.output[0])
            finally:
                release.set()
                result = await task
        self.assertEqual(result, "private worker output")
        self.assertEqual(len(captured.output), 2)
        self.assertIn("outcome=ok duration_seconds=", captured.output[1])
        self.assertIn("worker_index=2 worker_count=3", captured.output[1])
        self.assertNotIn("private worker output", "\n".join(captured.output))


class JudgeLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_judge_failure_retains_an_unverified_draft(self) -> None:
        verdict, restarting = await agent._judge_draft(
            "req", "draft", "/tmp",
            summary_backend_name="missing-backend",
            summary_model=None,
            inject_queue=None,
            injected=[],
        )
        self.assertFalse(restarting)
        self.assertFalse(verdict.should_continue)
        self.assertFalse(verdict.verified)

    async def test_awaiting_draft_skips_the_judge(self) -> None:
        with mock.patch.object(
            agent, "run_internal_backend",
            side_effect=AssertionError("judge must not run"),
        ):
            verdict, restarting = await agent._judge_draft(
                "req", "which one?\n\n[[await]]", "/tmp",
                summary_backend_name="codex",
                summary_model=None,
                inject_queue=None,
                injected=[],
            )
        self.assertFalse(restarting)
        self.assertFalse(verdict.should_continue)

    async def test_direct_turn_continues_then_ships(self) -> None:
        drafts = ["partial answer", "full answer"]
        calls = {"drive": 0}

        async def fast_drive(*_args, **_kwargs):
            text = drafts[min(calls["drive"], 1)]
            calls["drive"] += 1
            return AgentResult(text=text), False

        async def fake_judge(*_args, **_kwargs):
            round_no = _kwargs.get("round_no", 1)
            if round_no == 1:
                return flexible.JudgeVerdict(
                    should_continue=True,
                    missing="evidence",
                    next_instruction="add evidence",
                ), False
            return flexible.JudgeVerdict(should_continue=False), False

        with tempfile.TemporaryDirectory() as ws:
            from Cozter import session
            data = await asyncio.to_thread(session.create_session, ws, name="Work")
            backend = type(
                "JudgeTestBackend", (),
                {
                    "name": "judge-test",
                    "default_summary_model": "m",
                    "supports_typed_plugins": True,
                    "supports_plugin_prelude": False,
                    "supports_detached_tasks": False,
                },
            )()
            with (
                mock.patch.object(
                    agent.backends_agent, "get_backend", return_value=backend,
                ),
                mock.patch.object(agent, "_drive_backend", fast_drive),
                mock.patch.object(agent, "_judge_draft", fake_judge),
                mock.patch.object(
                    agent.compaction, "maybe_compact",
                    new_callable=mock.AsyncMock,
                ),
                mock.patch.object(
                    agent.titling, "maybe_auto_title",
                    new_callable=mock.AsyncMock,
                ),
            ):
                result = await agent.run(
                    "answer me", ws, 1,
                    backend_name="judge-test", session_id=data["id"],
                )
        self.assertEqual(result.text, "full answer")
        self.assertEqual(calls["drive"], 2)
        self.assertTrue(result.completion_verified)

    async def test_exhausted_judge_flags_auto_chain(self) -> None:
        calls = {"drive": 0}

        async def fast_drive(*_args, **_kwargs):
            calls["drive"] += 1
            return AgentResult(text="PARTIAL + remainder: still going"), False

        async def always_continue(*_args, **_kwargs):
            return flexible.JudgeVerdict(
                should_continue=True,
                missing="rest",
                next_instruction="keep going",
            ), False

        with tempfile.TemporaryDirectory() as ws:
            from Cozter import session
            data = await asyncio.to_thread(session.create_session, ws, name="Work")
            backend = type(
                "JudgeTestBackend", (),
                {
                    "name": "judge-test",
                    "default_summary_model": "m",
                    "supports_typed_plugins": True,
                    "supports_plugin_prelude": False,
                    "supports_detached_tasks": False,
                },
            )()
            with (
                mock.patch.object(
                    agent.backends_agent, "get_backend", return_value=backend,
                ),
                mock.patch.object(agent, "_drive_backend", fast_drive),
                mock.patch.object(agent, "_judge_draft", always_continue),
                mock.patch.object(
                    agent.compaction, "maybe_compact",
                    new_callable=mock.AsyncMock,
                ),
                mock.patch.object(
                    agent.titling, "maybe_auto_title",
                    new_callable=mock.AsyncMock,
                ),
            ):
                result = await agent.run(
                    "finish it", ws, 1,
                    backend_name="judge-test", session_id=data["id"],
                )
        # Capped loop exhausts (drive + follow-ups) and flags the remainder.
        self.assertEqual(calls["drive"], 1 + flexible.JUDGE_MAX_CONTINUES)
        self.assertTrue(result.continue_instruction)
        self.assertIn("keep going", result.continue_instruction)
        self.assertFalse(result.completion_verified)
        self.assertEqual(result.continuation_reason, "judge_limit")

    async def test_done_draft_carries_no_chain_flag(self) -> None:
        async def fast_drive(*_args, **_kwargs):
            return AgentResult(text="all done"), False

        async def always_done(*_args, **_kwargs):
            return flexible.JudgeVerdict(should_continue=False), False

        with tempfile.TemporaryDirectory() as ws:
            from Cozter import session
            data = await asyncio.to_thread(session.create_session, ws, name="Work")
            backend = type(
                "JudgeTestBackend", (),
                {
                    "name": "judge-test",
                    "default_summary_model": "m",
                    "supports_typed_plugins": True,
                    "supports_plugin_prelude": False,
                    "supports_detached_tasks": False,
                },
            )()
            with (
                mock.patch.object(
                    agent.backends_agent, "get_backend", return_value=backend,
                ),
                mock.patch.object(agent, "_drive_backend", fast_drive),
                mock.patch.object(agent, "_judge_draft", always_done),
                mock.patch.object(
                    agent.compaction, "maybe_compact",
                    new_callable=mock.AsyncMock,
                ),
                mock.patch.object(
                    agent.titling, "maybe_auto_title",
                    new_callable=mock.AsyncMock,
                ),
            ):
                result = await agent.run(
                    "finish it", ws, 1,
                    backend_name="judge-test", session_id=data["id"],
                )
        self.assertEqual(result.text, "all done")
        self.assertFalse(result.continue_instruction)
        self.assertTrue(result.completion_verified)


if __name__ == "__main__":
    unittest.main()
