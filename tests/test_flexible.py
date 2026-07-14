"""Tests for the flexible meta-agent.

Flexible understands a request with the summary model, splits it into
difficulty-graded sub-tasks, runs each on the agent+model bound to its
tier, and merges the workers' reports into one reply. These cover the
plan parser (a cheap model writes it, so it must be forgiving), the
per-tier settings, and the routing/merge loop itself.
"""

import tempfile
import unittest
from unittest import mock

from Cozter import agent, flexible, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.backends_bot.base import BotPlatform


class PlanParsingTests(unittest.TestCase):
    def test_parses_understanding_and_tiers(self) -> None:
        raw = (
            "[UNDERSTANDING]\nUser wants the retry bug fixed.\n"
            "[/UNDERSTANDING]\n"
            "[PLAN]\n"
            "1. [low] add a validation check\n"
            "2. [medium] write unit tests for the helper\n"
            "3. [high] debug the non-obvious retry race\n"
            "[/PLAN]\n"
        )
        plan = flexible.parse_plan(raw, "fix retries")

        self.assertEqual(plan.understanding, "User wants the retry bug fixed.")
        self.assertIsNone(plan.question)
        self.assertEqual(
            [(t.tier, t.instruction) for t in plan.subtasks],
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
        self.assertEqual([t.tier for t in plan.subtasks], ["high", "low"])

    def test_subtask_count_is_capped(self) -> None:
        raw = "[PLAN]\n" + "".join(
            f"{i}. [low] task {i}\n" for i in range(1, 20)
        ) + "[/PLAN]"
        plan = flexible.parse_plan(raw, "req")
        self.assertEqual(len(plan.subtasks), flexible.MAX_SUBTASKS)

    def test_lines_with_an_unknown_tier_are_skipped(self) -> None:
        plan = flexible.parse_plan(
            "[PLAN]\n1. [urgent] nope\n2. [low] yep\n[/PLAN]", "req",
        )
        self.assertEqual(
            [(t.tier, t.instruction) for t in plan.subtasks], [("low", "yep")],
        )

    def test_unparseable_output_falls_back_to_one_strong_task(self) -> None:
        for raw in ("", "   ", "no idea, sorry", "[PLAN]\n[/PLAN]"):
            plan = flexible.parse_plan(raw, "the original request")
            self.assertEqual(len(plan.subtasks), 1)
            self.assertEqual(plan.subtasks[0].tier, flexible.FALLBACK_TIER)
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


class FlexibleSettingsTests(unittest.TestCase):
    def test_a_fresh_workspace_has_every_tier_wired_up(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            self.assertEqual(
                workspace.get_flexible_run_config(ws),
                {
                    "low": ("codex", "gpt-5.4-mini"),
                    "mid": ("codex", "gpt-5.6-luna"),
                    "high": ("codex", "gpt-5.6-sol"),
                },
            )

    def test_tier_model_defaults_follow_the_tier_agent(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            workspace.set_flexible_backend_name(ws, "high", "claude_code")

            self.assertEqual(workspace.get_flexible_model(ws, "high"), "opus")
            # Rebinding one tier leaves the others alone.
            self.assertEqual(
                workspace.get_flexible_model(ws, "low"), "gpt-5.4-mini",
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
            workspace.set_flexible_model(ws, "mid", "gpt-5.5")
            workspace.set_flexible_backend_name(ws, "mid", "claude_code")
            workspace.set_flexible_model(ws, "mid", "sonnet")
            self.assertEqual(workspace.get_flexible_model(ws, "mid"), "sonnet")

            # Switching back restores the model codex was last set to.
            workspace.set_flexible_backend_name(ws, "mid", "codex")
            self.assertEqual(workspace.get_flexible_model(ws, "mid"), "gpt-5.5")

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

    def test_flexible_is_the_default_agent(self) -> None:
        self.assertEqual(workspace.DEFAULT_BACKEND, "flexible")
        self.assertNotIn("flexible", workspace.DIRECT_BACKENDS)

    def test_every_tier_command_is_registered(self) -> None:
        for tier in workspace.FLEXIBLE_TIERS:
            self.assertIn(f"agent_flexible_{tier}", BotPlatform._COMMANDS)
            self.assertIn(f"model_flexible_{tier}", BotPlatform._COMMANDS)


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
            workspace.set_flexible_backend_name(ws, "high", "claude_code")

            result, restarting = await self._run(ws)

        self.assertFalse(restarting)
        self.assertEqual(
            self.driven, [("codex", "gpt-5.4-mini"), ("claude_code", "opus")],
        )
        self.assertEqual(
            self.internal, ["Flexible planner", "Flexible merge"],
        )

    async def test_only_the_merged_answer_is_sent_to_the_user(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, _ = await self._run(ws)

        # The workers' own text is internal — it feeds the merge step, and
        # would otherwise be posted to the chat as extra messages.
        texts = [e.content for e in result.events if e.kind == "text"]
        self.assertEqual(texts, ["merged answer"])
        self.assertEqual(result.text, "merged answer")
        # Their tool events still stream through as the turn's visible trace.
        self.assertEqual(
            len([e for e in result.events if e.kind == "tool"]), 2,
        )

    async def test_worker_usage_is_summed_into_one_total(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, _ = await self._run(ws)

        self.assertEqual(
            result.usage, {"input_tokens": 20, "output_tokens": 10},
        )

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
        self.assertEqual(self.driven, [])  # no worker ran
        self.assertEqual(self.internal, ["Flexible planner"])  # no merge

    async def test_an_autonomous_turn_never_stops_to_ask(self) -> None:
        self.plan_output = "[QUESTION]\nWhich retry path?\n[/QUESTION]"
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, _ = await self._run(ws, collaborative=False)

        # A scheduled turn has nobody to answer, so the question degrades
        # into the fallback plan: one worker on the strongest tier.
        self.assertNotIn("[[await]]", result.text)
        self.assertEqual(self.driven, [("codex", "gpt-5.6-sol")])

    async def test_a_merged_answer_may_end_on_a_blocking_question(
        self,
    ) -> None:
        # The merge writes the reply the user reads, so it is the only step
        # after the planner that can stop the turn: the marker must survive
        # to _send_result, which is what pauses the queue.
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
        # Nobody is there to answer a scheduled run, so a marker the merge
        # model emits anyway is dropped rather than left to pause nothing.
        self.merge_output = "Done, but which retry path?\n\n[[await]]"
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, _ = await self._run(ws, collaborative=False)

        self.assertNotIn("[[await]]", result.text)
        self.assertIn("which retry path?", result.text)

    async def test_a_blocked_worker_pauses_the_turn_without_the_marker(
        self,
    ) -> None:
        # Workers run under the autonomy policy, so one that asks anyway is
        # genuinely stuck. The merge model is a cheap one and may relay the
        # question as prose — the pause must not depend on it remembering
        # the marker, or the user's answer lands as an unrelated new turn.
        self.worker_texts = [
            "Blocked. Which retry path?\n\n[[await]]", "high report",
        ]
        self.merge_output = "I need to know: which retry path should I use?"
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, _ = await self._run(ws, collaborative=True)

        _, awaiting = agent.extract_await(result.text)
        self.assertTrue(awaiting)
        # The merge is told which sub-task blocked, so it can surface the
        # right question out of several reports.
        self.assertIn("BLOCKED", self.merge_prompt)
        # The worker's own marker stays internal; only the merged reply
        # carries the one the bot acts on.
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
        # Nothing blocked and the merge left the marker off, so the trailing
        # question is a suggestion, not a blocker. Pausing here would wedge
        # the queue on every "want me to also...?".
        self.merge_output = "Fixed the check. Want me to add tests too?"
        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            result, _ = await self._run(ws, collaborative=True)

        _, awaiting = agent.extract_await(result.text)
        self.assertFalse(awaiting)

    async def test_reports_are_concatenated_when_the_merge_agent_is_gone(
        self,
    ) -> None:
        async def missing_merge(backend, ws_path, prompt, model, **kwargs):
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
        async def silent_worker(
            backend, ws_path, prompt, model, approval, **kwargs,
        ):
            result = AgentResult()
            # Tool events but no text - the shape a backend leaves behind
            # when it dies partway through a turn.
            result.events.append(ChatEvent(kind="tool", content="ran a tool"))
            return result, False

        async def silent_merge(backend, ws_path, prompt, model, **kwargs):
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
        # The reply names the agent+model that went quiet, so the user knows
        # which tier to go fix rather than staring at a blank answer.
        self.assertIn(f"{low_agent}/{low_model}", result.text)

    async def test_a_failing_worker_reports_why_it_went_quiet(self) -> None:
        """A backend error must reach the user, not just the journal."""
        async def failing_worker(
            backend, ws_path, prompt, model, approval, **kwargs,
        ):
            result = AgentResult()
            result.error = "429 usage limit reached"
            return result, False

        async def silent_merge(backend, ws_path, prompt, model, **kwargs):
            if kwargs["label"] == "Flexible planner":
                return self.plan_output
            return ""  # the same broken agent runs the merge

        with tempfile.TemporaryDirectory() as ws:
            workspace.ensure_cozter_dir(ws)
            with mock.patch.object(
                agent, "run_internal_backend", silent_merge,
            ), mock.patch.object(agent, "_drive_backend", failing_worker):
                result, _ = await agent._run_flexible(
                    "context", "user message", ws,
                    approval="auto", effort=0, collaborative=False,
                    summary_backend_name="codex",
                    summary_model="gpt-5.6-luna",
                    on_event=None, inject_queue=None, injected=[],
                )

        self.assertIn("429 usage limit reached", result.text)


if __name__ == "__main__":
    unittest.main()
