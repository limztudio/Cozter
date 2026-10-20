"""An exhausted account cannot become another Flexible continuation round."""

import asyncio
import tempfile
import unittest
from unittest import mock

from Cozter import agent, session, workspace
from Cozter.agent import AgentResult


MONTHLY_LIMIT = (
    "You've hit your monthly spend limit · raise it at "
    "https://claude.ai/settings/usage?from=cc_cli_limit_message · "
    "your session limit resets 1pm (Asia/Seoul)"
)


class FlexibleMonthlyLimitTests(unittest.IsolatedAsyncioTestCase):
    def test_explicit_monthly_exhaustion_is_terminal_but_configuration_failure_is_not(self):
        for error in (
            MONTHLY_LIMIT,
            "You have reached the monthly spending limit",
            "Account exceeded monthly spend limit",
            "Monthly spending limit has been exceeded",
            "monthly spend limit reached",
        ):
            with self.subTest(error=error):
                self.assertTrue(agent._account_provider_unavailable(error))
                self.assertTrue(agent._terminal_provider_error(error))
        for error in (
            "failed to configure monthly spend limit",
            "Monthly spending limit configuration failed",
            "Set a monthly spend limit to control account costs",
        ):
            with self.subTest(error=error):
                self.assertFalse(agent._account_provider_unavailable(error))
                self.assertFalse(agent._terminal_provider_error(error))

    async def test_monthly_limit_skips_same_account_and_stops_judge_auto_chain_with_partial_evidence(self):
        attempts = []
        internal = []

        async def planner(*_args, **kwargs):
            internal.append(kwargs["label"])
            self.assertEqual(kwargs["label"], "Flexible planner")
            return (
                "[PLAN]\n1. [low] inspect current status\n"
                "2. [high] repair renderer\n3. [high] verify renderer\n[/PLAN]"
            )

        async def worker(backend, *_args, **_kwargs):
            attempts.append(backend.name)
            return (
                AgentResult(text="Status inspected; renderer repair remains.")
                if len(attempts) == 1 else AgentResult(error=MONTHLY_LIMIT)
            ), False

        with tempfile.TemporaryDirectory() as root:
            data = await asyncio.to_thread(session.create_session, root, name="Renderer repair")
            await asyncio.to_thread(workspace.set_flexible_backend_name, root, "low", "meta")
            await asyncio.to_thread(workspace.set_flexible_backend_name, root, "high", "claude_code")
            with (
                mock.patch.object(agent, "run_internal_backend", planner),
                mock.patch.object(agent, "_drive_backend", worker),
                mock.patch.object(agent, "_run_flexible", wraps=agent._run_flexible) as flexible_run,
                mock.patch.object(agent, "_judge_draft", side_effect=AssertionError("exhausted account must not be judged")),
                mock.patch.object(agent, "_run_post_turn_maintenance", mock.AsyncMock()),
            ):
                result = await agent.run(
                    "inspect, repair, and verify renderer", root, 1, backend_name="flexible",
                    summary_backend_name="meta", session_id=data["id"],
                )
                await asyncio.sleep(0)
        self.assertEqual(attempts, ["meta", "claude_code"])
        self.assertEqual(internal, ["Flexible planner"])
        flexible_run.assert_awaited_once()
        self.assertEqual(result.error, MONTHLY_LIMIT)
        self.assertFalse(result.continue_instruction)
        self.assertIn("Status inspected; renderer repair remains.", result.text)
        self.assertIn("verify renderer", result.text)
        self.assertIn("This plan item remains unfinished.", result.text)
        self.assertIn(MONTHLY_LIMIT, result.text)
        self.assertIn("PARTIAL", result.text)
