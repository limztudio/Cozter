"""Missing external prerequisites must not create unlimited work retries."""

import tempfile
import unittest
from unittest import mock

from Cozter import agent, flexible, session
from Cozter.backends_agent.base import AgentResult
from Cozter.backends_bot.base import BotPlatform
from Cozter.tests.test_reply_delivery import _ReplyDeliveryBot


class BlockedJudgeRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_durable_job_pauses_instead_of_renewing_the_same_blocked_work(self):
        with tempfile.TemporaryDirectory() as ws:
            bot = _ReplyDeliveryBot(ws, fail_final=False)
            result = AgentResult(text="PARTIAL: graphics remain unverified")
            agent._retain_blocked_verdict(result, flexible.JudgeVerdict(
                False, "Visible desktop unavailable", "Supply an interactive desktop",
                verified=False, blocked=True,
            ))
            worker = mock.AsyncMock(return_value=result)
            with mock.patch.object(bot, "_run_turn_with_budget", worker):
                complete = await bot._run_turn("u1", "chat", "Exercise required graphics paths")
            self.assertFalse(complete)
            self.assertEqual(worker.await_count, 1)
            self.assertIn("u1", bot._paused_jobs)
            self.assertIn("Supply an interactive desktop", bot.sent[-1])

    async def test_initial_and_tail_verdict_preserve_work_without_auto_continuation(self):
        for tail in (False, True):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as ws:
                data = session.create_session(ws, name="Blocked execution")
                blocked = flexible.JudgeVerdict(
                    False, "Required graphics checks have no visible desktop",
                    "Provide a visible interactive desktop before resuming", verified=False, blocked=True,
                )
                verdicts = ([(flexible.JudgeVerdict(True, "verify", "verify remaining checks"), False)]
                            if tail else []) + [(blocked, False)]
                worker = mock.AsyncMock(return_value=(AgentResult(text="Build passed; source changes retained"), False))
                with (mock.patch.object(agent, "_drive_backend", worker),
                      mock.patch.object(agent, "_judge_draft", mock.AsyncMock(side_effect=verdicts)),
                      mock.patch.object(agent, "_run_post_turn_maintenance", mock.AsyncMock()),
                      mock.patch.object(flexible, "JUDGE_MAX_CONTINUES", 1)):
                    result = await agent.run("Build and exercise graphics", ws, 1,
                                             backend_name="meta", session_id=data["id"])
                self.assertEqual(worker.await_count, 2 if tail else 1)
                self.assertFalse(result.completion_verified)
                self.assertIsNone(result.continuation_reason)
                self.assertFalse(result.continue_instruction)
                self.assertIn("source changes retained", result.text)
                self.assertIn("Required graphics checks have no visible desktop", result.text)
                self.assertIn("Provide a visible interactive desktop", result.text)
                self.assertTrue(BotPlatform._job_result_is_incomplete(result))
                await agent._wait_for_pending_session_writes(ws)


if __name__ == "__main__":
    unittest.main()
