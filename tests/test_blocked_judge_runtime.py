"""Blocked verdicts retain work and autonomous recovery instructions."""

import tempfile
import unittest
from unittest import mock

from Cozter import agent, flexible, session
from Cozter.backends_agent.base import AgentResult
from Cozter.backends_bot.base import BotPlatform


class BlockedJudgeRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocked_result_retains_work_and_prerequisite_for_automatic_recovery(self):
        for next_step in ("Recheck and recover the authorized interactive desktop", ""):
            with self.subTest(next_step=next_step):
                result = AgentResult(text="PARTIAL: graphics remain unverified; build passed")
                result.continuation_reason = "judge_limit"
                result.continue_instruction = "Previous continuation"
                missing = "Visible desktop unavailable"
                agent._retain_blocked_verdict(result, flexible.JudgeVerdict(
                    False, missing, next_step, verified=False, blocked=True,
                ))
                self.assertEqual(result.continuation_reason, "prerequisite")
                self.assertEqual(result.blocked_reason, missing)
                self.assertTrue(result.continue_instruction)
                self.assertFalse(result.completion_verified)
                self.assertIn("build passed", result.text)
                self.assertIn(missing, result.text)
                self.assertTrue(BotPlatform._job_result_is_incomplete(result))
                if next_step:
                    self.assertIn(next_step, result.continue_instruction)

    async def test_initial_and_tail_verdict_preserve_work_for_automatic_recovery(self):
        for tail in (False, True):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as ws:
                data = session.create_session(ws, name="Blocked execution")
                blocked = flexible.JudgeVerdict(
                    False, "Required graphics checks have no visible desktop",
                    "Recheck and recover the authorized visible desktop", verified=False, blocked=True,
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
                self.assertEqual(result.continuation_reason, "prerequisite")
                self.assertEqual(result.blocked_reason, "Required graphics checks have no visible desktop")
                self.assertIn("Recheck and recover the authorized visible desktop", result.continue_instruction)
                self.assertIn("source changes retained", result.text)
                self.assertIn("Required graphics checks have no visible desktop", result.text)
                self.assertIn("Recheck and recover the authorized visible desktop", result.text)
                self.assertTrue(BotPlatform._job_result_is_incomplete(result))
                await agent._wait_for_pending_session_writes(ws)


if __name__ == "__main__":
    unittest.main()
