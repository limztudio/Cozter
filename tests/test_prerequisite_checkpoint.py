"""Real flexible worker evidence survives blocked recovery and durable restore."""

import asyncio
import os
import tempfile
import unittest
from unittest import mock

from Cozter.tests.helpers import persist_queue_entry

from Cozter import agent, flexible, session, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.tests.test_reply_delivery import ReplyDeliveryBot


class PrerequisiteCheckpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_and_tail_blocked_verdicts_checkpoint_completed_flexible_workers(self):
        for tail in (False, True):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as ws:
                with mock.patch.object(workspace, "CONFIG_DIR", ws):
                    await self.check_blocked_checkpoint(ws, tail=tail)

    async def check_blocked_checkpoint(self, ws, *, tail):
        bot = ReplyDeliveryBot(ws, fail_final=False)
        sid = session.create_session(ws, name="Durable prerequisite recovery")["id"]
        original = "Repair sources, verify local tests and exercise the visible desktop."
        entry = await persist_queue_entry(bot, "u1", original, "chat")
        later = await persist_queue_entry(bot, "u1", "Later queued maintenance", "chat")
        queue = bot._ensure_message_queue("u1")
        queue.put_nowait((original, "chat", entry, False))
        queue.put_nowait(("Later queued maintenance", "chat", later, False))
        await bot._save_job_checkpoint("u1", entry, {
            "state": "running", "request": original, "workspace_path": ws, "session_id": sid,
        })
        for tier in flexible.TIERS:
            workspace.set_flexible_backend_name(ws, tier, "meta")
        artifact = os.path.join(ws, "repair-evidence.txt")
        with open(artifact, "w", encoding="utf-8") as output:
            output.write("Repairs and local tests verified")
        plans = 0

        async def internal(*_args, label, **_kwargs):
            nonlocal plans
            if label == "Flexible planner":
                plans += 1
                if plans > 1:
                    return "[PLAN]\n1. [low] Inspect remaining desktop readiness\n[/PLAN]"
                return (
                    "[PLAN]\n1. [low] Repair sources\n"
                    "2. [low] Verify local tests\n[/PLAN]"
                )
            if label == "Flexible merge":
                return "Source repairs and local tests passed; the required visible desktop is unavailable."
            raise AssertionError(f"Unexpected internal provider call: {label}")

        repair = AgentResult(
            text="Source repairs retained.\n[[attach: repair-evidence.txt]]",
            events=[ChatEvent(kind="text", content="[[attach: repair-evidence.txt]]")],
            usage={"input_tokens": 3},
        )
        verified = AgentResult(text="Local tests passed: 42 tests, zero failures.", usage={"input_tokens": 5})
        workers = [(repair, False), (verified, False)]
        verdicts = []
        if tail:
            verdicts.append((flexible.JudgeVerdict(True, "Desktop readiness", "Inspect desktop readiness"), False))
            workers.append((AgentResult(text="Desktop readiness probe confirmed Session 0.", usage={"input_tokens": 2}), False))
        verdicts.append((flexible.JudgeVerdict(
            False, "Required visible desktop is unavailable in Session 0",
            "Recover or recheck the authorized interactive desktop",
            verified=False, blocked=True,
        ), False))
        with (
            mock.patch.object(workspace, "get_run_config", return_value=("flexible", None, "summary", "full", "meta")),
            mock.patch.object(agent, "run_internal_backend", internal),
            mock.patch.object(agent, "_drive_backend", mock.AsyncMock(side_effect=workers)) as drive,
            mock.patch.object(agent, "_judge_draft", mock.AsyncMock(side_effect=verdicts)),
            mock.patch.object(agent, "_run_post_turn_maintenance", mock.AsyncMock()),
            mock.patch.object(flexible, "JUDGE_MAX_CONTINUES", 1),
        ):
            await bot._drain_message_queue("u1")
            await agent._wait_for_pending_session_writes(ws)
            await asyncio.sleep(0)
        self.assertEqual(drive.await_count, 3 if tail else 2)
        saved = await bot._load_job_checkpoint("u1", entry)
        self.assertEqual(saved["state"], "paused")
        self.assertEqual(saved["request"], original)
        self.assertEqual(saved["session_id"], sid)
        self.assertGreater(saved["next_retry_at"], 0)
        reports = saved["checkpoint"]["reports"]
        self.assertEqual(reports[0]["instruction"], "Repair sources")
        self.assertIn("Source repairs retained", reports[0]["report"])
        self.assertEqual(reports[1]["instruction"], "Verify local tests")
        self.assertIn("42 tests, zero failures", reports[1]["report"])
        if tail:
            self.assertEqual(reports[2]["instruction"], "Inspect remaining desktop readiness")
            self.assertIn("confirmed Session 0", reports[2]["report"])
        pending = "\n".join(saved["checkpoint"]["pending"])
        self.assertIn("Required visible desktop is unavailable", pending)
        self.assertIn("Preserve completed changes and valid build/test evidence", pending)
        self.assertEqual(saved["output"]["attachments"], [artifact])
        self.assertEqual(saved["output"]["usage"], {"input_tokens": 10 if tail else 8})
        self.assertEqual([row["id"] for row in bot._read_queue_file()["u1"]], [entry, later])
        self.assertIn("u1", bot._paused_jobs)
        self.assertFalse(bot.has_active_turns())
        restarted = ReplyDeliveryBot(ws, fail_final=False)
        with mock.patch.object(restarted, "_start_queue_drain"):
            await restarted.restore_queues()
        restored = await restarted._load_job_checkpoint("u1", entry)
        self.assertEqual(restored["checkpoint"], saved["checkpoint"])
        self.assertEqual([row[2] for row in restarted._message_queues["u1"]._queue], [entry, later])

