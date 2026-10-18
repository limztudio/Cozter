"""Detached launches participate in the updater's idle handoff."""

import asyncio
import unittest
from unittest import mock

from Cozter.tests.helpers import OfflineBot


class UpdateBackgroundAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_update_handoff_rejects_new_detached_launch(self):
        bot = OfflineBot(["user"])
        bot._update_restart_pending = True
        ctx = bot.make_context("user", "chat", command="bg", args="run checks")
        with (
            mock.patch.object(bot, "send_text", new=mock.AsyncMock()) as reply,
            mock.patch.object(bot, "_run_background_command", new=mock.AsyncMock()) as launch,
        ):
            await bot.cmd_background(ctx)
        launch.assert_not_awaited()
        self.assertEqual(bot._detached_launch_tasks, {})
        self.assertIn("Update handoff", reply.await_args.args[1])

    async def test_shutdown_rejects_new_detached_launch(self):
        bot = OfflineBot(["user"])
        bot._stopping = True
        ctx = bot.make_context("user", "chat", command="bg", args="run checks")
        with (
            mock.patch.object(bot, "send_text", new=mock.AsyncMock()) as reply,
            mock.patch.object(bot, "_run_background_command", new=mock.AsyncMock()) as launch,
        ):
            await bot.cmd_background(ctx)
        launch.assert_not_awaited()
        self.assertEqual(bot._detached_launch_tasks, {})
        self.assertIn("stopping", reply.await_args.args[1])

    async def test_registered_detached_launch_is_visible_to_idle_recheck(self):
        bot = OfflineBot(["user"])
        ctx = bot.make_context("user", "chat", command="bg", args="run checks")
        entered, release = asyncio.Event(), asyncio.Event()

        async def run(*_args):
            entered.set()
            await release.wait()

        with mock.patch.object(bot, "_run_background_command", side_effect=run):
            command = asyncio.create_task(bot.cmd_background(ctx))
            try:
                await asyncio.wait_for(entered.wait(), timeout=1)
                self.assertTrue(bot.has_active_turns())
                self.assertEqual(len(bot._detached_launch_tasks["user"]), 1)
            finally:
                release.set()
                await command
        self.assertFalse(bot.has_active_turns())
        self.assertEqual(bot._detached_launch_tasks, {})
