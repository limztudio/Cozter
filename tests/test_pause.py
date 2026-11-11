"""Edge case: /pause holds queued work until next user input resumes."""

import os
import tempfile
import unittest
from unittest import mock

from Cozter import workspace
from Cozter.tests.helpers import OfflineBot


class _PauseBot(OfflineBot):
    def __init__(self) -> None:
        super().__init__(["u1"])
        self.replies: list[str] = []

    @property
    def platform_id(self) -> str:
        return "test:pause"

    async def send_text(self, _chat_id: str, text: str, *, rich: bool = False):
        self.replies.append(text)
        return None


class PauseCommandTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = self._tmp.name
        self._orig_state = workspace.WORKSPACE_STATE_PATH
        workspace.WORKSPACE_STATE_PATH = os.path.join(self.ws, "state.json")
        self.bot = _PauseBot()
        self._config = mock.patch.object(workspace, "CONFIG_DIR", self.ws)
        self._config.start()
        self.uid = "u1"
        workspace.select_workspace(self.uid, self.ws, self.bot.platform_id)

    async def asyncTearDown(self) -> None:
        self._config.stop()
        workspace.WORKSPACE_STATE_PATH = self._orig_state
        self._tmp.cleanup()

    def _ctx(self, *, command=None, args="", text=""):
        return self.bot.make_context(
            self.uid, "c1", text=text, command=command, args=args,
        )

    async def test_pause_registered_and_listed(self) -> None:
        self.assertIn("pause", self.bot._COMMANDS)
        await self.bot.dispatch_command(self._ctx(command="nope"))
        self.assertIn("Unknown command: /nope", self.bot.replies[-1])
        self.assertIn("/pause", self.bot.replies[-1])

    async def test_pause_dispatch_holds_and_next_command_resumes(self) -> None:
        await self.bot.dispatch_command(self._ctx(command="pause"))
        self.assertIn(self.uid, self.bot._manually_paused)
        self.assertTrue(self.bot._holds_queue(self.uid))
        self.assertIsNone(self.bot._start_queue_drain(self.uid))
        await self.bot.dispatch_command(self._ctx(command="version"))
        self.assertNotIn(self.uid, self.bot._manually_paused)
        self.assertIn("Paused", self.bot.replies[0])

    async def test_pause_holds_reply_delivery_until_resume(self) -> None:
        await self.bot.dispatch_command(self._ctx(command="pause"))
        record = {"id": "r1", "user_id": self.uid, "chat_id": "c1"}
        self.assertFalse(await self.bot._deliver_staged_reply(record))
        self.assertTrue(self.bot._resume_manual_pause(self.uid))

    async def test_stop_clears_manual_pause(self) -> None:
        await self.bot.dispatch_command(self._ctx(command="pause"))
        await self.bot.cmd_stop(self._ctx())
        self.assertNotIn(self.uid, self.bot._manually_paused)
        self.assertEqual(self.bot.replies[-1], "Cancelled.")


if __name__ == "__main__":
    unittest.main()
