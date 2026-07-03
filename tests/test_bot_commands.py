"""Tests for bot command handlers, driven through a capturing fake bot.

Exercises the real BotPlatform command logic (validation + state mutation
+ replies) without any chat platform: a concrete subclass captures
send_text, and each handler runs under asyncio.run against a temp
workspace. The global workspace-state file is redirected to the tempdir.
"""

import asyncio
import os
import tempfile
import unittest

from Cozter import session, workspace
from Cozter.backends_bot.base import BotContext, BotPlatform


class _CmdBot(BotPlatform):
    def __init__(self) -> None:
        super().__init__([])
        self.replies: list[str] = []

    @property
    def platform_id(self) -> str:
        return "test:cmd"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send_text(self, chat_id: str, text: str, *, rich: bool = False):
        self.replies.append(text)
        return None

    async def edit_text(
        self, handle, text: str, *, rich: bool = False,
    ) -> None:
        pass

    async def delete_message(self, handle) -> None:
        pass

    async def send_file(self, chat_id: str, path: str) -> None:
        pass


class BotCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = self._tmp.name
        self._orig_state = workspace.WORKSPACE_STATE_PATH
        workspace.WORKSPACE_STATE_PATH = os.path.join(self.ws, "state.json")
        self.bot = _CmdBot()
        self.uid = "u1"
        workspace.select_workspace(self.uid, self.ws, self.bot.platform_id)

    def tearDown(self) -> None:
        workspace.WORKSPACE_STATE_PATH = self._orig_state
        self._tmp.cleanup()

    def _ctx(self, text: str = "", args: str = "") -> BotContext:
        return BotContext(
            user_id=self.uid, chat_id="c1", text=text,
            command=None, args=args, attachment=None, platform=self.bot,
        )

    def _last(self) -> str:
        return self.bot.replies[-1]

    def _run(self, coro) -> None:
        asyncio.run(coro)

    # -- /permission -------------------------------------------------------
    def test_permission_flow_sets_value(self) -> None:
        self._run(self.bot.cmd_permission(self._ctx()))
        self.assertIn("Available modes", self._last())
        self._run(self.bot._receive_permission(self._ctx(text="deny")))
        self.assertEqual(workspace.get_permission(self.ws), "deny")
        self.assertIn("deny", self._last())

    def test_permission_rejects_unknown(self) -> None:
        self._run(self.bot._receive_permission(self._ctx(text="bogus")))
        self.assertIn("Unknown mode", self._last())
        self.assertEqual(workspace.get_permission(self.ws), "auto")  # default

    # -- /style ------------------------------------------------------------
    def test_style_flow_sets_value(self) -> None:
        self._run(self.bot._receive_style(self._ctx(text="autonomous")))
        self.assertEqual(
            workspace.get_interaction_style(self.ws), "autonomous",
        )

    # -- /effort -----------------------------------------------------------
    def test_effort_flow_sets_value(self) -> None:
        self._run(self.bot._receive_effort(self._ctx(text="60")))
        self.assertEqual(workspace.get_reasoning_effort(self.ws), 60)

    def test_effort_rejects_out_of_range(self) -> None:
        self._run(self.bot._receive_effort(self._ctx(text="500")))
        self.assertIn("Out of range", self._last())

    # -- /context ----------------------------------------------------------
    def test_context_sets_budget(self) -> None:
        self._run(self.bot.cmd_context(self._ctx(args="8000")))
        self.assertEqual(workspace.get_history_budget(self.ws), 8000)

    # -- /doctor -----------------------------------------------------------
    def test_doctor_lists_every_backend(self) -> None:
        self._run(self.bot.cmd_doctor(self._ctx()))
        out = self._last()
        for name in ("codex", "claude_code", "copilot", "llama"):
            self.assertIn(name, out)

    # -- /sessions ---------------------------------------------------------
    def test_sessions_list_and_switch(self) -> None:
        first = session.create_session(self.ws, name="First")
        session.create_session(self.ws, name="Second")
        # Bare command lists them.
        self._run(self.bot.cmd_sessions(self._ctx()))
        listing = self._last()
        self.assertIn("First", listing)
        self.assertIn("Second", listing)
        # Switch by name.
        self._run(self.bot.cmd_sessions(self._ctx(args="First")))
        self.assertIn("Switched to session: First", self._last())
        self.assertEqual(
            session.get_last_session(self.ws, self.uid), first["id"],
        )

    def test_no_workspace_replies_gracefully(self) -> None:
        # A user with no selected workspace gets the no-workspace message,
        # not a crash.
        ctx = BotContext(
            user_id="nobody", chat_id="c1", text="", command=None,
            args="", attachment=None, platform=self.bot,
        )
        self._run(self.bot.cmd_permission(ctx))
        self.assertTrue(self.bot.replies)  # replied rather than raised


if __name__ == "__main__":
    unittest.main()
