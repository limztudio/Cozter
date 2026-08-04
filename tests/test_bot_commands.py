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
from unittest import mock

from Cozter import session, workspace
from Cozter.backends_bot.base import BotContext
from Cozter.tests.helpers import TestBot


class _CmdBot(TestBot):
    def __init__(self) -> None:
        super().__init__(["u1"])
        self.replies: list[str] = []
        self.ai_texts: list[str] = []

    @property
    def platform_id(self) -> str:
        return "test:cmd"

    async def send_text(self, chat_id: str, text: str, *, rich: bool = False):
        self.replies.append(text)
        return None

    async def _ai_chat(self, ctx: BotContext) -> None:
        self.ai_texts.append(ctx.text)


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
        return self.bot.make_context(
            self.uid,
            "c1",
            text=text,
            args=args,
        )

    def _last(self) -> str:
        return self.bot.replies[-1]

    def _run(self, coro) -> None:
        asyncio.run(coro)

    # -- text command aliases ---------------------------------------------
    def test_backslash_command_alias_dispatches_with_arguments(self) -> None:
        self._run(self.bot.dispatch_text(self._ctx(text=r"\context 8000")))

        self.assertEqual(workspace.get_history_budget(self.ws), 8000)
        self.assertEqual(self.bot.ai_texts, [])

    def test_unknown_backslash_text_remains_chat_input(self) -> None:
        text = r"\newcommand is LaTeX, not a Cozter command"
        self._run(self.bot.dispatch_text(self._ctx(text=text)))

        self.assertEqual(self.bot.ai_texts, [text])

    def test_backslash_cancel_exits_pending_input_flow(self) -> None:
        self._run(self.bot.cmd_open(self._ctx()))
        self.assertIn(self.uid, self.bot._pending_input)

        self._run(self.bot.dispatch_text(self._ctx(text=r"\cancel")))

        self.assertNotIn(self.uid, self.bot._pending_input)
        self.assertEqual(self._last(), "Cancelled.")

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

    def test_effort_rejects_overlong_numeric_input(self) -> None:
        self._run(self.bot._receive_effort(self._ctx(text="9" * 5_000)))
        self.assertIn("Not a number", self._last())

    # -- /context ----------------------------------------------------------
    def test_context_sets_budget(self) -> None:
        self._run(self.bot.cmd_context(self._ctx(args="8000")))
        self.assertEqual(workspace.get_history_budget(self.ws), 8000)

    def test_numeric_commands_reject_overlong_decimal_input(self) -> None:
        too_long = "9" * 5_000

        for handler in (
            self.bot.cmd_compact,
            self.bot.cmd_context,
            self.bot.cmd_colony,
        ):
            with self.subTest(handler=handler.__name__):
                self._run(handler(self._ctx(args=too_long)))
                self.assertEqual(self._last(), "Error: number is too large.")

    # -- /doctor -----------------------------------------------------------
    def test_doctor_lists_every_direct_backend(self) -> None:
        self._run(self.bot.cmd_doctor(self._ctx()))
        out = self._last()
        for name in ("codex", "claude_code", "copilot", "llama", "zai"):
            self.assertIn(name, out)

    # -- /agent ------------------------------------------------------------
    def test_agent_picker_reserves_zero_for_flexible(self) -> None:
        self._run(self.bot.cmd_agent(self._ctx()))

        out = self._last()
        self.assertIn("  0. flexible", out)
        self.assertIn("  1. codex", out)

        self._run(self.bot._receive_agent(self._ctx(text="0")))
        self.assertEqual(workspace.get_backend_name(self.ws), "flexible")

    def test_flexible_model_picker_uses_selected_tier_backend(self) -> None:
        workspace.set_flexible_backend_name(self.ws, "high", "claude_code")

        self._run(self.bot._receive_flexible_model(
            self._ctx(text="opus"), tier="high",
        ))

        self.assertEqual(
            workspace.get_flexible_model(self.ws, "high"), "opus",
        )
        self.assertIn("Flexible high model set to: opus", self._last())

    def test_invalid_model_choices_reprompt_without_crashing(self) -> None:
        cases = (
            (
                self.bot._receive_model(self._ctx(text="missing")),
                "get_available_models",
            ),
            (
                self.bot._receive_summarymodel(self._ctx(text="missing")),
                "get_available_summary_models",
            ),
            (
                self.bot._receive_flexible_model(
                    self._ctx(text="missing"), tier="high",
                ),
                "get_available_flexible_models",
            ),
        )

        for handler, fetcher in cases:
            with self.subTest(fetcher=fetcher), mock.patch.object(
                workspace, fetcher, return_value=["known-model"],
            ):
                self._run(handler)
                self.assertIn("Unknown model: missing", self._last())
                self.assertIn(self.uid, self.bot._pending_input)

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
        ctx = self.bot.make_context("nobody", "c1")
        self._run(self.bot.cmd_permission(ctx))
        self.assertTrue(self.bot.replies)  # replied rather than raised


if __name__ == "__main__":
    unittest.main()
