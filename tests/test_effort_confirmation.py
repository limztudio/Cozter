"""Effort confirmations describe actual model behavior across tier overrides."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Cozter import workspace
from Cozter.backends_agent import (
    ClaudeCodeBackend, CodexBackend, CopilotBackend, GrokBackend,
    LlamaBackend, MetaModelApiBackend, ZaiBackend,
)
from Cozter.tests.helpers import OfflineBot


class OfflineEffortDescriptionTests(unittest.TestCase):
    def test_all_direct_backends_describe_defaults_and_edges_without_discovery(self) -> None:
        codex = CodexBackend()
        backends = (
            (codex, "gpt-6.1-sol", 99, "max"),
            (CopilotBackend(), "auto", 100, "no adjustable effort"),
            (ClaudeCodeBackend(), "claude-opus-4-5", 100, "no adjustable effort"),
            (GrokBackend(), "private-grok", 99, "medium"),
            (LlamaBackend(), "auto", 100, "high"),
            (MetaModelApiBackend(), "muse-wonder-1.0", 99, "xhigh"),
            (ZaiBackend(), "glm-5.1", 49, "thinking disabled"),
        )
        probes = [
            self.enterContext(mock.patch.object(
                codex, "_discover_models", side_effect=AssertionError("unexpected Codex probe"),
            )),
            self.enterContext(mock.patch(
                "urllib.request.urlopen", side_effect=AssertionError("unexpected HTTP probe"),
            )),
            self.enterContext(mock.patch(
                "subprocess.run", side_effect=AssertionError("unexpected CLI probe"),
            )),
            self.enterContext(mock.patch(
                "asyncio.create_subprocess_exec", side_effect=AssertionError("unexpected CLI launch"),
            )),
        ]
        for backend, model, percent, expected in backends:
            with self.subTest(backend=backend.name):
                self.assertEqual(backend.describe_effort(0, model), "backend default (no override)")
                self.assertIn(expected, backend.describe_effort(percent, model))

        with codex._model_catalog_lock:
            self.assertEqual(codex.describe_effort(99, "gpt-6.1-sol"), "max")
        for probe in probes:
            probe.assert_not_called()


class EffortConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.ws))
        self.enterContext(mock.patch.object(
            workspace, "WORKSPACE_STATE_PATH", str(Path(self.ws, "workspaces.json")),
        ))
        self.enterContext(mock.patch(
            "urllib.request.urlopen", side_effect=AssertionError("unexpected catalog request"),
        ))
        self.bot = OfflineBot(["owner"])
        self.bot.send_text = mock.AsyncMock()
        workspace.select_workspace("owner", self.ws, self.bot.platform_id)
        workspace.set_backend_name(self.ws, "meta")
        workspace.set_model(self.ws, "muse-wonder-1.0")
        for tier in workspace.FLEXIBLE_TIERS:
            workspace.set_flexible_backend_name(self.ws, tier, "meta")
            workspace.set_flexible_model(self.ws, tier, "muse-spark-1.2")

    def context(self, *, text="", command=None, args=""):
        return self.bot.make_context("owner", "room", text=text, command=command, args=args)

    def reply_line(self, model: str) -> str:
        reply = self.bot.send_text.call_args.args[1]
        matches = [line for line in reply.splitlines() if model in line]
        self.assertEqual(len(matches), 1, reply)
        return matches[0]

    async def test_direct_confirmation_uses_model_scale_and_only_100_is_max(self) -> None:
        await self.bot.cmd_effort(self.context())
        await self.bot.dispatch_text(self.context(text="99%"))
        line = self.reply_line("meta/muse-wonder-1.0")
        self.assertIn("→ xhigh", line)
        self.assertEqual(workspace.get_reasoning_effort(self.ws), 99)

        await self.bot.cmd_effort(self.context())
        await self.bot.dispatch_text(self.context(text="100"))
        line = self.reply_line("meta/muse-wonder-1.0")
        self.assertIn("→ max", line)
        self.assertEqual(workspace.get_reasoning_effort(self.ws), 100)

        workspace.set_backend_name(self.ws, "claude_code")
        workspace.set_model(self.ws, "claude-haiku-4-5")
        await self.bot._receive_effort(self.context(text="100"))
        line = self.reply_line("claude_code/claude-haiku-4-5")
        self.assertIn("no adjustable effort", line)
        self.assertNotIn("→ max", line)

    async def test_direct_arguments_and_invalid_retry_preserve_settings(self) -> None:
        await self.bot.dispatch_command(self.context(command="effort", args="99%"))
        self.assertIn("→ xhigh", self.reply_line("meta/muse-wonder-1.0"))
        self.assertNotIn("owner", self.bot._pending_input)
        await self.bot.dispatch_command(self.context(command="effort", args="100"))
        self.assertIn("→ max", self.reply_line("meta/muse-wonder-1.0"))
        for invalid, expected in (("101", "Out of range"), ("banana", "Not a number")):
            with self.subTest(invalid=invalid):
                await self.bot.dispatch_command(self.context(command="effort", args=invalid))
                self.assertIn(expected, self.bot.send_text.call_args.args[1])
                self.assertEqual(workspace.get_reasoning_effort(self.ws), 100)
                self.assertIn("owner", self.bot._pending_input)
        await self.bot.dispatch_text(self.context(text="99"))
        self.assertIn("→ xhigh", self.reply_line("meta/muse-wonder-1.0"))
        self.assertEqual(workspace.get_reasoning_effort(self.ws), 99)
        self.assertNotIn("owner", self.bot._pending_input)

    async def test_flexible_confirmation_keeps_each_override_and_reset_distinct(self) -> None:
        workspace.set_backend_name(self.ws, "flexible")
        for tier, name, model in (
            ("low", "claude_code", "claude-haiku-4-5"),
            ("mid", "meta", "muse-wonder-1.0"),
            ("high", "zai", "glm-5.3"),
        ):
            workspace.set_flexible_backend_name(self.ws, tier, name)
            workspace.set_flexible_model(self.ws, tier, model)
        workspace.set_flexible_reasoning_effort(self.ws, "low", 100)
        workspace.set_flexible_reasoning_effort(self.ws, "mid", 99)
        workspace.set_flexible_reasoning_effort(self.ws, "high", 100)

        await self.bot._receive_effort(self.context(text="25"))
        low = self.reply_line("claude_code/claude-haiku-4-5")
        self.assertIn("low:", low)
        self.assertIn("100% (tier override)", low)
        self.assertIn("no adjustable effort", low)
        mid = self.reply_line("meta/muse-wonder-1.0")
        self.assertIn("mid:", mid)
        self.assertIn("99% (tier override)", mid)
        self.assertIn("→ xhigh", mid)
        high = self.reply_line("zai/glm-5.3")
        self.assertIn("high:", high)
        self.assertIn("100% (tier override)", high)
        self.assertIn("→ max", high)

        await self.bot.dispatch_command(self.context(command="effort_flexible_mid", args="100"))
        self.assertIn("→ max", self.reply_line("meta/muse-wonder-1.0"))
        await self.bot.dispatch_command(self.context(command="effort_flexible_mid", args="default"))
        line = self.reply_line("meta/muse-wonder-1.0")
        self.assertIn("25% (inherited from workspace)", self.bot.send_text.call_args.args[1])
        self.assertIn("→ low", line)
        self.assertIsNone(workspace.get_flexible_reasoning_effort_override(self.ws, "mid"))

        await self.bot.dispatch_command(self.context(command="effort_flexible_mid", args="0"))
        line = self.reply_line("meta/muse-wonder-1.0")
        self.assertIn("0% (tier override)", self.bot.send_text.call_args.args[1])
        self.assertIn("backend default (no override)", line)
        self.assertEqual(workspace.get_flexible_reasoning_effort_override(self.ws, "mid"), 0)
        self.assertEqual(workspace.get_reasoning_effort(self.ws), 25)

    async def test_binary_thinking_confirmation_uses_actual_http_fields(self) -> None:
        workspace.set_backend_name(self.ws, "zai")
        workspace.set_model(self.ws, "glm-5.1")
        for percent, expected in (
            (49, "thinking disabled"),
            (50, "thinking enabled"),
            (0, "backend default (no override)"),
        ):
            with self.subTest(percent=percent):
                await self.bot._receive_effort(self.context(text=str(percent)))
                line = self.reply_line("zai/glm-5.1")
                self.assertIn("→ " + expected, line)

    async def test_glm_5_2_confirmation_does_not_hide_maximum_behind_alias(self) -> None:
        workspace.set_backend_name(self.ws, "zai")
        workspace.set_model(self.ws, "glm-5.2[1m]")
        for percent, expected in ((1, "none"), (99, "high"), (100, "max")):
            with self.subTest(percent=percent):
                await self.bot.dispatch_command(self.context(command="effort", args=str(percent)))
                line = self.reply_line("zai/glm-5.2[1m]")
                self.assertIn("→ " + expected, line)
                self.assertNotIn("→ xhigh", line)
                if percent < 100:
                    self.assertNotIn("→ max", line)


if __name__ == "__main__":
    unittest.main()
