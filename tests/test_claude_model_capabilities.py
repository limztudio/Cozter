"""Regression coverage for Claude Code's model-specific CLI capabilities."""

import asyncio
import unittest
from unittest import mock

from Cozter.backends_agent import claude_code as claude_code_mod
from Cozter.backends_agent.claude_code import ClaudeCodeBackend


class ClaudeModelCapabilityTests(unittest.TestCase):
    def test_opus_45_api_effort_is_not_a_claude_code_capability(self) -> None:
        backend = ClaudeCodeBackend()
        for model in (
            "claude-opus-4-5",
            "claude-opus-4-5-20251101",
            " CLAUDE-OPUS-4-5[1M] ",
        ):
            with self.subTest(model=model):
                self.assertEqual(backend.effort_levels_for_model(model), ())
                for effort in (1, 50, 100):
                    command: list[str] = []
                    backend.append_launch_options(command, model, effort, "auto")
                    self.assertNotIn("--effort", command)
                    self.assertEqual(command[command.index("--model") + 1], model)

    def test_opus_45_print_launch_omits_unsupported_effort(self) -> None:
        async def run() -> None:
            with mock.patch.object(
                claude_code_mod, "create_prompt_subprocess",
                new=mock.AsyncMock(),
            ) as launch:
                await ClaudeCodeBackend().launch(
                    "/work", "prompt", "claude-opus-4-5-20251101", "auto",
                    effort=100,
                )
            command = launch.await_args.args[0]
            self.assertNotIn("--effort", command)
            self.assertIn("--print", command)
            self.assertEqual(launch.await_args.kwargs["cwd"], "/work")

        asyncio.run(run())

    def test_opus_45_detached_launch_omits_unsupported_effort(self) -> None:
        async def run() -> None:
            with mock.patch.object(
                claude_code_mod, "_run_claude_command",
                new=mock.AsyncMock(return_value=(0, "Backgrounded · task-123", "")),
            ) as launch:
                task_id = await ClaudeCodeBackend().launch_detached(
                    "/work", "prompt", "claude-opus-4-5", "auto", effort=100,
                )
            command = launch.await_args.args[0]
            self.assertNotIn("--effort", command)
            self.assertIn("--bg", command)
            self.assertEqual(command[-1], "prompt")
            self.assertEqual(task_id, "task-123")

        asyncio.run(run())

    def test_newer_models_keep_their_supported_effort_scales(self) -> None:
        backend = ClaudeCodeBackend()
        for model, expected in (
            ("claude-opus-4-6[1m]", ("low", "medium", "high", "max")),
            ("claude-sonnet-4-6", ("low", "medium", "high", "max")),
            ("claude-fable-5-1", ("low", "medium", "high", "xhigh", "max")),
            ("claude-opus-5-5", ("low", "medium", "high", "xhigh", "max")),
            ("claude-sonnet-5-5", ("low", "medium", "high", "xhigh", "max")),
        ):
            with self.subTest(model=model):
                self.assertEqual(backend.effort_levels_for_model(model), expected)
                command: list[str] = []
                backend.append_launch_options(command, model, 100, "auto")
                self.assertEqual(command[command.index("--effort") + 1], "max")


if __name__ == "__main__":
    unittest.main()
