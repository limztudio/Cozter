"""Only effort 100 requests a selected model's highest graded effort."""

import tempfile
import time
import unittest
from unittest import mock

from Cozter.backends_agent import claude_code, codex, copilot, grok
from Cozter.backends_agent.base import effort_band
from Cozter.backends_agent.llama import LlamaBackend
from Cozter.backends_agent.meta import MetaModelApiBackend
from Cozter.backends_agent.zai import ZaiBackend


class EffortMaximumPolicyTests(unittest.TestCase):
    def test_lower_cutoffs_remain_stable_and_only_100_selects_maximum(self) -> None:
        levels = ("low", "medium", "high", "xhigh", "max")
        cases = (
            (0, None), (1, "low"), (19, "low"),
            (20, "medium"), (39, "medium"), (40, "high"), (59, "high"),
            (60, "xhigh"), (79, "xhigh"), (80, "xhigh"),
            (99, "xhigh"), (100, "max"),
        )
        for percent, expected in cases:
            with self.subTest(percent=percent):
                self.assertEqual(effort_band(percent, levels), expected)

    def test_maximum_is_reserved_for_every_supported_vocabulary_size(self) -> None:
        for levels in (
            ("low", "high"),
            ("low", "high", "max"),
            ("low", "medium", "high", "max"),
            ("low", "medium", "high", "xhigh", "max"),
            ("minimal", "low", "medium", "high", "xhigh", "max"),
            ("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        ):
            with self.subTest(levels=levels):
                self.assertIsNone(effort_band(0, levels))
                self.assertEqual(effort_band(1, levels), levels[0])
                self.assertEqual(effort_band(99, levels), levels[-2])
                self.assertEqual(effort_band(100, levels), levels[-1])
                self.assertNotIn(levels[-1], [effort_band(value, levels) for value in range(1, 100)])

    def test_empty_and_single_level_capabilities_do_not_invent_lower_values(self) -> None:
        for effort in (0, 1, 50, 99, 100):
            with self.subTest(effort=effort):
                self.assertIsNone(effort_band(effort, ()))
        for effort in (0, 1, 50, 99):
            with self.subTest(effort=effort):
                self.assertIsNone(effort_band(effort, ("ultra",)))
        self.assertEqual(effort_band(100, ("ultra",)), "ultra")

    def test_cli_model_capabilities_choose_their_own_penultimate_and_maximum(self) -> None:
        cases = (
            (claude_code.ClaudeCodeBackend(), "claude-opus-5-5", "xhigh", "max"),
            (claude_code.ClaudeCodeBackend(), "CLAUDE-OPUS-5-5[1M]", "xhigh", "max"),
            (claude_code.ClaudeCodeBackend(), "claude-opus-4-6[1m]", "high", "max"),
            (claude_code.ClaudeCodeBackend(), "claude-opus-4-5", None, None),
            (claude_code.ClaudeCodeBackend(), "haiku", None, None),
            (codex.CodexBackend(), "gpt-6.1-sol", "max", "ultra"),
            (codex.CodexBackend(), "gpt-5.6-luna", "xhigh", "max"),
            (codex.CodexBackend(), "company-model", "high", "xhigh"),
            (copilot.CopilotBackend(), "company-approved", "xhigh", "max"),
            (copilot.CopilotBackend(), " AUTO ", None, None),
            (copilot.CopilotBackend(), None, None, None),
            (grok.GrokBackend(), "grok-4.7", "high", "xhigh"),
            (grok.GrokBackend(), "grok-4.5", "medium", "high"),
            (grok.GrokBackend(), "company-grok", "medium", "high"),
        )
        for backend, model, lower, maximum in cases:
            for effort, expected in ((0, None), (99, lower), (100, maximum)):
                with self.subTest(backend=backend.name, model=model, effort=effort):
                    command: list[str] = []
                    backend.append_launch_options(command, model, effort, "auto")
                    flag = getattr(backend, "effort_flag", "--effort")
                    if expected is None:
                        self.assertNotIn("--effort", command)
                        self.assertNotIn("--reasoning-effort", command)
                    else:
                        self.assertEqual(command[command.index(flag) + 1], expected)

    def test_codex_cached_custom_catalog_is_authoritative_without_new_discovery(self) -> None:
        backend = codex.CodexBackend()
        backend._cached_model_catalog = (
            ("company-graded", "company-single", "company-fixed"),
            {"company-graded": ("medium", "high", "ultra"),
             "company-single": ("ultra",), "company-fixed": ()},
        )
        backend._catalog_expires_at = time.monotonic() + 60
        with mock.patch.object(backend, "_discover_models", side_effect=AssertionError("Must not probe")):
            for model, lower, maximum in (
                ("company-graded", "high", "ultra"),
                ("company-single", None, "ultra"),
                ("company-fixed", None, None),
            ):
                for effort, expected in ((99, lower), (100, maximum)):
                    with self.subTest(model=model, effort=effort):
                        command: list[str] = []
                        backend.append_launch_options(command, model, effort, "auto")
                        if expected is None:
                            self.assertNotIn("--effort", command)
                        else:
                            self.assertEqual(command[command.index("--effort") + 1], expected)

    def test_http_models_send_the_selected_capability_maximum_only_at_100(self) -> None:
        cases = (
            (LlamaBackend(), "local-model", "medium", "high"),
            (MetaModelApiBackend(), "muse-spark-1.3", "xhigh", "max"),
            (MetaModelApiBackend(), "muse-spark-1.3-contributor", "high", "xhigh"),
            (MetaModelApiBackend(), "muse-spark-1.2", "high", "xhigh"),
            (MetaModelApiBackend(), "private-muse", "medium", "high"),
            (ZaiBackend(), "glm-5.2", "high", "max"),
            (ZaiBackend(), "GLM-5.2[1M]", "high", "max"),
            (ZaiBackend(), "glm-5.3", "high", "max"),
            (ZaiBackend(), "GLM-5.3-FLASH[1M]", "high", "max"),
            (ZaiBackend(), "glm-5.3-flashx", "high", "max"),
        )
        for backend, model, lower, maximum in cases:
            with self.subTest(backend=backend.name, model=model):
                self.assertEqual(backend._effort_fields(0, model), {})
                for effort, expected in ((99, lower), (100, maximum)):
                    fields = backend._effort_fields(effort, model)
                    self.assertEqual(fields["reasoning_effort"], expected)
                    if backend.name == "zai":
                        self.assertEqual(fields["thinking"], {"type": "enabled"})

    def test_zai_binary_thinking_policy_retains_its_existing_threshold(self) -> None:
        backend = ZaiBackend()
        for effort, expected in ((0, None), (1, "disabled"), (49, "disabled"),
                                 (50, "enabled"), (99, "enabled"), (100, "enabled")):
            with self.subTest(effort=effort):
                fields = backend._effort_fields(effort, "glm-5.1")
                self.assertEqual(fields, {"thinking": {"type": expected}} if expected else {})


class EffortLaunchArgumentTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_cli_launches_use_the_maximum_policy_and_native_flag_shape(self) -> None:
        cases = (
            (claude_code, claude_code.ClaudeCodeBackend(), "claude-opus-5-5", "xhigh", "max"),
            (codex, codex.CodexBackend(), "gpt-6.1-sol", "max", "ultra"),
            (copilot, copilot.CopilotBackend(), "company-approved", "xhigh", "max"),
            (grok, grok.GrokBackend(), "grok-4.7", "high", "xhigh"),
        )
        with tempfile.TemporaryDirectory() as workspace:
            for module, backend, model, lower, maximum in cases:
                for effort, expected in ((99, lower), (100, maximum)):
                    with self.subTest(backend=backend.name, effort=effort):
                        process = mock.Mock(pid=123)
                        spawn_name = ("create_captured_subprocess" if backend.name == "grok"
                                      else "create_prompt_subprocess")
                        with (
                            mock.patch.object(module, "executable_command", return_value=[backend.name]),
                            mock.patch.object(module, spawn_name, new=mock.AsyncMock(return_value=process)) as spawn,
                        ):
                            await backend.launch(workspace, "Offline effort test", model, "auto", effort=effort)
                        try:
                            command = spawn.await_args.args[0]
                            if backend.name == "codex":
                                self.assertIn(f"model_reasoning_effort={expected}", command)
                            else:
                                flag = getattr(backend, "effort_flag", "--effort")
                                self.assertEqual(command[command.index(flag) + 1], expected)
                        finally:
                            await backend.cleanup_process(process)


if __name__ == "__main__":
    unittest.main()
