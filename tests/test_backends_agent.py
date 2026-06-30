import unittest

from Cozter.backends_agent.claude_code import ClaudeCodeBackend
from Cozter.backends_agent.codex import CodexBackend
from Cozter.backends_agent.copilot import CopilotBackend


class StaticBackendModelTests(unittest.TestCase):
    def test_static_backend_defaults_are_selectable(self) -> None:
        for backend_cls in (
            CodexBackend,
            ClaudeCodeBackend,
            CopilotBackend,
        ):
            with self.subTest(backend=backend_cls.name):
                models = backend_cls.available_models
                self.assertEqual(len(models), len(set(models)))
                self.assertIn(backend_cls.default_model, models)
                self.assertIn(backend_cls.default_summary_model, models)

    def test_copilot_picker_includes_current_cli_capable_models(self) -> None:
        models = CopilotBackend.available_models
        for model in (
            "gpt-5.5",
            "gpt-5.4-mini",
            "gpt-5-mini",
            "claude-fable-5",
            "claude-opus-4.8",
            "claude-opus-4.8-fast",
            "claude-opus-4.6-fast",
            "gemini-2.5-pro",
        ):
            with self.subTest(model=model):
                self.assertIn(model, models)


if __name__ == "__main__":
    unittest.main()
