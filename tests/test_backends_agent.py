import unittest

from Cozter import config
from Cozter.backends_agent import copilot as copilot_mod
from Cozter.backends_agent.base import Backend
from Cozter.backends_agent.claude_code import ClaudeCodeBackend
from Cozter.backends_agent.codex import CodexBackend
from Cozter.backends_agent.copilot import CopilotBackend
from Cozter.backends_agent.llama import LlamaBackend


class _DummyBackend(Backend):
    """Minimal concrete Backend for exercising base-class behavior."""

    async def launch(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    def parse_event(self, event, result) -> None:
        return None

    def extract_agent_text(self, event):
        return None


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
            "gpt-5.4-nano",
            "gpt-5-mini",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-opus-4.8",
            "claude-opus-4.8-fast",
            "gemini-2.5-pro",
            "gemini-3-flash",
            "gemini-3.1-pro",
            "gemini-3.5-flash",
            "raptor-mini",
            "kimi-k2.7-code",
        ):
            with self.subTest(model=model):
                self.assertIn(model, models)
        self.assertNotIn("claude-opus-4.6-fast", models)

    def test_claude_code_picker_includes_current_models(self) -> None:
        models = ClaudeCodeBackend.available_models
        for model in (
            "sonnet",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-opus-4-8",
        ):
            with self.subTest(model=model):
                self.assertIn(model, models)


class CopilotPromptCapTests(unittest.TestCase):
    def test_max_prompt_chars_is_platform_sane(self) -> None:
        cap = copilot_mod._max_prompt_chars()
        self.assertIsInstance(cap, int)
        # Never below the Windows floor, never absurdly large. On POSIX
        # (ARG_MAX ~2 MB) this lands well above the old fixed 28K cap.
        self.assertGreaterEqual(cap, 28_000)
        self.assertLessEqual(cap, 1_000_000)


class BackendHealthCheckTests(unittest.TestCase):
    def _dummy(self, executable: str) -> _DummyBackend:
        backend = _DummyBackend()
        backend.executable = executable
        return backend

    def test_missing_executable_reports_unhealthy(self) -> None:
        ok, detail = self._dummy(
            "cozter-nonexistent-binary-zzz",
        ).health_check()
        self.assertFalse(ok)
        self.assertIn("not found", detail)

    def test_present_executable_reports_healthy(self) -> None:
        # "sh" is on PATH on the CI runner and every POSIX dev box.
        ok, _ = self._dummy("sh").health_check()
        self.assertTrue(ok)

    def test_llama_health_check_unreachable(self) -> None:
        def _dead_url() -> str:
            return "http://127.0.0.1:1"

        orig = config.get_llama_server_url
        config.get_llama_server_url = _dead_url
        try:
            ok, detail = LlamaBackend().health_check()
        finally:
            config.get_llama_server_url = orig
        self.assertFalse(ok)
        self.assertIn("unreachable", detail)


if __name__ == "__main__":
    unittest.main()
