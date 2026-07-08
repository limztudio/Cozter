import json
import shutil
import subprocess
import unittest

from Cozter import config
from Cozter.backends_agent import copilot as copilot_mod
from Cozter.backends_agent.base import Backend
from Cozter.backends_agent.claude_code import ClaudeCodeBackend
from Cozter.backends_agent.codex import CodexBackend
from Cozter.backends_agent.copilot import CopilotBackend
from Cozter.backends_agent.llama import LlamaBackend
from Cozter.backends_agent.zai import ZaiBackend


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

    def test_codex_picker_includes_current_codex_models(self) -> None:
        models = CodexBackend.available_models
        for model in (
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.3-codex-spark",
        ):
            with self.subTest(model=model):
                self.assertIn(model, models)
        # Deprecated or non-Codex-picker ids must not appear in the picker.
        self.assertNotIn("gpt-5.6", models)
        self.assertNotIn("gpt-5.4-nano", models)
        self.assertNotIn("gpt-5.3-codex", models)
        self.assertNotIn("gpt-5.2-codex", models)
        self.assertNotIn("gpt-5.1-codex", models)

    def test_codex_picker_matches_installed_cli_catalog(self) -> None:
        codex = shutil.which("codex")
        if not codex:
            self.skipTest("codex CLI is not installed")

        try:
            proc = subprocess.run(
                [codex, "debug", "models"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            self.skipTest("codex debug models timed out")

        if proc.returncode != 0:
            self.skipTest(f"codex debug models failed: {proc.stderr.strip()}")

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            self.fail(f"codex debug models returned invalid JSON: {exc}")

        catalog = payload.get("models")
        if not isinstance(catalog, list):
            self.fail("codex debug models returned no models list")

        visible_models = tuple(
            model["slug"]
            for model in catalog
            if isinstance(model, dict)
            and model.get("visibility") == "list"
            and isinstance(model.get("slug"), str)
        )
        if not visible_models:
            self.skipTest("codex debug models returned no visible models")

        self.assertEqual(CodexBackend.available_models, visible_models)

    def test_copilot_picker_includes_current_cli_capable_models(self) -> None:
        models = CopilotBackend.available_models
        for model in (
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5-mini",
            "gpt-5.3-codex",
            "claude-haiku-4.5",
            "claude-sonnet-5",
            "claude-sonnet-4.6",
            "claude-fable-5",
            "claude-opus-4.8",
            "claude-opus-4.8-fast",
            "claude-opus-4.5",
            "gemini-2.5-pro",
            "gemini-3-flash",
            "gemini-3.1-pro",
            "gemini-3.5-flash",
            "mai-code-1-flash",
            "raptor-mini",
            "kimi-k2.7-code",
        ):
            with self.subTest(model=model):
                self.assertIn(model, models)
        self.assertNotIn("claude-opus-4.6-fast", models)
        self.assertNotIn("gemini-3.1-pro-preview", models)

    def test_claude_code_picker_includes_current_models(self) -> None:
        models = ClaudeCodeBackend.available_models
        for model in (
            "sonnet",
            "opusplan[1m]",
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


class ZaiBackendTests(unittest.TestCase):
    def test_defaults_are_selectable(self) -> None:
        models = ZaiBackend.available_models
        self.assertEqual(len(models), len(set(models)))
        self.assertIn(ZaiBackend.default_model, models)
        self.assertIn(ZaiBackend.default_summary_model, models)

    def test_picker_includes_current_text_models(self) -> None:
        models = ZaiBackend.available_models
        for model in (
            "glm-5.2",
            "glm-5.1",
            "glm-5",
            "glm-5-turbo",
            "glm-4.7",
            "glm-4.7-flashx",
            "glm-4.7-flash",
            "glm-4.5-x",
            "glm-4.5-airx",
            "glm-4-32b-0414-128k",
        ):
            with self.subTest(model=model):
                self.assertIn(model, models)
        self.assertNotIn("glm-4-air", models)

    def test_chat_endpoint_appends_only_chat_completions(self) -> None:
        # Z.ai's base already carries /api/paas/v4, so no extra /v1.
        endpoint = ZaiBackend()._chat_endpoint()
        self.assertTrue(endpoint.endswith("/chat/completions"))
        self.assertNotIn("/v1/chat/completions", endpoint)

    def test_auth_headers_reflect_key(self) -> None:
        def _key() -> str:
            return "secret-key"

        def _nokey() -> str:
            return ""

        orig = config.get_zai_api_key
        try:
            config.get_zai_api_key = _key
            self.assertEqual(
                ZaiBackend()._auth_headers(),
                {"Authorization": "Bearer secret-key"},
            )
            config.get_zai_api_key = _nokey
            self.assertEqual(ZaiBackend()._auth_headers(), {})
        finally:
            config.get_zai_api_key = orig

    def test_health_check_reflects_key(self) -> None:
        def _key() -> str:
            return "k"

        def _nokey() -> str:
            return ""

        orig = config.get_zai_api_key
        try:
            config.get_zai_api_key = _nokey
            ok, detail = ZaiBackend().health_check()
            self.assertFalse(ok)
            self.assertIn("no API key", detail)
            config.get_zai_api_key = _key
            ok, _ = ZaiBackend().health_check()
            self.assertTrue(ok)
        finally:
            config.get_zai_api_key = orig

    def test_request_model_falls_back_to_default(self) -> None:
        backend = ZaiBackend()
        self.assertEqual(backend._request_model("glm-4.7"), "glm-4.7")
        self.assertEqual(backend._request_model(None), "glm-5.2")
        self.assertEqual(backend._request_model(""), "glm-5.2")

    def test_effort_maps_to_glm_supported_levels(self) -> None:
        backend = ZaiBackend()
        self.assertIsNone(backend.convert_effort(0))
        self.assertEqual(backend.convert_effort(1), "high")
        self.assertEqual(backend.convert_effort(49), "high")
        self.assertEqual(backend.convert_effort(50), "max")
        self.assertEqual(backend.convert_effort(100), "max")


if __name__ == "__main__":
    unittest.main()
