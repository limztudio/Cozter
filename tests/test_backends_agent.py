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
from Cozter.backends_agent.llama import LlamaBackend, _model_ids
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
        self.assertEqual(models, (
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.3-codex-spark",
        ))

    def test_codex_effort_uses_levels_supported_by_selected_picker_model(
        self,
    ) -> None:
        backend = CodexBackend()
        self.assertEqual(
            backend.effort_levels,
            ("low", "medium", "high", "xhigh", "max", "ultra"),
        )
        self.assertIsNone(backend.convert_effort(0))
        self.assertEqual(backend.convert_effort(1), "low")
        self.assertEqual(backend.convert_effort(100), "ultra")
        self.assertEqual(
            backend.effort_levels_for_model("gpt-5.6-luna"),
            ("low", "medium", "high", "xhigh", "max"),
        )
        self.assertEqual(
            backend.effort_levels_for_model("gpt-5.4"),
            ("low", "medium", "high", "xhigh"),
        )
        self.assertEqual(
            backend.effort_levels_for_model("custom-model"),
            ("low", "medium", "high", "xhigh"),
        )

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

        catalog_efforts = {
            model["slug"]: tuple(
                level["effort"]
                for level in model.get("supported_reasoning_levels", ())
            )
            for model in catalog
            if isinstance(model, dict)
            and model.get("visibility") == "list"
            and isinstance(model.get("slug"), str)
        }
        self.assertEqual(CodexBackend.model_effort_levels, catalog_efforts)

    def test_copilot_picker_matches_current_cli_capable_models(self) -> None:
        """Change-detector for the docs-derived tuple, not CLI verification.

        test_copilot_picker_matches_installed_cli_catalog is the authority on
        the real slugs, and it skips wherever the CLI is not installed.
        """
        self.assertEqual(CopilotBackend.available_models, (
            "auto",
            "claude-sonnet-5",
            "claude-sonnet-4.6",
            "claude-sonnet-4.5",
            "claude-haiku-4.5",
            "claude-fable-5",
            "claude-opus-4.8",
            "claude-opus-4.8-fast",
            "claude-opus-4.7",
            "claude-opus-4.6",
            "claude-opus-4.5",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.3-codex",
            "gpt-5.4-mini",
            "gpt-5-mini",
            "gemini-3.1-pro-preview",
            "gemini-3.5-flash",
            "kimi-k2.7-code",
        ))

    def test_copilot_effort_matches_current_cli_choices(self) -> None:
        backend = CopilotBackend()
        self.assertEqual(
            backend.effort_levels,
            ("low", "medium", "high", "xhigh", "max"),
        )
        self.assertIsNone(backend.convert_effort(0))
        self.assertEqual(backend.convert_effort(1), "low")
        self.assertEqual(backend.convert_effort(100), "max")

    def test_copilot_picker_matches_installed_cli_catalog(self) -> None:
        copilot = shutil.which("copilot")
        if not copilot:
            self.skipTest("copilot CLI is not installed")

        try:
            proc = subprocess.run(
                [copilot, "help", "config"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            self.skipTest("copilot help config timed out")

        if proc.returncode != 0:
            self.skipTest(
                f"copilot help config failed: {proc.stderr.strip()}",
            )

        models: list[str] = []
        in_model_section = False
        for line in proc.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("`model`:"):
                in_model_section = True
                continue
            if not in_model_section:
                continue
            if stripped.startswith('- "') and stripped.endswith('"'):
                models.append(stripped[3:-1])
                continue
            if models and stripped:
                break

        if not models:
            self.skipTest("copilot help config returned no model catalog")

        # ``auto`` is Cozter's supported sentinel and is not repeated in the
        # CLI's concrete-model configuration list.
        self.assertEqual(CopilotBackend.available_models[1:], tuple(models))

    def test_claude_code_picker_includes_current_models(self) -> None:
        models = ClaudeCodeBackend.available_models
        for model in (
            "sonnet",
            "opusplan[1m]",
            "fable[1m]",
            "claude-fable-5",
            "claude-sonnet-5",
            "claude-opus-4-8",
            "claude-opus-4-8[1m]",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-opus-4-5",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-6",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-sonnet-4-5-20250929[1m]",
        ):
            with self.subTest(model=model):
                self.assertIn(model, models)

    def test_claude_code_picker_excludes_ids_the_cli_rejects(self) -> None:
        """Guard the three ID shapes Claude Code's model registry refuses.

        Every one of these shipped in the picker at some point. The rules
        they violate are spelled out on ClaudeCodeBackend.available_models.
        """
        models = ClaudeCodeBackend.available_models
        for model in (
            # No dated snapshot is published from Opus/Sonnet 4.6 on, so a
            # date suffix here 404s.
            "claude-opus-4-6-20251101",
            "claude-sonnet-4-6-20251114",
            # Fast mode is the /fast session toggle, not a model ID.
            "claude-opus-4-6-fast",
            "claude-opus-4-7-fast",
            "claude-opus-4-8-fast",
            # Natively 1M, so these carry no [1m] suffix.
            "claude-sonnet-5[1m]",
            "claude-fable-5[1m]",
        ):
            with self.subTest(model=model):
                self.assertNotIn(model, models)


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

    def test_append_model_effort_args(self) -> None:
        backend = self._dummy("sh")
        backend.effort_levels = ("low", "high")
        cmd = ["tool"]

        backend.append_model_effort_args(
            cmd,
            "chosen-model",
            50,
            model_flag="--model",
            effort_flag="--effort",
        )

        self.assertEqual(
            cmd,
            ["tool", "--model", "chosen-model", "--effort", "high"],
        )

    def test_append_model_effort_args_supports_templates(self) -> None:
        backend = self._dummy("sh")
        backend.effort_levels = ("low", "high")
        cmd = ["tool"]

        backend.append_model_effort_args(
            cmd,
            None,
            1,
            model_flag="-m",
            effort_flag="-c",
            effort_template="model_reasoning_effort={effort}",
        )

        self.assertEqual(cmd, ["tool", "-c", "model_reasoning_effort=low"])

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

    def test_llama_effort_uses_openai_request_shape(self) -> None:
        self.assertEqual(
            LlamaBackend()._effort_fields(100),
            {"reasoning_effort": "high"},
        )

    def test_llama_model_ids_tolerate_malformed_payloads(self) -> None:
        for payload in (None, [], {}, {"data": None}, {"data": {}}):
            with self.subTest(payload=payload):
                self.assertEqual(_model_ids(payload), ())

        self.assertEqual(
            _model_ids({
                "data": [
                    {"id": "model-a"},
                    {"id": ""},
                    {"id": "model-a"},
                    {"id": "model-b"},
                    "bad entry",
                ],
            }),
            ("model-a", "model-b"),
        )


class ZaiBackendTests(unittest.TestCase):
    def test_defaults_are_selectable(self) -> None:
        models = ZaiBackend.available_models
        self.assertEqual(len(models), len(set(models)))
        self.assertIn(ZaiBackend.default_model, models)
        self.assertIn(ZaiBackend.default_summary_model, models)

    def test_picker_includes_current_text_models(self) -> None:
        self.assertEqual(ZaiBackend.available_models, (
            "glm-5.2",
            "glm-5.1",
            "glm-5-turbo",
            "glm-5",
            "glm-4.7",
            "glm-4.7-flash",
            "glm-4.7-flashx",
            "glm-4.6",
            "glm-4.5",
            "glm-4.5-air",
            "glm-4.5-x",
            "glm-4.5-airx",
            "glm-4.5-flash",
            "glm-4-32b-0414-128k",
        ))

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

    def test_glm_5_2_effort_uses_reasoning_request_shape(self) -> None:
        backend = ZaiBackend()
        self.assertEqual(backend._effort_fields(0, "glm-5.2"), {})
        self.assertEqual(
            backend._effort_fields(1, "glm-5.2"),
            {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "none",
            },
        )
        self.assertEqual(
            backend._effort_fields(50, "glm-5.2"),
            {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "medium",
            },
        )
        self.assertEqual(
            backend._effort_fields(100, "glm-5.2"),
            {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "max",
            },
        )

    def test_older_glm_effort_uses_thinking_switch(self) -> None:
        backend = ZaiBackend()
        self.assertEqual(
            backend._effort_fields(49, "glm-5.1"),
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(
            backend._effort_fields(50, "glm-5.1"),
            {"thinking": {"type": "enabled"}},
        )


if __name__ == "__main__":
    unittest.main()
