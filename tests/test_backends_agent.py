import asyncio
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from Cozter import config
from Cozter.backends_agent import claude_code as claude_code_mod
from Cozter.backends_agent import codex as codex_mod
from Cozter.backends_agent import copilot as copilot_mod
from Cozter.backends_agent.base import Backend
from Cozter.backends_agent.claude_code import ClaudeCodeBackend
from Cozter.backends_agent.codex import CodexBackend
from Cozter.backends_agent.copilot import CopilotBackend
from Cozter.backends_agent.llama import LlamaBackend
from Cozter.backends_agent._openai_agent import extract_model_ids
from Cozter.backends_agent import zai as zai_mod
from Cozter.backends_agent.zai import ZaiBackend


class _DummyBackend(Backend):
    """Minimal concrete Backend for exercising base-class behavior."""

    async def launch(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    def parse_event(self, event, result) -> None:
        return None

    def extract_agent_text(self, event):
        return None


class BackendPermissionCommandTests(unittest.TestCase):
    def _codex_command(
        self, approval: str, *, compaction: bool = False,
    ) -> tuple[str, ...]:
        async def run() -> tuple[str, ...]:
            proc = mock.Mock()
            with (
                mock.patch.object(
                    codex_mod, "executable_command", return_value=["codex"],
                ),
                mock.patch.object(
                    codex_mod,
                    "create_prompt_subprocess",
                    new=mock.AsyncMock(return_value=proc),
                ) as create_process,
            ):
                await CodexBackend().launch(
                    "/work", "summarize", None, approval,
                    compaction=compaction,
                )
            return tuple(create_process.await_args.args[0])

        return asyncio.run(run())

    def _claude_command(
        self, approval: str, *, compaction: bool = False,
    ) -> tuple[str, ...]:
        async def run() -> tuple[str, ...]:
            proc = mock.Mock()
            with (
                mock.patch.object(
                    claude_code_mod,
                    "executable_command",
                    return_value=["claude"],
                ),
                mock.patch.object(
                    claude_code_mod,
                    "create_prompt_subprocess",
                    new=mock.AsyncMock(return_value=proc),
                ) as create_process,
            ):
                await ClaudeCodeBackend().launch(
                    "/work", "summarize", None, approval,
                    compaction=compaction,
                )
            return tuple(create_process.await_args.args[0])

        return asyncio.run(run())

    def _copilot_command(
        self, approval: str, *, compaction: bool = False,
    ) -> tuple[str, ...]:
        async def run() -> tuple[str, ...]:
            with tempfile.TemporaryDirectory() as temp_dir:
                isolated_home = os.path.join(temp_dir, "copilot-home")
                os.mkdir(isolated_home)
                proc = mock.MagicMock()
                proc.pid = 123
                backend = CopilotBackend()
                with (
                    mock.patch.object(
                        copilot_mod,
                        "executable_command",
                        return_value=["copilot"],
                    ),
                    mock.patch.object(
                        copilot_mod,
                        "_create_isolated_copilot_home",
                        return_value=isolated_home,
                    ),
                    mock.patch.object(
                        copilot_mod.asyncio,
                        "create_subprocess_exec",
                        new=mock.AsyncMock(return_value=proc),
                    ) as create_process,
                ):
                    await backend.launch(
                        "/work", "summarize", None, approval,
                        compaction=compaction,
                    )
                command = tuple(create_process.await_args.args)
                await backend.cleanup_process(proc)
                return command

        return asyncio.run(run())

    def test_permission_argument_maps_are_explicit_and_fail_restricted(self) -> None:
        self.assertEqual(
            codex_mod._permission_args("full"),
            ["--dangerously-bypass-approvals-and-sandbox"],
        )
        self.assertEqual(codex_mod._permission_args("auto"), ["--full-auto"])
        for approval in ("confirm", "deny", "unknown"):
            with self.subTest(backend="codex", approval=approval):
                self.assertEqual(
                    codex_mod._permission_args(approval),
                    ["--sandbox", "read-only"],
                )

        self.assertEqual(
            claude_code_mod._permission_args("full"),
            ["--dangerously-skip-permissions"],
        )
        self.assertEqual(
            claude_code_mod._permission_args("auto"),
            ["--permission-mode", "acceptEdits"],
        )
        for approval in ("confirm", "deny", "unknown"):
            with self.subTest(backend="claude", approval=approval):
                self.assertEqual(
                    claude_code_mod._permission_args(approval),
                    ["--permission-mode", "plan"],
                )

        self.assertEqual(copilot_mod._permission_args("full"), ["--yolo"])
        self.assertEqual(
            copilot_mod._permission_args("auto"), ["--allow-all-tools"],
        )
        for approval in ("confirm", "deny", "unknown"):
            with self.subTest(backend="copilot", approval=approval):
                self.assertEqual(
                    copilot_mod._permission_args(approval),
                    ["--available-tools", ""],
                )

    def test_internal_compaction_command_never_grants_full_access(self) -> None:
        codex_command = self._codex_command("deny", compaction=True)
        self.assertIn("--sandbox", codex_command)
        self.assertEqual(
            codex_command[codex_command.index("--sandbox") + 1], "read-only",
        )
        self.assertNotIn(
            "--dangerously-bypass-approvals-and-sandbox", codex_command,
        )
        self.assertNotIn("--full-auto", codex_command)

        claude_command = self._claude_command("deny", compaction=True)
        self.assertIn("--permission-mode", claude_command)
        self.assertEqual(
            claude_command[claude_command.index("--permission-mode") + 1],
            "plan",
        )
        self.assertNotIn("--dangerously-skip-permissions", claude_command)
        self.assertNotIn("bypassPermissions", claude_command)

        copilot_command = self._copilot_command("deny", compaction=True)
        self.assertIn("--available-tools", copilot_command)
        self.assertEqual(
            copilot_command[copilot_command.index("--available-tools") + 1],
            "",
        )
        self.assertNotIn("--allow-all-tools", copilot_command)
        self.assertNotIn("--yolo", copilot_command)


class BackendModelTests(unittest.TestCase):
    def test_backend_catalogs_are_nonempty_and_deduped(self) -> None:
        for backend_cls in (
            CodexBackend,
            ClaudeCodeBackend,
            CopilotBackend,
        ):
            with self.subTest(backend=backend_cls.name):
                models = backend_cls().available_models
                self.assertTrue(models)
                self.assertEqual(len(models), len(set(models)))

    def test_codex_fallback_models_are_current_and_selectable(self) -> None:
        models = codex_mod._FALLBACK_MODELS
        self.assertEqual(models, (
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.3-codex-spark",
        ))
        self.assertIn(CodexBackend.default_model, models)
        self.assertIn(CodexBackend.default_summary_model, models)
        for model in CodexBackend.tier_models.values():
            self.assertIn(model, models)

    def test_codex_effort_uses_levels_supported_by_selected_picker_model(
        self,
    ) -> None:
        backend = CodexBackend()
        backend._cached_model_catalog = (
            codex_mod._FALLBACK_MODELS,
            codex_mod._FALLBACK_MODEL_EFFORT_LEVELS,
        )
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
            backend.effort_levels_for_model("gpt-5.4-mini"),
            ("low", "medium", "high", "xhigh"),
        )
        self.assertEqual(
            backend.effort_levels_for_model("custom-model"),
            ("low", "medium", "high", "xhigh"),
        )

    def test_codex_catalog_parser_uses_only_visible_models(self) -> None:
        payload = {
            "models": [
                {
                    "slug": "company-fast",
                    "visibility": "list",
                    "supported_reasoning_levels": [
                        {"effort": "low"},
                        {"effort": "high"},
                        {"effort": "low"},
                        {"effort": 123},
                    ],
                },
                {"slug": "hidden-model", "visibility": "hidden"},
                {
                    "slug": "company-fast", "visibility": "list",
                },
                {
                    "slug": "company-fixed",
                    "visibility": "list",
                    "supported_reasoning_levels": [],
                },
                {"slug": "", "visibility": "list"},
            ],
        }

        models, efforts = codex_mod._parse_debug_models_catalog(
            json.dumps(payload).encode("utf-8"),
        )

        self.assertEqual(models, ("company-fast", "company-fixed"))
        self.assertEqual(efforts, {
            "company-fast": ("low", "high"),
            "company-fixed": (),
        })

    def test_codex_catalog_parser_rejects_invalid_output(self) -> None:
        self.assertEqual(
            codex_mod._parse_debug_models_catalog(b"\xff\xfe\x00"),
            ((), {}),
        )

    def test_codex_discovery_caches_company_catalog(self) -> None:
        payload = json.dumps({
            "models": [{
                "slug": "company-model",
                "visibility": "list",
                "supported_reasoning_levels": [{"effort": "medium"}],
            }],
        }).encode("utf-8")
        completed = subprocess.CompletedProcess(
            ["codex", "debug", "models"], 0, stdout=payload, stderr=b"",
        )
        with (
            mock.patch.object(codex_mod.shutil, "which", return_value="codex"),
            mock.patch.object(
                codex_mod, "executable_command", return_value=["codex"],
            ),
            mock.patch.object(
                codex_mod.subprocess, "run", return_value=completed,
            ) as run_mock,
        ):
            backend = CodexBackend()
            self.assertEqual(backend.available_models, ("company-model",))
            self.assertEqual(backend.available_models, ("company-model",))
            backend._catalog_expires_at = 0
            self.assertEqual(backend.available_models, ("company-model",))
            self.assertEqual(
                backend.model_effort_levels,
                {"company-model": ("medium",)},
            )

        self.assertEqual(run_mock.call_count, 2)
        run_mock.assert_called_with(
            ["codex", "debug", "models"],
            capture_output=True,
            timeout=codex_mod._MODEL_DISCOVERY_TIMEOUT_SEC,
        )

    def test_codex_discovery_falls_back_after_failed_probes(self) -> None:
        failed = subprocess.CompletedProcess(
            ["codex", "debug", "models"], 1, stdout=b"", stderr=b"bad",
        )
        with (
            mock.patch.object(codex_mod.shutil, "which", return_value="codex"),
            mock.patch.object(
                codex_mod, "executable_command", return_value=["codex"],
            ),
            mock.patch.object(
                codex_mod.subprocess, "run", side_effect=[failed, failed],
            ) as run_mock,
        ):
            backend = CodexBackend()
            self.assertEqual(backend.available_models, codex_mod._FALLBACK_MODELS)
            self.assertEqual(
                backend.model_effort_levels,
                codex_mod._FALLBACK_MODEL_EFFORT_LEVELS,
            )

        self.assertEqual(run_mock.call_count, 2)

    def test_codex_discovery_recovers_from_stale_reasoning_config(self) -> None:
        failed = subprocess.CompletedProcess(
            ["codex", "debug", "models"], 1, stdout=b"", stderr=b"bad",
        )
        recovered = subprocess.CompletedProcess(
            ["codex", "debug", "models"],
            0,
            stdout=json.dumps({
                "models": [{
                    "slug": "company-model",
                    "visibility": "list",
                }],
            }).encode("utf-8"),
            stderr=b"",
        )
        with (
            mock.patch.object(codex_mod.shutil, "which", return_value="codex"),
            mock.patch.object(
                codex_mod, "executable_command", return_value=["codex"],
            ),
            mock.patch.object(
                codex_mod.subprocess,
                "run",
                side_effect=[failed, recovered],
            ) as run_mock,
        ):
            self.assertEqual(
                CodexBackend().available_models,
                ("company-model",),
            )

        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(
            run_mock.call_args_list[1].args[0],
            [
                "codex", "-c", 'model_reasoning_effort="high"',
                "debug", "models",
            ],
        )

    def test_codex_picker_matches_installed_cli_catalog(self) -> None:
        codex = shutil.which("codex")
        if not codex:
            self.skipTest("codex CLI is not installed")

        try:
            proc = subprocess.run(
                [codex, "debug", "models"],
                capture_output=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            self.skipTest("codex debug models timed out")

        if proc.returncode != 0:
            self.skipTest(
                "codex debug models failed: "
                f"{codex_mod._stderr_preview(proc.stderr)}",
            )

        visible_models, catalog_efforts = (
            codex_mod._parse_debug_models_catalog(proc.stdout)
        )
        if not visible_models:
            self.skipTest("codex debug models returned no visible models")

        backend = CodexBackend()
        self.assertEqual(backend.available_models, visible_models)
        self.assertEqual(backend.model_effort_levels, catalog_efforts)

    def test_copilot_fallback_is_policy_safe_auto_only(self) -> None:
        """A failed account probe must never revive generic model names."""
        self.assertEqual(copilot_mod._FALLBACK_MODELS, ("auto",))
        self.assertEqual(CopilotBackend.default_model, "auto")
        self.assertEqual(CopilotBackend.default_summary_model, "auto")
        self.assertEqual(CopilotBackend.tier_models, {})
        self.assertFalse(CopilotBackend.allow_unverified_extra_models)

    def test_copilot_isolated_home_copies_metadata_not_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_home = os.path.join(temp_dir, "source")
            isolated_home = os.path.join(temp_dir, "isolated")
            os.mkdir(source_home)
            os.mkdir(isolated_home)
            for name, content in (
                ("config.json", '{"lastLoggedInUser":"account"}'),
                ("settings.json", '{"model":"auto"}'),
            ):
                with open(os.path.join(source_home, name), "w", encoding="utf-8") as file:
                    file.write(content)
            history_dir = os.path.join(source_home, "session-state")
            os.mkdir(history_dir)
            with open(
                os.path.join(history_dir, "history.jsonl"),
                "w",
                encoding="utf-8",
            ) as file:
                file.write("old conversation")

            with (
                mock.patch.dict(
                    copilot_mod.os.environ,
                    {"COPILOT_HOME": source_home},
                    clear=False,
                ),
                mock.patch.object(
                    copilot_mod.tempfile,
                    "mkdtemp",
                    return_value=isolated_home,
                ),
            ):
                created_home = copilot_mod._create_isolated_copilot_home()

            self.assertEqual(created_home, isolated_home)
            for name in ("config.json", "settings.json"):
                self.assertTrue(os.path.isfile(os.path.join(isolated_home, name)))
            self.assertFalse(os.path.exists(os.path.join(isolated_home, "session-state")))

    def test_copilot_acp_parser_extracts_account_model_values(self) -> None:
        payload = {
            "sessionId": "catalog-only-session",
            "configOptions": [
                {
                    "id": "mode",
                    "category": "mode",
                    "type": "select",
                    "options": [{"value": "ask"}],
                },
                {
                    "id": "model",
                    "category": "model",
                    "type": "select",
                    "options": [
                        {"value": " auto "},
                        {"value": "company-allowed"},
                        {"value": " company-allowed "},
                        {"value": ""},
                        {"value": 42},
                        "bad",
                    ],
                },
            ],
        }
        self.assertEqual(
            copilot_mod._parse_acp_model_options(payload),
            ("auto", "company-allowed"),
        )

    def test_copilot_acp_metadata_catalog_beats_config_fallback(self) -> None:
        self.assertEqual(
            copilot_mod._parse_acp_model_options({
                "models": {
                    "availableModels": [
                        {"modelId": "company-allowed"},
                        {"modelId": "company-allowed"},
                    ],
                },
                "configOptions": [{
                    "id": "model", "category": "model",
                    "type": "select",
                    "options": [{"value": "generic-but-blocked"}],
                }],
            }),
            ("auto", "company-allowed"),
        )

    def test_copilot_acp_parser_rejects_missing_or_malformed_selector(self) -> None:
        for payload in (
            None,
            {},
            {"configOptions": {}},
            {"configOptions": []},
            {"configOptions": [{"id": "model", "type": "boolean"}]},
            {"configOptions": [{"id": "not-model", "type": "select",
                                "options": [{"value": "blocked"}]}]},
        ):
            with self.subTest(payload=payload):
                self.assertEqual(copilot_mod._parse_acp_model_options(payload), ())

    def test_copilot_discovery_uses_acp_not_generic_help_catalog(self) -> None:
        responses = "\n".join((
            json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "result": {"protocolVersion": 1, "agentCapabilities": {}},
            }),
            json.dumps({
                "jsonrpc": "2.0", "id": 2,
                "result": {
                    "sessionId": "catalog-only-session",
                    "configOptions": [{
                        "id": "model", "category": "model",
                        "type": "select",
                        "options": [{"value": "allowed-only"}],
                    }],
                },
            }),
        )) + "\n"
        proc = mock.MagicMock()
        proc.stdin = mock.MagicMock()
        proc.stdout = io.StringIO(responses)
        proc.poll.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            isolated_home = os.path.join(temp_dir, "copilot-home")
            os.mkdir(isolated_home)
            with (
                mock.patch.object(copilot_mod.shutil, "which", return_value="copilot"),
                mock.patch.object(copilot_mod, "executable_command", return_value=["copilot"]),
                mock.patch.object(
                    copilot_mod,
                    "_create_isolated_copilot_home",
                    return_value=isolated_home,
                ),
                mock.patch.object(copilot_mod.subprocess, "Popen", return_value=proc) as popen,
            ):
                self.assertEqual(
                    CopilotBackend().available_models,
                    ("auto", "allowed-only"),
                )

            self.assertEqual(popen.call_args.kwargs["env"]["COPILOT_HOME"], isolated_home)
            self.assertFalse(os.path.exists(isolated_home))

        command = popen.call_args.args[0]
        self.assertIn("--acp", command)
        self.assertIn("--stdio", command)
        self.assertNotIn("help", command)
        sent = [
            json.loads(call.args[0])
            for call in proc.stdin.write.call_args_list
        ]
        self.assertIn(
            {"jsonrpc": "2.0", "method": "initialized", "params": {}},
            sent,
        )
        proc.terminate.assert_called_once()

    def test_copilot_retries_failures_and_caches_only_success(self) -> None:
        backend = CopilotBackend()
        with mock.patch.object(
            backend,
            "_discover_models",
            side_effect=[None, ("auto", "allowed-only")],
        ) as discover:
            self.assertEqual(backend.available_models, ("auto",))
            # A short fallback throttle prevents an unavailable CLI from
            # spawning another ACP process for every picker interaction.
            self.assertEqual(backend.available_models, ("auto",))
            self.assertEqual(discover.call_count, 1)
            backend._fallback_expires_at = 0
            self.assertEqual(
                backend.available_models,
                ("auto", "allowed-only"),
            )
            self.assertEqual(
                backend.available_models,
                ("auto", "allowed-only"),
            )

        self.assertEqual(discover.call_count, 2)

    def test_copilot_stale_configured_model_fails_closed_to_auto(self) -> None:
        backend = CopilotBackend()
        backend._cached_models = ("auto", "company-allowed")
        backend._catalog_expires_at = time.monotonic() + 60
        self.assertEqual(
            backend.resolve_configured_model("company-allowed"),
            "company-allowed",
        )
        self.assertEqual(backend.resolve_configured_model("blocked-model"), "auto")

        backend._catalog_expires_at = 0
        self.assertEqual(
            backend.resolve_configured_model("company-allowed"), "auto",
        )

    def test_copilot_cmd_shim_cleanup_kills_its_process_tree(self) -> None:
        proc = mock.MagicMock()
        proc.pid = 12345
        proc.stdin = None
        proc.stdout = None
        proc.stderr = None
        proc.poll.return_value = 0

        with mock.patch.object(copilot_mod.subprocess, "run") as taskkill:
            copilot_mod._stop_acp_process(proc, kill_tree=True)

        taskkill.assert_called_once_with(
            ["taskkill", "/PID", "12345", "/T", "/F"],
            stdout=copilot_mod.subprocess.DEVNULL,
            stderr=copilot_mod.subprocess.DEVNULL,
            timeout=2,
            check=False,
        )

    def test_copilot_effort_matches_current_cli_choices(self) -> None:
        backend = CopilotBackend()
        self.assertEqual(
            backend.effort_levels,
            ("low", "medium", "high", "xhigh", "max"),
        )
        self.assertIsNone(backend.convert_effort(0))
        self.assertEqual(backend.convert_effort(1), "low")
        self.assertEqual(backend.convert_effort(100), "max")
        self.assertEqual(backend.effort_levels_for_model("auto"), ())
        self.assertEqual(backend.effort_levels_for_model(None), ())
        self.assertEqual(
            backend.effort_levels_for_model("company-allowed"),
            backend.effort_levels,
        )

    def test_copilot_auto_omits_unsupported_reasoning_effort(self) -> None:
        async def launch(model: str | None) -> tuple[str, ...]:
            with tempfile.TemporaryDirectory() as temp_dir:
                isolated_home = os.path.join(temp_dir, "copilot-home")
                os.mkdir(isolated_home)
                proc = mock.MagicMock()
                proc.pid = 123
                backend = CopilotBackend()
                with (
                    mock.patch.object(
                        copilot_mod,
                        "executable_command",
                        return_value=["copilot"],
                    ),
                    mock.patch.object(
                        copilot_mod,
                        "_create_isolated_copilot_home",
                        return_value=isolated_home,
                    ),
                    mock.patch.object(
                        copilot_mod.asyncio,
                        "create_subprocess_exec",
                        new_callable=mock.AsyncMock,
                        return_value=proc,
                    ) as create_process,
                ):
                    await backend.launch(
                        "C:/workspace",
                        "hello",
                        model,
                        "auto",
                        effort=100,
                    )
                self.assertEqual(
                    create_process.await_args.kwargs["env"]["COPILOT_HOME"],
                    isolated_home,
                )
                await backend.cleanup_process(proc)
                self.assertFalse(os.path.exists(isolated_home))
                return create_process.await_args.args

        auto_command = asyncio.run(launch("auto"))
        self.assertIn("--model", auto_command)
        self.assertIn("auto", auto_command)
        self.assertNotIn("--effort", auto_command)

        implicit_auto_command = asyncio.run(launch(None))
        self.assertNotIn("--model", implicit_auto_command)
        self.assertNotIn("--effort", implicit_auto_command)

        named_command = asyncio.run(launch("company-allowed"))
        self.assertIn("--model", named_command)
        self.assertIn("company-allowed", named_command)
        self.assertIn("--effort", named_command)
        self.assertIn("max", named_command)

    def test_claude_code_picker_includes_current_models(self) -> None:
        models = ClaudeCodeBackend.available_models
        for model in (
            "sonnet",
            "opusplan[1m]",
            "fable[1m]",
            "claude-fable-5",
            "claude-sonnet-5",
            "claude-opus-4-8",
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
            "claude-opus-4-8[1m]",
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
        # The interpreter running this test exists on every supported OS.
        ok, _ = self._dummy(sys.executable).health_check()
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

    def test_llama_model_catalog_refreshes_after_expiry(self) -> None:
        backend = LlamaBackend()
        with mock.patch.object(
            backend,
            "_fetch_models",
            side_effect=[("local-a",), ("local-b",)],
        ) as fetch:
            self.assertEqual(backend.available_models, ("local-a",))
            self.assertEqual(backend.available_models, ("local-a",))
            backend._catalog_expires_at = 0
            self.assertEqual(backend.available_models, ("local-b",))

        self.assertEqual(fetch.call_count, 2)

    def test_llama_model_ids_tolerate_malformed_payloads(self) -> None:
        for payload in (None, [], {}, {"data": None}, {"data": {}}):
            with self.subTest(payload=payload):
                self.assertEqual(extract_model_ids(payload), ())

        self.assertEqual(
            extract_model_ids({
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
    def test_fallback_models_are_current_and_selectable(self) -> None:
        models = zai_mod._FALLBACK_MODELS
        self.assertEqual(len(models), len(set(models)))
        self.assertIn(ZaiBackend.default_model, models)
        self.assertIn(ZaiBackend.default_summary_model, models)

    def test_fallback_picker_includes_current_text_models(self) -> None:
        self.assertEqual(zai_mod._FALLBACK_MODELS, (
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

    def test_model_ids_tolerate_malformed_payloads(self) -> None:
        for payload in (None, [], {}, {"data": None}, {"data": {}}):
            with self.subTest(payload=payload):
                self.assertEqual(extract_model_ids(payload), ())

        self.assertEqual(
            extract_model_ids({
                "data": [
                    {"id": "glm-company"},
                    {"id": " glm-private "},
                    {"id": "glm-company"},
                    {"id": ""},
                    {"id": 123},
                    "bad",
                ],
            }),
            ("glm-company", "glm-private"),
        )

    def test_available_models_queries_configured_account_once(self) -> None:
        response = mock.MagicMock()
        response.read.return_value = json.dumps({
            "data": [{"id": "glm-company"}, {"id": "glm-private"}],
        }).encode("utf-8")
        response.__enter__.return_value = response
        with (
            mock.patch.object(zai_mod.cfg, "get_zai_api_key", return_value="key"),
            mock.patch.object(
                zai_mod.cfg,
                "get_zai_base_url",
                return_value="https://models.example.test/v4/",
            ),
            mock.patch.object(
                zai_mod.urllib.request,
                "urlopen",
                return_value=response,
            ) as urlopen_mock,
        ):
            backend = ZaiBackend()
            self.assertEqual(
                backend.available_models,
                ("glm-company", "glm-private"),
            )
            self.assertEqual(
                backend.available_models,
                ("glm-company", "glm-private"),
            )
            backend._catalog_expires_at = 0
            self.assertEqual(
                backend.available_models,
                ("glm-company", "glm-private"),
            )

        self.assertEqual(urlopen_mock.call_count, 2)
        request = urlopen_mock.call_args.args[0]
        self.assertEqual(request.full_url, "https://models.example.test/v4/models")
        self.assertEqual(request.get_header("Authorization"), "Bearer key")
        self.assertEqual(
            urlopen_mock.call_args.kwargs["timeout"],
            zai_mod._MODEL_DISCOVERY_TIMEOUT_SEC,
        )

    def test_available_models_falls_back_without_key(self) -> None:
        with (
            mock.patch.object(zai_mod.cfg, "get_zai_api_key", return_value=""),
            mock.patch.object(zai_mod.urllib.request, "urlopen") as urlopen_mock,
        ):
            self.assertEqual(ZaiBackend().available_models, zai_mod._FALLBACK_MODELS)

        urlopen_mock.assert_not_called()

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
