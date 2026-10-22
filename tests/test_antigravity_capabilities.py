"""Unsupported internal roles cannot inherit Antigravity's write access."""

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Cozter import agent, backends_agent, session, workspace
from Cozter.backends_agent.base import AgentResult
from Cozter.tests.helpers import temporary_config


class AntigravityCapabilityBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(temporary_config({}))
        self.enterContext(mock.patch(
            "Cozter.backends_agent.antigravity_cli.AntigravityCliBackend._fetch_models",
            return_value=("gemini-4-argon-medium", "gemini-4-argon-high", "gemini-3.8-flash-medium"),
        ))
        self.backend = backends_agent.get_backend("antigravity_cli")
        self.summary = backends_agent.get_backend(backends_agent.DEFAULT_DIRECT_BACKEND)

    def test_chat_and_flexible_selection_cannot_save_an_unsupported_summary_role(self):
        workspace.set_backend_name(self.ws, "antigravity_cli")
        workspace.set_model(self.ws, self.backend.default_model)
        for tier in workspace.FLEXIBLE_TIERS:
            workspace.set_flexible_backend_name(self.ws, tier, "antigravity_cli")
        workspace.set_summary_backend_name(self.ws, self.summary.name)
        workspace.set_summary_model(self.ws, self.summary.default_summary_model)
        path = Path(workspace._settings_path(self.ws))
        original = path.read_bytes()

        with self.assertRaisesRegex(ValueError, "summary backend"):
            workspace.set_summary_backend_name(self.ws, "antigravity_cli")

        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(workspace.get_backend_name(self.ws), "antigravity_cli")
        self.assertEqual(workspace.get_model(self.ws), self.backend.default_model)
        for tier in workspace.FLEXIBLE_TIERS:
            self.assertEqual(workspace.get_flexible_backend_name(self.ws, tier), "antigravity_cli")
        self.assertEqual(workspace.get_summary_backend_name(self.ws), self.summary.name)
        self.assertIn("antigravity_cli", backends_agent.DIRECT_BACKENDS)
        self.assertNotIn("antigravity_cli", backends_agent.SUMMARY_BACKENDS)

    def test_stale_unsupported_summary_uses_supported_model_without_rewriting_saved_settings(self):
        workspace.set_backend_name(self.ws, "antigravity_cli")
        workspace.set_summary_backend_name(self.ws, self.summary.name)
        workspace.set_summary_model(self.ws, self.summary.default_summary_model)
        settings = workspace._load_settings(self.ws)
        settings["summary_backend"] = "antigravity_cli"
        settings[workspace._model_key("antigravity_cli", workspace.SUMMARY_SCOPE)] = "stale-argon-summary"
        workspace._save_settings(self.ws, settings)
        path = Path(workspace._settings_path(self.ws))
        original = path.read_bytes()

        self.assertEqual(workspace.get_summary_backend_name(self.ws), self.summary.name)
        self.assertEqual(workspace.get_summary_model(self.ws), self.summary.default_summary_model)
        self.assertEqual(workspace.get_backend_name(self.ws), "antigravity_cli")
        self.assertEqual(path.read_bytes(), original)

    async def test_direct_run_defaults_judge_and_maintenance_to_supported_backend(self):
        data = session.create_session(self.ws, name="Manual")
        session.set_last_session(self.ws, 7, data["id"])
        with (
            mock.patch.object(agent, "_drive_backend_impl", new=mock.AsyncMock(
                return_value=(AgentResult(text="Implemented and verified."), False),
            )) as drive,
            mock.patch.object(agent, "run_internal_backend", new=mock.AsyncMock(
                return_value="[VERDICT]DONE[/VERDICT]",
            )) as internal,
            mock.patch.object(agent, "_compaction_context_targets", return_value=()),
            mock.patch.object(agent, "_snapshot_attachment_images", return_value={}),
            mock.patch.object(agent, "_append_new_attachment_images", new=mock.AsyncMock()),
            mock.patch.object(agent, "_run_post_turn_maintenance", new=mock.AsyncMock()) as maintenance,
        ):
            result = await agent.run(
                "Complete this bounded repair.", self.ws, 7, model=self.backend.default_model,
                approval="full", backend_name="antigravity_cli", execution_id="unsupported-internal-default",
            )
            await asyncio.sleep(0)

        self.assertFalse(result.error)
        self.assertTrue(result.completion_verified)
        self.assertEqual(drive.await_args.args[0].name, "antigravity_cli")
        internal.assert_awaited_once()
        self.assertEqual(internal.await_args.args[0].name, self.summary.name)
        self.assertEqual(internal.await_args.args[3], self.summary.default_summary_model)
        self.assertEqual(internal.await_args.kwargs["label"], "Continue judge")
        maintenance.assert_awaited_once()
        self.assertEqual(maintenance.await_args.args[2:4],
                         (self.summary.default_summary_model, self.summary.name))

    async def test_explicit_unsupported_summary_is_rejected_before_chat_or_paid_fallback(self):
        with (
            mock.patch.object(agent, "_drive_backend_impl", new=mock.AsyncMock()) as drive,
            mock.patch.object(agent, "run_internal_backend", new=mock.AsyncMock()) as internal,
        ):
            with self.assertRaisesRegex(ValueError, "cannot enforce"):
                await agent.run(
                    "Do not silently choose another paid backend.", self.ws, 7,
                    model=self.backend.default_model, approval="full", backend_name="antigravity_cli",
                    summary_backend_name="antigravity_cli", execution_id="unsupported-internal-explicit",
                )
        drive.assert_not_awaited()
        internal.assert_not_awaited()
