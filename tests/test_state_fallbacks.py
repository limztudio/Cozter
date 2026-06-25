import json
import os
import tempfile
import unittest

from Cozter import colony, config, schedules, workspace


class WorkspaceStateFallbackTests(unittest.TestCase):
    def test_invalid_workspace_settings_fall_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, ".cozter"))
            with open(
                os.path.join(tmp, ".cozter", "settings.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump({
                    "backend": "missing",
                    "summary_backend": "also-missing",
                    "permission": "bad",
                    "codex_model": 123,
                    "codex_summary_model": "",
                }, f)

            self.assertEqual(workspace.get_backend_name(tmp), "codex")
            self.assertEqual(workspace.get_summary_backend_name(tmp), "codex")
            self.assertEqual(workspace.get_permission(tmp), "auto")

            backend, model, summary_model, permission, summary_backend = (
                workspace.get_run_config(tmp)
            )
            self.assertEqual(backend, "codex")
            self.assertEqual(model, "gpt-5.5")
            self.assertEqual(summary_model, "gpt-5.4-mini")
            self.assertEqual(permission, "auto")
            self.assertEqual(summary_backend, "codex")

    def test_workspace_index_ignores_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "workspaces.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)

            old_path = workspace.WORKSPACE_STATE_PATH
            workspace.WORKSPACE_STATE_PATH = path
            try:
                with self.assertLogs(workspace.logger, level="WARNING"):
                    self.assertIsNone(workspace.get_current("u1", "bot"))
                    self.assertEqual(workspace.get_recent("u1"), [])
            finally:
                workspace.WORKSPACE_STATE_PATH = old_path

    def test_set_permission_rejects_unknown_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                workspace.set_permission(tmp, "maybe")


class ColonyStateFallbackTests(unittest.TestCase):
    def test_colony_state_normalizes_missing_or_invalid_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, ".cozter"))
            with open(
                os.path.join(tmp, ".cozter", "colony.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump({"items": "not-list", "compact_count": True}, f)

            self.assertEqual(colony.get_items(tmp), [])
            self.assertEqual(colony.get_compact_count(tmp), 0)
            self.assertEqual(colony.bump_compact_count(tmp), 1)


class ConfigFallbackTests(unittest.TestCase):
    def test_boolean_config_values_do_not_count_as_positive_ints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"llama_max_agent_turns": True}, f)

            old_path = config.CONFIG_PATH
            config.CONFIG_PATH = path
            try:
                self.assertEqual(config.get_llama_max_agent_turns(), 60)
            finally:
                config.CONFIG_PATH = old_path


class ScheduleParserTests(unittest.TestCase):
    def test_parse_days_handles_case_and_spaces(self) -> None:
        self.assertEqual(
            schedules.parse_days(" Mon, WED, 5 "),
            ["mon", "wed", "fri"],
        )


if __name__ == "__main__":
    unittest.main()
