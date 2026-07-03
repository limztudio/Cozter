import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime

from Cozter import colony, config, schedules, session, workspace
from Cozter.backends_bot.base import BotPlatform


class QueueRestoreBot(BotPlatform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drained_users: list[str] = []

    @property
    def platform_id(self) -> str:
        return "test:queue"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send_text(self, chat_id: str, text: str, *, rich: bool = False):
        return None

    async def edit_text(
        self, handle, text: str, *, rich: bool = False,
    ) -> None:
        pass

    async def delete_message(self, handle) -> None:
        pass

    async def send_file(self, chat_id: str, path: str) -> None:
        pass

    async def _drain_message_queue(self, uid: str) -> None:
        self.drained_users.append(uid)


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

    def test_iter_current_workspaces_ignores_malformed_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "workspaces.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "u1": "not-object",
                    "u2": {"current": "not-object"},
                    "u3": {"current": {"bot": "/tmp/ws"}},
                    "u4": {"current": {"bot": ""}},
                }, f)

            old_path = workspace.WORKSPACE_STATE_PATH
            workspace.WORKSPACE_STATE_PATH = path
            try:
                self.assertEqual(
                    workspace.iter_current_workspaces("bot"),
                    [("u3", "/tmp/ws")],
                )
            finally:
                workspace.WORKSPACE_STATE_PATH = old_path

    def test_workspace_migration_normalizes_target_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "workspaces.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "source": {"current": {"bot": "/tmp/ws"}},
                    "target": {"current": "not-object", "recent": []},
                }, f)

            old_path = workspace.WORKSPACE_STATE_PATH
            workspace.WORKSPACE_STATE_PATH = path
            try:
                self.assertTrue(
                    workspace.migrate_current_workspace(
                        "source", "target", "bot",
                    )
                )
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                self.assertEqual(
                    data["target"]["current"]["bot"], "/tmp/ws",
                )
            finally:
                workspace.WORKSPACE_STATE_PATH = old_path

    def test_set_permission_rejects_unknown_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                workspace.set_permission(tmp, "maybe")

    def test_max_permission_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            old = config.CONFIG_PATH
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"max_permission": "auto"}, f)
            config.CONFIG_PATH = path
            try:
                self.assertEqual(config.get_max_permission(), "auto")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"max_permission": "nonsense"}, f)
                self.assertEqual(config.get_max_permission(), "full")
            finally:
                config.CONFIG_PATH = old

    def test_permission_ceiling_clamps_and_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            def _auto_ceiling() -> str:
                return "auto"

            orig = config.get_max_permission
            config.get_max_permission = _auto_ceiling
            try:
                # Setting above the ceiling is rejected.
                with self.assertRaises(ValueError):
                    workspace.set_permission(tmp, "full")
                # At/below the ceiling is allowed.
                workspace.set_permission(tmp, "auto")
                self.assertEqual(workspace.get_permission(tmp), "auto")
                # An already-stored higher value is clamped on read.
                with open(
                    os.path.join(tmp, ".cozter", "settings.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump({"permission": "full"}, f)
                self.assertEqual(workspace.get_permission(tmp), "auto")
            finally:
                config.get_max_permission = orig

    def test_interaction_style_falls_back_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Missing setting -> default.
            self.assertEqual(
                workspace.get_interaction_style(tmp), "collaborative",
            )
            # Invalid value -> default.
            os.makedirs(os.path.join(tmp, ".cozter"))
            with open(
                os.path.join(tmp, ".cozter", "settings.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump({"style": "chatty"}, f)
            self.assertEqual(
                workspace.get_interaction_style(tmp), "collaborative",
            )

    def test_set_interaction_style_round_trips_and_rejects_unknown(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace.set_interaction_style(tmp, "autonomous")
            self.assertEqual(
                workspace.get_interaction_style(tmp), "autonomous",
            )
            with self.assertRaises(ValueError):
                workspace.set_interaction_style(tmp, "verbose")

    def test_extra_models_parsing_tolerates_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "extra_models": {
                        "codex": ["gpt-5.6", "", 123],  # non-strings dropped
                        "copilot": "not-a-list",         # wrong type -> []
                    },
                }, f)
            old = config.CONFIG_PATH
            config.CONFIG_PATH = path
            try:
                self.assertEqual(config.get_extra_models("codex"), ["gpt-5.6"])
                self.assertEqual(config.get_extra_models("copilot"), [])
                self.assertEqual(config.get_extra_models("missing"), [])
            finally:
                config.CONFIG_PATH = old

    def test_extra_models_missing_or_non_object_returns_empty(self) -> None:
        # No config file (CLI mode) -> [].
        old = config.CONFIG_PATH
        config.CONFIG_PATH = "/nonexistent/config.json"
        try:
            self.assertEqual(config.get_extra_models("codex"), [])
        finally:
            config.CONFIG_PATH = old
        # extra_models present but not an object -> [].
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"extra_models": ["oops"]}, f)
            config.CONFIG_PATH = path
            try:
                self.assertEqual(config.get_extra_models("codex"), [])
            finally:
                config.CONFIG_PATH = old

    def test_history_budget_falls_back_and_enforces_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                workspace.get_history_budget(tmp),
                workspace.DEFAULT_HISTORY_BUDGET,
            )
            with self.assertRaises(ValueError):
                workspace.set_history_budget(
                    tmp, workspace.MIN_HISTORY_BUDGET - 1,
                )
            workspace.set_history_budget(tmp, 8_000)
            self.assertEqual(workspace.get_history_budget(tmp), 8_000)
            # A malformed stored value falls back to the default.
            with open(
                os.path.join(tmp, ".cozter", "settings.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump({"history_budget": "lots"}, f)
            self.assertEqual(
                workspace.get_history_budget(tmp),
                workspace.DEFAULT_HISTORY_BUDGET,
            )

    def test_available_models_appends_extras_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, ".cozter"))
            workspace.set_backend_name(tmp, "codex")

            def _fake_extras(name: str) -> list[str]:
                # "gpt-5.5" is already built-in; "gpt-5.6" is new.
                return ["gpt-5.6", "gpt-5.5"] if name == "codex" else []

            orig = config.get_extra_models
            config.get_extra_models = _fake_extras
            try:
                models = workspace.get_available_models(tmp)
            finally:
                config.get_extra_models = orig

            self.assertEqual(models[0], "gpt-5.5")   # built-ins first
            self.assertIn("gpt-5.6", models)          # extra appended
            self.assertEqual(models.count("gpt-5.5"), 1)  # no duplicate


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

    def test_non_object_config_exits_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)

            old_path = config.CONFIG_PATH
            config.CONFIG_PATH = path
            try:
                out = io.StringIO()
                with (
                    self.assertRaises(SystemExit) as raised,
                    contextlib.redirect_stdout(out),
                ):
                    config.load_config()

                self.assertEqual(raised.exception.code, 1)
                self.assertIn("must contain a JSON object", out.getvalue())
            finally:
                config.CONFIG_PATH = old_path

    def test_runtime_getters_reject_non_object_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)

            old_path = config.CONFIG_PATH
            config.CONFIG_PATH = path
            try:
                with self.assertRaisesRegex(ValueError, "JSON object"):
                    config.get_llama_max_agent_turns()
            finally:
                config.CONFIG_PATH = old_path


class ScheduleParserTests(unittest.TestCase):
    def test_parse_days_handles_case_and_spaces(self) -> None:
        self.assertEqual(
            schedules.parse_days(" Mon, WED, 5 "),
            ["mon", "wed", "fri"],
        )

    def test_schedule_parsers_ignore_malformed_values(self) -> None:
        self.assertEqual(schedules.parse_days(["mon"]), [])
        self.assertIsNone(schedules.parse_time(930))
        self.assertIsNone(schedules.parse_iso(930))

    def test_most_recent_slot_ignores_malformed_schedule_fields(self) -> None:
        now = datetime(2026, 1, 5, 10, 0)  # Monday

        self.assertIsNone(
            schedules.most_recent_slot(
                {"days": "mon", "time": "09:00"}, now,
            ),
        )
        self.assertIsNone(
            schedules.most_recent_slot(
                {"days": ["mon"], "time": 900}, now,
            ),
        )
        self.assertEqual(
            schedules.most_recent_slot(
                {"days": ["mon", 7], "time": "09:00"}, now,
            ),
            datetime(2026, 1, 5, 9, 0),
        )

    def test_schedule_store_ignores_malformed_user_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, ".cozter"))
            path = os.path.join(tmp, ".cozter", "schedules.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "u1": "not-list",
                    "u2": [{"id": "ok"}, "not-object"],
                    "u3": {"id": "not-list"},
                }, f)

            self.assertEqual(schedules.list_schedules(tmp, "u1"), [])
            self.assertEqual(schedules.list_schedules(tmp, "u2"), [{"id": "ok"}])
            self.assertEqual(schedules.list_schedule_user_ids(tmp), ["u2"])

            schedules.add_schedule(tmp, "u1", {"id": "new"})
            self.assertEqual(schedules.list_schedules(tmp, "u1"), [{"id": "new"}])

    def test_schedule_mutations_skip_non_dict_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, ".cozter"))
            path = os.path.join(tmp, ".cozter", "schedules.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"u1": ["not-object", {"id": "a"}]}, f)

            schedules.update_schedule_fired(tmp, "u1", "a", "2026-01-01T00:00:00")
            self.assertEqual(
                schedules.list_schedules(tmp, "u1"),
                [{"id": "a", "last_fired": "2026-01-01T00:00:00"}],
            )
            self.assertTrue(schedules.remove_schedule(tmp, "u1", "a"))
            self.assertEqual(schedules.list_schedules(tmp, "u1"), [])

    def test_scheduler_skips_schedule_without_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = os.path.join(tmp, "ws")
            os.makedirs(os.path.join(ws, ".cozter"))
            with open(
                os.path.join(ws, ".cozter", "schedules.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump({
                    "u1": [{
                        "days": list(schedules.DAY_ABBREV),
                        "time": "00:00",
                        "command": "run",
                        "created": "2000-01-01T00:00:00",
                        "chat_id": "u1",
                        "user_id": "u1",
                    }],
                }, f)

            old_path = workspace.WORKSPACE_STATE_PATH
            workspace.WORKSPACE_STATE_PATH = os.path.join(
                tmp, "workspaces.json",
            )
            try:
                workspace.select_workspace("u1", ws, "test:queue")
                bot = QueueRestoreBot(["u1"])
                asyncio.run(bot._scheduler_tick())
                self.assertEqual(bot.drained_users, [])
            finally:
                workspace.WORKSPACE_STATE_PATH = old_path


class SessionStateFallbackTests(unittest.TestCase):
    def test_session_loader_normalizes_malformed_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = os.path.join(tmp, ".cozter", "sessions")
            os.makedirs(sessions_dir)
            with open(
                os.path.join(sessions_dir, "abc123.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump({
                    "id": "abc123",
                    "name": 7,
                    "created": 8,
                    "messages": [
                        {"role": 4, "content": 123},
                        "not-object",
                    ],
                    "summary": ["bad"],
                    "long_term": ["keep", 5, ""],
                    "compacted_count": True,
                }, f)

            loaded = session.load_session(tmp, "abc123")
            self.assertIsNotNone(loaded)
            if loaded is None:
                self.fail("session should load after normalization")
            self.assertEqual(loaded["name"], "abc123")
            self.assertEqual(loaded["created"], "")
            self.assertEqual(loaded["messages"], [
                {"role": "?", "content": "123"},
            ])
            self.assertIsNone(loaded["summary"])
            self.assertEqual(loaded["long_term"], ["keep"])
            self.assertEqual(loaded["compacted_count"], 0)
            self.assertEqual(session.list_sessions(tmp), [{
                "id": "abc123",
                "name": "abc123",
                "created": "",
                "message_count": 1,
            }])

    def test_session_listing_ignores_missing_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = os.path.join(tmp, ".cozter", "sessions")
            os.makedirs(sessions_dir)
            with open(
                os.path.join(sessions_dir, "missing-id.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump({"messages": []}, f)

            with self.assertLogs(session.logger, level="WARNING"):
                self.assertEqual(session.list_sessions(tmp), [])

    def test_session_helpers_tolerate_bad_runtime_shapes(self) -> None:
        self.assertEqual(
            session.total_message_count({
                "compacted_count": "bad",
                "messages": "bad",
            }),
            0,
        )
        self.assertEqual(
            session.format_msg_line({"role": 4, "content": 123}),
            "?: 123",
        )


class QueueStateFallbackTests(unittest.TestCase):
    def test_queue_entries_filters_malformed_values(self) -> None:
        self.assertEqual(BotPlatform._queue_entries("not-list"), [])
        self.assertEqual(
            BotPlatform._queue_entries([{"id": "ok"}, "not-object"]),
            [{"id": "ok"}],
        )

    def test_restore_queues_keeps_entries_over_default_capacity(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                old_config_dir = workspace.CONFIG_DIR
                workspace.CONFIG_DIR = tmp
                try:
                    bot = QueueRestoreBot(["u1"], max_queue_size=1)
                    existing_queue = bot._ensure_message_queue("u1")
                    existing_queue.put_nowait((
                        "already queued", "chat", "existing-id", False,
                    ))
                    path = bot._queue_file_path()
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump({
                            "u1": [
                                {
                                    "id": "first-id",
                                    "text": "first",
                                    "chat_id": "chat",
                                },
                                {
                                    "id": "second-id",
                                    "text": "second",
                                    "chat_id": "chat",
                                    "ephemeral": True,
                                },
                            ],
                        }, f)

                    await bot.restore_queues()
                    await asyncio.sleep(0)

                    queue = bot._message_queues["u1"]
                    self.assertEqual(queue.qsize(), 3)
                    self.assertEqual(
                        queue.get_nowait(),
                        ("already queued", "chat", "existing-id", False),
                    )
                    self.assertEqual(
                        queue.get_nowait(),
                        ("first", "chat", "first-id", False),
                    )
                    self.assertEqual(
                        queue.get_nowait(),
                        ("second", "chat", "second-id", True),
                    )
                    self.assertEqual(bot.drained_users, ["u1"])
                finally:
                    workspace.CONFIG_DIR = old_config_dir

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
