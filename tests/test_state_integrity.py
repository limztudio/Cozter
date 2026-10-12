"""Regression coverage for durable state and maintenance transactions."""

import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from Cozter import colony, config, session, utils, workspace


INVALID_JSON_VALUES = (
    '{"number":' + "9" * 5_000 + "}",
    '{"nested":' + "[" * 2_000 + "0" + "]" * 2_000 + "}",
    # CPython 3.12+ allows substantially deeper C decoder stacks than 3.11.
    '{"nested":' + "[" * 100_000 + "0" + "]" * 100_000 + "}",
)


class StateValidationAndRecoveryTests(unittest.TestCase):
    def test_example_config_covers_current_defaults(self) -> None:
        example_path = os.path.join(os.path.dirname(config.__file__), ".config", "config.example.json")
        with open(example_path, encoding="utf-8") as file_handle:
            example = json.load(file_handle)
        self.assertEqual(example.keys(), config._DEFAULT_CONFIG.keys())
        for key, default in config._DEFAULT_CONFIG.items():
            if key in {"telegram_bot_tokens", "user_ids"}:
                continue
            with self.subTest(key=key):
                self.assertEqual(example[key], default)

    def test_invalid_numeric_or_nested_json_state_falls_back(self) -> None:
        for raw in INVALID_JSON_VALUES:
            with self.subTest(raw_length=len(raw)), tempfile.TemporaryDirectory() as ws:
                path = os.path.join(ws, "state.json")
                with open(path, "w", encoding="utf-8") as file_handle:
                    file_handle.write(raw)
                with self.assertLogs(utils.logger, level="WARNING"):
                    self.assertEqual(utils.load_json_object(path, "test state"), {})

    def test_invalid_numeric_or_nested_session_json_is_ignored(self) -> None:
        for raw in INVALID_JSON_VALUES:
            with self.subTest(raw_length=len(raw)), tempfile.TemporaryDirectory() as ws:
                data = session.create_session(ws)
                path = session._session_path(ws, data["id"])
                with open(path, "w", encoding="utf-8") as file_handle:
                    file_handle.write(raw)
                with self.assertLogs(session.logger, level="WARNING"):
                    self.assertIsNone(session.load_session(ws, data["id"]))

    def test_deeply_nested_config_reports_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            path = os.path.join(ws, "config.json")
            with open(path, "w", encoding="utf-8") as file_handle:
                file_handle.write(INVALID_JSON_VALUES[2])
            output = io.StringIO()
            with (
                mock.patch.object(config, "CONFIG_PATH", path),
                contextlib.redirect_stdout(output),
                self.assertRaises(SystemExit) as exit_result,
            ):
                config.load_config()
            self.assertEqual(exit_result.exception.code, 1)
            self.assertIn("corrupted or unreadable", output.getvalue())

    def test_newest_listing_skips_corrupt_files_without_parsing_older_valid_files(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            older = session.create_session(ws, name="Older valid")
            newest = session.create_session(ws, name="Newest valid")
            session_dir = session._sessions_dir(ws)
            paths = [
                session._session_path(ws, older["id"]),
                session._session_path(ws, newest["id"]),
                os.path.join(session_dir, "broken.json"),
                os.path.join(session_dir, "mismatched.json"),
            ]
            with open(paths[2], "w", encoding="utf-8") as file_handle:
                file_handle.write("not json")
            with open(paths[3], "w", encoding="utf-8") as file_handle:
                json.dump({"id": "another-id"}, file_handle)
            for index, path in enumerate(paths):
                os.utime(path, ns=(1_000_000_000 * (index + 1),) * 2)
            with (
                mock.patch.object(session, "_load_session_path", wraps=session._load_session_path) as load,
                self.assertLogs(session.logger, level="WARNING"),
            ):
                found, total = session.list_newest_sessions_with_data(ws, 1)
            self.assertEqual(total, 4)
            self.assertEqual([entry["id"] for entry in found], [newest["id"]])
            self.assertEqual(load.call_count, 3)
            self.assertNotIn(paths[0], [call.args[0] for call in load.call_args_list])

    def test_last_session_migration_repairs_unsafe_target_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            data = session.create_session(ws)
            session._save_last_session_map(ws, {
                "target": "../unsafe", "source": data["id"],
            })
            self.assertTrue(session.migrate_last_session(ws, ["source"], "target"))
            self.assertEqual(session.get_last_session(ws, "target"), data["id"])

    def test_state_path_rejects_windows_stream_and_drive_components(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            for component in ("settings.json:stream", "C:relative"):
                with self.subTest(component=component), self.assertRaises(ValueError):
                    workspace.workspace_state_path(ws, component)

    def test_escaping_state_link_cannot_create_its_missing_target(self) -> None:
        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as outside:
            target = os.path.join(outside, "must-not-be-created")
            try:
                os.symlink(target, os.path.join(ws, ".cozter"), target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")
            with self.assertRaises(ValueError):
                workspace.ensure_cozter_dir(ws)
            self.assertFalse(os.path.exists(target))


class JsonEventValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_numeric_and_nested_events_do_not_drop_later_valid_event(self) -> None:
        stream = asyncio.StreamReader()
        stream.feed_data(("\n".join([*INVALID_JSON_VALUES, '{"text":"complete"}']) + "\n").encode())
        stream.feed_eof()
        invalid = []
        events = [event async for event in utils.iter_json_events(stream, on_invalid=invalid.append)]
        self.assertEqual(events, [{"text": "complete"}])
        self.assertEqual(invalid, list(INVALID_JSON_VALUES))


class ColonyConsolidationConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_consolidation_preserves_concurrent_memory_and_session_changes(self) -> None:
        for change in ("local", "shared", "created", "deleted"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as ws:
                data = await asyncio.to_thread(session.create_session, ws)
                data["long_term"] = ["Original local fact."]
                await asyncio.to_thread(session.save_session, ws, data["id"], data)
                await asyncio.to_thread(colony.set_items, ws, ["Original shared fact."])
                output = f"[COLONY]\n- Stale shared fact.\n[/COLONY][SESSION:{data['id']}][/SESSION]"

                async def stale_result(*_args, **_kwargs):
                    if change == "local":
                        latest = session.load_session(ws, data["id"])
                        latest["long_term"] = ["New local fact."]
                        await asyncio.to_thread(session.save_session, ws, data["id"], latest)
                    elif change == "shared":
                        await asyncio.to_thread(colony.set_items, ws, ["New shared fact."])
                    elif change == "created":
                        await asyncio.to_thread(session.create_session, ws, name="New session")
                    else:
                        await asyncio.to_thread(session.delete_session, ws, data["id"])
                    return output

                with (
                    mock.patch.object(colony, "run_internal_backend", side_effect=stale_result),
                    self.assertLogs(colony.logger, level="WARNING"),
                ):
                    self.assertFalse(await colony.consolidate(ws, backend_name="codex"))
                expected_shared = "New shared fact." if change == "shared" else "Original shared fact."
                self.assertEqual(colony.get_items(ws), [expected_shared])
                latest = session.load_session(ws, data["id"])
                if change == "deleted":
                    self.assertIsNone(latest)
                else:
                    expected_local = "New local fact." if change == "local" else "Original local fact."
                    self.assertEqual(latest["long_term"], [expected_local])
                    self.assertEqual(len(session.list_sessions_with_data(ws)), 2 if change == "created" else 1)

    async def test_consolidation_accepts_appended_messages_and_preserves_rename(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            data = await asyncio.to_thread(session.create_session, ws)
            data["long_term"] = ["Original local fact."]
            await asyncio.to_thread(session.save_session, ws, data["id"], data)
            message = {"role": "user", "content": "A new conversation message."}

            async def result(*_args, **_kwargs):
                await asyncio.to_thread(session.append_messages, ws, data["id"], [message])
                await asyncio.to_thread(session.set_session_name, ws, data["id"], "User Renamed Session")
                return f"[COLONY]\n- Shared fact.\n[/COLONY][SESSION:{data['id']}]\n- Remaining local fact.\n[/SESSION]"

            with mock.patch.object(colony, "run_internal_backend", side_effect=result):
                self.assertTrue(await colony.consolidate(ws, backend_name="codex"))
            latest = session.load_session(ws, data["id"])
            self.assertEqual(latest["messages"], [message])
            self.assertEqual(latest["name"], "User Renamed Session")
            self.assertEqual(latest["long_term"], ["Remaining local fact."])
            self.assertEqual(colony.get_items(ws), ["Shared fact."])

    async def test_empty_workspace_cleanup_rechecks_sessions_after_waiting_for_lock(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            await asyncio.to_thread(colony.set_items, ws, ["Shared fact."])
            writer_lock = workspace.get_lock(ws)
            await writer_lock.acquire()
            consolidation = asyncio.create_task(colony.consolidate(ws, backend_name="codex"))
            try:
                await asyncio.sleep(0)
                self.assertFalse(consolidation.done())
                await asyncio.to_thread(session.create_session, ws)
                writer_lock.release()
                self.assertFalse(await consolidation)
                self.assertEqual(colony.get_items(ws), ["Shared fact."])
            finally:
                if writer_lock.locked():
                    writer_lock.release()
                await asyncio.gather(consolidation, return_exceptions=True)
