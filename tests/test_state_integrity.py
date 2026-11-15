"""Regression coverage for durable state and maintenance transactions."""

import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from Cozter import colony, config, schedules, session, utils, workspace


INVALID_JSON_VALUES = (
    '{"number":' + "9" * 5_000 + "}",
    '{"nested":' + "[" * 2_000 + "0" + "]" * 2_000 + "}",
    # CPython 3.12+ allows substantially deeper C decoder stacks than 3.11.
    '{"nested":' + "[" * 100_000 + "0" + "]" * 100_000 + "}",
)


class JsonCacheSnapshotRaceTests(unittest.TestCase):
    def _cases(self, root):
        state = os.path.join(root, ".cozter")
        os.makedirs(state, exist_ok=True)
        return (
            (config, os.path.join(root, "config.json"), config._load_config_object,
             {"max_permission": "auto"}, {"max_permission": "deny"}),
            (workspace, os.path.join(root, "workspaces.json"), workspace._load_all,
             {"u": {"current": {"bot": "aaa"}}}, {"u": {"current": {"bot": "bbb"}}}),
            (workspace, os.path.join(state, "settings.json"), lambda: workspace._load_settings(root),
             {"permission": "auto"}, {"permission": "deny"}),
            (colony, os.path.join(state, "colony.json"), lambda: colony._load(root),
             {"items": ["aaa"], "compact_count": 0}, {"items": ["bbb"], "compact_count": 0}),
            (session, os.path.join(state, "last_session.json"), lambda: session._load_last_session_map(root),
             {"u": "aaa"}, {"u": "bbb"}),
            (schedules, os.path.join(state, "schedules.json"), lambda: schedules._load_all(root),
             {"u": [{"command": "aaa"}]}, {"u": [{"command": "bbb"}]}),
        )

    def _write(self, path, data, stamp=None):
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(data, stream)
        if stamp is not None:
            os.utime(path, ns=(stamp, stamp))

    def test_same_stamp_rewrite_and_snapshot_race_refresh_all_six_caches(self):
        with tempfile.TemporaryDirectory() as root:
            cases = self._cases(root)
            with (mock.patch.object(config, "CONFIG_PATH", cases[0][1]),
                  mock.patch.object(workspace, "WORKSPACE_STATE_PATH", cases[1][1])):
                for module, path, load, old, new in cases:
                    with self.subTest(path=path):
                        self._write(path, old)
                        stamp = os.stat(path).st_mtime_ns
                        self.assertEqual(load(), old)
                        self._write(path, new, stamp)
                        self.assertEqual(load(), new)
                        self._write(path, old, stamp)
                        real = module.read_bytes_digest

                        def replace_after_read(name):
                            snapshot = real(name)
                            self._write(path, new, stamp)
                            return snapshot

                        with mock.patch.object(module, "read_bytes_digest", side_effect=replace_after_read):
                            self.assertEqual(load(), old)
                        self.assertEqual(load(), new)

    def test_replacement_after_save_is_not_cached_as_the_saved_value(self):
        with tempfile.TemporaryDirectory() as root:
            cases = self._cases(root)
            saves = (
                lambda value: workspace._save_all(value),
                lambda value: workspace._save_settings(root, value),
                lambda value: colony.set_items(root, value["items"]),
                lambda value: session._save_last_session_map(root, value),
                lambda value: schedules._save_all(root, value),
            )
            with mock.patch.object(workspace, "WORKSPACE_STATE_PATH", cases[1][1]):
                for (module, path, load, old, new), save in zip(cases[1:], saves):
                    with self.subTest(path=path):
                        self._write(path, old)
                        real = module.save_json_object

                        def replace_after_save(name, data):
                            real(name, data)
                            self._write(path, new, os.stat(path).st_mtime_ns)

                        with mock.patch.object(module, "save_json_object", side_effect=replace_after_save):
                            save(old)
                        self.assertEqual(load(), new)

    def test_permission_replacement_before_fingerprint_cannot_pin_old_value(self):
        with tempfile.TemporaryDirectory() as root:
            cases = self._cases(root)
            with mock.patch.object(config, "CONFIG_PATH", cases[0][1]):
                for module, path, load, old, new in (cases[0], cases[2]):
                    with self.subTest(path=path):
                        self._write(path, old)
                        stamp = os.stat(path).st_mtime_ns
                        real = module.read_bytes_digest

                        def replace_before_read(name):
                            self._write(path, new, stamp)
                            return real(name)

                        with mock.patch.object(module, "read_bytes_digest", side_effect=replace_before_read):
                            load()
                        self.assertEqual(load(), new)


class StateValidationAndRecoveryTests(unittest.TestCase):
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
