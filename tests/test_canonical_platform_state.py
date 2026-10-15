"""Current platform identities never adopt obsolete owners or API shapes."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Cozter import session, titling, workspace
from Cozter.backends_bot.signal import SignalBot


class CanonicalPlatformStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_signal_start_and_incoming_message_preserve_current_and_unclaimed_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "workspaces.json"
            current = root / "current"
            current.mkdir()
            old = root / "old"
            old.mkdir()
            uid = "signal-group:group"
            with (
                mock.patch.object(workspace, "WORKSPACE_STATE_PATH", str(state)),
                mock.patch.object(workspace, "CONFIG_DIR", str(root)),
            ):
                workspace.select_workspace("sender", str(old), "signal:+account")
                workspace.select_workspace(uid, str(current), "signal")
                data = session.create_session(str(current))
                session.set_last_session(str(current), uid, data["id"])
                session.set_last_session(str(current), "sender", "historical-session")
                queue = root / "queue_signal.json"
                queue.write_text(json.dumps({
                    uid: [{"id": "current-job", "text": "current requirement", "chat_id": "group",
                           "job": {"session_id": data["id"], "native_session_id": "exact-native-id",
                                   "verified_evidence": ["saved result"]}}],
                    "sender": [{"id": "old-job", "text": "unclaimed", "chat_id": "group"}],
                }), encoding="utf-8")
                schedule_file = current / ".cozter" / "schedules.json"
                schedule_file.write_text(json.dumps({
                    "sender": [{"id": "unclaimed-schedule", "chat_id": "group"}],
                }), encoding="utf-8")
                last = current / ".cozter" / "last_session.json"
                paths = (state, queue, last, schedule_file)
                before = [path.read_bytes() for path in paths]

                bot = SignalBot(["https://signal.group/#fixture"], jsonrpc_socket="tcp://127.0.0.1:7584")
                with (
                    mock.patch.object(bot._outbound_journal, "load", new=mock.AsyncMock()),
                    mock.patch.object(bot, "_connect_jsonrpc", new=mock.AsyncMock()),
                    mock.patch.object(bot, "_resolve_group_ids", new=mock.AsyncMock(return_value={"fixture": "group"})),
                    mock.patch.object(bot, "_start_daemon_services", new=mock.AsyncMock()),
                    mock.patch.object(bot, "_start_receive_subscription", new=mock.AsyncMock()),
                    mock.patch.object(bot, "dispatch_text", new=mock.AsyncMock()) as dispatch,
                ):
                    await bot.start()
                    await bot._handle_received_item({
                        "account": "account",
                        "envelope": {"source": "sender", "dataMessage": {
                            "message": "Continue the current work", "timestamp": 12345,
                            "groupInfo": {"groupId": "group"},
                        }},
                    })
                dispatch.assert_awaited_once()
                self.assertEqual(dispatch.await_args.args[0].user_id, uid)
                self.assertEqual(workspace.get_current(uid, "signal"), workspace.canonicalize_workspace_path(str(current)))
                self.assertEqual(session.get_last_session(str(current), uid), data["id"])
                self.assertEqual([path.read_bytes() for path in paths], before)

    async def test_workspace_lookup_requires_exact_platform_key_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "workspaces.json"
            state.write_text(json.dumps({
                "signal-group:group": {"current": {"signal:+account": temporary}, "recent": [temporary]},
            }), encoding="utf-8")
            before = state.read_bytes()
            with mock.patch.object(workspace, "WORKSPACE_STATE_PATH", str(state)):
                self.assertIsNone(workspace.get_current("signal-group:group", "signal"))
                self.assertEqual(workspace.iter_current_workspaces("signal"), [])
            self.assertEqual(state.read_bytes(), before)

    async def test_last_session_lookup_does_not_adopt_another_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = session.create_session(temporary)
            session.set_last_session(temporary, "sender", data["id"])
            path = Path(temporary) / ".cozter" / "last_session.json"
            before = path.read_bytes()
            self.assertIsNone(session.get_last_session(temporary, "signal-group:group"))
            self.assertEqual(path.read_bytes(), before)

    async def test_compaction_requires_explicit_valid_snapshot_count_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = session.create_session(temporary)
            path = Path(session._session_path(temporary, data["id"]))
            before = path.read_bytes()
            with self.assertRaises(TypeError):
                session.set_summary(temporary, data["id"], "summary")
            for invalid in (None, True, False, -1, 1.5, "2"):
                with self.subTest(count=invalid), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                    session.set_summary(temporary, data["id"], "summary", summarized_count=invalid)
                self.assertEqual(path.read_bytes(), before)

    async def test_compaction_snapshot_preserves_messages_appended_after_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = session.create_session(temporary)
            messages = [{"role": "user", "content": str(index)} for index in range(5)]
            session.append_messages(temporary, data["id"], messages)
            session.set_summary(temporary, data["id"], "summary", keep_recent=1, summarized_count=3)
            saved = session.load_session(temporary, data["id"])
            self.assertEqual(saved["messages"], messages[2:])
            self.assertEqual(saved["compacted_count"], 2)
            self.assertEqual(saved["summary"], "summary")

    async def test_title_generation_ignores_nonstring_summary_instead_of_coercing_it(self):
        class InvalidSummary:
            def __str__(self):
                raise AssertionError("Nonstring summaries must not be rendered")

        with (
            mock.patch.object(titling.backends_agent, "get_backend", return_value=mock.sentinel.backend),
            mock.patch.object(titling, "run_internal_backend", new=mock.AsyncMock(return_value="Current Topic")) as backend,
        ):
            title = await titling.generate("/fixture", "session", backend_name="test", _preloaded_data={
                "summary": InvalidSummary(),
                "messages": [{"role": "assistant", "content": "current exchange"}],
            })
        self.assertEqual(title, "Current Topic")
        self.assertNotIn("Previous summary:", backend.await_args.args[2])
        self.assertIn("current exchange", backend.await_args.args[2])
