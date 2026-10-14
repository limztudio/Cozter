"""Signal own-output replay protection without live RPCs or model calls."""

import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from Cozter import config, workspace
from Cozter.backends_bot import create_platforms
from Cozter.backends_bot import signal as signal_module
from Cozter.backends_bot.signal import (
    SignalBot, SignalCliError, _SignalOutboundJournal, _echo_hash,
    seed_signal_outbound_echoes,
)
from Cozter.tests.helpers import temporary_config


class SignalFeedbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = os.path.join(temporary.name, "state")
        self.bots: list[SignalBot] = []
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.path))
        self.enterContext(mock.patch.object(signal_module, "_SIGNAL_SEND_MIN_INTERVAL_SEC", 0))

    async def asyncTearDown(self) -> None:
        for bot in self.bots:
            await bot._flush_outbound_journal()

    def bot(self) -> SignalBot:
        bot = SignalBot([], jsonrpc_socket="tcp://127.0.0.1:7584")
        self.bots.append(bot)
        bot._group_ids = {"group", "other-group"}
        bot._rpc_request = mock.AsyncMock(return_value={"timestamp": "12345"})
        bot._migrate_group_workspace_state = mock.Mock()
        bot._migrate_group_workspace_files = mock.AsyncMock()
        bot.dispatch_text = mock.AsyncMock()
        bot.dispatch_command = mock.AsyncMock()
        return bot

    async def send_and_stop(self, text: str) -> None:
        sender = self.bot()
        await sender.send_text("group", text)
        await sender._flush_outbound_journal()

    @staticmethod
    def message(text: str, *, timestamp: int = 12345, group: str = "group",
                local: bool = True, sync: bool = True, device: int | None = None) -> dict:
        data = {"message": text, "timestamp": timestamp,
                "groupInfo": {"groupId": group}}
        envelope = {"source": "local" if local else "another-person"}
        if device is not None:
            envelope["sourceDevice"] = device
        if sync:
            envelope["syncMessage"] = {"sentMessage": data}
        else:
            envelope["dataMessage"] = data
        return {"account": "local", "envelope": envelope}

    async def test_constructor_and_empty_load_do_not_create_state_directory(self) -> None:
        bot = self.bot()
        self.assertFalse(os.path.exists(self.path))
        await bot._outbound_journal.load()
        self.assertFalse(os.path.exists(self.path))

    async def test_repeated_own_sync_echoes_are_not_consumed_and_survive_restart(self) -> None:
        sender = self.bot()
        await sender.send_text("group", "A finished report")
        await sender._flush_outbound_journal()
        for bot in (sender, self.bot()):
            for timestamp in (12345, 12346, 12347):
                await bot._handle_received_item(self.message("A finished report", timestamp=timestamp))
            bot.dispatch_text.assert_not_awaited()

    async def test_timestamp_provenance_survives_restart_and_is_group_bound(self) -> None:
        sender = self.bot()
        await sender.send_text("group", "Original report")
        await sender._flush_outbound_journal()
        restarted = self.bot()
        await restarted._handle_received_item(self.message("Edited content"))
        restarted.dispatch_text.assert_not_awaited()
        await restarted._handle_received_item(self.message("Edited content", group="other-group"))
        restarted.dispatch_text.assert_awaited_once()

    async def test_pre_send_fingerprint_handles_timeout_before_acknowledgement(self) -> None:
        sender = self.bot()
        sender._rpc_request.side_effect = TimeoutError("acknowledgement lost")
        with self.assertRaises(TimeoutError):
            await sender.send_text("group", "Timed out final reply")
        restarted = self.bot()
        for _ in range(3):
            await restarted._handle_received_item(self.message("Timed out final reply", timestamp=999))
        restarted.dispatch_text.assert_not_awaited()

    async def test_other_people_and_unproven_local_messages_are_preserved(self) -> None:
        await self.send_and_stop("A report")
        bot = self.bot()
        for data in (
            self.message("A report", local=False, sync=False),
            self.message("Please continue with my next task", timestamp=67890),
            self.message("A report", timestamp=67891, group="other-group"),
        ):
            await bot._handle_received_item(data)
        self.assertEqual(bot.dispatch_text.await_count, 3)

    async def test_local_data_message_echo_is_also_ignored(self) -> None:
        await self.send_and_stop("A report")
        bot = self.bot()
        await bot._handle_received_item(self.message("A report", sync=False))
        bot.dispatch_text.assert_not_awaited()

    async def test_configured_own_device_is_ignored_before_any_state_or_journal_io(self) -> None:
        bot = self.bot()
        bot.own_device_id = 2
        for timestamp, text in enumerate(("Thinking...", "Arbitrary bot report", "/stop"), 800):
            await bot._handle_received_item(self.message(text, timestamp=timestamp, device=2))
            await bot._handle_received_item(self.message(text, timestamp=timestamp, device=2, sync=False))
        bot.dispatch_text.assert_not_awaited()
        bot.dispatch_command.assert_not_awaited()
        bot._migrate_group_workspace_state.assert_not_called()
        self.assertFalse(os.path.exists(self.path))

    async def test_primary_and_other_linked_device_can_repeat_proven_bot_output(self) -> None:
        await self.send_and_stop("Repeat this text")
        bot = self.bot()
        bot.own_device_id = 2
        for device in (1, 3):
            # A matching timestamp/text must not override the known human device.
            await bot._handle_received_item(self.message("Repeat this text", device=device))
        # Dedupe still treats an identical original message timestamp as one request.
        self.assertEqual(bot.dispatch_text.await_count, 1)
        await bot._handle_received_item(self.message("Repeat this text", timestamp=33333, device=3))
        self.assertEqual(bot.dispatch_text.await_count, 2)
        await bot._handle_received_item(self.message("/stop", timestamp=44444, device=1))
        bot.dispatch_command.assert_awaited_once()

    async def test_other_account_with_same_device_id_is_preserved(self) -> None:
        await self.send_and_stop("Shared text")
        bot = self.bot()
        bot.own_device_id = 2
        await bot._handle_received_item(self.message("Shared text", device=2, sync=False, local=False))
        bot.dispatch_text.assert_awaited_once()

    async def test_missing_source_device_retains_durable_fallback_when_configured(self) -> None:
        await self.send_and_stop("Old echoed report")
        bot = self.bot()
        bot.own_device_id = 2
        await bot._handle_received_item(self.message("Old echoed report"))
        bot.dispatch_text.assert_not_awaited()
        await bot._handle_received_item(self.message("New human instruction", timestamp=44444))
        bot.dispatch_text.assert_awaited_once()

    async def test_exact_protocol_replies_are_rejected_before_pending_input(self) -> None:
        bot = self.bot()
        bot.dispatch_text = SignalBot.dispatch_text.__get__(bot)
        handler = mock.AsyncMock()
        uid = bot._state_user_id("group")
        bot._expect_input(uid, handler)
        wrapped = "Directory does not exist:\nQueued (3/5000).\n\nPlease enter a valid directory (or /cancel):"
        nested = f"Directory does not exist:\n{wrapped}\n\nUse /open to choose from recent workspaces."
        for local in (True, False):
            for text in ("Queued (3/5000).", wrapped, nested):
                await bot._handle_received_item(self.message(text, local=local, sync=local))
        handler.assert_not_awaited()
        self.assertIn(uid, bot._pending_input)
        bot._migrate_group_workspace_state.assert_not_called()
        bot.dispatch_command.assert_not_awaited()
        self.assertFalse(os.path.exists(self.path))
        await bot._handle_received_item(self.message("C:\\MyWorkspace", timestamp=77777))
        handler.assert_awaited_once()

    async def test_similar_natural_language_and_unrelated_directory_text_are_preserved(self) -> None:
        bot = self.bot()
        for index, text in enumerate((
            "Queued (1/5000). Please show me the task list.",
            "Directory does not exist:\nC:\\Missing\n\nPlease enter a valid directory (or /cancel):",
            "Queued (one/two).", "Why does it say Queued (1/2).?",
        )):
            await bot._handle_received_item(self.message(text, timestamp=50000 + index))
        self.assertEqual(bot.dispatch_text.await_count, 4)

    async def test_proven_command_echo_cannot_trigger_command_dispatch(self) -> None:
        await self.send_and_stop("/stop")
        bot = self.bot()
        await bot._handle_received_item(self.message("/stop"))
        bot.dispatch_command.assert_not_awaited()
        await bot._handle_received_item(self.message("/stop", local=False, sync=False))
        bot.dispatch_command.assert_awaited_once()

    async def test_failed_pre_send_journal_write_prevents_rpc(self) -> None:
        bot = self.bot()
        with mock.patch.object(signal_module, "save_json_object", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                await bot.send_text("group", "Cannot persist this reply")
        bot._rpc_request.assert_not_awaited()

    async def test_acknowledged_send_is_not_retried_when_timestamp_write_fails(self) -> None:
        bot = self.bot()
        save = signal_module.save_json_object
        calls = 0

        def fail_timestamp(path, payload):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("disk full after ACK")
            save(path, payload)

        with mock.patch.object(signal_module, "save_json_object", side_effect=fail_timestamp):
            with self.assertLogs(signal_module.logger, "WARNING"):
                handle = await bot.send_text("group", "Acknowledged reply")
                await bot._flush_outbound_journal()
        self.assertIsNotNone(handle)
        bot._rpc_request.assert_awaited_once()
        await bot._handle_received_item(self.message("Changed text, known timestamp"))
        bot.dispatch_text.assert_not_awaited()
        restarted = self.bot()
        await restarted._handle_received_item(self.message("Acknowledged reply", timestamp=789))
        restarted.dispatch_text.assert_not_awaited()

    async def test_post_ack_writer_cannot_turn_confirmed_send_into_cancellation_failure(self) -> None:
        bot = self.bot()
        entered, release = threading.Event(), threading.Event()
        save = signal_module.save_json_object
        writes = 0

        def block_timestamp(path, payload):
            nonlocal writes
            writes += 1
            if writes == 2:
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("test writer was not released")
            save(path, payload)

        with mock.patch.object(signal_module, "save_json_object", side_effect=block_timestamp):
            task = asyncio.create_task(bot.send_text("group", "Confirmed reply"))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                self.assertTrue(task.done())
                self.assertIsNotNone(task.result())
                task.cancel()
                self.assertFalse(task.cancelled())
                # Even shutdown cancellation owns the background disk write.
                writers = list(bot._outbound_journal_tasks)
                self.assertEqual(len(writers), 1)
                writers[0].cancel()
                await asyncio.sleep(0.01)
                self.assertFalse(writers[0].done())
            finally:
                release.set()
                await bot._flush_outbound_journal()
        bot._rpc_request.assert_awaited_once()
        restarted = self.bot()
        await restarted._handle_received_item(self.message("Different body, same confirmed timestamp"))
        restarted.dispatch_text.assert_not_awaited()

    async def test_cancellation_waits_for_journal_writer_and_never_starts_send(self) -> None:
        bot = self.bot()
        entered, release = threading.Event(), threading.Event()
        save = signal_module.save_json_object

        def blocking_save(path, payload):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test writer was not released")
            save(path, payload)

        with mock.patch.object(signal_module, "save_json_object", side_effect=blocking_save):
            task = asyncio.create_task(bot.send_text("group", "Cancelled send"))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                task.cancel()
                await asyncio.sleep(0.01)
                self.assertFalse(task.done())
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        bot._rpc_request.assert_not_awaited()
        restarted = self.bot()
        await restarted._handle_received_item(self.message("Cancelled send"))
        restarted.dispatch_text.assert_not_awaited()

    async def test_recovery_seed_contains_hashes_only_and_suppresses_repeated_old_echoes(self) -> None:
        await seed_signal_outbound_echoes([("group", "Proven quarantined report")])
        with open(os.path.join(self.path, "signal_outbound.json"), encoding="utf-8") as source:
            raw = source.read()
        self.assertNotIn("Proven quarantined report", raw)
        self.assertTrue(all(row["group"] != "group" for row in json.loads(raw)["records"]))
        bot = self.bot()
        for timestamp in (100, 101, 102):
            await bot._handle_received_item(self.message("Proven quarantined report", timestamp=timestamp))
        bot.dispatch_text.assert_not_awaited()

    async def test_journal_bounds_records_and_expires_old_provenance(self) -> None:
        now = time.time()
        journal = _SignalOutboundJournal()
        with mock.patch.object(signal_module, "_OUTGOING_ECHO_LIMIT", 3):
            await journal.remember([("group", "text", _echo_hash(str(index))) for index in range(8)])
        self.assertEqual(len(journal._records), 3)
        with mock.patch.object(signal_module.time, "time", return_value=now + 8 * 24 * 3600):
            restarted = _SignalOutboundJournal()
            await restarted.load()
            self.assertEqual(restarted._records, {})
            self.assertFalse(restarted.contains("group", text="7", timestamp=None))

    async def test_malformed_or_oversize_journal_fails_closed_before_dispatch_or_send(self) -> None:
        os.makedirs(self.path)
        path = os.path.join(self.path, "signal_outbound.json")
        for raw in ("{broken", '{"version":99,"records":[]}', "x" * 200):
            with open(path, "w", encoding="utf-8") as target:
                target.write(raw)
            with mock.patch.object(signal_module, "_OUTGOING_JOURNAL_MAX_BYTES", 100):
                bot = self.bot()
                with self.assertRaises(SignalCliError):
                    await bot._handle_received_item(self.message("Unproven request"))
                with self.assertRaises(SignalCliError):
                    await bot.send_text("group", "Reply")
            bot.dispatch_text.assert_not_awaited()
            bot._rpc_request.assert_not_awaited()


class SignalOwnDeviceConfigTests(unittest.TestCase):
    def test_explicit_numeric_device_is_loaded_and_wired_into_platform(self) -> None:
        body = {"signal_group_urls": ["https://signal.group/#fixture"],
                "signal_jsonrpc_socket": "tcp://127.0.0.1:7584", "signal_own_device_id": 2}
        with temporary_config(body):
            loaded = config.load_config()
        bots = create_platforms(loaded)
        self.assertEqual(len(bots), 1)
        self.assertIsInstance(bots[0], SignalBot)
        self.assertEqual(bots[0].own_device_id, 2)

    def test_missing_zero_and_null_preserve_unset_device_compatibility(self) -> None:
        base = {"signal_group_urls": ["https://signal.group/#fixture"],
                "signal_jsonrpc_socket": "tcp://127.0.0.1:7584"}
        for changes in ({}, {"signal_own_device_id": 0}, {"signal_own_device_id": None}):
            with self.subTest(changes=changes), temporary_config({**base, **changes}):
                loaded = config.load_config()
                self.assertEqual(loaded["signal_own_device_id"], 0)
                self.assertIsNone(create_platforms(loaded)[0].own_device_id)

    def test_invalid_device_config_is_rejected_instead_of_silently_disabling_guard(self) -> None:
        for value in (True, False, -1, 2.0, "2", [], {}):
            with self.subTest(value=value), temporary_config({"signal_own_device_id": value}):
                with mock.patch("builtins.print"):
                    with self.assertRaises(SystemExit) as error:
                        config.load_config()
                self.assertEqual(error.exception.code, 1)
                with self.assertRaisesRegex(ValueError, "signal_own_device_id"):
                    SignalBot([], jsonrpc_socket="tcp://127.0.0.1:7584", own_device_id=value)

