"""Recovery and lifecycle regressions found in the 2026-10-13 bot audit."""

from types import SimpleNamespace
import unittest
from unittest import mock

from telegram.error import BadRequest, NetworkError

from Cozter.backends_bot.base import MessageHandle
from Cozter.backends_bot.telegram import TelegramBot
from Cozter.tests.helpers import TestBot


class ReplyRecoveryQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_input_preserves_backlog_with_malformed_entries(self):
        bot = TestBot(["user"])
        queue = bot._ensure_message_queue("user")
        malformed = [("short",), None, {"id": "completed"}]
        completed = ("already answered", "chat", "completed", False)
        later = ("later prompt", "chat", "later", False)
        for entry in [*malformed, completed, later]:
            queue.put_nowait(entry)
        with mock.patch.object(bot, "_persist_complete", new=mock.AsyncMock()) as complete:
            self.assertTrue(await bot._complete_reply_input({
                "user_id": "user", "id": "completed",
            }))
        complete.assert_awaited_once_with("user", "completed")
        self.assertEqual(
            [queue.get_nowait() for _ in range(queue.qsize())],
            [*malformed, later],
        )


class TelegramRichFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_ambiguous_send_network_error_does_not_send_plain_duplicate(self):
        send = mock.AsyncMock(side_effect=NetworkError("lost acknowledgement"))
        bot = TelegramBot("fixture-token", ["user"])
        bot.app = SimpleNamespace(bot=SimpleNamespace(send_message=send))
        with self.assertRaises(NetworkError):
            await bot.send_text("chat", "**answer**", rich=True)
        send.assert_awaited_once_with(
            chat_id="chat", text="<b>answer</b>", parse_mode="HTML",
        )

    async def test_rejected_send_markup_falls_back_to_plain_text(self):
        send = mock.AsyncMock(side_effect=[
            BadRequest("Can't parse entities"), SimpleNamespace(message_id=7),
        ])
        bot = TelegramBot("fixture-token", ["user"])
        bot.app = SimpleNamespace(bot=SimpleNamespace(send_message=send))
        self.assertEqual(
            await bot.send_text("chat", "**answer**", rich=True),
            MessageHandle("chat", "7"),
        )
        self.assertEqual(send.await_args_list, [
            mock.call(chat_id="chat", text="<b>answer</b>", parse_mode="HTML"),
            mock.call(chat_id="chat", text="answer"),
        ])

    async def test_ambiguous_edit_network_error_is_propagated_without_retry(self):
        edit = mock.AsyncMock(side_effect=NetworkError("lost acknowledgement"))
        bot = TelegramBot("fixture-token", ["user"])
        bot.app = SimpleNamespace(bot=SimpleNamespace(edit_message_text=edit))
        with self.assertRaises(NetworkError):
            await bot.edit_text(MessageHandle("chat", "7"), "**answer**", rich=True)
        edit.assert_awaited_once()

    async def test_rejected_edit_markup_falls_back_to_plain_text(self):
        edit = mock.AsyncMock(side_effect=[BadRequest("Can't parse entities"), None])
        bot = TelegramBot("fixture-token", ["user"])
        bot.app = SimpleNamespace(bot=SimpleNamespace(edit_message_text=edit))
        await bot.edit_text(MessageHandle("chat", "7"), "**answer**", rich=True)
        self.assertEqual(edit.await_args_list, [
            mock.call(chat_id="chat", message_id=7, text="<b>answer</b>", parse_mode="HTML"),
            mock.call(chat_id="chat", message_id=7, text="answer"),
        ])


class TelegramShutdownTests(unittest.IsolatedAsyncioTestCase):
    def _bot(self, *, polling: bool, running: bool):
        bot = TelegramBot("fixture-token", ["user"])
        updater = SimpleNamespace(running=polling, stop=mock.AsyncMock())
        app = SimpleNamespace(
            updater=updater, running=running,
            stop=mock.AsyncMock(), shutdown=mock.AsyncMock(),
        )
        bot.app = app
        bot._stop_daemon_services = mock.AsyncMock()
        return bot, app

    async def test_partial_start_without_polling_still_stops_application(self):
        bot, app = self._bot(polling=False, running=True)
        app.updater.stop.side_effect = RuntimeError("Updater is not running")
        await bot.stop()
        app.updater.stop.assert_not_awaited()
        app.stop.assert_awaited_once()
        app.shutdown.assert_awaited_once()
        self.assertIsNone(bot.app)

    async def test_initialized_application_without_start_is_shut_down(self):
        bot, app = self._bot(polling=False, running=False)
        app.updater.stop.side_effect = RuntimeError("Updater is not running")
        app.stop.side_effect = RuntimeError("Application is not running")
        await bot.stop()
        app.updater.stop.assert_not_awaited()
        app.stop.assert_not_awaited()
        app.shutdown.assert_awaited_once()

    async def test_failed_polling_stop_still_attempts_remaining_cleanup(self):
        bot, app = self._bot(polling=True, running=True)
        failure = RuntimeError("fixture polling shutdown failure")
        app.updater.stop.side_effect = failure
        with self.assertRaises(RuntimeError) as raised:
            await bot.stop()
        self.assertIs(raised.exception, failure)
        app.stop.assert_awaited_once()
        app.shutdown.assert_awaited_once()
        self.assertIsNone(bot.app)

    async def test_running_application_shuts_down_once_and_stop_is_idempotent(self):
        bot, app = self._bot(polling=True, running=True)
        await bot.stop()
        await bot.stop()
        app.updater.stop.assert_awaited_once()
        app.stop.assert_awaited_once()
        app.shutdown.assert_awaited_once()
        self.assertEqual(bot._stop_daemon_services.await_count, 2)


if __name__ == "__main__":
    unittest.main()
