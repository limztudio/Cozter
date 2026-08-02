"""Regression coverage for bounded chat-platform attachments."""

import base64
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from Cozter import config
from Cozter.backends_bot.base import UploadTooLargeError, upload_limit_message
from Cozter.backends_bot.signal import SignalBot
from Cozter.backends_bot.slack import SlackBot, _download_private
from Cozter.backends_bot.telegram import (
    TelegramBot,
    _download_telegram_file,
)


class UploadLimitConfigTests(unittest.TestCase):
    def test_getter_uses_override_and_rejects_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            old_path = config.CONFIG_PATH
            config.CONFIG_PATH = path
            try:
                self.assertEqual(
                    config.get_max_upload_bytes(),
                    config.DEFAULT_MAX_UPLOAD_BYTES,
                )
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"max_upload_bytes": 123}, f)
                self.assertEqual(config.get_max_upload_bytes(), 123)

                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"max_upload_bytes": True}, f)
                self.assertEqual(
                    config.get_max_upload_bytes(),
                    config.DEFAULT_MAX_UPLOAD_BYTES,
                )
            finally:
                config.CONFIG_PATH = old_path

    def test_loader_normalizes_an_invalid_platform_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "telegram_bot_tokens": ["token"],
                    "user_ids": [123],
                    "max_upload_bytes": 0,
                }, f)

            old_path = config.CONFIG_PATH
            config.CONFIG_PATH = path
            try:
                loaded = config.load_config()
            finally:
                config.CONFIG_PATH = old_path

            self.assertEqual(
                loaded["max_upload_bytes"], config.DEFAULT_MAX_UPLOAD_BYTES,
            )


class UploadLimitPlatformTests(unittest.IsolatedAsyncioTestCase):
    async def test_outbound_uploads_are_checked_before_platform_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "large.bin")
            with open(path, "wb") as f:
                f.write(b"four")

            bots = [
                TelegramBot("token", ["user"], max_upload_bytes=3),
                SlackBot("bot-token", "app-token", ["C1"], max_upload_bytes=3),
                SignalBot(
                    ["https://signal.group/#test"],
                    jsonrpc_socket="/tmp/signal.sock",
                    max_upload_bytes=3,
                ),
            ]
            for bot in bots:
                with self.subTest(platform=type(bot).__name__):
                    with self.assertRaises(UploadTooLargeError):
                        await bot.send_file("chat", path)

    async def test_telegram_rejects_metadata_before_requesting_file(self) -> None:
        bot = TelegramBot("token", ["1"], max_upload_bytes=3)
        bot.app = SimpleNamespace(bot=SimpleNamespace(id=99))
        bot.send_text = mock.AsyncMock(return_value=None)
        media = SimpleNamespace(
            file_name="large.bin",
            file_id="file-id",
            file_size=4,
            get_file=mock.AsyncMock(),
        )
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=1, full_name="Test User"),
            effective_chat=SimpleNamespace(id=10),
            message=SimpleNamespace(
                document=media,
                photo=None,
                audio=None,
                video=None,
                voice=None,
                video_note=None,
                caption="",
            ),
        )

        with tempfile.TemporaryDirectory() as ws, mock.patch(
            "Cozter.backends_bot.telegram.workspace.get_current",
            return_value=ws,
        ):
            await bot._on_file(update, None)

        media.get_file.assert_not_awaited()
        bot.send_text.assert_awaited_once_with(
            "10", upload_limit_message(3), rich=False,
        )

    async def test_telegram_unknown_size_uses_bounded_downloader(
        self,
    ) -> None:
        bot = TelegramBot("token", ["1"], max_upload_bytes=3)
        bot.app = SimpleNamespace(bot=SimpleNamespace(id=99))
        bot.send_text = mock.AsyncMock(return_value=None)

        class DownloadedFile:
            file_id = "file-id"

        media = SimpleNamespace(
            file_name="large.bin",
            file_id="file-id",
            file_size=None,
            get_file=mock.AsyncMock(return_value=DownloadedFile()),
        )
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=1, full_name="Test User"),
            effective_chat=SimpleNamespace(id=10),
            message=SimpleNamespace(
                document=media,
                photo=None,
                audio=None,
                video=None,
                voice=None,
                video_note=None,
                caption="",
            ),
        )

        with tempfile.TemporaryDirectory() as ws, mock.patch(
            "Cozter.backends_bot.telegram.workspace.get_current",
            return_value=ws,
        ), mock.patch(
            "Cozter.backends_bot.telegram._download_telegram_file",
            new=mock.AsyncMock(side_effect=UploadTooLargeError(3)),
        ) as download:
            await bot._on_file(update, None)
            self.assertFalse(os.path.exists(
                os.path.join(ws, ".cozter", "uploads", "large.bin"),
            ))

        download.assert_awaited_once()

        bot.send_text.assert_awaited_once_with(
            "10", upload_limit_message(3), rich=False,
        )

    async def test_telegram_stream_limit_removes_partial_download(self) -> None:
        class Stream:
            async def iter_chunked(self, _size):
                yield b"abc"
                yield b"d"

        class Response:
            content_length = None
            content = Stream()

            def raise_for_status(self) -> None:
                return None

        class Request:
            async def __aenter__(self):
                return Response()

            async def __aexit__(self, *_args):
                return None

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def get(self, _url, **_kwargs):
                return Request()

        file = SimpleNamespace(file_path="https://files.example/large")
        with tempfile.TemporaryDirectory() as tmp:
            destination = os.path.join(tmp, "large.bin")
            with mock.patch(
                "Cozter.backends_bot.telegram.aiohttp.ClientSession",
                return_value=Session(),
            ):
                with self.assertRaises(UploadTooLargeError):
                    await _download_telegram_file(file, destination, 3)

            self.assertFalse(os.path.exists(destination))
            self.assertEqual(os.listdir(tmp), [])

    async def test_slack_metadata_rejection_skips_private_download(self) -> None:
        bot = SlackBot("bot-token", "app-token", ["C1"], max_upload_bytes=3)
        bot._bot_user_id = "B1"
        bot.send_text = mock.AsyncMock(return_value=None)

        with tempfile.TemporaryDirectory() as ws, mock.patch(
            "Cozter.backends_bot.slack.workspace.get_current",
            return_value=ws,
        ), mock.patch(
            "Cozter.backends_bot.slack._download_private",
            new=mock.AsyncMock(),
        ) as download:
            await bot._handle_files(
                {"user": "U1", "channel": "C1"},
                [{
                    "id": "F1",
                    "name": "large.bin",
                    "size": 4,
                    "url_private_download": "https://files.example/large",
                }],
                "",
            )

        download.assert_not_awaited()
        bot.send_text.assert_awaited_once_with(
            "C1",
            f"Not downloading large.bin: {upload_limit_message(3)}",
            rich=False,
        )

    async def test_slack_stream_limit_removes_partial_download(self) -> None:
        class Stream:
            async def iter_chunked(self, _size):
                yield b"abc"
                yield b"d"

        class Response:
            content_length = None
            content = Stream()

            def raise_for_status(self) -> None:
                return None

        class Request:
            async def __aenter__(self):
                return Response()

            async def __aexit__(self, *_args):
                return None

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def get(self, _url, **_kwargs):
                return Request()

        with tempfile.TemporaryDirectory() as tmp:
            destination = os.path.join(tmp, "large.bin")
            with mock.patch(
                "Cozter.backends_bot.slack.aiohttp.ClientSession",
                return_value=Session(),
            ):
                with self.assertRaises(UploadTooLargeError):
                    await _download_private(
                        "https://files.example/large",
                        "bot-token",
                        destination,
                        max_upload_bytes=3,
                    )

            self.assertFalse(os.path.exists(destination))
            self.assertEqual(os.listdir(tmp), [])

    async def test_signal_local_and_base64_attachments_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "signal-source.bin")
            upload_dir = os.path.join(tmp, "uploads")
            os.mkdir(upload_dir)
            with open(source, "wb") as f:
                f.write(b"four")

            bot = SignalBot(
                ["https://signal.group/#test"],
                jsonrpc_socket="/tmp/signal.sock",
                max_upload_bytes=3,
            )
            with self.assertRaises(UploadTooLargeError):
                await bot._materialize_attachment(
                    {"path": source, "filename": "local.bin"},
                    "group",
                    upload_dir,
                    "",
                )
            self.assertFalse(os.path.exists(os.path.join(upload_dir, "local.bin")))

            bot._rpc_request = mock.AsyncMock(
                return_value=base64.b64encode(b"four").decode(),
            )
            with self.assertRaises(UploadTooLargeError):
                await bot._materialize_attachment(
                    {"id": "remote", "filename": "remote.bin"},
                    "group",
                    upload_dir,
                    "",
                )
            self.assertFalse(os.path.exists(os.path.join(upload_dir, "remote.bin")))


if __name__ == "__main__":
    unittest.main()
