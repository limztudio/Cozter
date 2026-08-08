"""Regression coverage for bounded chat-platform attachments."""

import base64
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from Cozter import config
from Cozter.backends_bot.base import (
    AttachmentInfo,
    BotContext,
    _INLINE_SIZE_LIMIT,
    UploadTooLargeError,
    reserve_upload_path,
    upload_limit_message,
    write_bytes_atomically,
)
from Cozter.backends_bot.signal import SignalBot
from Cozter.backends_bot.slack import SlackBot, _download_private
from Cozter.backends_bot.telegram import (
    TelegramBot,
    _download_telegram_file,
)
from Cozter.tests.helpers import temporary_config


class _OversizeDownloadStream:
    async def iter_chunked(self, _size):
        yield b"abc"
        yield b"d"


class _OversizeDownloadResponse:
    content_length = None
    content = _OversizeDownloadStream()

    def raise_for_status(self) -> None:
        return None


class _OversizeDownloadRequest:
    async def __aenter__(self):
        return _OversizeDownloadResponse()

    async def __aexit__(self, *_args):
        return None


class _OversizeDownloadSession:
    def __init__(self):
        self.request_kwargs: dict[str, object] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def get(self, _url, **kwargs):
        self.request_kwargs = kwargs
        return _OversizeDownloadRequest()


class UploadLimitConfigTests(unittest.TestCase):
    def test_loader_normalizes_an_invalid_platform_limit(self) -> None:
        with temporary_config({
            "telegram_bot_tokens": ["token"],
            "user_ids": [123],
            "max_upload_bytes": 0,
        }):
            loaded = config.load_config()

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

    async def test_text_attachment_reads_only_inline_limit_plus_one(self) -> None:
        bot = SignalBot(
            ["https://signal.group/#test"], jsonrpc_socket="/tmp/signal.sock",
        )
        bot._require_ws = mock.AsyncMock(return_value="/workspace")
        bot._dispatch_ai = mock.AsyncMock()
        attachment = AttachmentInfo(
            local_path="/workspace/large.txt",
            filename="large.txt",
            kind="document",
        )
        ctx = BotContext(
            user_id="u1",
            chat_id="chat",
            text="",
            command=None,
            args="",
            attachment=attachment,
            platform=bot,
        )
        opened = mock.mock_open(
            read_data="x" * (_INLINE_SIZE_LIMIT + 1),
        )

        with mock.patch("builtins.open", opened):
            await bot._ai_file(ctx)

        opened.assert_called_once_with(
            attachment.local_path, encoding="utf-8", errors="replace",
        )
        opened.return_value.read.assert_called_once_with(_INLINE_SIZE_LIMIT + 1)
        prompt = bot._dispatch_ai.await_args.args[1]
        self.assertIn("over 50,000 chars", prompt)
        self.assertNotIn("[File contents", prompt)

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
        file = SimpleNamespace(file_path="https://files.example/large")
        with tempfile.TemporaryDirectory() as tmp:
            destination = os.path.join(tmp, "large.bin")
            download_session = _OversizeDownloadSession()
            with mock.patch(
                "Cozter.backends_bot.base.aiohttp.ClientSession",
                return_value=download_session,
            ):
                with self.assertRaises(UploadTooLargeError):
                    await _download_telegram_file(file, destination, 3)

            self.assertEqual(download_session.request_kwargs, {"headers": None})
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
        with tempfile.TemporaryDirectory() as tmp:
            destination = os.path.join(tmp, "large.bin")
            download_session = _OversizeDownloadSession()
            with mock.patch(
                "Cozter.backends_bot.base.aiohttp.ClientSession",
                return_value=download_session,
            ):
                with self.assertRaises(UploadTooLargeError):
                    await _download_private(
                        "https://files.example/large",
                        "bot-token",
                        destination,
                        max_upload_bytes=3,
                    )

            self.assertEqual(download_session.request_kwargs, {
                "headers": {"Authorization": "Bearer bot-token"},
            })
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

    async def test_signal_attachment_storage_writes_complete_files(self) -> None:
        """Both shared atomic writers preserve valid Signal attachments."""
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "signal-source.bin")
            upload_dir = os.path.join(tmp, "uploads")
            os.mkdir(upload_dir)
            with open(source, "wb") as f:
                f.write(b"abc")

            bot = SignalBot(
                ["https://signal.group/#test"],
                jsonrpc_socket="/tmp/signal.sock",
                max_upload_bytes=3,
            )
            local = await bot._materialize_attachment(
                {"path": source, "filename": "local.bin"},
                "group",
                upload_dir,
                "",
            )
            assert local is not None
            with open(local.local_path, "rb") as f:
                self.assertEqual(f.read(), b"abc")

            bot._rpc_request = mock.AsyncMock(
                return_value=base64.b64encode(b"xyz").decode(),
            )
            remote = await bot._materialize_attachment(
                {"id": "remote", "filename": "remote.bin"},
                "group",
                upload_dir,
                "",
            )
            assert remote is not None
            with open(remote.local_path, "rb") as f:
                self.assertEqual(f.read(), b"xyz")

    async def test_upload_path_reservation_keeps_same_names_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            with reserve_upload_path(upload_dir, "report.txt") as first:
                with reserve_upload_path(upload_dir, "report.txt") as second:
                    write_bytes_atomically(first, b"first")
                    write_bytes_atomically(second, b"second")

            self.assertEqual(os.path.basename(first), "report.txt")
            self.assertEqual(os.path.basename(second), "report (2).txt")
            with open(first, "rb") as f:
                self.assertEqual(f.read(), b"first")
            with open(second, "rb") as f:
                self.assertEqual(f.read(), b"second")

    async def test_upload_path_reservation_is_removed_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            with self.assertRaisesRegex(RuntimeError, "download failed"):
                with reserve_upload_path(upload_dir, "report.txt"):
                    raise RuntimeError("download failed")

            self.assertEqual(os.listdir(upload_dir), [])

    async def test_slack_same_name_attachments_keep_original_names(self) -> None:
        bot = SlackBot("bot-token", "app-token", ["C1"])
        bot._bot_user_id = "B1"
        attachments = []

        async def fake_download(
            url: str,
            _token: str,
            local_path: str,
            *,
            max_upload_bytes: int,
        ) -> None:
            self.assertGreater(max_upload_bytes, 0)
            write_bytes_atomically(local_path, url.encode())

        bot.dispatch_file = mock.AsyncMock(
            side_effect=lambda ctx: attachments.append(ctx.attachment),
        )
        files = [
            {
                "id": "F1",
                "name": "report.txt",
                "url_private_download": "https://files.example/first",
            },
            {
                "id": "F2",
                "name": "report.txt",
                "url_private_download": "https://files.example/second",
            },
        ]

        with tempfile.TemporaryDirectory() as ws, mock.patch(
            "Cozter.backends_bot.slack.workspace.get_current",
            return_value=ws,
        ), mock.patch(
            "Cozter.backends_bot.slack._download_private",
            side_effect=fake_download,
        ):
            await bot._handle_files(
                {"user": "U1", "channel": "C1"}, files, "",
            )
            with open(attachments[0].local_path, "rb") as f:
                self.assertEqual(f.read(), b"https://files.example/first")
            with open(attachments[1].local_path, "rb") as f:
                self.assertEqual(f.read(), b"https://files.example/second")

        self.assertEqual([att.filename for att in attachments], [
            "report.txt", "report.txt",
        ])
        self.assertNotEqual(
            attachments[0].local_path, attachments[1].local_path,
        )

    async def test_telegram_same_name_attachments_keep_original_names(
        self,
    ) -> None:
        bot = TelegramBot("token", ["1"])
        bot.app = SimpleNamespace(bot=SimpleNamespace(id=99))
        attachments = []

        def update_for(file_id: str):
            media = SimpleNamespace(
                file_name="report.txt",
                file_id=file_id,
                file_size=3,
                get_file=mock.AsyncMock(
                    return_value=SimpleNamespace(file_id=file_id),
                ),
            )
            return SimpleNamespace(
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

        async def fake_download(
            tg_file: object, local_path: str, _max_upload_bytes: int,
        ) -> None:
            write_bytes_atomically(
                local_path, str(getattr(tg_file, "file_id")).encode(),
            )

        bot.dispatch_file = mock.AsyncMock(
            side_effect=lambda ctx: attachments.append(ctx.attachment),
        )
        with tempfile.TemporaryDirectory() as ws, mock.patch(
            "Cozter.backends_bot.telegram.workspace.get_current",
            return_value=ws,
        ), mock.patch(
            "Cozter.backends_bot.telegram._download_telegram_file",
            side_effect=fake_download,
        ):
            await bot._on_file(update_for("first"), None)
            await bot._on_file(update_for("second"), None)
            with open(attachments[0].local_path, "rb") as f:
                self.assertEqual(f.read(), b"first")
            with open(attachments[1].local_path, "rb") as f:
                self.assertEqual(f.read(), b"second")

        self.assertEqual([att.filename for att in attachments], [
            "report.txt", "report.txt",
        ])
        self.assertNotEqual(
            attachments[0].local_path, attachments[1].local_path,
        )

    async def test_signal_same_name_attachments_keep_original_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_one = os.path.join(tmp, "source-one.txt")
            source_two = os.path.join(tmp, "source-two.txt")
            upload_dir = os.path.join(tmp, "uploads")
            os.mkdir(upload_dir)
            with open(source_one, "wb") as f:
                f.write(b"first")
            with open(source_two, "wb") as f:
                f.write(b"second")

            bot = SignalBot(
                ["https://signal.group/#test"],
                jsonrpc_socket="/tmp/signal.sock",
            )
            first = await bot._materialize_attachment(
                {"path": source_one, "filename": "report.txt"},
                "group", upload_dir, "",
            )
            second = await bot._materialize_attachment(
                {"path": source_two, "filename": "report.txt"},
                "group", upload_dir, "",
            )

            assert first is not None
            assert second is not None
            self.assertEqual(first.filename, "report.txt")
            self.assertEqual(second.filename, "report.txt")
            self.assertNotEqual(first.local_path, second.local_path)
            with open(first.local_path, "rb") as f:
                self.assertEqual(f.read(), b"first")
            with open(second.local_path, "rb") as f:
                self.assertEqual(f.read(), b"second")


if __name__ == "__main__":
    unittest.main()
