"""Offline regressions for Signal attachment whitespace normalization."""

import asyncio
import base64
import binascii
import re
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from Cozter.backends_bot import signal
from Cozter.backends_bot.base import UploadTooLargeError


class SignalAttachmentWhitespaceTests(unittest.TestCase):
    def test_unbroken_ascii_payload_reuses_input_without_regex(self):
        payload = base64.b64encode(b"attachment bytes").decode("ascii")
        with mock.patch.object(signal, "_SIGNAL_ATTACH_WHITESPACE_RE") as regex:
            normalized = signal._strip_attachment_whitespace(payload)
        self.assertIs(normalized, payload)
        regex.sub.assert_not_called()
        self.assertEqual(signal._decode_attachment_payload(normalized, 100), b"attachment bytes")

    def test_all_ascii_whitespace_preserves_decoded_equivalence(self):
        data = bytes(range(64))
        encoded = base64.b64encode(data).decode("ascii")
        for code in range(128):
            whitespace = chr(code)
            if not whitespace.isspace():
                continue
            with self.subTest(code=code):
                payload = whitespace + whitespace.join(encoded) + whitespace
                normalized = signal._strip_attachment_whitespace(payload)
                self.assertEqual(normalized, re.sub(r"\s+", "", payload))
                self.assertEqual(signal._decode_attachment_payload(normalized, len(data)), data)

    def test_unicode_whitespace_falls_back_to_original_normalization(self):
        regex = signal._SIGNAL_ATTACH_WHITESPACE_RE
        payload = "\u00a0Y\u0085W\u2003J\u2028j\u202f"
        with mock.patch.object(signal, "_SIGNAL_ATTACH_WHITESPACE_RE", wraps=regex) as wrapped:
            normalized = signal._strip_attachment_whitespace(payload)
        wrapped.sub.assert_called_once_with("", payload)
        self.assertEqual(signal._decode_attachment_payload(normalized, 3), b"abc")

    def test_invalid_base64_remains_strictly_rejected(self):
        for payload in ("YWJ$", "\tYW J$\n", "\x00YWJj"):
            with self.subTest(payload=repr(payload)), self.assertRaises(binascii.Error):
                signal._decode_attachment_payload(signal._strip_attachment_whitespace(payload), 100)

    def test_non_ascii_non_whitespace_remains_invalid(self):
        payload = "YWJj\ud800"
        self.assertEqual(signal._strip_attachment_whitespace(payload), payload)
        with self.assertRaises(ValueError):
            signal._decode_attachment_payload(payload, 100)

    def test_encoded_and_decoded_size_caps_are_unchanged(self):
        encoded = base64.b64encode(b"abc").decode("ascii")
        normalized = signal._strip_attachment_whitespace(" \n" + encoded + "\t\r")
        self.assertEqual(signal._decode_attachment_payload(normalized, 3), b"abc")
        with self.assertRaises(UploadTooLargeError):
            signal._decode_attachment_payload(normalized, 2)
        too_long = base64.b64encode(b"abcd").decode("ascii")
        with self.assertRaises(UploadTooLargeError):
            signal._decode_attachment_payload(signal._strip_attachment_whitespace(too_long), 3)

    def test_empty_or_whitespace_only_payload_stays_empty(self):
        for payload in ("", " \t\n\r\x1c", "\u00a0\u2003"):
            with self.subTest(payload=repr(payload)):
                normalized = signal._strip_attachment_whitespace(payload)
                self.assertEqual(normalized, "")
                self.assertEqual(signal._decode_attachment_payload(normalized, 0), b"")


class SignalAttachmentWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_decode_and_atomic_publish_yield_loop_and_preserve_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = signal.SignalBot([], jsonrpc_socket="tcp://127.0.0.1:7583")
            bot._rpc_request = mock.AsyncMock(return_value={"data": " YW Jj\n"})
            started = asyncio.Event()
            release = threading.Event()
            loop = asyncio.get_running_loop()
            main_thread = threading.get_ident()
            threads = []
            decode = signal._decode_attachment_payload

            def blocked_decode(*args):
                threads.append(threading.get_ident())
                loop.call_soon_threadsafe(started.set)
                if not release.wait(5):
                    raise AssertionError("attachment decode never released")
                return decode(*args)

            with mock.patch.object(signal, "_decode_attachment_payload", blocked_decode):
                task = asyncio.create_task(bot._materialize_attachment(
                    {"id": "fixture", "filename": "fixture.bin"}, "group", directory, "caption",
                ))
                try:
                    await asyncio.wait_for(started.wait(), 2)
                    self.assertFalse(task.done())
                    self.assertNotEqual(threads, [main_thread])
                    release.set()
                    attachment = await asyncio.wait_for(task, 2)
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)
            self.assertIsNotNone(attachment)
            self.assertEqual(Path(attachment.local_path).read_bytes(), b"abc")

    async def test_repeated_cancel_settles_writer_before_reservation_cleanup(self):
        for late_failure in (False, True):
            with self.subTest(late_failure=late_failure), tempfile.TemporaryDirectory() as directory:
                bot = signal.SignalBot([], jsonrpc_socket="tcp://127.0.0.1:7583")
                bot._rpc_request = mock.AsyncMock(return_value={"data": "YWJj"})
                started = asyncio.Event()
                release = threading.Event()
                settled = threading.Event()
                loop = asyncio.get_running_loop()
                write = signal.write_bytes_atomically

                def blocked_write(*args):
                    loop.call_soon_threadsafe(started.set)
                    try:
                        if not release.wait(5):
                            raise AssertionError("attachment writer never released")
                        write(*args)
                        if late_failure:
                            raise OSError("late attachment publish failure")
                    finally:
                        settled.set()

                with mock.patch.object(signal, "write_bytes_atomically", blocked_write):
                    task = asyncio.create_task(bot._materialize_attachment(
                        {"id": "fixture", "filename": "fixture.bin"}, "group", directory, "",
                    ))
                    try:
                        await asyncio.wait_for(started.wait(), 2)
                        task.cancel()
                        await asyncio.sleep(0)
                        task.cancel()
                        await asyncio.sleep(0)
                        self.assertFalse(task.done())
                        self.assertFalse(settled.is_set())
                        release.set()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    finally:
                        release.set()
                        await asyncio.gather(task, return_exceptions=True)
                self.assertTrue(settled.is_set())
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
