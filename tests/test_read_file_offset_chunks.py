"""Large line offsets retain exact text decoding, scan bounds, and ranges."""

import asyncio
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from Cozter.agent_tools.builtin import read_file
from Cozter.agent_tools.builtin.read_files import ReadFilesTool


def legacy_skip_lines(file, count):
    remaining = read_file._READ_FILE_MAX_SKIP_CHARS
    for _ in range(count):
        while True:
            if remaining <= 0:
                raise read_file._OffsetScanLimitExceededError
            chunk = file.readline(min(read_file._READ_FILE_SKIP_CHUNK_CHARS, remaining))
            if not chunk:
                return False
            remaining -= len(chunk)
            if chunk.endswith("\n"):
                break
    return True


def skip_outcome(raw, count, skip):
    with io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", errors="replace") as stream:
        try:
            result = skip(stream, count)
            return result, stream.read()
        except read_file._OffsetScanLimitExceededError:
            return "scan limit", stream.read()


class ReadFileOffsetChunkTests(unittest.TestCase):
    def test_generated_mixed_newline_unicode_and_invalid_byte_offsets_match_legacy(self):
        generator = random.Random(458012)
        contents = [b"ascii", "café🌐".encode(), b"invalid:\xff\xc3", b"", b" ",
                    "same line\u2028still same\u0085line".encode()]
        endings = [b"\n", b"\r\n", b"\r"]
        for _ in range(250):
            raw = b"".join(generator.choice(contents) + generator.choice(endings)
                           for _ in range(generator.randrange(128, 400)))
            if generator.randrange(2):
                raw += generator.choice(contents)
            count = generator.randrange(120, 420)
            budget = generator.choice([129, 257, 1024, 4096, 16_384])
            with (
                mock.patch.object(read_file, "_READ_FILE_MAX_SKIP_CHARS", budget),
                mock.patch.object(read_file, "_READ_FILE_SKIP_CHUNK_CHARS", 31),
            ):
                self.assertEqual(skip_outcome(raw, count, read_file._skip_lines),
                                 skip_outcome(raw, count, legacy_skip_lines))

    def test_exact_ceiling_eof_and_unterminated_last_line_match_legacy(self):
        for suffix in (b"", b"tail", b"\xc3\xfftail", b"\r", b"\r\n"):
            raw = b"a\n" * 129 + suffix
            for count in (0, 1, 128, 129, 130, 131):
                for budget in (0, 1, 257, 258, 259, 262, 263, 270):
                    with (
                        mock.patch.object(read_file, "_READ_FILE_MAX_SKIP_CHARS", budget),
                        mock.patch.object(read_file, "_READ_FILE_SKIP_CHUNK_CHARS", 17),
                    ):
                        self.assertEqual(skip_outcome(raw, count, read_file._skip_lines),
                                         skip_outcome(raw, count, legacy_skip_lines),
                                         (suffix, count, budget))

    def test_real_text_file_ranges_preserve_decoder_cookies_and_output_cap(self):
        raw = ("café🌐\r\nleft\rright\n".encode() + b"bad\xff\xc3\n") * 400
        raw += b"unterminated\xf0\x9f"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "mixed.log")
            path.write_bytes(raw)
            for offset in (128, 129, 255, 700, 1599, 1600, 1601):
                for limit in (None, 0, 1, 7):
                    with (
                        mock.patch.object(read_file, "_READ_FILE_MAX_CHARS", 41),
                        mock.patch.object(read_file, "_READ_FILE_SKIP_CHUNK_CHARS", 97),
                    ):
                        actual = read_file._read_text_range(str(path), offset, limit)
                        with mock.patch.object(read_file, "_skip_lines", legacy_skip_lines):
                            expected = read_file._read_text_range(str(path), offset, limit)
                    self.assertEqual(actual, expected, (offset, limit))

    def test_chunk_reads_stay_bounded_and_never_seek_past_the_authorized_prefix(self):
        class ObservedStream:
            def __init__(self):
                self.stream = io.StringIO("a\n" * 1000)
                self.read_sizes = []

            def tell(self):
                return self.stream.tell()

            def seekable(self):
                return True

            def seek(self, cookie):
                self.stream.seek(cookie)

            def read(self, count):
                self.read_sizes.append(count)
                return self.stream.read(count)

        stream = ObservedStream()
        with (
            mock.patch.object(read_file, "_READ_FILE_MAX_SKIP_CHARS", 1000),
            mock.patch.object(read_file, "_READ_FILE_SKIP_CHUNK_CHARS", 64),
        ):
            self.assertTrue(read_file._skip_lines(stream, 350))
        self.assertEqual(stream.tell(), 700)
        self.assertTrue(all(0 < size <= 64 for size in stream.read_sizes))
        self.assertLess(len(stream.read_sizes), 20)

    def test_small_offsets_keep_the_readline_path(self):
        class SmallStream(io.StringIO):
            def tell(self):
                raise AssertionError("small offsets should not need a decoder cookie")

            def read(self, *_args):
                raise AssertionError("small offsets should not pre-read a large chunk")

        stream = SmallStream("row\n" * 200)
        self.assertTrue(read_file._skip_lines(stream, 128))
        self.assertEqual(stream.readline(), "row\n")

    def test_nonseekable_and_readline_only_streams_keep_legacy_text_ceiling_and_eof(self):
        class NonSeekableRaw(io.RawIOBase):
            def __init__(self, raw):
                self.source = io.BytesIO(raw)

            def readable(self):
                return True

            def readinto(self, buffer):
                value = self.source.read(len(buffer))
                buffer[:len(value)] = value
                return len(value)

        class ReadlineOnly:
            def __init__(self, text):
                self.text = text

            def readline(self, size):
                return self.text.readline(size)

        raw = ("café🌐\r\n".encode() + b"invalid\xff\r") * 200 + b"unterminated\xc3"
        for readline_only in (False, True):
            for count in (129, 300, 400, 401, 450):
                for budget in (257, 1024, 16_384):
                    results = []
                    for skip in (legacy_skip_lines, read_file._skip_lines):
                        with io.TextIOWrapper(
                            io.BufferedReader(NonSeekableRaw(raw)),
                            encoding="utf-8", errors="replace",
                        ) as stream:
                            self.assertFalse(stream.seekable())
                            supplied = ReadlineOnly(stream) if readline_only else stream
                            with (
                                mock.patch.object(read_file, "_READ_FILE_MAX_SKIP_CHARS", budget),
                                mock.patch.object(read_file, "_READ_FILE_SKIP_CHUNK_CHARS", 31),
                            ):
                                try:
                                    result = skip(supplied, count)
                                except read_file._OffsetScanLimitExceededError:
                                    result = "scan limit"
                            results.append((result, stream.read()))
                    self.assertEqual(results[0], results[1], (readline_only, count, budget))

    def test_read_files_integrates_identical_large_offset_ranges_and_refusals(self):
        async def run(directory):
            Path(directory, "mixed.log").write_bytes(("café🌐\r\nnext\r".encode() + b"bad\xff\n") * 400)
            Path(directory, "short.txt").write_text("one\ntwo\n", encoding="utf-8")
            args = {"files": [
                {"path": "mixed.log", "offset": 350, "limit": 20},
                {"path": "short.txt", "offset": 1, "limit": 1},
                {"path": "missing.txt", "offset": 200},
                {"path": "mixed.log", "offset": 5000, "limit": 10},
            ]}
            actual = await ReadFilesTool().run(directory, args)
            with mock.patch.object(read_file, "_skip_lines", legacy_skip_lines):
                expected = await ReadFilesTool().run(directory, args)
            self.assertEqual(actual, expected)
            self.assertIn("two\n", actual)
            self.assertIn("File not found: missing.txt", actual)

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(directory))
