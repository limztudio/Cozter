"""Bounded patch validation retains exact UTF-8 and splitlines limits."""

import itertools
import unittest
from unittest import mock

from Cozter.agent_tools.base import utf8_byte_limit_exceeded
from Cozter.agent_tools.builtin import apply_patch


class PatchLimitOptimizationTests(unittest.TestCase):
    def test_utf8_limits_match_encoded_size_at_every_codepoint_width(self) -> None:
        samples = (
            "", "ASCII", "é", "한", "🙂", "aé한🙂", "\ud800", "\udfff",
            "left\ud800middle\udfffright", "\r\n", "\n", "\r\r\n",
            "a\nb\r\nc\rd\n",
        )
        for text in samples:
            for restore_crlf in (False, True):
                restored = (
                    text.replace("\r\n", "\n").replace("\n", "\r\n")
                    if restore_crlf else text
                )
                size = len(restored.encode("utf-8", errors="surrogatepass"))
                for limit in {0, 1, max(0, size - 1), size, size + 1}:
                    with self.subTest(text=repr(text), restore=restore_crlf, limit=limit):
                        self.assertEqual(utf8_byte_limit_exceeded(
                            text, limit, restore_crlf=restore_crlf,
                        ), size > limit)

    def test_crlf_and_multibyte_boundaries_remain_exact_across_chunks(self) -> None:
        for ending in ("\r\n", "\r\r\n", "\n\r\n", "🙂\n", "\ud800\n"):
            for padding in (4094, 4095, 4096, 8191):
                text = "a" * padding + ending + "끝\r\n"
                for restore in (False, True):
                    restored = (
                        text.replace("\r\n", "\n").replace("\n", "\r\n")
                        if restore else text
                    )
                    size = len(restored.encode("utf-8", errors="surrogatepass"))
                    self.assertFalse(utf8_byte_limit_exceeded(text, size, restore_crlf=restore))
                    self.assertTrue(utf8_byte_limit_exceeded(text, size - 1, restore_crlf=restore))

    def test_encoding_allocations_are_bounded_and_huge_arguments_reject_early(self) -> None:
        encoded_lengths = []

        class TrackedText(str):
            def __getitem__(self, key):
                return TrackedText(super().__getitem__(key))

            def encode(self, *args, **kwargs):
                encoded_lengths.append(len(self))
                return super().encode(*args, **kwargs)

        text = TrackedText("🙂" * 20_000)
        self.assertFalse(utf8_byte_limit_exceeded(text, 80_000))
        self.assertGreater(len(encoded_lengths), 1)
        self.assertLessEqual(max(encoded_lengths), 4096)
        encoded_lengths.clear()
        self.assertTrue(utf8_byte_limit_exceeded(text, 10))
        self.assertEqual(encoded_lengths, [])

    def test_splitline_limits_match_builtin_for_all_boundary_families(self) -> None:
        parts = (
            "a", "\n", "\r", "\r\n", "\v", "\f", "\x1c", "\x1d",
            "\x1e", "\x85", "\u2028", "\u2029", "\x1f",
        )
        for width in range(4):
            for sequence in itertools.product(parts, repeat=width):
                text = "".join(sequence)
                count = len(text.splitlines())
                for limit in (0, 1, 2, 3, 20_000):
                    self.assertEqual(apply_patch._splitline_limit_exceeded(
                        text, limit,
                    ), count > limit, (repr(text), limit))

    def test_line_counter_stops_at_cap_without_materializing_remainder(self) -> None:
        visited = []

        class Boundaries:
            def finditer(self, _text):
                for number in range(3):
                    visited.append(number)
                    yield mock.Mock(end=lambda: 1)
                raise AssertionError("line counter scanned beyond its cap")

        with mock.patch.object(apply_patch, "_LINE_BREAK_RE", Boundaries()):
            self.assertTrue(apply_patch._splitline_limit_exceeded("\n" * 100_000, 2))
        self.assertEqual(visited, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
