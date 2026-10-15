"""Signal style projection visits only overlapping message chunks."""

import unittest
from unittest import mock

from Cozter.backends_bot import signal


class SignalSpanProjectionTests(unittest.TestCase):
    def test_postorder_crossing_spans_preserve_order_and_utf16_offsets(self):
        body = "A😀B😀C"
        spans = [(3, 2, "ITALIC"), (1, 4, "BOLD"), (0, 5, "MONOSPACE")]
        with mock.patch.object(signal, "_md_to_signal_body_and_spans", return_value=(body, spans)):
            self.assertEqual(signal._signal_rich_text_chunks("fixture", limit=2), [
                ("A😀", ["1:2:BOLD", "0:3:MONOSPACE"]),
                ("B😀", ["1:2:ITALIC", "0:3:BOLD", "0:3:MONOSPACE"]),
                ("C", ["0:1:ITALIC", "0:1:BOLD", "0:1:MONOSPACE"]),
            ])

    def test_each_projection_receives_only_its_overlapping_spans(self):
        body = "x" * 20_000
        spans = [(index, 1, "BOLD") for index in range(0, len(body), 2)]
        original = signal._signal_style_strings_for_chunk
        visits = []

        def project(value, selected, start, end):
            visits.append(len(selected))
            self.assertTrue(all(pos < end and pos + length > start for pos, length, _ in selected))
            return original(value, selected, start, end)

        with (
            mock.patch.object(signal, "_md_to_signal_body_and_spans", return_value=(body, spans)),
            mock.patch.object(signal, "_signal_style_strings_for_chunk", side_effect=project),
        ):
            chunks = signal._signal_rich_text_chunks("fixture", limit=200)
        self.assertEqual("".join(text for text, _ in chunks), body)
        self.assertEqual(sum(visits), len(spans))
        self.assertEqual(len(chunks), 100)

    def test_empty_and_unstyled_text_keep_their_existing_chunks(self):
        self.assertEqual(signal._signal_rich_text_chunks(""), [])
        self.assertEqual(signal._signal_rich_text_chunks("abcd", limit=2), [("ab", []), ("cd", [])])
