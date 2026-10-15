"""Literal prose runs retain exact Signal body, spans, and rich chunks."""

import hashlib
import json
import random
import unittest
from unittest import mock

from Cozter.backends_bot import signal


def _literal_fixtures():
    yield from (
        "", "ordinary prose without markup", "literal🙂한글 " * 1800,
        "# Heading **bold _nested_** and `code with * stars`",
        "before **bold literal🙂text** then _italic literal_ and ~~gone~~ ||secret||",
        r"C:\folder_name\output.txt https://example.com/a_b?x=1&y=2 <tag> &amp; [label](url)",
        "\\_escaped \\*star \\**bold** and **unfinished",
        "```python\nplain \t _field **untouched**\n```\nplain after",
        "~~~~\nlong literal **markdown**\n~~~~\n**following**",
        "ordinary\x00\x1c\v\r\t\u2028\u2003Unicode🙂&entities; text",
        "**" + "literal🙂" * 2000 + "**",
        " _member *field bare~tilde and|pipe `unfinished",
    )
    generator = random.Random(372145)
    atoms = [
        "long ordinary literal passage ", "plain words ", "🙂한글", "\n", "\t", "\r",
        "\u2028", "\x00", "# heading ", "\\escaped", r"C:\path\item_name",
        "**bold**", "__bold__", "*italic*", "_italic_", "_identifier", "*member",
        "`literal_code`", "~~deleted~~", "||spoiler||", "[link](url)", "&amp;<tag>",
        "*", "_", "~", "|", "`", "**", "\n```python\n", "\n```\n", "\n~~~\n",
    ]
    for _ in range(500):
        yield "".join(generator.choice(atoms) for _ in range(generator.randint(10, 100)))


def _literal_fixture_digest():
    rendered = [
        [signal._md_to_signal_body_and_spans(text),
         signal._signal_rich_text_chunks(text, limit=23),
         signal._signal_rich_text_chunks(text)]
        for text in _literal_fixtures()
    ]
    encoded = json.dumps(rendered, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SignalLiteralRunTests(unittest.TestCase):
    def test_full_body_span_and_chunk_golden_matches_previous_character_parser(self):
        self.assertEqual(_literal_fixture_digest(), "02623922cb3e656d3868359331800c7e148a644048c2eeea1ab4aa5c4b519eaa")

    def test_four_mib_literal_run_is_preserved_without_per_character_marker_calls(self):
        text = "ordinary text with Unicode🙂한글 and punctuation.&<> ".ljust(64, "x") * (4 * 1024 * 1024 // 64)
        self.assertEqual(len(text), 4 * 1024 * 1024)
        calls = 0
        detector = signal._signal_inline_marker_at

        def observe(value, index):
            nonlocal calls
            calls += 1
            return detector(value, index)

        with mock.patch.object(signal, "_signal_inline_marker_at", observe):
            body, spans = signal._md_to_signal_body_and_spans(text)
        self.assertEqual(body, text)
        self.assertEqual(spans, [])
        self.assertLessEqual(calls, 2)

    def test_all_trigger_characters_still_reach_the_existing_marker_detector(self):
        text = "plain **bold** __under__ *italic* _italics_ ~~gone~~ ||hidden|| `code` tail"
        self.assertEqual(signal._signal_rich_text_chunks(text), [
            ("plain bold under italic italics gone hidden code tail", [
                "6:4:BOLD", "11:5:BOLD", "17:6:ITALIC", "24:7:ITALIC",
                "32:4:STRIKETHROUGH", "37:6:SPOILER", "44:4:MONOSPACE",
            ]),
        ])
