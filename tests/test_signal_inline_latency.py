"""Signal inline close searches stay bounded without changing rendered output."""

import hashlib
import json
import os
import random
import subprocess
import sys
import unittest

from Cozter.backends_bot.signal import _signal_rich_text_chunks


def _markdown_fixtures():
    fixtures = [
        "Plain text with _field and *member identifiers",
        "# Heading **bold _nested_** and `literal _identifier`",
        "\\_escaped \\*star \\**bold** and **unfinished",
        "Unmatched _names _here _followed by a closing underscore_ end",
        "**" + "🙂한글" * 1800 + "**",
        "```cpp\n _field *member **literal** 🙂\n```\n_following_",
        "~~~text\nunterminated _field\nstays literal",
        "**bold** __also__ *italic* _italic_ ~~gone~~ ||hidden||",
        "***" * 30 + " mixed delimiters ___" * 20,
    ]
    atoms = [
        "word", " ", "  ", "\n", "_field", "*member", "**bold**",
        "_italic_", "*italic*", "`_code`", "~~gone~~", "||secret||",
        "🙂", "한글", "\\_escaped", "\\*marker", "__under__", "**",
        "`", "```cpp\n", "\n```\n", "\n~~~\n", "# heading ", "C:\\path_name",
    ]
    generator = random.Random(417203)
    for _ in range(300):
        fixtures.append("".join(generator.choice(atoms) for _ in range(generator.randint(10, 100))))
    return fixtures


def _fixture_digest():
    rendered = [_signal_rich_text_chunks(text) for text in _markdown_fixtures()]
    encoded = json.dumps(rendered, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SignalInlineLatencyTests(unittest.TestCase):
    def test_valid_and_malformed_markdown_match_baseline_golden_chunks_and_styles(self):
        self.assertEqual(_fixture_digest(), "0fd9a932c993a33e808e11b3d19f189e28d8c4a4cbde43b01a1c5e032ef0fa0c")

    def test_unmatched_identifiers_and_unicode_keep_every_character(self):
        for marker in ("_", "*"):
            text = (" " + marker + "field🙂한글") * 6000
            chunks = _signal_rich_text_chunks(text)
            self.assertEqual("".join(chunk for chunk, _styles in chunks), text)
            self.assertTrue(all(not styles for _chunk, styles in chunks))

    def test_a_later_valid_closer_still_closes_the_first_opener(self):
        text = " _first _second _third closing_ suffix"
        chunks = _signal_rich_text_chunks(text)
        self.assertEqual(chunks, [(" first _second _third closing suffix", ["1:28:ITALIC"])])

    def test_many_unmatched_openers_finish_in_a_killable_process(self):
        # A future quadratic regression must not wedge the test runner itself.
        # Ten seconds includes startup/imports and is deliberately generous;
        # this checks bounded work, not a brittle subsecond timing threshold.
        script = """
from Cozter.backends_bot.signal import _signal_rich_text_chunks
for marker in ('_', '*'):
    text = (' ' + marker + 'field') * 30000
    chunks = _signal_rich_text_chunks(text)
    assert ''.join(chunk for chunk, _ in chunks) == text
    assert not any(styles for _, styles in chunks)
print('complete')
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
            env=dict(os.environ), timeout=10, check=True,
        )
        self.assertEqual(result.stdout.strip(), "complete")
