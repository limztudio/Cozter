"""Instruction identities preserve quote semantics without regex retry stalls."""

import itertools
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import unittest

from Cozter import flexible


_LEGACY_QUOTED_SPANS = re.compile(r'''(`[^`]*`|'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")''')


def legacy_identity(instruction):
    pieces = _LEGACY_QUOTED_SPANS.split(instruction)
    normalized = "".join(piece if index % 2 else re.sub(r"\s+", " ", piece)
                         for index, piece in enumerate(pieces)).strip()
    return re.sub(r"^Please\s+", "", normalized)


class FlexibleInstructionIdentityTests(unittest.TestCase):
    def test_literal_case_and_quoted_spacing_keep_existing_task_identities(self):
        instruction = '  Please  Repair\n target `a  b` and "x  y" then \'C:\\dir\\file\'.  '
        expected = 'Repair target `a  b` and "x  y" then \'C:\\dir\\file\'.'
        self.assertEqual(flexible.instruction_identity(instruction), expected)
        self.assertEqual(flexible.task_identity(flexible.Subtask("high", instruction)),
                         "high:" + expected)
        self.assertNotEqual(flexible.instruction_identity(instruction.lower()), expected)
        self.assertNotEqual(flexible.instruction_identity(instruction.replace("a  b", "a b")), expected)

    def test_exhaustive_short_inputs_match_the_original_split_regex(self):
        alphabet = "'\"`\\ a\n"
        for length in range(6):
            for pieces in itertools.product(alphabet, repeat=length):
                text = "".join(pieces)
                self.assertEqual(flexible.instruction_identity(text), legacy_identity(text), repr(text))

    def test_seeded_longer_mixed_quotes_and_unicode_whitespace_match_legacy(self):
        rng = random.Random(92701)
        alphabet = "'\"`\\ abc\n\r\t\u2003\u2028"
        for index in range(2_000):
            text = "".join(rng.choices(alphabet, k=rng.randrange(150)))
            if index % 8 == 0:
                text = "Please " + text
            self.assertEqual(flexible.instruction_identity(text), legacy_identity(text), repr(text))

    def test_failed_escape_boundaries_do_not_hide_later_valid_quotes(self):
        for quote in ("'", '"'):
            for delimiter in ("\n", "\r\n", "\r", "\u2028", ""):
                text = quote + "before\\" + quote + "\\" + delimiter + " middle " + quote + "x  y" + quote
                self.assertEqual(flexible.instruction_identity(text), legacy_identity(text))
                embedded = quote + "\\" + quote + "`x  y`\\\n" + quote + "z  w" + quote
                self.assertEqual(flexible.instruction_identity(embedded), legacy_identity(embedded))

    def test_large_unclosed_escaped_runs_finish_in_a_bounded_child_process(self):
        # The old splitter takes tens of seconds per 100K-character case.
        # Isolate the deadline so a regression cannot freeze the test runner.
        code = """import json
from Cozter.flexible import instruction_identity
results = []
for quote in (chr(39), chr(34)):
    text = (quote + chr(92)) * 50000
    results.append(instruction_identity(text) == text)
print(json.dumps(results))
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
            check=True, capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(json.loads(result.stdout), [True, True])
