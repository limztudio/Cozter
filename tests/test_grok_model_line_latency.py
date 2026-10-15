"""Grok's human model catalog retains captures without whitespace retry stalls."""

import itertools
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import unittest
from unittest import mock

from Cozter.backends_agent import grok


_LEGACY_MODEL_LINE = re.compile(r"^\s*[*-]\s+(\S+)\s*(?:\([^)]*\))?\s*$")


def _legacy_id(line):
    match = _LEGACY_MODEL_LINE.match(line)
    return match.group(1) if match is not None else None


class GrokModelLineLatencyTests(unittest.TestCase):
    def test_model_ids_description_whitespace_and_final_newline_capture_exactly(self):
        cases = [
            "- model", " * model (description)", "- model(attached)",
            "- model(attached) (description)", "- model (nested (description))",
            "- model (description) trailing", "- model (description", "- model )",
            "\t* model\u2003(description)\u2028", "- model\r\n", "- model\n",
            "- model\n\n", "- model\v\x1c", "\n * model", " - (model)",
            "- `quoted-model`,", "- model\u2003\u2028(Unicode🙂 description)",
            "- model\u2003(unfinished", "- model   trailing", "*", "- ", "",
            "- (((((( x", "- (((((( x)", "- model(desc with spaces)",
            "- (model with spaces)", "- ((model with spaces)",
        ]
        for line in cases:
            with self.subTest(line=line):
                self.assertEqual(grok._model_line_id(line), _legacy_id(line))

    def test_exhaustive_short_and_seeded_mixed_lines_match_legacy_captures(self):
        alphabet = "-* a()\t"
        for length in range(6):
            for pieces in itertools.product(alphabet, repeat=length):
                line = "".join(pieces)
                self.assertEqual(grok._model_line_id(line), _legacy_id(line), repr(line))
        generator = random.Random(802741)
        pieces = ["-", "*", " ", "  ", "\t", "\r", "\n", "\u2003", "\u2028",
                  "model", "(description)", "(", ")", "(attached)", "🙂", "`", ","]
        for _ in range(10_000):
            line = "".join(generator.choices(pieces, k=generator.randrange(1, 35)))
            self.assertEqual(grok._model_line_id(line), _legacy_id(line), repr(line))

    def test_final_catalog_heading_order_dedup_and_bytes_match_legacy(self):
        catalogs = [
            "Banner\nAvailable models:\n  - model (description)\n * model\n - other(attached)\nNext heading:\n - hidden",
            "AVAILABLE MODELS:\n - `model`,\n  malformed diagnostic\n - other (description)\n",
            "No catalog heading\n - hidden\n",
            "Available models:\n - first\n\n - after-blank\n",
            b"Available models:\n - utf8-\xff\n - other\n",
        ]
        generator = random.Random(9824)
        lines = [" - model", " * other (description)", " - model", " - `quoted`,",
                 " - model(attached)", " - model (unclosed", "  malformed", "",
                 "Next heading:", " * \u2003model\t(description)"]
        for _ in range(500):
            catalogs.append("Available models:\n" + "\n".join(generator.choices(lines, k=20)))
        for output in catalogs:
            with mock.patch.object(grok, "_model_line_id", _legacy_id):
                expected = grok._parse_models_output(output)
            self.assertEqual(grok._parse_models_output(output), expected)

    def test_large_malformed_padding_keeps_later_models_in_a_killable_child(self):
        script = """import json
from Cozter.backends_agent.grok import _parse_models_output
results = []
for suffix in ('(', '(missing description', 'trailing junk'):
    malformed = '  - grok-model' + ' ' * 100000 + suffix
    output = 'Available models:\\n' + malformed + '\\n - next-model (valid)\\n'
    results.append(_parse_models_output(output) == ('next-model',))
valid = 'Available models:\\n  - grok-model' + ' ' * 100000 + '(valid)\\n'
results.append(_parse_models_output(valid) == ('grok-model',))
for suffix in (' x', ' x) extra', ' (x) extra)'):
    malformed = '  - ' + '(' * 100000 + suffix
    output = 'Available models:\\n' + malformed + '\\n - next-model (valid)\\n'
    results.append(_parse_models_output(output) == ('next-model',))
print(json.dumps(results))
"""
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=True, timeout=5,
        )
        self.assertEqual(json.loads(result.stdout), [True] * 7)
