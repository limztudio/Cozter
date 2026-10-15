"""Control marker parsing retains exact legacy ranges without suffix rescans."""

import os
import random
import re
import subprocess
import sys
import unittest
from unittest import mock

from Cozter import agent


_LEGACY_ATTACH = re.compile(r"\[\[attach:\s*([^\]\n]+?)\s*\]\]", re.IGNORECASE)
_LEGACY_BACKGROUND = re.compile(
    r"(?ims)^[ \t]*\[\[background:\s*(?P<prompt>.*?)\s*\]\][ \t]*(?:\r?\n|$)",
)


def _fixtures():
    yield from (
        "[[attach:]]", "[[attach: ]]", "[[attach:\n]]", "[[attach:\n \n]]",
        "[[attach:\r\n]]", "[[attach: x \n\t]]", "[[attach: x\ny]]",
        "[[ATTACH:\t🙂한글.txt\r\n]]", "[[attach:outer [[attach:inner]] tail]]",
        "[[attach: missing ] text [[attach: valid]]", "prefix [[background: inline]]\n",
        " \t[[background:\n first\nsecond\t\n]] \t\r\nnext",
        "[[background: first]]not a line end\n[[background: second]]\n",
        "[[background:]]\n", "[[background:\n]]", "[[background: nested\n[[background: inner]]\n",
        "[[background: first]]\rtrailing", "[[bacKground: Kelvin letter]]\n",
        "[[background:overlapping]]]]\n", " [[background: whitespace\u2003]]\n",
    )
    generator = random.Random(8137)
    atoms = [
        "[[attach:", "[[ATTACH:", "[[background:", "[[bACKground:", "]", "]]",
        "x", " ", "\t", "\r", "\n", "\r\n", "\v", "\x1c", "\u2003", "🙂",
        "prefix", "suffix", "[",
    ]
    for _ in range(10000):
        yield "".join(generator.choice(atoms) for _ in range(generator.randrange(1, 35)))


class ProtocolMarkerLatencyTests(unittest.TestCase):
    def test_exact_ranges_and_captures_match_legacy_regexes(self):
        for text in _fixtures():
            for legacy, parse in (
                (_LEGACY_ATTACH, agent._iter_attachment_markers),
                (_LEGACY_BACKGROUND, agent._iter_detached_task_markers),
            ):
                expected = [(match.start(), match.end(), match.group(1)) for match in legacy.finditer(text)]
                self.assertEqual(list(parse(text)), expected, repr(text))

    def test_attachment_resolution_callbacks_and_cleaned_text_are_unchanged(self):
        text = "before [[attach: one.txt]] [[attach:missing]] [[attach: one.txt]]\n[[attach: ]]\n[[attach: two.txt]] after"

        def resolve(path, _workspace):
            return None if path == "missing" else ("/resolved/" + path, False)

        expected_paths = []
        expected_calls = []

        def remove(match):
            path = match.group(1).strip()
            if path:
                expected_calls.append(path)
                source = resolve(path, "workspace")
                if source is not None and source[0] not in expected_paths:
                    expected_paths.append(source[0])
            return ""

        expected = agent._collapse_extra_blank_lines(_LEGACY_ATTACH.sub(remove, text)), expected_paths
        with mock.patch.object(agent, "_resolve_attachment_source", side_effect=resolve) as resolver:
            actual = agent.extract_attachment_sources(text, "workspace")
        self.assertEqual(actual, expected)
        self.assertEqual([call.args[0] for call in resolver.call_args_list], expected_calls)

    def test_split_attach_keeps_original_marker_text_and_order(self):
        for text in _fixtures():
            markers = [match.group(0) for match in _LEGACY_ATTACH.finditer(text)]
            expected = (
                agent._collapse_extra_blank_lines(_LEGACY_ATTACH.sub("", text)), markers,
            ) if markers else (text, [])
            self.assertEqual(agent._split_attach_markers(text), expected, repr(text))

    def test_background_actions_keep_dedup_limit_empty_and_anchoring_rules(self):
        text = (
            "inline [[background: not an action]]\n"
            "[[background:first]]\n[[background:first]]\n[[background:  ]]\n"
            "[[BACKGROUND:second\ncontinued]]\r\n"
            "  [[background:third]]\n[[background:fourth]]\n"
            "[[background:unfinished"
        )
        expected_prompts = []

        def remove(match):
            prompt = match.group("prompt").strip()
            if prompt and len(prompt) <= agent._MAX_DETACHED_TASK_PROMPT_CHARS:
                if prompt not in expected_prompts and len(expected_prompts) < agent._MAX_DETACHED_TASK_REQUESTS:
                    expected_prompts.append(prompt)
            return ""

        expected = agent._collapse_extra_blank_lines(_LEGACY_BACKGROUND.sub(remove, text)), expected_prompts
        with mock.patch.object(agent.logger, "warning"):
            self.assertEqual(agent.extract_detached_task_requests(text), expected)

    def test_incomplete_and_whitespace_heavy_markers_finish_in_killable_process(self):
        # Keep a quadratic regression killable even if it holds the GIL. The
        # generous timeout includes Python/import startup on a slow host.
        script = """
from Cozter import agent
cases = [
    ('[[attach: missing ' * 10000, agent._iter_attachment_markers, False),
    ('[[background: missing\\n' * 10000, agent._iter_detached_task_markers, False),
    ('[[attach: missing ' * 10000 + '\\ninvalid continuation]]', agent._iter_attachment_markers, False),
    ('[[attach:x' + ' ' * 100000 + 'y]]', agent._iter_attachment_markers, True),
    ('[[background:x' + ' ' * 100000 + 'y]]\\n', agent._iter_detached_task_markers, True),
]
for text, parse, expected in cases:
    matches = list(parse(text))
    assert bool(matches) == expected
    if expected:
        assert matches[0][0] == 0 and matches[0][1] == len(text)
print('complete')
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
            env=dict(os.environ), timeout=10, check=True,
        )
        self.assertEqual(result.stdout.strip(), "complete")
