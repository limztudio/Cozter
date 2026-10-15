"""Formatting marker collisions preserve text without quadratic searches."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

from Cozter.backends_bot.formatting import render_inline_code, unused_format_marker
from Cozter.backends_bot.slack import _mrkdwn_prose


class FormatMarkerLatencyTests(unittest.TestCase):
    def test_inline_code_preserves_user_marker_and_literal_code(self):
        prefix = "\x00CODE\x00" + "\x00" * 7
        source = prefix + " user `a*b` tail"
        result = render_inline_code(
            source, render_prose=lambda value: value.replace("user", "USER"),
            render_code=lambda value: "<code>" + value + "</code>",
        )
        self.assertEqual(result, prefix + " USER <code>a*b</code> tail")

    def test_overlapping_markers_consider_the_longest_following_nul_run(self):
        marker = "\x00CODE\x00"
        source = "\x00CODE\x00CODE\x00" + "\x00" * 9
        self.assertEqual(unused_format_marker(source, marker), marker + "\x00" * 10)
        self.assertEqual(unused_format_marker("ordinary prose", marker), marker)
        self.assertEqual(unused_format_marker(marker, marker), marker + "\x00")

    def test_slack_preserves_both_private_marker_collisions(self):
        prefix = "\ue001" + "\x00" * 7 + "\ue002" + "\x00" * 11
        self.assertEqual(_mrkdwn_prose(prefix + " **bold** and *italic*"),
                         prefix + " *bold* and _italic_")

    def test_large_collision_runs_finish_in_a_killable_process(self):
        # The earlier loop extended and searched each prefix separately.
        # At this size both renderers together took many seconds; a child
        # timeout makes a regression bounded even on slower CI hosts.
        code = textwrap.dedent("""
            from Cozter.backends_bot.formatting import render_inline_code
            from Cozter.backends_bot.slack import _mrkdwn_prose
            zeros = '\\x00' * 150000
            prefix = '\\x00CODE\\x00' + zeros
            assert render_inline_code(prefix + ' `code`',
                render_prose=lambda value: value,
                render_code=lambda value: '<code>' + value + '</code>') == prefix + ' <code>code</code>'
            prefix = '\\ue001' + zeros + '\\ue002' + zeros
            assert _mrkdwn_prose(prefix + ' **bold**') == prefix + ' *bold*'
        """)
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[2], timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
