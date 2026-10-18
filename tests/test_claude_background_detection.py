"""Claude Bash background bookkeeping must not rescan overlapping suffixes."""

import random
import re
import unittest

from Cozter.backends_agent import claude_code as cc
from Cozter.backends_agent.base import AgentResult

# The old regex, kept here as the expected answer.
_LEGACY_BACKGROUND_BASH_RE = re.compile(
    r"(?:^|[\s;&|])claude\b[^\n]*--(?:bg|background)\b", re.IGNORECASE,
)


class ClaudeBackgroundDetectionTests(unittest.TestCase):
    def test_word_boundaries_case_and_newlines_match_legacy(self):
        cases = [
            "claude --bg task", "CLAUDE --BACKGROUND task", " claude --bG",
            "echo x;claude --bg", "x|claude --background", "x&claude --bg",
            "xclaude --bg", "x/claude --bg", "claudex --bg", "claude_ --bg",
            "claude\u00e9 --bg", "claude --backgrounded", "claude --bg_",
            "claude --bg\u00e9", "claude \n--bg", "claude\r\n--background",
            "x\nclaude --bg", "\n\nclaude --bg", "x\nclaude\n--bg",
            "claude\u2028--bg", "\u2003claude --bg", "--bg claude",
            "--bg claude claude --background", "claude\n--bg claude",
            "claude\n--bg claude --background", "claude --bg\n",
            "claude--bg", "claude/--bg", "claude claude --background",
            "\nclaude --bg\nclaude", "\n", "", "x\rclaude --bg",
        ]
        for command in cases:
            with self.subTest(command=command):
                self.assertEqual(cc._is_background_bash(command), bool(_LEGACY_BACKGROUND_BASH_RE.search(command)))

    def test_seeded_malformed_commands_match_legacy(self):
        randomizer = random.Random(217)
        pieces = [
            "claude", "CLAUDE", "claudex", "xclaude", "--bg", "--background",
            "--BACKGROUND", "--bG", "--bg_", "--backgrounded", "x", "\u00e9",
            " ", "\n", "\r", "\t", ";", "&", "|", "/", "_", "\u2003", "\u2028",
        ]
        for _ in range(15_000):
            command = "".join(randomizer.choices(pieces, k=randomizer.randrange(1, 35)))
            self.assertEqual(cc._is_background_bash(command), bool(_LEGACY_BACKGROUND_BASH_RE.search(command)), command)

    def test_large_repeated_launchers_preserve_flag_order(self):
        repeated = " claude" * 100_000
        self.assertFalse(cc._is_background_bash(repeated))
        self.assertFalse(cc._is_background_bash("--bg" + repeated))
        self.assertFalse(cc._is_background_bash(repeated + "\n--bg"))
        self.assertTrue(cc._is_background_bash(repeated + " --bg"))
        self.assertTrue(cc._is_background_bash(repeated + "\nclaude --background"))

    def test_tool_callback_keeps_background_ownership_and_status(self):
        for command, expected in (("echo x", set()), ("claude --bg task", {"use-1"})):
            with self.subTest(command=command):
                result = AgentResult()
                cc.ClaudeCodeBackend._emit_tool_event({
                    "type": "tool_use", "id": "use-1", "name": "Bash",
                    "input": {"command": command},
                }, result)
                self.assertEqual(result.detached_task_tool_use_ids, expected)
                self.assertEqual(len(result.events), 1)
                self.assertEqual(result.events[0].kind, "tool")
                self.assertEqual(result.events[0].content, "$ " + command)


if __name__ == "__main__":
    unittest.main()
