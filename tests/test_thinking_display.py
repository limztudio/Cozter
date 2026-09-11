"""Tests for the live 'Thinking...' preview composition.

During a turn the bot streams recent tool activity plus the latest answer
text into an editable status message. _compose_thinking_display builds that
string; it must keep the newest tool lines and the tail of long text.
"""

import unittest

from Cozter.backends_bot.base import BotPlatform


class ThinkingDisplayTests(unittest.TestCase):
    def test_includes_recent_tools_and_text(self) -> None:
        out = BotPlatform._compose_thinking_display(
            ["» ls", "» cat x"], "Here is the answer",
        )
        self.assertTrue(out.startswith("Thinking..."))
        self.assertIn("» cat x", out)
        self.assertIn("Here is the answer", out)

    def test_keeps_only_last_five_tool_lines(self) -> None:
        lines = [f"» step {i}" for i in range(8)]
        out = BotPlatform._compose_thinking_display(lines, "")
        self.assertIn("» step 7", out)
        self.assertNotIn("» step 0", out)  # oldest dropped

    def test_truncates_long_text_keeping_tail(self) -> None:
        long = "HEAD" + "b" * 1000 + "TAIL"
        out = BotPlatform._compose_thinking_display([], long)
        self.assertIn("TAIL", out)
        self.assertNotIn("HEAD", out)  # head dropped, tail kept
        self.assertIn("… [preview]", out)  # visible cut marker, not bare "…"
        self.assertLessEqual(len(out), 720)

    def test_ephemeral_label_clip_has_visible_marker(self) -> None:
        label = "x" * 60
        clipped = label[:40] + "… [clipped]"
        self.assertIn("[clipped]", clipped)

    def test_short_id_clip_has_visible_marker(self) -> None:
        from Cozter.backends_bot.signal import _short_id
        self.assertIn("[clipped]", _short_id("x" * 40))
        self.assertEqual(_short_id("short"), "short")

    def test_empty_is_just_thinking(self) -> None:
        self.assertEqual(BotPlatform._compose_thinking_display([], ""), "Thinking...")
        self.assertEqual(
            BotPlatform._compose_thinking_display([], "   "), "Thinking...",
        )


if __name__ == "__main__":
    unittest.main()
