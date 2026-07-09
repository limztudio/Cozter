import unittest

from Cozter.backends_bot.formatting import strip_html_markup
from Cozter.backends_bot.slack import _md_to_mrkdwn
from Cozter.backends_bot.telegram import _md_to_html


class PlatformFormattingTests(unittest.TestCase):
    def test_telegram_markdown_to_html_handles_inline_and_code_blocks(
        self,
    ) -> None:
        out = _md_to_html(
            "# Title\nA **bold** _it_ ~~gone~~ `code`\n```\n<x>\n```"
        )

        self.assertEqual(
            out,
            "<b>Title</b>\n"
            "A <b>bold</b> <i>it</i> <s>gone</s> <code>code</code>\n"
            "<pre>&lt;x&gt;</pre>",
        )

    def test_slack_markdown_to_mrkdwn_handles_inline_and_code_blocks(
        self,
    ) -> None:
        out = _md_to_mrkdwn(
            "# Title\nA **bold** *it* ~~gone~~ `code`\n```\n<x>\n```"
        )

        self.assertEqual(
            out,
            "*Title*\n"
            "A *bold* _it_ ~gone~ `code`\n"
            "```\n"
            "&lt;x&gt;\n"
            "```",
        )

    def test_strip_html_markup_removes_tags_and_unescapes_entities(
        self,
    ) -> None:
        out = strip_html_markup(
            "<b>Title</b>\n<pre>&lt;x &amp; y&gt;</pre>"
        )

        self.assertEqual(out, "Title\n<x & y>")


if __name__ == "__main__":
    unittest.main()
