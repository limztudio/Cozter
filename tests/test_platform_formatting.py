import asyncio
import re
import unittest
from types import SimpleNamespace

from Cozter.backends_bot.base import MessageHandle, attachment_kind_from_mime
from Cozter.backends_bot.formatting import strip_html_markup
from Cozter.backends_bot.slack import _md_to_mrkdwn
from Cozter.backends_bot.telegram import (
    TelegramBot, _TELEGRAM_TEXT_LIMIT, _md_to_html, _rich_telegram_chunks,
)


class PlatformFormattingTests(unittest.TestCase):
    def test_inline_code_preserves_markdown_characters_and_escapes_html(self):
        source = "**Run** `**literal** _arg_ ~~value~~ <x & y>`"
        self.assertEqual(
            _md_to_html(source),
            "<b>Run</b> <code>**literal** _arg_ ~~value~~ &lt;x &amp; y&gt;</code>",
        )
        self.assertEqual(
            _md_to_mrkdwn(source),
            "*Run* `**literal** _arg_ ~~value~~ &lt;x &amp; y&gt;`",
        )

    def test_inline_code_placeholders_cannot_replace_source_text(self):
        source = "\x00CODE\x000\x00CODE\x00 `*literal*`"
        self.assertEqual(_md_to_html(source), "\x00CODE\x000\x00CODE\x00 <code>*literal*</code>")
        self.assertEqual(_md_to_mrkdwn(source), source)

    def test_attachment_kind_from_mime_handles_all_transports(self) -> None:
        self.assertEqual(attachment_kind_from_mime("IMAGE/jpeg"), "photo")
        self.assertEqual(attachment_kind_from_mime("audio/ogg"), "audio")
        self.assertEqual(attachment_kind_from_mime("video/mp4"), "video")
        self.assertEqual(attachment_kind_from_mime(None), "document")

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

    def test_fenced_code_keeps_shorter_and_nonclosing_backtick_lines(self):
        source = "````python\n```literal\n**plain**\n````"
        self.assertEqual(
            _md_to_html(source), "<pre>```literal\n**plain**</pre>",
        )
        self.assertEqual(
            _md_to_mrkdwn(source), "```\n```literal\n**plain**\n```",
        )

    def test_tilde_fenced_code_is_literal(self):
        source = "~~~python\n**plain** <x>\n~~~"
        self.assertEqual(_md_to_html(source), "<pre>**plain** &lt;x&gt;</pre>")


class TelegramSendTextTests(unittest.TestCase):
    def test_plain_text_is_split_at_telegram_limit_without_losing_text(self) -> None:
        class CapturingApi:
            def __init__(self) -> None:
                self.messages: list[dict] = []

            async def send_message(self, **kwargs):
                self.messages.append(kwargs)
                return SimpleNamespace(message_id=len(self.messages))

        async def run() -> tuple[CapturingApi, object]:
            api = CapturingApi()
            bot = TelegramBot("token", ["1"])
            bot.app = SimpleNamespace(bot=api)
            text = "a" * (_TELEGRAM_TEXT_LIMIT - 1) + "\n" + "b"
            handle = await bot.send_text("42", text)
            return api, handle

        api, handle = asyncio.run(run())

        self.assertEqual(
            "".join(message["text"] for message in api.messages),
            "a" * (_TELEGRAM_TEXT_LIMIT - 1) + "\n" + "b",
        )
        self.assertTrue(all(
            len(message["text"]) <= _TELEGRAM_TEXT_LIMIT
            for message in api.messages
        ))
        self.assertTrue(all("parse_mode" not in message for message in api.messages))
        self.assertEqual(handle, MessageHandle("42", "2"))

    def test_rich_text_is_split_as_markdown_before_html_conversion(self) -> None:
        class CapturingApi:
            def __init__(self) -> None:
                self.messages: list[dict] = []

            async def send_message(self, **kwargs):
                self.messages.append(kwargs)
                return SimpleNamespace(message_id=len(self.messages))

        async def run() -> CapturingApi:
            api = CapturingApi()
            bot = TelegramBot("token", ["1"])
            bot.app = SimpleNamespace(bot=api)
            text = "a" * (_TELEGRAM_TEXT_LIMIT - 1) + "\n" + "b"
            await bot.send_text("42", text, rich=True)
            return api

        api = asyncio.run(run())
        self.assertGreaterEqual(len(api.messages), 2)
        self.assertTrue(all(
            len(message["text"]) <= _TELEGRAM_TEXT_LIMIT
            for message in api.messages
        ))
        self.assertTrue(all(
            message.get("parse_mode") == "HTML" for message in api.messages
        ))
        self.assertEqual(
            "".join(message["text"] for message in api.messages),
            "a" * (_TELEGRAM_TEXT_LIMIT - 1) + "\n" + "b",
        )

    def test_rich_chunks_do_not_split_html_tags(self) -> None:
        html = _md_to_html("**bold** and `code`")
        self.assertEqual(_rich_telegram_chunks("**bold** and `code`"), [html])
        self.assertIn("<b>bold</b>", html)
        self.assertIn("<code>code</code>", html)

    def test_expanding_entities_and_long_code_keep_text_and_balanced_tags(self):
        for source in (
            "&" * 1_000,
            "**" + "<x & y>" * 1_000 + "**",
            "```python\n" + "a < b && **literal**\n" * 1_000 + "```",
            "**_" + "x & y " * 1_000 + "** crossed_",
        ):
            with self.subTest(source=source[:20]):
                chunks = _rich_telegram_chunks(source)
                self.assertTrue(all(len(chunk) <= _TELEGRAM_TEXT_LIMIT for chunk in chunks))
                self.assertEqual(
                    "".join(strip_html_markup(chunk) for chunk in chunks),
                    strip_html_markup(_md_to_html(source)),
                )
                for chunk in chunks:
                    stack = []
                    for closing, tag in re.findall(r"<(/?)(b|i|s|code|pre)>", chunk):
                        if closing:
                            self.assertTrue(stack)
                            self.assertEqual(stack.pop(), tag)
                        else:
                            stack.append(tag)
                    self.assertEqual(stack, [])


if __name__ == "__main__":
    unittest.main()
