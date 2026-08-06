import asyncio
import unittest
from types import SimpleNamespace

from Cozter.backends_bot.base import MessageHandle, attachment_kind_from_mime
from Cozter.backends_bot.formatting import strip_html_markup
from Cozter.backends_bot.slack import _md_to_mrkdwn
from Cozter.backends_bot.telegram import (
    TelegramBot, _TELEGRAM_TEXT_LIMIT, _md_to_html,
)


class PlatformFormattingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
