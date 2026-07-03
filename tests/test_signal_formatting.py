import asyncio
import unittest
from typing import Any

from Cozter.backends_bot.base import MessageHandle
from Cozter.backends_bot.signal import (
    SignalBot,
    _md_to_signal_body_and_spans,
    _signal_rich_text_chunks,
)


class SignalFormattingTests(unittest.TestCase):
    def test_markdown_becomes_signal_body_and_styles(self) -> None:
        body, spans = _md_to_signal_body_and_spans(
            "# Title\nA **bold** _it_ ~~gone~~ `code` ||secret||"
        )

        self.assertEqual(body, "Title\nA bold it gone code secret")
        self.assertEqual(
            spans,
            [
                (0, 5, "BOLD"),
                (8, 4, "BOLD"),
                (13, 2, "ITALIC"),
                (16, 4, "STRIKETHROUGH"),
                (21, 4, "MONOSPACE"),
                (26, 6, "SPOILER"),
            ],
        )

    def test_rich_chunks_use_utf16_offsets(self) -> None:
        chunks = _signal_rich_text_chunks("😀 **ok**")

        self.assertEqual(chunks, [("😀 ok", ["3:2:BOLD"])])

    def test_code_block_is_monospace_without_fences(self) -> None:
        chunks = _signal_rich_text_chunks("a\n```\nx*y\n```\nb")

        self.assertEqual(chunks, [("a\nx*y\nb", ["2:3:MONOSPACE"])])

    def test_styles_are_clipped_across_chunks(self) -> None:
        chunks = _signal_rich_text_chunks("**abcdef**", limit=3)

        self.assertEqual(
            chunks,
            [
                ("abc", ["0:3:BOLD"]),
                ("def", ["0:3:BOLD"]),
            ],
        )


class _CapturingSignalBot(SignalBot):
    def __init__(self) -> None:
        super().__init__(["https://signal.group/#test"], jsonrpc_socket="/tmp/s")
        self._group_ids = {"group"}
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def _rpc_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: int | float | None = 60,
    ) -> Any:
        self.calls.append((method, params))
        return {"timestamp": "123"}


class SignalSendFormattingTests(unittest.TestCase):
    def test_rich_send_uses_signal_text_styles_shape(self) -> None:
        async def run() -> _CapturingSignalBot:
            bot = _CapturingSignalBot()
            await bot.send_text(
                "group", "A **bold** and `code`", rich=True,
            )
            return bot

        bot = asyncio.run(run())

        self.assertEqual(len(bot.calls), 1)
        self.assertEqual(
            bot.calls[0],
            (
                "send",
                {
                    "groupId": "group",
                    "message": "A bold and code",
                    "textStyle": ["2:4:BOLD", "11:4:MONOSPACE"],
                },
            ),
        )

    def test_single_rich_span_uses_singular_param(self) -> None:
        async def run() -> _CapturingSignalBot:
            bot = _CapturingSignalBot()
            await bot.send_text("group", "A **bold**", rich=True)
            return bot

        bot = asyncio.run(run())

        self.assertEqual(
            bot.calls[0][1],
            {
                "groupId": "group",
                "message": "A bold",
                "textStyle": "2:4:BOLD",
            },
        )

    def test_rich_edit_uses_signal_text_style_shape(self) -> None:
        async def run() -> _CapturingSignalBot:
            bot = _CapturingSignalBot()
            await bot.edit_text(
                MessageHandle(chat_id="group", message_id="42"),
                "Thinking...\n\n**partial** `reply`",
                rich=True,
            )
            return bot

        bot = asyncio.run(run())

        self.assertEqual(
            bot.calls[0][1],
            {
                "groupId": "group",
                "editTimestamp": 42,
                "message": "Thinking...\n\npartial reply",
                "textStyle": ["13:7:BOLD", "21:5:MONOSPACE"],
            },
        )


if __name__ == "__main__":
    unittest.main()
