"""Tests for Slack plain-message command aliases."""

import asyncio
import unittest

from Cozter.backends_bot.base import MessageHandle
from Cozter.backends_bot.slack import SlackBot


class _CapturingSlackBot(SlackBot):
    def __init__(self) -> None:
        super().__init__("bot-token", "app-token", ["C1"])
        self._bot_user_id = "B1"
        self.replies: list[str] = []

    async def send_text(
        self, chat_id: str, text: str, *, rich: bool = False,
    ) -> MessageHandle | None:
        self.replies.append(text)
        return None


class SlackMessageCommandTests(unittest.TestCase):
    def test_backslash_command_message_dispatches_alias(self) -> None:
        bot = _CapturingSlackBot()

        asyncio.run(bot._on_message({
            "user": "U1",
            "channel": "C1",
            "text": r"\start",
        }))

        self.assertEqual(bot.replies, ["Cozter bot is running."])


if __name__ == "__main__":
    unittest.main()
