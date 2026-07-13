"""Slack native-Markdown block rendering tests."""

import unittest
from types import SimpleNamespace

from Cozter.backends_bot.base import MessageHandle
from Cozter.backends_bot.slack import (
    SlackBot,
    _SLACK_MARKDOWN_LIMIT,
    _split_slack_markdown,
)
from slack_sdk.errors import SlackApiError


class _Client:
    def __init__(self) -> None:
        self.posts: list[dict] = []
        self.updates: list[dict] = []

    async def chat_postMessage(self, **kwargs):
        self.posts.append(kwargs)
        return {"ts": f"{len(self.posts)}.0"}

    async def chat_update(self, **kwargs):
        self.updates.append(kwargs)


class _MarkdownRejectingClient(_Client):
    async def chat_postMessage(self, **kwargs):
        if "blocks" in kwargs:
            raise SlackApiError(
                "Markdown blocks are unavailable",
                {"ok": False, "error": "invalid_blocks"},
            )
        return await super().chat_postMessage(**kwargs)


class SlackFormattingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = _Client()
        self.bot = SlackBot("bot-token", "app-token", ["C1"])
        self.bot.app = SimpleNamespace(client=self.client)

    async def test_rich_reply_uses_native_markdown_block(self) -> None:
        markdown = (
            "# Plan\n\n"
            "- [x] Done\n"
            "- [ ] Next\n\n"
            "| Item | Status |\n| --- | --- |\n| Slack | Ready |\n\n"
            "```python\nprint('ready')\n```"
        )

        handle = await self.bot.send_text("C1", markdown, rich=True)

        self.assertEqual(handle, MessageHandle("C1", "1.0"))
        self.assertEqual(self.client.posts, [{
            "channel": "C1",
            "text": markdown,
            "blocks": [{"type": "markdown", "text": markdown}],
        }])

    async def test_plain_reply_keeps_existing_text_payload(self) -> None:
        await self.bot.send_text("C1", "# Plain", rich=False)

        self.assertEqual(self.client.posts, [{
            "channel": "C1", "text": "# Plain",
        }])

    async def test_rich_reply_falls_back_when_markdown_blocks_are_rejected(self):
        self.client = _MarkdownRejectingClient()
        self.bot.app = SimpleNamespace(client=self.client)

        await self.bot.send_text("C1", "# Title", rich=True)

        self.assertEqual(self.client.posts, [{
            "channel": "C1", "text": "*Title*",
        }])

    async def test_rich_edit_uses_native_markdown_block(self) -> None:
        markdown = "## Working\n\n- [ ] Running"

        await self.bot.edit_text(
            MessageHandle("C1", "10.0"), markdown, rich=True,
        )

        self.assertEqual(self.client.updates, [{
            "channel": "C1",
            "ts": "10.0",
            "text": markdown,
            "blocks": [{"type": "markdown", "text": markdown}],
        }])

    def test_long_markdown_is_split_with_fences_balanced(self) -> None:
        source = "```python\n" + ("print('x')\n" * 1_500) + "```\n"

        chunks = _split_slack_markdown(source, limit=1_000)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 1_000 for chunk in chunks))
        self.assertTrue(all(chunk.count("```") % 2 == 0 for chunk in chunks))
        self.assertTrue(chunks[0].startswith("```python\n"))
        self.assertTrue(chunks[1].startswith("```python\n"))
        self.assertLessEqual(max(map(len, chunks)), _SLACK_MARKDOWN_LIMIT)


if __name__ == "__main__":
    unittest.main()
