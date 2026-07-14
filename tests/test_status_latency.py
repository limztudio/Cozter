"""Regression tests for noncritical chat status I/O.

The live "Thinking..." preview is useful, but a delayed platform API call
must never block the agent stream or the final answer.
"""

import asyncio
import tempfile
import unittest
from unittest import mock

from Cozter import agent, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.backends_bot.base import BotPlatform, MessageHandle


class _StatusBot(BotPlatform):
    def __init__(self, workspace_path: str) -> None:
        super().__init__([])
        self.workspace_path = workspace_path
        self.sent: list[str] = []
        self.edit_started = asyncio.Event()
        self.edit_release = asyncio.Event()
        self.delete_started = asyncio.Event()
        self.delete_release = asyncio.Event()

    @property
    def platform_id(self) -> str:
        return "test:status"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send_text(
        self, _chat_id: str, text: str, *, rich: bool = False,
    ) -> MessageHandle | None:
        self.sent.append(text)
        if text == "Thinking...":
            return MessageHandle("chat", "thinking")
        return None

    async def edit_text(
        self, _handle: MessageHandle, _text: str, *, rich: bool = False,
    ) -> None:
        self.edit_started.set()
        await self.edit_release.wait()

    async def delete_message(self, _handle: MessageHandle) -> None:
        self.delete_started.set()
        await self.delete_release.wait()

    async def send_file(self, _chat_id: str, _path: str) -> None:
        pass

    async def _current_workspace_for_turn(
        self, _uid: str, _chat_id: str,
    ) -> str | None:
        return self.workspace_path


class StatusLatencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_operation_times_out_without_raising(self) -> None:
        async def never_finishes() -> None:
            await asyncio.Event().wait()

        with tempfile.TemporaryDirectory() as tmp:
            bot = _StatusBot(tmp)
            with mock.patch(
                "Cozter.backends_bot.base._STATUS_OPERATION_TIMEOUT_SEC", 0.01,
            ):
                result = await bot._run_status_operation(
                    never_finishes(), action="testing a slow status call",
                )

        self.assertIsNone(result)

    async def test_final_reply_does_not_wait_for_status_edit_or_delete(self) -> None:
        async def fake_run(*_args, on_event, **_kwargs) -> AgentResult:
            await on_event(ChatEvent(kind="text", content="preview"))
            await asyncio.wait_for(bot.edit_started.wait(), timeout=0.1)
            return AgentResult(events=[ChatEvent(kind="text", content="final")])

        with tempfile.TemporaryDirectory() as tmp:
            bot = _StatusBot(tmp)
            with (
                mock.patch.object(agent, "run", side_effect=fake_run),
                mock.patch.object(
                    workspace,
                    "get_run_config",
                    return_value=("codex", "auto", "auto", "auto", "codex"),
                ),
            ):
                await asyncio.wait_for(
                    bot._run_turn("u1", "chat", "hello"), timeout=0.2,
                )

            await asyncio.wait_for(bot.delete_started.wait(), timeout=0.2)
            self.assertTrue(bot.edit_started.is_set())
            self.assertIn("final", bot.sent)

            # Let the background cleanup finish so the test leaves no task.
            bot.edit_release.set()
            bot.delete_release.set()
            await asyncio.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
