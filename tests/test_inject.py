"""Regression tests for the live ``/inject`` interruption path."""

import asyncio
import tempfile
import unittest
from unittest import mock

from Cozter import agent, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.backends_bot.base import BotContext, BotPlatform


class _InjectRaceBot(BotPlatform):
    """A bot whose final reply remains in flight for the race window."""

    def __init__(self, workspace_path: str) -> None:
        super().__init__([])
        self.workspace_path = workspace_path
        self.sent: list[str] = []
        self.final_reply_started = asyncio.Event()
        self.release_final_reply = asyncio.Event()

    @property
    def platform_id(self) -> str:
        return "test:inject"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send_text(
        self, _chat_id: str, text: str, *, rich: bool = False,
    ) -> None:
        self.sent.append(text)
        if text == "final reply":
            self.final_reply_started.set()
            await self.release_final_reply.wait()

    async def edit_text(
        self, _handle, _text: str, *, rich: bool = False,
    ) -> None:
        pass

    async def delete_message(self, _handle) -> None:
        pass

    async def send_file(self, _chat_id: str, _path: str) -> None:
        pass

    async def _current_workspace_for_turn(
        self, _uid: str, _chat_id: str,
    ) -> str | None:
        return self.workspace_path


class InjectCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_inject_is_rejected_once_final_reply_delivery_starts(
        self,
    ) -> None:
        """Never acknowledge an inject after the agent's final phase ends."""
        async def completed_run(*_args, **_kwargs) -> AgentResult:
            return AgentResult(events=[
                ChatEvent(kind="text", content="final reply"),
            ])

        with tempfile.TemporaryDirectory() as ws:
            bot = _InjectRaceBot(ws)
            with (
                mock.patch.object(agent, "run", new=completed_run),
                mock.patch.object(
                    workspace,
                    "get_run_config",
                    return_value=("flexible", "auto", "auto", "auto", "codex"),
                ),
            ):
                turn = asyncio.create_task(
                    bot._run_turn("u1", "chat", "original request"),
                )
                await asyncio.wait_for(bot.final_reply_started.wait(), timeout=1)

                await bot.cmd_inject(BotContext(
                    user_id="u1", chat_id="chat", text="", command="inject",
                    args="late requirement", attachment=None, platform=bot,
                ))

                self.assertIn("The task has already finished.", bot.sent)
                self.assertNotIn("Injected.", bot.sent)

                bot.release_final_reply.set()
                await asyncio.wait_for(turn, timeout=1)

            # ``_run_turn`` is normally wrapped by ``_dispatch_ai``, which
            # owns the usual map cleanup.  This direct unit call leaves the
            # closed queue behind only for this test.
            bot._inject_queues.pop("u1", None)


if __name__ == "__main__":
    unittest.main()
