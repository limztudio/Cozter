"""Regression tests for the live ``/inject`` interruption path."""

import asyncio
import tempfile
import unittest
from unittest import mock

from Cozter import agent, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.backends_bot.base import BotContext
from Cozter.tests.helpers import TestBot


class _InjectRaceBot(TestBot):
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

    async def send_text(
        self, _chat_id: str, text: str, *, rich: bool = False,
    ) -> None:
        self.sent.append(text)
        if text == "final reply":
            self.final_reply_started.set()
            await self.release_final_reply.wait()

    async def _current_workspace_for_turn(
        self, _uid: str, _chat_id: str,
    ) -> str | None:
        return self.workspace_path


class InjectCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_inject_queues_followup_once_final_reply_delivery_starts(
        self,
    ) -> None:
        """Queue a late inject as a follow-up instead of dropping it."""
        async def completed_run(*_args, **_kwargs) -> AgentResult:
            return AgentResult(events=[
                ChatEvent(kind="text", content="final reply"),
            ])

        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as cfg:
            from Cozter import workspace as workspace_mod
            old_config_dir = workspace_mod.CONFIG_DIR
            workspace_mod.CONFIG_DIR = cfg
            self.addCleanup(setattr, workspace_mod, "CONFIG_DIR", old_config_dir)
            bot = _InjectRaceBot(ws)
            bot.notify_targets = ["u1"]
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

                # A late inject must never be dropped: the live window is
                # closed, so it queues as a follow-up turn instead.
                self.assertNotIn("The task has already finished.", bot.sent)
                self.assertNotIn("Injected.", bot.sent)
                self.assertTrue(
                    any(text.startswith("Queued (") for text in bot.sent),
                    bot.sent,
                )

                # The background drain may already have consumed the
                # in-memory entry, so verify via the queued reply plus the
                # durable ledger instead of racing the queue.
                queued_entry = None
                try:
                    queued_entry = bot._message_queues["u1"].get_nowait()
                except Exception:
                    queued_entry = None
                if queued_entry is not None:
                    self.assertEqual(queued_entry[0], "late requirement")
                    self.assertEqual(queued_entry[1], "chat")
                else:
                    data = bot._read_queue_file()
                    entries = bot._queue_entries(data.get("u1"))
                    self.assertTrue(
                        any(
                            isinstance(entry, dict)
                            and entry.get("text") == "late requirement"
                            and entry.get("chat_id") == "chat"
                            for entry in entries
                        ) or any(
                            text == "late requirement" for text in bot.sent
                        ),
                        bot.sent,
                    )

                bot.release_final_reply.set()
                await asyncio.wait_for(turn, timeout=1)

            # Direct unit call skips _dispatch_ai cleanup; pop the closed queue here.
            bot._inject_queues.pop("u1", None)


if __name__ == "__main__":
    unittest.main()


class InjectAckBoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_inject_ack_is_bounded_when_send_hangs(self) -> None:
        """A slow ack send must not stall /inject past the ack bound.

        Regression: while a scheduled (ephemeral) turn is processing,
        a flaky platform send (e.g. signal-cli reconnects) held
        ``cmd_inject``'s "Injected." reply for 60s+, making /inject
        look dead. The message itself is queued synchronously, so the
        ack is now bounded and the agent still receives it.
        """
        import time

        from Cozter.backends_bot import base as bot_base

        class _SlowAckBot(_InjectRaceBot):
            async def send_text(
                self, _chat_id: str, text: str, *, rich: bool = False,
            ):
                if text == "Injected.":
                    await asyncio.sleep(60)
                    return None
                return await super().send_text(
                    _chat_id, text, rich=rich,
                )

        async def waiting_run(*_args, **kwargs) -> AgentResult:
            iq = kwargs.get("inject_queue")
            assert iq is not None
            msg = await asyncio.wait_for(iq.get(), timeout=20)
            assert msg == "urgent!"
            return AgentResult(events=[
                ChatEvent(kind="text", content="after:" + msg),
            ])

        with tempfile.TemporaryDirectory() as ws:
            bot = _SlowAckBot(ws)
            with (
                mock.patch.object(agent, "run", new=waiting_run),
                mock.patch.object(
                    workspace,
                    "get_run_config",
                    return_value=("flexible", "auto", "auto", "auto", "codex"),
                ),
            ):
                turn = asyncio.create_task(
                    bot._run_ephemeral_turn("u1", "chat", "sched cmd"),
                )
                await asyncio.sleep(0.5)
                t0 = time.monotonic()
                await asyncio.wait_for(
                    bot.cmd_inject(BotContext(
                        user_id="u1", chat_id="chat", text="",
                        command="inject", args="urgent!",
                        attachment=None, platform=bot,
                    )),
                    timeout=bot_base._INJECT_ACK_TIMEOUT_SEC + 15,
                )
                self.assertLess(
                    time.monotonic() - t0,
                    bot_base._INJECT_ACK_TIMEOUT_SEC + 15,
                )
                await asyncio.wait_for(turn, timeout=25)

            bot._inject_queues.pop("u1", None)
