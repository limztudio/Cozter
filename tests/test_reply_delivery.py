"""Regression coverage for durable completed-turn text delivery."""

import asyncio
import tempfile
import unittest
from unittest import mock

from Cozter import agent, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.backends_bot.base import BotContext, BotPlatform


class _ReplyDeliveryBot(BotPlatform):
    def __init__(self, workspace_path: str, *, fail_final: bool) -> None:
        super().__init__(["u1"])
        self.workspace_path = workspace_path
        self.fail_final = fail_final
        self.sent: list[str] = []

    @property
    def platform_id(self) -> str:
        return "test:reply-delivery"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def start_detached_task_watcher(self) -> None:
        # This test drives retry passes explicitly.
        return

    async def send_text(
        self, _chat_id: str, text: str, *, rich: bool = False,
    ) -> None:
        if text == "final response" and self.fail_final:
            raise OSError("platform unavailable")
        self.sent.append(text)
        return None

    async def edit_text(self, _handle, _text: str, *, rich: bool = False) -> None:
        pass

    async def delete_message(self, _handle) -> None:
        pass

    async def send_file(self, _chat_id: str, _path: str) -> None:
        pass

    async def _current_workspace_for_turn(
        self, _uid: str, _chat_id: str,
    ) -> str | None:
        return self.workspace_path


class ReplyDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_final_send_retains_reply_and_retry_does_not_rerun_agent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_config_dir = workspace.CONFIG_DIR
            workspace.CONFIG_DIR = tmp
            self.addCleanup(setattr, workspace, "CONFIG_DIR", old_config_dir)
            bot = _ReplyDeliveryBot(tmp, fail_final=True)
            ctx = BotContext(
                user_id="u1", chat_id="chat", text="prompt",
                command=None, args="", attachment=None, platform=bot,
            )
            result = AgentResult(events=[
                ChatEvent(kind="text", content="final response"),
            ])
            queued_id = await bot._persist_enqueue("u1", "later prompt", "chat")
            queue = bot._ensure_message_queue("u1")
            queue.put_nowait(("later prompt", "chat", queued_id, False))

            with (
                mock.patch.object(
                    agent, "run", new=mock.AsyncMock(return_value=result),
                ) as run_agent,
                mock.patch.object(
                    workspace,
                    "get_run_config",
                    return_value=("flexible", "", "", "auto", "codex"),
                ),
            ):
                with self.assertLogs("Cozter.backends_bot.base", level="WARNING"):
                    await bot._dispatch_ai(ctx, "prompt")

                # The inbound prompt is safely complete: recovery uses the
                # persisted completed reply, never another agent invocation.
                self.assertEqual(
                    [entry["text"] for entry in bot._read_queue_file()["u1"]],
                    ["later prompt"],
                )
                records = await bot._list_reply_delivery_records()
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["messages"], ["final response"])
                run_agent.assert_awaited_once()
                # A later prompt must not overtake the undelivered answer.
                self.assertEqual(queue.qsize(), 1)

                # Simulate a restart: a fresh platform instance loads the
                # ledger and sends the already-computed payload directly.
                retry_bot = _ReplyDeliveryBot(tmp, fail_final=False)
                await retry_bot.restore_reply_deliveries()
                await retry_bot._check_reply_deliveries()

                self.assertEqual(retry_bot.sent, ["final response"])
                self.assertEqual(
                    await retry_bot._list_reply_delivery_records(), [],
                )
                run_agent.assert_awaited_once()

    async def test_restart_consumes_staged_reply_before_rerunning_prompt(
        self,
    ) -> None:
        """A crash between staging and queue completion must not rerun tools."""
        with tempfile.TemporaryDirectory() as tmp:
            old_config_dir = workspace.CONFIG_DIR
            workspace.CONFIG_DIR = tmp
            self.addCleanup(setattr, workspace, "CONFIG_DIR", old_config_dir)
            before_crash = _ReplyDeliveryBot(tmp, fail_final=False)
            entry_id = await before_crash._persist_enqueue(
                "u1", "prompt", "chat",
            )
            result = AgentResult(events=[
                ChatEvent(kind="text", content="final response"),
            ])

            # Deliberately stop after the durable reply write, simulating a
            # process crash before the foreground handler completes the
            # matching inbound entry.
            await before_crash._stage_reply_delivery(
                "u1", entry_id, "chat", tmp, result, allow_await=True,
            )
            self.assertIn("u1", before_crash._read_queue_file())
            self.assertEqual(
                len(await before_crash._list_reply_delivery_records()), 1,
            )

            retry_bot = _ReplyDeliveryBot(tmp, fail_final=False)
            with mock.patch.object(
                agent, "run", new=mock.AsyncMock(),
            ) as run_agent:
                await retry_bot.restore_queues()
                self.assertEqual(retry_bot._message_queues["u1"].qsize(), 1)
                await retry_bot._check_reply_deliveries()
                await asyncio.sleep(0)

            self.assertEqual(retry_bot.sent, ["final response"])
            self.assertEqual(retry_bot._read_queue_file(), {})
            self.assertEqual(
                await retry_bot._list_reply_delivery_records(), [],
            )
            run_agent.assert_not_awaited()

    async def test_cancel_prevents_a_stale_delivery_snapshot_from_sending(
        self,
    ) -> None:
        """A queued watcher snapshot must honor /cancel before it sends."""
        with tempfile.TemporaryDirectory() as tmp:
            old_config_dir = workspace.CONFIG_DIR
            workspace.CONFIG_DIR = tmp
            self.addCleanup(setattr, workspace, "CONFIG_DIR", old_config_dir)
            bot = _ReplyDeliveryBot(tmp, fail_final=False)
            entry_id = await bot._persist_enqueue("u1", "prompt", "chat")
            record = await bot._stage_reply_delivery(
                "u1", entry_id, "chat", tmp,
                AgentResult(events=[
                    ChatEvent(kind="text", content="final response"),
                ]),
                allow_await=True,
            )

            # Queue /cancel ahead of a delivery retry which already holds a
            # stale record object, simulating the poller snapshot race.
            delivery_lock = bot._reply_delivery_lock("u1")
            await delivery_lock.acquire()
            cancel_ctx = BotContext(
                user_id="u1", chat_id="chat", text="/cancel",
                command="cancel", args="", attachment=None, platform=bot,
            )
            cancel_task = asyncio.create_task(bot.cmd_cancel(cancel_ctx))
            await asyncio.sleep(0)
            delivery_task = asyncio.create_task(
                bot._deliver_staged_reply(record),
            )
            await asyncio.sleep(0)
            delivery_lock.release()
            await asyncio.gather(cancel_task, delivery_task)

            self.assertEqual(bot.sent, ["Cancelled."])
            self.assertEqual(bot._read_queue_file(), {})
            self.assertEqual(
                await bot._list_reply_delivery_records(), [],
            )


if __name__ == "__main__":
    unittest.main()
