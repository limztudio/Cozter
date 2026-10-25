"""Regression coverage for durable completed-turn text delivery."""

import asyncio
import os
import tempfile
import unittest
from unittest import mock

from Cozter import agent, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.backends_bot.base import BotContext
from Cozter.tests.helpers import (
    persist_queue_entry,
    OfflineBot,
    create_python_script_process,
)


class ReplyDeliveryBot(OfflineBot):
    def __init__(self, workspace_path: str, *, fail_final: bool) -> None:
        super().__init__(["u1"])
        self.workspace_path = workspace_path
        self.fail_final = fail_final
        self.sent: list[str] = []

    @property
    def platform_id(self) -> str:
        return "test:reply-delivery"

    def start_detached_task_watcher(self) -> None:
        return

    async def send_text(
        self, _chat_id: str, text: str, *, rich: bool = False,
    ) -> None:
        if text == "final response" and self.fail_final:
            raise OSError("platform unavailable")
        self.sent.append(text)
        return None

    async def _current_workspace_for_turn(
        self, _uid: str, _chat_id: str,
    ) -> str | None:
        return self.workspace_path


class ReplyDeliveryTests(unittest.IsolatedAsyncioTestCase):
    def _temporary_workspace(self) -> str:
        """Keep standalone diagnostic/PID fixtures under unittest cleanup."""
        return self.enterContext(tempfile.TemporaryDirectory())

    async def test_detached_reply_preserves_an_interactive_answer_pause(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = ReplyDeliveryBot(tmp, fail_final=False)
            bot._awaiting_answer.add("u1")
            result = AgentResult(events=[ChatEvent(kind="text", content="background done [[await]]")])

            await bot._send_detached_result("chat", tmp, result, "u1")

            self.assertEqual(bot.sent, ["background done"])
            self.assertIn("u1", bot._awaiting_answer)

    async def test_failed_final_send_retains_reply_and_retry_does_not_rerun_agent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_config_dir = workspace.CONFIG_DIR
            workspace.CONFIG_DIR = tmp
            self.addCleanup(setattr, workspace, "CONFIG_DIR", old_config_dir)
            bot = ReplyDeliveryBot(tmp, fail_final=True)
            ctx = BotContext(
                user_id="u1", chat_id="chat", text="prompt",
                command=None, args="", attachment=None, platform=bot,
            )
            result = AgentResult(events=[
                ChatEvent(kind="text", content="final response"),
            ])
            queue = bot._ensure_message_queue("u1")

            async def run_with_later_inbound(*_args, **_kwargs):
                await bot._admit_queue_entry("u1", "later prompt", "chat", 0)
                return result

            with (
                mock.patch.object(
                    agent, "run", new=mock.AsyncMock(side_effect=run_with_later_inbound),
                ) as run_agent,
                mock.patch.object(
                    workspace,
                    "get_run_config",
                    return_value=("flexible", "", "", "auto", "codex"),
                ),
            ):
                with self.assertLogs("Cozter.backends_bot.base", level="WARNING"):
                    await bot._dispatch_ai(ctx, "prompt")

                self.assertEqual(
                    [entry["text"] for entry in bot._read_queue_file()["u1"]],
                    ["later prompt"],
                )
                records = await bot._list_reply_delivery_records()
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["messages"], ["final response"])
                run_agent.assert_awaited_once()
                # A later prompt must not overtake the answer.
                self.assertEqual(queue.qsize(), 1)

                # Simulate a restart via a fresh platform instance.
                retry_bot = ReplyDeliveryBot(tmp, fail_final=False)
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
            before_crash = ReplyDeliveryBot(tmp, fail_final=False)
            entry_id = await persist_queue_entry(before_crash,
                "u1", "prompt", "chat",
            )
            result = AgentResult(events=[
                ChatEvent(kind="text", content="final response"),
            ])

            # Stop after the durable write (crash before completion).
            await before_crash._stage_reply_delivery(
                "u1", entry_id, "chat", tmp, result, allow_await=True,
            )
            self.assertIn("u1", before_crash._read_queue_file())
            self.assertEqual(
                len(await before_crash._list_reply_delivery_records()), 1,
            )

            retry_bot = ReplyDeliveryBot(tmp, fail_final=False)
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
            bot = ReplyDeliveryBot(tmp, fail_final=False)
            entry_id = await persist_queue_entry(bot, "u1", "prompt", "chat")
            record = await bot._stage_reply_delivery(
                "u1", entry_id, "chat", tmp,
                AgentResult(events=[
                    ChatEvent(kind="text", content="final response"),
                ]),
                allow_await=True,
            )

            # Queue /cancel ahead of a stale-record retry.
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

    async def test_drain_drops_malformed_entry_and_continues(self) -> None:
        """A malformed in-memory entry must not wedge the drain loop."""
        with tempfile.TemporaryDirectory() as tmp:
            old_config_dir = workspace.CONFIG_DIR
            workspace.CONFIG_DIR = tmp
            self.addCleanup(setattr, workspace, "CONFIG_DIR", old_config_dir)
            bot = ReplyDeliveryBot(tmp, fail_final=False)
            result = AgentResult(events=[
                ChatEvent(kind="text", content="final response"),
            ])
            entry_id = await persist_queue_entry(bot, "u1", "prompt", "chat")
            queue = bot._ensure_message_queue("u1")
            queue.put_nowait(("short",))
            queue.put_nowait(("prompt", "chat", entry_id, False))

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
                with self.assertLogs(
                    "Cozter.backends_bot.base", level="WARNING",
                ):
                    await bot._drain_message_queue("u1")
                run_agent.assert_awaited_once()
            self.assertIn("final response", bot.sent)

    async def test_explicit_cancellation_interrupts_hanging_delayed_reply(self) -> None:
        for command in ("stop", "cancel"):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                with mock.patch.object(workspace, "CONFIG_DIR", tmp):
                    bot = ReplyDeliveryBot(tmp, fail_final=False)
                    entry_id = await persist_queue_entry(bot, "u1", "prompt", "chat")
                    record = await bot._stage_reply_delivery(
                        "u1", entry_id, "chat", tmp,
                        AgentResult(events=[ChatEvent(kind="text", content="final response")]),
                        allow_await=True,
                    )
                    started = asyncio.Event()

                    async def hanging_send(_chat, text, *, rich=False):
                        if text == "final response":
                            started.set()
                            await asyncio.Event().wait()
                        bot.sent.append(text)

                    with mock.patch.object(bot, "send_text", side_effect=hanging_send):
                        delivery = asyncio.create_task(bot._deliver_staged_reply(record))
                        await asyncio.wait_for(started.wait(), timeout=2)
                        ctx = BotContext(
                            user_id="u1", chat_id="chat", text=f"/{command}",
                            command=command, args="", attachment=None, platform=bot,
                        )
                        await asyncio.wait_for(getattr(bot, f"cmd_{command}")(ctx), timeout=2)
                        self.assertFalse(await asyncio.wait_for(delivery, timeout=2))
                    self.assertEqual(bot.sent, ["Cancelled."])
                    self.assertEqual(await bot._list_reply_delivery_records(), [])
                    self.assertEqual(bot._read_queue_file(), {})
                    self.assertEqual(bot._reply_delivery_tasks, {})

    async def test_blocked_standalone_retry_counts_as_an_active_turn(self) -> None:
        """A standalone retry's blocked send must hold the update idle wait.

        This is the ``BotPlatform`` regression for the standalone durable
        reply retry: a restart-retry re-sends an already staged reply
        without holding the per-user turn lock, so the update loop's
        ``has_active_turns()`` check is the only thing that keeps a
        restart from racing ahead of the in-flight send.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(workspace, "CONFIG_DIR", tmp):
                bot = ReplyDeliveryBot(tmp, fail_final=False)
                entry_id = await persist_queue_entry(bot, "u1", "prompt", "chat")
                await bot._stage_reply_delivery(
                    "u1", entry_id, "chat", tmp,
                    AgentResult(events=[
                        ChatEvent(kind="text", content="final response"),
                    ]),
                    allow_await=False,
                )
                entered = asyncio.Event()
                release = asyncio.Event()

                async def blocked_send(_chat, text, *, rich=False):
                    if text == "final response":
                        entered.set()
                        await release.wait()
                    bot.sent.append(text)
                    return None

                with mock.patch.object(
                    bot, "send_text", side_effect=blocked_send,
                ), mock.patch.object(
                    bot, "start_detached_task_watcher",
                ):
                    retry = asyncio.create_task(bot._check_reply_deliveries())
                    try:
                        await asyncio.wait_for(entered.wait(), timeout=5)
                        # The owned retry is in flight without a held lock.
                        self.assertEqual(
                            len(bot._reply_delivery_tasks.get("u1", ())), 1,
                        )
                        self.assertTrue(bot.has_active_turns())
                        self.assertIn(
                            "pending_deliveries=[u1(1)]",
                            bot.stuck_turn_diagnostics(),
                        )
                        release.set()
                        await asyncio.wait_for(retry, timeout=5)
                    finally:
                        release.set()
                        if not retry.done():
                            retry.cancel()
                            await asyncio.gather(
                                retry, return_exceptions=True,
                            )
                self.assertEqual(bot.sent, ["final response"])
                self.assertEqual(
                    await bot._list_reply_delivery_records(), [],
                )
                self.assertEqual(bot._reply_delivery_tasks, {})
                self.assertFalse(bot.has_active_turns())

    async def test_stop_services_cancels_a_blocked_standalone_retry(self) -> None:
        """Shutdown cancels a blocked retry while keeping its ledger row.

        Cancelling (not awaiting) the stuck send lets ``stop()`` proceed
        without preserving delivery order, and the retained staged record
        resumes after restart via ``restore_reply_deliveries``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(workspace, "CONFIG_DIR", tmp):
                bot = ReplyDeliveryBot(tmp, fail_final=False)
                entry_id = await persist_queue_entry(bot, "u1", "prompt", "chat")
                record = await bot._stage_reply_delivery(
                    "u1", entry_id, "chat", tmp,
                    AgentResult(events=[
                        ChatEvent(kind="text", content="final response"),
                    ]),
                    allow_await=False,
                )
                entered = asyncio.Event()

                async def blocked_send(_chat, text, *, rich=False):
                    if text == "final response":
                        entered.set()
                        await asyncio.Event().wait()
                    bot.sent.append(text)
                    return None

                with mock.patch.object(
                    bot, "send_text", side_effect=blocked_send,
                ):
                    delivery = asyncio.create_task(
                        bot._deliver_staged_reply(record),
                    )
                    await asyncio.wait_for(entered.wait(), timeout=5)
                    self.assertTrue(bot.has_active_turns())
                    await asyncio.wait_for(
                        bot._stop_daemon_services(), timeout=5,
                    )
                    self.assertFalse(await asyncio.wait_for(
                        delivery, timeout=5,
                    ))
                self.assertEqual(bot.sent, [])
                # The staged text survives for post-restart retry.
                self.assertEqual(
                    len(await bot._list_reply_delivery_records()), 1,
                )
                self.assertEqual(bot._reply_delivery_tasks, {})
                self.assertFalse(bot.has_active_turns())

    async def test_pipelined_uploads_overlap_next_turn(self) -> None:
        """Texts return fast; pictures upload behind the next agent turn."""
        with tempfile.TemporaryDirectory() as tmp:
            old_config_dir = workspace.CONFIG_DIR
            workspace.CONFIG_DIR = tmp
            self.addCleanup(setattr, workspace, "CONFIG_DIR", old_config_dir)
            first_pic = os.path.join(tmp, "first.png")
            second_pic = os.path.join(tmp, "second.png")
            with open(first_pic, "wb") as file_handle:
                file_handle.write(b"first-picture")
            with open(second_pic, "wb") as file_handle:
                file_handle.write(b"second-picture")

            class _UploadBot(ReplyDeliveryBot):
                def __init__(self, workspace_path: str) -> None:
                    super().__init__(workspace_path, fail_final=False)
                    self.files: list[str] = []
                    self.release_uploads = asyncio.Event()
                    self.text_order: list[str] = []

                async def send_text(
                    self, chat_id: str, text: str, *, rich: bool = False,
                ) -> None:
                    self.sent.append(text)
                    self.text_order.append(text)
                    return None

                async def send_file(self, chat_id: str, path: str) -> None:
                    self.files.append(os.path.basename(path))
                    await self.release_uploads.wait()

            bot = _UploadBot(tmp)
            first = AgentResult(events=[
                ChatEvent(kind="text", content="first answer"),
                ChatEvent(kind="attachment", content=first_pic),
            ])
            await bot._send_result("chat", tmp, first, uid="u1")
            # Texts out while the picture uploads: next turn starts.
            self.assertEqual(bot.sent, ["first answer"])
            self.assertEqual(bot.files, [])
            self.assertTrue(bot.has_active_turns())

            second = AgentResult(events=[
                ChatEvent(kind="text", content="second answer"),
                ChatEvent(kind="attachment", content=second_pic),
            ])
            second_task = asyncio.create_task(
                bot._send_result("chat", tmp, second, uid="u1"),
            )
            await asyncio.sleep(0.05)
            # Text waits for the first picture.
            self.assertNotIn("second answer", bot.sent)
            bot.release_uploads.set()
            await second_task
            for _ in range(20):
                await asyncio.sleep(0)
            self.assertEqual(bot.sent, ["first answer", "second answer"])
            self.assertEqual(bot.files, ["first.png", "second.png"])
            self.assertFalse(bot.has_active_turns())

    async def test_cancel_discards_background_uploads(self) -> None:
        """Cancelled pictures never send and unblock has_active_turns."""
        with tempfile.TemporaryDirectory() as tmp:
            old_config_dir = workspace.CONFIG_DIR
            workspace.CONFIG_DIR = tmp
            self.addCleanup(setattr, workspace, "CONFIG_DIR", old_config_dir)
            pic = os.path.join(tmp, "pic.png")
            with open(pic, "wb") as file_handle:
                file_handle.write(b"picture")

            class _StuckUploadBot(ReplyDeliveryBot):
                def __init__(self, workspace_path: str) -> None:
                    super().__init__(workspace_path, fail_final=False)
                    self.files: list[str] = []

                async def send_file(self, chat_id: str, path: str) -> None:
                    self.files.append(os.path.basename(path))
                    await asyncio.sleep(3600)

            bot = _StuckUploadBot(tmp)
            result = AgentResult(events=[
                ChatEvent(kind="text", content="answer"),
                ChatEvent(kind="attachment", content=pic),
            ])
            await bot._send_result("chat", tmp, result, uid="u1")
            self.assertEqual(bot.sent, ["answer"])
            self.assertTrue(bot.has_active_turns())
            task_running, cancelled_work = await bot._cancel_user_work("u1")
            self.assertTrue(cancelled_work)
            self.assertFalse(task_running)
            self.assertEqual(bot.files, [])
            for _ in range(20):
                await asyncio.sleep(0)
            self.assertFalse(bot.has_active_turns())

    async def test_stuck_diag_reports_backend_model_pid_and_last_progress(
        self,
    ) -> None:
        """A stuck wait names backend/model/PID/elapsed/last-progress."""
        bot = ReplyDeliveryBot(self._temporary_workspace(), fail_final=False)
        task = asyncio.current_task()
        assert task is not None
        bot._running_tasks["u1"] = task
        bot._begin_turn_progress("u1", backend="meta", model="muse-spark-1.3")
        bot._note_turn_process("u1", 42424)
        bot._record_turn_event_progress(
            "u1", ChatEvent(kind="tool", content="$ sleep 3600"),
        )
        diag = bot.stuck_turn_diagnostics()
        self.assertIn("backend=meta", diag)
        self.assertIn("model=muse-spark-1.3", diag)
        self.assertIn("pid=42424", diag)
        self.assertIn("elapsed=", diag)
        self.assertIn("last=tool", diag)
        self.assertIn("tool: shell", diag)

    async def test_stuck_diag_redacts_prompts_args_and_credentials(
        self,
    ) -> None:
        """Progress labels never carry command text, paths, or secrets."""
        bot = ReplyDeliveryBot(self._temporary_workspace(), fail_final=False)
        task = asyncio.current_task()
        assert task is not None
        bot._running_tasks["u1"] = task
        bot._begin_turn_progress("u1", backend="codex", model="gpt-5")
        bot._note_turn_process("u1", 777)
        bot._record_turn_event_progress(
            "u1",
            ChatEvent(
                kind="tool",
                content="bash: run hunter2 --token s3cr3t /etc/passwd",
            ),
        )
        diag = bot.stuck_turn_diagnostics()
        for sensitive in ("hunter2", "s3cr3t", "--token", "/etc/passwd", "run "):
            self.assertNotIn(sensitive, diag)
        # Only the redacted tool name survives.
        self.assertIn("tool: bash", diag)
        bot._record_turn_event_progress(
            "u1",
            ChatEvent(kind="file", content="changed: /etc/passwd hunter2"),
        )
        diag = bot.stuck_turn_diagnostics()
        for sensitive in ("hunter2", "/etc/passwd"):
            self.assertNotIn(sensitive, diag)
        self.assertIn("file: change", diag)

    async def test_stuck_diag_without_progress_uses_current_unknown_fields(
        self,
    ) -> None:
        """Missing progress uses explicit unknown fields in the current diagnostic format."""
        bot = ReplyDeliveryBot(self._temporary_workspace(), fail_final=False)
        task = asyncio.current_task()
        assert task is not None
        bot._running_tasks["u1"] = task
        diag = bot.stuck_turn_diagnostics()
        self.assertIn("u1(cancelled=False)", diag)
        self.assertIn("backend=? model=? pid=? elapsed=0.0s", diag)
        self.assertIn("last=<no progress events yet>", diag)

    async def test_turn_progress_cleared_on_cleanup(self) -> None:
        """Finished/cancelled turns drop their attribution record."""
        bot = ReplyDeliveryBot(self._temporary_workspace(), fail_final=False)
        bot._begin_turn_progress("u1", backend="meta", model="m")
        bot._record_turn_event_progress(
            "u1", ChatEvent(kind="tool", content="bash"),
        )
        self.assertIn("u1", bot._turn_progress)
        lock = bot._ensure_task_lock("u1")
        await lock.acquire()
        bot._running_tasks["u1"] = asyncio.current_task()
        try:
            bot._cleanup_turn("u1", lock)
        except RuntimeError:
            lock.release()
            raise
        self.assertNotIn("u1", bot._turn_progress)
        self.assertNotIn("backend=", bot.stuck_turn_diagnostics())

    async def test_drive_backend_reports_pid_through_on_process_hook(
        self,
    ) -> None:
        """_drive_backend calls on_process with the launched process."""
        seen: list[int | None] = []

        class _PidBackend:
            name = "pid-test"

            async def launch(self, *_args, **_kwargs):
                return await create_python_script_process(
                    "print('{}')\n",
                )

            @staticmethod
            def parse_event(_event, _result) -> None:
                return None

        proc_holder: list[object] = []
        try:
            result, restarting = await agent._drive_backend(
                _PidBackend(), self._temporary_workspace(), "prompt", None, "auto",
                effort=0,
                on_process=lambda proc: (
                    seen.append(proc.pid), proc_holder.append(proc),
                ),
            )
        finally:
            for proc in proc_holder:
                try:
                    proc.kill()  # type: ignore[union-attr]
                    await proc.wait()  # type: ignore[union-attr]
                except (ProcessLookupError, OSError):
                    pass
        self.assertFalse(restarting)
        self.assertEqual(len(seen), 1)
        self.assertIsInstance(seen[0], int)
        self.assertGreater(seen[0], 0)
        self.assertIsNone(result.error)


if __name__ == "__main__":
    unittest.main()
