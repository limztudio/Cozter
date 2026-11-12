"""Edge case: /pause holds queued work until next user input resumes."""

import os
import tempfile
import unittest
from unittest import mock

from Cozter import workspace
from Cozter.tests.helpers import OfflineBot


class _PauseBot(OfflineBot):
    def __init__(self) -> None:
        super().__init__(["u1"])
        self.replies: list[str] = []

    @property
    def platform_id(self) -> str:
        return "test:pause"

    async def send_text(self, _chat_id: str, text: str, *, rich: bool = False):
        self.replies.append(text)
        return None


class PauseCommandTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = self._tmp.name
        self._orig_state = workspace.WORKSPACE_STATE_PATH
        workspace.WORKSPACE_STATE_PATH = os.path.join(self.ws, "state.json")
        self.bot = _PauseBot()
        self._config = mock.patch.object(workspace, "CONFIG_DIR", self.ws)
        self._config.start()
        self.uid = "u1"
        workspace.select_workspace(self.uid, self.ws, self.bot.platform_id)

    async def asyncTearDown(self) -> None:
        self._config.stop()
        workspace.WORKSPACE_STATE_PATH = self._orig_state
        self._tmp.cleanup()

    def _ctx(self, *, command=None, args="", text=""):
        return self.bot.make_context(
            self.uid, "c1", text=text, command=command, args=args,
        )

    async def test_pause_registered_and_listed(self) -> None:
        self.assertIn("pause", self.bot._COMMANDS)
        await self.bot.dispatch_command(self._ctx(command="nope"))
        self.assertIn("Unknown command: /nope", self.bot.replies[-1])
        self.assertIn("/pause", self.bot.replies[-1])

    async def test_pause_dispatch_holds_and_next_command_resumes(self) -> None:
        await self.bot.dispatch_command(self._ctx(command="pause"))
        self.assertIn(self.uid, self.bot._manually_paused)
        self.assertTrue(self.bot._holds_queue(self.uid))
        self.assertIsNone(self.bot._start_queue_drain(self.uid))
        await self.bot.dispatch_command(self._ctx(command="version"))
        self.assertNotIn(self.uid, self.bot._manually_paused)
        self.assertIn("Paused", self.bot.replies[0])

    async def test_pause_holds_reply_delivery_until_resume(self) -> None:
        await self.bot.dispatch_command(self._ctx(command="pause"))
        record = {"id": "r1", "user_id": self.uid, "chat_id": "c1"}
        self.assertFalse(await self.bot._deliver_staged_reply(record))
        self.assertTrue(self.bot._resume_manual_pause(self.uid))

    async def test_pause_freezes_live_turn_and_resumes_it(self) -> None:
        import asyncio

        admitted = await self.bot._admit_queue_entry(
            self.uid, "live work", "c1",
            self.bot._cancel_generations.get(self.uid, 0),
        )
        assert admitted is not None and admitted.entry is not None
        entry_id = admitted.entry["id"]
        msg_queue = self.bot._ensure_message_queue(self.uid)
        lock = self.bot._ensure_task_lock(self.uid)
        started = asyncio.Event()
        entered = asyncio.Event()

        async def fake_turn(*args, **kwargs):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise
            return True

        async def fake_checkpoint(*args, **kwargs):
            entered.set()
            return True

        drain = asyncio.create_task(
            self.bot._drain_message_queue(self.uid),
        )
        try:
            with mock.patch.object(self.bot, "_run_turn", fake_turn):
                await asyncio.wait_for(started.wait(), timeout=5)
                self.assertIn(self.uid, self.bot._running_tasks)
                with mock.patch.object(
                    self.bot, "_save_job_checkpoint", fake_checkpoint,
                ):
                    await self.bot.dispatch_command(self._ctx(command="pause"))
                self.assertIn(self.uid, self.bot._manually_paused)
                # Acknowledged freeze: no "Cancelled." notice for the live turn.
                self.assertNotIn("Cancelled.", self.bot.replies)
                # The frozen entry stays durable and re-queued for resume.
                data = self.bot._read_queue_file()
                entries = self.bot._queue_entries(data.get(self.uid))
                self.assertTrue(any(
                    entry.get("id") == entry_id for entry in entries
                ))
                self.assertTrue(any(
                    isinstance(item, tuple) and len(item) == 4
                    and item[2] == entry_id for item in msg_queue._queue
                ))
                await asyncio.wait_for(drain, timeout=5)
            # Resume input restarts the retained work.
            self.assertFalse(lock.locked())
            self.assertTrue(self.bot._resume_manual_pause(self.uid))
        finally:
            if not drain.done():
                drain.cancel()
                try:
                    await drain
                except (asyncio.CancelledError, Exception):
                    pass
            if lock.locked():
                lock.release()
            self.bot._running_tasks.pop(self.uid, None)
            self.bot._active_queue_entries.pop(self.uid, None)
            self.bot._inject_queues.pop(self.uid, None)
            self.bot._cancel_acknowledged.discard(self.uid)

    async def test_pause_rejects_inject_until_resume(self) -> None:
        await self.bot.dispatch_command(self._ctx(command="pause"))
        await self.bot.cmd_inject(self._ctx(command="inject", args="late"))
        self.assertIn("Paused", self.bot.replies[-1])
        self.assertIn(self.uid, self.bot._manually_paused)

    async def test_stop_clears_manual_pause(self) -> None:
        await self.bot.dispatch_command(self._ctx(command="pause"))
        await self.bot.cmd_stop(self._ctx())
        self.assertNotIn(self.uid, self.bot._manually_paused)
        self.assertEqual(self.bot.replies[-1], "Cancelled.")


    async def test_bare_resume_text_wakes_without_queueing(self) -> None:
        import asyncio
        from unittest import mock

        async def fake_turn(*args, **kwargs):
            entry_id = kwargs.get("queue_entry_id")
            await self.bot._persist_complete(args[1] if len(args) > 1 else self.uid, entry_id)
            return True

        with mock.patch.object(self.bot, "_run_turn", fake_turn):
            await self.bot._admit_queue_entry(
                self.uid, "held task", "c1",
                self.bot._cancel_generations.get(self.uid, 0),
            )
            await self.bot.dispatch_command(self._ctx(command="pause"))
            await self.bot.dispatch_text(self._ctx(text="resume"))
            for _ in range(100):
                data = self.bot._read_queue_file()
                if not self.bot._queue_entries(data.get(self.uid)):
                    break
                await asyncio.sleep(0.05)
            data = self.bot._read_queue_file()
            self.assertEqual(
                [entry.get("text") for entry in self.bot._queue_entries(data.get(self.uid))],
                [],
            )
            self.assertIn("Resumed.", self.bot.replies)
            self.assertNotIn(self.uid, self.bot._manually_paused)

    async def test_real_resume_text_still_runs_as_new_task(self) -> None:
        import asyncio
        from unittest import mock

        seen: list[str] = []

        async def fake_turn(*args, **kwargs):
            text = args[2] if len(args) > 2 else kwargs.get("text", "")
            seen.append(text)
            await self.bot._persist_complete(args[1] if len(args) > 1 else self.uid, kwargs.get("queue_entry_id"))
            return True

        with mock.patch.object(self.bot, "_run_turn", fake_turn):
            await self.bot._admit_queue_entry(
                self.uid, "held task", "c1",
                self.bot._cancel_generations.get(self.uid, 0),
            )
            await self.bot.dispatch_command(self._ctx(command="pause"))
            await self.bot.dispatch_text(self._ctx(text="now add logging"))
            for _ in range(100):
                if len(seen) >= 2:
                    break
                await asyncio.sleep(0.05)
            self.assertEqual(seen[:2], ["held task", "now add logging"])
