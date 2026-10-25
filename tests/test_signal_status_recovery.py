"""Owned Signal status creation and recovery never rerun or delay agent work."""

import asyncio
import tempfile
import unittest
from collections import Counter
from unittest import mock

from Cozter import agent, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.backends_bot import base, signal
from Cozter.backends_bot.signal import SignalBot, SignalCliError


class _RecoverySignalBot(SignalBot):
    def __init__(self, ws):
        super().__init__([], jsonrpc_socket="tcp://127.0.0.1:7583")
        self.ws = ws
        self._group_ids = {"group"}
        self.status_initial_timeout_sec = .06
        self.status_recovery_interval_sec = .04
        self.status_update_interval_sec = .005
        self.calls = []
        self.posts = []
        self.edits = []
        self.deletes = []
        self.posted = asyncio.Queue()
        self.edited = asyncio.Queue()
        self.post_started = asyncio.Event()
        self.release_initial = asyncio.Event()
        self.initial_behavior = "success"
        self.fail_first_handle_edits = False
        self.initial_cancelled = False

    async def _current_workspace_for_turn(self, _uid, _chat):
        return self.ws

    async def _rpc_request(self, method, params=None, *, timeout=60):
        self.calls.append((method, dict(params)))
        if method == "remoteDelete":
            self.deletes.append(params["targetTimestamp"])
            return {}
        if "editTimestamp" in params:
            self.edits.append(dict(params))
            self.edited.put_nowait(dict(params))
            if self.fail_first_handle_edits and params["editTimestamp"] == 1001:
                raise SignalCliError("fixture first status edit unavailable")
            return {"timestamp": 2000 + len(self.edits)}
        if params.get("message") == "final":
            return {"timestamp": 9000}
        self.posts.append((asyncio.get_running_loop().time(), dict(params)))
        self.post_started.set()
        if len(self.posts) == 1:
            if self.initial_behavior == "error":
                raise SignalCliError("fixture initial status unavailable")
            if self.initial_behavior == "missing_timestamp":
                return {}
            if self.initial_behavior == "stall":
                try:
                    await self.release_initial.wait()
                except asyncio.CancelledError:
                    self.initial_cancelled = True
                    raise
        timestamp = 1000 + len(self.posts)
        self.posted.put_nowait(timestamp)
        return {"timestamp": timestamp}


class SignalStatusRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.ws))
        self.enterContext(mock.patch.object(workspace, "get_run_config", return_value=(
            "codex", "auto", "auto", "auto", "codex",
        )))
        self.enterContext(mock.patch.object(signal._SignalOutboundJournal, "remember", mock.AsyncMock()))
        self.enterContext(mock.patch.object(signal, "_SIGNAL_SEND_MIN_INTERVAL_SEC", 0))
        self.enterContext(mock.patch.object(base, "_STATUS_OPERATION_TIMEOUT_SEC", .01))
        self.enterContext(mock.patch.object(base, "_STATUS_EDIT_INTERVAL_SEC", 0))
        self.bot = _RecoverySignalBot(self.ws)

    async def asyncTearDown(self):
        await self.bot._flush_outbound_journal()
        await asyncio.sleep(0)
        self.assertFalse(self.status_owners())

    def status_owners(self):
        return [task for task in asyncio.all_tasks()
                if task.get_name().startswith("signal:status-owner:")]

    def final(self):
        return AgentResult(events=[ChatEvent(kind="text", content="final")])

    async def test_initial_send_lock_can_outlast_foreground_wait_and_adopt_one_late_handle(self):
        await self.bot._send_lock.acquire()
        started, finish = asyncio.Event(), asyncio.Event()

        async def run(*_args, on_event, **_kwargs):
            await on_event(ChatEvent(kind="tool", content="older activity"))
            await on_event(ChatEvent(kind="tool", content="latest buffered activity"))
            started.set()
            await finish.wait()
            return self.final()

        with mock.patch.object(agent, "run", side_effect=run) as execute:
            task = asyncio.create_task(self.bot._run_turn("u1", "group", "work"))
            await asyncio.wait_for(started.wait(), .2)
            self.assertEqual(self.bot.posts, [])
            self.bot._send_lock.release()
            edited = await asyncio.wait_for(self.bot.edited.get(), .2)
            self.assertEqual(len(self.bot.posts), 1)
            self.assertEqual(self.bot.posts[0][1]["message"], "Thinking...")
            self.assertEqual(edited["editTimestamp"], 1001)
            self.assertIn("latest buffered activity", edited["message"])
            finish.set()
            await asyncio.wait_for(task, .3)
            execute.assert_awaited_once()
        await asyncio.sleep(.05)
        self.assertEqual(len(self.bot.posts), 1)
        self.assertIn(1001, self.bot.deletes)

    async def test_actual_initial_failure_recovers_latest_activity_after_bounded_backoff(self):
        for behavior in ("error", "missing_timestamp", "stall"):
            with self.subTest(behavior=behavior):
                self.bot = _RecoverySignalBot(self.ws)
                self.bot.initial_behavior = behavior

                async def run(*_args, on_event, **_kwargs):
                    for index in range(50):
                        await on_event(ChatEvent(kind="tool", content=f"coalesced step {index}"))
                    self.assertEqual(await asyncio.wait_for(self.bot.posted.get(), .5), 1002)
                    self.assertEqual(len(self.bot.posts), 2)
                    elapsed = self.bot.posts[1][0] - self.bot.posts[0][0]
                    minimum = .09 if behavior == "stall" else .035
                    self.assertGreaterEqual(elapsed, minimum)
                    self.assertIn("coalesced step 49", self.bot.posts[1][1]["message"])
                    return self.final()

                with (mock.patch.object(agent, "run", side_effect=run) as execute,
                      self.assertLogs("Cozter.performance", "INFO") as logs):
                    await asyncio.wait_for(self.bot._run_turn("u1", "group", "work"), .8)
                    execute.assert_awaited_once()
                await self.bot._flush_outbound_journal()
                self.assertTrue(any("stage=status_post platform=signal reason=recovery outcome=posted"
                                    in line for line in logs.output))
                self.assertFalse(self.status_owners())
                if behavior == "stall":
                    self.assertTrue(self.bot.initial_cancelled)

    async def test_ten_failed_edits_rotate_to_new_handle_and_resume_edits_without_extra_posts(self):
        self.bot.fail_first_handle_edits = True

        async def run(*_args, on_event, **_kwargs):
            self.assertEqual(await self.bot.posted.get(), 1001)
            await on_event(ChatEvent(kind="tool", content="preserved live activity"))
            self.assertEqual(await asyncio.wait_for(self.bot.posted.get(), .5), 1002)
            self.assertEqual(Counter(event["editTimestamp"] for event in self.bot.edits), {1001: 10})
            self.assertGreaterEqual(self.bot.posts[1][0] - self.bot.posts[0][0], .035)
            await on_event(ChatEvent(kind="tool", content="new handle activity"))
            while (edited := await asyncio.wait_for(self.bot.edited.get(), .2))["editTimestamp"] == 1001:
                pass
            self.assertEqual(edited["editTimestamp"], 1002)
            self.assertIn("new handle activity", edited["message"])
            return self.final()

        with mock.patch.object(agent, "run", side_effect=run) as execute:
            await asyncio.wait_for(self.bot._run_turn("u1", "group", "work"), .8)
            execute.assert_awaited_once()
        await asyncio.sleep(.05)
        self.assertEqual(len(self.bot.posts), 2)
        self.assertCountEqual(self.bot.deletes, [1001, 1002])
        self.assertEqual(Counter(event["editTimestamp"] for event in self.bot.edits), {1001: 10, 1002: 1})

    async def test_final_or_stop_during_pending_initial_creation_cannot_send_a_late_status(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                self.bot = _RecoverySignalBot(self.ws)
                await self.bot._send_lock.acquire()
                started, finish = asyncio.Event(), asyncio.Event()

                async def run(*_args, on_event, **_kwargs):
                    await on_event(ChatEvent(kind="tool", content="buffered activity"))
                    started.set()
                    await finish.wait()
                    return self.final()

                with (mock.patch.object(agent, "run", side_effect=run) as execute,
                      mock.patch.object(self.bot, "_send_result", mock.AsyncMock()) as deliver):
                    task = asyncio.create_task(self.bot._run_turn("u1", "group", "work"))
                    await asyncio.wait_for(started.wait(), .2)
                    if cancel:
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await asyncio.wait_for(task, .2)
                        deliver.assert_not_awaited()
                    else:
                        finish.set()
                        await asyncio.wait_for(task, .2)
                        deliver.assert_awaited_once()
                    execute.assert_awaited_once()
                self.bot._send_lock.release()
                await asyncio.sleep(.08)
                self.assertEqual(self.bot.posts, [])
                self.assertFalse(self.status_owners())

    async def test_final_during_recovery_backoff_cancels_future_post(self):
        self.bot.initial_behavior = "error"

        async def run(*_args, on_event, **_kwargs):
            await on_event(ChatEvent(kind="tool", content="no delayed recovery"))
            await asyncio.sleep(0)
            return self.final()

        with mock.patch.object(agent, "run", side_effect=run):
            await asyncio.wait_for(self.bot._run_turn("u1", "group", "work"), .3)
        await asyncio.sleep(.08)
        self.assertEqual(len(self.bot.posts), 1)
        self.assertFalse(self.status_owners())

    async def test_stop_during_initial_foreground_wait_cancels_creator_without_starting_agent(self):
        self.bot.initial_behavior = "stall"
        with (mock.patch.object(base, "_STATUS_OPERATION_TIMEOUT_SEC", .2),
              mock.patch.object(agent, "run", mock.AsyncMock()) as execute):
            task = asyncio.create_task(self.bot._run_turn("u1", "group", "work"))
            await asyncio.wait_for(self.bot.post_started.wait(), .1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, .2)
            execute.assert_not_awaited()
        self.bot.release_initial.set()
        await asyncio.sleep(.08)
        self.assertTrue(self.bot.initial_cancelled)
        self.assertEqual(len(self.bot.posts), 1)
        self.assertFalse(self.status_owners())
        self.assertFalse(self.bot._send_lock.locked())
