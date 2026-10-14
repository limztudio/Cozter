"""Signal progress uses one bounded editable message, with no live RPCs."""

import asyncio
import os
import tempfile
from typing import Any
import unittest
from unittest import mock

from Cozter import agent, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.backends_bot import base as bot_base, signal as signal_module
from Cozter.backends_bot.signal import SignalBot, SignalCliError


class _StatusSignalBot(SignalBot):
    def __init__(self, path: str) -> None:
        super().__init__([], jsonrpc_socket="tcp://127.0.0.1:7583")
        self.path = path
        self._group_ids = {"group"}
        self.calls: list[tuple[float, str, dict[str, Any]]] = []
        self.edits = 0
        self.edit_events = [asyncio.Event() for _ in range(10)]
        self.deleted = asyncio.Event()
        self.initial_behavior = "success"
        self.stall_edits = False
        self.fail_edits = False
        self.edit_cancelled = asyncio.Event()

    async def _current_workspace_for_turn(self, _uid: str, _chat: str) -> str:
        return self.path

    async def _rpc_request(
        self, method: str, params: dict[str, Any] | None = None,
        *, timeout: int | float | None = 60,
    ) -> Any:
        assert params is not None
        self.calls.append((asyncio.get_running_loop().time(), method, params))
        if method == "remoteDelete":
            self.deleted.set()
            return {}
        if params.get("message") == "Thinking...":
            if self.initial_behavior == "error":
                raise SignalCliError("fixture status send failure")
            if self.initial_behavior == "missing_timestamp":
                return {}
            if self.initial_behavior == "stall":
                await asyncio.Event().wait()
            return {"timestamp": "42"}
        if "editTimestamp" in params:
            self.edits += 1
            self.edit_events[self.edits - 1].set()
            if self.fail_edits:
                raise SignalCliError("fixture edit failure")
            if self.stall_edits:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.edit_cancelled.set()
                    raise
        return {"timestamp": str(100 + len(self.calls))}


class SignalStatusTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = tmp.name
        self.bots: list[_StatusSignalBot] = []
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.path))
        self.enterContext(mock.patch.object(workspace, "get_run_config", return_value=(
            "codex", "auto", "auto", "auto", "codex",
        )))
        self.enterContext(mock.patch.object(signal_module, "_SIGNAL_SEND_MIN_INTERVAL_SEC", 0))
        self.enterContext(mock.patch.object(bot_base, "_STATUS_EDIT_INTERVAL_SEC", 0))
        # Feedback tests cover physical journal writes and cancellation. Keep
        # these millisecond RPC/status timing fixtures independent of disk fsync.
        self.enterContext(mock.patch.object(
            signal_module._SignalOutboundJournal, "remember", new=mock.AsyncMock(),
        ))

    def _bot(self, *, interval: float = 0.02) -> _StatusSignalBot:
        bot = _StatusSignalBot(self.path)
        self.bots.append(bot)
        # Shorten only wall-clock waits; preserve the real Signal edit budget.
        bot.status_update_interval_sec = interval
        return bot

    async def asyncTearDown(self) -> None:
        for bot in self.bots:
            await bot._flush_outbound_journal()

    async def test_visible_initial_status_and_coalesced_real_edit_payload(self) -> None:
        bot = self._bot(interval=0.05)

        async def run(*_args, on_event, **_kwargs):
            self.assertEqual(bot.calls[0][2], {
                "groupId": "group", "message": "Thinking...",
            })
            for index in range(9):
                await on_event(ChatEvent(kind="tool", content=f"step {index}"))
                await on_event(ChatEvent(kind="text", content=f"preview {index}"))
            await asyncio.wait_for(bot.edit_events[0].wait(), timeout=0.5)
            for index in range(9, 18):
                await on_event(ChatEvent(kind="tool", content=f"step {index}"))
            await asyncio.wait_for(bot.edit_events[1].wait(), timeout=0.5)
            return AgentResult(events=[ChatEvent(kind="text", content="final")])

        with (
            mock.patch.object(agent, "run", side_effect=run),
            self.assertLogs("Cozter.performance", level="INFO") as trace,
        ):
            await asyncio.wait_for(bot._run_turn("user", "group", "work"), timeout=1)
        thinking_trace = [line for line in trace.output if "stage=thinking_status " in line]
        self.assertEqual(len(thinking_trace), 1)
        self.assertIn("stage=thinking_status platform=signal outcome=posted", thinking_trace[0])
        self.assertNotIn("group", thinking_trace[0])
        self.assertNotIn("user", thinking_trace[0])
        await asyncio.wait_for(bot.deleted.wait(), timeout=0.5)
        edits = [(when, params) for when, _, params in bot.calls if "editTimestamp" in params]
        self.assertEqual(len(edits), 2)
        self.assertGreaterEqual(edits[0][0] - bot.calls[0][0], 0.045)
        self.assertGreaterEqual(edits[1][0] - edits[0][0], 0.045)
        self.assertTrue(all(params["editTimestamp"] == 42 for _, params in edits))
        self.assertIn("step 8", edits[0][1]["message"])
        self.assertIn("preview 8", edits[0][1]["message"])
        self.assertIn("step 17", edits[-1][1]["message"])
        self.assertIn(("remoteDelete", {"groupId": "group", "targetTimestamp": 42}), [
            (method, params) for _, method, params in bot.calls
        ])
        fresh = [params["message"] for _, method, params in bot.calls
                 if method == "send" and "editTimestamp" not in params]
        self.assertEqual(fresh, ["Thinking...", "final"])

    async def test_ten_edit_attempt_cap_includes_failures_and_resets_each_turn(self) -> None:
        bot = self._bot(interval=0.002)

        async def run(*_args, on_event, **_kwargs):
            offset = bot.edits
            for index in range(10):
                await on_event(ChatEvent(kind="tool", content=f"step {index}"))
                await asyncio.wait_for(bot.edit_events[index].wait(), timeout=0.5)
            for index in range(100):
                await on_event(ChatEvent(kind="tool", content=f"over cap {index}"))
            await asyncio.sleep(0.01)
            self.assertEqual(bot.edits - offset, 10)
            return AgentResult(events=[ChatEvent(kind="text", content="final")])

        with mock.patch.object(agent, "run", side_effect=run):
            await asyncio.wait_for(bot._run_turn("user", "group", "work"), timeout=1)
            await asyncio.wait_for(bot.deleted.wait(), timeout=0.5)
            bot.edits = 0
            bot.fail_edits = True
            bot.edit_events = [asyncio.Event() for _ in range(10)]
            bot.deleted.clear()
            await asyncio.wait_for(bot._run_turn("user", "group", "work again"), timeout=1)
            await asyncio.wait_for(bot.deleted.wait(), timeout=0.5)
        self.assertEqual(sum("editTimestamp" in params for _, _, params in bot.calls), 20)

    async def test_initial_status_failure_or_timeout_never_spams_or_hides_final(self) -> None:
        for behavior in ("error", "stall", "missing_timestamp"):
            with self.subTest(behavior=behavior):
                bot = self._bot()
                bot.initial_behavior = behavior

                async def run(*_args, on_event, **_kwargs):
                    for _ in range(20):
                        await on_event(ChatEvent(kind="tool", content="progress"))
                        await on_event(ChatEvent(kind="text", content="preview"))
                    return AgentResult(events=[ChatEvent(kind="text", content="final")])

                with (
                    mock.patch.object(agent, "run", side_effect=run),
                    mock.patch.object(bot_base, "_STATUS_OPERATION_TIMEOUT_SEC", 0.01),
                    self.assertLogs("Cozter.performance", level="INFO") as trace,
                ):
                    await asyncio.wait_for(bot._run_turn("user", "group", "work"), timeout=0.5)
                thinking_trace = [line for line in trace.output if "stage=thinking_status " in line]
                # The foreground wait may finish one Windows clock tick
                # before the background failure has settled.
                self.assertTrue(any(f"outcome={outcome}" in thinking_trace[0]
                                    for outcome in ("pending", "no_handle")))
                await asyncio.sleep(0)
                self.assertEqual([params["message"] for _, _, params in bot.calls], [
                    "Thinking...", "final",
                ])
                self.assertEqual(bot.edits, 0)
                self.assertFalse(bot.deleted.is_set())

    async def test_stalled_edit_is_cancelled_without_delaying_final(self) -> None:
        bot = self._bot()
        bot.stall_edits = True

        async def run(*_args, on_event, **_kwargs):
            await on_event(ChatEvent(kind="tool", content="progress"))
            await asyncio.wait_for(bot.edit_events[0].wait(), timeout=0.5)
            return AgentResult(events=[ChatEvent(kind="text", content="final")])

        with mock.patch.object(agent, "run", side_effect=run):
            await asyncio.wait_for(bot._run_turn("user", "group", "work"), timeout=0.5)
        await asyncio.wait_for(bot.deleted.wait(), timeout=0.5)
        self.assertTrue(bot.edit_cancelled.is_set())
        self.assertIn("final", [params.get("message") for _, _, params in bot.calls])
        self.assertFalse(bot._send_lock.locked())

    async def test_cancelled_turn_cleans_status_and_pending_throttle(self) -> None:
        bot = self._bot(interval=60)
        running = asyncio.Event()

        async def run(*_args, on_event, **_kwargs):
            await on_event(ChatEvent(kind="tool", content="pending progress"))
            running.set()
            await asyncio.Event().wait()

        with mock.patch.object(agent, "run", side_effect=run):
            turn = asyncio.create_task(bot._run_turn("user", "group", "work"))
            await asyncio.wait_for(running.wait(), timeout=0.5)
            turn.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await turn
        await asyncio.wait_for(bot.deleted.wait(), timeout=0.5)
        self.assertEqual(bot.edits, 0)
        self.assertEqual(len(bot.calls), 2)

    async def test_scheduled_turn_uses_thinking_and_removes_ephemeral_session(self) -> None:
        bot = self._bot(interval=60)
        sessions = []

        async def run(*_args, on_event, session_id, **_kwargs):
            sessions.append(session_id)
            self.assertIsNotNone(session_id)
            self.assertEqual(bot.calls[1][2]["message"], "Thinking...")
            await on_event(ChatEvent(kind="tool", content="scheduled progress"))
            return AgentResult(events=[ChatEvent(kind="text", content="scheduled final")])

        with mock.patch.object(agent, "run", side_effect=run):
            await asyncio.wait_for(bot._run_ephemeral_turn(
                "user", "group", "scheduled work",
            ), timeout=1)
        await asyncio.wait_for(bot.deleted.wait(), timeout=0.5)
        messages = [params["message"] for _, method, params in bot.calls if method == "send"]
        self.assertEqual(messages, [
            "⏰ Starting scheduled task: scheduled work", "Thinking...", "scheduled final",
        ])
        self.assertEqual(len(sessions), 1)
        self.assertEqual(os.listdir(os.path.join(self.path, ".cozter", "sessions")), [])


if __name__ == "__main__":
    unittest.main()
