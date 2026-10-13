"""Tests for bot command handlers, driven through a capturing fake bot.

Exercises the real BotPlatform command logic (validation + state mutation
+ replies) without any chat platform: a concrete subclass captures
send_text, and each handler runs under asyncio.run against a temp
workspace. The global workspace-state file is redirected to the tempdir.
"""

import asyncio
from datetime import datetime
import os
import tempfile
import unittest
from unittest import mock

from Cozter import schedules, session, workspace
from Cozter.backends_bot.base import BotContext
from Cozter.backends_bot.signal import SignalBot
from Cozter.tests.helpers import TestBot


class _CommandBot(TestBot):
    def __init__(self) -> None:
        super().__init__(["u1"])
        self.replies: list[str] = []
        self.ai_texts: list[str] = []

    @property
    def platform_id(self) -> str:
        return "test:cmd"

    async def send_text(self, chat_id: str, text: str, *, rich: bool = False):
        self.replies.append(text)
        return None

    async def _ai_chat(self, ctx: BotContext) -> None:
        self.ai_texts.append(ctx.text)


class BotCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = self._tmp.name
        self._orig_state = workspace.WORKSPACE_STATE_PATH
        workspace.WORKSPACE_STATE_PATH = os.path.join(self.ws, "state.json")
        self.bot = _CommandBot()
        self.uid = "u1"
        workspace.select_workspace(self.uid, self.ws, self.bot.platform_id)

    def tearDown(self) -> None:
        workspace.WORKSPACE_STATE_PATH = self._orig_state
        self._tmp.cleanup()

    def _ctx(self, text: str = "", args: str = "") -> BotContext:
        return self.bot.make_context(
            self.uid,
            "c1",
            text=text,
            args=args,
        )

    def _last(self) -> str:
        return self.bot.replies[-1]

    def _run(self, coro) -> None:
        asyncio.run(coro)

    def test_restored_queue_shrinks_without_losing_backlog_or_identity(self) -> None:
        self.bot.max_queue_size = 2
        restored = self.bot._ensure_message_queue(self.uid, min_size=5)
        entries = [(f"prompt {index}", "c1", str(index), False) for index in range(5)]
        for entry in entries:
            restored.put_nowait(entry)

        current = self.bot._ensure_message_queue(self.uid)

        self.assertIs(current, restored)
        self.assertEqual(current.maxsize, 2)
        self.assertTrue(current.full())
        self.assertEqual([current.get_nowait() for _ in range(5)], entries)

    def test_queue_growth_wakes_a_producer_waiting_on_existing_queue(self) -> None:
        async def run() -> None:
            self.bot.max_queue_size = 1
            original = self.bot._ensure_message_queue(self.uid)
            first = ("first", "c1", "1", False)
            second = ("second", "c1", "2", False)
            original.put_nowait(first)
            producer = asyncio.create_task(original.put(second))
            try:
                await asyncio.sleep(0)
                self.assertFalse(producer.done())
                grown = self.bot._ensure_message_queue(self.uid, min_size=2)
                self.assertIs(grown, original)
                await asyncio.wait_for(producer, timeout=1)
                self.assertEqual([grown.get_nowait(), grown.get_nowait()], [first, second])
            finally:
                if not producer.done():
                    producer.cancel()
                    await asyncio.gather(producer, return_exceptions=True)

        self._run(run())

    # text command aliases
    def test_backslash_command_alias_dispatches_with_arguments(self) -> None:
        self._run(self.bot.dispatch_text(self._ctx(text=r"\context 8000")))

        self.assertEqual(workspace.get_history_budget(self.ws), 8000)
        self.assertEqual(self.bot.ai_texts, [])

    def test_unknown_backslash_text_remains_chat_input(self) -> None:
        text = r"\newcommand is LaTeX, not a Cozter command"
        self._run(self.bot.dispatch_text(self._ctx(text=text)))

        self.assertEqual(self.bot.ai_texts, [text])

    def test_unknown_slash_command_lists_available(self) -> None:
        ctx = self.bot.make_context(
            self.uid, "c1", text="/nope", command="nope", args="",
        )
        self._run(self.bot.dispatch_command(ctx))

        self.assertIn("Unknown command: /nope", self._last())
        self.assertIn("/cancel", self._last())
        self.assertIn("/doctor", self._last())
        self.assertEqual(self.bot.ai_texts, [])

    def test_backslash_cancel_exits_pending_input_flow(self) -> None:
        self._run(self.bot.cmd_open(self._ctx()))
        self.assertIn(self.uid, self.bot._pending_input)

        self._run(self.bot.dispatch_text(self._ctx(text=r"\cancel")))

        self.assertNotIn(self.uid, self.bot._pending_input)
        self.assertEqual(self._last(), "Cancelled.")

    # /permission
    def test_permission_flow_sets_value(self) -> None:
        self._run(self.bot.cmd_permission(self._ctx()))
        self.assertIn("Available modes", self._last())
        self._run(self.bot._receive_permission(self._ctx(text="deny")))
        self.assertEqual(workspace.get_permission(self.ws), "deny")
        self.assertIn("deny", self._last())

    def test_permission_rejects_unknown(self) -> None:
        self._run(self.bot._receive_permission(self._ctx(text="bogus")))
        self.assertIn("Unknown mode", self._last())
        self.assertEqual(workspace.get_permission(self.ws), "auto")  # default

    # /effort
    def test_effort_rejects_out_of_range(self) -> None:
        self._run(self.bot._receive_effort(self._ctx(text="500")))
        self.assertIn("Out of range", self._last())

    def test_effort_rejects_overlong_numeric_input(self) -> None:
        self._run(self.bot._receive_effort(self._ctx(text="9" * 5_000)))
        self.assertIn("Not a number", self._last())

    # /context
    def test_numeric_commands_reject_overlong_decimal_input(self) -> None:
        too_long = "9" * 5_000

        for handler in (
            self.bot.cmd_compact,
            self.bot.cmd_context,
            self.bot.cmd_colony,
        ):
            with self.subTest(handler=handler.__name__):
                self._run(handler(self._ctx(args=too_long)))
                self.assertEqual(self._last(), "Error: number is too large.")

    # /reserve
    def test_reserve_wizard_binds_days_and_time_across_callbacks(self) -> None:
        self._run(self.bot.cmd_reserve(self._ctx()))
        self._run(self.bot.dispatch_text(self._ctx(text="mon,wed")))
        self.assertIn("Enter time", self._last())

        self._run(self.bot.dispatch_text(self._ctx(text="09:30")))
        self.assertIn("Enter the command", self._last())

        self._run(self.bot.dispatch_text(self._ctx(text="run report")))

        self.assertIn("Schedule created", self._last())
        entries = schedules.list_schedules(self.ws, self.uid)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["days"], ["mon", "wed"])
        self.assertEqual(entries[0]["time"], "09:30")
        self.assertEqual(entries[0]["command"], "run report")
        self.assertEqual(entries[0]["platform_id"], self.bot.platform_id)

    def test_scheduler_preserves_other_platforms_schedule(self) -> None:
        for schedule_id, platform_id in (
            ("foreign", "another:bot"), ("own", self.bot.platform_id), ("legacy", None),
        ):
            schedule = {
                "id": schedule_id, "days": list(schedules.DAY_ABBREV),
                "time": "00:00", "created": "2000-01-01T00:00:00",
                "chat_id": "c1", "command": "run",
            }
            if platform_id is not None:
                schedule["platform_id"] = platform_id
            schedules.add_schedule(self.ws, self.uid, schedule)

        with mock.patch.object(self.bot, "_fire_schedule", new=mock.AsyncMock()) as fire:
            self._run(self.bot._scheduler_tick())

        self.assertEqual({call.args[1]["id"] for call in fire.await_args_list}, {"own", "legacy"})
        self.assertNotIn("last_fired", schedules.list_schedules(self.ws, self.uid)[0])

    def test_schedule_picker_lists_and_deletes_only_this_platforms_entries(self):
        for schedule_id, platform_id in (("foreign", "another:bot"), ("own", self.bot.platform_id)):
            schedules.add_schedule(self.ws, self.uid, {
                "id": schedule_id, "platform_id": platform_id,
                "days": ["mon"], "time": "09:00", "command": schedule_id,
            })

        self._run(self.bot.cmd_schedules(self._ctx()))
        self.assertNotIn("foreign", self._last())
        self._run(self.bot._receive_schedules(self._ctx(text="1")))
        self.assertEqual([item["id"] for item in schedules.list_schedules(self.ws, self.uid)], ["foreign"])

    def test_concurrent_scheduler_ticks_claim_a_due_slot_once(self):
        schedules.add_schedule(self.ws, self.uid, {
            "id": "once", "days": list(schedules.DAY_ABBREV),
            "time": "00:00", "created": "2000-01-01T00:00:00",
            "chat_id": "c1", "command": "run",
        })

        async def run():
            lock = asyncio.Lock()
            await lock.acquire()
            with (
                mock.patch.object(workspace, "get_lock", return_value=lock),
                mock.patch.object(self.bot, "_fire_schedule", new=mock.AsyncMock()) as fire,
            ):
                ticks = [asyncio.create_task(self.bot._scheduler_tick()) for _ in range(2)]
                await asyncio.sleep(0)
                lock.release()
                await asyncio.gather(*ticks)
                fire.assert_awaited_once()

        self._run(run())

    def test_scheduler_rechecks_edited_time_after_waiting_for_workspace_lock(self) -> None:
        async def run(updated_time: str, expected_slot: str | None) -> None:
            schedules._save_all(self.ws, {self.uid: [{
                "id": "edited", "days": list(schedules.DAY_ABBREV),
                "time": "08:30", "created": "2026-10-12T08:00:00",
                "chat_id": "c1", "command": "run",
            }]})
            lock = asyncio.Lock()
            await lock.acquire()
            with (
                mock.patch.object(workspace, "get_lock", return_value=lock),
                mock.patch("Cozter.backends_bot.base.datetime") as clock,
                mock.patch.object(self.bot, "_fire_schedule", new=mock.AsyncMock()) as fire,
            ):
                clock.now.return_value = datetime(2026, 10, 12, 9, 0)
                tick = asyncio.create_task(self.bot._scheduler_tick())
                try:
                    await asyncio.sleep(0)
                    self.assertFalse(tick.done())
                    current = schedules._load_all(self.ws)
                    current[self.uid][0]["time"] = updated_time
                    schedules._save_all(self.ws, current)
                    lock.release()
                    await asyncio.wait_for(tick, timeout=1)
                finally:
                    if lock.locked():
                        lock.release()
                    if not tick.done():
                        tick.cancel()
                        await asyncio.gather(tick, return_exceptions=True)
                if expected_slot is None:
                    fire.assert_not_awaited()
                    self.assertNotIn("last_fired", schedules.list_schedules(self.ws, self.uid)[0])
                else:
                    fire.assert_awaited_once()
                    self.assertEqual(fire.await_args.args[1]["last_fired"], expected_slot)

        for updated_time, expected_slot in (
            ("10:00", None),
            ("08:45", "2026-10-12T08:45:00"),
        ):
            with self.subTest(updated_time=updated_time):
                self._run(run(updated_time, expected_slot))

    def test_signal_legacy_adoption_preserves_other_platforms_schedules(self):
        bot = SignalBot(["https://signal.group/#test"], jsonrpc_socket="/tmp/signal.sock")
        bot._group_ids = {"group"}
        target_uid = "signal-group:group"
        workspace.select_workspace(target_uid, self.ws, bot.platform_id)
        schedules.add_schedule(self.ws, self.uid, {"id": "foreign", "platform_id": self.bot.platform_id})
        schedules.add_schedule(self.ws, self.uid, {"id": "legacy"})

        moved = bot._migrate_legacy_workspace_schedules(self.ws, target_uid, "group")

        self.assertEqual(moved, 1)
        self.assertEqual([record["id"] for record in schedules.list_schedules(self.ws, self.uid)], ["foreign"])
        migrated = schedules.list_schedules(self.ws, target_uid)
        self.assertEqual(migrated[0]["id"], "legacy")
        self.assertEqual(migrated[0]["platform_id"], "signal")
        self.assertEqual(migrated[0]["chat_id"], "group")

    # /agent
    def test_agent_picker_reserves_zero_for_flexible(self) -> None:
        self._run(self.bot.cmd_agent(self._ctx()))

        out = self._last()
        self.assertIn("  0. flexible", out)
        self.assertIn("  1. codex", out)

        self._run(self.bot._receive_agent(self._ctx(text="0")))
        self.assertEqual(workspace.get_backend_name(self.ws), "flexible")

    def test_flexible_model_picker_uses_selected_tier_backend(self) -> None:
        workspace.set_flexible_backend_name(self.ws, "high", "claude_code")

        self._run(self.bot._receive_flexible_model(
            self._ctx(text="opus"), tier="high",
        ))

        self.assertEqual(
            workspace.get_flexible_model(self.ws, "high"), "opus",
        )
        self.assertIn("Flexible high model set to: opus", self._last())

    def test_invalid_model_choices_reprompt_without_crashing(self) -> None:
        cases = (
            (
                self.bot._receive_model(self._ctx(text="missing")),
                "get_available_models",
            ),
            (
                self.bot._receive_summary_model(self._ctx(text="missing")),
                "get_available_summary_models",
            ),
            (
                self.bot._receive_flexible_model(
                    self._ctx(text="missing"), tier="high",
                ),
                "get_available_flexible_models",
            ),
        )

        for handler, fetcher in cases:
            with self.subTest(fetcher=fetcher), mock.patch.object(
                workspace, fetcher, return_value=["known-model"],
            ):
                self._run(handler)
                self.assertIn("Unknown model: missing", self._last())
                self.assertIn(self.uid, self.bot._pending_input)

    # /sessions
    def test_sessions_list_and_switch(self) -> None:
        first = session.create_session(self.ws, name="First")
        session.create_session(self.ws, name="Second")
        # Bare command lists them.
        self._run(self.bot.cmd_sessions(self._ctx()))
        listing = self._last()
        self.assertIn("First", listing)
        self.assertIn("Second", listing)
        # Switch by name.
        self._run(self.bot.cmd_sessions(self._ctx(args="First")))
        self.assertIn("Switched to session: First", self._last())
        self.assertEqual(
            session.get_last_session(self.ws, self.uid), first["id"],
        )

    def test_no_workspace_replies_gracefully(self) -> None:
        # No workspace: expect the no-workspace message, not a crash.
        ctx = self.bot.make_context("nobody", "c1")
        self._run(self.bot.cmd_permission(ctx))
        self.assertTrue(self.bot.replies)  # replied rather than raised


if __name__ == "__main__":
    unittest.main()
