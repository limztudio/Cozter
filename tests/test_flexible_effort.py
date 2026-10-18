"""Flexible tiers capture independent effort without losing default inheritance."""

import asyncio
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Cozter import agent, execution_state, flexible, session, workspace
from Cozter.backends_agent.base import AgentResult
from Cozter.backends_bot.base import BotPlatform, COMMAND_NAMES
from Cozter.tests.helpers import TestBot
from Cozter.utils import json_digest


class FlexibleEffortSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = self.enterContext(tempfile.TemporaryDirectory())

    def test_inherits_workspace_or_explicit_caller_default_without_writing(self) -> None:
        workspace.set_reasoning_effort(self.ws, 35)
        path = Path(workspace._settings_path(self.ws))
        before = path.read_bytes()
        self.assertIsNone(workspace.get_flexible_reasoning_effort_override(self.ws, "low"))
        self.assertEqual(workspace.get_flexible_reasoning_efforts(self.ws), dict.fromkeys(flexible.TIERS, 35))
        self.assertEqual(workspace.get_flexible_reasoning_efforts(self.ws, default=70), dict.fromkeys(flexible.TIERS, 70))
        self.assertEqual(path.read_bytes(), before)

    def test_persists_separate_tiers_explicit_zero_and_reset(self) -> None:
        workspace.set_reasoning_effort(self.ws, 50)
        for tier, value in zip(flexible.TIERS, (0, 70, 100)):
            workspace.set_flexible_reasoning_effort(self.ws, tier, value)
        saved = json.loads(Path(workspace._settings_path(self.ws)).read_text(encoding="utf-8"))
        self.assertEqual([saved[f"flexible_{tier}_reasoning_effort"] for tier in flexible.TIERS], [0, 70, 100])
        self.assertEqual(workspace.get_flexible_reasoning_efforts(self.ws, default=25), {"low": 0, "mid": 70, "high": 100})
        workspace.set_flexible_reasoning_effort(self.ws, "mid", None)
        self.assertIsNone(workspace.get_flexible_reasoning_effort_override(self.ws, "mid"))
        self.assertEqual(workspace.get_flexible_reasoning_effort(self.ws, "mid"), 50)
        self.assertNotIn("flexible_mid_reasoning_effort", workspace._load_settings(self.ws))
        self.assertEqual(workspace.get_reasoning_effort(self.ws), 50)

    def test_snapshot_uses_one_settings_read_for_both_global_and_overrides(self) -> None:
        with mock.patch.object(workspace, "_load_settings", side_effect=[
            {"reasoning_effort": 40, "flexible_high_reasoning_effort": 100},
            {"reasoning_effort": 90},
        ]) as load:
            self.assertEqual(workspace.get_flexible_reasoning_efforts(self.ws), {"low": 40, "mid": 40, "high": 100})
        load.assert_called_once_with(self.ws)

    def test_corrupt_overrides_inherit_and_invalid_setters_reject(self) -> None:
        for bad in (True, False, 2.5, "80", [], {}, -1, 101):
            with self.subTest(bad=bad):
                workspace._set_setting(self.ws, "flexible_low_reasoning_effort", bad)
                self.assertIsNone(workspace.get_flexible_reasoning_effort_override(self.ws, "low"))
                self.assertEqual(workspace.get_flexible_reasoning_effort(self.ws, "low", default=65), 65)
        for bad in (True, False, 2.5, "80", []):
            with self.subTest(setter=bad), self.assertRaises(ValueError):
                workspace.set_flexible_reasoning_effort(self.ws, "low", bad)
        for method in (workspace.get_flexible_reasoning_effort_override, workspace.get_flexible_reasoning_effort):
            with self.assertRaises(ValueError):
                method(self.ws, "unknown")
        with self.assertRaises(ValueError):
            workspace.set_flexible_reasoning_effort(self.ws, "unknown", 25)
        workspace.set_flexible_reasoning_effort(self.ws, "low", 120)
        self.assertEqual(workspace.get_flexible_reasoning_effort_override(self.ws, "low"), 100)


class FlexibleEffortCommandTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.ws))
        self.enterContext(mock.patch.object(workspace, "WORKSPACE_STATE_PATH", str(Path(self.ws, "workspaces.json"))))
        self.bot = TestBot(["owner"])
        self.bot.send_text = mock.AsyncMock()
        workspace.select_workspace("owner", self.ws, self.bot.platform_id)

    def context(self, *, text="", command=None, args=""):
        return self.bot.make_context("owner", "room", text=text, command=command, args=args)

    async def test_registered_commands_support_direct_values_and_inheritance(self) -> None:
        workspace.set_reasoning_effort(self.ws, 60)
        for tier in flexible.TIERS:
            command = f"effort_flexible_{tier}"
            self.assertIn(command, COMMAND_NAMES)
            await self.bot.dispatch_command(self.context(command=command, args="0"))
            self.assertEqual(workspace.get_flexible_reasoning_effort_override(self.ws, tier), 0)
            self.assertIn("tier override", self.bot.send_text.call_args.args[1])
            await self.bot.dispatch_command(self.context(command=command, args="default"))
            self.assertIsNone(workspace.get_flexible_reasoning_effort_override(self.ws, tier))
            self.assertIn("60% (inherited", self.bot.send_text.call_args.args[1])

    async def test_picker_retries_invalid_input_and_cancel_does_not_change_setting(self) -> None:
        workspace.set_flexible_reasoning_effort(self.ws, "mid", 40)
        await self.bot.dispatch_command(self.context(command="effort_flexible_mid"))
        self.assertIn("40%", self.bot.send_text.call_args.args[1])
        for invalid in ("banana", "101", "-1", "false"):
            await self.bot.dispatch_text(self.context(text=invalid))
            self.assertIn("owner", self.bot._pending_input)
            self.assertEqual(workspace.get_flexible_reasoning_effort_override(self.ws, "mid"), 40)
        await self.bot.dispatch_text(self.context(text="75%"))
        self.assertEqual(workspace.get_flexible_reasoning_effort_override(self.ws, "mid"), 75)
        await self.bot.dispatch_command(self.context(command="effort_flexible_mid"))
        await self.bot.dispatch_command(self.context(command="cancel"))
        self.assertNotIn("owner", self.bot._pending_input)
        self.assertEqual(workspace.get_flexible_reasoning_effort_override(self.ws, "mid"), 75)

    async def test_summary_and_global_picker_describe_tier_effort_and_maximum(self) -> None:
        workspace.set_reasoning_effort(self.ws, 30)
        workspace.set_flexible_reasoning_effort(self.ws, "high", 100)
        with mock.patch.object(workspace, "get_flexible_run_config", return_value={tier: ("meta", "model") for tier in flexible.TIERS}):
            summary = BotPlatform._flexible_summary(self.ws)
        self.assertIn("30% (inherited", summary)
        self.assertIn("100% (tier override)", summary)
        self.assertIn("/effort_flexible_<tier>", summary)
        await self.bot.cmd_effort(self.context())
        text = self.bot.send_text.call_args.args[1]
        self.assertIn("100% selects the highest", text)
        self.assertNotIn("75-100 high", text)


class FlexibleEffortRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.plan = "[PLAN]\n1. [low] first\n2. [mid] second\n3. [high] third\n[/PLAN]"
        self.planner_calls = 0
        self.sid = session.create_session(self.ws, name="Effort routing")["id"]
        self.tiers = {tier: ("meta", "muse-spark-1.3") for tier in flexible.TIERS}
        self.enterContext(mock.patch.object(workspace, "get_flexible_run_config", side_effect=lambda _ws: self.tiers))

    async def internal(self, _backend, _ws, _prompt, _model, *, label, **_kwargs):
        if label == "Flexible planner":
            self.planner_calls += 1
            return self.plan
        return "Final merged response"

    async def invoke(self, drive, *, effort=70, job="job", implementation=False):
        async with execution_state.execution_scope(self.ws, job, "Original request"):
            execution_state.bind_conversation(self.sid)
            with (mock.patch.object(agent, "run_internal_backend", self.internal),
                  mock.patch.object(agent, "_drive_backend_impl" if implementation else "_drive_backend", drive)):
                return await agent._run_flexible(
                    "Original request", "Original request", self.ws, approval="full", effort=effort,
                    collaborative=False, summary_backend_name="meta", summary_model="summary",
                    on_event=None, inject_queue=None, injected=[],
                )

    async def test_routes_actual_workers_using_snapshot_with_explicit_zero(self) -> None:
        workspace.set_reasoning_effort(self.ws, 5)
        workspace.set_flexible_reasoning_effort(self.ws, "low", 0)
        workspace.set_flexible_reasoning_effort(self.ws, "high", 100)
        observed = []

        async def drive(_backend, _ws, prompt, *_args, effort, **_kwargs):
            observed.append(effort)
            # Mid-run edits affect the next invocation, never later workers
            # in an invocation whose checkpoint signature is already saved.
            workspace.set_flexible_reasoning_effort(self.ws, "high", 20)
            return AgentResult(text="Completed worker evidence"), False

        await self.invoke(drive, effort=70)
        self.assertEqual(observed, [0, 70, 100])

    async def test_changed_tier_policy_invalidates_completed_worker_checkpoint(self) -> None:
        calls = Counter()

        async def drive(_backend, _ws, prompt, *_args, **_kwargs):
            index = int(prompt.split("[Sub-task ")[1].split("/")[0])
            calls[index] += 1
            if index == 2 and calls[index] == 1:
                raise asyncio.CancelledError
            return AgentResult(text="Completed worker evidence"), False

        with self.assertRaises(asyncio.CancelledError):
            await self.invoke(drive)
        workspace.set_flexible_reasoning_effort(self.ws, "high", 100)
        await self.invoke(drive)
        self.assertEqual(self.planner_calls, 2)
        self.assertEqual(calls[1], 2)

    async def test_equal_explicit_policy_keeps_inherited_plan_and_completed_workers(self) -> None:
        calls = Counter()

        async def drive(_backend, _ws, prompt, *_args, **_kwargs):
            index = int(prompt.split("[Sub-task ")[1].split("/")[0])
            calls[index] += 1
            if index == 2 and calls[index] == 1:
                raise asyncio.CancelledError
            return AgentResult(text="Completed worker evidence"), False

        with self.assertRaises(asyncio.CancelledError):
            await self.invoke(drive)
        for tier in flexible.TIERS:
            workspace.set_flexible_reasoning_effort(self.ws, tier, 70)
        await self.invoke(drive, effort=10)
        self.assertEqual(self.planner_calls, 1)
        self.assertEqual(calls[1], 1)

    async def test_native_worker_state_identity_includes_each_tier_effort(self) -> None:
        self.tiers = {tier: ("codex", "gpt-6-sol") for tier in flexible.TIERS}
        for tier, value in zip(flexible.TIERS, (20, 60, 100)):
            workspace.set_flexible_reasoning_effort(self.ws, tier, value)
        observed = []

        async def implementation(_backend, _ws, _prompt, _model, _approval, *, effort, **_kwargs):
            state = execution_state.current_run()
            self.assertIsNotNone(state)
            identity = {"backend": "codex", "model": "gpt-6-sol", "approval": "full", "effort": effort, "worker": "main"}
            self.assertEqual(state.key, json_digest({
                "native_generation": state.get("native_record_id"), "backend": identity,
            }))
            observed.append((effort, state.key))
            return AgentResult(text="Completed worker evidence"), False

        await self.invoke(implementation, implementation=True)
        self.assertEqual([value[0] for value in observed], [20, 60, 100])
        self.assertEqual(len({value[1] for value in observed}), 3)


if __name__ == "__main__":
    unittest.main()
