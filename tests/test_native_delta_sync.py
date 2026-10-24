"""Real native event loops persist receipts before completion and sync only gaps."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

from Cozter import agent, execution_state, flexible, session
from Cozter.backends_agent import claude_code, codex, copilot
from Cozter.tests import test_native_session_turns as native_turns


_REAL_DRIVE = agent._drive_backend_impl
_CASES = native_turns._CASES


class NativeDeltaSyncTests(unittest.IsolatedAsyncioTestCase):
    _started = staticmethod(native_turns.NativeSessionTurnsTests._started)
    _answer = staticmethod(native_turns.NativeSessionTurnsTests._answer)
    _missing = staticmethod(native_turns.NativeSessionTurnsTests._missing)
    _conversation = native_turns.NativeSessionTurnsTests._conversation
    _turn = native_turns.NativeSessionTurnsTests._turn
    _drive = native_turns.NativeSessionTurnsTests._drive

    async def asyncSetUp(self):
        await native_turns.NativeSessionTurnsTests.asyncSetUp(self)
        # Keep the real driver, provider parsers, native scopes and disk writes.
        # Only subprocess creation, byte transport and OS teardown are fixtures.
        self.enterContext(mock.patch.object(agent, "_drive_backend_impl", new=_REAL_DRIVE))
        self.enterContext(mock.patch.object(agent, "iter_process_json_events", side_effect=self._events))
        self.enterContext(mock.patch.object(agent, "drain_text_stream", new=mock.AsyncMock(return_value="")))
        self.enterContext(mock.patch.object(agent, "wait_for_process_exit", new=mock.AsyncMock()))
        self.enterContext(mock.patch.object(agent, "kill_and_wait", side_effect=self._kill))
        self.enterContext(mock.patch.object(agent, "finish_process_stderr", new=mock.AsyncMock(return_value="")))
        self.enterContext(mock.patch.object(agent, "has_managed_process_group", return_value=False))
        self.enterContext(mock.patch.object(agent, "terminate_process_group_async", side_effect=self._kill))
        self.mode = "complete"
        self.receipt_ready = asyncio.Event()
        self.release_activity = asyncio.Event()
        self.initialization_receipts = []
        self.activity_receipts = []
        self.processes = []
        self.backends["flexible"] = SimpleNamespace(name="flexible")
        self.models["flexible"] = "unused-flexible-model"
        self.worker_name = "codex"
        self.plan = "[PLAN]\n1. [low] Inspect the target.\n2. [high] Repair and verify it.\n[/PLAN]"
        self.merges = 0
        self.enterContext(mock.patch.object(
            agent.workspace_mod, "get_flexible_run_config", side_effect=lambda _ws: {
                tier: (self.worker_name, self.models[self.worker_name]) for tier in flexible.TIERS
            },
        ))
        self.enterContext(mock.patch.object(agent, "run_internal_backend", side_effect=self._internal))

    @staticmethod
    def _disk_values(record):
        return json.loads(record.path.read_text(encoding="utf-8"))["values"]

    async def _internal(self, _backend, _ws, _prompt, _model, *, label, **_kwargs):
        if label == "Flexible planner":
            return self.plan
        self.assertEqual(label, "Flexible merge")
        self.merges += 1
        return f"UNSEEN-MERGED-ANSWER-{self.merges}"

    async def _create_process(self, command, *args, **kwargs):
        command = list(command)
        prompt = (Path(command[command.index("--prompt-file") + 1]).read_text(encoding="utf-8")
                  if command[0] == "grok" else args[0])
        record = execution_state.current_run()
        native_id = record.get("native_session_id") or record.get("native_candidate_session_id") or str(uuid.uuid4())
        if command[0] == "copilot":
            (record.directory / "copilot-home" / "session-state" / native_id).mkdir(parents=True, exist_ok=True)
        launch = {"backend": command[0], "command": command, "prompt": prompt,
                  "native_id": native_id, "previous_id": record.get("native_session_id"),
                  "home": kwargs.get("env", {}).get("COPILOT_HOME"), "record": record}
        self.launches.append(launch)
        process = mock.Mock()
        process.fixture_launch = launch
        process.pid = 10000 + len(self.launches)
        process.returncode = None
        process.stderr = None
        self.processes.append(process)
        return process

    async def _kill(self, process):
        process.returncode = -9

    @staticmethod
    def _activity(module):
        if module is codex:
            return {"type": "turn.started"}
        if module is copilot:
            return {"type": "assistant.message_delta", "data": {"deltaContent": ""}}
        if module is claude_code:
            return {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "status"}]}}
        return {"type": "user", "message": {"content": []}}

    async def _events(self, process, **_kwargs):
        launch = process.fixture_launch
        module = self.modules[launch["backend"]]
        record = launch["record"]
        if self.mode == "stderr-missing-session":
            self.mode = "complete"
            process.returncode = 1
            return
        yield self._started(module, launch["native_id"])
        await record.flush()
        self.initialization_receipts.append(self._disk_values(record))
        if self.mode == "error-before-work":
            event = self._missing(module, launch["native_id"])
            yield json.loads(json.dumps(event).replace(
                "No session found with ID: " + launch["native_id"], "Provider rejected the request before work.",
            ))
            process.returncode = 1
            return
        # These provider activity events deliberately emit no ChatEvent.
        # Receipts must follow accepted work, not require visible output.
        yield self._activity(module)
        await record.flush()
        self.activity_receipts.append(self._disk_values(record))
        self.receipt_ready.set()
        if self.mode == "hold-after-activity":
            await self.release_activity.wait()
        yield self._answer(module, "COMPLETED-NATIVE-ANSWER")
        process.returncode = 0

    async def test_first_native_activity_is_durable_before_completion_and_reload_for_all_four(self):
        for module, cls, _model in _CASES:
            with self.subTest(backend=cls.name):
                sid = self._conversation()
                self.mode = "hold-after-activity"
                self.receipt_ready = asyncio.Event()
                self.release_activity = asyncio.Event()
                task = asyncio.create_task(agent.run(
                    "LONG-RUNNING-REQUEST", self.ws, 7, model=self.models[cls.name], approval="full",
                    backend_name=cls.name, summary_backend_name="codex", summary_model="offline-summary",
                    session_id=sid, execution_id="forced-reload-" + cls.name,
                ))
                try:
                    await asyncio.wait_for(self.receipt_ready.wait(), 5)
                    self.assertFalse(task.done())
                    first_launch = self.launches[-1]
                    receipt = self.activity_receipts[-1].get("native_prompt_ledger", {})
                    self.assertEqual(receipt.get("session_id"), first_launch["native_id"])
                    self.assertTrue(receipt.get("sections"))
                    self.assertTrue(receipt.get("messages"))
                    initial = self.initialization_receipts[-1]
                    self.assertFalse(initial.get("native_prompt_ledger"))
                finally:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                self.mode = "complete"
                # agent.run constructs another Execution and reloads the saved
                # pin/receipt as a recovered queue job would after daemon loss.
                await agent.run(
                    "LONG-RUNNING-REQUEST", self.ws, 7, model=self.models[cls.name], approval="full",
                    backend_name=cls.name, summary_backend_name="codex", summary_model="offline-summary",
                    session_id=sid, execution_id="forced-reload-" + cls.name,
                    original_request="LONG-RUNNING-REQUEST",
                )
                returned = self.launches[-1]
                self.assertEqual(returned["previous_id"], first_launch["native_id"])
                self.assertEqual(returned["native_id"], first_launch["native_id"])
                self.assertIn("Continue the current job from this native conversation", returned["prompt"])
                self.assertIn("Interrupted turn", returned["prompt"])
                self.assertNotIn("LONG-RUNNING-REQUEST", returned["prompt"])
                for marker in ("HISTORICAL-REQUEST-MARKER", "HISTORICAL-ANSWER-MARKER", "POLICY-PRELUDE-MARKER"):
                    self.assertNotIn(marker, returned["prompt"])
                flag = "resume" if module is codex else "--resume"
                self.assertEqual(returned["command"][returned["command"].index(flag) + 1], first_launch["native_id"])
                if module is copilot:
                    self.assertEqual(returned["home"], first_launch["home"])

    async def test_initialization_and_prework_error_never_receipt_unprocessed_history_for_all_four(self):
        for _module, cls, _model in _CASES:
            with self.subTest(backend=cls.name):
                sid = self._conversation()
                self.mode = "error-before-work"
                result = await self._turn(cls.name, "REJECTED-REQUEST", sid=sid)
                rejected = self.launches[-1]
                self.assertTrue(result.error)
                self.assertFalse(self.initialization_receipts[-1].get("native_prompt_ledger"))
                self.assertFalse(self._disk_values(rejected["record"]).get("native_prompt_ledger"))
                self.mode = "complete"
                await self._turn(cls.name, "FIRST-ACTUAL-WORK", sid=sid)
                accepted = self.launches[-1]
                self.assertEqual(accepted["previous_id"], rejected["native_id"])
                for marker in ("HISTORICAL-REQUEST-MARKER", "HISTORICAL-ANSWER-MARKER", "POLICY-PRELUDE-MARKER"):
                    self.assertIn(marker, accepted["prompt"])

    async def test_flexible_skips_delivered_user_request_but_supplies_unseen_merged_reply_for_all_four(self):
        for module, cls, _model in _CASES:
            with self.subTest(backend=cls.name):
                sid = self._conversation()
                self.worker_name = cls.name
                self.plan = "[PLAN]\n1. [low] Inspect the target.\n2. [high] Repair and verify it.\n[/PLAN]"
                merged = await self._turn("flexible", "ALREADY-DELIVERED-FLEXIBLE-REQUEST", sid=sid)
                native_id = self.launches[-1]["native_id"]
                first_home = self.launches[-1]["home"]
                self.plan = "[PLAN]\n1. [mid] Handle the next assigned task.\n[/PLAN]"
                await self._turn("flexible", "NEXT-FLEXIBLE-REQUEST", sid=sid)
                returned = self.launches[-1]
                self.assertEqual(returned["previous_id"], native_id)
                self.assertEqual(returned["prompt"].count(merged.text), 1)
                self.assertNotIn("ALREADY-DELIVERED-FLEXIBLE-REQUEST", returned["prompt"])
                self.assertNotIn("HISTORICAL-REQUEST-MARKER", returned["prompt"])
                self.assertIn("NEXT-FLEXIBLE-REQUEST", returned["prompt"])
                await self._turn("flexible", "THIRD-FLEXIBLE-REQUEST", sid=sid)
                later = self.launches[-1]
                self.assertEqual(later["previous_id"], native_id)
                self.assertNotIn(merged.text, later["prompt"])
                self.assertNotIn("NEXT-FLEXIBLE-REQUEST", later["prompt"])
                if module is copilot:
                    self.assertEqual(first_home, later["home"])

    async def test_flexible_log_acknowledges_exact_user_before_a_detached_completion(self):
        sid = self._conversation()
        persist = agent._persist_session_write
        appended = False

        async def append_after_log(workspace_path, write):
            nonlocal appended
            result = await persist(workspace_path, write)
            if not appended:
                appended = True
                async with agent.workspace_mod.get_lock(workspace_path):
                    session.append_messages(workspace_path, sid, [
                        {"role": "user", "content": "UNSEEN-DETACHED-REQUEST"},
                        {"role": "assistant", "content": "UNSEEN-DETACHED-ANSWER"},
                    ])
            return result

        with mock.patch.object(agent, "_persist_session_write", side_effect=append_after_log):
            merged = await self._turn("flexible", "DELIVERED-FOREGROUND-REQUEST", sid=sid)
        native_id = self.launches[-1]["native_id"]
        self.plan = "[PLAN]\n1. [high] Continue after detached completion.\n[/PLAN]"
        await self._turn("flexible", "NEXT-FOREGROUND-REQUEST", sid=sid)
        returned = self.launches[-1]
        self.assertTrue(appended)
        self.assertEqual(returned["previous_id"], native_id)
        self.assertNotIn("DELIVERED-FOREGROUND-REQUEST", returned["prompt"])
        for marker in (merged.text, "UNSEEN-DETACHED-REQUEST", "UNSEEN-DETACHED-ANSWER"):
            self.assertEqual(returned["prompt"].count(marker), 1)

    async def test_stderr_missing_session_before_events_retries_once_with_full_bootstrap_for_all_four(self):
        for module, cls, _model in _CASES:
            with self.subTest(backend=cls.name):
                sid = self._conversation()
                await self._turn(cls.name, "BEFORE-STDERR-MISSING-REQUEST", sid=sid)
                original_id = self.launches[-1]["native_id"]
                start = len(self.launches)
                self.mode = "stderr-missing-session"
                stderr_msg = (
                    f"Error: thread/resume: thread/resume failed: no rollout found for thread id {original_id} (code -32600)"
                    if module is codex else
                    f"Error: No session, task, or name matched '{original_id}'.\nTo resume: copilot --resume=<id>"
                    if module is copilot else
                    f"No conversation found with session ID: {original_id}"
                )
                with mock.patch.object(agent, "finish_process_stderr", new=mock.AsyncMock(return_value=stderr_msg)):
                    result = await self._turn(cls.name, "RECOVERED-AFTER-STDERR-MISSING", sid=sid)
                self.assertFalse(result.error)
                launches = self.launches[start:]
                self.assertEqual(len(launches), 2)
                self.assertEqual(launches[0]["previous_id"], original_id)
                self.assertIsNone(launches[1]["previous_id"])
                self.assertNotEqual(launches[1]["native_id"], original_id)
                for marker in (
                    "HISTORICAL-REQUEST-MARKER", "HISTORICAL-ANSWER-MARKER",
                    "BEFORE-STDERR-MISSING-REQUEST", "RECOVERED-AFTER-STDERR-MISSING",
                ):
                    self.assertIn(marker, launches[1]["prompt"])


if __name__ == "__main__":
    unittest.main()
