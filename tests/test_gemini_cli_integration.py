"""Gemini keeps native receipts across HTTP handoffs and restricted helper lanes."""

import asyncio
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

from Cozter import agent, backends_agent, execution_state, flexible, workspace
from Cozter.backends_agent import gemini_cli
from Cozter.backends_agent.base import AgentResult
from Cozter.backends_agent.meta import MetaModelApiBackend
from Cozter.tests import test_native_delta_sync as delta_fixture
from Cozter.tests.helpers import OfflineBot, temporary_config
from Cozter.tests.test_flexible_owner_runtime import owner_plan


_REAL_DRIVER = agent._drive_backend_impl
_REAL_PROMPT = agent._build_backend_prompt


class GeminiNativeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    _conversation = delta_fixture.NativeDeltaSyncTests._conversation
    _turn = delta_fixture.NativeDeltaSyncTests._turn
    _kill = delta_fixture.NativeDeltaSyncTests._kill
    _disk_values = staticmethod(delta_fixture.NativeDeltaSyncTests._disk_values)

    async def asyncSetUp(self):
        await delta_fixture.NativeDeltaSyncTests.asyncSetUp(self)
        self.backend = gemini_cli.GeminiCliBackend()
        self.backends["gemini_cli"] = self.backend
        self.models["gemini_cli"] = self.backend.default_model
        self.modules["gemini_cli"] = gemini_cli
        self.backends["meta"] = MetaModelApiBackend()
        self.models["meta"] = self.backends["meta"].default_model
        self.backends["flexible"] = SimpleNamespace(name="flexible")
        self.models["flexible"] = "unused-flexible-model"
        self.tiers = {tier: ("gemini_cli", self.backend.default_model) for tier in flexible.TIERS}
        self.tiers["mid"] = ("meta", self.models["meta"])
        self.enterContext(mock.patch.object(agent.workspace_mod, "get_flexible_run_config", side_effect=lambda _ws: self.tiers))
        self.enterContext(mock.patch.object(gemini_cli, "executable_command", return_value=["gemini"]))
        self.enterContext(mock.patch.object(gemini_cli, "create_prompt_subprocess", side_effect=self._create_process))
        self.enterContext(mock.patch.object(gemini_cli, "_check_restricted_policy_available", return_value=None))
        self.enterContext(mock.patch.object(agent, "_drive_backend_impl", side_effect=self._drive))
        self.enterContext(mock.patch.object(agent, "finish_process_stderr", new=mock.AsyncMock(
            side_effect=lambda process, *_args, **_kwargs: process.fixture_launch.get("stderr", ""),
        )))
        self.enterContext(mock.patch.object(agent, "_build_backend_prompt", side_effect=_REAL_PROMPT))
        self.enterContext(mock.patch.object(agent.agent_tools, "cli_plugin_prelude", return_value=(
            "Plugins (bash): SYNTHETIC-PLUGIN-MARKER; python -m fixture_plugin '<JSON>'"
        )))
        self.plan = owner_plan(tier="high", specialists=[])
        self.parent_lease = False
        self.hold_after_activity = False
        self.receipt_ready = asyncio.Event()
        self.receipts = []
        self.http_calls = []
        self.planner_prompts = []
        self.next_exit_failure = None

    async def _internal(self, _backend, _ws, prompt, _model, *, label, **_kwargs):
        if label == "Flexible planner":
            self.planner_prompts.append(prompt)
            return self.plan
        self.assertEqual(label, "Flexible merge")
        self.merges += 1
        return "UNSEEN-GEMINI-MERGED-ANSWER"

    async def _create_process(self, command, prompt, **kwargs):
        command = list(command)
        record = execution_state.current_run()
        native_id = record.get("native_session_id") or record.get("native_candidate_session_id") or str(uuid.uuid4())
        policy = None
        if "--admin-policy" in command:
            policy = Path(command[command.index("--admin-policy") + 1]).read_text(encoding="utf-8")
        launch = {
            "command": command, "prompt": prompt, "native_id": native_id,
            "previous_id": record.get("native_session_id"), "record": record,
            "scope": execution_state._WORKER.get(), "policy": policy, "env": kwargs.get("env", {}),
            "exit_failure": self.next_exit_failure,
        }
        self.next_exit_failure = None
        if launch["exit_failure"]:
            launch["stderr"] = launch["exit_failure"]["stderr"]
        self.launches.append(launch)
        process = mock.Mock()
        process.fixture_launch = launch
        process.pid = 11000 + len(self.launches)
        process.returncode = None
        process.stderr = None
        self.processes.append(process)
        return process

    async def _drive(self, backend, *args, **kwargs):
        if backend.name == "gemini_cli":
            return await _REAL_DRIVER(backend, *args, **kwargs)
        self.assertEqual(backend.name, "meta")
        scope = execution_state._WORKER.get()
        self.http_calls.append({"scope": scope, "prompt": args[1], "approval": args[3]})
        answer = "HTTP-HELPER-EVIDENCE-MARKER" if scope.startswith("flexible:lease:") else "HTTP-EXTERNAL-ANSWER-MARKER"
        return AgentResult(text=answer), False

    async def _events(self, process, **_kwargs):
        launch = process.fixture_launch
        failure = launch["exit_failure"]
        if failure and not failure.get("after_activity"):
            process.returncode = failure["returncode"]
            return
        yield {"type": "init", "session_id": launch["native_id"], "model": self.backend.default_model}
        yield {"type": "tool_use", "tool_name": "read_file", "tool_id": "synthetic-read",
               "parameters": {"file_path": "existing-evidence.txt"}}
        self.receipts.append(self._disk_values(launch["record"]))
        self.receipt_ready.set()
        if failure:
            process.returncode = failure["returncode"]
            return
        if self.hold_after_activity:
            await asyncio.Event().wait()
        yield {"type": "tool_result", "tool_id": "synthetic-read", "status": "success", "output": "Existing proof"}
        if self.parent_lease and launch["scope"] == "flexible:owner":
            self.parent_lease = False
            answer = (
                "Retain the repair and verification in this owner.\n[LEASE]\ntier: mid\n"
                "reason: Compare a focused invariant without transferring implementation\n"
                "task: Inspect the existing cache invariant and report evidence without changes\n[/LEASE]"
            )
        else:
            answer = f"GEMINI-ANSWER-MARKER-{len(self.launches)}"
        yield {"type": "message", "role": "assistant", "content": answer, "delta": True}
        yield {"type": "result", "status": "success"}
        process.returncode = 0

    async def test_http_interlude_and_flexible_helper_resume_exact_gemini_uuid_with_only_unseen_history(self):
        sid = self._conversation()
        await self._turn("gemini_cli", "FIRST-GEMINI-REQUEST-MARKER", sid=sid)
        first = self.launches[-1]
        self.assertIn("HISTORICAL-REQUEST-MARKER", first["prompt"])
        await self._turn("meta", "HTTP-INTERVENING-REQUEST-MARKER", sid=sid)
        self.parent_lease = True

        result = await self._turn("flexible", "FLEXIBLE-PARENT-REQUEST-MARKER", sid=sid)

        self.assertFalse(result.error)
        parent, resumed = self.launches[-2:]
        self.assertEqual(parent["previous_id"], first["native_id"])
        self.assertEqual(resumed["previous_id"], first["native_id"])
        self.assertEqual(parent["record"].path, resumed["record"].path)
        self.assertEqual(parent["prompt"].count("HTTP-INTERVENING-REQUEST-MARKER"), 1)
        self.assertEqual(parent["prompt"].count("HTTP-EXTERNAL-ANSWER-MARKER"), 1)
        self.assertNotIn("HISTORICAL-REQUEST-MARKER", parent["prompt"])
        self.assertNotIn("FIRST-GEMINI-REQUEST-MARKER", parent["prompt"])
        self.assertIn("HTTP-HELPER-EVIDENCE-MARKER", resumed["prompt"])
        self.assertNotIn("HTTP-INTERVENING-REQUEST-MARKER", resumed["prompt"])
        self.assertNotIn("FLEXIBLE-PARENT-REQUEST-MARKER", resumed["prompt"])
        self.assertEqual(self.http_calls[-1]["approval"], "confirm")
        self.assertNotIn("SYNTHETIC-PLUGIN-MARKER", self.http_calls[-1]["prompt"])
        self.assertIn("SYNTHETIC-PLUGIN-MARKER", first["prompt"])
        await self._turn("gemini_cli", "NEXT-GEMINI-REQUEST-MARKER", sid=sid)
        later = self.launches[-1]
        self.assertEqual(later["previous_id"], first["native_id"])
        for marker in ("HISTORICAL-REQUEST-MARKER", "HTTP-INTERVENING-REQUEST-MARKER", "HTTP-EXTERNAL-ANSWER-MARKER"):
            self.assertNotIn(marker, later["prompt"])
        self.assertEqual(later["command"][later["command"].index("--resume") + 1], first["native_id"])

    async def test_interruption_after_receipt_reloads_exact_uuid_without_resending_accepted_request(self):
        sid = self._conversation()
        self.hold_after_activity = True
        task = asyncio.create_task(agent.run(
            "GEMINI-INTERRUPTED-REQUEST-MARKER", self.ws, 7, model=self.backend.default_model,
            approval="full", backend_name="gemini_cli", summary_backend_name="codex",
            summary_model="offline-summary", session_id=sid, execution_id="gemini-interrupted",
        ))
        try:
            await asyncio.wait_for(self.receipt_ready.wait(), 5)
            first = self.launches[-1]
            ledger = self.receipts[-1].get("native_prompt_ledger", {})
            self.assertEqual(ledger.get("session_id"), first["native_id"])
            self.assertTrue(ledger.get("messages"))
            self.assertFalse(task.done())
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.hold_after_activity = False

        result = await agent.run(
            "GEMINI-INTERRUPTED-REQUEST-MARKER", self.ws, 7, model=self.backend.default_model,
            approval="full", backend_name="gemini_cli", summary_backend_name="codex",
            summary_model="offline-summary", session_id=sid, execution_id="gemini-interrupted",
            original_request="GEMINI-INTERRUPTED-REQUEST-MARKER",
        )

        self.assertFalse(result.error)
        resumed = self.launches[-1]
        self.assertEqual(resumed["previous_id"], first["native_id"])
        self.assertIn("Continue the current job from this native conversation", resumed["prompt"])
        self.assertNotIn("GEMINI-INTERRUPTED-REQUEST-MARKER", resumed["prompt"])
        self.assertNotIn("HISTORICAL-REQUEST-MARKER", resumed["prompt"])

    async def test_initial_read_only_gemini_helper_has_separate_uuid_policy_and_plugin_shell_boundary(self):
        sid = self._conversation()
        self.plan = owner_plan(tier="low", specialists=[{
            "tier": "high", "reason": "Inspect the coupled invariant before edits",
            "task": "Read the existing invariant and report evidence without edits or commands",
        }])
        for tier in flexible.TIERS:
            workspace.set_flexible_reasoning_effort(self.ws, tier, 0)

        result = await self._turn("flexible", "READ-ONLY-HELPER-PARENT-MARKER", sid=sid)

        self.assertFalse(result.error)
        helper, owner = self.launches
        self.assertNotEqual(helper["native_id"], owner["native_id"])
        self.assertNotEqual(helper["record"].path, owner["record"].path)
        self.assertIn("--admin-policy", helper["command"])
        self.assertIsNotNone(helper["policy"])
        rules = tomllib.loads(helper["policy"])["rule"]
        catchall = next(rule for rule in rules if rule["toolName"] == "*" and "mcpName" not in rule)
        allow = next(rule for rule in rules if rule["decision"] == "allow")
        self.assertEqual(catchall["decision"], "deny")
        self.assertEqual(set(allow["toolName"]), {"read_file", "list_directory", "glob", "grep_search"})
        self.assertGreater(allow["priority"], catchall["priority"])
        self.assertTrue(any(rule.get("mcpName") == "*" and rule["decision"] == "deny" for rule in rules))
        self.assertTrue({"run_shell_command", "write_file", "replace", "exit_plan_mode", "fixture_plugin"}
                        .isdisjoint(allow["toolName"]))
        self.assertEqual(helper["command"][helper["command"].index("--approval-mode") + 1], "plan")
        self.assertFalse(Path(helper["command"][helper["command"].index("--admin-policy") + 1]).exists())
        self.assertNotIn("--yolo", helper["command"])
        self.assertNotIn("yolo", helper["command"])
        self.assertIn("yolo", owner["command"])
        self.assertEqual(owner["env"].get("GEMINI_SANDBOX"), "false")
        self.assertIn("SYNTHETIC-PLUGIN-MARKER", owner["prompt"])
        self.assertEqual(owner["prompt"].count("GEMINI-ANSWER-MARKER-1"), 1)
        self.assertEqual(len(self.planner_prompts), 1)
        self.assertEqual(self.merges, 0)

    async def test_exact_missing_session_stderr_before_work_bootstraps_original_scope_into_one_new_uuid(self):
        sid = self._conversation()
        first_result = await self._turn("gemini_cli", "GEMINI-BEFORE-MISSING-REQUEST", sid=sid)
        first = self.launches[-1]
        self.next_exit_failure = {
            "returncode": 42,
            "stderr": "Error resuming session: No previous sessions found for this project.\n",
        }

        recovered = await self._turn("gemini_cli", "GEMINI-MISSING-RESUME-REQUEST", sid=sid)

        self.assertFalse(recovered.error)
        self.assertEqual(len(self.launches), 3)
        failed, replacement = self.launches[-2:]
        self.assertEqual(failed["command"][failed["command"].index("--resume") + 1], first["native_id"])
        self.assertNotIn("HISTORICAL-REQUEST-MARKER", failed["prompt"])
        self.assertNotEqual(replacement["native_id"], first["native_id"])
        self.assertIn("--session-id", replacement["command"])
        self.assertNotIn("--resume", replacement["command"])
        self.assertNotIn("latest", replacement["command"])
        for marker in ("HISTORICAL-REQUEST-MARKER", "HISTORICAL-ANSWER-MARKER",
                       "GEMINI-BEFORE-MISSING-REQUEST", first_result.text, "GEMINI-MISSING-RESUME-REQUEST"):
            self.assertIn(marker, replacement["prompt"])
        await self._turn("gemini_cli", "GEMINI-AFTER-RECOVERY-REQUEST", sid=sid)
        self.assertEqual(self.launches[-1]["command"][self.launches[-1]["command"].index("--resume") + 1],
                         replacement["native_id"])
        self.assertNotIn("HISTORICAL-REQUEST-MARKER", self.launches[-1]["prompt"])

    async def test_authentication_or_missing_session_after_activity_never_replays_the_request(self):
        for failure in (
            {"returncode": 42, "stderr": "Error resuming session: Authentication failed.\n"},
            {"returncode": 42, "stderr": "Error resuming session: No previous sessions found for this project.\n",
             "after_activity": True},
        ):
            with self.subTest(failure=failure):
                sid = self._conversation()
                await self._turn("gemini_cli", "GEMINI-NO-REPLAY-SEED", sid=sid)
                first = self.launches[-1]
                count = len(self.launches)
                self.next_exit_failure = failure

                result = await self._turn("gemini_cli", "GEMINI-AMBIGUOUS-FAILURE-REQUEST", sid=sid)

                self.assertTrue(result.error)
                self.assertEqual(len(self.launches), count + 1)
                failed = self.launches[-1]
                self.assertEqual(failed["native_id"], first["native_id"])
                self.assertIn("--resume", failed["command"])
                self.assertNotIn("--session-id", failed["command"])
                self.assertFalse(failed["record"].get("native_resume_unavailable"))


class GeminiConfigurationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.ws))
        self.enterContext(mock.patch.object(workspace, "WORKSPACE_STATE_PATH", str(Path(self.ws, "workspaces.json"))))
        self.enterContext(temporary_config({"extra_models": {"gemini_cli": ["private-gemini-model"]}}))
        self.bot = OfflineBot(["owner"])
        self.bot.send_text = mock.AsyncMock()
        workspace.select_workspace("owner", self.ws, self.bot.platform_id)

    async def test_registered_cli_preserves_custom_model_and_reports_unsupported_effort_for_direct_and_flexible(self):
        backend = backends_agent.get_backend("gemini_cli")
        self.assertIn("gemini_cli", backends_agent.DIRECT_BACKENDS)
        workspace.set_backend_name(self.ws, "gemini_cli")
        workspace.set_summary_backend_name(self.ws, "gemini_cli")
        workspace.set_model(self.ws, "private-gemini-model")
        workspace.set_flexible_backend_name(self.ws, "high", "gemini_cli")
        self.assertEqual(workspace.get_summary_model(self.ws), backend.default_summary_model)
        self.assertEqual(workspace.get_flexible_model(self.ws, "high"), backend.tier_model("high"))
        self.assertIn("private-gemini-model", workspace.get_available_models(self.ws))
        with mock.patch("subprocess.run", side_effect=AssertionError("effort must not discover or launch")):
            for effort in (99, 100):
                context = self.bot.make_context("owner", "room", command="effort", args=str(effort))
                await self.bot.dispatch_command(context)
                reply = self.bot.send_text.call_args.args[1]
                self.assertIn("gemini_cli/private-gemini-model", reply)
                self.assertIn("no adjustable effort", reply)
                self.assertEqual(workspace.get_model(self.ws), "private-gemini-model")
            workspace.set_backend_name(self.ws, "flexible")
            workspace.set_flexible_model(self.ws, "high", "private-gemini-model")
            context = self.bot.make_context("owner", "room", command="effort_flexible_high", args="100")
            await self.bot.dispatch_command(context)
            self.assertIn("gemini_cli/private-gemini-model", self.bot.send_text.call_args.args[1])
            self.assertIn("no adjustable effort", self.bot.send_text.call_args.args[1])
            self.assertEqual(workspace.get_flexible_reasoning_effort(self.ws, "high"), 100)

