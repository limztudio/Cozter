"""Antigravity native receipts survive handoffs, failures and interruption."""

import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

from Cozter import agent, execution_state, flexible, session, skills
from Cozter.backends_agent import antigravity_cli
from Cozter.backends_agent.base import AgentResult
from Cozter.backends_agent.meta import MetaModelApiBackend
from Cozter.tests import test_native_delta_sync as delta_fixture
from Cozter.tests.test_flexible_owner_runtime import owner_plan


_REAL_DRIVER = agent._drive_backend_impl
_REAL_PROMPT = agent._build_backend_prompt


class AntigravityNativeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    _conversation = delta_fixture.NativeDeltaSyncTests._conversation
    _turn = delta_fixture.NativeDeltaSyncTests._turn
    _kill = delta_fixture.NativeDeltaSyncTests._kill
    _disk_values = staticmethod(delta_fixture.NativeDeltaSyncTests._disk_values)

    async def asyncSetUp(self):
        await delta_fixture.NativeDeltaSyncTests.asyncSetUp(self)
        self.home = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(skills.Path, "home", return_value=self.home))
        self.enterContext(mock.patch.dict(os.environ, {
            "CODEX_HOME": "", "CLAUDE_CONFIG_DIR": "", "COPILOT_HOME": "", "GROK_HOME": "",
            "COPILOT_SKILLS_DIRS": "",
        }))
        self.backend = antigravity_cli.AntigravityCliBackend()
        self.backends["antigravity_cli"] = self.backend
        self.models["antigravity_cli"] = self.backend.default_model
        self.modules["antigravity_cli"] = antigravity_cli
        self.backends["meta"] = MetaModelApiBackend()
        self.models["meta"] = self.backends["meta"].default_model
        self.backends["flexible"] = SimpleNamespace(name="flexible")
        self.models["flexible"] = "unused-flexible-model"
        self.tiers = {tier: ("antigravity_cli", self.backend.default_model) for tier in flexible.TIERS}
        self.tiers["mid"] = ("meta", self.models["meta"])
        self.enterContext(mock.patch.object(agent.workspace_mod, "get_flexible_run_config", side_effect=lambda _ws: self.tiers))
        self.enterContext(mock.patch.object(antigravity_cli, "_command_prefix", return_value=["agy"]))
        self.enterContext(mock.patch.object(antigravity_cli, "create_prompt_subprocess", side_effect=self._create_process))
        self.enterContext(mock.patch.object(antigravity_cli, "create_piped_subprocess", side_effect=self._create_piped_process))
        self.enterContext(mock.patch.object(antigravity_cli, "reap_failed_subprocess", side_effect=self._kill))
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
        self.initialization_receipts = []
        self.http_calls = []
        self.planner_prompts = []
        self.next_failure = None
        self.next_response = None
        self.truncate_stream = False
        self.next_init_mismatch = False

    async def _internal(self, _backend, _ws, prompt, _model, *, label, **_kwargs):
        if label == "Flexible planner":
            self.planner_prompts.append(prompt)
            return self.plan
        self.assertEqual(label, "Flexible merge")
        self.merges += 1
        return "UNSEEN-ANTIGRAVITY-MERGED-ANSWER"

    async def _create_process(self, command, wire, **kwargs):
        return await self._new_process(command, wire, piped=False, **kwargs)

    async def _create_piped_process(self, command, **kwargs):
        self.assertEqual(kwargs.pop("stdin"), asyncio.subprocess.PIPE)
        return await self._new_process(command, "", piped=True, **kwargs)

    async def _new_process(self, command, wire, *, piped, **kwargs):
        command = list(command)
        self.assertEqual(command[command.index("--input-format") + 1], "stream-json")
        self.assertEqual(command[command.index("--output-format") + 1], "stream-json")
        self.assertEqual(command[command.index("--print-timeout") + 1], "0")
        self.assertNotIn("--continue", command)
        record = execution_state.current_run()
        native_id = (command[command.index("--conversation") + 1]
                     if "--conversation" in command else str(uuid.uuid4()))
        if self.next_init_mismatch:
            self.assertTrue(piped)
            native_id = str(uuid.uuid4())
            self.next_init_mismatch = False
        launch = {
            "command": command, "prompt": None, "wire": "", "wire_bytes": bytearray(), "native_id": native_id,
            "previous_id": record.get("native_session_id"), "record": record,
            "scope": execution_state._WORKER.get(), "env": kwargs.get("env", {}),
            "failure": self.next_failure, "response": self.next_response,
            "piped": piped, "writes": [], "input_ready": asyncio.Event(),
        }
        self.next_failure = None
        self.next_response = None
        self.launches.append(launch)
        process = mock.Mock()
        process.fixture_launch = launch
        process.pid = 12000 + len(self.launches)
        process.returncode = None
        process.stderr = None
        if piped:
            process.stdout = asyncio.StreamReader()
            process.stdout.feed_data((json.dumps({
                "event": "init", "conversation_id": native_id, "init": {"cwd": self.ws},
            }) + "\n").encode("utf-8"))
            process.stdin = SimpleNamespace(
                write=lambda value: self._accept_wire(launch, value),
                drain=mock.AsyncMock(), close=mock.Mock(), wait_closed=mock.AsyncMock(),
                is_closing=lambda: False,
            )
        else:
            self._accept_wire(launch, wire.encode("utf-8"))
        self.processes.append(process)
        return process

    def _accept_wire(self, launch, value):
        launch["writes"].append(bytes(value))
        launch["wire_bytes"].extend(value)
        if not launch["wire_bytes"].endswith(b"\n"):
            return
        launch["wire"] = launch["wire_bytes"].decode("utf-8")
        message = json.loads(launch["wire"])
        self.assertEqual(message["event"], "user")
        launch["prompt"] = message["message"]["content"]
        self.assertIsInstance(launch["prompt"], str)
        launch["input_ready"].set()

    async def _drive(self, backend, *args, **kwargs):
        if backend.name == "antigravity_cli":
            return await _REAL_DRIVER(backend, *args, **kwargs)
        self.assertEqual(backend.name, "meta")
        scope = execution_state._WORKER.get()
        self.http_calls.append({"scope": scope, "prompt": args[1], "approval": args[3]})
        answer = "HTTP-HELPER-EVIDENCE-MARKER" if scope.startswith("flexible:lease:") else "HTTP-EXTERNAL-ANSWER-MARKER"
        return AgentResult(text=answer), False

    async def _events(self, process, **_kwargs):
        launch = process.fixture_launch
        native_id = launch["native_id"]
        failure = launch["failure"]
        if not launch["piped"]:
            yield {"event": "init", "conversation_id": native_id, "init": {
                "cwd": self.ws, "model": self.backend.default_model, "permission_mode": "always-proceed",
            }}
        await launch["input_ready"].wait()
        await launch["record"].flush()
        self.initialization_receipts.append(self._disk_values(launch["record"]))
        if not failure or failure.get("after_activity"):
            yield {"event": "step_update", "step_update": {
                "conversation_id": native_id, "step_index": 0, "state": "DONE", "step_type": "user_input",
            }}
            yield {"event": "step_update", "step_update": {
                "conversation_id": native_id, "step_index": 1, "state": "ACTIVE", "step_type": "tool",
                "tool_name": "run_command", "tool_info": {
                    "name": "run_command", "parameters": {"CommandLine": "git status --short"},
                },
            }}
            await launch["record"].flush()
            self.receipts.append(self._disk_values(launch["record"]))
            self.receipt_ready.set()
            if self.hold_after_activity:
                await asyncio.Event().wait()
        if failure:
            yield {"event": "result", "result": {
                "conversation_id": native_id, "status": "ERROR", "response": failure.get("response", ""),
                "error": failure["error"],
            }}
            process.returncode = 1
            return
        if self.parent_lease and launch["scope"] == "flexible:owner":
            self.parent_lease = False
            answer = (
                "Retain implementation and verification in this owner.\n[LEASE]\ntier: mid\n"
                "reason: Compare a focused invariant without transferring implementation\n"
                "task: Inspect the existing cache invariant and report evidence without changes\n[/LEASE]"
            )
        else:
            answer = launch["response"] or f"ANTIGRAVITY-ANSWER-MARKER-{len(self.launches)}"
        for index, fragment in enumerate(("INTERIM-RESPONSE-MARKER", "-SECOND-FRAGMENT")):
            yield {"event": "step_update", "step_update": {
                "conversation_id": native_id, "step_index": 2, "state": "ACTIVE" if index == 0 else "DONE",
                "step_type": "agent_response", "text_delta": fragment,
            }}
        if self.truncate_stream:
            process.returncode = 0
            return
        yield {"event": "result", "result": {
            "conversation_id": native_id, "status": "SUCCESS", "response": answer,
            "usage": {"input_tokens": 11, "output_tokens": 3, "cache_read_tokens": 7},
        }}
        process.returncode = 0

    async def test_http_interlude_and_flexible_helper_resume_exact_conversation_with_only_unseen_history(self):
        sid = self._conversation()
        await self._turn("antigravity_cli", "FIRST-ANTIGRAVITY-REQUEST-MARKER", sid=sid)
        first = self.launches[-1]
        self.assertIn("HISTORICAL-REQUEST-MARKER", first["prompt"])
        self.assertNotIn("--conversation", first["command"])
        self.assertIsNone(first["previous_id"])
        await self._turn("meta", "HTTP-INTERVENING-REQUEST-MARKER", sid=sid)
        self.parent_lease = True

        result = await self._turn("flexible", "FLEXIBLE-PARENT-REQUEST-MARKER", sid=sid)

        self.assertFalse(result.error)
        parent, resumed = self.launches[-2:]
        for launch in (parent, resumed):
            self.assertEqual(launch["previous_id"], first["native_id"])
            self.assertEqual(launch["command"][launch["command"].index("--conversation") + 1], first["native_id"])
        self.assertEqual(parent["record"].path, resumed["record"].path)
        self.assertEqual(parent["prompt"].count("HTTP-INTERVENING-REQUEST-MARKER"), 1)
        self.assertEqual(parent["prompt"].count("HTTP-EXTERNAL-ANSWER-MARKER"), 1)
        self.assertNotIn("HISTORICAL-REQUEST-MARKER", parent["prompt"])
        self.assertNotIn("FIRST-ANTIGRAVITY-REQUEST-MARKER", parent["prompt"])
        self.assertIn("HTTP-HELPER-EVIDENCE-MARKER", resumed["prompt"])
        self.assertNotIn("HTTP-INTERVENING-REQUEST-MARKER", resumed["prompt"])
        self.assertNotIn("FLEXIBLE-PARENT-REQUEST-MARKER", resumed["prompt"])
        self.assertEqual(self.http_calls[-1]["approval"], "confirm")
        self.assertNotIn("SYNTHETIC-PLUGIN-MARKER", self.http_calls[-1]["prompt"])
        self.assertIn("SYNTHETIC-PLUGIN-MARKER", first["prompt"])
        await self._turn("antigravity_cli", "NEXT-ANTIGRAVITY-REQUEST-MARKER", sid=sid)
        later = self.launches[-1]
        self.assertEqual(later["native_id"], first["native_id"])
        self.assertNotIn(result.text, later["prompt"])
        for marker in ("HISTORICAL-REQUEST-MARKER", "HTTP-INTERVENING-REQUEST-MARKER", "HTTP-EXTERNAL-ANSWER-MARKER"):
            self.assertNotIn(marker, later["prompt"])
        self.assertEqual(len(self.planner_prompts), 1)
        self.assertEqual(self.merges, 0)

    async def test_external_flexible_merge_is_supplied_once_while_native_final_reply_is_not_replayed(self):
        sid = self._conversation()
        await self._turn("antigravity_cli", "BEFORE-EXTERNAL-MERGE-MARKER", sid=sid)
        native_id = self.launches[-1]["native_id"]
        self.plan = "[PLAN]\n1. [low] Inspect the target.\n2. [high] Repair and verify it.\n[/PLAN]"

        merged = await self._turn("flexible", "EXTERNAL-MERGE-REQUEST-MARKER", sid=sid)

        self.assertEqual(merged.text, "UNSEEN-ANTIGRAVITY-MERGED-ANSWER")
        self.assertEqual(self.merges, 1)
        await self._turn("antigravity_cli", "AFTER-EXTERNAL-MERGE-MARKER", sid=sid)
        first = self.launches[-1]
        self.assertEqual(first["native_id"], native_id)
        self.assertEqual(first["prompt"].count(merged.text), 1)
        self.assertNotIn("EXTERNAL-MERGE-REQUEST-MARKER", first["prompt"])
        await self._turn("antigravity_cli", "AFTER-MERGE-SYNC-MARKER", sid=sid)
        self.assertEqual(self.launches[-1]["native_id"], native_id)
        self.assertNotIn(merged.text, self.launches[-1]["prompt"])

    async def test_skill_edits_and_deletion_refresh_each_role_without_resetting_native_owner(self):
        source = Path(self.ws, ".cozter", "skills", "workflow.md")
        native = self.home / ".gemini/antigravity-cli/skills/review/SKILL.md"

        def write(path, body):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "---\nname: " + path.parent.name + "\ndescription: Stable workflow metadata\n---\n" + body,
                encoding="utf-8",
            )

        write(source, "WORKSPACE-INSTRUCTIONS-V1")
        write(native, "NATIVE-INSTRUCTIONS-V1")
        initial = skills.discover(self.ws)
        first_source = next(entry for entry in initial.entries if entry.path == source.resolve())
        first_native = next(entry for entry in initial.entries if entry.path == native.resolve())
        sid = self._conversation()
        await self._turn("antigravity_cli", "BEFORE-SKILL-EDITS-MARKER", sid=sid)
        first = self.launches[-1]
        self.assertIn(first_source.revision, first["prompt"])
        self.assertIn(first_native.revision, first["prompt"])
        self.parent_lease = True
        helper_revisions = {}
        changed = False

        async def events(process, **kwargs):
            nonlocal changed
            async for event in self._events(process, **kwargs):
                if (not changed and process.fixture_launch["scope"] == "flexible:owner"
                        and event.get("event") == "result"):
                    changed = True
                    write(source, "WORKSPACE-INSTRUCTIONS-V2")
                    write(native, "NATIVE-INSTRUCTIONS-V2")
                yield event

        async def drive(backend, *args, **kwargs):
            if backend.name == "meta":
                catalog = skills.current_catalog(self.ws)
                self.assertTrue(execution_state._WORKER.get().startswith("flexible:lease:"))
                for previous, body in ((first_source, "WORKSPACE-INSTRUCTIONS-V2"),
                                       (first_native, "NATIVE-INSTRUCTIONS-V2")):
                    current = next(entry for entry in catalog.entries if entry.id == previous.id)
                    helper_revisions[current.id] = current.revision
                    self.assertNotEqual(current.revision, previous.revision)
                    self.assertIn(body, json.loads(skills.read_skill(self.ws, current.id))["content"])
                write(source, "WORKSPACE-INSTRUCTIONS-V3")
                native.unlink()
            return await self._drive(backend, *args, **kwargs)

        with (mock.patch.object(agent, "iter_process_json_events", side_effect=events),
              mock.patch.object(agent, "_drive_backend_impl", side_effect=drive)):
            result = await self._turn("flexible", "WORK-THROUGH-SKILL-EDITS-MARKER", sid=sid)

        self.assertFalse(result.error)
        self.assertEqual(len(helper_revisions), 2)
        parent, resumed = self.launches[-2:]
        latest = skills.discover(self.ws)
        latest_source = next(entry for entry in latest.entries if entry.id == first_source.id)
        self.assertNotEqual(latest_source.revision, helper_revisions[first_source.id])
        self.assertFalse(any(entry.id == first_native.id for entry in latest.entries))
        self.assertNotIn(first_source.revision, parent["prompt"])
        self.assertIn("[Updated skills catalog; replaces the previous version]", resumed["prompt"])
        self.assertIn(latest_source.revision, resumed["prompt"])
        for stale in (first_native.id, first_native.revision, helper_revisions[first_native.id],
                      "HISTORICAL-REQUEST-MARKER", "WORK-THROUGH-SKILL-EDITS-MARKER"):
            self.assertNotIn(stale, resumed["prompt"])
        self.assertIn("HTTP-HELPER-EVIDENCE-MARKER", resumed["prompt"])
        for launch in (parent, resumed):
            self.assertEqual(launch["native_id"], first["native_id"])
            self.assertEqual(launch["record"].path, first["record"].path)
            self.assertIn("re-read every selected skill", launch["prompt"])

        await self._turn("antigravity_cli", "AFTER-SKILL-EDITS-MARKER", sid=sid)
        later = self.launches[-1]
        self.assertEqual(later["native_id"], first["native_id"])
        self.assertIn("re-read every selected skill", later["prompt"])
        self.assertNotIn("[Available workspace and shared native skills]", later["prompt"])
        self.assertNotIn("WORK-THROUGH-SKILL-EDITS-MARKER", later["prompt"])
        self.assertEqual(len(self.planner_prompts), 1)

    async def test_unsupported_native_helper_cannot_launch_or_replace_existing_owner_lane(self):
        sid = self._conversation()
        await self._turn("antigravity_cli", "BEFORE-UNSUPPORTED-HELPER-MARKER", sid=sid)
        first = self.launches[-1]
        self.tiers["mid"] = ("antigravity_cli", self.backend.default_model)
        self.parent_lease = True

        result = await self._turn("flexible", "WORK-WITH-UNSUPPORTED-HELPER-MARKER", sid=sid)

        self.assertFalse(result.error)
        self.assertEqual(len(self.launches), 3)
        parent, resumed = self.launches[-2:]
        self.assertEqual(self.http_calls, [])
        self.assertIn("configured helper backend cannot enforce read-only work", resumed["prompt"])
        for launch in (parent, resumed):
            self.assertEqual(launch["scope"], "flexible:owner")
            self.assertEqual(launch["native_id"], first["native_id"])
            self.assertEqual(launch["record"].path, first["record"].path)
            self.assertIn("--dangerously-skip-permissions", launch["command"])
            self.assertNotIn("HISTORICAL-REQUEST-MARKER", launch["prompt"])
        await self._turn("antigravity_cli", "AFTER-UNSUPPORTED-HELPER-MARKER", sid=sid)
        self.assertEqual(self.launches[-1]["native_id"], first["native_id"])
        self.assertNotIn("WORK-WITH-UNSUPPORTED-HELPER-MARKER", self.launches[-1]["prompt"])

    async def test_cancel_after_native_acceptance_resumes_exact_id_without_replaying_original_work(self):
        sid = self._conversation()
        self.hold_after_activity = True
        task = asyncio.create_task(agent.run(
            "ANTIGRAVITY-INTERRUPTED-REQUEST-MARKER", self.ws, 7, model=self.backend.default_model,
            approval="full", backend_name="antigravity_cli", summary_backend_name="codex",
            summary_model="offline-summary", session_id=sid, execution_id="antigravity-interrupted",
        ))
        try:
            await asyncio.wait_for(self.receipt_ready.wait(), 5)
            first = self.launches[-1]
            ledger = self.receipts[-1].get("native_prompt_ledger", {})
            self.assertEqual(ledger.get("session_id"), first["native_id"])
            self.assertTrue(ledger.get("messages"))
            self.assertFalse(self.initialization_receipts[-1].get("native_prompt_ledger"))
            self.assertFalse(task.done())
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.hold_after_activity = False

        result = await agent.run(
            "ANTIGRAVITY-INTERRUPTED-REQUEST-MARKER", self.ws, 7, model=self.backend.default_model,
            approval="full", backend_name="antigravity_cli", summary_backend_name="codex",
            summary_model="offline-summary", session_id=sid, execution_id="antigravity-interrupted",
            original_request="ANTIGRAVITY-INTERRUPTED-REQUEST-MARKER",
        )

        self.assertFalse(result.error)
        resumed = self.launches[-1]
        self.assertEqual(resumed["native_id"], first["native_id"])
        self.assertEqual(resumed["command"][resumed["command"].index("--conversation") + 1], first["native_id"])
        self.assertIn("Continue the current job from this native conversation", resumed["prompt"])
        self.assertNotIn("ANTIGRAVITY-INTERRUPTED-REQUEST-MARKER", resumed["prompt"])
        self.assertNotIn("HISTORICAL-REQUEST-MARKER", resumed["prompt"])

    async def test_error_before_acceptance_keeps_unprocessed_history_for_explicit_retry(self):
        sid = self._conversation()
        self.next_failure = {"error": "Authentication required"}
        rejected = await self._turn("antigravity_cli", "REJECTED-REQUEST-MARKER", sid=sid)
        failed = self.launches[-1]
        self.assertTrue(rejected.error)
        self.assertEqual(len(self.launches), 1)
        self.assertFalse(self.initialization_receipts[-1].get("native_prompt_ledger"))
        self.assertFalse(failed["record"].get("native_resume_unavailable"))

        accepted = await self._turn("antigravity_cli", "EXPLICIT-RETRY-REQUEST-MARKER", sid=sid)

        self.assertFalse(accepted.error)
        retried = self.launches[-1]
        self.assertEqual(retried["native_id"], failed["native_id"])
        self.assertIn("HISTORICAL-REQUEST-MARKER", retried["prompt"])
        self.assertIn("HISTORICAL-ANSWER-MARKER", retried["prompt"])
        self.assertIn("EXPLICIT-RETRY-REQUEST-MARKER", retried["prompt"])

    async def test_provider_error_after_activity_preserves_receipt_without_automatic_replay(self):
        sid = self._conversation()
        await self._turn("antigravity_cli", "ANTIGRAVITY-SEED-REQUEST", sid=sid)
        first = self.launches[-1]
        self.next_failure = {"error": "Conversation unavailable while reading tool result", "after_activity": True}

        failed = await self._turn("antigravity_cli", "FAILED-AFTER-ACTIVITY-MARKER", sid=sid)

        self.assertTrue(failed.error)
        self.assertEqual(len(self.launches), 2)
        attempt = self.launches[-1]
        self.assertEqual(attempt["native_id"], first["native_id"])
        self.assertFalse(attempt["record"].get("native_resume_unavailable"))
        self.assertEqual(self.receipts[-1]["native_prompt_ledger"]["session_id"], first["native_id"])
        self.assertNotIn("HISTORICAL-REQUEST-MARKER", attempt["prompt"])
        await self._turn("antigravity_cli", "NEW-GUIDANCE-AFTER-FAILURE", sid=sid)
        self.assertNotIn("FAILED-AFTER-ACTIVITY-MARKER", self.launches[-1]["prompt"])
        self.assertEqual(self.launches[-1]["native_id"], first["native_id"])

    async def test_failed_native_reply_is_synced_once_as_external_session_evidence(self):
        sid = self._conversation()
        await self._turn("antigravity_cli", "BEFORE-PARTIAL-ERROR-MARKER", sid=sid)
        native_id = self.launches[-1]["native_id"]
        self.next_failure = {"error": "Provider stopped after partial work", "after_activity": True,
                             "response": "FAILED-NATIVE-REPLY-MARKER"}

        failed = await self._turn("flexible", "PARTIAL-ERROR-REQUEST-MARKER", sid=sid)

        self.assertTrue(failed.error)
        await self._turn("antigravity_cli", "AFTER-PARTIAL-ERROR-MARKER", sid=sid)
        resumed = self.launches[-1]
        self.assertEqual(resumed["native_id"], native_id)
        self.assertEqual(resumed["prompt"].count("FAILED-NATIVE-REPLY-MARKER"), 1)
        self.assertNotIn("PARTIAL-ERROR-REQUEST-MARKER", resumed["prompt"])
        await self._turn("antigravity_cli", "AFTER-PARTIAL-ERROR-SYNC-MARKER", sid=sid)
        self.assertNotIn("FAILED-NATIVE-REPLY-MARKER", self.launches[-1]["prompt"])

    async def test_final_response_corrects_streamed_fragments_without_duplicate_reply_or_history(self):
        sid = self._conversation()
        self.next_response = "AUTHORITATIVE-CORRECTED-ANSWER"

        result = await self._turn("antigravity_cli", "CORRECTION-REQUEST-MARKER", sid=sid)

        self.assertFalse(result.error)
        self.assertEqual(result.text, "AUTHORITATIVE-CORRECTED-ANSWER")
        self.assertNotIn("INTERIM-RESPONSE-MARKER", result.text)
        self.assertEqual(result.usage, {"input_tokens": 11, "output_tokens": 3})
        saved = session.load_session(self.ws, sid)
        self.assertEqual(saved["messages"][-1]["content"], "AUTHORITATIVE-CORRECTED-ANSWER")
        await self._turn("antigravity_cli", "AFTER-CORRECTION-REQUEST", sid=sid)
        self.assertNotIn("AUTHORITATIVE-CORRECTED-ANSWER", self.launches[-1]["prompt"])
        self.assertNotIn("INTERIM-RESPONSE-MARKER", self.launches[-1]["prompt"])

    async def test_clean_exit_without_terminal_result_preserves_partial_but_never_reports_completion(self):
        sid = self._conversation()
        self.truncate_stream = True

        result = await self._turn("antigravity_cli", "TRUNCATED-STREAM-REQUEST-MARKER", sid=sid)

        self.assertTrue(result.error)
        self.assertFalse(result.completion_verified)
        self.assertIn("INTERIM-RESPONSE-MARKER-SECOND-FRAGMENT", result.text)
        self.assertEqual(len(self.launches), 1)
        self.assertFalse(self.launches[-1]["record"].get("native_resume_unavailable"))

    async def test_mismatched_resume_init_is_killed_before_delta_delivery_then_bootstraps_once(self):
        sid = self._conversation()
        first_result = await self._turn("antigravity_cli", "BEFORE-MISSING-CONVERSATION-MARKER", sid=sid)
        first = self.launches[-1]
        self.next_init_mismatch = True

        recovered = await self._turn("antigravity_cli", "RESUME-MISSING-CONVERSATION-MARKER", sid=sid)

        self.assertFalse(recovered.error)
        self.assertEqual(len(self.launches), 3)
        rejected, replacement = self.launches[-2:]
        self.assertEqual(rejected["command"][rejected["command"].index("--conversation") + 1], first["native_id"])
        self.assertNotEqual(rejected["native_id"], first["native_id"])
        self.assertEqual(rejected["writes"], [])
        self.assertIsNone(rejected["prompt"])
        self.assertEqual(self.processes[-2].returncode, -9)
        self.assertFalse(rejected["input_ready"].is_set())
        self.assertNotIn("--conversation", replacement["command"])
        self.assertNotIn("--continue", replacement["command"])
        self.assertNotEqual(replacement["native_id"], first["native_id"])
        self.assertNotEqual(replacement["native_id"], rejected["native_id"])
        for marker in ("HISTORICAL-REQUEST-MARKER", "HISTORICAL-ANSWER-MARKER",
                       "BEFORE-MISSING-CONVERSATION-MARKER", first_result.text, "RESUME-MISSING-CONVERSATION-MARKER"):
            self.assertEqual(replacement["prompt"].count(marker), 1, marker)
        await self._turn("antigravity_cli", "AFTER-CONVERSATION-RECOVERY-MARKER", sid=sid)
        later = self.launches[-1]
        self.assertEqual(later["command"][later["command"].index("--conversation") + 1], replacement["native_id"])
        self.assertNotIn("HISTORICAL-REQUEST-MARKER", later["prompt"])
        self.assertNotIn("RESUME-MISSING-CONVERSATION-MARKER", later["prompt"])

    async def test_stale_trajectory_error_on_clean_completed_turn_succeeds_while_nonzero_exit_fails(self):
        sid = self._conversation()

        async def clean_recovered_events(process, **kwargs):
            async for event in self._events(process, **kwargs):
                if event.get("event") == "result":
                    event = {
                        "event": "result",
                        "result": {
                            **event["result"],
                            "status": "ERROR",
                            "error": "The stream was interrupted. Please continue the task you were working on.",
                        },
                    }
                yield event

        with mock.patch.object(agent, "iter_process_json_events", side_effect=clean_recovered_events):
            result = await self._turn("antigravity_cli", "RECOVERED-MID-TURN-ERROR-MARKER", sid=sid)

        self.assertFalse(result.error)
        self.assertEqual(result.text, "ANTIGRAVITY-ANSWER-MARKER-1")
        self.assertEqual(len(self.launches), 1)

        async def nonzero_error_events(process, **kwargs):
            async for event in clean_recovered_events(process, **kwargs):
                if event.get("event") == "result":
                    event["result"]["error"] = "Provider rejected final commit"
                yield event
            process.returncode = 1

        with mock.patch.object(agent, "iter_process_json_events", side_effect=nonzero_error_events):
            failed = await self._turn("antigravity_cli", "GENUINE-EXIT-ERROR-MARKER", sid=sid)

        self.assertTrue(failed.error)
        self.assertIn("Provider rejected final commit", failed.error)
        self.assertEqual(len(self.launches), 2)

    async def test_transient_stream_timeout_recovers_in_same_native_session_and_defers_when_repeated(self):
        sid = self._conversation()
        failures = [
            {
                "error": "agent executor error: generating and executing: request failed: stream idle timeout (5m0s): context deadline exceeded",
                "after_activity": True,
            },
            None,
        ]

        async def recover_once_events(process, **kwargs):
            process.fixture_launch["failure"] = failures.pop(0)
            async for event in self._events(process, **kwargs):
                yield event

        with mock.patch.object(agent, "iter_process_json_events", side_effect=recover_once_events):
            recovered = await self._turn("flexible", "TRANSIENT-TIMEOUT-RECOVERY-MARKER", sid=sid)

        self.assertFalse(recovered.error)
        self.assertEqual(len(self.launches), 2)
        first, second = self.launches
        self.assertEqual(first["native_id"], second["native_id"])
        self.assertEqual(second["command"][second["command"].index("--conversation") + 1], first["native_id"])
        self.assertIn("Automatic stream recovery", second["prompt"])
        self.assertIn("Continue the current job from this native conversation", second["prompt"])
        self.assertNotIn("TRANSIENT-TIMEOUT-RECOVERY-MARKER", second["prompt"])

        async def always_timeout_events(process, **kwargs):
            process.fixture_launch["failure"] = {
                "error": "API error (attempt 7): request failed: stream idle timeout (5m0s): context deadline exceeded",
                "after_activity": True,
            }
            async for event in self._events(process, **kwargs):
                yield event

        with mock.patch.object(agent, "iter_process_json_events", side_effect=always_timeout_events):
            deferred = await agent.run(
                "REPEATED-TIMEOUT-JOB-MARKER", self.ws, 7, model="unused-flexible-model",
                approval="full", backend_name="flexible", summary_backend_name="codex",
                summary_model="offline-summary", session_id=sid, execution_id="agy-repeated-timeout-job",
            )

        self.assertEqual(deferred.continuation_reason, "prerequisite")
        self.assertIn("stream idle timeout", deferred.error)
        self.assertEqual(len(self.launches), 5)
        async with execution_state.execution_scope(self.ws, "agy-repeated-timeout-job", "REPEATED-TIMEOUT-JOB-MARKER"):
            plan = await execution_state.record("flexible_plan")
            self.assertTrue(plan.get("active"))
            self.assertEqual(plan.get("cursor"), 0)
