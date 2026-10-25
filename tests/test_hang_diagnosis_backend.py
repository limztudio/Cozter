"""The diagnostic model has local reads and no original execution ownership."""

import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent_tools, backends_agent, config, execution_state, process_observation, supervision, utils
from Cozter.backends_agent import _openai_agent as oa
from Cozter.tests.helpers import OfflineHttpBackend
from Cozter.tests.test_http_execution_continuation import make_tool_call


def _decision(verdict="uncertain", **changes):
    return {"verdict": verdict, "evidence": "source/log evidence", "diagnosis": "inspection conclusion",
            "repair": "preserve existing work and inspect the relevant source", "progress_paths": []} | changes


class HangDiagnosisBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_hidden_confirmation_diagnosis_reads_children_and_settles_without_changing_job_state(self):
        payloads = []
        settled = asyncio.Event()

        async def stream(_endpoint, payload, *_args, **_kwargs):
            payloads.append(copy.deepcopy(payload))
            if len(payloads) == 1:
                prompt = payload["messages"][1]["content"]
                self.assertIn("hidden interactive input", prompt)
                self.assertIn("matching target state", prompt)
                self.assertIn("Do not repeat the same inconclusive checks", prompt)
                self.assertIn("Previous review did not inspect directory children", prompt)
                return "", "", [make_tool_call("inspect-parent", "list_dir", "crashes")]
            if len(payloads) == 2:
                self.assertEqual(payload["messages"][-1]["content"], "packages/")
                return "", "", [make_tool_call("inspect-children", "list_dir", "crashes/packages")]
            self.assertEqual(payload["messages"][-1]["content"], "evidence.nwbcrashpkg")
            return json.dumps(_decision(
                "confirmed_hang", evidence="crashes/packages contains evidence.nwbcrashpkg; command omits -Recurse",
                diagnosis="Remove-Item waits for a confirmation that this unattended shell cannot receive.",
                repair="Preserve the crash package; inspect exact scope and use -NonInteractive without automatic deletion.",
            )), "", []

        async def work():
            try:
                await asyncio.Event().wait()
            finally:
                settled.set()

        with tempfile.TemporaryDirectory() as workspace:
            crash = Path(workspace, "crashes", "packages", "evidence.nwbcrashpkg")
            crash.parent.mkdir(parents=True)
            crash.write_bytes(b"preserved crash evidence")
            identity = process_observation.ProcessIdentity(123, "fixture:owned")
            ticket = supervision.Operation(supervision.Worker(workspace, "offline", "model-a"), "native")
            ticket.command = "Remove-Item -Force -ErrorAction SilentlyContinue crashes/*"
            ticket.process = SimpleNamespace(pid=123)
            ticket.identity = identity
            ticket.snapshot = process_observation.ProcessSnapshot(identity, ())
            ticket.progress_at -= 60
            ticket.review_history = [{"verdict": "uncertain", "evidence": "Previous review did not inspect directory children"}]
            with (
                mock.patch.object(backends_agent, "get_backend", return_value=OfflineHttpBackend()),
                mock.patch.object(oa, "_stream_completion", side_effect=stream),
                mock.patch.object(supervision, "_observe", new=mock.AsyncMock()),
                mock.patch.object(config, "get_hang_check_interval_seconds", return_value=.001),
                mock.patch.object(config, "get_hang_diagnose_after_seconds", return_value=.001),
                mock.patch.object(config, "get_hang_review_interval_seconds", return_value=0),
            ):
                async with execution_state.execution_scope(workspace, "same-job", "Finish remaining target"):
                    plan = await execution_state.record("flexible_plan")
                    await plan.save(active=True, cursor=1, completed=["verified target A"])
                    async with execution_state.backend_scope("codex", "model", "full", 0, native=True) as record:
                        await record.save(native_session_id="67f170b2-d850-4661-a99a-aa7eda414504")
                        original_record, original_plan = record.path.read_bytes(), plan.path.read_bytes()
                        task = asyncio.create_task(work())
                        try:
                            result = await asyncio.wait_for(supervision.wait_owned(task, ticket), 5)
                        finally:
                            if not task.done():
                                task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                        self.assertTrue(task.cancelled())
                        self.assertTrue(settled.is_set())
                        self.assertIn("confirmation", result)
                        self.assertEqual(record.path.read_bytes(), original_record)
                        self.assertEqual(plan.path.read_bytes(), original_plan)
            self.assertEqual(crash.read_bytes(), b"preserved crash evidence")
            self.assertEqual(len(payloads), 3)

    async def test_diagnostic_cannot_expand_into_original_workflow_or_continue_tool_segments(self):
        class LongWorkerBackend(OfflineHttpBackend):
            supports_vision = True

            def _max_agent_turns(self):
                return 100

            def _max_segments(self):
                return 100

            def _auto_continue_after_tool_limit(self):
                return True

        payloads = []
        events = []

        async def stream(_endpoint, payload, *_args, **_kwargs):
            payloads.append(copy.deepcopy(payload))
            if "tools" in payload:
                return "", "", [make_tool_call(str(len(payloads)), path=f"file-{len(payloads)}.txt")]
            return json.dumps(_decision()), "", []

        with (
            tempfile.TemporaryDirectory() as workspace,
            supervision.diagnostic_scope(),
            agent_tools.skill_planning_scope(),
            agent_tools.diagnostic_reading_scope(),
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
            mock.patch.object(oa.tools, "execute_tool", new=mock.AsyncMock(return_value="local evidence")) as execute,
            mock.patch.object(oa, "_vision_parts_for_prompt", side_effect=AssertionError("No diagnostic vision expansion")),
        ):
            await LongWorkerBackend()._run_agent(
                SimpleNamespace(emit=events.append), workspace,
                "Diagnose supplied evidence only. attachment saved to: evidence.png", "model", "confirm", False, 20,
            )
        self.assertEqual(len(payloads), 9)
        self.assertEqual(execute.await_count, 8)
        self.assertNotIn("tools", payloads[-1])
        prompt = payloads[0]["messages"][0]["content"]
        for worker_rule in (oa.WHOLE_SCOPE_RULE, oa.DOC_FOLLOWING_RULE, oa.PROGRESS_RULE, oa.VISION_RULE):
            self.assertNotIn(worker_rule, prompt)
        self.assertIn("strict JSON diagnosis", prompt)
        self.assertEqual({row["function"]["name"] for row in payloads[0]["tools"]}, agent_tools.DIAGNOSTIC_READ_TOOL_NAMES)
        answers = [event["text"] for event in events if event.get("type") == "assistant_text"]
        self.assertEqual(json.loads(answers[-1])["verdict"], "uncertain")

    async def test_diagnostic_launch_is_read_only_suspended_and_cannot_recursively_supervise(self):
        with tempfile.TemporaryDirectory() as workspace:
            ticket = supervision.Operation(supervision.Worker(workspace, "offline", "model-a"), "tool:bash")
            ticket.output = "original output"
            proc = SimpleNamespace(returncode=0)

            async def launch(path, _prompt, model, *, approval, compaction, effort):
                self.assertEqual((path, model, approval, compaction), (workspace, "model-a", "confirm", False))
                self.assertEqual(effort, 20)
                self.assertTrue(agent_tools.diagnostic_reading_active())
                self.assertIsNone(execution_state.current_run())
                self.assertIsNone(supervision.new_operation(workspace, "generation"))
                supervision.note_output("doctor-private output")
                schema = oa._tools_for_approval(approval, compaction)
                self.assertEqual({row["function"]["name"] for row in schema}, agent_tools.DIAGNOSTIC_READ_TOOL_NAMES)

                async def inherited_checks():
                    self.assertTrue(agent_tools.diagnostic_reading_active())
                    self.assertIsNone(execution_state.current_run())
                    self.assertIsNone(supervision.new_operation(workspace, "tool:read_file"))

                await asyncio.create_task(inherited_checks())
                return proc

            backend = SimpleNamespace(launch=mock.AsyncMock(side_effect=launch))
            with (
                mock.patch.object(backends_agent, "get_backend", return_value=backend),
                mock.patch.object(utils, "drain_llm_subprocess", new=mock.AsyncMock(return_value=json.dumps(_decision()))),
                mock.patch.object(utils, "cleanup_backend_process", new=mock.AsyncMock()) as cleanup,
                supervision.worker_scope(workspace, summary_backend_name="offline", summary_model="model-a"),
                supervision.operation_scope(ticket),
            ):
                async with execution_state.execution_scope(workspace, "original-job", "original request"):
                    async with execution_state.backend_scope("original", "model", "full", 99) as record:
                        await record.save(evidence="retained")
                        original = Path(record.path).read_bytes()
                        result = await supervision._diagnose(ticket, {"observation": "original evidence"}, 1)
                        self.assertEqual(result["verdict"], "uncertain")
                        self.assertIs(execution_state.current_run(), record)
                        self.assertEqual(Path(record.path).read_bytes(), original)
                self.assertFalse(agent_tools.diagnostic_reading_active())
                self.assertIsNotNone(supervision.new_operation(workspace, "generation"))
                self.assertEqual(ticket.output, "original output")
                cleanup.assert_awaited_once_with(backend, proc, log=supervision.logger)

    async def test_diagnostic_deadline_settles_its_process_and_restores_context(self):
        entered = asyncio.Event()
        drained = asyncio.Event()
        proc = SimpleNamespace(returncode=None)

        async def drain(*_args, **_kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()

        async def kill(process):
            self.assertIs(process, proc)
            process.returncode = 130

        backend = SimpleNamespace(launch=mock.AsyncMock(return_value=proc))
        with tempfile.TemporaryDirectory() as workspace:
            ticket = supervision.Operation(supervision.Worker(workspace, "offline", "model-a"), "generation")
            with (
                mock.patch.object(backends_agent, "get_backend", return_value=backend),
                mock.patch.object(utils, "drain_llm_subprocess", side_effect=drain),
                mock.patch.object(utils, "kill_and_wait", side_effect=kill) as reap,
                mock.patch.object(utils, "cleanup_backend_process", new=mock.AsyncMock()) as cleanup,
            ):
                with self.assertRaises(TimeoutError):
                    await supervision._diagnose(ticket, {}, .02)
                self.assertTrue(entered.is_set())
                self.assertTrue(drained.is_set())
                reap.assert_awaited_once_with(proc)
                cleanup.assert_awaited_once_with(backend, proc, log=supervision.logger)
            self.assertFalse(agent_tools.diagnostic_reading_active())

    async def test_model_launch_failure_does_not_change_original_operation_or_permissions(self):
        backend = SimpleNamespace(launch=mock.AsyncMock(side_effect=RuntimeError("model unavailable")))
        with tempfile.TemporaryDirectory() as workspace:
            ticket = supervision.Operation(supervision.Worker(workspace, "offline", "model-a"), "tool:bash")
            with (
                mock.patch.object(backends_agent, "get_backend", return_value=backend),
                mock.patch.object(utils, "cleanup_backend_process", new=mock.AsyncMock()) as cleanup,
                supervision.operation_scope(ticket),
            ):
                with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                    await supervision._diagnose(ticket, {}, 1)
                self.assertFalse(agent_tools.diagnostic_reading_active())
                supervision.note_output("original work still observable")
                self.assertIn("original work still observable", ticket.output)
                cleanup.assert_not_awaited()


class HangDiagnosisParserTests(unittest.TestCase):
    def test_structured_decision_requires_a_causal_explanation_before_recovery(self):
        for raw in (
            "not JSON", "[]", json.dumps({"verdict": "done"}),
            json.dumps(_decision("confirmed_hang", evidence="")),
            json.dumps(_decision("confirmed_hang", diagnosis="  ")),
            json.dumps(_decision("confirmed_hang", repair=None)),
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                supervision._parse_diagnosis(raw)

    def test_fenced_decision_retains_bounded_inspection_paths(self):
        value = _decision("progressing", progress_paths=["progress.log", 42] + ["other.log"] * 20)
        result = supervision._parse_diagnosis("```json\n" + json.dumps(value) + "\n```")
        self.assertEqual(result["verdict"], "progressing")
        self.assertEqual(result["progress_paths"][0], "progress.log")
        self.assertLessEqual(len(result["progress_paths"]), 16)
        self.assertTrue(all(isinstance(path, str) for path in result["progress_paths"]))


if __name__ == "__main__":
    unittest.main()
