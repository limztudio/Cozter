"""Folded guidance resumes the same Flexible role and native CLI UUID offline."""

import asyncio
import os
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from Cozter import agent, execution_state
from Cozter.backends_agent import claude_code, codex, copilot, grok
from Cozter.backends_agent.base import AgentResult


_NATIVE_ID = "01a126fa-e54a-7753-8078-3e31ff8fc962"
_CASES = (
    (codex, codex.CodexBackend, "gpt-6.1-sol"),
    (claude_code, claude_code.ClaudeCodeBackend, "opus"),
    (copilot, copilot.CopilotBackend, "fixture-model"),
    (grok, grok.GrokBackend, "grok-4.7"),
)
_PLAN = (
    "[UNDERSTANDING]\nFinish the requested target.\n[/UNDERSTANDING]\n"
    "[PLAN]\n1. [high] Finish and verify the remaining target.\n[/PLAN]"
)


class FlexibleNativeInjectionContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def test_budget_reload_keeps_all_four_native_sessions_after_guidance_fold(self):
        for module, backend_type, model in _CASES:
            with self.subTest(backend=backend_type.name), tempfile.TemporaryDirectory() as workspace:
                backend = backend_type()
                original = "Complete the offline fixture target."
                guidance = "Preserve the completed checks and finish the pending target."
                folded = original + "\n\n[Added while working]:\n" + guidance
                launches = []
                native_ids = []
                record_keys = []
                process = mock.Mock()
                factory = "create_captured_subprocess" if module is grok else "create_prompt_subprocess"
                summary = SimpleNamespace(name="summary")

                async def drive(real_backend, path, prompt, selected_model, approval, **kwargs):
                    # Keep Flexible's real worker scope, _drive_backend wrapper,
                    # native record and launch argv construction in this test.
                    record = execution_state.current_run()
                    self.assertIsNotNone(record)
                    record_keys.append(record.key)
                    await real_backend.launch(path, prompt, selected_model, approval,
                                              effort=kwargs["effort"])
                    native_id = record.get("native_session_id")
                    if module is copilot and native_id is None:
                        native_id = record.get("native_candidate_session_id")
                        (record.directory / "copilot-home" / "session-state" / native_id).mkdir(parents=True)
                    native_id = native_id or _NATIVE_ID
                    if module is codex:
                        event = {"type": "thread.started", "thread_id": native_id}
                    elif module is copilot:
                        event = {"type": "session.start", "data": {"sessionId": native_id}}
                    else:
                        event = {"type": "system", "subtype": "init", "session_id": native_id}
                    real_backend.parse_event(event, AgentResult())
                    await record.flush()
                    native_ids.append(record.get("native_session_id"))
                    await real_backend.cleanup_process(process)
                    if len(native_ids) == 1:
                        raise asyncio.CancelledError("synthetic allowance boundary")
                    return AgentResult(text="The requested target is complete and verified."), False

                async def create(argv, *args, **kwargs):
                    launches.append(list(argv))
                    return process

                def get_backend(name):
                    if name == "summary":
                        return summary
                    self.assertEqual(name, backend.name)
                    return backend

                async def run(request, injected):
                    return await agent._run_flexible(
                        request, request, workspace, approval="full", effort=100,
                        collaborative=False, summary_backend_name="summary",
                        summary_model="summary-model", on_event=None,
                        inject_queue=None, injected=injected,
                    )

                with (
                    mock.patch.dict(os.environ, {"COPILOT_HOME": workspace}),
                    mock.patch.object(agent.backends_agent, "get_backend", side_effect=get_backend),
                    mock.patch.object(agent.workspace_mod, "get_flexible_run_config", return_value={
                        tier: (backend.name, model) for tier in ("low", "mid", "high")
                    }),
                    mock.patch.object(agent, "run_internal_backend", new=mock.AsyncMock(side_effect=[
                        _PLAN, _PLAN.replace("Finish and verify", "Repeat the inspection and verify"),
                    ])) as planner,
                    mock.patch.object(agent, "_drive_backend_impl", side_effect=drive),
                    mock.patch.object(module, "executable_command", return_value=[backend.name]),
                    mock.patch.object(module, factory, side_effect=create),
                ):
                    async with execution_state.execution_scope(workspace, "durable-job", original):
                        with self.assertRaises(asyncio.CancelledError):
                            await run(original, [guidance])
                        saved = await execution_state.record("flexible_plan")
                        self.assertTrue(saved.get("active"))
                        self.assertEqual(saved.get("cursor"), 0)
                    # A distinct Execution object simulates the next durable
                    # queue allowance after BotPlatform folds accepted input.
                    async with execution_state.execution_scope(workspace, "durable-job", folded):
                        result, restarting = await run(folded, [])

                self.assertFalse(restarting)
                self.assertEqual(result.text, "The requested target is complete and verified.")
                self.assertEqual(planner.await_count, 1)
                self.assertEqual(len(launches), 2)
                self.assertEqual(native_ids[0], native_ids[1])
                self.assertEqual(record_keys[0], record_keys[1])
                flag = "resume" if module is codex else "--resume"
                self.assertNotIn(flag, launches[0])
                self.assertEqual(launches[1][launches[1].index(flag) + 1], native_ids[0])
                self.assertNotIn("--last", launches[1])
                self.assertNotIn("--continue", launches[1])
                expected_permission = {
                    "codex": "--dangerously-bypass-approvals-and-sandbox",
                    "claude_code": "--dangerously-skip-permissions",
                    "copilot": "--yolo", "grok": "--always-approve",
                }[backend.name]
                self.assertIn(expected_permission, launches[1])


if __name__ == "__main__":
    unittest.main()
