"""Regression coverage for audit findings in launcher and orchestration."""

import re
import tempfile
import unittest
from unittest import mock

from Cozter import agent, flexible, workspace
from Cozter.backends_agent.base import AgentResult, append_text_result
from Cozter.tests.test_runtime_diagnostics import _load_main_module


class LauncherExitTests(unittest.TestCase):
    def test_cli_crash_returns_a_nonzero_exit_status(self):
        main = _load_main_module()

        def crash(coro):
            coro.close()
            raise RuntimeError("fixture startup problem")

        with (
            mock.patch.object(main, "setup_logging"),
            mock.patch.object(main, "_enable_faulthandler"),
            mock.patch.object(main, "_cli_mode_requested", return_value=True),
            mock.patch.object(main.asyncio, "run", side_effect=crash),
            mock.patch.object(main, "log_crash", return_value="fixture.log") as log,
            self.assertLogs(main.logger, level="CRITICAL"),
            self.assertRaises(SystemExit) as exit_status,
        ):
            main.run()

        self.assertEqual(exit_status.exception.code, 1)
        log.assert_called_once()


class FollowupCoverageTests(unittest.IsolatedAsyncioTestCase):
    def test_parser_keeps_all_reported_obligations(self):
        instructions = [f"fix issue {i}" for i in range(flexible.MAX_SUBTASKS + 3)]
        parsed = flexible.parse_followups("\n".join(
            f"[followup:low] {instruction}" for instruction in instructions
        ))
        self.assertEqual([task.instruction for task in parsed], instructions)

    async def test_queue_limit_keeps_unrun_tasks_in_the_final_reply(self):
        calls = []
        reported = set()
        executed = set()

        async def internal(*_args, **kwargs):
            if kwargs["label"] == "Flexible planner":
                return "[PLAN]\n1. [low] original task\n[/PLAN]"
            return "Completed the available tasks."

        async def drive(*_args, **_kwargs):
            calls.append(1)
            if task := re.search(r"\nYour task: issue (\d+)\n", _args[2]):
                executed.add(int(task[1]))
            result = AgentResult()
            text = "Reviewed."
            if len(calls) <= 8:
                # Alternating tiers cannot be combined into one worker without
                # changing providers or dependency order. Eight reports also
                # exceed the global queue cap, independently of the per-report cap.
                issues = range((len(calls) - 1) * 27, len(calls) * 27)
                reported.update(issues)
                text += "\n" + "\n".join(
                    f"[followup:{flexible.TIERS[i % 3]}] issue {i}" for i in issues
                )
            append_text_result(result, text)
            return result, False

        # Skill prep writes .cozter/ into the workspace, so keep it out of cwd.
        with (
            tempfile.TemporaryDirectory() as ws,
            mock.patch.object(agent, "run_internal_backend", side_effect=internal),
            mock.patch.object(agent, "_drive_backend", side_effect=drive),
            mock.patch.object(workspace, "get_flexible_run_config", return_value={
                tier: ("codex", "fixture-model") for tier in flexible.TIERS
            }),
        ):
            result, restarting = await agent._run_flexible(
                "context", "request", ws, approval="auto", effort=0,
                collaborative=False, summary_backend_name="codex",
                summary_model="fixture-model", on_event=None,
                inject_queue=None, injected=[],
            )

        self.assertFalse(restarting)
        self.assertEqual(len(calls), flexible.MAX_SUBTASKS * 2)
        self.assertIn("PARTIAL", result.text)
        self.assertIn("these tasks were not run", result.text)
        self.assertEqual(len(reported), 216)
        self.assertEqual(len(executed), 23)
        remainder = {line for line in result.text.splitlines() if line.startswith("- [")}
        self.assertEqual(remainder, {
            f"- [{flexible.TIERS[issue % 3]}] issue {issue}" for issue in reported - executed
        })
