"""Diagnosed tool hangs settle ownership and return evidence to the agent."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent_tools, supervision
from Cozter.agent_tools.builtin import bash, grep
from Cozter.backends_agent import _openai_agent as oa
from Cozter.utils import close_subprocess_pipe, kill_and_wait


class ToolHangRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_diagnostic_reading_permits_local_reads_but_blocks_all_other_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "evidence.txt").write_text("retained evidence", encoding="utf-8")
            with agent_tools.skill_planning_scope(), agent_tools.diagnostic_reading_scope():
                self.assertTrue(agent_tools.diagnostic_reading_active())
                result = await agent_tools.execute_tool(
                    "read_file", {"path": "evidence.txt"}, workspace, "full", lambda _event: None,
                )
                self.assertIn("retained evidence", result)
                for name in ("write_file", "bash", "web_search", "web_fetch", "list_skills",
                             "read_skill", "tree", "send_message", "custom_plugin"):
                    with self.subTest(name=name):
                        execute = mock.AsyncMock(return_value="mutation must never run")
                        tool = SimpleNamespace(name=name, run=execute, file_action=None, is_plugin=False)
                        with mock.patch.dict(agent_tools.TOOLS_BY_NAME, {name: tool}):
                            result = await agent_tools.execute_tool(
                                name, {}, workspace, "full", lambda _event: None,
                            )
                        self.assertTrue(result.startswith("Blocked:"))
                        execute.assert_not_awaited()
                execute = mock.AsyncMock(return_value="shadowed plugin must never run")
                shadow = SimpleNamespace(name="read_file", run=execute, file_action=None, is_plugin=True)
                with mock.patch.dict(agent_tools.TOOLS_BY_NAME, {"read_file": shadow}):
                    result = await agent_tools.execute_tool(
                        "read_file", {}, workspace, "full", lambda _event: None,
                    )
                self.assertTrue(result.startswith("Blocked:"))
                execute.assert_not_awaited()
        self.assertFalse(agent_tools.diagnostic_reading_active())
        self.assertEqual({row["function"]["name"] for row in agent_tools.DIAGNOSTIC_READ_TOOL_SCHEMA},
                         agent_tools.DIAGNOSTIC_READ_TOOL_NAMES)

    async def test_diagnostic_scope_is_inherited_by_children_and_restored_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            execute = mock.AsyncMock(return_value="normal tool result")
            tool = SimpleNamespace(name="write_file", run=execute, file_action=None, is_plugin=False)
            with mock.patch.dict(agent_tools.TOOLS_BY_NAME, {"write_file": tool}):
                with self.assertRaisesRegex(RuntimeError, "diagnostic failure"):
                    with agent_tools.diagnostic_reading_scope():
                        child = asyncio.create_task(agent_tools.execute_tool(
                            "write_file", {}, workspace, "full", lambda _event: None,
                        ))
                        self.assertTrue((await child).startswith("Blocked:"))
                        raise RuntimeError("diagnostic failure")
                self.assertFalse(agent_tools.diagnostic_reading_active())
                self.assertEqual(await agent_tools.execute_tool(
                    "write_file", {}, workspace, "full", lambda _event: None,
                ), "normal tool result")
            execute.assert_awaited_once()

    async def test_diagnostic_reads_preserve_deny_and_compaction_restrictions(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, agent_tools.diagnostic_reading_scope():
            for approval in ("deny", "restricted", "invalid", ""):
                with self.subTest(approval=approval):
                    self.assertIsNone(oa._tools_for_approval(approval, False))
                    result = await agent_tools.execute_tool(
                        "read_file", {"path": "missing"}, workspace, approval, lambda _event: None,
                    )
                    self.assertTrue(result.startswith("Blocked:"))
            for approval in ("full", "auto", "confirm"):
                with self.subTest(approval=approval):
                    self.assertIsNone(oa._tools_for_approval(approval, True))
                    self.assertEqual({row["function"]["name"] for row in oa._tools_for_approval(approval, False)},
                                     agent_tools.DIAGNOSTIC_READ_TOOL_NAMES)

    async def test_diagnostic_recursive_search_requires_a_narrow_subtree_before_starting(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "tests").mkdir()
            for name in ("grep", "glob"):
                execute = mock.AsyncMock(return_value="narrow result")
                tool = SimpleNamespace(name=name, run=execute, file_action=None, is_plugin=False)
                pattern_key = "glob" if name == "grep" else "pattern"
                with mock.patch.dict(agent_tools.TOOLS_BY_NAME, {name: tool}):
                    with agent_tools.diagnostic_reading_scope():
                        broad = [
                            {pattern_key: "**/*.cpp"},
                            {"path": workspace, pattern_key: "**/*.cpp"},
                            {pattern_key: "./**/*.cpp"},
                            {pattern_key: "*/**/*.cpp"},
                            {pattern_key: "tests/../**/*.cpp"},
                        ]
                        if name == "grep":
                            broad.append({"pattern": "cause"})
                        for args in broad:
                            with self.subTest(name=name, args=args):
                                result = await agent_tools.execute_tool(
                                    name, args, workspace, "full", lambda _event: None,
                                )
                                self.assertIn("search is too broad", result)
                        execute.assert_not_awaited()
                        for args in (
                            {pattern_key: "tests/**/*.cpp"},
                            {"path": "tests", pattern_key: "**/*.cpp"},
                            {pattern_key: "*.py"},
                        ):
                            self.assertEqual(await agent_tools.execute_tool(
                                name, args, workspace, "full", lambda _event: None,
                            ), "narrow result")
                    # An ordinary root search is unaffected by the diagnostic guard.
                    self.assertEqual(await agent_tools.execute_tool(
                        name, {pattern_key: "**/*.cpp"}, workspace, "full", lambda _event: None,
                    ), "narrow result")
                self.assertEqual(execute.await_count, 4)

    async def test_diagnostic_search_budget_settles_cleanup_before_returning_feedback(self) -> None:
        cleaning = asyncio.Event()
        release = asyncio.Event()
        settled = asyncio.Event()
        events = []

        async def search(*_args):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaning.set()
                await release.wait()
                settled.set()
                raise

        tool = SimpleNamespace(name="grep", run=search, file_action=None, is_plugin=False)
        with tempfile.TemporaryDirectory() as workspace, mock.patch.dict(
            agent_tools.TOOLS_BY_NAME, {"grep": tool},
        ), mock.patch.object(agent_tools, "_DIAGNOSTIC_GREP_BUDGET_SECONDS", 0.01):
            with agent_tools.diagnostic_reading_scope():
                owner = asyncio.create_task(agent_tools.execute_tool(
                    "grep", {"path": "tests", "pattern": "cause"}, workspace, "full", events.append,
                ))
            try:
                await asyncio.wait_for(cleaning.wait(), 2)
                self.assertFalse(owner.done())
                self.assertEqual([event["type"] for event in events], ["tool_use"])
                release.set()
                result = await asyncio.wait_for(owner, 2)
                self.assertTrue(settled.is_set())
                self.assertIn("exceeded the diagnostic budget", result)
                self.assertIn("narrower source/test path and glob", result)
                self.assertEqual([event["type"] for event in events], ["tool_use", "tool_result"])
            finally:
                release.set()
                if not owner.done():
                    owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)

    async def test_operator_stop_during_diagnostic_budget_cleanup_remains_cancellation(self) -> None:
        cleaning = asyncio.Event()
        release = asyncio.Event()
        settled = asyncio.Event()
        events = []

        async def search(*_args):
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                settled.set()

        tool = SimpleNamespace(name="grep", run=search, file_action=None, is_plugin=False)
        with tempfile.TemporaryDirectory() as workspace, mock.patch.dict(
            agent_tools.TOOLS_BY_NAME, {"grep": tool},
        ), mock.patch.object(agent_tools, "_DIAGNOSTIC_GREP_BUDGET_SECONDS", 0.01):
            with agent_tools.diagnostic_reading_scope():
                owner = asyncio.create_task(agent_tools.execute_tool(
                    "grep", {"path": "tests", "pattern": "cause"}, workspace, "full", events.append,
                ))
            try:
                await asyncio.wait_for(cleaning.wait(), 2)
                owner.cancel("operator stop during diagnostic cleanup")
                await asyncio.sleep(0)
                owner.cancel("repeated operator stop")
                await asyncio.sleep(0)
                self.assertFalse(owner.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError) as caught:
                    await asyncio.wait_for(owner, 2)
                self.assertEqual(caught.exception.args, ("operator stop during diagnostic cleanup",))
                self.assertTrue(settled.is_set())
                self.assertEqual([event["type"] for event in events], ["tool_use"])
            finally:
                release.set()
                if not owner.done():
                    owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)

    async def test_diagnostic_grep_budget_reaps_the_real_scan_process(self) -> None:
        processes = []
        start = grep._start_scan_worker

        def capture(*args):
            proc, receive = start(*args)
            processes.append(proc)
            return proc, receive

        before = asyncio.all_tasks()
        with tempfile.TemporaryDirectory() as workspace:
            target = Path(workspace, "tests")
            target.mkdir()
            (target / "stuck.txt").write_text("a" * 100_000, encoding="utf-8")
            with (
                agent_tools.diagnostic_reading_scope(),
                mock.patch.object(agent_tools, "_DIAGNOSTIC_GREP_BUDGET_SECONDS", 0.1),
                mock.patch.object(grep, "_start_scan_worker", side_effect=capture),
            ):
                result = await asyncio.wait_for(agent_tools.execute_tool(
                    "grep", {"path": "tests", "glob": "*.txt", "pattern": "(a+)+b"},
                    workspace, "full", lambda _event: None,
                ), 5)
            self.assertIn("exceeded the diagnostic budget", result)
            self.assertEqual(len(processes), 1)
            self.assertFalse(processes[0].is_alive())
            self.assertIsNotNone(processes[0].exitcode)
        await asyncio.sleep(0)
        self.assertFalse(asyncio.all_tasks() - before)

    async def test_ordinary_grep_has_no_diagnostic_execution_budget(self) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def search(*_args):
            entered.set()
            await release.wait()
            return "ordinary search finished"

        tool = SimpleNamespace(name="grep", run=search, file_action=None, is_plugin=False)
        with tempfile.TemporaryDirectory() as workspace, mock.patch.dict(
            agent_tools.TOOLS_BY_NAME, {"grep": tool},
        ), mock.patch.object(agent_tools, "_DIAGNOSTIC_GREP_BUDGET_SECONDS", 0.001):
            owner = asyncio.create_task(agent_tools.execute_tool(
                "grep", {"pattern": "cause"}, workspace, "full", lambda _event: None,
            ))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                await asyncio.sleep(0.02)
                self.assertFalse(owner.done())
                release.set()
                self.assertEqual(await asyncio.wait_for(owner, 2), "ordinary search finished")
            finally:
                release.set()
                await asyncio.gather(owner, return_exceptions=True)

    async def test_internal_hang_recovery_reaps_real_shell_and_returns_one_normal_result(self) -> None:
        ticket = object()
        inherited = ContextVar("observed_tool_ticket", default=None)
        output_started = asyncio.Event()
        attached = []
        captured = []
        events = []
        recovery = "Confirmed stalled command stopped. Retained partial output and inspect existing artifacts before repairing."

        @contextmanager
        def scope(operation):
            token = inherited.set(operation)
            try:
                yield
            finally:
                inherited.reset(token)

        def attach(process, *, command=""):
            self.assertIs(inherited.get(), ticket)
            self.assertIn("partial evidence", command)
            attached.append(process)

        def output(data):
            captured.append(data)
            output_started.set()

        async def confirmed_hang(task, operation):
            self.assertIs(operation, ticket)
            await asyncio.wait_for(output_started.wait(), 2)
            task.cancel("confirmed hang")
            await asyncio.gather(task, return_exceptions=True)
            self.assertTrue(task.cancelled())
            self.assertIsNotNone(attached[0].returncode)
            return recovery

        before = asyncio.all_tasks()
        executable = getattr(sys, "_base_executable", sys.executable)
        with tempfile.TemporaryDirectory() as workspace:
            try:
                with (
                    mock.patch.object(supervision, "new_operation", return_value=ticket) as operation,
                    mock.patch.object(supervision, "operation_scope", side_effect=scope),
                    mock.patch.object(supervision, "attach_process", side_effect=attach),
                    mock.patch.object(supervision, "note_output", side_effect=output),
                    mock.patch.object(supervision, "wait_owned", side_effect=confirmed_hang),
                    mock.patch.object(bash, "_find_shell", return_value=[executable, "-u", "-c"]),
                    self.assertLogs("Cozter.performance", level="INFO") as logs,
                ):
                    result = await asyncio.wait_for(agent_tools.execute_tool(
                        "bash", {"command": "import time; print('partial evidence', flush=True); time.sleep(30)"},
                        workspace, "full", events.append,
                    ), 4)
                operation.assert_called_once_with(workspace, kind="tool:bash", details=mock.ANY)
                self.assertIn("partial evidence", operation.call_args.kwargs["details"])
                self.assertEqual(result, recovery)
                self.assertIn(b"partial evidence", b"".join(captured))
                self.assertEqual([event["type"] for event in events], ["tool_use", "tool_result"])
                self.assertEqual(events[-1]["output"], recovery)
                self.assertTrue(any("outcome=recovered_hang" in line for line in logs.output))
                await asyncio.sleep(0)
                self.assertFalse(asyncio.all_tasks() - before)
            finally:
                for process in attached:
                    await kill_and_wait(process)
                    close_subprocess_pipe(process, 1)

    async def test_external_cancellation_waits_for_tool_cleanup_and_emits_no_recovery_result(self) -> None:
        entered = asyncio.Event()
        cleaning = asyncio.Event()
        release = asyncio.Event()
        settled = asyncio.Event()
        events = []

        async def operation(*_args):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaning.set()
                await release.wait()
                settled.set()
                raise

        tool = SimpleNamespace(run=operation, file_action=None, is_plugin=True)
        with tempfile.TemporaryDirectory() as workspace, mock.patch.dict(
            agent_tools.TOOLS_BY_NAME, {"owned_wait": tool},
        ):
            owner = asyncio.create_task(agent_tools.execute_tool(
                "owned_wait", {}, workspace, "full", events.append,
            ))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                owner.cancel("operator stop")
                await asyncio.wait_for(cleaning.wait(), 2)
                owner.cancel("repeated operator stop")
                await asyncio.sleep(0)
                self.assertFalse(owner.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(owner, 2)
                self.assertTrue(settled.is_set())
                self.assertEqual([event["type"] for event in events], ["tool_use"])
            finally:
                release.set()
                if not owner.done():
                    owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)

    async def test_recovery_is_a_result_barrier_before_the_next_mutation(self) -> None:
        ticket = object()
        settled = asyncio.Event()
        entered = asyncio.Event()
        trace = []

        async def blocked(*_args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                trace.append("old mutation settled")
                settled.set()

        async def next_mutation(*_args):
            self.assertTrue(settled.is_set())
            trace.append("next mutation")
            return "next result"

        async def wait_owned(task, _ticket):
            await entered.wait()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return "Diagnosed stalled mutation; partial work retained."

        with tempfile.TemporaryDirectory() as workspace, mock.patch.dict(
            agent_tools.TOOLS_BY_NAME,
            {"blocked": SimpleNamespace(run=blocked, file_action=None, is_plugin=True),
             "next": SimpleNamespace(run=next_mutation, file_action=None, is_plugin=True)},
        ):
            with (
                mock.patch.object(supervision, "new_operation", return_value=ticket),
                mock.patch.object(supervision, "wait_owned", side_effect=wait_owned),
            ):
                first = await agent_tools.execute_tool("blocked", {}, workspace, "full", lambda _event: None)
            self.assertIn("partial work retained", first)
            second = await agent_tools.execute_tool("next", {}, workspace, "full", lambda _event: None)
        self.assertEqual(second, "next result")
        self.assertEqual(trace, ["old mutation settled", "next mutation"])


if __name__ == "__main__":
    unittest.main()
