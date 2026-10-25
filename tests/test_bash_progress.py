"""A quiet command emits liveness metadata without losing output or cancellation."""

import asyncio
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent_tools, supervision
from Cozter.agent_tools.builtin import bash
from Cozter.tests.helpers import create_python_script_process
from Cozter.utils import close_subprocess_pipe, kill_and_wait


class BashProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_heartbeat_is_not_reported_as_useful_output(self):
        proc = await create_python_script_process(
            'import time;time.sleep(.18);print("completed output",flush=True)'
        )
        try:
            with (
                mock.patch.object(bash, "_BASH_PROGRESS_INTERVAL_SEC", .02),
                mock.patch.object(supervision, "note_output") as output,
                self.assertLogs("Cozter.performance", level="INFO") as capture,
            ):
                result, truncated, incomplete = await asyncio.wait_for(
                    bash._read_command_output(proc, 4096), 3,
                )
            self.assertGreaterEqual(len(capture.output), 2)
            self.assertTrue(all("outcome=running" in line for line in capture.output))
            self.assertTrue(all("stage=tool_wait tool=bash pid=" in line for line in capture.output))
            self.assertEqual(b"".join(call.args[0] for call in output.call_args_list), result)
            self.assertIn(b"completed output", result)
            self.assertFalse(truncated)
            self.assertFalse(incomplete)
        finally:
            await kill_and_wait(proc)
            close_subprocess_pipe(proc, 1)
            close_subprocess_pipe(proc, 2)

    async def test_unconfigured_quiet_tool_keeps_running_without_diagnosis(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def quiet(*_args):
            started.set()
            await release.wait()
            return "quiet work completed"

        tool = SimpleNamespace(run=quiet, file_action=None, is_plugin=True)
        with tempfile.TemporaryDirectory() as workspace, mock.patch.dict(
            agent_tools.TOOLS_BY_NAME, {"quiet_fixture": tool},
        ):
            self.assertIsNone(supervision.new_operation(workspace, kind="tool:quiet_fixture"))
            running = asyncio.create_task(agent_tools.execute_tool(
                "quiet_fixture", {}, workspace, "full", lambda _event: None,
            ))
            try:
                await asyncio.wait_for(started.wait(), 1)
                await asyncio.sleep(.05)
                self.assertFalse(running.done())
                release.set()
                self.assertEqual(await asyncio.wait_for(running, 1), "quiet work completed")
            finally:
                release.set()
                await asyncio.gather(running, return_exceptions=True)

    async def test_heartbeat_wait_does_not_delay_tool_cancellation(self):
        tool = bash.BashTool()
        proc = await create_python_script_process('import time;time.sleep(30)')
        try:
            with tempfile.TemporaryDirectory() as ws:
                with (mock.patch.object(bash, "_BASH_PROGRESS_INTERVAL_SEC", .02),
                      mock.patch.object(bash, "_spawn_command", return_value=proc) as spawn):
                    command = asyncio.create_task(tool.run(ws, {"command": "owned quiet child"}))
                    await asyncio.sleep(.05)
                    spawn.assert_awaited_once()
                    command.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(command, 3)
                    self.assertIsNotNone(proc.returncode)
        finally:
            await kill_and_wait(proc)
            close_subprocess_pipe(proc, 1)
            close_subprocess_pipe(proc, 2)
