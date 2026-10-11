"""Native subprocess regressions for shell output and inherited pipes."""

import asyncio
import os
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock

from Cozter import utils
from Cozter.agent_tools.builtin import bash


class BashProcessOutputTests(unittest.IsolatedAsyncioTestCase):
    async def _run_script(self, path: str, script: str) -> str:
        # Windows venv redirectors retain their own stdio handles. Use the
        # interpreter itself so the fixture models a single exiting parent.
        executable = getattr(sys, "_base_executable", sys.executable)
        with mock.patch.object(bash, "_find_shell", return_value=[
            executable, "-u", "-c",
        ]):
            return await asyncio.wait_for(
                bash.BashTool().run(path, {"command": script}), timeout=5,
            )

    @staticmethod
    def _kill_fixture_child(pid_path: str) -> None:
        if not os.path.exists(pid_path):
            return
        with open(pid_path, encoding="utf-8") as file_handle:
            pid = int(file_handle.read())
        if os.name == "nt":
            utils.terminate_windows_process_tree(pid)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    async def test_preserves_normal_output_larger_than_pipe_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            result = await self._run_script(path, (
                "import sys\n"
                "sys.stdout.write('x' * (512 * 1024))\n"
                "sys.stdout.flush()\n"
                "sys.stderr.write('stderr-end')\n"
            ))
        self.assertEqual(result, "x" * (512 * 1024) + "stderr-end")

    async def test_preserves_nonzero_exit_code_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            result = await self._run_script(path, (
                "import sys\nprint('failure details')\nsys.exit(13)\n"
            ))
        self.assertEqual(result.replace("\r\n", "\n"), "$ exit 13\nfailure details\n")

    async def test_post_exit_grace_does_not_limit_running_command(self) -> None:
        with tempfile.TemporaryDirectory() as path, mock.patch.object(
            bash, "_BASH_POST_EXIT_DRAIN_TIMEOUT", 0.04,
        ):
            started = time.monotonic()
            result = await self._run_script(path, (
                "import time\nprint('start',flush=True)\n"
                "time.sleep(0.3)\nprint('finished')\n"
            ))
        self.assertGreaterEqual(time.monotonic() - started, 0.3)
        self.assertEqual(result.replace("\r\n", "\n"), "start\nfinished\n")

    async def test_closed_output_still_waits_for_parent_exit(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            started = time.monotonic()
            result = await self._run_script(path, (
                "import os,time\nprint('output complete',flush=True)\n"
                "os.close(1)\nos.close(2)\ntime.sleep(0.3)\n"
            ))
        self.assertGreaterEqual(time.monotonic() - started, 0.3)
        self.assertEqual(result.strip(), "output complete")

    async def test_exited_parent_with_inherited_pipe_returns_captured_output(self) -> None:
        with tempfile.TemporaryDirectory() as path, mock.patch.object(
            bash, "_BASH_POST_EXIT_DRAIN_TIMEOUT", 0.1,
        ):
            pid_path = os.path.join(path, "child.pid")
            script = (
                "import subprocess,sys\n"
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(10)'],stdin=subprocess.DEVNULL)\n"
                f"open({pid_path!r},'w').write(str(child.pid))\n"
                "print('parent completed',flush=True)\n"
            )
            try:
                started = time.monotonic()
                result = await self._run_script(path, script)
                self.assertLess(time.monotonic() - started, 2)
                self.assertTrue(result.startswith("parent completed"))
                self.assertIn("output may be incomplete", result)
            finally:
                self._kill_fixture_child(pid_path)

    async def test_preserves_descendant_output_within_exit_grace(self) -> None:
        with tempfile.TemporaryDirectory() as path, mock.patch.object(
            bash, "_BASH_POST_EXIT_DRAIN_TIMEOUT", 1.0,
        ):
            pid_path = os.path.join(path, "child.pid")
            script = (
                "import subprocess,sys\n"
                "child=subprocess.Popen([sys.executable,'-c',"
                "\"import time; time.sleep(0.1); print('child output',flush=True)\"],"
                "stdin=subprocess.DEVNULL)\n"
                f"open({pid_path!r},'w').write(str(child.pid))\n"
                "print('parent output',flush=True)\n"
            )
            try:
                result = await self._run_script(path, script)
                self.assertEqual(
                    result.replace("\r\n", "\n"), "parent output\nchild output\n",
                )
            finally:
                self._kill_fixture_child(pid_path)

    async def test_output_ceiling_kills_live_parent(self) -> None:
        original_spawn = asyncio.create_subprocess_exec
        processes: list[asyncio.subprocess.Process] = []

        async def record_process(*args, **kwargs):
            proc = await original_spawn(*args, **kwargs)
            processes.append(proc)
            return proc

        with tempfile.TemporaryDirectory() as path, mock.patch.object(
            bash, "_BASH_MAX_OUTPUT_BYTES", 16 * 1024,
        ), mock.patch.object(asyncio, "create_subprocess_exec", record_process):
            result = await self._run_script(path, (
                "import sys,time\n"
                "sys.stdout.write('z' * (128 * 1024))\nsys.stdout.flush()\n"
                "time.sleep(10)\n"
            ))
        self.assertTrue(result.startswith("z" * (16 * 1024)))
        self.assertIn("output truncated at 16384 bytes", result)
        self.assertIsNotNone(processes[0].returncode)

    async def test_cancellation_reaps_parent_and_output_tasks(self) -> None:
        original_spawn = asyncio.create_subprocess_exec
        processes: list[asyncio.subprocess.Process] = []

        async def record_process(*args, **kwargs):
            proc = await original_spawn(*args, **kwargs)
            processes.append(proc)
            return proc

        with tempfile.TemporaryDirectory() as path, mock.patch.object(
            asyncio, "create_subprocess_exec", record_process,
        ):
            before = asyncio.all_tasks()
            task = asyncio.create_task(self._run_script(path, (
                "import time\nprint('started',flush=True)\ntime.sleep(10)\n"
            )))
            try:
                async def wait_for_launch() -> None:
                    while not processes:
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(wait_for_launch(), timeout=2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=3)
                self.assertIsNotNone(processes[0].returncode)
                await asyncio.sleep(0)
                self.assertFalse(asyncio.all_tasks() - before)
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                for proc in processes:
                    if proc.returncode is None:
                        await utils.kill_and_wait(proc)
                    utils.close_subprocess_pipe(proc, 1)


if __name__ == "__main__":
    unittest.main()
