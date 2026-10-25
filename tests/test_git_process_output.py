"""Native Git-runner regressions for verbose output and inherited pipes."""

import asyncio
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

from Cozter import utils
from Cozter.agent_tools.plugins import _git_common as git


class GitProcessOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_stderr_only_diagnostic_reaches_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            out, err, rc = await self._script(path, (
                "import sys\nsys.stderr.write('warning: fixture diagnostic\\n')\n"
            ))
        self.assertEqual(rc, 0)
        result = git.finish_git_output(out, err)
        self.assertEqual(result, "OK\n\ngit said:\nwarning: fixture diagnostic")

    async def _script(self, path: str, script: str) -> tuple[str, str, int]:
        executable = getattr(sys, "_base_executable", sys.executable)
        return await asyncio.wait_for(
            git.git_once([executable, "-u", "-c", script], path, dict(os.environ)),
            timeout=5,
        )

    async def test_both_streams_larger_than_pipe_capacity_keep_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            out, err, rc = await self._script(path, (
                "import sys\n"
                "sys.stdout.write('x'*1048576)\nsys.stdout.flush()\n"
                "sys.stderr.write('failure detail\\n'+'y'*1048576)\n"
                "sys.stderr.flush()\nsys.exit(13)\n"
            ))
        self.assertEqual(rc, 13)
        self.assertLess(len(out), git._GIT_STDOUT_BYTES + 200)
        self.assertLess(len(err), git._GIT_STDERR_BYTES + 200)
        self.assertTrue(out.startswith("x" * 100))
        self.assertTrue(err.startswith("failure detail"))
        self.assertIn("git output truncated", out)
        self.assertIn("git output truncated", err)

    async def test_running_command_outlives_post_exit_grace(self) -> None:
        with tempfile.TemporaryDirectory() as path, mock.patch.object(
            git, "_GIT_POST_EXIT_DRAIN_SECONDS", .03,
        ):
            started = time.monotonic()
            out, _, rc = await self._script(path, (
                "import time\nprint('start',flush=True)\n"
                "time.sleep(.2)\nprint('done')\n"
            ))
        self.assertGreaterEqual(time.monotonic() - started, .2)
        self.assertEqual(out.replace("\r\n", "\n"), "start\ndone\n")
        self.assertEqual(rc, 0)

    async def test_exited_parent_with_inherited_pipes_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as path, mock.patch.object(
            git, "_GIT_POST_EXIT_DRAIN_SECONDS", .1,
        ):
            pid_path = os.path.join(path, "fixture-child.pid")
            script = (
                "import subprocess,sys\n"
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(10)'],stdin=subprocess.DEVNULL)\n"
                f"open({pid_path!r},'w').write(str(child.pid))\n"
                "print('parent complete',flush=True)\n"
            )
            try:
                started = time.monotonic()
                out, _, rc = await self._script(path, script)
                self.assertLess(time.monotonic() - started, 2)
                self.assertEqual(rc, 0)
                self.assertTrue(out.startswith("parent complete"))
                self.assertIn("output may be incomplete", out)
            finally:
                await self._kill_fixture(pid_path)

    async def test_post_exit_grace_keeps_late_child_output(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            out, err, rc = await self._script(path, (
                "import subprocess,sys\n"
                "subprocess.Popen([sys.executable,'-u','-c',"
                "'import time; time.sleep(.05); print(\"child tail\")'])\n"
                "print('parent complete',flush=True)\n"
            ))
        self.assertEqual(rc, 0)
        self.assertEqual(err, "")
        self.assertEqual(out.replace("\r\n", "\n"), "parent complete\nchild tail\n")

    async def test_cancel_reaps_parent_and_closes_inherited_pipes(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            pid_path = os.path.join(path, "fixture-parent.pid")
            child_pid_path = os.path.join(path, "fixture-child.pid")
            spawned: list[asyncio.subprocess.Process] = []
            launch = git.create_captured_subprocess

            async def capture_launch(*args, **kwargs):
                proc = await launch(*args, **kwargs)
                spawned.append(proc)
                return proc

            with mock.patch.object(git, "create_captured_subprocess", capture_launch):
                task = asyncio.create_task(self._script(path, (
                    "import os,subprocess,sys,time\n"
                    "child=subprocess.Popen([sys.executable,'-c',"
                    "'import time; time.sleep(10)'],stdin=subprocess.DEVNULL)\n"
                    f"open({child_pid_path!r},'w').write(str(child.pid))\n"
                    f"open({pid_path!r},'w').write(str(os.getpid()))\n"
                    "time.sleep(10)\n"
                )))
                try:
                    async with asyncio.timeout(3):
                        while not os.path.exists(pid_path):
                            await asyncio.sleep(.01)
                    started = time.monotonic()
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, timeout=3)
                    self.assertLess(time.monotonic() - started, 3)
                    self.assertEqual(len(spawned), 1)
                    self.assertIsNotNone(spawned[0].returncode)
                finally:
                    await self._kill_fixture(pid_path)
                    await self._kill_fixture(child_pid_path)
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)

    @staticmethod
    async def _kill_fixture(pid_path: str) -> None:
        if not os.path.exists(pid_path):
            return
        with open(pid_path, encoding="utf-8") as file_handle:
            pid = int(file_handle.read())
        if os.name == "nt":
            # taskkill spawns a child process (~0.1s); keep it off the
            # loop so asyncio debug mode does not flag this teardown.
            await asyncio.to_thread(utils.terminate_windows_process_tree, pid)
        else:
            import signal
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    async def test_repeated_cancel_during_spawn_retains_process_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            spawned: list[asyncio.subprocess.Process] = []
            created = asyncio.Event()
            handover = asyncio.Event()
            launch = git.create_captured_subprocess

            async def delayed_launch(*args, **kwargs):
                proc = await launch(*args, **kwargs)
                spawned.append(proc)
                created.set()
                await handover.wait()
                return proc

            with mock.patch.object(git, "create_captured_subprocess", delayed_launch):
                executable = getattr(sys, "_base_executable", sys.executable)
                task = asyncio.create_task(git.git_once(
                    [executable, "-u", "-c", "import time; time.sleep(10)"],
                    path, dict(os.environ),
                ))
                try:
                    await asyncio.wait_for(created.wait(), timeout=3)
                    task.cancel()
                    await asyncio.sleep(0)
                    task.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                    handover.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, timeout=3)
                    self.assertIsNotNone(spawned[0].returncode)
                finally:
                    handover.set()
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    for proc in spawned:
                        if proc.returncode is None:
                            await utils.kill_and_wait(proc)

    async def test_cancelled_spawn_failure_is_consumed_without_loop_diagnostics(self) -> None:
        started, release = asyncio.Event(), asyncio.Event()
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        diagnostics: list[dict] = []
        loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
        self.addCleanup(loop.set_exception_handler, previous_handler)

        async def fail_creation(*_args, **_kwargs):
            started.set()
            await release.wait()
            raise OSError("fixture git spawn failed after cancellation")

        with mock.patch.object(git, "create_captured_subprocess", side_effect=fail_creation):
            task = asyncio.create_task(git._spawn_git(["fixture-git"], ".", {}))
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1)
                # Flush callbacks so a consumed launch failure cannot become
                # an "exception in shielded future" diagnostic on Python 3.14.
                await asyncio.sleep(0)
                self.assertEqual(diagnostics, [])
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_reader_failure_reaps_live_parent_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            launch = git.create_captured_subprocess
            spawned: list[asyncio.subprocess.Process] = []

            async def failing_reader_launch(*args, **kwargs):
                proc = await launch(*args, **kwargs)
                spawned.append(proc)
                assert proc.stdout is not None
                proc.stdout.read = mock.AsyncMock(side_effect=OSError("fixture read failed"))
                return proc

            with mock.patch.object(git, "create_captured_subprocess", failing_reader_launch):
                with self.assertRaisesRegex(OSError, "fixture read failed"):
                    await asyncio.wait_for(
                        self._script(path, "import time; time.sleep(10)"), timeout=3,
                    )
            self.assertEqual(len(spawned), 1)
            self.assertIsNotNone(spawned[0].returncode)


if __name__ == "__main__":
    unittest.main()
