"""Git CLI startup shares process ownership without loading provider adapters."""

import asyncio
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from Cozter import process_launch, utils


_FIXTURE_PYTHON = getattr(sys, "_base_executable", None) or sys.executable


class GitColdImportTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("git"), "Git is required for the local fixture")
    def test_real_status_stays_cold_and_keeps_unicode_output(self):
        env = os.environ.copy() | {
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "PYTHONIOENCODING": "utf-8",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
        }
        source = textwrap.dedent("""
            import runpy
            import sys
            sys.argv[:] = ['-m', '{"action":"status"}']
            runpy.run_module('Cozter.agent_tools.plugins.git_info', run_name='__main__')
            assert 'aiohttp' not in sys.modules
            assert not any(name.startswith('Cozter.backends_agent') for name in sys.modules)
        """)
        with tempfile.TemporaryDirectory() as workspace:
            subprocess.run(
                ["git", "init", "--quiet", "--initial-branch=fixture", workspace],
                env=env, capture_output=True, check=True, timeout=10,
            )
            Path(workspace, "untracked-한글.txt").write_text("synthetic fixture\n", encoding="utf-8")
            child = subprocess.run(
                [sys.executable, "-B", "-c", source], cwd=workspace, env=env,
                capture_output=True, timeout=10,
            )
        self.assertEqual(child.returncode, 0, child.stderr.decode("utf-8", errors="replace"))
        self.assertEqual(child.stderr, b"")
        self.assertEqual(child.stdout.decode("utf-8").splitlines(),
                         ["## No commits yet on fixture", "?? untracked-한글.txt"])


class SharedProcessLaunchTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_creation_settles_child_and_preserves_first_cancel(self):
        original = process_launch.create_piped_subprocess
        started, release = asyncio.Event(), asyncio.Event()
        processes = []

        async def delayed_creation(*args, **kwargs):
            proc = await original(*args, **kwargs)
            processes.append(proc)
            started.set()
            await release.wait()
            return proc

        with mock.patch.object(process_launch, "create_piped_subprocess", side_effect=delayed_creation):
            owner = asyncio.create_task(process_launch.create_captured_subprocess(
                [_FIXTURE_PYTHON, "-B", "-c", "import time; time.sleep(120)"],
            ))
            try:
                await asyncio.wait_for(started.wait(), 5)
                owner.cancel("first")
                await asyncio.sleep(0)
                owner.cancel("second")
                await asyncio.sleep(0)
                self.assertFalse(owner.done())
                self.assertIsNone(processes[0].returncode)
                release.set()
                with self.assertRaises(asyncio.CancelledError) as raised:
                    await asyncio.wait_for(owner, 5)
                self.assertEqual(raised.exception.args, ("first",))
                self.assertIsNotNone(processes[0].returncode)
            finally:
                release.set()
                await asyncio.gather(owner, return_exceptions=True)
                for proc in processes:
                    if proc.returncode is None:
                        await utils.kill_and_wait(proc)
                    utils.close_subprocess_pipe(proc, 1)
                    utils.close_subprocess_pipe(proc, 2)

    async def test_creation_failure_after_cancel_is_consumed(self):
        started, release = asyncio.Event(), asyncio.Event()
        errors = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()

        async def failing_creation(*_args, **_kwargs):
            started.set()
            await release.wait()
            raise RuntimeError("synthetic late creation failure")

        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        try:
            with mock.patch.object(process_launch, "create_piped_subprocess", side_effect=failing_creation):
                owner = asyncio.create_task(process_launch.create_captured_subprocess(["synthetic-command"]))
                try:
                    await asyncio.wait_for(started.wait(), 2)
                    owner.cancel("first")
                    await asyncio.sleep(0)
                    owner.cancel("second")
                    release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(owner, 2)
                finally:
                    release.set()
                    await asyncio.gather(owner, return_exceptions=True)
            gc.collect()
            await asyncio.sleep(0)
            self.assertEqual(errors, [])
        finally:
            loop.set_exception_handler(previous_handler)

    async def test_custom_cancel_cleanup_keeps_its_ownership_policy(self):
        started, release = asyncio.Event(), asyncio.Event()
        proc = mock.Mock(returncode=0)

        async def delayed_creation(*_args, **_kwargs):
            started.set()
            await release.wait()
            return proc

        cleanup = mock.AsyncMock()
        with (
            mock.patch.object(process_launch, "create_piped_subprocess", side_effect=delayed_creation),
            mock.patch.object(process_launch, "kill_and_wait", new=mock.AsyncMock()) as kill,
        ):
            owner = asyncio.create_task(process_launch.create_captured_subprocess(
                ["synthetic-launcher"], cancel_cleanup=cleanup,
            ))
            try:
                await asyncio.wait_for(started.wait(), 2)
                owner.cancel()
                await asyncio.sleep(0)
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(owner, 2)
            finally:
                release.set()
                await asyncio.gather(owner, return_exceptions=True)
        cleanup.assert_awaited_once_with(proc)
        kill.assert_not_awaited()

    async def test_cleanup_closes_exited_managed_group_pipes(self):
        proc = mock.Mock(returncode=0)
        with (
            mock.patch.object(process_launch, "has_managed_process_group", return_value=True) as has_group,
            mock.patch.object(process_launch, "kill_and_wait", new=mock.AsyncMock()) as kill,
            mock.patch.object(process_launch, "close_subprocess_pipe") as close_pipe,
        ):
            await process_launch.reap_failed_subprocess(proc)
        has_group.assert_called_once_with(proc)
        kill.assert_awaited_once_with(proc)
        self.assertEqual(close_pipe.call_args_list, [mock.call(proc, 1), mock.call(proc, 2)])

    async def test_shared_default_cleanup_closes_both_pipes_after_group_kill_failure(self):
        proc = mock.Mock(returncode=0)
        with (
            mock.patch.object(process_launch, "has_managed_process_group", return_value=True) as has_group,
            mock.patch.object(process_launch, "kill_and_wait", new=mock.AsyncMock(
                side_effect=RuntimeError("synthetic cleanup failure"),
            )) as kill,
            mock.patch.object(process_launch, "close_subprocess_pipe") as close_pipe,
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic cleanup failure"):
                await process_launch.reap_failed_subprocess(proc)
        has_group.assert_called_once_with(proc)
        kill.assert_awaited_once_with(proc)
        self.assertEqual(close_pipe.call_args_list, [mock.call(proc, 1), mock.call(proc, 2)])

    async def test_spawn_keeps_group_marking_and_exact_arguments(self):
        proc = mock.Mock(returncode=None)
        with (
            mock.patch.object(process_launch.asyncio, "create_subprocess_exec", new=mock.AsyncMock(return_value=proc)) as create,
            mock.patch.object(process_launch, "mark_process_group_leader") as mark_group,
        ):
            mark_group.__bool__.return_value = False
            result = await process_launch.create_piped_subprocess(
                ["synthetic-command", "arg"], stdin=7, cwd="synthetic-workspace",
                env={"SYNTHETIC_ENV": "value"}, start_new_session=True,
            )
        self.assertIs(result, proc)
        mark_group.assert_called_once_with(proc)
        create.assert_awaited_once_with(
            "synthetic-command", "arg", stdin=7,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd="synthetic-workspace", env={"SYNTHETIC_ENV": "value"}, start_new_session=True,
        )


if __name__ == "__main__":
    unittest.main()
