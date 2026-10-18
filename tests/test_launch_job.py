"""Launcher job coverage; native tests protect only isolated fixture trees."""

import asyncio
import ctypes
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import launch_job


class LaunchJobUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment: dict[str, str] = {}
        self.api = SimpleNamespace(
            CreateJobObjectW=mock.Mock(return_value=11),
            OpenJobObjectW=mock.Mock(return_value=22),
            SetInformationJobObject=mock.Mock(return_value=1),
            AssignProcessToJobObject=mock.Mock(return_value=1),
            GetCurrentProcess=mock.Mock(return_value=-1),
            IsProcessInJob=mock.Mock(return_value=1),
            CloseHandle=mock.Mock(return_value=1),
        )
        patches = (
            mock.patch.object(launch_job, "os", SimpleNamespace(
                name="nt", environ=self.environment, getpid=lambda: 4321,
            )),
            mock.patch.object(launch_job, "_kernel32", return_value=self.api),
            mock.patch.object(launch_job, "_job_handle", None),
            mock.patch.object(launch_job, "_protected", False),
            mock.patch.object(launch_job, "_last_error", return_value=0),
            mock.patch.object(launch_job.ctypes, "set_last_error", create=True),
            mock.patch.object(launch_job.ctypes, "FormatError", return_value="test failure", create=True),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_non_windows_does_not_load_native_api(self) -> None:
        with mock.patch.object(launch_job.os, "name", "posix"):
            self.assertFalse(launch_job.ensure_launch_job())
        launch_job._kernel32.assert_not_called()
        self.assertEqual(self.environment, {})

    def test_current_process_is_assigned_before_marker_and_owner_keeps_handle(self) -> None:
        def assign(handle, process):
            self.assertNotIn(launch_job._LAUNCH_JOB_ENV, self.environment)
            self.assertEqual((handle, process), (11, -1))
            return 1

        self.api.AssignProcessToJobObject.side_effect = assign
        self.assertTrue(launch_job.ensure_launch_job())
        self.assertEqual(launch_job._job_handle, 11)
        self.api.CloseHandle.assert_not_called()
        security, name = self.api.CreateJobObjectW.call_args.args
        self.assertIsNone(security)  # NULL attributes produce a noninherited handle.
        self.assertTrue(name.startswith("Local\\CozterLaunch-4321-"))
        self.assertEqual(self.environment[launch_job._LAUNCH_JOB_ENV], name)
        handle, info_class, info, size = self.api.SetInformationJobObject.call_args.args
        self.assertEqual(handle, 11)
        self.assertEqual(info_class, 9)
        self.assertEqual(info._obj.BasicLimitInformation.LimitFlags, 0x2000)
        self.assertEqual(size, ctypes.sizeof(launch_job._ExtendedLimitInformation))

    def test_repeated_calls_create_and_assign_only_once(self) -> None:
        self.assertTrue(launch_job.ensure_launch_job())
        self.assertTrue(launch_job.ensure_launch_job())
        self.api.CreateJobObjectW.assert_called_once()
        self.api.AssignProcessToJobObject.assert_called_once()

    def test_descendant_verifies_membership_and_closes_query_handle(self) -> None:
        marker = "Local\\parent-launch-job"
        self.environment[launch_job._LAUNCH_JOB_ENV] = marker

        def belongs(process, handle, result):
            self.assertEqual((process, handle), (-1, 22))
            result._obj.value = 1
            return 1

        self.api.IsProcessInJob.side_effect = belongs
        self.assertTrue(launch_job.ensure_launch_job())
        self.api.OpenJobObjectW.assert_called_once_with(0x0004, False, marker)
        self.api.CloseHandle.assert_called_once_with(22)
        self.api.CreateJobObjectW.assert_not_called()
        self.api.AssignProcessToJobObject.assert_not_called()
        self.assertIsNone(launch_job._job_handle)
        self.assertEqual(self.environment[launch_job._LAUNCH_JOB_ENV], marker)

    def test_unverified_marker_does_not_skip_new_protection(self) -> None:
        self.environment[launch_job._LAUNCH_JOB_ENV] = "Local\\unrelated-job"
        # A FALSE query output alone is not proof of protection.
        self.assertTrue(launch_job.ensure_launch_job())
        self.api.CloseHandle.assert_called_once_with(22)
        self.api.CreateJobObjectW.assert_called_once()
        self.api.AssignProcessToJobObject.assert_called_once()
        self.assertNotEqual(self.environment[launch_job._LAUNCH_JOB_ENV], "Local\\unrelated-job")

    def test_stale_marker_creates_new_protection(self) -> None:
        self.environment[launch_job._LAUNCH_JOB_ENV] = "Local\\expired-job"
        self.api.OpenJobObjectW.return_value = None
        self.assertTrue(launch_job.ensure_launch_job())
        self.api.CreateJobObjectW.assert_called_once()
        self.api.CloseHandle.assert_not_called()

    def test_failed_membership_query_closes_handle_and_fails_clearly(self) -> None:
        self.environment[launch_job._LAUNCH_JOB_ENV] = "Local\\parent-job"
        self.api.IsProcessInJob.return_value = 0
        with self.assertRaisesRegex(RuntimeError, "IsProcessInJob"):
            launch_job.ensure_launch_job()
        self.api.CloseHandle.assert_called_once_with(22)
        self.api.CreateJobObjectW.assert_not_called()
        self.assertFalse(launch_job._protected)

    def test_failed_creation_never_assigns_or_marks_protected(self) -> None:
        self.api.CreateJobObjectW.return_value = None
        with self.assertRaisesRegex(RuntimeError, "CreateJobObjectW"):
            launch_job.ensure_launch_job()
        self.api.AssignProcessToJobObject.assert_not_called()
        self.api.CloseHandle.assert_not_called()
        self.assertEqual(self.environment, {})
        self.assertFalse(launch_job._protected)

    def test_failed_limit_setup_closes_only_unassigned_handle(self) -> None:
        self.api.SetInformationJobObject.return_value = 0
        with self.assertRaisesRegex(RuntimeError, "SetInformationJobObject"):
            launch_job.ensure_launch_job()
        self.api.CloseHandle.assert_called_once_with(11)
        self.api.AssignProcessToJobObject.assert_not_called()
        self.assertEqual(self.environment, {})

    def test_incompatible_parent_job_fails_before_children_can_spawn(self) -> None:
        self.api.AssignProcessToJobObject.return_value = 0
        with mock.patch.object(launch_job, "_last_error", return_value=5):
            with self.assertRaisesRegex(RuntimeError, "parent job may prohibit nested jobs"):
                launch_job.ensure_launch_job()
        self.api.CloseHandle.assert_called_once_with(11)
        self.assertIsNone(launch_job._job_handle)
        self.assertFalse(launch_job._protected)
        self.assertEqual(self.environment, {})


@unittest.skipUnless(os.name == "nt", "Windows job-object lifetime semantics")
class WindowsLaunchJobLifetimeTests(unittest.IsolatedAsyncioTestCase):
    async def _assert_fixture_tree_dies(self, *, existing_parent_job: bool = False) -> None:
        # Every job below belongs to a new fixture launcher, and every
        # termination targets that fixture, never the unittest runner itself.
        child_script = (
            "import json, os, subprocess, sys, time\n"
            "from Cozter import launch_job\n"
            "launch_job.ensure_launch_job()\n"
            "grandchild = subprocess.Popen([sys.executable, '-B', '-c', "
            "'import time; time.sleep(120)'], stdin=subprocess.DEVNULL, "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=False)\n"
            "print(json.dumps({'child': os.getpid(), 'grandchild': grandchild.pid, "
            "'child_kept_job_handle': launch_job._job_handle is not None}), flush=True)\n"
            "time.sleep(120)\n"
        )
        outer_setup = (
            "api = launch_job._kernel32()\n"
            "outer_job = api.CreateJobObjectW(None, None)\n"
            "assert outer_job and api.AssignProcessToJobObject(outer_job, api.GetCurrentProcess())\n"
        ) if existing_parent_job else ""
        parent_script = (
            "import json, subprocess, sys, time\n"
            "from Cozter import launch_job\n"
            + outer_setup
            + "launch_job.ensure_launch_job()\n"
            + f"child = subprocess.Popen([sys.executable, '-B', '-c', {child_script!r}], "
            "stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, "
            "close_fds=False)\n"
            "line = child.stdout.readline()\n"
            "if not line:\n"
            "    raise RuntimeError(child.stderr.read().decode())\n"
            "data = json.loads(line)\n"
            "data['owner_kept_job_handle'] = launch_job._job_handle is not None\n"
            "print(json.dumps(data), flush=True)\n"
            "time.sleep(120)\n"
        )
        environment = dict(os.environ)
        environment.pop(launch_job._LAUNCH_JOB_ENV, None)
        package_parent = str(Path(__file__).resolve().parents[2])
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-B", "-c", parent_script,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=package_parent, env=environment,
        )
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
        api.OpenProcess.restype = ctypes.c_void_p
        api.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        api.WaitForSingleObject.restype = ctypes.c_uint32
        api.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        api.TerminateProcess.restype = ctypes.c_int32
        api.CloseHandle.argtypes = [ctypes.c_void_p]
        api.CloseHandle.restype = ctypes.c_int32
        handles: list[int] = []
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=10)
            if not line:
                error = await asyncio.wait_for(proc.stderr.read(), timeout=2)
                self.fail(f"fixture launcher exited before readiness: {error.decode()}")
            data = json.loads(line)
            self.assertTrue(data["owner_kept_job_handle"])
            self.assertFalse(data["child_kept_job_handle"])
            for key in ("child", "grandchild"):
                # Track exact processes via sync handles (PIDs can be reused).
                handle = api.OpenProcess(0x100000 | 0x1000 | 0x0001, False, data[key])
                self.assertTrue(handle, f"could not open fixture {key}")
                handles.append(handle)
                self.assertEqual(api.WaitForSingleObject(handle, 0), 258)
            proc.kill()  # Only the root fixture dies directly, as with task stop.
            await asyncio.wait_for(proc.wait(), timeout=5)
            for handle in handles:
                result = await asyncio.to_thread(api.WaitForSingleObject, handle, 5_000)
                self.assertEqual(result, 0, "fixture descendant survived launcher termination")
        finally:
            if proc.returncode is None:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=5)
            for handle in handles:
                if api.WaitForSingleObject(handle, 0) == 258:
                    # Cleanup remains restricted to known fixture handles.
                    api.TerminateProcess(handle, 1)
                    await asyncio.to_thread(api.WaitForSingleObject, handle, 5_000)
                api.CloseHandle(handle)

    async def test_forced_launcher_termination_kills_child_and_grandchild(self) -> None:
        await self._assert_fixture_tree_dies()

    async def test_launcher_job_nests_inside_existing_parent_job(self) -> None:
        await self._assert_fixture_tree_dies(existing_parent_job=True)


if __name__ == "__main__":
    unittest.main()
