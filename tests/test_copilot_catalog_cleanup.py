"""Offline catalog subprocess ownership and bounded stdout reader cleanup."""

import io
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from Cozter.backends_agent import copilot


class CopilotCatalogCleanupTests(unittest.TestCase):
    def test_posix_catalog_launch_creates_owned_session(self):
        with tempfile.TemporaryDirectory() as directory:
            home = os.path.join(directory, "home")
            os.mkdir(home)
            proc = mock.Mock(stdout=None)
            with (
                mock.patch.object(copilot.os, "name", "posix"),
                mock.patch.object(copilot.shutil, "which", return_value="copilot"),
                mock.patch.object(copilot, "executable_command", return_value=["copilot"]),
                mock.patch.object(copilot, "_create_isolated_copilot_home", return_value=home),
                mock.patch.object(copilot.subprocess, "Popen", return_value=proc) as launch,
                mock.patch.object(copilot, "_stop_acp_process") as stop,
            ):
                self.assertIsNone(copilot.CopilotBackend()._discover_models(directory))
            self.assertTrue(launch.call_args.kwargs["start_new_session"])
            stop.assert_called_once_with(proc, kill_tree=True, reader=None, reader_stop=mock.ANY)

    def test_posix_cleanup_kills_owned_group_after_parent_exits(self):
        proc = mock.Mock(pid=12345, stdin=None, stdout=None, stderr=None)
        proc.poll.return_value = 0
        with (
            mock.patch.object(copilot.os, "name", "posix"),
            mock.patch.object(copilot.os, "getpgid", return_value=54321, create=True),
            mock.patch.object(copilot.os, "killpg", create=True) as killpg,
            mock.patch.object(copilot.signal, "SIGKILL", 9, create=True),
        ):
            copilot._stop_acp_process(proc, kill_tree=True)
        killpg.assert_called_once_with(12345, 9)
        proc.terminate.assert_not_called()

    def test_cleanup_never_closes_stream_owned_by_busy_reader(self):
        proc = mock.Mock(stdin=None, stderr=None)
        proc.poll.return_value = 0
        reader = mock.Mock()
        stop = threading.Event()
        copilot._stop_acp_process(proc, reader=reader, reader_stop=stop)
        self.assertTrue(stop.is_set())
        reader.join.assert_called_once_with(timeout=2)
        proc.stdout.close.assert_not_called()

    def test_posix_poll_decodes_partial_utf8_and_unterminated_final_line(self):
        stop = threading.Event()
        stream = mock.Mock()
        stream.fileno.return_value = 123
        with (
            mock.patch.object(copilot.select, "select", return_value=([123], [], [])),
            mock.patch.object(copilot.os, "read", side_effect=[b'first\n\xe2', b'\x82\xac', b'']),
        ):
            self.assertEqual(list(copilot._poll_acp_lines(stream, stop)), ["first\n", "€"])

    def test_posix_poll_stop_does_not_wait_for_retained_pipe(self):
        stop = threading.Event()
        stream = mock.Mock()

        def retained_pipe(*_args):
            stop.set()
            return [], [], []

        with mock.patch.object(copilot.select, "select", side_effect=retained_pipe):
            self.assertEqual(list(copilot._poll_acp_lines(stream, stop)), [])

    def test_production_reader_closes_its_stream_when_stopped(self):
        stop = threading.Event()
        stream = io.StringIO("first\n")
        messages = queue.Queue()
        copilot._read_acp_stdout(stream, messages, stop)
        self.assertTrue(stream.closed)
        self.assertEqual(messages.get_nowait(), "first\n")
        self.assertIsNone(messages.get_nowait())

    @unittest.skipUnless(os.name == "posix", "requires native POSIX process groups")
    def test_catalog_timeout_reaps_descendant_retaining_stdout_after_parent_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            identities = Path(directory) / "pids.json"
            script = Path(directory) / "catalog.py"
            script.write_text(
                "import json,os,subprocess,sys\n"
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
                f"open({str(identities)!r},'w').write(json.dumps([os.getpid(),child.pid]))\n",
                encoding="utf-8",
            )
            home = Path(directory) / "home"
            home.mkdir()
            real_popen = subprocess.Popen
            processes = []

            def launch(*args, **kwargs):
                proc = real_popen(*args, **kwargs)
                processes.append(proc)
                proc.wait(timeout=5)
                return proc

            started = time.monotonic()
            try:
                with (
                    mock.patch.object(copilot.shutil, "which", return_value=sys.executable),
                    mock.patch.object(copilot, "executable_command", return_value=[sys.executable, str(script)]),
                    mock.patch.object(copilot, "_create_isolated_copilot_home", return_value=str(home)),
                    mock.patch.object(copilot.subprocess, "Popen", side_effect=launch),
                    mock.patch.object(copilot, "_MODEL_DISCOVERY_TIMEOUT_SEC", 0.25),
                ):
                    self.assertIsNone(copilot.CopilotBackend()._discover_models(directory))
                self.assertLess(time.monotonic() - started, 5)
                self.assertFalse(home.exists())
                self.assertTrue(processes[0].stdout.closed)
                parent, child = json.loads(identities.read_text(encoding="utf-8"))
                # A killed child may remain a zombie until init reaps it;
                # the closed pipe is evidence that it stopped executing.
                self.assertEqual(parent, processes[0].pid)
                try:
                    os.kill(child, 0)
                except ProcessLookupError:
                    pass
                else:
                    state = Path(f"/proc/{child}/stat")
                    if state.exists():
                        self.assertEqual(state.read_text().split()[2], "Z")
            finally:
                for proc in processes:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
