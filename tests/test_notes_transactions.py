"""Separate plugin processes serialize notes append and clear transactions."""

import asyncio
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from Cozter.agent_tools.plugins import notes


_PROCESS = """
import asyncio
from pathlib import Path
import sys
import time
from Cozter.agent_tools.plugins import notes

workspace, role, action = sys.argv[1:]
ready = Path(workspace, 'first-reading')
second = Path(workspace, 'second-started')
if role == 'first':
    original = notes._read_notes_text
    def held_read(target):
        result = original(target)
        ready.touch()
        deadline = time.monotonic() + 10
        while not second.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('second process did not start')
            time.sleep(0.01)
        time.sleep(0.5)
        return result
    notes._read_notes_text = held_read
else:
    second.touch()
result = asyncio.run(notes.NotesTool().run(workspace, {
    'action': action, 'text': role + ' durable entry',
}))
print(result)
if result.startswith('Error:'):
    raise SystemExit(1)
"""


class NotesProcessTransactionTests(unittest.TestCase):
    def invoke_pair(self, workspace, second_action):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            entry for entry in (str(Path(__file__).resolve().parents[2]),
                                environment.get("PYTHONPATH", "")) if entry
        )
        processes = []
        try:
            first = subprocess.Popen(
                [sys.executable, "-c", _PROCESS, workspace, "first", "append"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment,
            )
            processes.append(first)
            deadline = time.monotonic() + 10
            while not Path(workspace, "first-reading").exists():
                if first.poll() is not None or time.monotonic() >= deadline:
                    output = first.communicate(timeout=2)
                    self.fail(f"first process did not begin its transaction: {output}")
                time.sleep(0.01)
            second = subprocess.Popen(
                [sys.executable, "-c", _PROCESS, workspace, "second", second_action],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment,
            )
            processes.append(second)
            results = [process.communicate(timeout=15) for process in processes]
            for process, (stdout, stderr) in zip(processes, results):
                self.assertEqual(process.returncode, 0, (stdout, stderr))
                self.assertFalse(stderr, stderr)
            return results
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)

    def test_simultaneous_cli_appends_preserve_both_successful_entries(self):
        with tempfile.TemporaryDirectory() as workspace:
            results = self.invoke_pair(workspace, "append")
            for stdout, _stderr in results:
                self.assertTrue(stdout.startswith("Noted"), stdout)
            content = Path(workspace, ".cozter", "notes.md").read_text(encoding="utf-8")
            self.assertEqual(content.count("first durable entry"), 1)
            self.assertEqual(content.count("second durable entry"), 1)

    def test_cli_clear_waits_for_the_earlier_append_transaction(self):
        with tempfile.TemporaryDirectory() as workspace:
            results = self.invoke_pair(workspace, "clear")
            self.assertIn("Notes cleared.", results[1][0])
            self.assertFalse(Path(workspace, ".cozter", "notes.md").exists())


class NotesCancelledTransactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_owner_holds_lock_until_write_completes(self):
        entered, release, second_read = threading.Event(), threading.Event(), threading.Event()
        original_write, original_read = notes.write_text_after_edit, notes._read_notes_text

        def held_write(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("notes write was not released")
            return original_write(*args, **kwargs)

        def observe_read(target):
            if entered.is_set():
                second_read.set()
            return original_read(target)

        with (
            tempfile.TemporaryDirectory() as workspace,
            mock.patch.object(notes, "write_text_after_edit", side_effect=held_write),
            mock.patch.object(notes, "_read_notes_text", side_effect=observe_read),
        ):
            first = asyncio.create_task(notes.NotesTool().run(workspace, {
                "action": "append", "text": "first cancelled entry",
            }))
            second = None
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                first.cancel()
                second = asyncio.create_task(notes.NotesTool().run(workspace, {
                    "action": "append", "text": "second entry",
                }))
                await asyncio.sleep(0.1)
                self.assertFalse(second_read.is_set())
                self.assertFalse(first.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                self.assertTrue((await second).startswith("Noted"))
                content = Path(workspace, ".cozter", "notes.md").read_text(encoding="utf-8")
                self.assertIn("first cancelled entry", content)
                self.assertIn("second entry", content)
            finally:
                release.set()
                await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)
