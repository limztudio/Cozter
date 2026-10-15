"""Cancelling notes mutations waits for their owned filesystem work."""

import asyncio
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from Cozter.agent_tools.plugins import notes


class NotesOwnedWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def wait_entered(self, event: threading.Event) -> None:
        self.assertTrue(await asyncio.wait_for(asyncio.to_thread(event.wait, 2), 3))

    async def test_repeated_append_cancellation_waits_for_durable_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp, ".cozter", "notes.md")
            entered, release, finished = threading.Event(), threading.Event(), threading.Event()
            write = notes.write_text_after_edit

            def held_write(*args, **kwargs):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test did not release notes write")
                try:
                    return write(*args, **kwargs)
                finally:
                    finished.set()

            with mock.patch.object(notes, "write_text_after_edit", side_effect=held_write):
                task = asyncio.create_task(notes.NotesTool().run(tmp, {
                    "action": "append", "text": "first durable entry",
                }))
                try:
                    await self.wait_entered(entered)
                    task.cancel()
                    await asyncio.sleep(0)
                    task.cancel()
                    await asyncio.sleep(0.02)
                    self.assertFalse(task.done(), "cancellation acknowledged before its write settled")
                    self.assertFalse(target.exists())
                    release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 3)
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)
            self.assertTrue(finished.is_set())
            self.assertIn("first durable entry", target.read_text(encoding="utf-8"))
            # A follow-on job sees the completed mutation rather than racing
            # a cancelled caller's still-running read/modify/write thread.
            result = await notes.NotesTool().run(tmp, {"action": "append", "text": "second durable entry"})
            self.assertTrue(result.startswith("Noted"), result)
            saved = target.read_text(encoding="utf-8")
            self.assertEqual(saved.count("first durable entry"), 1)
            self.assertEqual(saved.count("second durable entry"), 1)

    async def test_clear_cancellation_waits_until_named_entry_is_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp, ".cozter", "notes.md")
            target.parent.mkdir()
            target.write_text("existing notes", encoding="utf-8")
            entered, release, finished = threading.Event(), threading.Event(), threading.Event()
            unlink = notes.os.unlink

            def held_unlink(*args, **kwargs):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test did not release notes delete")
                try:
                    return unlink(*args, **kwargs)
                finally:
                    finished.set()

            with mock.patch.object(notes.os, "unlink", side_effect=held_unlink):
                task = asyncio.create_task(notes.NotesTool().run(tmp, {"action": "clear"}))
                try:
                    await self.wait_entered(entered)
                    task.cancel()
                    await asyncio.sleep(0.02)
                    self.assertFalse(task.done(), "cancellation acknowledged before its delete settled")
                    self.assertTrue(target.exists())
                    release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 3)
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)
            self.assertTrue(finished.is_set())
            self.assertFalse(target.exists())

    async def test_read_remains_immediately_cancellable(self) -> None:
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def held_read(*args):
            entered.set()
            try:
                release.wait(3)
                return "bounded notes preview"
            finally:
                finished.set()

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(notes.NotesTool, "_read", side_effect=held_read):
            task = asyncio.create_task(notes.NotesTool().run(tmp, {"action": "read"}))
            try:
                await self.wait_entered(entered)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 1)
                self.assertFalse(finished.is_set())
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                await self.wait_entered(finished)


if __name__ == "__main__":
    unittest.main()
