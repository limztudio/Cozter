"""Explicit attachment copies yield the loop while retaining delivery ownership."""

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from Cozter import agent
from Cozter.agent_tools.base import get_tool_permission, tool_permission_scope
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.tests.helpers import TestBot


class AttachmentPreparationOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_external_copy_allows_status_and_preserves_order_dedup_and_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = root / "external"
            external.mkdir()
            workspace = root / "workspace"
            workspace.mkdir()
            source = external / "fixture.png"
            source.write_bytes(b"synthetic image fixture")
            bot = TestBot(["u1"])
            bot.send_text = mock.AsyncMock()
            started = asyncio.Event()
            release = threading.Event()
            loop = asyncio.get_running_loop()
            main_thread = threading.get_ident()
            observed = []
            original_copy = agent._copy_to_unique_path

            def blocked_copy(*args):
                observed.append((threading.get_ident(), get_tool_permission()))
                loop.call_soon_threadsafe(started.set)
                if not release.wait(5):
                    raise AssertionError("event loop never released the copy")
                return original_copy(*args)

            result = AgentResult(events=[
                ChatEvent(kind="attachment", content=str(source)),
                ChatEvent(kind="text", content=f"Done [[attach:{source}]]"),
                ChatEvent(kind="attachment", content=str(source)),
            ])
            with (
                mock.patch.object(agent, "_external_attachment_roots", return_value=[str(external)]),
                mock.patch.object(agent, "_copy_to_unique_path", blocked_copy),
                mock.patch.object(bot, "_track_attachment_upload") as upload,
                tool_permission_scope("full"),
            ):
                task = asyncio.create_task(bot._send_result("chat", str(workspace), result, uid="u1"))
                try:
                    await asyncio.wait_for(started.wait(), 2)
                    # This status operation runs while the worker is blocked;
                    # no wall-clock latency assertion is needed.
                    await bot.send_status("chat", "Thinking...")
                    self.assertFalse(task.done())
                    self.assertEqual(bot.send_text.await_args.args, ("chat", "Thinking..."))
                    self.assertNotEqual(observed[0][0], main_thread)
                    self.assertEqual(observed[0][1], "full")
                    release.set()
                    await task
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)
            self.assertEqual(len(observed), 1)
            upload.assert_called_once()
            owner, chat, paths = upload.call_args.args
            self.assertEqual((owner, chat), ("u1", "chat"))
            self.assertEqual(len(paths), 1)
            self.assertEqual(Path(paths[0]).read_bytes(), source.read_bytes())
            self.assertEqual([call.args[1] for call in bot.send_text.await_args_list], ["Thinking...", "Done"])

    async def test_repeated_cancel_settles_copy_before_ack_and_never_prepares_later_file(self):
        loop = asyncio.get_running_loop()
        diagnostics = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
        self.addCleanup(loop.set_exception_handler, previous_handler)
        for late_failure in (False, True):
            with self.subTest(late_failure=late_failure), tempfile.TemporaryDirectory() as temporary:
                started = asyncio.Event()
                release = threading.Event()
                settled = threading.Event()
                published = Path(temporary) / "copied.png"
                attempts = []
                bot = TestBot(["u1"])
                bot.send_text = mock.AsyncMock()

                def blocked_prepare(path, _workspace):
                    attempts.append(path)
                    loop.call_soon_threadsafe(started.set)
                    try:
                        if not release.wait(5):
                            raise AssertionError("event loop never released the copy")
                        published.write_bytes(b"published")
                        if late_failure:
                            raise OSError("late copy failure")
                        return str(published)
                    finally:
                        settled.set()

                result = AgentResult(events=[
                    ChatEvent(kind="attachment", content="first.png"),
                    ChatEvent(kind="attachment", content="second.png"),
                    ChatEvent(kind="text", content="Done"),
                ])
                with (
                    mock.patch.object(agent, "attachment_source_path", side_effect=lambda path, _ws: path),
                    mock.patch.object(agent, "prepare_attachment_path", blocked_prepare),
                    mock.patch.object(bot, "_track_attachment_upload") as upload,
                ):
                    task = asyncio.create_task(bot._send_result("chat", temporary, result, uid="u1"))
                    try:
                        await asyncio.wait_for(started.wait(), 2)
                        task.cancel("first stop")
                        await asyncio.sleep(0)
                        self.assertFalse(task.done())
                        task.cancel("second stop")
                        await asyncio.sleep(0)
                        self.assertFalse(task.done())
                        self.assertFalse(published.exists())
                        release.set()
                        with self.assertRaises(asyncio.CancelledError) as caught:
                            await task
                        self.assertEqual(caught.exception.args, ("first stop",))
                        self.assertTrue(settled.is_set())
                        self.assertEqual(published.read_bytes(), b"published")
                    finally:
                        release.set()
                        await asyncio.gather(task, return_exceptions=True)
                    self.assertEqual(attempts, ["first.png"])
                    upload.assert_not_called()
                    bot.send_text.assert_not_awaited()
        await asyncio.sleep(0)
        self.assertEqual(diagnostics, [])
