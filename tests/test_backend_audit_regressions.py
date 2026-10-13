"""Backend regressions found while reviewing launch and attachment ownership."""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest import mock

from Cozter.backends_agent import base, claude_code, grok, _openai_agent as oa
from Cozter import utils


_FIXTURE_PYTHON = getattr(sys, "_base_executable", None) or sys.executable


class AttachmentMarkerTests(unittest.TestCase):
    def test_case_insensitive_marker_reaches_shared_vision_resolver(self):
        with tempfile.TemporaryDirectory() as workspace:
            path = os.path.join(workspace, "photo.png")
            with open(path, "wb") as handle:
                handle.write(b"image fixture")
            for marker in ("attachment saved to:", "ATTACHMENT SAVED TO:", "Attachment Saved To:"):
                with self.subTest(marker=marker):
                    self.assertEqual(
                        base.attachment_image_paths(f"[Photo {marker} photo.png]", workspace),
                        [os.path.realpath(path)],
                    )


class HttpAttachmentMarkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_uppercase_attachment_marker_reaches_http_payload(self):
        class Backend(oa.OpenAIChatBackend):
            name = "fixture"

            def _chat_endpoint(self):
                return "http://fixture.invalid/chat/completions"

        prompt = "[Photo ATTACHMENT SAVED TO: photo.png]"
        parts = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": "fixture"}}]
        with (
            mock.patch.object(oa, "_vision_parts_for_prompt", return_value=parts) as prepare,
            mock.patch.object(oa, "_stream_completion", return_value=("done", "", [])) as stream,
        ):
            await Backend()._run_agent(mock.Mock(), ".", prompt, "model", "deny", False, 0)
        prepare.assert_called_once_with(prompt, ".")
        self.assertEqual(stream.await_args.args[1]["messages"][1]["content"], parts)


class CapturedLaunchCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_captured_creation_reaps_process_before_return(self):
        original = base._create_piped_subprocess
        started, release = asyncio.Event(), asyncio.Event()
        processes = []

        async def delayed_creation(*args, **kwargs):
            proc = await original(*args, **kwargs)
            processes.append(proc)
            started.set()
            await release.wait()
            return proc

        with mock.patch.object(base, "_create_piped_subprocess", side_effect=delayed_creation):
            launch = asyncio.create_task(base.create_captured_subprocess(
                [_FIXTURE_PYTHON, "-B", "-c", "import time; time.sleep(120)"],
            ))
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                launch.cancel()
                await asyncio.sleep(0)
                launch.cancel()
                await asyncio.sleep(0)
                self.assertFalse(launch.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(launch, timeout=5)
                self.assertIsNotNone(processes[0].returncode)
            finally:
                release.set()
                if not launch.done():
                    launch.cancel()
                await asyncio.gather(launch, return_exceptions=True)
                for proc in processes:
                    if proc.returncode is None:
                        await utils.kill_and_wait(proc)
                    utils.close_subprocess_pipe(proc, 1)
                    utils.close_subprocess_pipe(proc, 2)

    async def test_claude_startup_cancel_uses_launcher_only_cleanup(self):
        started, release = asyncio.Event(), asyncio.Event()
        proc = mock.Mock(returncode=0)

        async def delayed_creation(*args, **kwargs):
            started.set()
            await release.wait()
            return proc

        with (
            mock.patch.object(base, "_create_piped_subprocess", side_effect=delayed_creation),
            mock.patch.object(claude_code, "_stop_claude_command_launcher", new=mock.AsyncMock()) as stop,
            mock.patch.object(base, "_reap_failed_prompt_subprocess", new=mock.AsyncMock()) as reap_tree,
        ):
            launch = asyncio.create_task(claude_code._run_claude_command(["fixture"], cwd="."))
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                launch.cancel()
                await asyncio.sleep(0)
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(launch, timeout=1)
            finally:
                release.set()
                if not launch.done():
                    launch.cancel()
                await asyncio.gather(launch, return_exceptions=True)
        stop.assert_awaited_once_with(proc)
        reap_tree.assert_not_awaited()

    async def test_grok_cancelled_creation_keeps_prompt_until_process_is_reaped(self):
        original = base._create_piped_subprocess
        started, release = asyncio.Event(), asyncio.Event()
        processes, prompt_paths = [], []

        async def delayed_creation(command, **kwargs):
            prompt_paths.append(command[command.index("--prompt-file") + 1])
            proc = await original(
                [_FIXTURE_PYTHON, "-B", "-c", "import time; time.sleep(120)"], **kwargs,
            )
            processes.append(proc)
            started.set()
            await release.wait()
            return proc

        with tempfile.TemporaryDirectory() as workspace, mock.patch.object(
            base, "_create_piped_subprocess", side_effect=delayed_creation,
        ):
            launch = asyncio.create_task(grok.GrokBackend().launch(workspace, "fixture", None, "deny"))
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                launch.cancel()
                await asyncio.sleep(0)
                self.assertTrue(os.path.exists(prompt_paths[0]))
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(launch, timeout=5)
                self.assertIsNotNone(processes[0].returncode)
                self.assertFalse(os.path.exists(prompt_paths[0]))
            finally:
                release.set()
                if not launch.done():
                    launch.cancel()
                await asyncio.gather(launch, return_exceptions=True)
                for proc in processes:
                    if proc.returncode is None:
                        await utils.kill_and_wait(proc)
                    utils.close_subprocess_pipe(proc, 1)
                    utils.close_subprocess_pipe(proc, 2)


if __name__ == "__main__":
    unittest.main()
