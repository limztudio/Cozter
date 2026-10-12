"""Prompt delivery cannot wait for a CLI's captured startup output."""

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from Cozter.backends_agent import base as backend_base
from Cozter import utils


# Bypass the Windows venv redirector in native fixtures, so the handles under
# test belong to the Python process executing the fixture itself.
_FIXTURE_PYTHON = getattr(sys, "_base_executable", None) or sys.executable


class NpmShimResolutionTests(unittest.TestCase):
    def _fixture(self, root):
        script = os.path.join(root, "node_modules", "fixture", "loader.js")
        os.makedirs(os.path.dirname(script))
        with open(script, "w", encoding="utf-8") as handle:
            handle.write("// offline fixture")
        shim = os.path.join(root, "fixture.cmd")
        with open(shim, "w", encoding="utf-8") as handle:
            handle.write("\n".join(backend_base._NPM_NODE_SHIM_HEADER) + "\n")
            handle.write(
                'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & '
                '"%_prog%" "%dp0%\\node_modules\\fixture\\loader.js" %*\n'
            )
        return shim, script

    def test_standard_npm_shim_uses_bundled_node_before_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="npm shim ") as root:
            shim, script = self._fixture(root)
            node = os.path.join(root, "node.exe")
            with open(node, "wb") as handle:
                handle.write(b"offline existence fixture")
            with mock.patch.object(backend_base.shutil, "which") as which:
                self.assertEqual(backend_base._npm_node_shim_prefix(shim), [node, script])
            which.assert_not_called()

    def test_standard_npm_shim_uses_path_node_when_unbundled(self) -> None:
        with tempfile.TemporaryDirectory(prefix="npm shim ") as root:
            shim, script = self._fixture(root)
            with mock.patch.object(backend_base.shutil, "which", return_value="node-on-path") as which:
                self.assertEqual(backend_base._npm_node_shim_prefix(shim), ["node-on-path", script])
            which.assert_called_once_with("node")

    def test_custom_missing_script_missing_node_and_unreadable_shims_keep_fallback(self) -> None:
        for invalid in ("custom", "missing_script", "missing_node", "unreadable", "custom_header"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as root:
                shim, script = self._fixture(root)
                if invalid == "custom":
                    with open(shim, "w", encoding="utf-8") as handle:
                        handle.write("@echo off\nnode custom-loader.js %*\n")
                elif invalid == "custom_header":
                    with open(shim, "a", encoding="utf-8") as handle:
                        handle.write("echo custom side effect\n")
                elif invalid == "missing_script":
                    os.unlink(script)
                with (
                    mock.patch.object(backend_base.sys, "platform", "win32"),
                    mock.patch.object(backend_base.shutil, "which", side_effect=lambda name: shim if name == "fixture" else None),
                ):
                    if invalid == "unreadable":
                        with mock.patch("builtins.open", side_effect=OSError("fixture unreadable")):
                            prefix = backend_base.resolve_executable_prefix("fixture")
                    else:
                        prefix = backend_base.resolve_executable_prefix("fixture")
                self.assertEqual(prefix, ["cmd.exe", "/c", shim])


class PromptSubprocessTests(unittest.IsolatedAsyncioTestCase):
    def _track_temporary_input(self):
        original = tempfile.TemporaryFile
        files = []

        def create(*args, **kwargs):
            handle = original(*args, **kwargs)
            files.append(handle)
            return handle

        return files, mock.patch.object(backend_base.tempfile, "TemporaryFile", side_effect=create)

    async def _finish_fixture(self, proc):
        if proc.returncode is None:
            await utils.kill_and_wait(proc)
        utils.close_subprocess_pipe(proc, 1)
        utils.close_subprocess_pipe(proc, 2)

    async def test_prompt_utf8_bytes_and_eof_survive_parent_file_close(self) -> None:
        script = (
            "import sys\n"
            "data=sys.stdin.buffer.read()\n"
            "sys.stdout.buffer.write(data)\n"
            "sys.stdout.buffer.flush()\n"
            "print('EOF=' + str(sys.stdin.buffer.read() == b''), file=sys.stderr)\n"
        )
        for prompt in ("", "서울 / café / 😀\r\nline two\n\x00end"):
            with self.subTest(prompt=prompt):
                files, track = self._track_temporary_input()
                with track:
                    proc = await asyncio.wait_for(backend_base.create_prompt_subprocess(
                        [_FIXTURE_PYTHON, "-B", "-c", script], prompt,
                    ), timeout=5)
                try:
                    self.assertEqual(len(files), 1)
                    self.assertTrue(files[0].closed)
                    self.assertIsNone(proc.stdin)
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
                    self.assertEqual(proc.returncode, 0)
                    self.assertEqual(stdout, prompt.encode("utf-8"))
                    self.assertEqual(stderr.strip(), b"EOF=True")
                finally:
                    await self._finish_fixture(proc)

    async def test_startup_stdout_and_stderr_flood_do_not_block_prompt_launch(self) -> None:
        flood_bytes = 1024 * 1024
        script = (
            "import hashlib,json,sys\n"
            f"sys.stdout.buffer.write(b'o'*{flood_bytes})\n"
            "sys.stdout.buffer.flush()\n"
            f"sys.stderr.buffer.write(b'e'*{flood_bytes})\n"
            "sys.stderr.buffer.flush()\n"
            "data=sys.stdin.buffer.read()\n"
            "print(json.dumps({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}), flush=True)\n"
        )
        # A size threshold would leave the small-prompt wait_closed risk.
        for prompt in ("small prompt", "서울 😀\n" * 100_000):
            with self.subTest(prompt_bytes=len(prompt.encode("utf-8"))):
                proc = await asyncio.wait_for(backend_base.create_prompt_subprocess(
                    [_FIXTURE_PYTHON, "-B", "-c", script], prompt,
                ), timeout=5)
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
                    self.assertEqual(proc.returncode, 0)
                    self.assertEqual(stdout[:flood_bytes], b"o" * flood_bytes)
                    self.assertEqual(stderr, b"e" * flood_bytes)
                    self.assertEqual(json.loads(stdout[flood_bytes:]), {
                        "bytes": len(prompt.encode("utf-8")),
                        "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    })
                finally:
                    await self._finish_fixture(proc)

    async def test_spawn_failure_closes_temporary_input_and_preserves_error(self) -> None:
        files, track = self._track_temporary_input()
        failure = FileNotFoundError("fixture CLI is missing")
        with track, mock.patch.object(
            backend_base, "_create_piped_subprocess", side_effect=failure,
        ):
            with self.assertRaises(FileNotFoundError) as raised:
                await backend_base.create_prompt_subprocess(["fixture-cli"], "private fixture prompt")
        self.assertIs(raised.exception, failure)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].closed)

    async def test_prompt_encoding_failure_closes_input_without_spawning(self) -> None:
        files, track = self._track_temporary_input()
        with track, mock.patch.object(backend_base, "_create_piped_subprocess") as spawn:
            with self.assertRaises(UnicodeEncodeError):
                await backend_base.create_prompt_subprocess(["fixture-cli"], "\ud800")
        spawn.assert_not_called()
        self.assertTrue(files[0].closed)

    async def test_cancellation_during_creation_reaps_child_and_closes_input(self) -> None:
        original = backend_base._create_piped_subprocess
        created = []
        started, release = asyncio.Event(), asyncio.Event()
        files, track = self._track_temporary_input()

        async def delayed_spawn(*args, **kwargs):
            proc = await original(*args, **kwargs)
            created.append(proc)
            started.set()
            await release.wait()
            return proc

        with track, mock.patch.object(backend_base, "_create_piped_subprocess", side_effect=delayed_spawn):
            launch = asyncio.create_task(backend_base.create_prompt_subprocess(
                [_FIXTURE_PYTHON, "-B", "-c", "import time; time.sleep(120)"], "fixture prompt",
            ))
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                launch.cancel()
                await asyncio.sleep(0)
                launch.cancel()  # Repeated stop cannot interrupt ownership cleanup.
                await asyncio.sleep(0)
                self.assertFalse(launch.done())
                self.assertFalse(files[0].closed)
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(launch, timeout=5)
                self.assertIsNotNone(created[0].returncode)
                self.assertTrue(files[0].closed)
            finally:
                release.set()
                if not launch.done():
                    launch.cancel()
                await asyncio.gather(launch, return_exceptions=True)
                for proc in created:
                    await self._finish_fixture(proc)

    async def test_cancelled_creation_failure_preserves_cancellation_and_closes_input(self) -> None:
        started, release = asyncio.Event(), asyncio.Event()
        files, track = self._track_temporary_input()
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        diagnostics = []
        loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
        self.addCleanup(loop.set_exception_handler, previous_handler)

        async def fail_creation(*_args, **_kwargs):
            started.set()
            await release.wait()
            raise OSError("fixture spawn failed after cancellation")

        with track, mock.patch.object(backend_base, "_create_piped_subprocess", side_effect=fail_creation):
            launch = asyncio.create_task(backend_base.create_prompt_subprocess(["fixture-cli"], "fixture"))
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                launch.cancel()
                await asyncio.sleep(0)
                self.assertFalse(files[0].closed)
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(launch, timeout=1)
                self.assertTrue(files[0].closed)
                # Flush callbacks: a consumed spawn failure must not become
                # an "exception in shielded future" diagnostic on Python 3.14.
                await asyncio.sleep(0)
                self.assertEqual(diagnostics, [])
            finally:
                release.set()
                if not launch.done():
                    launch.cancel()
                await asyncio.gather(launch, return_exceptions=True)

    async def test_cancelled_creation_of_exited_parent_does_not_wait_for_held_pipes(self) -> None:
        started, release = asyncio.Event(), asyncio.Event()
        files, track = self._track_temporary_input()
        proc = mock.Mock(returncode=0)

        async def wait_for_held_pipes():
            await asyncio.Future()

        proc.wait = mock.AsyncMock(side_effect=wait_for_held_pipes)

        async def exited_parent(*_args, **_kwargs):
            started.set()
            await release.wait()
            return proc

        with (
            track,
            mock.patch.object(backend_base, "_create_piped_subprocess", side_effect=exited_parent),
            mock.patch.object(backend_base, "has_managed_process_group", return_value=False),
            mock.patch.object(backend_base, "close_subprocess_pipe") as close,
        ):
            launch = asyncio.create_task(backend_base.create_prompt_subprocess(["fixture-cli"], "fixture"))
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                launch.cancel()
                await asyncio.sleep(0)
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(launch, timeout=1)
                self.assertTrue(files[0].closed)
                proc.wait.assert_not_awaited()
                self.assertEqual(close.call_args_list, [mock.call(proc, 1), mock.call(proc, 2)])
            finally:
                release.set()
                if not launch.done():
                    launch.cancel()
                await asyncio.gather(launch, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
