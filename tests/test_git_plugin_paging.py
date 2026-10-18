"""Git read pages preserve UTF-8 coverage without retaining whole patches."""

import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Cozter.agent_tools.plugins import _git_common as git
from Cozter.agent_tools.plugins import git_info
from Cozter.agent_tools.plugins.git_info import GitInfoTool, _PAGE_CONTENT_CHARS, _page_output
from Cozter.agent_tools.plugins.git_ops import GitOpsTool
from Cozter.agent_tools.plugins.git_sync import GitSyncTool


class GitTextWindowTests(unittest.IsolatedAsyncioTestCase):
    async def test_character_offsets_survive_split_utf8_and_invalid_bytes(self):
        raw = "zero 한글 🙂 tail".encode("utf-8") + b"\xff end"
        decoded = raw.decode("utf-8", errors="replace")
        for offset in (0, 6, len(decoded), len(decoded) + 2):
            with self.subTest(offset=offset):
                stream = asyncio.StreamReader()
                window = git.GitTextWindow(offset, 7)
                task = asyncio.create_task(window.drain(stream))
                for byte in raw:
                    stream.feed_data(bytes((byte,)))
                    await asyncio.sleep(0)
                stream.feed_eof()
                await task
                self.assertEqual(window.text(), decoded[offset:offset + 7])
                self.assertEqual(window.total_chars, len(decoded))

    async def test_large_discarded_prefix_and_tail_never_accumulate(self):
        class RepeatingStream:
            calls = 0

            async def read(self, size):
                self.calls += 1
                return b"x" * size if self.calls <= 256 else b""

        stream = RepeatingStream()
        window = git.GitTextWindow(12 * 1024 * 1024, 11_000)
        await window.drain(stream)
        self.assertEqual(window.total_chars, 16 * 1024 * 1024)
        self.assertEqual(window.retained_chars, 11_000)
        self.assertEqual(sum(map(len, window.parts)), 11_000)
        self.assertEqual(window.text(), "x" * 11_000)

    async def test_short_and_exactly_full_output_have_no_remainder_marker(self):
        for length in (0, 12, _PAGE_CONTENT_CHARS):
            with self.subTest(length=length):
                window = git.GitTextWindow(0, _PAGE_CONTENT_CHARS)
                window._accept_text("x" * length)
                self.assertEqual(_page_output(window, ""), "x" * length)

    async def test_eof_and_beyond_eof_do_not_offer_another_page(self):
        for offset in (12, 20):
            with self.subTest(offset=offset):
                window = git.GitTextWindow(offset, _PAGE_CONTENT_CHARS)
                window._accept_text("x" * 12)
                text = _page_output(window, "")
                self.assertIn("end of output", text)
                self.assertNotIn("next_offset=", text)
                self.assertNotIn("truncated", text)

    async def test_incomplete_pipe_drain_never_claims_end_of_output(self):
        window = git.GitTextWindow(0, _PAGE_CONTENT_CHARS)
        window._accept_text("known prefix")
        window.incomplete = True
        text = _page_output(window, "")
        self.assertIn("incomplete", text)
        self.assertNotIn("end of output", text)

    async def test_noninteractive_git_overrides_are_process_local(self):
        runner = mock.AsyncMock(return_value=("ok", "", 0))
        with (
            mock.patch.dict(os.environ, {"GCM_INTERACTIVE": "1", "GIT_TERMINAL_PROMPT": "1"}),
            mock.patch.object(git, "git_once", runner),
        ):
            await git.run_git(["git", "status"], ".")
            child_env = runner.await_args.args[2]
            self.assertEqual(child_env["GCM_INTERACTIVE"], "0")
            self.assertEqual(child_env["GIT_TERMINAL_PROMPT"], "0")
            self.assertEqual(os.environ["GCM_INTERACTIVE"], "1")
            self.assertEqual(os.environ["GIT_TERMINAL_PROMPT"], "1")

    async def test_readonly_git_is_noninteractive_without_changing_parent_environment(self):
        runner = mock.AsyncMock(return_value=("ok", "", 0))
        with (
            mock.patch.dict(os.environ, {
                "GCM_INTERACTIVE": "1", "GIT_TERMINAL_PROMPT": "1", "GIT_OPTIONAL_LOCKS": "1",
            }),
            mock.patch.object(git_info, "_git_once", runner),
        ):
            await GitInfoTool._run_git(["git", "-C", ".", "status"], ".")
            child_env = runner.await_args.args[2]
            self.assertEqual(child_env["GCM_INTERACTIVE"], "0")
            self.assertEqual(child_env["GIT_TERMINAL_PROMPT"], "0")
            self.assertEqual(child_env["GIT_OPTIONAL_LOCKS"], "0")
            self.assertEqual(os.environ["GCM_INTERACTIVE"], "1")
            self.assertEqual(os.environ["GIT_TERMINAL_PROMPT"], "1")
            self.assertEqual(os.environ["GIT_OPTIONAL_LOCKS"], "1")

    async def test_windowed_capture_keeps_cancellation_process_ownership(self):
        started = asyncio.Event()
        processes = []
        launch = git.create_captured_subprocess

        async def observed_launch(*args, **kwargs):
            proc = await launch(*args, **kwargs)
            processes.append(proc)
            started.set()
            return proc

        window = git.GitTextWindow(0, 20)
        executable = getattr(sys, "_base_executable", sys.executable)
        with tempfile.TemporaryDirectory() as path, mock.patch.object(
            git, "create_captured_subprocess", observed_launch,
        ):
            task = asyncio.create_task(git.git_once(
                [executable, "-u", "-c", "import time; print('started'); time.sleep(10)"],
                path, dict(os.environ), stdout_window=window,
            ))
            try:
                await asyncio.wait_for(started.wait(), timeout=3)
                async with asyncio.timeout(3):
                    while not window.total_chars:
                        await asyncio.sleep(.01)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=3)
                self.assertEqual(len(processes), 1)
                self.assertIsNotNone(processes[0].returncode)
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                for proc in processes:
                    if proc.returncode is None:
                        await git.kill_and_wait(proc)


@unittest.skipUnless(shutil.which("git"), "git not available")
class GitPluginPagingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self._git("init", "-q")
        self._git("config", "user.name", "Cozter tests")
        self._git("config", "user.email", "tests@example.invalid")
        (self.root / "sample.txt").write_text("original\n", encoding="utf-8")
        self._git("add", "sample.txt")
        self._git("-c", "commit.gpgsign=false", "commit", "-qm", "fixture")

    def _git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.root), *args], capture_output=True, check=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"},
        ).stdout.decode("utf-8", errors="replace")

    async def test_all_pages_reassemble_large_utf8_patch_without_loss(self):
        (self.root / "sample.txt").write_text(
            "".join(f"line {number:05} 한글 🙂 payload\n" for number in range(5000)),
            encoding="utf-8",
        )
        expected = await asyncio.to_thread(self._git,
            "-c", "core.quotepath=false", "diff", "--no-ext-diff", "--no-textconv", "HEAD",
        )
        self.assertGreater(len(expected), 110_000)
        pieces = []
        offset = 0
        while True:
            result = await GitInfoTool().run(str(self.root), {
                "action": "diff", "patch": True, "offset": offset,
            })
            self.assertLessEqual(len(result), git.MAX_OUTPUT_CHARS)
            payload, footer = result.rsplit("\n[git_info output ", 1)
            pieces.append(payload)
            self.assertIn("repository is unchanged", footer)
            match = re.search(r"next_offset=(\d+)", footer)
            if match is None:
                self.assertIn("end of output", footer)
                break
            following = int(match.group(1))
            self.assertEqual(following, offset + len(payload))
            self.assertGreater(following, offset)
            offset = following
            self.assertLess(len(pieces), 100)
        self.assertEqual("".join(pieces), expected)

    async def test_blame_range_returns_several_lines_in_one_read(self):
        (self.root / "sample.txt").write_text("first\nsecond\nthird\nfourth\n", encoding="utf-8")
        await asyncio.to_thread(self._git, "add", "sample.txt")
        await asyncio.to_thread(self._git, "-c", "commit.gpgsign=false", "commit", "-qm", "more lines")
        result = await GitInfoTool().run(str(self.root), {
            "action": "blame", "path": "sample.txt", "line": 2, "line_count": 2,
        })
        self.assertEqual(len(result.splitlines()), 2)
        self.assertIn("second", result)
        self.assertIn("third", result)
        self.assertNotIn("fourth", result)

    async def test_invalid_paging_or_range_does_not_launch_git(self):
        runner = mock.AsyncMock()
        with mock.patch.object(GitInfoTool, "_run_git", runner):
            for offset in (-1, True, 1.5, "12"):
                result = await GitInfoTool().run(str(self.root), {"action": "status", "offset": offset})
                self.assertTrue(result.startswith("Error:"), result)
            for count in (0, 201, True, 1.5):
                result = await GitInfoTool().run(str(self.root), {
                    "action": "blame", "path": "sample.txt", "line_count": count,
                })
                self.assertTrue(result.startswith("Error:"), result)
        runner.assert_not_awaited()

    async def test_zero_commit_diff_fallback_resets_window(self):
        empty = self.root / "empty"
        empty.mkdir()
        await asyncio.to_thread(
            subprocess.run, ["git", "-C", str(empty), "init", "-q"], capture_output=True, check=True,
        )
        (empty / "new.txt").write_text("new staged text\n", encoding="utf-8")
        await asyncio.to_thread(
            subprocess.run, ["git", "-C", str(empty), "add", "new.txt"], capture_output=True, check=True,
        )
        result = await GitInfoTool().run(str(empty), {"action": "diff", "patch": True})
        self.assertIn("+new staged text", result)
        self.assertNotIn("Error:", result)

    async def test_git_tools_declare_their_existing_twelve_thousand_char_budget(self):
        for tool in (GitInfoTool, GitOpsTool, GitSyncTool):
            self.assertEqual(tool.max_result_chars, 12_000)
        self.assertTrue(GitInfoTool.parallel_safe)
        self.assertFalse(getattr(GitOpsTool, "parallel_safe", False))
        self.assertFalse(getattr(GitSyncTool, "parallel_safe", False))
