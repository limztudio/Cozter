"""Offline regressions for progress traffic, Signal deadlines, and shells."""

import asyncio
import os
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent, workspace
from Cozter.agent_tools.builtin import bash as bash_tool
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.backends_bot import base as bot_base
from Cozter.backends_bot.base import MessageHandle
from Cozter.backends_bot.signal import SignalBot
from Cozter.tests.helpers import TestBot


class _StatusProbe(TestBot):
    def __init__(self, path: str, *, editable: bool = True) -> None:
        super().__init__([])
        self.path = path
        self.editable = editable
        self.sent: list[str] = []
        self.edits: list[tuple[float, str]] = []
        self.first_dispatch_delay = 0.0
        self.first_edit = asyncio.Event()
        self.second_edit = asyncio.Event()

    async def _current_workspace_for_turn(self, _uid: str, _chat: str) -> str:
        return self.path

    async def send_text(self, chat: str, text: str, *, rich: bool = False):
        self.sent.append(text)
        if text == "Thinking..." and self.editable:
            return MessageHandle(chat, "thinking")
        return None

    async def edit_text(self, handle, text: str, *, rich: bool = False) -> None:
        if not self.edits and self.first_dispatch_delay:
            await asyncio.sleep(self.first_dispatch_delay)
        self.edits.append((asyncio.get_running_loop().time(), text))
        self.first_edit.set()
        if len(self.edits) == 2:
            self.second_edit.set()


class StatusCoalescingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = temporary.name
        run_config = mock.patch.object(workspace, "get_run_config", return_value=(
            "codex", "auto", "auto", "auto", "codex",
        ))
        run_config.start()
        self.addCleanup(run_config.stop)

    async def test_burst_is_throttled_and_latest_preview_survives(self) -> None:
        bot = _StatusProbe(self.path)
        bot.first_dispatch_delay = 0.03
        real_sleep = asyncio.sleep
        woke_early = False

        async def early_sleep(delay):
            nonlocal woke_early
            if delay > 0 and bot.first_edit.is_set() and not woke_early:
                woke_early = True
                await real_sleep(0)
            else:
                await real_sleep(delay)

        async def run(*args, on_event, **kwargs):
            await on_event(ChatEvent(kind="tool", content="step 0"))
            await asyncio.wait_for(bot.first_edit.wait(), timeout=0.2)
            for index in range(1, 9):
                await on_event(ChatEvent(kind="tool", content=f"step {index}"))
            await asyncio.wait_for(bot.second_edit.wait(), timeout=0.3)
            return AgentResult(events=[ChatEvent(kind="text", content="final")])

        with (
            mock.patch.object(agent, "run", side_effect=run),
            mock.patch.object(bot_base, "_STATUS_EDIT_INTERVAL_SEC", 0.06),
            mock.patch.object(bot_base.asyncio, "sleep", side_effect=early_sleep),
        ):
            await asyncio.wait_for(bot._run_turn("user", "chat", "work"), timeout=0.5)
        self.assertTrue(woke_early)
        self.assertEqual(len(bot.edits), 2)
        self.assertGreaterEqual(bot.edits[1][0] - bot.edits[0][0], 0.05)
        self.assertIn("step 8", bot.edits[-1][1])
        self.assertIn("final", bot.sent)

    async def test_final_reply_does_not_wait_for_pending_throttle(self) -> None:
        bot = _StatusProbe(self.path)

        async def run(*args, on_event, **kwargs):
            await on_event(ChatEvent(kind="tool", content="first"))
            await asyncio.wait_for(bot.first_edit.wait(), timeout=0.2)
            await on_event(ChatEvent(kind="tool", content="pending"))
            return AgentResult(events=[ChatEvent(kind="text", content="final")])

        with (
            mock.patch.object(agent, "run", side_effect=run),
            mock.patch.object(bot_base, "_STATUS_EDIT_INTERVAL_SEC", 60.0),
        ):
            await asyncio.wait_for(bot._run_turn("user", "chat", "work"), timeout=0.3)
        await asyncio.sleep(0)
        self.assertEqual(len(bot.edits), 1)
        self.assertIn("final", bot.sent)

    async def test_cancelled_turn_cancels_pending_status_worker(self) -> None:
        bot = _StatusProbe(self.path)
        pending = asyncio.Event()

        async def run(*args, on_event, **kwargs):
            await on_event(ChatEvent(kind="tool", content="first"))
            await asyncio.wait_for(bot.first_edit.wait(), timeout=0.2)
            await on_event(ChatEvent(kind="tool", content="pending"))
            pending.set()
            await asyncio.Event().wait()

        with (
            mock.patch.object(agent, "run", side_effect=run),
            mock.patch.object(bot_base, "_STATUS_EDIT_INTERVAL_SEC", 0.06),
        ):
            task = asyncio.create_task(bot._run_turn("user", "chat", "work"))
            try:
                await asyncio.wait_for(pending.wait(), timeout=0.3)
            finally:
                task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.08)
        self.assertEqual(len(bot.edits), 1)

    async def test_noneditable_slow_status_does_not_block_agent_or_reply(self) -> None:
        bot = _StatusProbe(self.path, editable=False)
        status_started = asyncio.Event()

        async def slow_status(chat, text):
            status_started.set()
            await asyncio.Event().wait()

        async def run(*args, on_event, **kwargs):
            await on_event(ChatEvent(kind="tool", content="progress"))
            await asyncio.wait_for(status_started.wait(), timeout=0.2)
            return AgentResult(events=[ChatEvent(kind="text", content="final")])

        with (
            mock.patch.object(bot, "send_status", side_effect=slow_status),
            mock.patch.object(agent, "run", side_effect=run),
        ):
            await asyncio.wait_for(bot._run_turn("user", "chat", "work"), timeout=0.3)
        self.assertIn("final", bot.sent)

    async def test_signal_shows_thinking_and_delivers_reply(self) -> None:
        bot = SignalBot([], jsonrpc_socket="tcp://127.0.0.1:7583")

        async def run(*args, on_event, **kwargs):
            await on_event(ChatEvent(kind="tool", content="progress"))
            await on_event(ChatEvent(kind="text", content="preview"))
            return AgentResult(events=[ChatEvent(kind="text", content="final")])

        with (
            mock.patch.object(bot, "_current_workspace_for_turn", return_value=self.path),
            mock.patch.object(bot, "send_text", new=mock.AsyncMock(
                return_value=MessageHandle("chat", "reply"),
            )) as send,
            mock.patch.object(bot, "edit_text", new=mock.AsyncMock()) as edit,
            mock.patch.object(bot, "delete_message", new=mock.AsyncMock()) as delete,
            mock.patch.object(bot, "send_status", new=mock.AsyncMock()) as status,
            mock.patch.object(agent, "run", side_effect=run),
        ):
            await asyncio.wait_for(bot._run_turn("user", "chat", "work"), timeout=0.3)
            await asyncio.sleep(0)
            self.assertEqual(send.await_count, 2)
            self.assertEqual(send.await_args_list[0].args[1], "Thinking...")
            self.assertEqual(send.await_args.args[1], "final")
            edit.assert_not_awaited()
            delete.assert_awaited_once_with(MessageHandle("chat", "reply"))
            status.assert_not_awaited()


class SignalWriteDeadlineTests(unittest.IsolatedAsyncioTestCase):
    def _bot(self, drain) -> tuple[SignalBot, mock.Mock]:
        bot = SignalBot([], jsonrpc_socket="tcp://127.0.0.1:7583")
        writer = mock.Mock()
        writer.drain = mock.AsyncMock(side_effect=drain)
        bot._jsonrpc_writer = writer
        bot._jsonrpc_connected = mock.Mock(return_value=True)
        return bot, writer

    async def test_stuck_drain_obeys_rpc_deadline_and_cleans_pending(self) -> None:
        async def stuck():
            await asyncio.Event().wait()

        bot, writer = self._bot(stuck)
        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(bot._rpc_request_once("send", timeout=0.02), timeout=0.3)
        writer.write.assert_called_once()
        self.assertEqual(bot._jsonrpc_pending, {})
        self.assertFalse(bot._jsonrpc_write_lock.locked())

    async def test_busy_write_lock_obeys_rpc_deadline_without_sending(self) -> None:
        async def drain():
            return None

        bot, writer = self._bot(drain)
        await bot._jsonrpc_write_lock.acquire()
        try:
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(bot._rpc_request_once("send", timeout=0.02), timeout=0.3)
        finally:
            bot._jsonrpc_write_lock.release()
        writer.write.assert_not_called()
        self.assertEqual(bot._jsonrpc_pending, {})

    async def test_cancellation_during_drain_cleans_pending(self) -> None:
        started = asyncio.Event()

        async def stuck():
            started.set()
            await asyncio.Event().wait()

        bot, _writer = self._bot(stuck)
        task = asyncio.create_task(bot._rpc_request_once("send", timeout=None))
        try:
            await asyncio.wait_for(started.wait(), timeout=0.2)
        finally:
            task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(bot._jsonrpc_pending, {})
        self.assertFalse(bot._jsonrpc_write_lock.locked())


class ShellSelectionTests(unittest.TestCase):
    def _select(self, platform: str, paths: dict[str, str]) -> list[str] | None:
        with (
            mock.patch.object(bash_tool, "os", SimpleNamespace(name=platform)),
            mock.patch.object(bash_tool.shutil, "which", side_effect=paths.get),
        ):
            return bash_tool._find_shell()

    def test_windowsapps_alias_selects_native_powershell(self) -> None:
        powershell = r"C:\Program Files\PowerShell\7\pwsh.exe"
        self.assertEqual(self._select("nt", {
            "bash": r"C:\Users\user\AppData\Local\Microsoft\WindowsApps\bash.exe",
            "pwsh": powershell,
        }), [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command"])

    def test_system32_wsl_launcher_uses_legacy_powershell_fallback(self) -> None:
        powershell = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
        self.assertEqual(self._select("nt", {
            "bash": r"C:\Windows\System32\bash.exe", "powershell": powershell,
        }), [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command"])

    def test_real_windows_bash_stays_preferred(self) -> None:
        bash = r"C:\Program Files\Git\bin\bash.exe"
        self.assertEqual(self._select("nt", {"bash": bash, "pwsh": "pwsh.exe"}), [bash, "-c"])

    def test_cmd_fallback_disables_autorun(self) -> None:
        cmd = r"C:\Windows\System32\cmd.exe"
        self.assertEqual(self._select("nt", {"cmd.exe": cmd}), [cmd, "/d", "/c"])

    def test_posix_shell_selection_is_preserved(self) -> None:
        self.assertEqual(self._select("posix", {"bash": "/bin/bash"}), ["/bin/bash", "-c"])
        self.assertEqual(self._select("posix", {"sh": "/bin/sh"}), ["/bin/sh", "-c"])
        self.assertIsNone(self._select("posix", {}))

    def test_shell_context_reports_selected_dialect(self) -> None:
        for shell, expected in (
            ([r"C:\PowerShell\pwsh.exe", "-Command"], "PowerShell syntax"),
            ([r"C:\Windows\cmd.exe", "/c"], "cmd syntax"),
            (["/bin/bash", "-c"], "POSIX shell syntax"),
            (None, "unavailable"),
        ):
            with self.subTest(shell=shell), mock.patch.object(bash_tool, "_find_shell", return_value=shell):
                self.assertIn(expected, bash_tool.shell_context())


class NativeShellSmokeTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(os.name == "nt", "Windows native shell smoke test")
    async def test_native_powershell_runs_without_profile_or_interactive_setup(self) -> None:
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell unavailable")
        with tempfile.TemporaryDirectory() as path, mock.patch.object(
            bash_tool, "_find_shell", return_value=[
                powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
            ],
        ):
            result = await asyncio.wait_for(bash_tool.BashTool().run(
                path, {"command": "Write-Output 'native shell probe'"},
            ), timeout=5)
        self.assertEqual(result.strip(), "native shell probe")


if __name__ == "__main__":
    unittest.main()
