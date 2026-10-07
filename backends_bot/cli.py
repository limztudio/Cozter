"""CLI adapter: turns the launching terminal into a chat surface.

Activated by running ``python -m Cozter -cli`` (or ``--cli``). No tokens,
no networking - the bot reads commands and messages from stdin and prints
replies to stdout. Used for local development and for users who don't
want to set up Telegram or Slack.

Commands work the same as the other adapters: lines starting with ``/``
are slash commands, and registered commands may also use ``\\`` as a
message-friendly prefix. Everything else is treated as a chat message
routed to the AI agent. Status events emitted during an AI turn print
directly (since the terminal can't edit prior lines).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
import threading

from ..config import DEFAULT_MESSAGE_QUEUE_SIZE, DEFAULT_RECENT_WORKSPACE_LIMIT
from ..utils import await_cancelled, create_background_task
from .base import (
    BotContext,
    BotPlatform,
    MessageHandle,
)

logger = logging.getLogger(__name__)

# Faux local-user id: no collision with Telegram/Slack ids.
_LOCAL_ID = "local"


class CliBot(BotPlatform):
    """Local interactive REPL over stdin/stdout."""

    def __init__(
        self,
        *,
        recent_limit: int = DEFAULT_RECENT_WORKSPACE_LIMIT,
        max_queue_size: int = DEFAULT_MESSAGE_QUEUE_SIZE,
    ):
        super().__init__(
            [_LOCAL_ID],
            recent_limit=recent_limit,
            max_queue_size=max_queue_size,
        )
        self._stop_requested = asyncio.Event()
        self._input_task: asyncio.Task | None = None

    @property
    def platform_id(self) -> str:
        # Stable, prefixed: persists across sessions, disjoint from chat ids.
        return f"cli:{_LOCAL_ID}"

    def authorized(self, user_id: str, _chat_id: str) -> bool:
        # Local binary implies shell access; authorize unconditionally.
        return True

    # send/edit

    async def send_text(
        self, chat_id: str, text: str, *, rich: bool = False,
    ) -> MessageHandle | None:
        if not text:
            return None
        # Trailing newline: don't run into the next prompt.
        print(text)
        # Returning None prints each event as it arrives.
        return None

    async def edit_text(
        self, handle: MessageHandle, text: str, *, rich: bool = False,
    ) -> None:
        # No-op: unreachable for the CLI; kept for the contract.
        pass

    async def delete_message(self, handle: MessageHandle) -> None:
        pass

    async def send_file(self, chat_id: str, path: str) -> None:
        # Local-only files: absolute path.
        print(f"[Attached file: {os.path.abspath(path)}]")

    async def send_status(self, chat_id: str, text: str) -> None:
        """Print transient progress lines in dim gray so they're visually
        distinct from the agent's final reply.

        Falls back to plain text if the terminal can't render ANSI.
        """
        if not text:
            return
        if _ANSI_ENABLED:
            # ESC[2m = dim, ESC[90m = gray.
            print(f"\x1b[2;90m{text}\x1b[0m")
        else:
            print(text)

    # startup/shutdown

    async def start(self) -> None:
        _prepare_console()
        _install_force_exit_on_sigint()
        print("=== Cozter CLI mode ===")
        print(
            "Type /new or /open to select a workspace, /agent to switch"
            " agents, /help-like commands as usual."
        )
        print("Plain text goes to the AI. Ctrl-D or Ctrl-C exits.")
        print()
        # Load the staged reply first.
        await self.restore_reply_deliveries()
        self.start_detached_task_watcher()
        self._input_task = asyncio.create_task(self._input_loop())

    async def stop(self) -> None:
        self._stop_requested.set()
        await self.stop_detached_task_watcher()
        if self._input_task and not self._input_task.done():
            self._input_task.cancel()
            await await_cancelled(self._input_task)

    async def wait_until_exit(self) -> None:
        """Block the caller until the input loop terminates."""
        if self._input_task is None:
            return
        await await_cancelled(self._input_task)

    async def send_startup_messages(
        self, version: str, commit_date: str,
    ) -> None:
        # start() banner suffices; skip the greeting.
        return

    async def _input_loop(self) -> None:
        # Daemon-thread stdin.
        loop = asyncio.get_running_loop()
        line_q: asyncio.Queue[str | None] = asyncio.Queue()

        def _safe_post(value: str | None) -> bool:
            """Hand *value* to the loop; return False if it's already closed."""
            try:
                loop.call_soon_threadsafe(line_q.put_nowait, value)
                return True
            except RuntimeError:
                # Loop closed; nothing left to do.
                return False

        def _reader() -> None:
            # Prompt prints from the asyncio side.
            while True:
                try:
                    line = input()
                except (EOFError, KeyboardInterrupt):
                    _safe_post(None)
                    return
                except Exception:
                    # Unexpected: treat as EOF.
                    _safe_post(None)
                    return
                if not _safe_post(line):
                    return

        threading.Thread(target=_reader, daemon=True).start()

        try:
            while not self._stop_requested.is_set():
                # Reprint the prompt below the last output.
                print("> ", end="", flush=True)
                line = await line_q.get()
                if line is None:  # EOF / Ctrl-D
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    await self._handle_line(line)
                except KeyboardInterrupt:
                    print("(interrupted)")
                except Exception:
                    logger.exception("CLI dispatch failed")
                    print("Error: see logs for details.")
        finally:
            print("\nGoodbye.")

    async def _handle_line(self, line: str) -> None:
        if line.startswith("/"):
            parts = line[1:].split(None, 1)
            if not parts:
                return
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            ctx = self._ctx(command=cmd, args=args)
            await self.dispatch_command(ctx)
        else:
            parts = line[1:].split(None, 1) if line.startswith("\\") else []
            command = parts[0].split("@", 1)[0].lower() if parts else ""
            if _LOCAL_ID in self._pending_input or command in self._COMMANDS:
                # Picker answers and aliases must finish before the next line
                # can replace their pending handler. Agent turns stay concurrent
                # so /stop can still interrupt them.
                await self.dispatch_text(self._ctx(text=line))
            else:
                create_background_task(
                    self.dispatch_text(self._ctx(text=line)),
                    name="cli-dispatch",
                    log=logger,
                )

    def _ctx(
        self,
        *,
        text: str = "",
        command: str | None = None,
        args: str = "",
    ) -> BotContext:
        return self.make_context(
            _LOCAL_ID,
            _LOCAL_ID,
            text=text,
            command=command,
            args=args,
        )


# Helpers

def _prepare_console() -> None:
    """Make stdout/stderr UTF-8 so tool/file emojis don't crash cp1252."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")

    _enable_ansi()

    # Console: WARNING+ only.
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler,
        ):
            handler.setLevel(logging.WARNING)


# ANSI status colors: TTYs only.
_ANSI_ENABLED = False


def _enable_ansi() -> None:
    """Best-effort enable ANSI escape processing in the current console.

    Sets the module-level ``_ANSI_ENABLED`` flag based on whether stdout
    is a TTY and (on Windows) whether we can switch the console into
    Virtual Terminal Processing mode. Modern Windows Terminal and
    cmd.exe on Windows 10 1903+ support VT processing once enabled.
    """
    global _ANSI_ENABLED
    if not sys.stdout.isatty():
        _ANSI_ENABLED = False
        return
    if sys.platform != "win32":
        # POSIX terminals honor ANSI.
        _ANSI_ENABLED = True
        return
    # Windows: enable VT processing via SetConsoleMode.
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        STD_OUTPUT_HANDLE = -11
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            _ANSI_ENABLED = False
            return
        new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        _ANSI_ENABLED = bool(kernel32.SetConsoleMode(handle, new_mode))
    except (OSError, AttributeError):
        _ANSI_ENABLED = False


_force_exit_installed = False


def _install_force_exit_on_sigint() -> None:
    """Make Ctrl-C terminate the process immediately.

    With the daemon-thread reader the asyncio side cleans up fast, but
    we still skip the cancellation handshake on Ctrl-C so the user gets
    instant exit rather than a brief shutdown-message flicker.
    """
    global _force_exit_installed
    if _force_exit_installed:
        return
    _force_exit_installed = True

    def _force_exit() -> None:
        # Newline-prefixed; flush (os._exit skips the flush).
        with contextlib.suppress(Exception):
            print("\n(interrupted)", flush=True)
        os._exit(130)  # 128 + SIGINT

    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _force_exit)
    except (NotImplementedError, RuntimeError):
        # Windows: fall back to the synchronous signal API.
        signal.signal(signal.SIGINT, lambda *_: _force_exit())
