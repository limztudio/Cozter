"""CLI adapter: turns the launching terminal into a chat surface.

Activated by running ``python -m Cozter -cli`` (or ``--cli``). No tokens,
no networking - the bot reads commands and messages from stdin and prints
replies to stdout. Used for local development and for users who don't
want to set up Telegram or Slack.

Commands work the same as the other adapters: lines starting with ``/``
are slash commands, everything else is treated as a chat message routed
to the AI agent. Status events emitted during an AI turn print directly
(since the terminal can't edit prior lines).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading

from .base import (
    AttachmentInfo,
    BotContext,
    BotPlatform,
    MessageHandle,
)

logger = logging.getLogger(__name__)

# Single faux user for state-keying. Workspace/session files end up under
# this id so they don't collide with real Telegram user IDs (numeric) or
# Slack channel IDs ("C..."/"D...").
_LOCAL_ID = "local"


class CliBot(BotPlatform):
    """Local interactive REPL over stdin/stdout."""

    def __init__(
        self,
        *,
        recent_limit: int = 10,
        max_queue_size: int = 50,
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
        # Stable string so workspace/session state persists across CLI
        # sessions. Prefixed to keep it disjoint from Slack/Telegram ids.
        return f"cli:{_LOCAL_ID}"

    def authorized(self, user_id: str, chat_id: str) -> bool:
        # Anyone running this binary already has shell access; authorize
        # the lone local user unconditionally.
        return True

    # ----- send/edit primitives ------------------------------------------

    async def send_text(
        self, chat_id: str, text: str, *, rich: bool = False,
    ) -> MessageHandle | None:
        if not text:
            return None
        # Trailing newline so successive prints don't run together with
        # the next input prompt.
        print(text)
        # Returning None disables the editable-status code path in base
        # so on_event prints each event as it arrives.
        return None

    async def edit_text(
        self, handle: MessageHandle, text: str, *, rich: bool = False,
    ) -> None:
        # No-op: we never hand out MessageHandles, so this path is
        # unreachable for the CLI. Kept for the abstract-method contract.
        return None

    async def delete_message(self, handle: MessageHandle) -> None:
        return None

    async def send_file(self, chat_id: str, path: str) -> None:
        # Files only "exist" on the local filesystem; just point the user
        # at the absolute path.
        print(f"[Attached file: {os.path.abspath(path)}]")

    async def send_status(self, chat_id: str, text: str) -> None:
        """Print transient progress lines in dim gray so they're visually
        distinct from the agent's final reply.

        Falls back to plain text if the terminal can't render ANSI.
        """
        if not text:
            return
        if _ANSI_ENABLED:
            # ESC[2m = dim, ESC[90m = bright black ("gray").
            print(f"\x1b[2;90m{text}\x1b[0m")
        else:
            print(text)

    # ----- lifecycle ------------------------------------------------------

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
        self._input_task = asyncio.create_task(self._input_loop())

    async def stop(self) -> None:
        self._stop_requested.set()
        if self._input_task and not self._input_task.done():
            self._input_task.cancel()
            try:
                await self._input_task
            except asyncio.CancelledError:
                pass

    async def wait_until_exit(self) -> None:
        """Block the caller until the input loop terminates."""
        if self._input_task is None:
            return
        try:
            await self._input_task
        except asyncio.CancelledError:
            pass

    async def send_startup_messages(
        self, version: str, commit_date: str,
    ) -> None:
        # The start() banner already covers what the user needs; suppress
        # the per-platform startup message so the screen isn't cluttered
        # before the first input prompt.
        return None

    # ----- input loop -----------------------------------------------------

    async def _input_loop(self) -> None:
        # Drive stdin from a daemon thread so the asyncio loop never has
        # to wait for ``input()`` to return at shutdown. The daemon thread
        # is killed automatically when the interpreter exits, avoiding
        # the ``executor.shutdown(wait=True)`` hang that the previous
        # ``asyncio.to_thread`` version had on unhandled exceptions.
        loop = asyncio.get_running_loop()
        line_q: asyncio.Queue[str | None] = asyncio.Queue()

        def _safe_post(value: str | None) -> bool:
            """Hand *value* to the loop; return False if it's already closed."""
            try:
                loop.call_soon_threadsafe(line_q.put_nowait, value)
                return True
            except RuntimeError:
                # Loop was closed (process exiting); nothing left to do.
                return False

        def _reader() -> None:
            # Bare ``input()`` (no prompt arg) so the reader thread only
            # blocks on stdin. The "> " prompt is printed from the
            # asyncio side just before awaiting each line, so it always
            # lands AFTER the previous turn's AI output (status lines,
            # reply text) instead of being scrolled out of view by an
            # eager re-prompt.
            while True:
                try:
                    line = input()
                except (EOFError, KeyboardInterrupt):
                    _safe_post(None)
                    return
                except Exception:
                    # Unexpected (e.g. stdin closed). Treat as EOF.
                    _safe_post(None)
                    return
                if not _safe_post(line):
                    return

        threading.Thread(target=_reader, daemon=True).start()

        try:
            while not self._stop_requested.is_set():
                # Print the prompt fresh on each iteration so it always
                # appears at the bottom of the scrollback after the
                # previous turn's output finishes.
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
            # Fire-and-forget the AI dispatch so the input loop returns
            # immediately and the next line's "> " prompt shows up
            # right away. If a turn is already running, _dispatch_ai
            # sees the held task lock and routes the new message
            # through the per-user queue (printing the
            # "Queued (X/N)." feedback) instead of having it sit
            # silently in the CLI's line buffer waiting for the
            # current await to return.
            asyncio.create_task(
                self.dispatch_text(self._ctx(text=line)),
            )

    def _ctx(
        self,
        *,
        text: str = "",
        command: str | None = None,
        args: str = "",
        attachment: AttachmentInfo | None = None,
    ) -> BotContext:
        return BotContext(
            user_id=_LOCAL_ID,
            chat_id=_LOCAL_ID,
            text=text,
            command=command,
            args=args,
            attachment=attachment,
            platform=self,
        )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _prepare_console() -> None:
    """Make stdout/stderr UTF-8 so tool/file emojis don't crash cp1252."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            # Older Python or non-tty stream (e.g. redirected to a file
            # without a reconfigure-able encoding) - leave as-is.
            pass

    _enable_ansi()

    # Suppress INFO-level logging on the console so it doesn't interleave
    # with chat output. The file handler installed by setup_logging still
    # captures WARNING+ records.
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler,
        ):
            handler.setLevel(logging.WARNING)


# Whether to emit ANSI color sequences from ``send_status``. Decided once
# in ``_enable_ansi``; we disable for non-TTY stdout (piped/redirected)
# so escape codes don't appear literally in log files.
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
        # POSIX terminals universally honor ANSI for tty output.
        _ANSI_ENABLED = True
        return
    # Windows: try to enable VT processing via SetConsoleMode.
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
        # Newline-prefixed so the message doesn't run into the prompt;
        # flush=True because os._exit skips the normal stdout flush.
        try:
            print("\n(interrupted)", flush=True)
        except Exception:
            pass
        os._exit(130)  # 128 + SIGINT

    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _force_exit)
    except (NotImplementedError, RuntimeError):
        # Windows: add_signal_handler is unsupported. Fall back to the
        # synchronous signal API, which is enough for SIGINT here.
        signal.signal(signal.SIGINT, lambda *_: _force_exit())
