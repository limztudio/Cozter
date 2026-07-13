"""Chat-platform abstraction and shared command handlers.

Each concrete BotPlatform (telegram, slack, ...) is responsible for:
  1. Starting/stopping its framework and dispatching platform events.
  2. Building a BotContext per event and calling into the shared
     ``dispatch_command`` / ``dispatch_text`` / ``dispatch_file`` hooks.
  3. Providing the send/edit/delete/file primitives through the abstract
     methods at the bottom of BotPlatform.

All command logic lives here on BotPlatform itself, so the per-platform
adapters only deal with framework plumbing.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
import re
import shutil
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from .. import (
    agent, backends_agent, colony, config, schedules, session, updater,
    workspace,
)
from ..utils import COZTER_DIR
from ..utils import await_cancelled
from ..utils import create_background_task
from ..utils import drain_queue as _drain_queue
from ..utils import load_json_object
from ..utils import save_json_object

logger = logging.getLogger(__name__)

_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".sh", ".bash", ".zsh", ".fish",
    ".html", ".htm", ".css", ".scss", ".xml", ".csv",
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".swift", ".kt", ".r", ".m",
    ".sql", ".graphql", ".proto",
    ".dockerfile", ".gitignore", ".env",
    ".log", ".diff", ".patch",
})
_INLINE_SIZE_LIMIT = 50_000
UPLOADS_DIR = "uploads"
NO_WORKSPACE_TEXT = (
    "No workspace selected (or it was deleted). Use /new or /open."
)


def ensure_upload_dir(workspace_path: str) -> str:
    """Return the workspace upload directory, creating it if needed."""
    upload_dir = os.path.join(workspace_path, COZTER_DIR, UPLOADS_DIR)
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir

# ---------------------------------------------------------------------------
# Message handle + attachment info
# ---------------------------------------------------------------------------

@dataclass
class MessageHandle:
    """A reference to a sent message that can be edited or deleted later."""
    chat_id: str
    message_id: str


@dataclass
class AttachmentInfo:
    """An incoming file upload already downloaded to the workspace."""
    local_path: str
    filename: str
    kind: str  # "document", "photo", "audio", "video", "voice", "video note"
    caption: str = ""


# ---------------------------------------------------------------------------
# BotContext - per-event facade handed to every command handler.
# ---------------------------------------------------------------------------

@dataclass
class BotContext:
    """Per-event context passed to command handlers.

    All identifiers are stringified so the same dicts/logic work on both
    Telegram (int ids) and Slack (U... string ids).
    """
    user_id: str
    chat_id: str
    text: str
    command: str | None
    args: str
    attachment: AttachmentInfo | None
    platform: BotPlatform
    # Set by dispatch_command when a pending text-input flow existed and
    # was cleared by this command arriving. /cancel uses this to avoid
    # treating a wizard-exit command as a request to clear AI work.
    had_pending: bool = False

    async def reply_text(
        self, text: str, *, rich: bool = False,
    ) -> MessageHandle | None:
        return await self.platform.send_text(
            self.chat_id, text, rich=rich,
        )

    async def edit_text(
        self, handle: MessageHandle, text: str, *, rich: bool = False,
    ) -> None:
        await self.platform.edit_text(handle, text, rich=rich)

    async def delete_message(self, handle: MessageHandle) -> None:
        await self.platform.delete_message(handle)

    async def send_file(self, path: str) -> None:
        await self.platform.send_file(self.chat_id, path)


# Handler signature: callback taking a BotContext.
Handler = Callable[[BotContext], Awaitable[None]]
QueueEntry = tuple[str, str, str, bool]


# ---------------------------------------------------------------------------
# BotPlatform - shared state and command logic.
# ---------------------------------------------------------------------------

class BotPlatform(ABC):
    """Base class: holds all command logic, leaves I/O to subclasses."""

    def __init__(
        self,
        notify_targets: list[str],
        *,
        recent_limit: int = 10,
        max_queue_size: int = 50,
    ):
        # notify_targets are the chat IDs the bot greets on startup and uses
        # as its authorization set by default. For Telegram these are user
        # IDs; for Slack they are channel IDs (see SlackBot).
        self.notify_targets: list[str] = [str(t) for t in notify_targets]
        self.recent_limit = recent_limit
        self.max_queue_size = max_queue_size

        # Per-user runtime state (all keyed by str user_id).
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._task_locks: dict[str, asyncio.Lock] = {}
        self._message_queues: dict[str, asyncio.Queue] = {}
        self._inject_queues: dict[str, asyncio.Queue] = {}
        # Users whose last agent reply ended with [[await]] — their queue
        # drain is paused until the next message from them arrives.
        self._awaiting_answer: set[str] = set()
        # Multi-step flow state: maps user_id -> next text-input callback.
        self._pending_input: dict[str, Handler] = {}
        # Scheduler state. Double-fire within one tick cycle is prevented
        # by the persisted ``last_fired`` timestamp on each schedule,
        # which also survives bot restarts to enable catch-up firing.
        self._scheduler_task: asyncio.Task | None = None
        # Set when an auto-update has been pulled and the process is
        # waiting for active AI replies to finish before restarting.
        self._update_check_pending = False
        self._update_restart_pending = False
        # Serializes read-modify-write on the persistent-queue file so
        # concurrent enqueue/complete calls don't clobber each other.
        self._queue_file_lock: asyncio.Lock = asyncio.Lock()
        # Users whose running task was already acknowledged by /cancel
        # or /stop, so the cancelled task should not send a second reply.
        self._cancel_acknowledged: set[str] = set()

    # ----- platform identity + I/O primitives (abstract) ------------------

    @property
    @abstractmethod
    def platform_id(self) -> str:
        """Stable id used to scope workspace/session state to this bot."""

    @abstractmethod
    async def start(self) -> None:
        """Start listening for events."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop listening and shut down."""

    @abstractmethod
    async def send_text(
        self, chat_id: str, text: str, *, rich: bool = False,
    ) -> MessageHandle | None:
        """Send a message; return a handle for later edit/delete if supported."""

    @abstractmethod
    async def edit_text(
        self, handle: MessageHandle, text: str, *, rich: bool = False,
    ) -> None:
        """Edit a previously sent message."""

    @abstractmethod
    async def delete_message(self, handle: MessageHandle) -> None:
        """Delete a previously sent message."""

    @abstractmethod
    async def send_file(self, chat_id: str, path: str) -> None:
        """Upload a file to the chat."""

    async def notify_users(self, message: str) -> None:
        """Best-effort broadcast of *message* to every notify target."""
        for target in self.notify_targets:
            try:
                await self.send_text(target, message)
            except Exception as e:
                logger.warning("Failed to notify %s: %s", target, e)

    async def send_status(self, chat_id: str, text: str) -> None:
        """Emit a transient progress/status message ("» <tool>", etc.).

        Default: delegate to ``send_text``. Platforms that can render
        these visually distinct from the final reply (e.g. CLI mode
        with ANSI dim grey) should override.
        """
        await self.send_text(chat_id, text)

    async def _send_text_best_effort(
        self, chat_id: str, text: str, *, rich: bool = False,
    ) -> bool:
        """Send text and return whether it succeeded, swallowing I/O errors."""
        try:
            await self.send_text(chat_id, text, rich=rich)
        except Exception:
            return False
        return True

    def _start_queue_drain(self, uid: str) -> None:
        create_background_task(
            self._drain_message_queue(uid),
            name=f"{self.platform_id}:drain:{uid}",
            log=logger,
        )

    async def send_startup_messages(
        self, version: str, commit_date: str,
    ) -> None:
        """Announce startup to each notify target.

        Default: generic "Cozter started" broadcast. Platforms where the
        notify target is a specific user (Telegram) may override to
        include per-user workspace info.
        """
        msg = (
            f"Cozter started.\nVersion: {version}\nUpdated: {commit_date}"
        )
        await self.notify_users(msg)

    async def begin_update_check(self) -> None:
        """Pause new AI turns while an auto-update check/pull is in flight."""
        self._update_check_pending = True
        await self.stop_scheduler()

    async def cancel_update_check(self) -> None:
        """Resume normal processing after an update check found no update."""
        self._update_check_pending = False
        if not self._update_restart_pending:
            await self.start_scheduler()
            for uid in list(self._message_queues):
                self._start_queue_drain(uid)

    async def begin_update_restart(self) -> None:
        """Pause new AI turns while an auto-update restart is pending."""
        self._update_check_pending = False
        self._update_restart_pending = True
        await self.stop_scheduler()

    async def cancel_update_restart(self) -> None:
        """Resume normal processing if an update restart is aborted."""
        self._update_check_pending = False
        self._update_restart_pending = False
        await self.start_scheduler()
        for uid in list(self._message_queues):
            self._start_queue_drain(uid)

    def has_active_turns(self) -> bool:
        """Return True while any agent reply is still in progress."""
        if any(not task.done() for task in self._running_tasks.values()):
            return True
        return any(lock.locked() for lock in self._task_locks.values())

    def stuck_turn_diagnostics(self) -> str:
        """Human-readable dump of turn-tracking state, for stuck waits.

        Called when the update loop's idle wait blows past its ceiling
        (``update_idle_timeout``) so the critical log names the exact
        uid(s) still holding state, plus whether it's a not-yet-done
        task, a held lock, or both — the key evidence for diagnosing a
        wedged ``has_active_turns()`` that would otherwise only surface
        as repeated ``Delaying update check`` lines.
        """
        tasks = {
            uid: task for uid, task in self._running_tasks.items()
            if not task.done()
        }
        locks = {
            uid for uid, lock in self._task_locks.items() if lock.locked()
        }
        parts: list[str] = []
        if tasks:
            # Report the uid (the dict key) plus the task's own name and
            # done/cancelled flags for completeness. The task itself is
            # the coroutine running _run_turn; its default asyncio name
            # is "Task-N", but cancelled() distinguishes a task that was
            # signalled to stop but hasn't unwound its finally yet.
            names = [
                f"{uid}(cancelled={task.cancelled()})"
                for uid, task in tasks.items()
            ]
            parts.append("running_tasks=[" + ", ".join(names) + "]")
        if locks:
            parts.append("held_locks=[" + ", ".join(sorted(locks)) + "]")
        return "; ".join(parts) or "<no stuck state found>"

    # ----- event dispatch hooks (called by platform adapters) -------------

    def authorized(self, user_id: str, chat_id: str) -> bool:
        """Return True if an event from *(user_id, chat_id)* is allowed.

        Default scopes by user_id (Telegram semantics). Platforms that
        scope by channel instead (Slack) override to check *chat_id*.
        """
        return str(user_id) in self.notify_targets

    async def dispatch_command(self, ctx: BotContext) -> None:
        """Entry point for slash commands."""
        if not self.authorized(ctx.user_id, ctx.chat_id):
            logger.warning(
                "Unauthorized command from user=%s chat=%s",
                ctx.user_id, ctx.chat_id,
            )
            return
        handler = self._COMMANDS.get((ctx.command or "").lower())
        if handler is None:
            pending = self._pending_input.pop(ctx.user_id, None)
            if pending is not None:
                ctx.text = "/" + (ctx.command or "")
                if ctx.args:
                    ctx.text += f" {ctx.args}"
                ctx.command = None
                ctx.args = ""
                await pending(ctx)
                return
            await ctx.reply_text(f"Unknown command: /{ctx.command}")
            return
        # Any new command cancels a pending text-input flow. /cancel
        # uses ctx.had_pending to decide its reply.
        ctx.had_pending = (
            self._pending_input.pop(ctx.user_id, None) is not None
        )
        await handler(self, ctx)

    async def dispatch_text(self, ctx: BotContext) -> None:
        """Entry point for plain text and backslash command aliases."""
        if not self.authorized(ctx.user_id, ctx.chat_id):
            return
        if not ctx.text or not ctx.text.strip():
            return

        # Message-based surfaces such as Slack may reserve slash commands
        # for workspace-installed integrations.  Accept ``\open`` (and the
        # other registered names) as ordinary-message aliases so users do
        # not need a separately installed Slack slash command.  Only promote
        # known commands: arbitrary backslash-prefixed text is valid chat
        # input, especially for paths and LaTeX.
        text = ctx.text.strip()
        if text.startswith("\\"):
            parts = text[1:].split(None, 1)
            command = parts[0].split("@", 1)[0] if parts else ""
            if command.lower() in self._COMMANDS:
                ctx.text = ""
                ctx.command = command
                ctx.args = parts[1] if len(parts) > 1 else ""
                await self.dispatch_command(ctx)
                return

        pending = self._pending_input.pop(ctx.user_id, None)
        if pending is not None:
            # Multi-step flow continuation; handler decides whether to
            # re-arm itself by calling self._expect_input.
            await pending(ctx)
            return
        await self._ai_chat(ctx)

    async def dispatch_file(self, ctx: BotContext) -> None:
        """Entry point for file/attachment uploads."""
        if not self.authorized(ctx.user_id, ctx.chat_id):
            return
        if ctx.attachment is None:
            return
        # File uploads cancel any pending text-input flow to avoid surprises.
        self._pending_input.pop(ctx.user_id, None)
        await self._ai_file(ctx)

    def _expect_input(self, user_id: str, callback: Handler) -> None:
        """Arm a one-shot text-input handler for *user_id*."""
        self._pending_input[user_id] = callback

    def _ensure_task_lock(self, uid: str) -> asyncio.Lock:
        lock = self._task_locks.get(uid)
        if lock is None:
            lock = asyncio.Lock()
            self._task_locks[uid] = lock
        return lock

    def _ensure_message_queue(
        self, uid: str, *, min_size: int = 0,
    ) -> asyncio.Queue:
        q = self._message_queues.get(uid)
        maxsize = max(self.max_queue_size, min_size)
        if q is None:
            q = asyncio.Queue(maxsize=maxsize)
            self._message_queues[uid] = q
        elif q.maxsize and q.maxsize < maxsize:
            replacement: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
            while not q.empty():
                replacement.put_nowait(q.get_nowait())
            q = replacement
            self._message_queues[uid] = q
        return q

    @staticmethod
    def _pop_next_queue_entry(
        q: asyncio.Queue, *, ephemeral_only: bool = False,
    ) -> QueueEntry | None:
        """Pop the next runnable queue entry.

        When a collaborative turn is awaiting an answer, normal chat
        entries must remain paused, but scheduled ``/reserve`` entries
        are independent ephemeral runs and may continue.
        """
        if not ephemeral_only:
            return q.get_nowait()

        selected: QueueEntry | None = None
        buffered: list[QueueEntry] = []
        while not q.empty():
            entry = q.get_nowait()
            if selected is None and entry[3]:
                selected = entry
            else:
                buffered.append(entry)

        for entry in buffered:
            q.put_nowait(entry)
        return selected

    @staticmethod
    def _promote_queue_entry(
        q: asyncio.Queue, entry_id: str,
    ) -> None:
        """Move a queued entry to the front, preserving all others."""
        selected: QueueEntry | None = None
        buffered: list[QueueEntry] = []
        while not q.empty():
            entry = q.get_nowait()
            if selected is None and entry[2] == entry_id:
                selected = entry
            else:
                buffered.append(entry)

        if selected is not None:
            q.put_nowait(selected)
        for entry in buffered:
            q.put_nowait(entry)

    async def _persist_promote(
        self, uid: str, entry_id: str,
    ) -> None:
        """Move a persisted queue entry to the front for restart parity."""
        async with self._queue_file_lock:
            data = self._read_queue_file()
            entries = self._queue_entries(data.get(uid))
            selected: dict | None = None
            remaining: list[dict] = []
            for entry in entries:
                if selected is None and entry.get("id") == entry_id:
                    selected = entry
                else:
                    remaining.append(entry)
            if selected is None:
                return
            data[uid] = [selected, *remaining]
            self._write_queue_file(data)

    async def _resume_awaiting_answer(
        self, uid: str, q: asyncio.Queue, entry_id: str, *,
        reason: str,
    ) -> bool:
        """Clear an answer pause and promote the answer before backlog."""
        if uid not in self._awaiting_answer:
            return False
        self._awaiting_answer.discard(uid)
        self._promote_queue_entry(q, entry_id)
        await self._persist_promote(uid, entry_id)
        logger.info("User %s answered while %s; queue resumed", uid, reason)
        return True

    # ----- simple commands ------------------------------------------------

    async def cmd_start(self, ctx: BotContext) -> None:
        await ctx.reply_text("Cozter bot is running.")

    async def cmd_version(self, ctx: BotContext) -> None:
        ver, date = await asyncio.gather(
            asyncio.to_thread(updater.get_current_version),
            asyncio.to_thread(updater.get_last_commit_date),
        )
        await ctx.reply_text(f"Version: {ver}\nUpdated: {date}")

    async def cmd_doctor(self, ctx: BotContext) -> None:
        """Report readiness of every backend (CLI on PATH / server up)."""
        lines = ["Backend readiness:"]
        # Only the direct backends have anything to probe — flexible is a
        # meta-agent whose readiness is exactly its tiers' readiness.
        for name in backends_agent.DIRECT_BACKENDS:
            backend = backends_agent.get_backend(name)
            ok, detail = await asyncio.to_thread(backend.health_check)
            lines.append(f"  {'ok' if ok else 'XX'} {name}: {detail}")

        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if ws and os.path.isdir(ws):
            chat_backend = workspace.get_backend_name(ws)
            summary_backend = workspace.get_summary_backend_name(ws)
            lines.append("")
            lines.append(
                f"This workspace: chat={chat_backend},"
                f" summary={summary_backend}"
            )
            if chat_backend == workspace.FLEXIBLE_BACKEND:
                for tier, (name, model) in workspace.get_flexible_run_config(
                    ws,
                ).items():
                    lines.append(f"  flexible {tier:<4} = {name}/{model}")
        await ctx.reply_text("\n".join(lines))

    async def cmd_cancel(self, ctx: BotContext) -> None:
        if ctx.had_pending:
            await ctx.reply_text("Cancelled.")
            return

        uid = ctx.user_id
        was_awaiting = uid in self._awaiting_answer
        self._awaiting_answer.discard(uid)

        task = self._running_tasks.get(uid)
        task_running = task is not None and not task.done()
        if task_running:
            self._cancel_acknowledged.add(uid)
            task.cancel()

        drained: list = []
        _drain_queue(self._message_queues.get(uid), collect=drained)
        persisted = await self._clear_persistent_queue(uid)
        cleared = max(len(drained), persisted)

        if task_running:
            await ctx.reply_text("Cancelled.")
            return
        if was_awaiting or cleared:
            await ctx.reply_text("Cancelled.")
            return
        await ctx.reply_text("Nothing to cancel.")

    # ----- /new (dir-input flow) -----------------------------------------

    async def cmd_new(self, ctx: BotContext) -> None:
        current = workspace.get_current(ctx.user_id, self.platform_id)
        await ctx.reply_text(
            f"Current workspace: {current or '(none)'}\n\n"
            "Enter the full path for the new workspace directory"
            " (or /cancel):"
        )
        self._expect_input(ctx.user_id, self._receive_new_dir)

    async def _receive_new_dir(self, ctx: BotContext) -> None:
        path = ctx.text.strip()
        if os.path.exists(path):
            await ctx.reply_text(
                f"Directory already exists:\n{path}\n\n"
                "Please choose a different path (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_new_dir)
            return
        try:
            os.makedirs(path)
        except OSError as e:
            await ctx.reply_text(
                f"Failed to create directory: {e}\n\n"
                "Please try again (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_new_dir)
            return
        workspace.ensure_cozter_dir(path)
        workspace.select_workspace(ctx.user_id, path, self.platform_id)
        await ctx.reply_text(f"Workspace created and selected:\n{path}")

    # ----- /open ----------------------------------------------------------

    async def cmd_open(self, ctx: BotContext) -> None:
        if ctx.args.strip():
            await self._open_workspace_from_text(
                ctx, ctx.args.strip(), rearm_on_error=False,
            )
            return

        current = workspace.get_current(ctx.user_id, self.platform_id)
        recent = workspace.get_recent(ctx.user_id, self.recent_limit)
        lines = [f"Current workspace: {current or '(none)'}"]
        if recent:
            lines.append("\nRecent workspaces:")
            for i, r in enumerate(recent, 1):
                lines.append(f"  {i}. {r}")
        else:
            lines.append("\nNo recent workspaces.")
        lines.append(
            "\nEnter a directory path or number from the list (or /cancel):"
        )
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_open_dir)

    async def _receive_open_dir(self, ctx: BotContext) -> None:
        await self._open_workspace_from_text(
            ctx, ctx.text.strip(), rearm_on_error=True,
        )

    async def _open_workspace_from_text(
        self, ctx: BotContext, text: str, *, rearm_on_error: bool,
    ) -> None:
        recent = workspace.get_recent(ctx.user_id, self.recent_limit)
        if text.isdecimal():
            idx = int(text) - 1
            if 0 <= idx < len(recent):
                path = recent[idx]
            else:
                if rearm_on_error:
                    await ctx.reply_text(
                        "Invalid number. Please try again (or /cancel):"
                    )
                    self._expect_input(ctx.user_id, self._receive_open_dir)
                else:
                    await ctx.reply_text(
                        "Invalid number. Use /open to choose from recent"
                        " workspaces."
                    )
                return
        else:
            path = text
        if not os.path.isdir(path):
            if rearm_on_error:
                await ctx.reply_text(
                    f"Directory does not exist:\n{path}\n\n"
                    "Please enter a valid directory (or /cancel):"
                )
                self._expect_input(ctx.user_id, self._receive_open_dir)
            else:
                await ctx.reply_text(
                    f"Directory does not exist:\n{path}\n\n"
                    "Use /open to choose from recent workspaces."
                )
            return
        workspace.ensure_cozter_dir(path)
        workspace.select_workspace(ctx.user_id, path, self.platform_id)
        await ctx.reply_text(f"Workspace selected:\n{path}")

    # ----- /model ---------------------------------------------------------

    async def cmd_model(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        backend_name = workspace.get_backend_name(ws)
        # Flexible carries one model per difficulty tier rather than one
        # of its own, so there is nothing here to pick from.
        if backend_name == workspace.FLEXIBLE_BACKEND:
            await ctx.reply_text(self._flexible_summary(ws))
            return
        current = workspace.get_model(ws)
        options = await asyncio.to_thread(workspace.get_available_models, ws)
        lines = [
            f"Current model: {current} (backend: {backend_name})\n",
            "Available models:",
            *self._option_lines(options, current),
            "\nEnter a number or model name (or /cancel):",
        ]
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_model)

    async def _receive_model(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        text = ctx.text.strip()
        options = await asyncio.to_thread(workspace.get_available_models, ws)
        model = self._pick_option(text, options)
        if model is None:
            await ctx.reply_text(
                f"Unknown model: {text}\nTry again (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_model)
            return
        workspace.set_model(ws, model)
        await ctx.reply_text(f"Model set to: {model}")

    # ----- /summarymodel --------------------------------------------------

    async def cmd_summarymodel(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_summary_model(ws)
        summary_backend = workspace.get_summary_backend_name(ws)
        options = await asyncio.to_thread(
            workspace.get_available_summary_models, ws,
        )
        lines = [
            f"Current summary model: {current}"
            f" (summary agent: {summary_backend})\n",
            "Available models:",
            *self._option_lines(options, current),
            "\nEnter a number or model name (or /cancel):",
        ]
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_summarymodel)

    async def _receive_summarymodel(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        text = ctx.text.strip()
        options = await asyncio.to_thread(
            workspace.get_available_summary_models, ws,
        )
        model = self._pick_option(text, options)
        if model is None:
            await ctx.reply_text(
                f"Unknown model: {text}\nTry again (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_summarymodel)
            return
        workspace.set_summary_model(ws, model)
        await ctx.reply_text(f"Summary model set to: {model}")

    # ----- /agent ---------------------------------------------------------

    async def cmd_agent(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_backend_name(ws)
        options = workspace.AVAILABLE_BACKENDS
        lines = [
            f"Current agent: {current}\n",
            "Available agents:",
            *self._option_lines(options, current),
            f"\n{workspace.FLEXIBLE_BACKEND} splits each request by"
            " difficulty and routes the parts to its low/mid/high agents.",
            "\nEnter a number or agent name (or /cancel):",
        ]
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_agent)

    async def _receive_agent(self, ctx: BotContext) -> None:
        selected = await self._receive_backend_choice(
            ctx,
            options=workspace.AVAILABLE_BACKENDS,
            retry_handler=self._receive_agent,
        )
        if selected is None:
            return
        ws, name = selected
        workspace.set_backend_name(ws, name)
        if name == workspace.FLEXIBLE_BACKEND:
            await ctx.reply_text(
                f"Agent set to: {name}\n\n{self._flexible_summary(ws)}"
            )
            return
        _, model, summary_model, _, summary_backend = (
            workspace.get_run_config(ws)
        )
        # Always show the summary agent so the user doesn't read the
        # following ``Summary model`` line as belonging to *name* - it
        # belongs to whatever agent ``summary_backend`` points at.
        await ctx.reply_text(
            f"Agent set to: {name}\n"
            f"Model: {model}\n"
            f"Summary agent: {summary_backend}\n"
            f"Summary model: {summary_model}"
        )

    async def _receive_backend_choice(
        self,
        ctx: BotContext,
        *,
        options: list[str],
        retry_handler: Callable[[BotContext], Awaitable[None]],
    ) -> tuple[str, str] | None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return None
        text = ctx.text.strip()
        name = self._pick_option(text, options)
        if name is None:
            await ctx.reply_text(
                f"Unknown agent: {text}\nTry again (or /cancel):"
            )
            self._expect_input(ctx.user_id, retry_handler)
            return None
        return ws, name

    @staticmethod
    def _option_lines(options: list[str], current: str) -> list[str]:
        """Number a picker's options and mark the active one."""
        return [
            f"  {i}. {name}{' <-' if name == current else ''}"
            for i, name in enumerate(options, 1)
        ]

    @staticmethod
    def _flexible_summary(ws: str) -> str:
        """Describe how the flexible agent is currently wired up."""
        lines = [
            "The flexible agent has no single model — it splits your"
            " request and routes each part to a difficulty tier:",
            "",
        ]
        for tier, (name, model) in workspace.get_flexible_run_config(
            ws,
        ).items():
            lines.append(
                f"  {tier:<4} {name}/{model}"
                f" — {workspace.FLEXIBLE_TIER_DESCRIPTIONS[tier]}"
            )
        lines.append(
            "\nPlanning and merging run on the summary agent:"
            f" {workspace.get_summary_backend_name(ws)}"
            f"/{workspace.get_summary_model(ws)}"
        )
        lines.append(
            "\nRebind a tier with /agent_flexible_<tier> or"
            " /model_flexible_<tier>, where <tier> is "
            + "|".join(workspace.FLEXIBLE_TIERS)
            + "."
        )
        return "\n".join(lines)

    # ----- /agent_flexible_<tier>, /model_flexible_<tier> ------------------

    async def _show_flexible_agent(self, ctx: BotContext, tier: str) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_flexible_backend_name(ws, tier)
        # Flexible itself is excluded: a tier pointing back at the
        # meta-agent would plan and split forever.
        options = workspace.DIRECT_BACKENDS
        lines = [
            f"Flexible {tier} tier —"
            f" {workspace.FLEXIBLE_TIER_DESCRIPTIONS[tier]}",
            f"Current agent: {current}\n",
            "Available agents:",
            *self._option_lines(options, current),
            "\nEnter a number or agent name (or /cancel):",
        ]
        await ctx.reply_text("\n".join(lines))
        self._expect_input(
            ctx.user_id,
            functools.partial(self._receive_flexible_agent, tier=tier),
        )

    async def _receive_flexible_agent(
        self, ctx: BotContext, *, tier: str,
    ) -> None:
        selected = await self._receive_backend_choice(
            ctx,
            options=workspace.DIRECT_BACKENDS,
            retry_handler=functools.partial(
                self._receive_flexible_agent, tier=tier,
            ),
        )
        if selected is None:
            return
        ws, name = selected
        workspace.set_flexible_backend_name(ws, tier, name)
        await ctx.reply_text(
            f"Flexible {tier} agent set to: {name}\n"
            f"Flexible {tier} model: {workspace.get_flexible_model(ws, tier)}"
        )

    async def _show_flexible_model(self, ctx: BotContext, tier: str) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_flexible_model(ws, tier)
        backend_name = workspace.get_flexible_backend_name(ws, tier)
        options = workspace.get_available_flexible_models(ws, tier)
        lines = [
            f"Flexible {tier} tier —"
            f" {workspace.FLEXIBLE_TIER_DESCRIPTIONS[tier]}",
            f"Current model: {current} (agent: {backend_name})\n",
            "Available models:",
            *self._option_lines(options, current),
            "\nEnter a number or model name (or /cancel):",
        ]
        await ctx.reply_text("\n".join(lines))
        self._expect_input(
            ctx.user_id,
            functools.partial(self._receive_flexible_model, tier=tier),
        )

    async def _receive_flexible_model(
        self, ctx: BotContext, *, tier: str,
    ) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        text = ctx.text.strip()
        model = self._pick_option(
            text, workspace.get_available_flexible_models(ws, tier),
        )
        if model is None:
            await ctx.reply_text(
                f"Unknown model: {text}\nTry again (or /cancel):"
            )
            self._expect_input(
                ctx.user_id,
                functools.partial(self._receive_flexible_model, tier=tier),
            )
            return
        workspace.set_flexible_model(ws, tier, model)
        await ctx.reply_text(f"Flexible {tier} model set to: {model}")

    # ----- /summaryagent --------------------------------------------------

    async def cmd_summaryagent(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_summary_backend_name(ws)
        # The summary agent runs compaction, titling, and flexible's plan
        # and merge steps — all real turns, which flexible cannot serve.
        options = workspace.DIRECT_BACKENDS
        lines = [
            f"Current summary agent: {current}\n",
            "Available agents:",
            *self._option_lines(options, current),
            "\nEnter a number or agent name (or /cancel):",
        ]
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_summaryagent)

    async def _receive_summaryagent(self, ctx: BotContext) -> None:
        selected = await self._receive_backend_choice(
            ctx,
            options=workspace.DIRECT_BACKENDS,
            retry_handler=self._receive_summaryagent,
        )
        if selected is None:
            return
        ws, name = selected
        workspace.set_summary_backend_name(ws, name)
        summary_model = workspace.get_summary_model(ws)
        await ctx.reply_text(
            f"Summary agent set to: {name}\n"
            f"Summary model: {summary_model}"
        )

    # ----- /permission ----------------------------------------------------

    async def cmd_permission(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_permission(ws)
        options = workspace.AVAILABLE_PERMISSIONS
        lines = [f"Current permission: {current}\n", "Available modes:"]
        for i, p in enumerate(options, 1):
            marker = " <-" if p == current else ""
            desc = workspace.PERMISSION_DESCRIPTIONS[p]
            lines.append(f"  {i}. {p} - {desc}{marker}")
        ceiling = workspace.permission_ceiling()
        if ceiling != "full":
            lines.append(
                f"\n(Capped at '{ceiling}' by config max_permission;"
                " higher modes are rejected.)"
            )
        lines.append(
            "\nTip: chat surfaces can't show a per-tool approval dialog, so"
            " 'confirm' is best-effort. For ask-before-acting behavior on"
            " any backend use /style collaborative — it pauses the turn for"
            " your reply."
        )
        lines.append("\nEnter a number or mode name (or /cancel):")
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_permission)

    async def _receive_permission(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        text = ctx.text.strip().lower()
        options = workspace.AVAILABLE_PERMISSIONS
        perm = self._pick_option(text, options)
        if perm is None:
            await ctx.reply_text(
                f"Unknown mode: {text}\nTry again (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_permission)
            return
        try:
            workspace.set_permission(ws, perm)
        except ValueError as e:
            await ctx.reply_text(f"{e}\nTry again (or /cancel):")
            self._expect_input(ctx.user_id, self._receive_permission)
            return
        desc = workspace.PERMISSION_DESCRIPTIONS[perm]
        await ctx.reply_text(f"Permission set to: {perm}\n{desc}")

    # ----- /style ---------------------------------------------------------

    async def cmd_style(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_interaction_style(ws)
        options = workspace.AVAILABLE_STYLES
        lines = [
            f"Current interaction style: {current}\n",
            "Available styles:",
        ]
        for i, s in enumerate(options, 1):
            marker = " <-" if s == current else ""
            desc = workspace.STYLE_DESCRIPTIONS[s]
            lines.append(f"  {i}. {s} - {desc}{marker}")
        lines.append(
            "\nCollaborative makes the agent ask before big or ambiguous"
            " actions and pause for your reply; autonomous makes it decide"
            " and proceed. Applies to chat turns (scheduled runs are always"
            " autonomous)."
        )
        lines.append("\nEnter a number or style name (or /cancel):")
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_style)

    async def _receive_style(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        text = ctx.text.strip().lower()
        options = workspace.AVAILABLE_STYLES
        style = self._pick_option(text, options)
        if style is None:
            await ctx.reply_text(
                f"Unknown style: {text}\nTry again (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_style)
            return
        workspace.set_interaction_style(ws, style)
        desc = workspace.STYLE_DESCRIPTIONS[style]
        await ctx.reply_text(f"Interaction style set to: {style}\n{desc}")

    # ----- /effort --------------------------------------------------------

    async def cmd_effort(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_reasoning_effort(ws)
        await ctx.reply_text(
            f"Current reasoning effort: {current}%\n\n"
            "Pick a percentage 0-100. Higher = more reasoning,"
            " slower & costlier. 0 = no override (server picks).\n\n"
            "Each agent maps to its native scale:\n"
            "  codex:       1-19 minimal / 20-39 low / 40-59 medium"
            " / 60-79 high / 80-100 xhigh\n"
            "  llama:       1-24 minimal / 25-49 low / 50-74 medium"
            " / 75-100 high\n"
            "  claude_code: 1-19 low / 20-39 medium / 40-59 high"
            " / 60-79 xhigh / 80-100 max\n"
            "  copilot:     1-19 low / 20-39 medium / 40-59 high"
            " / 60-79 xhigh / 80-100 max\n\n"
            "Applies to your chat turns (not compaction/routing)."
            "\nEnter a number 0-100 (or /cancel):"
        )
        self._expect_input(ctx.user_id, self._receive_effort)

    async def _receive_effort(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        text = ctx.text.strip().rstrip("%").strip()
        try:
            value = int(text)
        except ValueError:
            await ctx.reply_text(
                f"Not a number: {ctx.text!r}\nEnter 0-100 (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_effort)
            return
        if not 0 <= value <= 100:
            await ctx.reply_text(
                f"Out of range: {value}\nEnter 0-100 (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_effort)
            return
        workspace.set_reasoning_effort(ws, value)
        await ctx.reply_text(
            f"Reasoning effort set to: {value}%"
            + ("\n(0 = no override - each backend uses its default)"
               if value == 0 else "")
        )

    # ----- /refresh -------------------------------------------------------

    async def cmd_refresh(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        codex_dir = os.path.join(ws, ".codex")
        cleared = False
        if os.path.isdir(codex_dir):
            shutil.rmtree(codex_dir, ignore_errors=True)
            logger.info("Cleared codex session dir: %s", codex_dir)
            cleared = True
        backend_name = workspace.get_backend_name(ws)
        if cleared:
            msg = (
                f"{backend_name} CLI session refreshed."
                " Your conversation history is preserved."
            )
        else:
            msg = (
                f"{backend_name} has no workspace-local state to clear."
                " Your conversation history is preserved."
            )
        await ctx.reply_text(msg)

    # ----- /compact -------------------------------------------------------

    async def cmd_compact(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        arg = ctx.args.strip().lower().split()
        first = arg[0] if arg else ""

        if first.isdecimal():
            interval = int(first)
            try:
                workspace.set_compact_interval(ws, interval)
            except ValueError as e:
                await ctx.reply_text(f"Error: {e}")
                return
            await ctx.reply_text(
                f"Compact interval set to {interval} messages."
            )
            return

        current = workspace.get_compact_interval(ws)
        sessions = session.list_sessions(ws)
        total_msgs = sum(s.get("message_count", 0) for s in sessions)
        lines = [
            f"Compact interval: {current} messages",
            f"Sessions: {len(sessions)} ({total_msgs} total messages)",
            "",
            "Compaction is automatic — each session compacts itself "
            f"every {current} messages.",
            "",
            "Usage:",
            "  /compact <number> - set interval",
        ]
        await ctx.reply_text("\n".join(lines))

    # ----- /context -------------------------------------------------------

    async def cmd_context(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        arg = ctx.args.strip().lower().split()
        first = arg[0] if arg else ""

        if first.isdecimal():
            budget = int(first)
            try:
                workspace.set_history_budget(ws, budget)
            except ValueError as e:
                await ctx.reply_text(f"Error: {e}")
                return
            await ctx.reply_text(
                f"Context budget set to {budget} characters."
            )
            return

        current = workspace.get_history_budget(ws)
        lines = [
            f"Context budget: {current} characters",
            "",
            "Colony memory, long-term memory, the session summary, and"
            " recent messages are prepended to each turn up to this"
            " budget; the oldest recent messages are dropped first to fit."
            " Measured in characters as a provider-agnostic proxy for"
            " tokens — raise it for large-context models, lower it for"
            " small local ones.",
            "",
            "Usage:",
            "  /context <number> - set the budget (characters)",
        ]
        await ctx.reply_text("\n".join(lines))

    # ----- /newsession ----------------------------------------------------

    async def cmd_newsession(self, ctx: BotContext) -> None:
        """Start a fresh session in the current workspace.

        Normally every turn continues the user's last-active session
        (persisted in ``.cozter/last_session.json``, so a bot restart
        resumes where the user left off). This command creates a new
        session and pins it as the new last_session, so the next
        message starts a clean conversation without affecting any
        existing session's history.
        """
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        data = session.create_session(ws)
        session.set_last_session(ws, ctx.user_id, data["id"])
        await ctx.reply_text(
            f"Started new session: {data['name']}\n"
            "Your next message goes into this fresh session.",
        )

    # ----- /sessions ------------------------------------------------------

    @staticmethod
    def _pick_session(choice: str, sessions: list[dict]) -> dict | None:
        """Resolve a /sessions selection: 1-based number, exact, or substring."""
        choice = choice.strip()
        if choice.isdecimal():
            idx = int(choice) - 1
            return sessions[idx] if 0 <= idx < len(sessions) else None
        low = choice.lower()
        for s in sessions:
            if s["name"].lower() == low:
                return s
        for s in sessions:
            if low and low in s["name"].lower():
                return s
        return None

    async def cmd_sessions(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        arg = ctx.args.strip()
        if arg:
            await self._switch_session(ctx, ws, arg)
            return
        sessions = session.list_sessions(ws)
        if not sessions:
            await ctx.reply_text(
                "No sessions yet — send a message to start one.",
            )
            return
        current = session.get_last_session(ws, ctx.user_id)
        lines = ["Sessions (newest first):"]
        for i, s in enumerate(sessions, 1):
            marker = " <-" if s["id"] == current else ""
            lines.append(
                f"  {i}. {s['name']} ({s['message_count']} msgs){marker}"
            )
        lines.append(
            "\nReply with a number or name to switch, /newsession for a"
            " fresh one (or /cancel):"
        )
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_session_switch)

    async def _receive_session_switch(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        await self._switch_session(ctx, ws, ctx.text.strip())

    async def _switch_session(
        self, ctx: BotContext, ws: str, choice: str,
    ) -> None:
        sessions = session.list_sessions(ws)
        target = self._pick_session(choice, sessions)
        if target is None:
            await ctx.reply_text(
                f"No session matches {choice!r}. Use /sessions to list them.",
            )
            return
        session.set_last_session(ws, ctx.user_id, target["id"])
        await ctx.reply_text(
            f"Switched to session: {target['name']}\n"
            "Your next message continues this conversation.",
        )

    # ----- /colony --------------------------------------------------------

    async def cmd_colony(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        arg = ctx.args.strip().lower().split()
        first = arg[0] if arg else ""

        if first == "now":
            await ctx.reply_text("Consolidating colony...")
            summary_model = workspace.get_summary_model(ws)
            summary_backend = workspace.get_summary_backend_name(ws)
            ok = await colony.consolidate(
                ws, summary_model, backend_name=summary_backend,
            )
            if ok:
                items = colony.get_items(ws)
                await ctx.reply_text(
                    f"Colony consolidated ({len(items)} item(s))."
                )
            else:
                await ctx.reply_text(
                    "Colony pass produced no changes (see logs)."
                )
            return

        if first.isdecimal():
            n = int(first)
            try:
                workspace.set_colony_interval(ws, n)
            except ValueError as e:
                await ctx.reply_text(f"Error: {e}")
                return
            await ctx.reply_text(f"Colony interval set to {n} compaction(s).")
            return

        items = colony.get_items(ws)
        count = colony.get_compact_count(ws)
        interval = workspace.get_colony_interval(ws)
        # interval - (count % interval) is in [1, interval]: when count
        # is a multiple of interval (start, or just-consolidated state),
        # a full interval of compactions is needed before the next pass.
        until = interval - (count % interval) if interval > 0 else 0
        lines = [
            f"Colony items: {len(items)}",
            f"Compactions since last pass: {count % interval}/{interval}",
            f"Compactions until next pass: {until}",
            "",
        ]
        if items:
            lines.append("Items:")
            for i, it in enumerate(items, 1):
                lines.append(f"  {i}. {it}")
            lines.append("")
        lines.extend([
            "Usage:",
            "  /colony <number> - set interval (compactions per pass)",
            "  /colony now - run consolidation immediately",
        ])
        await ctx.reply_text("\n".join(lines))

    # ----- /stop ----------------------------------------------------------

    async def cmd_stop(self, ctx: BotContext) -> None:
        # /stop unconditionally clears the await flag — the user is
        # giving up on the pending question, so the queue should be
        # free to drain (or stay drained, since /stop also empties it).
        was_awaiting = ctx.user_id in self._awaiting_answer
        self._awaiting_answer.discard(ctx.user_id)
        task = self._running_tasks.get(ctx.user_id)
        if task and not task.done():
            self._cancel_acknowledged.add(ctx.user_id)
            task.cancel()
            _drain_queue(self._message_queues.get(ctx.user_id))
            # Clear the persistent queue so cancelled work doesn't
            # come back on the next restart.
            await self._clear_persistent_queue(ctx.user_id)
            await ctx.reply_text("Cancelled.")
            return
        if was_awaiting:
            self._start_queue_drain(ctx.user_id)
            await ctx.reply_text(
                "Cleared pending question; resuming queued work."
            )
            return
        await ctx.reply_text("Nothing is running.")

    # ----- /inject --------------------------------------------------------

    async def cmd_inject(self, ctx: BotContext) -> None:
        text = ctx.args.strip()
        if not text:
            await ctx.reply_text("Usage: /inject <message>")
            return
        inject_q = self._inject_queues.get(ctx.user_id)
        if inject_q is None:
            await ctx.reply_text("No task is running.")
            return
        if inject_q.full():
            await ctx.reply_text("Inject queue full.")
            return
        await inject_q.put(text)
        await ctx.reply_text("Injected.")

    # ----- /reserve (recurring schedule wizard) --------------------------

    async def cmd_reserve(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        await ctx.reply_text(
            "Select days for this schedule.\n"
            "Use comma-separated names or numbers, or 'all'.\n"
            "1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun\n\n"
            "Examples: '1,3,5' or 'mon,wed,fri' or 'all'\n\n"
            "Enter days (or /cancel):"
        )
        self._expect_input(ctx.user_id, self._receive_reserve_days)

    async def _receive_reserve_days(self, ctx: BotContext) -> None:
        text = ctx.text.strip().lower()
        days = schedules.parse_days(text)
        if not days:
            await ctx.reply_text(
                "Could not parse days. Try again (e.g., 'mon,wed,fri'"
                " or '1,3,5') (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_reserve_days)
            return
        await ctx.reply_text(
            f"Days: {', '.join(days)}\n\n"
            "Enter time in HH:MM (24-hour) format (or /cancel):"
        )

        async def _time_cb(next_ctx: BotContext) -> None:
            await self._receive_reserve_time(next_ctx, days)

        self._expect_input(ctx.user_id, _time_cb)

    async def _receive_reserve_time(
        self, ctx: BotContext, days: list[str],
    ) -> None:
        time_str = schedules.parse_time(ctx.text.strip())
        if time_str is None:
            await ctx.reply_text(
                "Invalid time. Enter HH:MM (24-hour, e.g., 09:30)"
                " (or /cancel):"
            )

            async def _retry(next_ctx: BotContext) -> None:
                await self._receive_reserve_time(next_ctx, days)

            self._expect_input(ctx.user_id, _retry)
            return
        await ctx.reply_text(
            f"Time: {time_str}\n\n"
            "Enter the command to run at that time (or /cancel):"
        )

        async def _cmd_cb(next_ctx: BotContext) -> None:
            await self._receive_reserve_command(next_ctx, days, time_str)

        self._expect_input(ctx.user_id, _cmd_cb)

    async def _receive_reserve_command(
        self, ctx: BotContext, days: list[str], time_str: str,
    ) -> None:
        command = ctx.text.strip()
        if not command:
            await ctx.reply_text(
                "Command cannot be empty. Try again (or /cancel):"
            )

            async def _retry(next_ctx: BotContext) -> None:
                await self._receive_reserve_command(
                    next_ctx, days, time_str,
                )

            self._expect_input(ctx.user_id, _retry)
            return

        ws = await self._require_ws(ctx)
        if ws is None:
            return

        schedule = {
            "id": uuid.uuid4().hex[:12],
            "days": days,
            "time": time_str,
            "command": command,
            "created": datetime.now().isoformat(),
            "chat_id": ctx.chat_id,
            "user_id": ctx.user_id,
        }
        # Hold the workspace lock so a concurrent scheduler tick
        # writing last_fired can't clobber this insert.
        async with workspace.get_lock(ws):
            schedules.add_schedule(ws, ctx.user_id, schedule)
        await ctx.reply_text(
            f"Schedule created:\n"
            f"  Days: {', '.join(days)}\n"
            f"  Time: {time_str}\n"
            f"  Command: {command}"
        )

    # ----- /schedules (list/delete) --------------------------------------

    async def cmd_schedules(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        user_schedules = schedules.list_schedules(ws, ctx.user_id)
        if not user_schedules:
            await ctx.reply_text("No schedules.")
            return
        lines = ["Schedules:"]
        for i, s in enumerate(user_schedules, 1):
            raw_days = s.get("days", [])
            days = raw_days if isinstance(raw_days, list) else []
            days_str = ",".join(d for d in days if isinstance(d, str))
            lines.append(
                f"  {i}. [{days_str}] {s.get('time', '?')}"
                f" — {s.get('command', '')}"
            )
        lines.append(
            "\nEnter a number to delete, or /cancel to exit:"
        )
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_schedules)

    async def _receive_schedules(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        user_schedules = schedules.list_schedules(ws, ctx.user_id)
        text = ctx.text.strip()
        if not text.isdecimal():
            await ctx.reply_text(
                "Invalid input. Enter a number or /cancel:"
            )
            self._expect_input(ctx.user_id, self._receive_schedules)
            return
        idx = int(text) - 1
        if not (0 <= idx < len(user_schedules)):
            await ctx.reply_text(
                "Invalid number. Try again (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_schedules)
            return
        removed = user_schedules[idx]
        schedule_id = removed.get("id")
        if not isinstance(schedule_id, str) or not schedule_id:
            await ctx.reply_text(
                "That schedule entry is malformed and cannot be deleted "
                "by id. Fix or remove it from .cozter/schedules.json."
            )
            return
        async with workspace.get_lock(ws):
            schedules.remove_schedule(ws, ctx.user_id, schedule_id)
        raw_days = removed.get("days", [])
        days = raw_days if isinstance(raw_days, list) else []
        await ctx.reply_text(
            f"Removed: [{','.join(d for d in days if isinstance(d, str))}]"
            f" {removed.get('time', '?')} — {removed.get('command', '')}"
        )

    # ----- Persistent message queue --------------------------------------
    #
    # The in-memory asyncio.Queue is volatile — a bot restart (crash or
    # intentional) would drop everything the user had in flight. We
    # mirror each pending message to a per-platform JSON file so a
    # restart can resume from the last committed state.
    #
    # Writing order matters:
    #   1. enqueue: persist first, then put in in-memory queue
    #   2. complete: remove from persistent file after the turn settled
    #   3. cancellation (CancelledError) does NOT complete — on a clean
    #      shutdown the entry must survive restart; on /stop, cmd_stop
    #      clears the file explicitly.

    def _queue_file_path(self) -> str:
        os.makedirs(workspace.CONFIG_DIR, exist_ok=True)
        # Sanitize platform_id for the filesystem: ``cli:local`` and
        # ``slack:U123ABC`` would otherwise produce filenames with a
        # colon, which Windows rejects (and POSIX treats as legal but
        # cross-platform-hostile). Strip the full Windows-reserved set.
        safe = re.sub(
            r'[<>:"/\\|?*\x00-\x1f]', '_', self.platform_id,
        )
        return os.path.join(
            workspace.CONFIG_DIR, f"queue_{safe}.json",
        )

    def _read_queue_file(self) -> dict:
        return load_json_object(
            self._queue_file_path(), "queue file", logger,
        )

    @staticmethod
    def _queue_entries(value: object) -> list[dict]:
        if not isinstance(value, list):
            return []
        return [entry for entry in value if isinstance(entry, dict)]

    def _write_queue_file(self, data: dict) -> None:
        save_json_object(self._queue_file_path(), data)

    async def _persist_enqueue(
        self, uid: str, text: str, chat_id: str,
        *, ephemeral: bool = False,
    ) -> str:
        entry_id = uuid.uuid4().hex[:12]
        entry = {
            "id": entry_id, "text": text, "chat_id": chat_id,
        }
        if ephemeral:
            entry["ephemeral"] = True
        async with self._queue_file_lock:
            data = self._read_queue_file()
            entries = self._queue_entries(data.get(uid))
            entries.append(entry)
            data[uid] = entries
            self._write_queue_file(data)
        return entry_id

    async def _persist_complete(
        self, uid: str, entry_id: str,
    ) -> None:
        async with self._queue_file_lock:
            data = self._read_queue_file()
            entries = self._queue_entries(data.get(uid))
            if not entries:
                if uid in data:
                    data.pop(uid, None)
                    self._write_queue_file(data)
                return
            remaining = [e for e in entries if e.get("id") != entry_id]
            if len(remaining) == len(entries):
                return  # entry already gone
            if remaining:
                data[uid] = remaining
            else:
                data.pop(uid, None)
            self._write_queue_file(data)

    async def _clear_persistent_queue(self, uid: str) -> int:
        async with self._queue_file_lock:
            data = self._read_queue_file()
            entries = data.pop(uid, None)
            if entries is not None:
                self._write_queue_file(data)
                return len(self._queue_entries(entries)) or 1
        return 0

    async def restore_queues(self) -> None:
        """Rehydrate in-memory queues from disk and resume processing.

        Called once after the platform has started. For each user with
        persisted entries, refill the in-memory queue and spawn a drain
        task so the oldest entry runs first.
        """
        async with self._queue_file_lock:
            data = self._read_queue_file()
        if not data:
            return

        drained_users: list[str] = []
        for uid, raw_entries in data.items():
            entries = self._queue_entries(raw_entries)
            if not entries:
                continue
            self._ensure_task_lock(uid)
            existing_q = self._message_queues.get(uid)
            existing_size = existing_q.qsize() if existing_q else 0
            q = self._ensure_message_queue(
                uid, min_size=existing_size + len(entries),
            )
            for entry in entries:
                # Oldest-first preserves the user's original ordering.
                q.put_nowait((
                    entry.get("text", ""),
                    entry.get("chat_id", ""),
                    entry.get("id", ""),
                    bool(entry.get("ephemeral", False)),
                ))
            drained_users.append(uid)

        for uid in drained_users:
            self._start_queue_drain(uid)

    # ----- Scheduler loop ------------------------------------------------

    async def start_scheduler(self) -> None:
        """Kick off the background scheduler task. Idempotent."""
        if self._scheduler_task and not self._scheduler_task.done():
            return
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop_scheduler(self) -> None:
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            await await_cancelled(self._scheduler_task)
        self._scheduler_task = None

    async def _scheduler_loop(self) -> None:
        """Poll every 30s; fire schedules whose day+time match 'now'."""
        # Give the platform a moment to fully start.
        await asyncio.sleep(5)
        while True:
            try:
                await self._scheduler_tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler tick failed")
            await asyncio.sleep(30)

    async def _scheduler_tick(self) -> None:
        """Fire every schedule whose most-recent matching slot hasn't run.

        A schedule's "slot" is the most recent (day, time) match at or
        before ``now``. If that slot is newer than the schedule's
        persisted ``last_fired`` (or the schedule has never fired and
        the slot is newer than ``created``), fire it.

        Because ``last_fired`` lives in the session JSON on disk, this
        correctly catches up after a bot restart: any slot that should
        have run while the bot was down will fire once on the next tick.
        """
        now = datetime.now()

        # (uid, ws, sched, slot) tuples; sorted by creation order so
        # two schedules firing at the same slot run in the order the
        # user made them.
        to_fire: list[tuple[str, str, dict, datetime]] = []

        # Iterate workspace state (not notify_targets) so Slack — where
        # notify_targets are channels — correctly finds user sessions.
        for uid, ws in workspace.iter_current_workspaces(self.platform_id):
            if not os.path.isdir(ws):
                continue
            for sched in schedules.list_schedules(ws, uid):
                schedule_id = sched.get("id")
                if not isinstance(schedule_id, str) or not schedule_id:
                    continue
                slot = schedules.most_recent_slot(sched, now)
                if slot is None:
                    continue
                last_fired = schedules.parse_iso(sched.get("last_fired"))
                baseline = last_fired or schedules.parse_iso(
                    sched.get("created"),
                )
                # Never fire for slots at or before the schedule's
                # creation time (the user shouldn't see a fire at the
                # instant they created it).
                if baseline is not None and slot <= baseline:
                    continue
                to_fire.append((uid, ws, sched, slot))

        to_fire.sort(key=lambda item: item[2].get("created", ""))

        for uid, ws, sched, slot in to_fire:
            schedule_id = sched.get("id")
            if not isinstance(schedule_id, str) or not schedule_id:
                continue
            # Persist last_fired BEFORE queueing the command, so a
            # crash between marking and queueing at worst drops the
            # fire rather than firing it twice.
            async with workspace.get_lock(ws):
                schedules.update_schedule_fired(
                    ws, uid, schedule_id, slot.isoformat(),
                )
            await self._fire_schedule(uid, sched)

    async def _fire_schedule(self, uid: str, sched: dict) -> None:
        """Push a scheduled command onto the user's message queue.

        The queue entry is flagged ``ephemeral=True`` so the drain loop
        runs it in a fresh session that is deleted after the turn,
        rather than appending to whichever session was current at the
        time the schedule fires.
        """
        command = sched.get("command", "")
        chat_id = sched.get("chat_id", "")
        if not command or not chat_id:
            return

        # A user who was authorized when creating the schedule may have
        # been removed since — re-check so stale schedules don't bypass
        # authorization.
        if not self.authorized(uid, chat_id):
            logger.warning(
                "Skipping schedule for unauthorized user=%s chat=%s",
                uid, chat_id,
            )
            return

        # After a fresh bot start the scheduler can fire before the user
        # types anything, so create the per-user lock and queue on demand.
        self._ensure_task_lock(uid)
        q = self._ensure_message_queue(uid)

        # Check capacity BEFORE announcing — otherwise a user with a full
        # queue sees "⏰ Scheduled: X" immediately followed by "Queue full
        # — dropped scheduled command: X", which is confusing.
        if q.full():
            await self._send_text_best_effort(
                chat_id,
                f"Queue full — dropped scheduled command: {command}",
            )
            return

        try:
            await self.send_text(
                chat_id, f"⏰ Scheduled: {command}",
            )
        except Exception:
            logger.warning("Failed to announce scheduled command")

        entry_id = await self._persist_enqueue(
            uid, command, chat_id, ephemeral=True,
        )
        await q.put((command, chat_id, entry_id, True))

        # Kick the queue drainer in a background task; if a turn is
        # already running, _drain_message_queue breaks out immediately
        # and the running handler's own drain-after-turn picks up later.
        self._start_queue_drain(uid)

    # ----- AI chat + file -------------------------------------------------

    async def _require_ws(self, ctx: BotContext) -> str | None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws or not os.path.isdir(ws):
            await ctx.reply_text(NO_WORKSPACE_TEXT)
            return None
        return ws

    async def _ai_chat(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        await self._dispatch_ai(ctx, ctx.text.strip())

    async def _ai_file(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        att = ctx.attachment
        assert att is not None  # dispatch_file already checked

        rel_path = os.path.relpath(att.local_path, ws)
        parts: list[str] = []
        if att.caption:
            parts.append(att.caption)
        parts.append(
            f"[{att.kind.capitalize()} attachment saved to: {rel_path}]"
        )

        ext = os.path.splitext(att.filename)[1].lower()
        if ext in _TEXT_EXTENSIONS:
            try:
                with open(
                    att.local_path, encoding="utf-8", errors="replace",
                ) as f:
                    content = f.read()
                if len(content) <= _INLINE_SIZE_LIMIT:
                    parts.append(
                        f"[File contents of {att.filename}]\n"
                        f"{content}\n"
                        f"[End of file]"
                    )
                else:
                    parts.append(
                        f"[File too large to inline ({len(content):,} chars);"
                        f" read it from {rel_path}]"
                    )
            except OSError:
                pass

        await self._dispatch_ai(ctx, "\n".join(parts))

    # ----- dispatch + AI turn --------------------------------------------

    def _cleanup_turn(self, uid: str, lock: asyncio.Lock) -> None:
        self._running_tasks.pop(uid, None)
        self._inject_queues.pop(uid, None)
        self._cancel_acknowledged.discard(uid)
        lock.release()

    async def _dispatch_ai(self, ctx: BotContext, text: str) -> None:
        uid = ctx.user_id
        chat_id = ctx.chat_id
        if self._update_check_pending or self._update_restart_pending:
            self._ensure_task_lock(uid)
            q = self._ensure_message_queue(uid)
            if q.full():
                await ctx.reply_text("Queue full. Try again after restart.")
            else:
                entry_id = await self._persist_enqueue(
                    uid, text, chat_id,
                )
                await q.put((text, chat_id, entry_id, False))
                await self._resume_awaiting_answer(
                    uid, q, entry_id, reason="update was pending",
                )
                if self._update_restart_pending:
                    message = (
                        "Update restart pending. Queued for after restart."
                    )
                else:
                    message = "Update check in progress. Queued briefly."
                await ctx.reply_text(message)
            return

        lock = self._ensure_task_lock(uid)

        if lock.locked():
            q = self._ensure_message_queue(uid)
            if q.full():
                await ctx.reply_text("Queue full. Wait or /stop first.")
            else:
                was_awaiting = uid in self._awaiting_answer
                entry_id = await self._persist_enqueue(
                    uid, text, chat_id,
                )
                await q.put((text, chat_id, entry_id, False))
                if was_awaiting:
                    await self._resume_awaiting_answer(
                        uid, q, entry_id, reason="turn was finishing",
                    )
                await ctx.reply_text(
                    f"Queued ({q.qsize()}/{self.max_queue_size})."
                )
                # Race guard: the previous turn could have finished and
                # its drain could have exited (on an empty queue) while
                # we were awaiting _persist_enqueue above. Kick a fresh
                # drain task so the entry we just put isn't orphaned.
                # If a drain is already active it will no-op on the
                # locked() check at the top of the loop.
                self._start_queue_drain(uid)
            return

        # Direct path: persist BEFORE acquiring the lock so a crash
        # between persist and processing still leaves the entry on
        # disk to be resumed by restore_queues() after restart.
        entry_id = await self._persist_enqueue(uid, text, chat_id)
        await lock.acquire()
        # The user is sending a message — if the previous turn paused
        # for an answer ([[await]]), this message is the answer. Clear
        # the flag *after* lock acquisition: any concurrent drain task
        # racing during the persist_enqueue await will then see the
        # flag still set and break, instead of grabbing the lock and
        # processing a queued (non-answer) message ahead of us.
        self._awaiting_answer.discard(uid)
        # Register the task as soon as the lock is held, so /stop can
        # find it even if the turn yields on its initial Telegram/Slack
        # API calls (send "Thinking..." etc.) before it fully starts.
        self._running_tasks[uid] = asyncio.current_task()
        try:
            await self._run_turn(uid, chat_id, text)
        except asyncio.CancelledError:
            # /stop path already cleared the persistent queue.
            # Shutdown path deliberately leaves the entry so restart
            # can resume it — do NOT call _persist_complete here.
            # Wrap the reply: during shutdown the platform may be
            # tearing down and send_text can raise; that shouldn't
            # mask the fact that we handled the cancel cleanly.
            if uid not in self._cancel_acknowledged:
                await self._send_text_best_effort(chat_id, "Cancelled.")
            return
        except Exception as e:
            # Error is user-facing; consume the entry so it doesn't
            # re-run with the same failure after restart. Complete
            # BEFORE replying so a failing reply (e.g., platform
            # torn down) doesn't skip the completion and leave a
            # stale entry for restart to replay.
            logger.exception("AI turn failed")
            await self._persist_complete(uid, entry_id)
            await self._send_text_best_effort(chat_id, f"Error: {e}")
        else:
            await self._persist_complete(uid, entry_id)
        finally:
            self._cleanup_turn(uid, lock)

        await self._drain_message_queue(uid)

    @staticmethod
    def _compose_thinking_display(
        status_lines: list[str], latest_text: str,
    ) -> str:
        """Build the live 'Thinking...' preview.

        Recent tool/file activity plus the latest streamed answer text
        (tail-truncated so the newest content is kept). Rendered into the
        editable status message during a turn; the final reply is sent
        separately after the message is deleted.
        """
        parts = ["Thinking..."]
        if status_lines:
            parts.append("\n".join(status_lines[-5:]))
        text = latest_text.strip()
        if text:
            if len(text) > 600:
                text = "…" + text[-600:]
            parts.append(text)
        return "\n\n".join(parts)

    async def _run_turn(
        self, uid: str, chat_id: str, text: str,
        *, session_id: str | None = None,
    ) -> None:
        """Send a "Thinking..." status, run the agent, then post the reply.

        ``session_id`` pins the run to a specific session — used by
        ``_run_ephemeral_turn`` to route a scheduled command into a
        throwaway session. When None, ``agent.run`` routes the prompt
        to the best-matching session via ``router.select_or_create_session``.

        ``_running_tasks`` is already populated by the caller
        (``_dispatch_ai`` or ``_drain_message_queue``) right after
        lock.acquire(), so /stop can cancel this turn during any of
        its await points.
        """
        ws = await self._current_workspace_for_turn(uid, chat_id)
        if ws is None:
            return
        backend_name, model, summary_model, perm, summary_backend = (
            workspace.get_run_config(ws)
        )

        inject_q: asyncio.Queue[str] = asyncio.Queue(
            maxsize=self.max_queue_size,
        )
        self._inject_queues[uid] = inject_q

        thinking_handle = await self.send_text(
            chat_id, "Thinking...",
        )

        status_lines: list[str] = []
        latest_text = ""
        last_edit = 0.0

        async def on_event(ev: agent.ChatEvent) -> None:
            nonlocal last_edit, latest_text
            if ev.kind == "text":
                if ev.content.strip():
                    latest_text = ev.content
            elif ev.kind == "tool":
                status_lines.append(f"» {ev.content.split(chr(10))[0][:80]}")
            elif ev.kind == "file":
                status_lines.append(f"» {ev.content[:80]}")
            else:
                return
            if thinking_handle is None:
                # Platform doesn't support editable status messages (e.g.
                # CLI mode). Emit tool/file progress as fresh messages; the
                # answer text arrives whole in the final reply, so skip it
                # here to avoid printing it twice.
                if ev.kind != "text":
                    with contextlib.suppress(Exception):
                        await self.send_status(chat_id, status_lines[-1])
                return
            now = asyncio.get_running_loop().time()
            if now - last_edit < 1.5:
                return
            last_edit = now
            with contextlib.suppress(Exception):
                await self.edit_text(
                    thinking_handle,
                    self._compose_thinking_display(status_lines, latest_text),
                )

        try:
            result = await agent.run(
                text, ws, user_id=uid,
                model=model, summary_model=summary_model,
                approval=perm,
                on_event=on_event, inject_queue=inject_q,
                backend_name=backend_name,
                summary_backend_name=summary_backend,
                session_id=session_id,
            )
        finally:
            if thinking_handle is not None:
                with contextlib.suppress(Exception):
                    await asyncio.shield(
                        self.delete_message(thinking_handle)
                    )

        # Only opt into the [[await]] pause for interactive turns —
        # ephemeral schedule turns (session_id is set) get their session
        # deleted right after, so there's nothing to resume into.
        await self._send_result(
            chat_id, ws, result,
            uid=uid if session_id is None else None,
        )

    async def _run_ephemeral_turn(
        self, uid: str, chat_id: str, text: str,
    ) -> None:
        """Run a scheduled command in a fresh, throwaway session.

        The session is created right before the run and deleted in a
        ``finally`` so a crash, /stop, or backend error still cleans up.
        It never becomes the user's current session — interactive
        chat continues against whatever session was current before.
        """
        ws = await self._current_workspace_for_turn(uid, chat_id)
        if ws is None:
            return

        # Distinguishable name — the auto-router keys off session
        # names/topics and the "⏰" prefix together with the non-default
        # name keeps the auto-title task from racing the delete.
        label = text.strip().splitlines()[0] if text.strip() else "scheduled"
        if len(label) > 40:
            label = label[:40] + "…"
        sess_data = session.create_session(ws, name=f"⏰ {label}")
        sid = sess_data["id"]
        try:
            await self._run_turn(uid, chat_id, text, session_id=sid)
        finally:
            # Always tear down the throwaway session — including on
            # /stop (CancelledError propagates after this finally) and
            # on backend errors. The non-default session name keeps
            # the auto-title task from racing against the delete.
            try:
                session.delete_session(ws, sid)
            except Exception:
                logger.warning(
                    "Failed to delete ephemeral session %s", sid,
                    exc_info=True,
                )

    async def _current_workspace_for_turn(
        self, uid: str, chat_id: str,
    ) -> str | None:
        ws = workspace.get_current(uid, self.platform_id)
        if not ws or not os.path.isdir(ws):
            await self.send_text(
                chat_id,
                "Workspace not available (deleted?). Use /new or /open.",
            )
            return None
        return ws

    async def _send_result(
        self, chat_id: str, ws: str, result: agent.AgentResult,
        *, uid: str | None = None,
    ) -> None:
        """Send the agent's reply (text + attachments) and honor [[await]].

        ``uid``, when provided, opts the user's queue into the await
        pause: if any text event contains ``[[await]]``, the marker is
        stripped and ``self._awaiting_answer`` is set so the next
        drain pass will block until the user's next message clears it.
        Ephemeral schedule turns omit ``uid`` because their session is
        deleted right after, so there's nothing to resume into.
        """
        awaiting = False
        sent_sources: set[str] = set()

        async def send_attachment(path: str) -> None:
            source_path = agent.attachment_source_path(path, ws)
            if source_path is None:
                logger.warning(
                    "Refusing to send missing or unsafe attachment: %s", path,
                )
                return
            if source_path in sent_sources:
                return
            abs_path = agent.prepare_attachment_path(path, ws)
            if abs_path is None:
                logger.warning(
                    "Failed to prepare attachment for sending: %s", path,
                )
                return
            sent_sources.add(source_path)
            try:
                await self.send_file(chat_id, abs_path)
            except Exception as e:
                logger.warning(
                    "Failed to send attachment %s: %s", abs_path, e,
                )
                await self._send_text_best_effort(
                    chat_id,
                    f"Failed to attach {os.path.basename(abs_path)}: {e}",
                )

        for ev in result.events:
            if ev.kind == "attachment":
                await send_attachment(ev.content)
                continue
            if ev.kind != "text":
                continue
            text, attach_paths = agent.extract_attachment_sources(
                ev.content, ws,
            )
            text, ev_awaiting = agent.extract_await(text)
            if ev_awaiting:
                awaiting = True
            if text:
                await self.send_text(chat_id, text, rich=True)
            for path in attach_paths:
                await send_attachment(path)

        # Compact per-turn token/cost footer, when the backend reported
        # usage and the operator hasn't disabled it.
        if result.usage and config.get_show_usage():
            footer = agent.format_usage(result.usage)
            if footer:
                await self._send_text_best_effort(
                    chat_id, footer, rich=True,
                )

        if awaiting and uid is not None:
            self._awaiting_answer.add(uid)
            logger.info(
                "User %s awaiting answer; queue paused until next message",
                uid,
            )

    async def _drain_message_queue(self, uid: str) -> None:
        q = self._message_queues.get(uid)
        if not q:
            return
        # Defensive: if a queue exists, a lock must too. Create on demand
        # so a missing lock doesn't crash the drain task.
        lock = self._ensure_task_lock(uid)

        while not q.empty():
            if self._update_check_pending or self._update_restart_pending:
                break
            if lock.locked():
                break
            await lock.acquire()
            try:
                entry = self._pop_next_queue_entry(
                    q,
                    ephemeral_only=uid in self._awaiting_answer,
                )
            except asyncio.QueueEmpty:
                lock.release()
                break
            if entry is None:
                # Agent ended its last interactive reply with [[await]].
                # Leave normal chat entries paused until the user sends
                # an answer, but scheduled ephemeral entries above can
                # still drain while the queue is in that state.
                lock.release()
                break
            text, msg_chat_id, entry_id, ephemeral = entry

            # Entries survive across restarts, which means a user may
            # have been de-authorized between persist and drain. Drop
            # unauthorized entries without running them. Persist
            # completion is best-effort - if the on-disk queue write
            # fails (disk full, etc.) we still release the lock so
            # the next drain pass can run, instead of leaking it and
            # wedging the user forever.
            if not self.authorized(uid, msg_chat_id):
                logger.warning(
                    "Dropping queued entry for unauthorized user=%s chat=%s",
                    uid, msg_chat_id,
                )
                try:
                    await self._persist_complete(uid, entry_id)
                except Exception:
                    logger.exception(
                        "Failed to persist completion of unauthorized entry",
                    )
                finally:
                    lock.release()
                continue

            # Register the task before run_ai_turn yields on any API call,
            # so /stop can cancel it even during the initial "Thinking..."
            # send. Pops in _cleanup_turn via finally.
            self._running_tasks[uid] = asyncio.current_task()
            try:
                if ephemeral:
                    await self._run_ephemeral_turn(uid, msg_chat_id, text)
                else:
                    await self._run_turn(uid, msg_chat_id, text)
            except asyncio.CancelledError:
                # Leave the entry on disk so restart resumes it
                # (or, if /stop caused the cancel, cmd_stop already
                # cleared the persistent queue).
                if uid not in self._cancel_acknowledged:
                    await self._send_text_best_effort(
                        msg_chat_id, "Cancelled.",
                    )
                break
            except Exception as e:
                logger.exception("Queued AI chat failed")
                await self._send_text_best_effort(
                    msg_chat_id, f"Error: {e}",
                )
                await self._persist_complete(uid, entry_id)
            else:
                await self._persist_complete(uid, entry_id)
            finally:
                self._cleanup_turn(uid, lock)

    # ----- helpers --------------------------------------------------------

    @staticmethod
    def _pick_option(text: str, options: list[str]) -> str | None:
        text = text.strip()
        if text.isdecimal():
            idx = int(text) - 1
            if 0 <= idx < len(options):
                return options[idx]
            return None
        if text in options:
            return text
        return None

    # Command registry --------------------------------------------------
    # Populated at class body end to avoid forward-reference issues.
    _COMMANDS: ClassVar[dict[str, Handler]] = {}


def _flexible_tier_commands() -> dict[str, Handler]:
    """One /agent_flexible_<tier> and /model_flexible_<tier> per tier.

    Generated rather than written out so a new difficulty tier is a
    one-line change in :mod:`Cozter.flexible` instead of six here.
    """
    commands: dict[str, Handler] = {}
    for tier in workspace.FLEXIBLE_TIERS:
        commands[f"agent_flexible_{tier}"] = _flexible_command(
            BotPlatform._show_flexible_agent, tier,
        )
        commands[f"model_flexible_{tier}"] = _flexible_command(
            BotPlatform._show_flexible_model, tier,
        )
    return commands


def _flexible_command(show: Callable, tier: str) -> Handler:
    async def command(self: BotPlatform, ctx: BotContext) -> None:
        await show(self, ctx, tier)
    return command


def _build_command_registry() -> dict[str, Handler]:
    return {
        "start":        BotPlatform.cmd_start,
        "version":      BotPlatform.cmd_version,
        "doctor":       BotPlatform.cmd_doctor,
        "cancel":       BotPlatform.cmd_cancel,
        "new":          BotPlatform.cmd_new,
        "open":         BotPlatform.cmd_open,
        "model":        BotPlatform.cmd_model,
        "summarymodel": BotPlatform.cmd_summarymodel,
        "agent":        BotPlatform.cmd_agent,
        "summaryagent": BotPlatform.cmd_summaryagent,
        **_flexible_tier_commands(),
        "permission":   BotPlatform.cmd_permission,
        "style":        BotPlatform.cmd_style,
        "effort":       BotPlatform.cmd_effort,
        "refresh":      BotPlatform.cmd_refresh,
        "compact":      BotPlatform.cmd_compact,
        "context":      BotPlatform.cmd_context,
        "newsession":   BotPlatform.cmd_newsession,
        "sessions":     BotPlatform.cmd_sessions,
        "colony":       BotPlatform.cmd_colony,
        "stop":         BotPlatform.cmd_stop,
        "inject":       BotPlatform.cmd_inject,
        "reserve":      BotPlatform.cmd_reserve,
        "schedules":    BotPlatform.cmd_schedules,
    }


BotPlatform._COMMANDS = _build_command_registry()

COMMAND_NAMES: tuple[str, ...] = tuple(BotPlatform._COMMANDS.keys())
