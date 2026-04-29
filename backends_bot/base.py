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
import json
import logging
import os
import re
import shutil
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta

from .. import agent, colony, schedules, session, updater, workspace
from ..utils import atomic_write as _atomic_write
from ..utils import drain_queue as _drain_queue

logger = logging.getLogger(__name__)

_NO_WS_MSG = "No workspace selected. Use /new or /open first."

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

_ATTACH_RE = re.compile(
    r"\[\[attach:\s*([^\]\n]+?)\s*\]\]", re.IGNORECASE,
)


def extract_attachments(text: str, ws: str) -> tuple[str, list[str]]:
    """Parse [[attach: PATH]] markers. See bot docstring for details."""
    ws_real = os.path.realpath(ws)
    paths: list[str] = []

    def _sub(m: re.Match) -> str:
        rel = m.group(1).strip()
        if not rel:
            return ""
        try:
            abs_path = rel if os.path.isabs(rel) else os.path.join(ws, rel)
            abs_path = os.path.realpath(abs_path)
            inside = (
                abs_path == ws_real
                or abs_path.startswith(ws_real + os.sep)
            )
            if inside and os.path.isfile(abs_path):
                paths.append(abs_path)
        except (ValueError, OSError):
            pass
        return ""

    cleaned = _ATTACH_RE.sub(_sub, text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, paths


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
    # was cleared by this command arriving. /cancel reads this to know
    # whether to say "Cancelled." or "Nothing to cancel."
    had_pending: bool = False

    async def reply_text(
        self, text: str, *, rich: bool = False,
    ) -> MessageHandle | None:
        return await self.platform.send_text(
            self.chat_id, text, rich=rich,
        )

    async def edit_text(self, handle: MessageHandle, text: str) -> None:
        await self.platform.edit_text(handle, text)

    async def delete_message(self, handle: MessageHandle) -> None:
        await self.platform.delete_message(handle)

    async def send_file(self, path: str) -> None:
        await self.platform.send_file(self.chat_id, path)


# Handler signature: callback taking a BotContext.
Handler = Callable[[BotContext], Awaitable[None]]


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
        # Multi-step flow state: maps user_id -> next text-input callback.
        self._pending_input: dict[str, Handler] = {}
        # Scheduler state. Double-fire within one tick cycle is prevented
        # by the persisted ``last_fired`` timestamp on each schedule,
        # which also survives bot restarts to enable catch-up firing.
        self._scheduler_task: asyncio.Task | None = None
        # Serializes read-modify-write on the persistent-queue file so
        # concurrent enqueue/complete calls don't clobber each other.
        self._queue_file_lock: asyncio.Lock = asyncio.Lock()

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
        self, handle: MessageHandle, text: str,
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
        handler = self._COMMANDS.get(ctx.command or "")
        if handler is None:
            await ctx.reply_text(f"Unknown command: /{ctx.command}")
            return
        # Any new command cancels a pending text-input flow. /cancel
        # uses ctx.had_pending to decide its reply.
        ctx.had_pending = (
            self._pending_input.pop(ctx.user_id, None) is not None
        )
        await handler(self, ctx)

    async def dispatch_text(self, ctx: BotContext) -> None:
        """Entry point for plain text messages (not slash commands)."""
        if not self.authorized(ctx.user_id, ctx.chat_id):
            return
        if not ctx.text or not ctx.text.strip():
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

    # ----- simple commands ------------------------------------------------

    async def cmd_start(self, ctx: BotContext) -> None:
        await ctx.reply_text("Cozter bot is running.")

    async def cmd_version(self, ctx: BotContext) -> None:
        ver, date = await asyncio.gather(
            asyncio.to_thread(updater.get_current_version),
            asyncio.to_thread(updater.get_last_commit_date),
        )
        await ctx.reply_text(f"Version: {ver}\nUpdated: {date}")

    async def cmd_cancel(self, ctx: BotContext) -> None:
        if ctx.had_pending:
            await ctx.reply_text("Cancelled.")
        else:
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
        text = ctx.text.strip()
        recent = workspace.get_recent(ctx.user_id, self.recent_limit)
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < len(recent):
                path = recent[idx]
            else:
                await ctx.reply_text(
                    "Invalid number. Please try again (or /cancel):"
                )
                self._expect_input(ctx.user_id, self._receive_open_dir)
                return
        else:
            path = text
        if not os.path.isdir(path):
            await ctx.reply_text(
                f"Directory does not exist:\n{path}\n\n"
                "Please enter a valid directory (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_open_dir)
            return
        workspace.ensure_cozter_dir(path)
        workspace.select_workspace(ctx.user_id, path, self.platform_id)
        await ctx.reply_text(f"Workspace selected:\n{path}")

    # ----- /model ---------------------------------------------------------

    async def cmd_model(self, ctx: BotContext) -> None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
            return
        current = workspace.get_model(ws)
        backend_name = workspace.get_backend_name(ws)
        options = workspace.get_available_models(ws)
        lines = [
            f"Current model: {current} (backend: {backend_name})\n",
            "Available models:",
        ]
        for i, m in enumerate(options, 1):
            marker = " <-" if m == current else ""
            lines.append(f"  {i}. {m}{marker}")
        lines.append("\nEnter a number or model name (or /cancel):")
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_model)

    async def _receive_model(self, ctx: BotContext) -> None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
            return
        text = ctx.text.strip()
        options = workspace.get_available_models(ws)
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
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
            return
        current = workspace.get_summary_model(ws)
        backend_name = workspace.get_backend_name(ws)
        options = workspace.get_available_models(ws)
        lines = [
            f"Current summary model: {current} (backend: {backend_name})\n",
            "Available models:",
        ]
        for i, m in enumerate(options, 1):
            marker = " <-" if m == current else ""
            lines.append(f"  {i}. {m}{marker}")
        lines.append("\nEnter a number or model name (or /cancel):")
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_summarymodel)

    async def _receive_summarymodel(self, ctx: BotContext) -> None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
            return
        text = ctx.text.strip()
        options = workspace.get_available_models(ws)
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
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
            return
        current = workspace.get_backend_name(ws)
        options = workspace.AVAILABLE_BACKENDS
        lines = [f"Current agent: {current}\n", "Available agents:"]
        for i, name in enumerate(options, 1):
            marker = " <-" if name == current else ""
            lines.append(f"  {i}. {name}{marker}")
        lines.append("\nEnter a number or agent name (or /cancel):")
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_agent)

    async def _receive_agent(self, ctx: BotContext) -> None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
            return
        text = ctx.text.strip()
        options = workspace.AVAILABLE_BACKENDS
        name = self._pick_option(text, options)
        if name is None:
            await ctx.reply_text(
                f"Unknown agent: {text}\nTry again (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_agent)
            return
        workspace.set_backend_name(ws, name)
        _, model, summary_model, _ = workspace.get_run_config(ws)
        await ctx.reply_text(
            f"Agent set to: {name}\n"
            f"Model: {model}\nSummary model: {summary_model}"
        )

    # ----- /permission ----------------------------------------------------

    async def cmd_permission(self, ctx: BotContext) -> None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
            return
        current = workspace.get_permission(ws)
        options = workspace.AVAILABLE_PERMISSIONS
        lines = [f"Current permission: {current}\n", "Available modes:"]
        for i, p in enumerate(options, 1):
            marker = " <-" if p == current else ""
            desc = workspace.PERMISSION_DESCRIPTIONS[p]
            lines.append(f"  {i}. {p} - {desc}{marker}")
        lines.append("\nEnter a number or mode name (or /cancel):")
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_permission)

    async def _receive_permission(self, ctx: BotContext) -> None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
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
        workspace.set_permission(ws, perm)
        desc = workspace.PERMISSION_DESCRIPTIONS[perm]
        await ctx.reply_text(f"Permission set to: {perm}\n{desc}")

    # ----- /refresh -------------------------------------------------------

    async def cmd_refresh(self, ctx: BotContext) -> None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
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
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
            return
        arg = ctx.args.strip().lower().split()
        first = arg[0] if arg else ""

        if first.isdigit():
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

    # ----- /colony --------------------------------------------------------

    async def cmd_colony(self, ctx: BotContext) -> None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
            return
        arg = ctx.args.strip().lower().split()
        first = arg[0] if arg else ""

        if first == "now":
            await ctx.reply_text("Consolidating colony...")
            summary_model = workspace.get_summary_model(ws)
            backend_name = workspace.get_backend_name(ws)
            ok = await agent.colony_consolidate(
                ws, summary_model, backend_name=backend_name,
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

        if first.isdigit():
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
        task = self._running_tasks.get(ctx.user_id)
        if task and not task.done():
            task.cancel()
            _drain_queue(self._message_queues.get(ctx.user_id))
            # Clear the persistent queue so cancelled work doesn't
            # come back on the next restart.
            await self._clear_persistent_queue(ctx.user_id)
            await ctx.reply_text("Cancelling...")
        else:
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

    _DAY_ABBREV: tuple[str, ...] = (
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    )

    async def cmd_reserve(self, ctx: BotContext) -> None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
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
        days = self._parse_days(text)
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
        time_str = self._parse_time(ctx.text.strip())
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

        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
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
        async with agent.get_workspace_lock(ws):
            schedules.add_schedule(ws, ctx.user_id, schedule)
        await ctx.reply_text(
            f"Schedule created:\n"
            f"  Days: {', '.join(days)}\n"
            f"  Time: {time_str}\n"
            f"  Command: {command}"
        )

    # ----- /schedules (list/delete) --------------------------------------

    async def cmd_schedules(self, ctx: BotContext) -> None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
            return
        user_schedules = schedules.list_schedules(ws, ctx.user_id)
        if not user_schedules:
            await ctx.reply_text("No schedules.")
            return
        lines = ["Schedules:"]
        for i, s in enumerate(user_schedules, 1):
            days_str = ",".join(s.get("days", []))
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
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
            return
        user_schedules = schedules.list_schedules(ws, ctx.user_id)
        text = ctx.text.strip()
        if not text.isdigit():
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
        async with agent.get_workspace_lock(ws):
            schedules.remove_schedule(ws, ctx.user_id, removed["id"])
        await ctx.reply_text(
            f"Removed: [{','.join(removed.get('days', []))}]"
            f" {removed.get('time', '?')} — {removed.get('command', '')}"
        )

    @classmethod
    def _parse_days(cls, text: str) -> list[str]:
        """Parse a days spec into ordered, de-duplicated abbreviations."""
        if text == "all":
            return list(cls._DAY_ABBREV)
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            return []
        days: list[str] = []
        for p in parts:
            if p.isdigit():
                n = int(p)
                if not (1 <= n <= 7):
                    return []
                days.append(cls._DAY_ABBREV[n - 1])
            else:
                abbr = p[:3]
                if abbr not in cls._DAY_ABBREV:
                    return []
                days.append(abbr)
        seen: set[str] = set()
        unique: list[str] = []
        for d in days:
            if d not in seen:
                seen.add(d)
                unique.append(d)
        return unique

    @staticmethod
    def _parse_time(text: str) -> str | None:
        parts = text.split(":")
        if len(parts) != 2:
            return None
        try:
            h = int(parts[0])
            m = int(parts[1])
        except ValueError:
            return None
        if not (0 <= h <= 23 and 0 <= m <= 59):
            return None
        return f"{h:02d}:{m:02d}"

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
        return os.path.join(
            workspace.CONFIG_DIR, f"queue_{self.platform_id}.json",
        )

    def _read_queue_file(self) -> dict:
        path = self._queue_file_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Corrupt or unreadable queue file (%s): %s", path, e,
            )
            return {}

    def _write_queue_file(self, data: dict) -> None:
        _atomic_write(
            self._queue_file_path(), data, workspace.CONFIG_DIR,
        )

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
            data.setdefault(uid, []).append(entry)
            self._write_queue_file(data)
        return entry_id

    async def _persist_complete(
        self, uid: str, entry_id: str,
    ) -> None:
        async with self._queue_file_lock:
            data = self._read_queue_file()
            entries = data.get(uid)
            if not entries:
                return
            remaining = [e for e in entries if e.get("id") != entry_id]
            if len(remaining) == len(entries):
                return  # entry already gone
            if remaining:
                data[uid] = remaining
            else:
                data.pop(uid, None)
            self._write_queue_file(data)

    async def _clear_persistent_queue(self, uid: str) -> None:
        async with self._queue_file_lock:
            data = self._read_queue_file()
            if data.pop(uid, None) is not None:
                self._write_queue_file(data)

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
        for uid, entries in data.items():
            if not entries:
                continue
            if uid not in self._task_locks:
                self._task_locks[uid] = asyncio.Lock()
            if uid not in self._message_queues:
                self._message_queues[uid] = asyncio.Queue(
                    maxsize=self.max_queue_size,
                )
            q = self._message_queues[uid]
            for entry in entries:
                # Oldest-first preserves the user's original ordering.
                try:
                    q.put_nowait((
                        entry.get("text", ""),
                        entry.get("chat_id", ""),
                        entry.get("id", ""),
                        bool(entry.get("ephemeral", False)),
                    ))
                except asyncio.QueueFull:
                    logger.warning(
                        "Restore dropped entry for user=%s"
                        " (queue at capacity)", uid,
                    )
                    break
            drained_users.append(uid)

        for uid in drained_users:
            asyncio.create_task(self._drain_message_queue(uid))

    # ----- Scheduler loop ------------------------------------------------

    async def start_scheduler(self) -> None:
        """Kick off the background scheduler task. Idempotent."""
        if self._scheduler_task and not self._scheduler_task.done():
            return
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop_scheduler(self) -> None:
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
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
                slot = self._most_recent_slot(sched, now)
                if slot is None:
                    continue
                last_fired = self._parse_iso(sched.get("last_fired"))
                baseline = last_fired or self._parse_iso(
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
            # Persist last_fired BEFORE queueing the command, so a
            # crash between marking and queueing at worst drops the
            # fire rather than firing it twice.
            async with agent.get_workspace_lock(ws):
                schedules.update_schedule_fired(
                    ws, uid, sched["id"], slot.isoformat(),
                )
            await self._fire_schedule(uid, sched)

    @classmethod
    def _most_recent_slot(
        cls, sched: dict, now: datetime,
    ) -> datetime | None:
        """Return the latest (day, time) match <= now, or None."""
        time_str = sched.get("time", "")
        parsed = cls._parse_time(time_str)
        if parsed is None:
            return None
        h, m = map(int, parsed.split(":"))
        target = dt_time(h, m)
        days = sched.get("days", [])
        if not days:
            return None
        # Walk back up to 7 days; the first day-match whose datetime
        # is <= now is the most recent slot.
        for offset in range(8):
            candidate_date = (now - timedelta(days=offset)).date()
            day_name = cls._DAY_ABBREV[candidate_date.weekday()]
            if day_name not in days:
                continue
            candidate = datetime.combine(candidate_date, target)
            if candidate <= now:
                return candidate
        return None

    @staticmethod
    def _parse_iso(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

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

        if uid not in self._message_queues:
            self._message_queues[uid] = asyncio.Queue(
                maxsize=self.max_queue_size,
            )
        q = self._message_queues[uid]

        # Check capacity BEFORE announcing — otherwise a user with a full
        # queue sees "⏰ Scheduled: X" immediately followed by "Queue full
        # — dropped scheduled command: X", which is confusing.
        if q.full():
            try:
                await self.send_text(
                    chat_id,
                    f"Queue full — dropped scheduled command: {command}",
                )
            except Exception:
                pass
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
        asyncio.create_task(self._drain_message_queue(uid))

    # ----- AI chat + file -------------------------------------------------

    async def _require_ws(self, ctx: BotContext) -> str | None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws or not os.path.isdir(ws):
            await ctx.reply_text(
                "No workspace selected (or it was deleted)."
                " Use /new or /open."
            )
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
        lock.release()

    async def _dispatch_ai(self, ctx: BotContext, text: str) -> None:
        uid = ctx.user_id
        chat_id = ctx.chat_id
        if uid not in self._task_locks:
            self._task_locks[uid] = asyncio.Lock()
        lock = self._task_locks[uid]

        if lock.locked():
            if uid not in self._message_queues:
                self._message_queues[uid] = asyncio.Queue(
                    maxsize=self.max_queue_size,
                )
            q = self._message_queues[uid]
            if q.full():
                await ctx.reply_text("Queue full. Wait or /stop first.")
            else:
                entry_id = await self._persist_enqueue(
                    uid, text, chat_id,
                )
                await q.put((text, chat_id, entry_id, False))
                await ctx.reply_text(
                    f"Queued ({q.qsize()}/{self.max_queue_size})."
                )
                # Race guard: the previous turn could have finished and
                # its drain could have exited (on an empty queue) while
                # we were awaiting _persist_enqueue above. Kick a fresh
                # drain task so the entry we just put isn't orphaned.
                # If a drain is already active it will no-op on the
                # locked() check at the top of the loop.
                asyncio.create_task(self._drain_message_queue(uid))
            return

        # Direct path: persist BEFORE acquiring the lock so a crash
        # between persist and processing still leaves the entry on
        # disk to be resumed by restore_queues() after restart.
        entry_id = await self._persist_enqueue(uid, text, chat_id)
        await lock.acquire()
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
            try:
                await ctx.reply_text("Cancelled.")
            except Exception:
                pass
            return
        except Exception as e:
            # Error is user-facing; consume the entry so it doesn't
            # re-run with the same failure after restart. Complete
            # BEFORE replying so a failing reply (e.g., platform
            # torn down) doesn't skip the completion and leave a
            # stale entry for restart to replay.
            logger.exception("AI turn failed")
            await self._persist_complete(uid, entry_id)
            try:
                await ctx.reply_text(f"Error: {e}")
            except Exception:
                pass
        else:
            await self._persist_complete(uid, entry_id)
        finally:
            self._cleanup_turn(uid, lock)

        await self._drain_message_queue(uid)

    async def _run_turn(
        self, uid: str, chat_id: str, text: str,
        *, session_id: str | None = None,
    ) -> None:
        """Send a "Thinking..." status, run the agent, then post the reply.

        ``session_id`` pins the run to a specific session — used by
        ``_run_ephemeral_turn`` to route a scheduled command into a
        throwaway session. When None, ``agent.run`` resolves the
        user's current session itself.

        ``_running_tasks`` is already populated by the caller
        (``_dispatch_ai`` or ``_drain_message_queue``) right after
        lock.acquire(), so /stop can cancel this turn during any of
        its await points.
        """
        ws = workspace.get_current(uid, self.platform_id)
        if not ws or not os.path.isdir(ws):
            await self.send_text(
                chat_id,
                "Workspace not available (deleted?). Use /new or /open.",
            )
            return
        backend_name, model, summary_model, perm = (
            workspace.get_run_config(ws)
        )

        inject_q: asyncio.Queue[str] = asyncio.Queue(
            maxsize=self.max_queue_size,
        )
        self._inject_queues[uid] = inject_q

        thinking_handle = await self.send_text(chat_id, "Thinking...")

        status_lines: list[str] = []
        last_edit = 0.0

        async def on_event(ev: agent.ChatEvent) -> None:
            nonlocal last_edit
            if ev.kind == "tool":
                status_lines.append(
                    f"» {ev.content.split(chr(10))[0][:80]}"
                )
            elif ev.kind == "file":
                status_lines.append(f"» {ev.content[:80]}")
            else:
                return
            if thinking_handle is None:
                return
            now = asyncio.get_running_loop().time()
            if now - last_edit < 1.5:
                return
            last_edit = now
            display = "Thinking...\n\n" + "\n".join(status_lines[-5:])
            try:
                await self.edit_text(thinking_handle, display)
            except Exception:
                pass

        try:
            result = await agent.run(
                text, ws, user_id=uid,
                model=model, summary_model=summary_model, approval=perm,
                on_event=on_event, inject_queue=inject_q,
                backend_name=backend_name,
                session_id=session_id,
            )
        finally:
            if thinking_handle is not None:
                try:
                    await asyncio.shield(
                        self.delete_message(thinking_handle)
                    )
                except Exception:
                    pass

        await self._send_result(chat_id, ws, result)

    async def _run_ephemeral_turn(
        self, uid: str, chat_id: str, text: str,
    ) -> None:
        """Run a scheduled command in a fresh, throwaway session.

        The session is created right before the run and deleted in a
        ``finally`` so a crash, /stop, or backend error still cleans up.
        It never becomes the user's current session — interactive
        chat continues against whatever session was current before.
        """
        ws = workspace.get_current(uid, self.platform_id)
        if not ws or not os.path.isdir(ws):
            await self.send_text(
                chat_id,
                "Workspace not available (deleted?). Use /new or /open.",
            )
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

    async def _send_result(
        self, chat_id: str, ws: str, result: agent.AgentResult,
    ) -> None:
        for ev in result.events:
            if ev.kind != "text":
                continue
            text, attach_paths = extract_attachments(ev.content, ws)
            if text:
                await self.send_text(chat_id, text, rich=True)
            for path in attach_paths:
                try:
                    await self.send_file(chat_id, path)
                except Exception as e:
                    logger.warning(
                        "Failed to send attachment %s: %s", path, e,
                    )
                    try:
                        await self.send_text(
                            chat_id,
                            f"Failed to attach {os.path.basename(path)}: {e}",
                        )
                    except Exception:
                        pass

    async def _drain_message_queue(self, uid: str) -> None:
        q = self._message_queues.get(uid)
        if not q:
            return
        lock = self._task_locks[uid]

        while not q.empty():
            if lock.locked():
                break
            await lock.acquire()
            try:
                text, msg_chat_id, entry_id, ephemeral = q.get_nowait()
            except asyncio.QueueEmpty:
                lock.release()
                break

            # Entries survive across restarts, which means a user may
            # have been de-authorized between persist and drain. Drop
            # unauthorized entries without running them.
            if not self.authorized(uid, msg_chat_id):
                logger.warning(
                    "Dropping queued entry for unauthorized user=%s chat=%s",
                    uid, msg_chat_id,
                )
                await self._persist_complete(uid, entry_id)
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
                try:
                    await self.send_text(msg_chat_id, "Cancelled.")
                except Exception:
                    pass
                break
            except Exception as e:
                logger.exception("Queued AI chat failed")
                try:
                    await self.send_text(msg_chat_id, f"Error: {e}")
                except Exception:
                    pass
                await self._persist_complete(uid, entry_id)
            else:
                await self._persist_complete(uid, entry_id)
            finally:
                self._cleanup_turn(uid, lock)

    # ----- helpers --------------------------------------------------------

    @staticmethod
    def _pick_option(text: str, options: list[str]) -> str | None:
        text = text.strip()
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < len(options):
                return options[idx]
            return None
        if text in options:
            return text
        return None

    # Command registry --------------------------------------------------
    # Populated at class body end to avoid forward-reference issues.
    _COMMANDS: dict[str, Handler] = {}


def _build_command_registry() -> dict[str, Handler]:
    return {
        "start":        BotPlatform.cmd_start,
        "version":      BotPlatform.cmd_version,
        "cancel":       BotPlatform.cmd_cancel,
        "new":          BotPlatform.cmd_new,
        "open":         BotPlatform.cmd_open,
        "model":        BotPlatform.cmd_model,
        "summarymodel": BotPlatform.cmd_summarymodel,
        "agent":        BotPlatform.cmd_agent,
        "permission":   BotPlatform.cmd_permission,
        "refresh":      BotPlatform.cmd_refresh,
        "compact":      BotPlatform.cmd_compact,
        "colony":       BotPlatform.cmd_colony,
        "stop":         BotPlatform.cmd_stop,
        "inject":       BotPlatform.cmd_inject,
        "reserve":      BotPlatform.cmd_reserve,
        "schedules":    BotPlatform.cmd_schedules,
    }


BotPlatform._COMMANDS = _build_command_registry()

COMMAND_NAMES: tuple[str, ...] = tuple(BotPlatform._COMMANDS.keys())
