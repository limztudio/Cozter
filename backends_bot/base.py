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
import logging
import os
import re
import shutil
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from .. import agent, session, updater, workspace
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
        # Scheduler state.
        self._scheduler_task: asyncio.Task | None = None
        # Keyed by schedule_id — tracks the "HH:MM" minute when we last
        # fired each schedule, so a 30s poll loop can't double-fire.
        self._scheduler_fired: dict[str, str] = {}

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

    async def cmd_clear(self, ctx: BotContext) -> None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
            return
        new_sess = session.create_session(ws)
        session.set_current_session_id(ws, ctx.user_id, new_sess["id"])
        await ctx.reply_text(
            "Conversation cleared. Next message starts a new session."
        )

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

    # ----- /session -------------------------------------------------------

    async def cmd_session(self, ctx: BotContext) -> None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
            return
        current_sid = session.get_current_session_id(ws, ctx.user_id)
        sessions = session.list_sessions(ws)

        lines = []
        if current_sid:
            current_meta = next(
                (s for s in sessions if s["id"] == current_sid), None,
            )
            if current_meta:
                lines.append(
                    f"Current session: {current_meta['name']}"
                    f" ({current_meta['message_count']} msgs)"
                )
            else:
                lines.append("Current session: (invalid)")
        else:
            lines.append("Current session: (none)")

        if sessions:
            lines.append("\nSessions:")
            for i, s in enumerate(sessions, 1):
                marker = " <-" if s["id"] == current_sid else ""
                lines.append(
                    f"  {i}. {s['name']}"
                    f" ({s['message_count']} msgs){marker}"
                )
        else:
            lines.append("\nNo sessions yet.")

        lines.append(
            "\nEnter a number to switch, or 'new' to create (or /cancel):"
        )
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_session)

    async def _receive_session(self, ctx: BotContext) -> None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws:
            await ctx.reply_text(_NO_WS_MSG)
            return
        text = ctx.text.strip()
        if text.lower() == "new":
            new_sess = session.create_session(ws)
            session.set_current_session_id(ws, ctx.user_id, new_sess["id"])
            await ctx.reply_text(
                f"New session created: {new_sess['name']}"
            )
            return
        if text.isdigit():
            sessions = session.list_sessions(ws)
            idx = int(text) - 1
            if 0 <= idx < len(sessions):
                chosen = sessions[idx]
                session.set_current_session_id(ws, ctx.user_id, chosen["id"])
                await ctx.reply_text(f"Switched to: {chosen['name']}")
                return
        await ctx.reply_text(
            "Invalid input. Enter a number, 'new', or /cancel:"
        )
        self._expect_input(ctx.user_id, self._receive_session)

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

        if first == "now":
            sid = session.ensure_session(ws, ctx.user_id)
            await ctx.reply_text("Compacting session...")
            summary_model = workspace.get_summary_model(ws)
            backend_name = workspace.get_backend_name(ws)
            new_summary, new_long_term = await agent.compact_session(
                ws, sid, summary_model, backend_name=backend_name,
            )
            if new_summary:
                async with agent.get_workspace_lock(ws):
                    session.set_summary(
                        ws, sid, new_summary,
                        keep_recent=agent.KEEP_RECENT_AFTER_COMPACT,
                        long_term_rewrite=new_long_term,
                    )
                lt_count = (
                    len(new_long_term)
                    if new_long_term is not None else "?"
                )
                await ctx.reply_text(
                    f"Session compacted ({lt_count} long-term items)."
                )
            else:
                await ctx.reply_text(
                    "Compaction produced no output - session unchanged."
                )
            return

        if first.isdigit():
            sid = session.ensure_session(ws, ctx.user_id)
            interval = int(first)
            session.set_compact_interval(ws, sid, interval)
            await ctx.reply_text(
                f"Compact interval set to {interval} messages."
            )
            return

        sid = session.get_current_session_id(ws, ctx.user_id)
        if not sid:
            await ctx.reply_text(
                "No session yet. Send a message to start one.\n\n"
                "Usage:\n"
                "  /compact <number> - set interval\n"
                "  /compact now - compact immediately"
            )
            return
        sess_data = session.load_session(ws, sid) or {}
        current = sess_data.get(
            "compact_interval", session.DEFAULT_COMPACT_INTERVAL,
        )
        total = session.total_message_count(sess_data)
        summary = sess_data.get("summary")
        long_term = sess_data.get("long_term") or []
        lines = [
            f"Compact interval: {current} messages",
            f"Total messages: {total}",
            f"Has summary: {'yes' if summary else 'no'}",
            f"Long-term memory items: {len(long_term)}",
            "",
            "Usage:",
            "  /compact <number> - set interval",
            "  /compact now - compact immediately",
        ]
        await ctx.reply_text("\n".join(lines))

    # ----- /stop ----------------------------------------------------------

    async def cmd_stop(self, ctx: BotContext) -> None:
        task = self._running_tasks.get(ctx.user_id)
        if task and not task.done():
            task.cancel()
            _drain_queue(self._message_queues.get(ctx.user_id))
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

        sid = session.ensure_session(ws, ctx.user_id)
        schedule = {
            "id": uuid.uuid4().hex[:12],
            "days": days,
            "time": time_str,
            "command": command,
            "created": datetime.now().isoformat(),
            "chat_id": ctx.chat_id,
            "user_id": ctx.user_id,
        }
        session.add_schedule(ws, sid, schedule)
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
        sid = session.get_current_session_id(ws, ctx.user_id)
        if not sid:
            await ctx.reply_text(
                "No session yet. Send a message to start one."
            )
            return
        schedules = session.list_schedules(ws, sid)
        if not schedules:
            await ctx.reply_text("No schedules in this session.")
            return
        lines = ["Schedules in this session:"]
        for i, s in enumerate(schedules, 1):
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
        sid = session.get_current_session_id(ws, ctx.user_id)
        if not sid:
            await ctx.reply_text("No session.")
            return
        schedules = session.list_schedules(ws, sid)
        text = ctx.text.strip()
        if not text.isdigit():
            await ctx.reply_text(
                "Invalid input. Enter a number or /cancel:"
            )
            self._expect_input(ctx.user_id, self._receive_schedules)
            return
        idx = int(text) - 1
        if not (0 <= idx < len(schedules)):
            await ctx.reply_text(
                "Invalid number. Try again (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_schedules)
            return
        removed = schedules[idx]
        session.remove_schedule(ws, sid, removed["id"])
        # Forget the fired-this-minute marker so a recreated schedule
        # with the same id (unlikely but possible) isn't suppressed.
        self._scheduler_fired.pop(removed.get("id", ""), None)
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
        now = datetime.now()
        minute_str = now.strftime("%H:%M")
        day = now.strftime("%a").lower()

        # Collect every matching fire across all authorized users before
        # pushing anything — lets us sort by creation time so that two
        # schedules at the same minute fire in the order they were made.
        #
        # We iterate workspace state (not notify_targets) because on
        # Slack notify_targets are channel ids, not user ids. workspace
        # state maps real users to their current workspaces regardless
        # of how the platform identifies authorization targets.
        to_fire: list[tuple[str, dict]] = []
        for uid, ws in workspace.iter_current_workspaces(self.platform_id):
            if not os.path.isdir(ws):
                continue
            sid = session.get_current_session_id(ws, uid)
            if not sid:
                continue
            for sched in session.list_schedules(ws, sid):
                sid_key = sched.get("id", "")
                if not sid_key:
                    continue
                if self._scheduler_fired.get(sid_key) == minute_str:
                    continue
                if (
                    day in sched.get("days", [])
                    and sched.get("time") == minute_str
                ):
                    to_fire.append((uid, sched))
                    self._scheduler_fired[sid_key] = minute_str

        to_fire.sort(key=lambda item: item[1].get("created", ""))

        # Keep the fired-markers dict from growing unbounded: once we've
        # crossed to a new minute, drop everything older.
        if len(self._scheduler_fired) > 500:
            self._scheduler_fired = {
                k: v for k, v in self._scheduler_fired.items()
                if v == minute_str
            }

        for uid, sched in to_fire:
            await self._fire_schedule(uid, sched)

    async def _fire_schedule(self, uid: str, sched: dict) -> None:
        """Push a scheduled command onto the user's message queue."""
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

        try:
            await self.send_text(
                chat_id, f"⏰ Scheduled: {command}",
            )
        except Exception:
            logger.warning("Failed to announce scheduled command")

        if q.full():
            try:
                await self.send_text(
                    chat_id,
                    f"Queue full — dropped scheduled command: {command}",
                )
            except Exception:
                pass
            return

        await q.put((command, chat_id))

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
                await q.put((text, chat_id))
                await ctx.reply_text(
                    f"Queued ({q.qsize()}/{self.max_queue_size})."
                )
            return

        await lock.acquire()
        try:
            await self._run_ai_turn(uid, chat_id, text)
        except asyncio.CancelledError:
            # /stop path: cmd_stop already drained the queue, so don't
            # re-process anything here.
            await ctx.reply_text("Cancelled.")
            return
        except Exception as e:
            # Turn failed but messages queued while it ran are still
            # valid user input; fall through to drain them.
            logger.exception("AI turn failed")
            await ctx.reply_text(f"Error: {e}")
        finally:
            self._cleanup_turn(uid, lock)

        await self._drain_message_queue(uid)

    async def _run_ai_turn(
        self, uid: str, chat_id: str, text: str,
    ) -> None:
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
        self._running_tasks[uid] = asyncio.current_task()

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
                text, msg_chat_id = q.get_nowait()
            except asyncio.QueueEmpty:
                lock.release()
                break
            try:
                await self._run_ai_turn(uid, msg_chat_id, text)
            except asyncio.CancelledError:
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
        "clear":        BotPlatform.cmd_clear,
        "new":          BotPlatform.cmd_new,
        "open":         BotPlatform.cmd_open,
        "model":        BotPlatform.cmd_model,
        "summarymodel": BotPlatform.cmd_summarymodel,
        "agent":        BotPlatform.cmd_agent,
        "permission":   BotPlatform.cmd_permission,
        "session":      BotPlatform.cmd_session,
        "refresh":      BotPlatform.cmd_refresh,
        "compact":      BotPlatform.cmd_compact,
        "stop":         BotPlatform.cmd_stop,
        "inject":       BotPlatform.cmd_inject,
        "reserve":      BotPlatform.cmd_reserve,
        "schedules":    BotPlatform.cmd_schedules,
    }


BotPlatform._COMMANDS = _build_command_registry()

COMMAND_NAMES: tuple[str, ...] = tuple(BotPlatform._COMMANDS.keys())
