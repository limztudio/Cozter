import asyncio
import functools
import logging
import os
import re
import shutil

from telegram import Update
from telegram.error import NetworkError
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from . import codex
from . import session
from . import updater
from . import workspace
from .utils import drain_queue as _drain_queue

logger = logging.getLogger(__name__)


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _split_html(text: str, limit: int = 4096) -> list[str]:
    """
    Split text into chunks no larger than limit, breaking at newlines.

    Splitting at arbitrary byte offsets risks cutting inside an HTML tag,
    which causes Telegram to reject the message. Newline boundaries are safe
    because _md_to_html never produces tags that span multiple lines.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            # No newline - hard split
            chunks.append(text[:limit])
            text = text[limit:]
        else:
            # Split at newline, consume the newline character
            chunks.append(text[:split_at])
            text = text[split_at + 1:]
    return chunks


def _md_to_html(text: str) -> str:
    """Convert common Markdown to Telegram-compatible HTML."""
    lines = text.split("\n")
    result = []
    in_code_block = False
    code_buf: list[str] = []

    for line in lines:
        # Fenced code blocks
        if line.strip().startswith("```"):
            if in_code_block:
                escaped = _escape_html("\n".join(code_buf))
                result.append(f"<pre>{escaped}</pre>")
                code_buf.clear()
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_buf.append(line)
            continue

        line = _escape_html(line)

        # Headers -> bold
        line = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", line)
        # Bold: **text** or __text__
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        line = re.sub(r"__(.+?)__", r"<b>\1</b>", line)
        # Italic: *text* or _text_ (but not inside words with underscores)
        line = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<i>\1</i>", line)
        line = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>", line)
        # Inline code: `text`
        line = re.sub(r"`([^`]+?)`", r"<code>\1</code>", line)
        # Strikethrough: ~~text~~
        line = re.sub(r"~~(.+?)~~", r"<s>\1</s>", line)

        result.append(line)

    # Unclosed code block
    if in_code_block and code_buf:
        escaped = _escape_html("\n".join(code_buf))
        result.append(f"<pre>{escaped}</pre>")

    return "\n".join(result)


# Conversation states
NEW_AWAITING_DIR = 0
OPEN_AWAITING_DIR = 1
MODEL_AWAITING = 2
PERMISSION_AWAITING = 3
SESSION_AWAITING = 4
SUMMARY_MODEL_AWAITING = 5

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
_INLINE_SIZE_LIMIT = 50_000  # characters

_ATTACH_RE = re.compile(
    r"\[\[attach:\s*([^\]\n]+?)\s*\]\]", re.IGNORECASE,
)


def _extract_attachments(text: str, ws: str) -> tuple[str, list[str]]:
    """Parse [[attach: PATH]] markers.

    Returns (text_without_markers, list_of_absolute_file_paths).
    Only files that resolve inside the workspace are accepted, so the AI
    can't attach arbitrary system files.
    """
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
            pass  # malformed path (e.g. null byte) — drop silently
        return ""

    cleaned = _ATTACH_RE.sub(_sub, text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, paths


def _authorized(user_ids: list[int]):
    """Decorator that restricts handler to authorized user_ids."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if update.effective_user.id not in user_ids:
                logger.warning(
                    "Unauthorized access attempt from user %s (%s)",
                    update.effective_user.id,
                    update.effective_user.full_name,
                )
                return
            return await func(update, context)
        return wrapper
    return decorator


class CozterBot:
    def __init__(
        self,
        token: str,
        user_ids: list[int],
        recent_limit: int = 10,
        max_queue_size: int = 50,
    ):
        self.token = token
        self.user_ids = user_ids
        self.recent_limit = recent_limit
        self.max_queue_size = max_queue_size
        self.app: Application | None = None
        self._running_tasks: dict[int, asyncio.Task] = {}
        self._task_locks: dict[int, asyncio.Lock] = {}
        self._message_queues: dict[int, asyncio.Queue] = {}
        self._inject_queues: dict[int, asyncio.Queue] = {}

    @property
    def bot_id(self) -> int:
        return self.app.bot.id

    async def start(self) -> None:
        self.app = (
            Application.builder()
            .token(self.token)
            .concurrent_updates(True)
            .build()
        )

        # --- simple commands ---

        @_authorized(self.user_ids)
        async def cmd_start(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            await update.message.reply_text("Cozter bot is running.")

        @_authorized(self.user_ids)
        async def cmd_version(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            ver, date = await asyncio.gather(
                asyncio.to_thread(updater.get_current_version),
                asyncio.to_thread(updater.get_last_commit_date),
            )
            await update.message.reply_text(
                f"Version: {ver}\nUpdated: {date}"
            )

        @_authorized(self.user_ids)
        async def cmd_clear(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text(_NO_WS_MSG)
                return
            new_sess = session.create_session(ws)
            session.set_current_session_id(ws, uid, new_sess["id"])
            await update.message.reply_text(
                "Conversation cleared. Next message starts a new session."
            )

        # --- /new conversation ---

        @_authorized(self.user_ids)
        async def cmd_new(update: Update, _context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            current = workspace.get_current(uid, self.bot_id)
            msg = (
                f"Current workspace: {current or '(none)'}\n\n"
                "Enter the full path for the new workspace directory"
                " (or /cancel):"
            )
            await update.message.reply_text(msg)
            return NEW_AWAITING_DIR

        @_authorized(self.user_ids)
        async def new_receive_dir(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            path = update.message.text.strip()

            if os.path.exists(path):
                await update.message.reply_text(
                    f"Directory already exists:\n{path}\n\n"
                    "Please choose a different path (or /cancel):"
                )
                return NEW_AWAITING_DIR

            try:
                os.makedirs(path)
            except OSError as e:
                await update.message.reply_text(
                    f"Failed to create directory: {e}\n\n"
                    "Please try again (or /cancel):"
                )
                return NEW_AWAITING_DIR

            workspace.ensure_cozter_dir(path)
            workspace.select_workspace(uid, path, self.bot_id)
            await update.message.reply_text(
                f"Workspace created and selected:\n{path}"
            )
            return ConversationHandler.END

        # --- /open conversation ---

        @_authorized(self.user_ids)
        async def cmd_open(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            current = workspace.get_current(uid, self.bot_id)
            recent = workspace.get_recent(uid, self.recent_limit)

            lines = [f"Current workspace: {current or '(none)'}"]
            if recent:
                lines.append("\nRecent workspaces:")
                for i, r in enumerate(recent, 1):
                    lines.append(f"  {i}. {r}")
            else:
                lines.append("\nNo recent workspaces.")
            lines.append(
                "\nEnter a directory path or number"
                " from the list (or /cancel):"
            )
            await update.message.reply_text("\n".join(lines))
            return OPEN_AWAITING_DIR

        @_authorized(self.user_ids)
        async def open_receive_dir(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            text = update.message.text.strip()
            recent = workspace.get_recent(uid, self.recent_limit)

            if text.isdigit():
                idx = int(text) - 1
                if 0 <= idx < len(recent):
                    path = recent[idx]
                else:
                    await update.message.reply_text(
                        "Invalid number. Please try again (or /cancel):"
                    )
                    return OPEN_AWAITING_DIR
            else:
                path = text

            if not os.path.isdir(path):
                await update.message.reply_text(
                    f"Directory does not exist:\n{path}\n\n"
                    "Please enter a valid directory (or /cancel):"
                )
                return OPEN_AWAITING_DIR

            workspace.ensure_cozter_dir(path)
            workspace.select_workspace(uid, path, self.bot_id)
            await update.message.reply_text(f"Workspace selected:\n{path}")
            return ConversationHandler.END

        # --- /model conversation ---

        @_authorized(self.user_ids)
        async def cmd_model(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text(_NO_WS_MSG)
                return ConversationHandler.END

            current = workspace.get_model(ws)
            options = workspace.AVAILABLE_MODELS
            lines = [f"Current model: {current}\n", "Available models:"]
            for i, m in enumerate(options, 1):
                marker = " <-" if m == current else ""
                lines.append(f"  {i}. {m}{marker}")
            lines.append("\nEnter a number or model name (or /cancel):")
            await update.message.reply_text("\n".join(lines))
            return MODEL_AWAITING

        @_authorized(self.user_ids)
        async def model_receive(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text(_NO_WS_MSG)
                return ConversationHandler.END
            text = update.message.text.strip()
            options = workspace.AVAILABLE_MODELS

            if text.isdigit():
                idx = int(text) - 1
                if 0 <= idx < len(options):
                    model = options[idx]
                else:
                    await update.message.reply_text(
                        "Invalid number. Try again (or /cancel):"
                    )
                    return MODEL_AWAITING
            elif text in options:
                model = text
            else:
                await update.message.reply_text(
                    f"Unknown model: {text}\nTry again (or /cancel):"
                )
                return MODEL_AWAITING

            workspace.set_model(ws, model)
            await update.message.reply_text(f"Model set to: {model}")
            return ConversationHandler.END

        # --- /summarymodel conversation ---

        @_authorized(self.user_ids)
        async def cmd_summarymodel(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text(_NO_WS_MSG)
                return ConversationHandler.END

            current = workspace.get_summary_model(ws)
            options = workspace.AVAILABLE_MODELS
            lines = [
                f"Current summary model: {current}\n", "Available models:",
            ]
            for i, m in enumerate(options, 1):
                marker = " <-" if m == current else ""
                lines.append(f"  {i}. {m}{marker}")
            lines.append("\nEnter a number or model name (or /cancel):")
            await update.message.reply_text("\n".join(lines))
            return SUMMARY_MODEL_AWAITING

        @_authorized(self.user_ids)
        async def summarymodel_receive(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text(_NO_WS_MSG)
                return ConversationHandler.END
            text = update.message.text.strip()
            options = workspace.AVAILABLE_MODELS

            if text.isdigit():
                idx = int(text) - 1
                if 0 <= idx < len(options):
                    model = options[idx]
                else:
                    await update.message.reply_text(
                        "Invalid number. Try again (or /cancel):"
                    )
                    return SUMMARY_MODEL_AWAITING
            elif text in options:
                model = text
            else:
                await update.message.reply_text(
                    f"Unknown model: {text}\nTry again (or /cancel):"
                )
                return SUMMARY_MODEL_AWAITING

            workspace.set_summary_model(ws, model)
            await update.message.reply_text(f"Summary model set to: {model}")
            return ConversationHandler.END

        # --- /permission conversation ---

        @_authorized(self.user_ids)
        async def cmd_permission(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text(_NO_WS_MSG)
                return ConversationHandler.END

            current = workspace.get_permission(ws)
            options = workspace.AVAILABLE_PERMISSIONS
            lines = [f"Current permission: {current}\n", "Available modes:"]
            for i, p in enumerate(options, 1):
                marker = " <-" if p == current else ""
                desc = workspace.PERMISSION_DESCRIPTIONS[p]
                lines.append(f"  {i}. {p} - {desc}{marker}")
            lines.append("\nEnter a number or mode name (or /cancel):")
            await update.message.reply_text("\n".join(lines))
            return PERMISSION_AWAITING

        @_authorized(self.user_ids)
        async def permission_receive(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text(_NO_WS_MSG)
                return ConversationHandler.END
            text = update.message.text.strip().lower()
            options = workspace.AVAILABLE_PERMISSIONS

            perm = None
            if text.isdigit():
                idx = int(text) - 1
                if 0 <= idx < len(options):
                    perm = options[idx]
            elif text in options:
                perm = text

            if perm is None:
                await update.message.reply_text(
                    f"Unknown mode: {text}\nTry again (or /cancel):"
                )
                return PERMISSION_AWAITING

            workspace.set_permission(ws, perm)
            desc = workspace.PERMISSION_DESCRIPTIONS[perm]
            await update.message.reply_text(
                f"Permission set to: {perm}\n{desc}"
            )
            return ConversationHandler.END

        # --- /session conversation ---

        @_authorized(self.user_ids)
        async def cmd_session(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text(_NO_WS_MSG)
                return ConversationHandler.END

            current_sid = session.get_current_session_id(ws, uid)
            sessions = session.list_sessions(ws)

            lines = []
            if current_sid:
                current_data = session.load_session(ws, current_sid)
                if current_data:
                    count = session.total_message_count(current_data)
                    name = current_data.get("name", current_sid[:8])
                    lines.append(f"Current session: {name} ({count} msgs)")
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
            await update.message.reply_text("\n".join(lines))
            return SESSION_AWAITING

        @_authorized(self.user_ids)
        async def session_receive(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text(_NO_WS_MSG)
                return ConversationHandler.END
            text = update.message.text.strip()
            sessions = session.list_sessions(ws)

            if text.lower() == "new":
                new_sess = session.create_session(ws)
                session.set_current_session_id(ws, uid, new_sess["id"])
                await update.message.reply_text(
                    f"New session created: {new_sess['name']}"
                )
                return ConversationHandler.END

            if text.isdigit():
                idx = int(text) - 1
                if 0 <= idx < len(sessions):
                    chosen = sessions[idx]
                    session.set_current_session_id(ws, uid, chosen["id"])
                    await update.message.reply_text(
                        f"Switched to: {chosen['name']}"
                    )
                    return ConversationHandler.END

            await update.message.reply_text(
                "Invalid input. Enter a number, 'new', or /cancel:"
            )
            return SESSION_AWAITING

        # --- /refresh command ---

        @_authorized(self.user_ids)
        async def cmd_refresh(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text(_NO_WS_MSG)
                return

            codex_dir = os.path.join(ws, ".codex")
            if os.path.isdir(codex_dir):
                shutil.rmtree(codex_dir, ignore_errors=True)
                logger.info("Cleared codex session dir: %s", codex_dir)

            await update.message.reply_text(
                "Codex CLI session refreshed."
                " Your conversation history is preserved."
            )

        # --- /compact command ---

        @_authorized(self.user_ids)
        async def cmd_compact(
            update: Update, context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text(_NO_WS_MSG)
                return

            sid = session.ensure_session(ws, uid)
            args = (context.args[0] if context.args else "").strip().lower()

            if args == "now":
                await update.message.reply_text("Compacting session...")
                summary_model = workspace.get_summary_model(ws)
                new_summary, new_long_term = await codex.compact_session(
                    ws, sid, summary_model,
                )
                if new_summary:
                    async with codex.get_workspace_lock(ws):
                        session.set_summary(
                            ws, sid, new_summary,
                            keep_recent=codex.KEEP_RECENT_AFTER_COMPACT,
                            long_term_rewrite=new_long_term,
                        )
                    lt_count = (
                        len(new_long_term)
                        if new_long_term is not None else "?"
                    )
                    await update.message.reply_text(
                        f"Session compacted ({lt_count} long-term items)."
                    )
                else:
                    await update.message.reply_text(
                        "Compaction produced no output - session unchanged."
                    )
                return

            if args.isdigit():
                interval = int(args)
                session.set_compact_interval(ws, sid, interval)
                await update.message.reply_text(
                    f"Compact interval set to {interval} messages."
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
            await update.message.reply_text("\n".join(lines))

        # --- shared cancel ---

        @_authorized(self.user_ids)
        async def cancel(update: Update, _context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text("Cancelled.")
            return ConversationHandler.END

        # --- /stop command ---

        @_authorized(self.user_ids)
        async def cmd_stop(
            update: Update, _context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            task = self._running_tasks.get(uid)
            if task and not task.done():
                task.cancel()
                _drain_queue(self._message_queues.get(uid))
                await update.message.reply_text("Cancelling...")
            else:
                await update.message.reply_text("Nothing is running.")

        # --- /inject command ---

        @_authorized(self.user_ids)
        async def cmd_inject(
            update: Update, context: ContextTypes.DEFAULT_TYPE,
        ):
            uid = update.effective_user.id
            text = " ".join(context.args) if context.args else ""
            if not text:
                await update.message.reply_text("Usage: /inject <message>")
                return

            inject_q = self._inject_queues.get(uid)
            if inject_q is None:
                await update.message.reply_text("No task is running.")
                return

            if inject_q.full():
                await update.message.reply_text("Inject queue full.")
                return
            await inject_q.put(text)
            await update.message.reply_text("Injected.")

        async def _require_ws(update: Update, uid: int) -> str | None:
            ws = workspace.get_current(uid, self.bot_id)
            if not ws or not os.path.isdir(ws):
                await update.message.reply_text(
                    "No workspace selected (or it was deleted). Use /new or /open."
                )
                return None
            return ws

        # --- AI chat (default for non-command messages) ---

        @_authorized(self.user_ids)
        async def ai_chat(update: Update, _context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            text = update.message.text.strip()
            if not text:
                return
            if await _require_ws(update, uid) is None:
                return
            await self._dispatch(uid, update.effective_chat.id, text, update)

        # --- AI file attachment ---

        @_authorized(self.user_ids)
        async def ai_file(update: Update, _context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = await _require_ws(update, uid)
            if ws is None:
                return

            message = update.message
            if message.document:
                tg_file = await message.document.get_file()
                filename = (
                    message.document.file_name
                    or f"file_{message.document.file_id}"
                )
                file_type = "document"
            elif message.photo:
                photo = message.photo[-1]  # largest resolution
                tg_file = await photo.get_file()
                filename = f"photo_{photo.file_id}.jpg"
                file_type = "photo"
            elif message.audio:
                tg_file = await message.audio.get_file()
                filename = (
                    message.audio.file_name
                    or f"audio_{message.audio.file_id}.ogg"
                )
                file_type = "audio"
            elif message.video:
                tg_file = await message.video.get_file()
                filename = (
                    message.video.file_name
                    or f"video_{message.video.file_id}.mp4"
                )
                file_type = "video"
            elif message.voice:
                tg_file = await message.voice.get_file()
                filename = f"voice_{message.voice.file_id}.ogg"
                file_type = "voice"
            elif message.video_note:
                tg_file = await message.video_note.get_file()
                filename = f"videonote_{message.video_note.file_id}.mp4"
                file_type = "video note"
            else:
                return

            caption = (message.caption or "").strip()

            # Strip any path components from user-supplied filenames to prevent
            # path traversal (e.g. file_name="../../etc/passwd").
            filename = os.path.basename(filename) or f"file_{tg_file.file_id}"

            upload_dir = os.path.join(ws, ".cozter", "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            local_path = os.path.join(upload_dir, filename)
            try:
                await tg_file.download_to_drive(local_path)
            except Exception as e:
                await update.message.reply_text(f"Failed to download file: {e}")
                return

            ext = os.path.splitext(filename)[1].lower()
            rel_path = os.path.relpath(local_path, ws)

            parts: list[str] = []
            if caption:
                parts.append(caption)
            parts.append(f"[{file_type.capitalize()} attachment saved to: {rel_path}]")

            if ext in _TEXT_EXTENSIONS:
                try:
                    with open(local_path, encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    if len(content) <= _INLINE_SIZE_LIMIT:
                        parts.append(
                            f"[File contents of {filename}]\n"
                            f"{content}\n"
                            f"[End of file]"
                        )
                    else:
                        parts.append(
                            f"[File too large to inline ({len(content):,} chars);"
                            f" read it from {rel_path}]"
                        )
                except OSError:
                    pass  # binary or unreadable — path reference is enough

            await self._dispatch(
                uid, update.effective_chat.id, "\n".join(parts), update,
            )

        # --- register handlers ---

        cancel_handler = CommandHandler("cancel", cancel)

        new_conv = ConversationHandler(
            entry_points=[CommandHandler("new", cmd_new)],
            states={
                NEW_AWAITING_DIR: [
                    cancel_handler,
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND, new_receive_dir,
                    ),
                ],
            },
            fallbacks=[cancel_handler],
        )

        open_conv = ConversationHandler(
            entry_points=[CommandHandler("open", cmd_open)],
            states={
                OPEN_AWAITING_DIR: [
                    cancel_handler,
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND, open_receive_dir,
                    ),
                ],
            },
            fallbacks=[cancel_handler],
        )

        model_conv = ConversationHandler(
            entry_points=[CommandHandler("model", cmd_model)],
            states={
                MODEL_AWAITING: [
                    cancel_handler,
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND, model_receive,
                    ),
                ],
            },
            fallbacks=[cancel_handler],
        )

        summarymodel_conv = ConversationHandler(
            entry_points=[CommandHandler("summarymodel", cmd_summarymodel)],
            states={
                SUMMARY_MODEL_AWAITING: [
                    cancel_handler,
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND, summarymodel_receive,
                    ),
                ],
            },
            fallbacks=[cancel_handler],
        )

        permission_conv = ConversationHandler(
            entry_points=[CommandHandler("permission", cmd_permission)],
            states={
                PERMISSION_AWAITING: [
                    cancel_handler,
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND, permission_receive,
                    ),
                ],
            },
            fallbacks=[cancel_handler],
        )

        session_conv = ConversationHandler(
            entry_points=[CommandHandler("session", cmd_session)],
            states={
                SESSION_AWAITING: [
                    cancel_handler,
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND, session_receive,
                    ),
                ],
            },
            fallbacks=[cancel_handler],
        )

        self.app.add_handler(new_conv)
        self.app.add_handler(open_conv)
        self.app.add_handler(model_conv)
        self.app.add_handler(summarymodel_conv)
        self.app.add_handler(permission_conv)
        self.app.add_handler(session_conv)
        self.app.add_handler(CommandHandler("start", cmd_start))
        self.app.add_handler(CommandHandler("version", cmd_version))
        self.app.add_handler(CommandHandler("clear", cmd_clear))
        self.app.add_handler(CommandHandler("refresh", cmd_refresh))
        self.app.add_handler(CommandHandler("compact", cmd_compact))
        self.app.add_handler(CommandHandler("stop", cmd_stop))
        self.app.add_handler(CommandHandler("inject", cmd_inject))
        self.app.add_handler(MessageHandler(filters.Document.ALL, ai_file))
        self.app.add_handler(MessageHandler(filters.PHOTO, ai_file))
        self.app.add_handler(MessageHandler(filters.AUDIO, ai_file))
        self.app.add_handler(MessageHandler(filters.VIDEO, ai_file))
        self.app.add_handler(MessageHandler(filters.VOICE, ai_file))
        self.app.add_handler(MessageHandler(filters.VIDEO_NOTE, ai_file))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, ai_chat)
        )

        for attempt in range(1, 6):
            try:
                await self.app.initialize()
                break
            except NetworkError as e:
                if attempt == 5:
                    raise
                logger.warning(
                    "Network error during init (attempt %d/5): %s",
                    attempt, e,
                )
                await asyncio.sleep(5 * attempt)
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        logger.info("Bot started polling.")

    async def stop(self) -> None:
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Bot stopped.")

    # ------------------------------------------------------------------
    # Dispatch: acquire lock or queue, then run AI turn
    # ------------------------------------------------------------------

    def _cleanup_turn(self, uid: int, lock: asyncio.Lock) -> None:
        """Per-user state cleanup shared by dispatch and queue-drain paths."""
        self._running_tasks.pop(uid, None)
        self._inject_queues.pop(uid, None)
        lock.release()

    async def _dispatch(
        self, uid: int, chat_id: int, text: str, update: Update,
    ) -> None:
        """Acquire the per-user lock and run an AI turn, or queue if busy."""
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
                await update.message.reply_text("Queue full. Wait or /stop first.")
            else:
                await q.put((text, chat_id))
                await update.message.reply_text(
                    f"Queued ({q.qsize()}/{self.max_queue_size})."
                )
            return
        await lock.acquire()

        try:
            await self._run_ai_turn(uid, chat_id, text)
        except asyncio.CancelledError:
            await update.message.reply_text("Cancelled.")
            return
        except Exception as e:
            logger.exception("AI turn failed")
            await update.message.reply_text(f"Error: {e}")
            return
        finally:
            self._cleanup_turn(uid, lock)

        await self._drain_message_queue(uid)

    # ------------------------------------------------------------------
    # Core AI turn logic (used by _dispatch and queue drain)
    # ------------------------------------------------------------------

    async def _run_ai_turn(self, uid: int, chat_id: int, text: str) -> None:
        """Run a single AI turn: send thinking msg, stream events, send result.

        Caller must already hold the per-user lock.
        """
        ws = workspace.get_current(uid, self.bot_id)
        if not ws or not os.path.isdir(ws):
            await self.app.bot.send_message(
                chat_id=chat_id,
                text="Workspace not available (deleted?). Use /new or /open.",
            )
            return
        model, summary_model, perm = workspace.get_run_config(ws)

        inject_q: asyncio.Queue[str] = asyncio.Queue(
            maxsize=self.max_queue_size,
        )
        self._inject_queues[uid] = inject_q

        thinking_msg = await self.app.bot.send_message(
            chat_id=chat_id, text="Thinking...",
        )
        self._running_tasks[uid] = asyncio.current_task()

        status_lines: list[str] = []
        last_edit = 0.0

        async def on_event(ev: codex.ChatEvent) -> None:
            nonlocal last_edit
            if ev.kind == "tool":
                status_lines.append(
                    f"» {ev.content.split(chr(10))[0][:80]}"
                )
            elif ev.kind == "file":
                status_lines.append(f"» {ev.content[:80]}")
            else:
                return

            now = asyncio.get_running_loop().time()
            if now - last_edit < 1.5:
                return
            last_edit = now

            display = "Thinking...\n\n" + "\n".join(status_lines[-5:])
            try:
                await thinking_msg.edit_text(display)
            except Exception:
                pass  # Telegram rejects identical edits or rate-limits

        try:
            result = await codex.run(
                text, ws, user_id=uid,
                model=model, summary_model=summary_model, approval=perm,
                on_event=on_event, inject_queue=inject_q,
            )
        finally:
            try:
                await asyncio.shield(thinking_msg.delete())
            except Exception:
                pass

        await self._send_result(chat_id, ws, result)

    async def _send_text(self, chat_id: int, text: str) -> None:
        """Send markdown text as HTML, falling back to plain on parse error."""
        if not text:
            return
        html = _md_to_html(text)
        for chunk in _split_html(html):
            if not chunk.strip():
                continue  # Telegram rejects empty messages
            try:
                await self.app.bot.send_message(
                    chat_id=chat_id, text=chunk, parse_mode="HTML",
                )
            except Exception:
                plain = re.sub(r"<[^>]+>", "", chunk)
                plain = (
                    plain.replace("&lt;", "<")
                         .replace("&gt;", ">")
                         .replace("&amp;", "&")
                )
                if plain.strip():
                    await self.app.bot.send_message(
                        chat_id=chat_id, text=plain,
                    )

    async def _send_attachment(self, chat_id: int, path: str) -> None:
        """Send a file as a Telegram document, notifying the chat on failure."""
        name = os.path.basename(path)
        try:
            with open(path, "rb") as f:
                await self.app.bot.send_document(
                    chat_id=chat_id, document=f, filename=name,
                )
        except Exception as e:
            logger.warning("Failed to send attachment %s: %s", path, e)
            try:
                await self.app.bot.send_message(
                    chat_id=chat_id, text=f"Failed to attach {name}: {e}",
                )
            except Exception:
                pass

    async def _send_result(
        self, chat_id: int, ws: str, result: codex.CodexResult,
    ) -> None:
        """Send the AI's text reply plus any [[attach: ...]] files."""
        for ev in result.events:
            if ev.kind != "text":
                continue
            text, attach_paths = _extract_attachments(ev.content, ws)
            await self._send_text(chat_id, text)
            for path in attach_paths:
                await self._send_attachment(chat_id, path)

    async def _drain_message_queue(self, uid: int) -> None:
        """Process any messages that were queued while the AI was busy."""
        q = self._message_queues.get(uid)
        if not q:
            return
        lock = self._task_locks[uid]

        while not q.empty():
            if lock.locked():
                break  # something else acquired - stop draining

            # Check lock BEFORE dequeuing so we never drop a message.
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
                    await self.app.bot.send_message(
                        chat_id=msg_chat_id, text="Cancelled.",
                    )
                except Exception:
                    pass
                break
            except Exception as e:
                logger.exception("Queued AI chat failed")
                try:
                    await self.app.bot.send_message(
                        chat_id=msg_chat_id, text=f"Error: {e}",
                    )
                except Exception:
                    pass
            finally:
                self._cleanup_turn(uid, lock)

    async def notify_users(self, message: str) -> None:
        """Send a message to all authorized users."""
        if not self.app:
            return
        for uid in self.user_ids:
            try:
                await self.app.bot.send_message(chat_id=uid, text=message)
            except Exception as e:
                logger.warning("Failed to notify user %s: %s", uid, e)
