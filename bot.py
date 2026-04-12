import asyncio
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
from . import workspace

logger = logging.getLogger(__name__)


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _split_html(text: str, limit: int = 4096) -> list[str]:
    """Split text into chunks no larger than limit, breaking at newlines when possible.

    Splitting at arbitrary byte offsets risks cutting inside an HTML tag, which
    causes Telegram to reject the message. Newline boundaries are safe because
    _md_to_html never produces tags that span multiple lines.
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
            # No newline — hard split, keep all characters
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
                result.append(f"<pre>{_escape_html(chr(10).join(code_buf))}</pre>")
                code_buf.clear()
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_buf.append(line)
            continue

        line = _escape_html(line)

        # Headers → bold
        line = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", line)
        # Bold: **text** or __text__
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        line = re.sub(r"__(.+?)__", r"<b>\1</b>", line)
        # Italic: *text* or _text_  (but not inside words with underscores)
        line = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<i>\1</i>", line)
        line = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>", line)
        # Inline code: `text`
        line = re.sub(r"`([^`]+?)`", r"<code>\1</code>", line)
        # Strikethrough: ~~text~~
        line = re.sub(r"~~(.+?)~~", r"<s>\1</s>", line)

        result.append(line)

    # Unclosed code block
    if in_code_block and code_buf:
        result.append(f"<pre>{_escape_html(chr(10).join(code_buf))}</pre>")

    return "\n".join(result)

# Conversation states
NEW_AWAITING_DIR = 0
OPEN_AWAITING_DIR = 1
MODEL_AWAITING = 2
PERMISSION_AWAITING = 3
SESSION_AWAITING = 4
SUMMARY_MODEL_AWAITING = 5


def _authorized(user_ids: list[int]):
    """Decorator that restricts handler to authorized user_ids."""
    def decorator(func):
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if update.effective_user.id not in user_ids:
                logger.warning(
                    "Unauthorized access attempt from user %s (%s)",
                    update.effective_user.id, update.effective_user.full_name,
                )
                return
            return await func(update, context)
        return wrapper
    return decorator


class CozterBot:
    def __init__(self, token: str, user_ids: list[int], recent_limit: int = 10, max_queue_size: int = 50):
        self.token = token
        self.user_ids = user_ids
        self.recent_limit = recent_limit
        self.max_queue_size = max_queue_size
        self.app: Application | None = None
        self._running_tasks: dict[int, asyncio.Task] = {}     # user_id -> task
        self._task_locks: dict[int, asyncio.Lock] = {}         # user_id -> lock
        self._message_queues: dict[int, asyncio.Queue] = {}    # user_id -> queued messages
        self._inject_queues: dict[int, asyncio.Queue] = {}     # user_id -> inject queue
        self._thinking_msgs: dict[int, object] = {}            # user_id -> Message

    @property
    def bot_id(self) -> int:
        return self.app.bot.id

    async def start(self) -> None:
        self.app = Application.builder().token(self.token).concurrent_updates(True).build()

        # --- simple commands ---

        @_authorized(self.user_ids)
        async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text("Cozter bot is running.")

        @_authorized(self.user_ids)
        async def cmd_version(update: Update, context: ContextTypes.DEFAULT_TYPE):
            from . import updater
            ver, date = await asyncio.gather(
                asyncio.to_thread(updater.get_current_version),
                asyncio.to_thread(updater.get_last_commit_date),
            )
            await update.message.reply_text(f"Version: {ver}\nUpdated: {date}")

        @_authorized(self.user_ids)
        async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text("No workspace selected. Use /new or /open first.")
                return
            new_sess = session.create_session(ws)
            session.set_current_session_id(ws, uid, new_sess["id"])
            await update.message.reply_text("Conversation cleared. Next message starts a new session.")

        # --- /new conversation ---

        @_authorized(self.user_ids)
        async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            current = workspace.get_current(uid, self.bot_id)
            msg = f"Current workspace: {current or '(none)'}\n\nEnter the full path for the new workspace directory (or /cancel):"
            await update.message.reply_text(msg)
            return NEW_AWAITING_DIR

        @_authorized(self.user_ids)
        async def new_receive_dir(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            path = update.message.text.strip()

            if os.path.exists(path):
                await update.message.reply_text(
                    f"Directory already exists:\n{path}\n\nPlease choose a different path (or /cancel):"
                )
                return NEW_AWAITING_DIR

            try:
                os.makedirs(path)
            except OSError as e:
                await update.message.reply_text(
                    f"Failed to create directory: {e}\n\nPlease try again (or /cancel):"
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
        async def cmd_open(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
            lines.append("\nEnter a directory path or number from the list (or /cancel):")
            await update.message.reply_text("\n".join(lines))
            return OPEN_AWAITING_DIR

        @_authorized(self.user_ids)
        async def open_receive_dir(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
                    f"Directory does not exist:\n{path}\n\nPlease enter a valid directory (or /cancel):"
                )
                return OPEN_AWAITING_DIR

            workspace.ensure_cozter_dir(path)
            workspace.select_workspace(uid, path, self.bot_id)
            await update.message.reply_text(
                f"Workspace selected:\n{path}"
            )
            return ConversationHandler.END

        # --- /model conversation ---

        @_authorized(self.user_ids)
        async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text("No workspace selected. Use /new or /open first.")
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
        async def model_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text("No workspace selected. Use /new or /open first.")
                return ConversationHandler.END
            text = update.message.text.strip()
            options = workspace.AVAILABLE_MODELS

            if text.isdigit():
                idx = int(text) - 1
                if 0 <= idx < len(options):
                    model = options[idx]
                else:
                    await update.message.reply_text("Invalid number. Try again (or /cancel):")
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
        async def cmd_summarymodel(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text("No workspace selected. Use /new or /open first.")
                return ConversationHandler.END

            current = workspace.get_summary_model(ws)
            options = workspace.AVAILABLE_MODELS
            lines = [f"Current summary model: {current}\n", "Available models:"]
            for i, m in enumerate(options, 1):
                marker = " <-" if m == current else ""
                lines.append(f"  {i}. {m}{marker}")
            lines.append("\nEnter a number or model name (or /cancel):")
            await update.message.reply_text("\n".join(lines))
            return SUMMARY_MODEL_AWAITING

        @_authorized(self.user_ids)
        async def summarymodel_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text("No workspace selected. Use /new or /open first.")
                return ConversationHandler.END
            text = update.message.text.strip()
            options = workspace.AVAILABLE_MODELS

            if text.isdigit():
                idx = int(text) - 1
                if 0 <= idx < len(options):
                    model = options[idx]
                else:
                    await update.message.reply_text("Invalid number. Try again (or /cancel):")
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
        async def cmd_permission(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text("No workspace selected. Use /new or /open first.")
                return ConversationHandler.END

            current = workspace.get_permission(ws)
            options = workspace.AVAILABLE_PERMISSIONS
            lines = [f"Current permission: {current}\n", "Available modes:"]
            for i, p in enumerate(options, 1):
                marker = " <-" if p == current else ""
                desc = workspace.PERMISSION_DESCRIPTIONS[p]
                lines.append(f"  {i}. {p} — {desc}{marker}")
            lines.append("\nEnter a number or mode name (or /cancel):")
            await update.message.reply_text("\n".join(lines))
            return PERMISSION_AWAITING

        @_authorized(self.user_ids)
        async def permission_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text("No workspace selected. Use /new or /open first.")
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
            await update.message.reply_text(f"Permission set to: {perm}\n{desc}")
            return ConversationHandler.END

        # --- /session conversation ---

        @_authorized(self.user_ids)
        async def cmd_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text("No workspace selected. Use /new or /open first.")
                return ConversationHandler.END

            current_sid = session.get_current_session_id(ws, uid)
            sessions = session.list_sessions(ws)

            lines = []
            if current_sid:
                current_data = session.load_session(ws, current_sid)
                if current_data:
                    count = len(current_data.get("messages", []))
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
                    lines.append(f"  {i}. {s['name']} ({s['message_count']} msgs){marker}")
            else:
                lines.append("\nNo sessions yet.")

            lines.append("\nEnter a number to switch, or 'new' to create (or /cancel):")
            await update.message.reply_text("\n".join(lines))
            return SESSION_AWAITING

        @_authorized(self.user_ids)
        async def session_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text("No workspace selected. Use /new or /open first.")
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
        async def cmd_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text("No workspace selected. Use /new or /open first.")
                return

            # Clear codex CLI's internal session state if it exists
            codex_dir = os.path.join(ws, ".codex")
            if os.path.isdir(codex_dir):
                shutil.rmtree(codex_dir, ignore_errors=True)
                logger.info("Cleared codex session dir: %s", codex_dir)

            await update.message.reply_text(
                "Codex CLI session refreshed. Your conversation history is preserved."
            )

        # --- /compact command ---

        @_authorized(self.user_ids)
        async def cmd_compact(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid, self.bot_id)
            if not ws:
                await update.message.reply_text("No workspace selected. Use /new or /open first.")
                return

            sid = session.ensure_session(ws, uid)
            args = (context.args[0] if context.args else "").strip().lower()

            if args == "now":
                await update.message.reply_text("Compacting session...")
                summary_model = workspace.get_summary_model(ws)
                new_summary = await codex._compact_session(ws, sid, summary_model)
                if new_summary:
                    async with codex._get_workspace_lock(ws):
                        session.set_summary(ws, sid, new_summary,
                                            keep_recent=codex.KEEP_RECENT_AFTER_COMPACT)
                    await update.message.reply_text("Session compacted.")
                else:
                    await update.message.reply_text("Compaction produced no output — session unchanged.")
                return

            if args.isdigit():
                interval = int(args)
                session.set_compact_interval(ws, sid, interval)
                await update.message.reply_text(f"Compact interval set to {interval} messages.")
                return

            sess_data = session.load_session(ws, sid) or {}
            current = sess_data.get("compact_interval", 20)
            total = sess_data.get("compacted_count", 0) + len(sess_data.get("messages", []))
            summary = sess_data.get("summary")
            lines = [
                f"Compact interval: {current} messages",
                f"Total messages: {total}",
                f"Has summary: {'yes' if summary else 'no'}",
                "",
                "Usage:",
                "  /compact <number> — set interval",
                "  /compact now — compact immediately",
            ]
            await update.message.reply_text("\n".join(lines))

        # --- shared cancel ---

        @_authorized(self.user_ids)
        async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text("Cancelled.")
            return ConversationHandler.END

        # --- /stop command ---

        @_authorized(self.user_ids)
        async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            task = self._running_tasks.get(uid)
            if task and not task.done():
                task.cancel()
                # Also clear the message queue so queued messages don't run
                q = self._message_queues.get(uid)
                if q:
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                await update.message.reply_text("Cancelling...")
            else:
                await update.message.reply_text("Nothing is running.")

        # --- /inject command ---

        @_authorized(self.user_ids)
        async def cmd_inject(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

        # --- AI chat (default for non-command messages) ---

        @_authorized(self.user_ids)
        async def ai_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            text = update.message.text.strip()
            if not text:
                return
            ws = workspace.get_current(uid, self.bot_id)
            if not ws or not os.path.isdir(ws):
                await update.message.reply_text(
                    "No workspace selected (or it was deleted). Use /new or /open."
                )
                return

            if uid not in self._task_locks:
                self._task_locks[uid] = asyncio.Lock()
            lock = self._task_locks[uid]

            # If AI is already running, queue the message for later.
            if lock.locked():
                if uid not in self._message_queues:
                    self._message_queues[uid] = asyncio.Queue(maxsize=self.max_queue_size)
                q = self._message_queues[uid]
                if q.full():
                    await update.message.reply_text("Queue full. Wait or /stop first.")
                else:
                    await q.put((text, update.effective_chat.id))
                    await update.message.reply_text(
                        f"Queued ({q.qsize()}/{self.max_queue_size})."
                    )
                return
            await lock.acquire()

            chat_id = update.effective_chat.id
            try:
                await self._run_ai_turn(uid, chat_id, text)
            except asyncio.CancelledError:
                await update.message.reply_text("Cancelled.")
                return
            except Exception as e:
                logger.exception("AI chat failed")
                await update.message.reply_text(f"Error: {e}")
                return
            finally:
                self._running_tasks.pop(uid, None)
                self._inject_queues.pop(uid, None)
                self._thinking_msgs.pop(uid, None)
                lock.release()

            # Drain the message queue while there are pending messages.
            await self._drain_message_queue(uid)

        # --- register handlers ---

        cancel_handler = CommandHandler("cancel", cancel)

        new_conv = ConversationHandler(
            entry_points=[CommandHandler("new", cmd_new)],
            states={
                NEW_AWAITING_DIR: [
                    cancel_handler,
                    MessageHandler(filters.TEXT & ~filters.COMMAND, new_receive_dir),
                ],
            },
            fallbacks=[cancel_handler],
        )

        open_conv = ConversationHandler(
            entry_points=[CommandHandler("open", cmd_open)],
            states={
                OPEN_AWAITING_DIR: [
                    cancel_handler,
                    MessageHandler(filters.TEXT & ~filters.COMMAND, open_receive_dir),
                ],
            },
            fallbacks=[cancel_handler],
        )

        model_conv = ConversationHandler(
            entry_points=[CommandHandler("model", cmd_model)],
            states={
                MODEL_AWAITING: [
                    cancel_handler,
                    MessageHandler(filters.TEXT & ~filters.COMMAND, model_receive),
                ],
            },
            fallbacks=[cancel_handler],
        )

        summarymodel_conv = ConversationHandler(
            entry_points=[CommandHandler("summarymodel", cmd_summarymodel)],
            states={
                SUMMARY_MODEL_AWAITING: [
                    cancel_handler,
                    MessageHandler(filters.TEXT & ~filters.COMMAND, summarymodel_receive),
                ],
            },
            fallbacks=[cancel_handler],
        )

        permission_conv = ConversationHandler(
            entry_points=[CommandHandler("permission", cmd_permission)],
            states={
                PERMISSION_AWAITING: [
                    cancel_handler,
                    MessageHandler(filters.TEXT & ~filters.COMMAND, permission_receive),
                ],
            },
            fallbacks=[cancel_handler],
        )

        session_conv = ConversationHandler(
            entry_points=[CommandHandler("session", cmd_session)],
            states={
                SESSION_AWAITING: [
                    cancel_handler,
                    MessageHandler(filters.TEXT & ~filters.COMMAND, session_receive),
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
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ai_chat))

        for attempt in range(1, 6):
            try:
                await self.app.initialize()
                break
            except NetworkError as e:
                if attempt == 5:
                    raise
                logger.warning("Network error during init (attempt %d/5): %s", attempt, e)
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
    # Core AI turn logic (used by ai_chat handler and queue drain)
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
        model = workspace.get_model(ws)
        summary_model = workspace.get_summary_model(ws)
        perm = workspace.get_permission(ws)

        # Create inject queue for this run
        inject_q: asyncio.Queue[str] = asyncio.Queue(maxsize=self.max_queue_size)
        self._inject_queues[uid] = inject_q

        # Send the "Thinking..." message that we'll update with progress
        thinking_msg = await self.app.bot.send_message(chat_id=chat_id, text="Thinking...")
        self._thinking_msgs[uid] = thinking_msg
        self._running_tasks[uid] = asyncio.current_task()

        # Streaming callback — update the Thinking message with tool/file events
        status_lines: list[str] = []
        last_edit = 0.0

        async def on_event(ev: codex.ChatEvent) -> None:
            nonlocal last_edit
            if ev.kind == "tool":
                status_lines.append(f"» {ev.content.split(chr(10))[0][:80]}")
            elif ev.kind == "file":
                status_lines.append(f"» {ev.content[:80]}")
            else:
                return  # don't update for text events

            now = asyncio.get_running_loop().time()
            if now - last_edit < 1.5:
                return
            last_edit = now

            display = "Thinking...\n\n" + "\n".join(status_lines[-5:])
            try:
                await thinking_msg.edit_text(display)
            except Exception:
                pass  # Telegram rejects identical edits or rate-limits

        result = await codex.run(
            text, ws, user_id=uid,
            model=model, summary_model=summary_model, approval=perm,
            on_event=on_event, inject_queue=inject_q,
        )

        # Delete the thinking message now that we have the answer
        try:
            await thinking_msg.delete()
        except Exception:
            pass

        await self._send_result(chat_id, result)

    async def _send_result(self, chat_id: int, result: codex.CodexResult) -> None:
        """Send the final AI text response to the chat."""
        for ev in result.events:
            if ev.kind != "text":
                continue
            html = _md_to_html(ev.content)
            for chunk in _split_html(html):
                if not chunk.strip():
                    continue  # skip empty chunks — Telegram rejects them
                try:
                    await self.app.bot.send_message(
                        chat_id=chat_id, text=chunk, parse_mode="HTML",
                    )
                except Exception:
                    plain = re.sub(r"<[^>]+>", "", chunk)
                    plain = (plain.replace("&lt;", "<")
                                  .replace("&gt;", ">")
                                  .replace("&amp;", "&"))
                    if plain.strip():
                        await self.app.bot.send_message(chat_id=chat_id, text=plain)

    async def _drain_message_queue(self, uid: int) -> None:
        """Process any messages that were queued while the AI was busy."""
        q = self._message_queues.get(uid)
        if not q:
            return

        while not q.empty():
            lock = self._task_locks[uid]
            if lock.locked():
                break  # something else acquired — stop draining

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
                self._running_tasks.pop(uid, None)
                self._inject_queues.pop(uid, None)
                self._thinking_msgs.pop(uid, None)
                lock.release()

    async def notify_users(self, message: str) -> None:
        """Send a message to all authorized users."""
        if not self.app:
            return
        for uid in self.user_ids:
            try:
                await self.app.bot.send_message(chat_id=uid, text=message)
            except Exception as e:
                logger.warning("Failed to notify user %s: %s", uid, e)
