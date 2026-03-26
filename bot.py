import logging
import os

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from . import auth
from . import session
from . import workspace
from .chatgpt import ChatGPTClient

logger = logging.getLogger(__name__)

# Conversation states
NEW_AWAITING_DIR = 0
OPEN_AWAITING_DIR = 1
MODEL_AWAITING = 2
EFFORT_AWAITING = 3
SESSION_AWAITING = 4


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
    def __init__(self, token: str, user_ids: list[int], recent_limit: int = 10):
        self.token = token
        self.user_ids = user_ids
        self.recent_limit = recent_limit
        self.app: Application | None = None
        self.ai = ChatGPTClient()

    async def start(self) -> None:
        self.app = Application.builder().token(self.token).build()

        # --- simple commands ---

        @_authorized(self.user_ids)
        async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text("Cozter bot is running.")

        @_authorized(self.user_ids)
        async def cmd_version(update: Update, context: ContextTypes.DEFAULT_TYPE):
            from . import updater
            ver = updater.get_current_version()
            date = updater.get_last_commit_date()
            await update.message.reply_text(f"Version: {ver}\nUpdated: {date}")

        @_authorized(self.user_ids)
        async def cmd_account(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if not auth.is_logged_in():
                await update.message.reply_text(
                    "Not logged in.\nRestart the script to trigger login."
                )
                return
            tokens = auth.get_tokens()
            await update.message.reply_text(
                f"Account: {tokens.get('email', '?')}\n"
                f"Plan: {tokens.get('plan', '?')}"
            )

        @_authorized(self.user_ids)
        async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid)
            if ws:
                # Create a fresh session
                new_sess = session.create_session(ws)
                session.set_current_session_id(ws, uid, new_sess["id"])
            self.ai.clear_history(uid)
            await update.message.reply_text("Conversation cleared. New session started.")

        # --- /new conversation ---

        @_authorized(self.user_ids)
        async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            current = workspace.get_current(uid)
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
            workspace.select_workspace(uid, path)
            await update.message.reply_text(
                f"Workspace created and selected:\n{path}"
            )
            return ConversationHandler.END

        # --- /open conversation ---

        @_authorized(self.user_ids)
        async def cmd_open(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            current = workspace.get_current(uid)
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
            workspace.select_workspace(uid, path)
            await update.message.reply_text(
                f"Workspace selected:\n{path}"
            )
            return ConversationHandler.END

        # --- /model conversation ---

        @_authorized(self.user_ids)
        async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid)
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
            ws = workspace.get_current(uid)
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

        # --- /effort conversation ---

        @_authorized(self.user_ids)
        async def cmd_effort(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid)
            if not ws:
                await update.message.reply_text("No workspace selected. Use /new or /open first.")
                return ConversationHandler.END

            current = workspace.get_effort(ws)
            options = workspace.AVAILABLE_EFFORTS
            lines = [f"Current effort: {current}\n", "Available levels:"]
            for i, e in enumerate(options, 1):
                marker = " <-" if e == current else ""
                lines.append(f"  {i}. {e}{marker}")
            lines.append("\nEnter a number or level name (or /cancel):")
            await update.message.reply_text("\n".join(lines))
            return EFFORT_AWAITING

        @_authorized(self.user_ids)
        async def effort_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid)
            text = update.message.text.strip().lower()
            options = workspace.AVAILABLE_EFFORTS

            if text.isdigit():
                idx = int(text) - 1
                if 0 <= idx < len(options):
                    effort = options[idx]
                else:
                    await update.message.reply_text("Invalid number. Try again (or /cancel):")
                    return EFFORT_AWAITING
            elif text in options:
                effort = text
            else:
                await update.message.reply_text(
                    f"Unknown effort level: {text}\nTry again (or /cancel):"
                )
                return EFFORT_AWAITING

            workspace.set_effort(ws, effort)
            await update.message.reply_text(f"Effort set to: {effort}")
            return ConversationHandler.END

        # --- /session conversation ---

        @_authorized(self.user_ids)
        async def cmd_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid)
            if not ws:
                await update.message.reply_text("No workspace selected. Use /new or /open first.")
                return ConversationHandler.END

            current_sid = session.get_current_session_id(ws, uid)
            sessions = session.list_sessions(ws)

            lines = []
            if current_sid:
                current_data = session.load_session(ws, current_sid)
                if current_data:
                    lines.append(f"Current session: {current_data.get('name', current_sid[:8])} ({current_data.get('message_count', len(current_data.get('messages', [])))} msgs)")
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
            ws = workspace.get_current(uid)
            text = update.message.text.strip()
            sessions = session.list_sessions(ws)

            if text.lower() == "new":
                new_sess = session.create_session(ws)
                session.set_current_session_id(ws, uid, new_sess["id"])
                self.ai.clear_history(uid)
                await self.ai.load_session(uid, ws, new_sess["id"])
                await update.message.reply_text(
                    f"New session created: {new_sess['name']}\nConversation cleared."
                )
                return ConversationHandler.END

            if text.isdigit():
                idx = int(text) - 1
                if 0 <= idx < len(sessions):
                    chosen = sessions[idx]
                    session.set_current_session_id(ws, uid, chosen["id"])
                    self.ai.clear_history(uid)
                    await self.ai.load_session(uid, ws, chosen["id"])
                    await update.message.reply_text(
                        f"Switched to: {chosen['name']}\nPrevious conversation loaded."
                    )
                    return ConversationHandler.END

            await update.message.reply_text(
                "Invalid input. Enter a number, 'new', or /cancel:"
            )
            return SESSION_AWAITING

        # --- shared cancel ---

        @_authorized(self.user_ids)
        async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text("Cancelled.")
            return ConversationHandler.END

        # --- AI chat (default for non-command messages) ---

        @_authorized(self.user_ids)
        async def ai_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            ws = workspace.get_current(uid)
            if not ws:
                await update.message.reply_text(
                    "No workspace selected. Use /new or /open first."
                )
                return

            model = workspace.get_model(ws)
            effort = workspace.get_effort(ws)

            await update.message.reply_text("Thinking...")
            try:
                reply = await self.ai.chat(uid, update.message.text, ws, model=model, effort=effort)
            except Exception as e:
                logger.exception("AI chat failed")
                await update.message.reply_text(f"Error: {e}")
                return

            # Telegram has a 4096 char limit per message
            for i in range(0, len(reply), 4096):
                await update.message.reply_text(reply[i:i + 4096])

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

        effort_conv = ConversationHandler(
            entry_points=[CommandHandler("effort", cmd_effort)],
            states={
                EFFORT_AWAITING: [
                    cancel_handler,
                    MessageHandler(filters.TEXT & ~filters.COMMAND, effort_receive),
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
        self.app.add_handler(effort_conv)
        self.app.add_handler(session_conv)
        self.app.add_handler(CommandHandler("start", cmd_start))
        self.app.add_handler(CommandHandler("version", cmd_version))
        self.app.add_handler(CommandHandler("account", cmd_account))
        self.app.add_handler(CommandHandler("clear", cmd_clear))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ai_chat))

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        logger.info("Bot started polling.")

    async def stop(self) -> None:
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Bot stopped.")

    async def notify_users(self, message: str) -> None:
        """Send a message to all authorized users."""
        if not self.app:
            return
        for uid in self.user_ids:
            try:
                await self.app.bot.send_message(chat_id=uid, text=message)
            except Exception as e:
                logger.warning("Failed to notify user %s: %s", uid, e)
