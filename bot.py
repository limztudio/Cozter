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
from . import workspace

logger = logging.getLogger(__name__)

# Conversation states
NEW_AWAITING_DIR = 0
OPEN_AWAITING_DIR = 1


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

        # --- shared cancel ---

        @_authorized(self.user_ids)
        async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text("Cancelled.")
            return ConversationHandler.END

        # --- fallback echo (only outside conversations) ---

        @_authorized(self.user_ids)
        async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text(update.message.text)

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

        self.app.add_handler(new_conv)
        self.app.add_handler(open_conv)
        self.app.add_handler(CommandHandler("start", cmd_start))
        self.app.add_handler(CommandHandler("version", cmd_version))
        self.app.add_handler(CommandHandler("account", cmd_account))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

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
