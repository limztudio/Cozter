import asyncio
import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logger = logging.getLogger(__name__)


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
    def __init__(self, token: str, user_ids: list[int]):
        self.token = token
        self.user_ids = user_ids
        self.app: Application | None = None

    async def start(self) -> None:
        self.app = Application.builder().token(self.token).build()

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
        async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text(update.message.text)

        self.app.add_handler(CommandHandler("start", cmd_start))
        self.app.add_handler(CommandHandler("version", cmd_version))
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
