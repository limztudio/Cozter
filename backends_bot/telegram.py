"""Telegram adapter: wires python-telegram-bot events to BotPlatform."""

from __future__ import annotations

import asyncio
import logging
import os
import re

from telegram import Update
from telegram.error import NetworkError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .. import workspace
from ..utils import split_text_chunks
from .base import (
    AttachmentInfo,
    BotContext,
    BotPlatform,
    COMMAND_NAMES,
    MessageHandle,
    NO_WORKSPACE_TEXT,
    ensure_upload_dir,
)
from .formatting import render_fenced_markdown

logger = logging.getLogger(__name__)

_TELEGRAM_PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_TELEGRAM_TEXT_LIMIT = 4096


# ---------------------------------------------------------------------------
# Markdown -> Telegram HTML
# ---------------------------------------------------------------------------

def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _md_to_html(text: str) -> str:
    """Convert common Markdown to Telegram-compatible HTML."""
    return render_fenced_markdown(
        text,
        render_line=_html_line,
        render_code_block=_html_code_block,
    )


def _html_line(line: str) -> str:
    line = _escape_html(line)
    line = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", line)
    line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
    line = re.sub(r"__(.+?)__", r"<b>\1</b>", line)
    line = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<i>\1</i>", line)
    line = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>", line)
    line = re.sub(r"`([^`]+?)`", r"<code>\1</code>", line)
    return re.sub(r"~~(.+?)~~", r"<s>\1</s>", line)


def _html_code_block(lines: list[str]) -> list[str]:
    escaped = _escape_html("\n".join(lines))
    return [f"<pre>{escaped}</pre>"]


# ---------------------------------------------------------------------------
# Telegram platform adapter
# ---------------------------------------------------------------------------

class TelegramBot(BotPlatform):
    """One-to-one adapter around a python-telegram-bot Application."""

    def __init__(
        self,
        token: str,
        user_ids: list[int | str],
        *,
        recent_limit: int = 10,
        max_queue_size: int = 50,
    ):
        super().__init__(
            user_ids,
            recent_limit=recent_limit,
            max_queue_size=max_queue_size,
        )
        self.token = token
        self.app: Application | None = None

    @property
    def platform_id(self) -> str:
        # Use the numeric Telegram bot id so workspace state stays
        # compatible with installations that pre-date the platform split.
        if self.app is None:
            raise RuntimeError("platform_id is only valid after start()")
        return str(self.app.bot.id)

    # ----- send/edit primitives ------------------------------------------

    async def send_text(
        self, chat_id: str, text: str, *, rich: bool = False,
    ) -> MessageHandle | None:
        if not text:
            return None
        if not rich:
            msg = await self.app.bot.send_message(
                chat_id=chat_id, text=text,
            )
            return MessageHandle(chat_id=str(chat_id), message_id=str(msg.message_id))

        # Rich path: convert markdown → HTML and split for Telegram limits.
        html = _md_to_html(text)
        last: MessageHandle | None = None
        for chunk in split_text_chunks(html, _TELEGRAM_TEXT_LIMIT):
            if not chunk.strip():
                continue
            try:
                msg = await self.app.bot.send_message(
                    chat_id=chat_id, text=chunk, parse_mode="HTML",
                )
            except Exception:
                plain = re.sub(r"<[^>]+>", "", chunk)
                plain = (
                    plain.replace("&lt;", "<")
                         .replace("&gt;", ">")
                         .replace("&amp;", "&")
                )
                if not plain.strip():
                    continue
                msg = await self.app.bot.send_message(
                    chat_id=chat_id, text=plain,
                )
            last = MessageHandle(
                chat_id=str(chat_id), message_id=str(msg.message_id),
            )
        return last

    async def edit_text(
        self, handle: MessageHandle, text: str, *, rich: bool = False,
    ) -> None:
        if not rich:
            await self.app.bot.edit_message_text(
                chat_id=handle.chat_id,
                message_id=int(handle.message_id),
                text=text,
            )
            return

        html = _md_to_html(text)
        try:
            await self.app.bot.edit_message_text(
                chat_id=handle.chat_id,
                message_id=int(handle.message_id),
                text=html,
                parse_mode="HTML",
            )
        except Exception:
            plain = re.sub(r"<[^>]+>", "", html)
            plain = (
                plain.replace("&lt;", "<")
                     .replace("&gt;", ">")
                     .replace("&amp;", "&")
            )
            await self.app.bot.edit_message_text(
                chat_id=handle.chat_id,
                message_id=int(handle.message_id),
                text=plain,
            )

    async def delete_message(self, handle: MessageHandle) -> None:
        await self.app.bot.delete_message(
            chat_id=handle.chat_id,
            message_id=int(handle.message_id),
        )

    async def send_file(self, chat_id: str, path: str) -> None:
        name = os.path.basename(path)
        ext = os.path.splitext(name)[1].lower()
        if ext in _TELEGRAM_PHOTO_EXTENSIONS:
            try:
                with open(path, "rb") as f:
                    await self.app.bot.send_photo(
                        chat_id=chat_id, photo=f, filename=name,
                    )
                return
            except Exception:
                logger.warning(
                    "Failed to send %s as photo; falling back to document",
                    path,
                    exc_info=True,
                )
        with open(path, "rb") as f:
            await self.app.bot.send_document(
                chat_id=chat_id, document=f, filename=name,
            )

    # ----- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        self.app = (
            Application.builder()
            .token(self.token)
            .concurrent_updates(True)
            .build()
        )

        for name in COMMAND_NAMES:
            self.app.add_handler(
                CommandHandler(name, self._make_command_handler(name)),
            )

        attachment_filter = (
            filters.Document.ALL
            | filters.PHOTO
            | filters.AUDIO
            | filters.VIDEO
            | filters.VOICE
            | filters.VIDEO_NOTE
        )
        self.app.add_handler(
            MessageHandler(attachment_filter, self._on_file),
        )
        self.app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND, self._on_text,
            ),
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
        # Restore in-flight / queued messages before polling begins so a
        # new user message can't race past the restored backlog and run
        # out of order. app.bot.send_message works after initialize(), so
        # drain can still post "Thinking..." during restore.
        await self.restore_queues()
        await self.start_scheduler()
        await self.app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started polling.")

    async def stop(self) -> None:
        await self.stop_scheduler()
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram bot stopped.")

    # ----- event handlers -------------------------------------------------

    def _make_command_handler(self, name: str):
        async def handler(
            update: Update, context: ContextTypes.DEFAULT_TYPE,
        ) -> None:
            if not self._precheck(update):
                return
            args = " ".join(context.args) if context.args else ""
            ctx = self._build_context(
                update, text="", command=name, args=args,
            )
            if ctx is None:
                return
            await self.dispatch_command(ctx)
        return handler

    async def _on_text(
        self, update: Update, _context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not self._precheck(update):
            return
        text = (update.message.text or "").strip()
        ctx = self._build_context(update, text=text)
        if ctx is None:
            return
        await self.dispatch_text(ctx)

    async def _on_file(
        self, update: Update, _context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        # Early auth check: refuse to download files for non-whitelisted
        # users or for events without an effective_user (channel posts).
        if not self._precheck(update):
            return
        uid = str(update.effective_user.id)
        ws = workspace.get_current(uid, self.platform_id)
        message = update.message

        if message.document:
            tg_file = await message.document.get_file()
            filename = (
                message.document.file_name
                or f"file_{message.document.file_id}"
            )
            kind = "document"
        elif message.photo:
            photo = message.photo[-1]
            tg_file = await photo.get_file()
            filename = f"photo_{photo.file_id}.jpg"
            kind = "photo"
        elif message.audio:
            tg_file = await message.audio.get_file()
            filename = (
                message.audio.file_name
                or f"audio_{message.audio.file_id}.ogg"
            )
            kind = "audio"
        elif message.video:
            tg_file = await message.video.get_file()
            filename = (
                message.video.file_name
                or f"video_{message.video.file_id}.mp4"
            )
            kind = "video"
        elif message.voice:
            tg_file = await message.voice.get_file()
            filename = f"voice_{message.voice.file_id}.ogg"
            kind = "voice"
        elif message.video_note:
            tg_file = await message.video_note.get_file()
            filename = f"videonote_{message.video_note.file_id}.mp4"
            kind = "video note"
        else:
            return

        caption = (message.caption or "").strip()
        # Strip any path components to prevent traversal from malicious
        # filenames like "../../etc/passwd".
        filename = os.path.basename(filename) or f"file_{tg_file.file_id}"

        if not ws or not os.path.isdir(ws):
            ctx = self._build_context(update, text="")
            if ctx:
                await ctx.reply_text(NO_WORKSPACE_TEXT)
            return

        upload_dir = ensure_upload_dir(ws)
        local_path = os.path.join(upload_dir, filename)
        try:
            await tg_file.download_to_drive(local_path)
        except Exception as e:
            ctx = self._build_context(update, text="")
            if ctx:
                await ctx.reply_text(f"Failed to download file: {e}")
            return

        ctx = self._build_context(
            update, text="",
            attachment=AttachmentInfo(
                local_path=local_path,
                filename=filename,
                kind=kind,
                caption=caption,
            ),
        )
        if ctx is None:
            return
        await self.dispatch_file(ctx)

    def _precheck(self, update: Update) -> bool:
        """Reject events without an effective_user or from non-whitelisted users.

        Runs before any side effects (file download, subprocess spawn) so an
        unauthorized user can't cause work to happen.
        """
        user = update.effective_user
        if user is None:
            return False
        chat = update.effective_chat
        chat_id = str(chat.id) if chat else ""
        if not self.authorized(str(user.id), chat_id):
            logger.warning(
                "Unauthorized access attempt from user %s (%s)",
                user.id, user.full_name,
            )
            return False
        return True

    async def send_startup_messages(
        self, version: str, commit_date: str,
    ) -> None:
        """Greet each authorized user with their current workspace info."""
        for uid in self.notify_targets:
            ws = workspace.get_current(uid, self.platform_id)
            msg = (
                f"Cozter started.\nVersion: {version}\nUpdated: {commit_date}"
            )
            if ws:
                msg += f"\nWorkspace: {ws}"
            else:
                msg += "\nNo workspace selected. Use /new or /open."
            try:
                await self.send_text(uid, msg)
            except Exception as e:
                logger.warning(
                    "Failed to notify user %s: %s", uid, e,
                )

    def _build_context(
        self,
        update: Update,
        *,
        text: str,
        command: str | None = None,
        args: str = "",
        attachment: AttachmentInfo | None = None,
    ) -> BotContext | None:
        if not update.effective_user:
            return None
        return BotContext(
            user_id=str(update.effective_user.id),
            chat_id=str(update.effective_chat.id),
            text=text,
            command=command,
            args=args,
            attachment=attachment,
            platform=self,
        )
