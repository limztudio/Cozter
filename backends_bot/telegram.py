"""Telegram adapter: wires python-telegram-bot events to BotPlatform."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from urllib.parse import urlparse

import aiohttp
from telegram import Update
from telegram.error import NetworkError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..config import (
    DEFAULT_MAX_UPLOAD_BYTES,
    DEFAULT_MESSAGE_QUEUE_SIZE,
    DEFAULT_RECENT_WORKSPACE_LIMIT,
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
    UploadTooLargeError,
    copy_file_with_limit,
    ensure_upload_dir,
    upload_limit_message,
    upload_size_exceeds_limit,
    write_limited_async_stream,
)
from .formatting import render_fenced_markdown
from .formatting import escape_html_entities, strip_html_markup

logger = logging.getLogger(__name__)

_TELEGRAM_PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_TELEGRAM_TEXT_LIMIT = 4096


# ---------------------------------------------------------------------------
# Markdown -> Telegram HTML
# ---------------------------------------------------------------------------

# Telegram and Slack share the same HTML-entity escapes; the shared
# helper lives in formatting.py.
_escape_html = escape_html_entities


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
        recent_limit: int = DEFAULT_RECENT_WORKSPACE_LIMIT,
        max_queue_size: int = DEFAULT_MESSAGE_QUEUE_SIZE,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    ):
        super().__init__(
            user_ids,
            recent_limit=recent_limit,
            max_queue_size=max_queue_size,
            max_upload_bytes=max_upload_bytes,
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
                plain = strip_html_markup(chunk)
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
            plain = strip_html_markup(html)
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
        self._check_upload_path(path)
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
        # ``filters.UpdateType.MESSAGE`` restricts these handlers to genuine
        # new messages. Without it, an edited message also matches (its text
        # lives in ``update.edited_message``, so ``update.message`` is None)
        # and ``_on_text`` / ``_on_file`` would crash dereferencing it - a
        # crash on the very common action of fixing a typo in a prior message.
        self.app.add_handler(
            MessageHandler(
                attachment_filter & filters.UpdateType.MESSAGE,
                self._on_file,
            ),
        )
        self.app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND & filters.UpdateType.MESSAGE,
                self._on_text,
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
        await self._start_daemon_services()
        await self.app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started polling.")

    async def stop(self) -> None:
        await self._stop_daemon_services()
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
            media = message.document
            filename = (
                message.document.file_name
                or f"file_{message.document.file_id}"
            )
            kind = "document"
        elif message.photo:
            media = message.photo[-1]
            filename = f"photo_{media.file_id}.jpg"
            kind = "photo"
        elif message.audio:
            media = message.audio
            filename = (
                message.audio.file_name
                or f"audio_{message.audio.file_id}.ogg"
            )
            kind = "audio"
        elif message.video:
            media = message.video
            filename = (
                message.video.file_name
                or f"video_{message.video.file_id}.mp4"
            )
            kind = "video"
        elif message.voice:
            media = message.voice
            filename = f"voice_{message.voice.file_id}.ogg"
            kind = "voice"
        elif message.video_note:
            media = message.video_note
            filename = f"videonote_{message.video_note.file_id}.mp4"
            kind = "video note"
        else:
            return

        try:
            self._check_upload_size(getattr(media, "file_size", None))
        except UploadTooLargeError:
            ctx = self._build_context(update, text="")
            if ctx:
                await ctx.reply_text(upload_limit_message(self.max_upload_bytes))
            return

        tg_file = await media.get_file()
        try:
            # Some update objects omit a size but the subsequent getFile
            # response includes it. Reject before opening the download in
            # that case too.
            self._check_upload_size(getattr(tg_file, "file_size", None))
        except UploadTooLargeError:
            ctx = self._build_context(update, text="")
            if ctx:
                await ctx.reply_text(upload_limit_message(self.max_upload_bytes))
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
            await _download_telegram_file(
                tg_file, local_path, self.max_upload_bytes,
            )
        except UploadTooLargeError:
            ctx = self._build_context(update, text="")
            if ctx:
                await ctx.reply_text(upload_limit_message(self.max_upload_bytes))
            return
        except Exception as e:
            ctx = self._build_context(update, text="")
            if ctx:
                await ctx.reply_text(f"Failed to download file: {e}")
            return

        ctx = self._build_context(
            update, text="",
            attachment=AttachmentInfo(local_path, filename, kind, caption),
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
        return self.make_context(
            update.effective_user.id,
            update.effective_chat.id,
            text=text,
            command=command,
            args=args,
            attachment=attachment,
        )


async def _download_telegram_file(
    tg_file: object,
    local_path: str,
    max_upload_bytes: int,
) -> None:
    """Stream a Telegram file into an atomically-created bounded upload.

    ``python-telegram-bot``'s convenient ``download_to_drive`` first
    materializes the complete response in memory, then writes it to disk.
    That is unsafe when an update lacks a trustworthy ``file_size``. Files
    returned by ``get_file`` carry either a signed HTTP(S) URL or a local Bot
    API path, both of which can be copied while enforcing our own byte cap.
    """
    file_path = getattr(tg_file, "file_path", None)
    if not isinstance(file_path, str) or not file_path:
        raise RuntimeError("Telegram did not supply an attachment download path")

    get_encoded_url = getattr(tg_file, "_get_encoded_url", None)
    url = get_encoded_url() if callable(get_encoded_url) else file_path
    if not isinstance(url, str):
        raise RuntimeError("Telegram supplied an invalid attachment download URL")

    if urlparse(url).scheme in ("http", "https"):
        await _download_telegram_url(url, local_path, max_upload_bytes)
        return

    await asyncio.to_thread(
        copy_file_with_limit, file_path, local_path, max_upload_bytes,
    )


async def _download_telegram_url(
    url: str,
    local_path: str,
    max_upload_bytes: int,
) -> None:
    """Stream an HTTP(S) Telegram file without writing beyond the cap."""
    async with (
        aiohttp.ClientSession() as session,
        session.get(url) as response,
    ):
        response.raise_for_status()
        if upload_size_exceeds_limit(
            response.content_length, max_upload_bytes,
        ):
            raise UploadTooLargeError(max_upload_bytes)
        await write_limited_async_stream(
            response.content.iter_chunked(64 * 1024),
            local_path,
            max_upload_bytes,
        )
