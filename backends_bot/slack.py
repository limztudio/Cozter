"""Slack adapter: wires slack_bolt (socket mode) events to BotPlatform.

Slack's non-interactive flows differ from Telegram in several ways:
  - Slash commands must be ack()'d within 3s; heavy work runs after ack.
  - Events and commands originate from different APIs but both land here.
  - There is no native multi-step "ConversationHandler"; we rely on the
    base class's ``_pending_input`` state for follow-ups.
  - Rich text uses mrkdwn (`*bold*`, `_italic_`, `` `code` ``).
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse

import aiohttp

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import (
    AsyncSocketModeHandler,
)

from .. import workspace
from .base import (
    AttachmentInfo,
    BotContext,
    BotPlatform,
    COMMAND_NAMES,
    MessageHandle,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown -> Slack mrkdwn
# ---------------------------------------------------------------------------

def _escape_mrkdwn(text: str) -> str:
    """Slack mrkdwn uses HTML-style escaping for `<`, `>`, `&`."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


# Private Use Area placeholders that won't collide with user text or
# interfere with re.sub replacement-template parsing.
# (Defined just below; also paired with _bold_sub for bold-first rewriting.)
_BOLD_OPEN = ""
_BOLD_CLOSE = ""


def _bold_sub(m: re.Match) -> str:
    """Wrap group(1) in bold placeholders (module-level for perf)."""
    return _BOLD_OPEN + m.group(1) + _BOLD_CLOSE


def _md_to_mrkdwn(text: str) -> str:
    """Convert common Markdown to Slack-compatible mrkdwn.

    Differences from Telegram HTML rendering:
      - Bold is single ``*`` (not ``**``) in Slack.
      - Italic is single ``_`` (not ``*``).
      - Strikethrough is single ``~`` (not ``~~``).
      - Code blocks use triple backticks (same as input).
    """
    lines = text.split("\n")
    result: list[str] = []
    in_code_block = False
    code_buf: list[str] = []

    for line in lines:
        if line.strip().startswith("```"):
            if in_code_block:
                # Emit the accumulated block as-is, escaped for safety.
                result.append("```")
                result.extend(_escape_mrkdwn(x) for x in code_buf)
                result.append("```")
                code_buf.clear()
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_buf.append(line)
            continue

        line = _escape_mrkdwn(line)
        # Bold first, into placeholders, so the single-asterisk italic
        # regex below can't mis-match the `*bold*` we're about to emit.
        # Headers -> bold (Slack has no heading syntax).
        line = re.sub(r"^#{1,6}\s+(.+)$", _bold_sub, line)
        line = re.sub(r"\*\*(.+?)\*\*", _bold_sub, line)
        line = re.sub(r"__(.+?)__", _bold_sub, line)
        # Italic: single `*text*` → `_text_`; leave `_text_` as-is since
        # that's already valid mrkdwn.
        line = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"_\1_", line)
        # Strikethrough: `~~text~~` → `~text~`.
        line = re.sub(r"~~(.+?)~~", r"~\1~", line)
        # Swap bold placeholders back to Slack's single-asterisk bold.
        line = line.replace(_BOLD_OPEN, "*").replace(_BOLD_CLOSE, "*")
        # Inline code stays as `text`.
        result.append(line)

    if in_code_block and code_buf:
        result.append("```")
        result.extend(_escape_mrkdwn(x) for x in code_buf)
        result.append("```")

    return "\n".join(result)


_SLACK_MAX_CHARS = 39_000  # Slack hard-caps around 40K; stay under.


def _split_mrkdwn(text: str, limit: int = _SLACK_MAX_CHARS) -> list[str]:
    """Split mrkdwn text at newline boundaries so no tag is cut mid-way."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            chunks.append(text[:limit])
            text = text[limit:]
        else:
            chunks.append(text[:split_at])
            text = text[split_at + 1:]
    return chunks


# ---------------------------------------------------------------------------
# Slack platform adapter
# ---------------------------------------------------------------------------

class SlackBot(BotPlatform):
    """Slack Socket-Mode adapter."""

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        channel_ids: list[str],
        *,
        recent_limit: int = 10,
        max_queue_size: int = 50,
    ):
        # channel_ids IS the authorization set for Slack: the bot listens
        # only in these channels (public C..., private G..., DMs D..., or
        # multi-party DMs MP...). Stored in ``notify_targets`` on base.
        super().__init__(
            channel_ids,
            recent_limit=recent_limit,
            max_queue_size=max_queue_size,
        )
        self.bot_token = bot_token
        self.app_token = app_token
        self.app: AsyncApp | None = None
        self._handler: AsyncSocketModeHandler | None = None
        self._bot_user_id: str | None = None

    @property
    def platform_id(self) -> str:
        # Prefix with "slack:" so Slack state never collides with a
        # Telegram bot id that happens to be numerically identical.
        if self._bot_user_id is None:
            raise RuntimeError("platform_id is only valid after start()")
        return f"slack:{self._bot_user_id}"

    def authorized(self, user_id: str, chat_id: str) -> bool:
        """Slack authorization is channel-scoped: allow if *chat_id* is listed."""
        return str(chat_id) in self.notify_targets

    # ----- send/edit primitives ------------------------------------------

    async def send_text(
        self, chat_id: str, text: str, *, rich: bool = False,
    ) -> MessageHandle | None:
        if not text:
            return None
        assert self.app is not None
        client = self.app.client
        body = _md_to_mrkdwn(text) if rich else text
        last: MessageHandle | None = None
        for chunk in _split_mrkdwn(body):
            if not chunk.strip():
                continue
            resp = await client.chat_postMessage(channel=chat_id, text=chunk)
            last = MessageHandle(
                chat_id=str(chat_id), message_id=str(resp["ts"]),
            )
        return last

    async def edit_text(self, handle: MessageHandle, text: str) -> None:
        assert self.app is not None
        await self.app.client.chat_update(
            channel=handle.chat_id, ts=handle.message_id, text=text,
        )

    async def delete_message(self, handle: MessageHandle) -> None:
        assert self.app is not None
        await self.app.client.chat_delete(
            channel=handle.chat_id, ts=handle.message_id,
        )

    async def send_file(self, chat_id: str, path: str) -> None:
        assert self.app is not None
        name = os.path.basename(path)
        await self.app.client.files_upload_v2(
            channel=chat_id, file=path, filename=name,
        )

    # ----- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        self.app = AsyncApp(token=self.bot_token)

        auth = await self.app.client.auth_test()
        self._bot_user_id = auth["user_id"]

        for name in COMMAND_NAMES:
            self.app.command(f"/{name}")(self._make_command_handler(name))

        self.app.event("message")(self._on_message)

        self._handler = AsyncSocketModeHandler(self.app, self.app_token)
        # Restore in-flight / queued messages before connecting so new
        # user events can't race past the restored backlog. app.client
        # is already ready (auth_test succeeded above), so drain can
        # still post messages via chat_postMessage during restore.
        await self.restore_queues()
        await self.start_scheduler()
        await self._handler.connect_async()
        logger.info(
            "Slack bot started in Socket Mode (bot_user=%s).",
            self._bot_user_id,
        )

    async def stop(self) -> None:
        await self.stop_scheduler()
        if self._handler is not None:
            try:
                await self._handler.close_async()
            except Exception:
                logger.exception("Error during Slack socket close")
        self._handler = None
        self.app = None
        logger.info("Slack bot stopped.")

    # ----- event handlers -------------------------------------------------

    # Allow only plain user messages and file uploads; every other subtype
    # (system messages, edits, pins, joins, bot-to-bot chatter, ...) is
    # discarded so it can't be interpreted as user input.
    _ALLOWED_SUBTYPES = frozenset({None, "file_share"})

    def _ctx(
        self,
        uid: str,
        channel: str,
        *,
        text: str = "",
        command: str | None = None,
        args: str = "",
        attachment: AttachmentInfo | None = None,
    ) -> BotContext:
        return BotContext(
            user_id=uid, chat_id=channel, text=text,
            command=command, args=args, attachment=attachment, platform=self,
        )

    def _make_command_handler(self, name: str):
        async def handler(ack, command, **_kwargs) -> None:
            await ack()
            uid = str(command.get("user_id") or "")
            channel = str(command.get("channel_id") or "")
            if not self.authorized(uid, channel):
                logger.warning(
                    "Unauthorized Slack command /%s in channel=%s from user=%s",
                    name, channel, uid,
                )
                return
            args = (command.get("text") or "").strip()
            ctx = self._ctx(uid, channel, command=name, args=args)
            try:
                await self.dispatch_command(ctx)
            except Exception:
                logger.exception("Slack command /%s failed", name)
        return handler

    async def _on_message(self, event, **_kwargs) -> None:
        if event.get("subtype") not in self._ALLOWED_SUBTYPES:
            return
        uid = str(event.get("user") or "")
        channel = str(event.get("channel") or "")
        if event.get("bot_id") or not uid or not channel:
            return
        # Slack authorization is channel-scoped: only process events from
        # channels on the allowlist. This is checked before any download
        # work so unauthorized senders can't cause side effects.
        if not self.authorized(uid, channel):
            return

        text = (event.get("text") or "").strip()
        files = event.get("files") or []

        if files:
            await self._handle_files(event, files, text)
            return

        if not text:
            return

        await self.dispatch_text(self._ctx(uid, channel, text=text))

    async def _handle_files(
        self, event: dict, files: list[dict], caption: str,
    ) -> None:
        # Caller (_on_message) has already verified channel authorization.
        uid = str(event["user"])
        channel = str(event["channel"])
        ws = workspace.get_current(uid, self.platform_id)

        ctx_for_reply = self._ctx(uid, channel, text=caption)
        if not ws or not os.path.isdir(ws):
            await ctx_for_reply.reply_text(
                "No workspace selected (or it was deleted)."
                " Use /new or /open."
            )
            return

        upload_dir = os.path.join(ws, ".cozter", "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        for f in files:
            url = f.get("url_private_download") or f.get("url_private")
            if not url:
                continue
            # Use the user-supplied name if meaningful; otherwise fall back
            # to the Slack file id. basename() guards against path chars
            # that would otherwise escape upload_dir.
            filename = os.path.basename(f.get("name") or "")
            if not filename:
                filename = f.get("id") or "file"
            kind = _slack_file_kind(f)
            local_path = os.path.join(upload_dir, filename)
            try:
                await _download_private(url, self.bot_token, local_path)
            except Exception as e:
                await ctx_for_reply.reply_text(
                    f"Failed to download {filename}: {e}"
                )
                continue

            await self.dispatch_file(self._ctx(
                uid, channel,
                attachment=AttachmentInfo(
                    local_path=local_path,
                    filename=filename,
                    kind=kind,
                    caption=caption,
                ),
            ))


def _slack_file_kind(f: dict) -> str:
    """Map a Slack file payload to one of our coarse kind labels."""
    mime = (f.get("mimetype") or "").lower()
    if mime.startswith("image/"):
        return "photo"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"
    return "document"


async def _download_private(
    url: str, bot_token: str, local_path: str,
) -> None:
    """Download a `url_private` file using the bot token."""
    # Slack private URLs require Authorization: Bearer <bot-token>.
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"Refusing to fetch non-http url: {url!r}")
    headers = {"Authorization": f"Bearer {bot_token}"}
    async with aiohttp.ClientSession() as s:
        async with s.get(url, headers=headers) as resp:
            resp.raise_for_status()
            with open(local_path, "wb") as fp:
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    fp.write(chunk)
