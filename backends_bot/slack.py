"""Slack adapter: wires slack_bolt (socket mode) events to BotPlatform.

Slack's non-interactive flows differ from Telegram in several ways:
  - Slash commands must be ack()'d within 3s; heavy work runs after ack.
  - Events and commands originate from different APIs but both land here.
  - There is no native multi-step "ConversationHandler"; we rely on the
    base class's ``_pending_input`` state for follow-ups.
  - Rich AI replies use Slack's native Markdown blocks so headings, tables,
    task lists, links, and syntax-highlighted code render as intended.
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
from slack_sdk.errors import SlackApiError

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
    return render_fenced_markdown(
        text,
        render_line=_mrkdwn_line,
        render_code_block=_mrkdwn_code_block,
    )


def _mrkdwn_line(line: str) -> str:
    line = _escape_mrkdwn(line)
    # Bold first, into placeholders, so the single-asterisk italic
    # regex below can't mis-match the `*bold*` we're about to emit.
    # Headers -> bold (Slack has no heading syntax).
    line = re.sub(r"^#{1,6}\s+(.+)$", _bold_sub, line)
    line = re.sub(r"\*\*(.+?)\*\*", _bold_sub, line)
    line = re.sub(r"__(.+?)__", _bold_sub, line)
    # Italic: single `*text*` -> `_text_`; leave `_text_` as-is since
    # that's already valid mrkdwn.
    line = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"_\1_", line)
    # Strikethrough: `~~text~~` -> `~text~`.
    line = re.sub(r"~~(.+?)~~", r"~\1~", line)
    # Swap bold placeholders back to Slack's single-asterisk bold.
    return line.replace(_BOLD_OPEN, "*").replace(_BOLD_CLOSE, "*")


def _mrkdwn_code_block(lines: list[str]) -> list[str]:
    # Emit the accumulated block as-is, escaped for safety.
    return ["```", *(_escape_mrkdwn(line) for line in lines), "```"]


_SLACK_MAX_CHARS = 39_000  # Slack hard-caps around 40K; stay under.
_SLACK_MARKDOWN_LIMIT = 12_000  # Cumulative Markdown-block text per payload.
_FENCE_OPEN_RE = re.compile(r"^\s*(`{3,}|~{3,}).*$")


def _fence_open(line: str) -> tuple[str, str] | None:
    """Return the original opener and marker for a fenced code block."""
    match = _FENCE_OPEN_RE.match(line)
    if match is None:
        return None
    return line.strip(), match.group(1)


def _fence_closes(line: str, marker: str) -> bool:
    """Whether *line* is a valid close for the active fenced block."""
    character = re.escape(marker[0])
    return re.fullmatch(
        rf"\s*{character}{{{len(marker)},}}\s*", line,
    ) is not None


def _split_slack_markdown(
    text: str, limit: int = _SLACK_MARKDOWN_LIMIT,
) -> list[str]:
    """Split Markdown for Slack while balancing any fenced code blocks.

    Slack limits all Markdown-block text in one payload to 12,000
    characters. A long AI reply therefore becomes multiple Slack messages.
    When a split falls inside a fenced block, close the current block and
    reopen it (including its language hint) in the next message.
    """
    if len(text) <= limit:
        return [text]

    # A split inside a fence adds a prefix (the reopened opener + "\n") to the
    # continuation chunk and a suffix ("\n" + the closing marker) to the
    # preceding chunk. The prefix reopens the fence active at the END of the
    # previous chunk while the suffix closes the fence active at the END of the
    # current chunk - which can be a DIFFERENT fence. Reserve the independent
    # maxima so the worst-case prefix+suffix pairing still fits; a single
    # combined max (opener+marker of the same fence) can under-reserve when a
    # long-opener fence and a long-marker fence interleave across a boundary.
    max_opener = 0
    max_marker = 0
    for line in text.splitlines():
        fence = _fence_open(line)
        if fence is None:
            continue
        opener, marker = fence
        max_opener = max(max_opener, len(opener))
        max_marker = max(max_marker, len(marker))
    fence_wrap_reserve = (
        max_opener + max_marker + 2 if (max_opener or max_marker) else 0
    )
    if limit <= fence_wrap_reserve:
        # A single fence line is longer than a Slack Markdown block. We
        # cannot preserve that malformed/exceptional fence, but still send
        # the reply as bounded chunks rather than dropping it with an error.
        return split_text_chunks(text, limit)

    # Keep raw chunks deliberately below Slack's limit. That leaves enough
    # room to close/reopen even the longest fence in the response.
    raw_chunks = split_text_chunks(text, limit - fence_wrap_reserve)
    chunks: list[str] = []
    active_fence: tuple[str, str] | None = None

    for raw_chunk in raw_chunks:
        prefix = f"{active_fence[0]}\n" if active_fence else ""
        for line in raw_chunk.splitlines():
            if active_fence is not None:
                if _fence_closes(line, active_fence[1]):
                    active_fence = None
                continue
            active_fence = _fence_open(line)

        chunk = prefix + raw_chunk
        if active_fence:
            if not chunk.endswith("\n"):
                chunk += "\n"
            chunk += active_fence[1]
        chunks.append(chunk)

    return chunks


def _markdown_block_rejected(error: SlackApiError) -> bool:
    """Whether Slack rejected the new Markdown block type for this app."""
    response = getattr(error, "response", None)
    code = response.get("error") if response is not None else None
    return code in {"invalid_arguments", "invalid_blocks"}


async def _post_rich_markdown(client, chat_id: str, markdown: str) -> dict:
    """Post one native-Markdown payload, falling back to legacy mrkdwn."""
    try:
        return await client.chat_postMessage(
            channel=chat_id,
            text=markdown,
            blocks=[{"type": "markdown", "text": markdown}],
        )
    except SlackApiError as error:
        if not _markdown_block_rejected(error):
            raise
        logger.warning(
            "Slack Markdown blocks unavailable; falling back to mrkdwn: %s",
            error,
        )
        return await client.chat_postMessage(
            channel=chat_id, text=_md_to_mrkdwn(markdown),
        )


async def _update_rich_markdown(
    client, handle: MessageHandle, markdown: str,
) -> None:
    """Update a short rich message, with legacy mrkdwn fallback."""
    try:
        await client.chat_update(
            channel=handle.chat_id,
            ts=handle.message_id,
            text=markdown,
            blocks=[{"type": "markdown", "text": markdown}],
        )
    except SlackApiError as error:
        if not _markdown_block_rejected(error):
            raise
        logger.warning(
            "Slack Markdown blocks unavailable; falling back to mrkdwn: %s",
            error,
        )
        await client.chat_update(
            channel=handle.chat_id,
            ts=handle.message_id,
            text=_md_to_mrkdwn(markdown),
        )


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
        last: MessageHandle | None = None
        if rich:
            for chunk in _split_slack_markdown(text):
                if not chunk.strip():
                    continue
                resp = await _post_rich_markdown(client, chat_id, chunk)
                last = MessageHandle(
                    chat_id=str(chat_id), message_id=str(resp["ts"]),
                )
            return last

        for chunk in split_text_chunks(text, _SLACK_MAX_CHARS):
            if not chunk.strip():
                continue
            resp = await client.chat_postMessage(channel=chat_id, text=chunk)
            last = MessageHandle(
                chat_id=str(chat_id), message_id=str(resp["ts"]),
            )
        return last

    async def edit_text(
        self, handle: MessageHandle, text: str, *, rich: bool = False,
    ) -> None:
        assert self.app is not None
        if not rich:
            await self.app.client.chat_update(
                channel=handle.chat_id, ts=handle.message_id, text=text,
            )
            return

        chunks = _split_slack_markdown(text)
        if len(chunks) == 1:
            await _update_rich_markdown(self.app.client, handle, chunks[0])
            return

        # A message handle can only update one Slack message. Rich reply
        # chunks are posted separately, so preserve the old readable mrkdwn
        # behavior for an unusually large editable status message.
        await self.app.client.chat_update(
            channel=handle.chat_id,
            ts=handle.message_id,
            text=_md_to_mrkdwn(text),
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
        self.app.event("app_mention")(self._on_app_mention)

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

    async def _on_message(
        self, event, *, _from_app_mention: bool = False, **_kwargs,
    ) -> None:
        if event.get("subtype") not in self._ALLOWED_SUBTYPES:
            return
        uid = str(event.get("user") or "")
        channel = str(event.get("channel") or "")
        if (
            event.get("bot_id")
            or uid == self._bot_user_id
            or not uid
            or not channel
        ):
            return
        # Slack authorization is channel-scoped: only process events from
        # channels on the allowlist. This is checked before any download
        # work so unauthorized senders can't cause side effects.
        if not self.authorized(uid, channel):
            return

        text = (event.get("text") or "").strip()
        # If Slack delivers both ``message`` and ``app_mention`` for the
        # same post, let the latter own it.  It strips the bot marker before
        # dispatching, so the input is processed exactly once.
        marker = (
            f"<@{self._bot_user_id}>" if self._bot_user_id is not None else ""
        )
        if not _from_app_mention and marker and marker in text:
            return
        files = event.get("files") or []

        if files:
            await self._handle_files(event, files, text)
            return

        if not text:
            return

        await self.dispatch_text(self._ctx(uid, channel, text=text))

    async def _on_app_mention(self, event, **_kwargs) -> None:
        """Handle a direct Slack mention as ordinary input.

        Slack sends a distinct ``app_mention`` event instead of ``message``
        whenever the bot is tagged.  Remove this bot's markup so aliases
        such as ``@Cozter \\open`` reach the shared backslash-command parser.
        """
        text = str(event.get("text") or "")
        if self._bot_user_id is None:
            return
        marker = f"<@{self._bot_user_id}>"
        if marker not in text:
            return
        text = text.replace(marker, "").strip()
        await self._on_message(
            {**event, "text": text}, _from_app_mention=True,
        )

    async def _handle_files(
        self, event: dict, files: list[dict], caption: str,
    ) -> None:
        # Caller (_on_message) has already verified channel authorization.
        uid = str(event["user"])
        channel = str(event["channel"])
        ws = workspace.get_current(uid, self.platform_id)

        ctx_for_reply = self._ctx(uid, channel, text=caption)
        if not ws or not os.path.isdir(ws):
            await ctx_for_reply.reply_text(NO_WORKSPACE_TEXT)
            return

        upload_dir = ensure_upload_dir(ws)

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
    async with (
        aiohttp.ClientSession() as s,
        s.get(url, headers=headers) as resp,
    ):
        resp.raise_for_status()
        with open(local_path, "wb") as fp:
            async for chunk in resp.content.iter_chunked(64 * 1024):
                fp.write(chunk)
