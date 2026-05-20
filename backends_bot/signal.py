"""Signal adapter: wires signal-cli group messages to BotPlatform.

This backend talks to Signal through the installed ``signal-cli`` binary.
It resolves configured group invite URLs to group ids on startup, then
polls ``signal-cli receive -o json`` for messages from those groups.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
from typing import Any

from .. import workspace
from .base import AttachmentInfo, BotContext, BotPlatform, MessageHandle

logger = logging.getLogger(__name__)

_DEFAULT_RECEIVE_TIMEOUT = 30
_ERROR_RETRY_DELAY = 10
_SIGNAL_TEXT_LIMIT = 4_000


class SignalCliError(RuntimeError):
    """Raised when signal-cli exits unsuccessfully."""


class SignalBot(BotPlatform):
    """Signal group adapter backed by signal-cli."""

    def __init__(
        self,
        phone_number: str,
        group_urls: list[str],
        *,
        recent_limit: int = 10,
        max_queue_size: int = 50,
        signal_cli_path: str = "signal-cli",
        receive_timeout: int = _DEFAULT_RECEIVE_TIMEOUT,
    ):
        super().__init__(
            group_urls,
            recent_limit=recent_limit,
            max_queue_size=max_queue_size,
        )
        self.phone_number = phone_number
        self.group_urls = _dedupe_group_urls(group_urls)
        self.signal_cli_path = signal_cli_path
        self.receive_timeout = receive_timeout
        self._group_ids_by_url: dict[str, str] = {}
        self._group_ids: set[str] = set()
        self._receive_task: asyncio.Task | None = None
        self._stop_requested = asyncio.Event()

    @property
    def platform_id(self) -> str:
        return f"signal:{self.phone_number}"

    def authorized(self, user_id: str, chat_id: str) -> bool:
        return str(chat_id) in self._group_ids

    # ----- send/edit primitives ------------------------------------------

    async def send_text(
        self, chat_id: str, text: str, *, rich: bool = False,
    ) -> MessageHandle | None:
        if not text:
            return None
        group_id = self._group_id_for_chat(chat_id)
        last: MessageHandle | None = None
        for chunk in _split_text(text):
            raw = await self._run_signal_cli(
                "send", "-g", group_id, "--message-from-stdin",
                input_text=chunk,
            )
            timestamp = _extract_timestamp(raw)
            if timestamp:
                last = MessageHandle(
                    chat_id=group_id,
                    message_id=timestamp,
                )
        return last

    async def edit_text(self, handle: MessageHandle, text: str) -> None:
        group_id = self._group_id_for_chat(handle.chat_id)
        await self._run_signal_cli(
            "send", "-g", group_id,
            "--edit-timestamp", handle.message_id,
            "--message-from-stdin",
            input_text=text,
        )

    async def delete_message(self, handle: MessageHandle) -> None:
        group_id = self._group_id_for_chat(handle.chat_id)
        await self._run_signal_cli(
            "remoteDelete", "-g", group_id,
            "-t", handle.message_id,
        )

    async def send_file(self, chat_id: str, path: str) -> None:
        group_id = self._group_id_for_chat(chat_id)
        await self._run_signal_cli(
            "send", "-g", group_id, "-a", path,
        )

    async def send_status(self, chat_id: str, text: str) -> None:
        # Signal has no cheap, reliable transient status surface if
        # message timestamps are unavailable, so avoid spamming the group
        # with every tool event. The final reply still arrives normally.
        return None

    # ----- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        if shutil.which(self.signal_cli_path) is None:
            raise RuntimeError(
                f"signal-cli executable not found: {self.signal_cli_path}"
            )
        self._group_ids_by_url = await self._resolve_group_ids()
        self._group_ids = set(self._group_ids_by_url.values())
        self.notify_targets = list(
            dict.fromkeys(self._group_ids_by_url.values())
        )
        self._stop_requested.clear()
        await self.restore_queues()
        await self.start_scheduler()
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info(
            "Signal bot started for %d group URL(s).",
            len(self._group_ids),
        )

    async def stop(self) -> None:
        await self.stop_scheduler()
        self._stop_requested.set()
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        self._receive_task = None
        logger.info("Signal bot stopped.")

    async def send_startup_messages(
        self, version: str, commit_date: str,
    ) -> None:
        msg = (
            f"Cozter started.\nVersion: {version}\nUpdated: {commit_date}"
        )
        await self.notify_users(msg)

    # ----- receive loop ---------------------------------------------------

    async def _receive_loop(self) -> None:
        while not self._stop_requested.is_set():
            try:
                raw = await self._run_signal_cli(
                    "receive",
                    "-t", str(self.receive_timeout),
                    "--ignore-stories",
                    timeout=self.receive_timeout + 30,
                )
                for item in _parse_json_items(raw):
                    await self._handle_received_item(item)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Signal receive loop failed")
                try:
                    await asyncio.wait_for(
                        self._stop_requested.wait(),
                        timeout=_ERROR_RETRY_DELAY,
                    )
                except asyncio.TimeoutError:
                    pass

    async def _handle_received_item(self, item: dict[str, Any]) -> None:
        envelope = item.get("envelope") if isinstance(item, dict) else None
        if not isinstance(envelope, dict):
            envelope = item
        data = envelope.get("dataMessage")
        if not isinstance(data, dict):
            return

        group_id = _extract_message_group_id(data)
        if group_id not in self._group_ids:
            return

        uid = _extract_sender_id(envelope)
        if not uid or uid == self.phone_number:
            return

        text = str(data.get("message") or data.get("body") or "").strip()
        attachments = data.get("attachments") or []

        if text.startswith("/"):
            await self.dispatch_command(
                self._ctx(uid, group_id, text=text),
            )
            return

        if isinstance(attachments, list) and attachments:
            await self._handle_attachments(uid, group_id, text, attachments)
            return

        if text:
            await self.dispatch_text(self._ctx(uid, group_id, text=text))

    async def _handle_attachments(
        self, uid: str, group_id: str, caption: str, attachments: list[Any],
    ) -> None:
        ctx_for_reply = self._ctx(uid, group_id, text=caption)
        ws = workspace.get_current(uid, self.platform_id)
        if not ws or not os.path.isdir(ws):
            await ctx_for_reply.reply_text(
                "No workspace selected (or it was deleted)."
                " Use /new or /open."
            )
            return

        upload_dir = os.path.join(ws, ".cozter", "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        for att in attachments:
            if not isinstance(att, dict):
                continue
            try:
                info = await self._materialize_attachment(
                    att, group_id, upload_dir, caption,
                )
            except Exception as e:
                logger.warning("Failed to import Signal attachment: %s", e)
                await ctx_for_reply.reply_text(
                    f"Failed to download attachment: {e}"
                )
                continue
            if info is None:
                continue
            await self.dispatch_file(
                self._ctx(uid, group_id, attachment=info),
            )

    async def _materialize_attachment(
        self,
        att: dict[str, Any],
        group_id: str,
        upload_dir: str,
        caption: str,
    ) -> AttachmentInfo | None:
        filename = _attachment_filename(att)
        local_path = os.path.join(upload_dir, filename)
        source_path = _attachment_local_path(att)

        if source_path and os.path.isfile(source_path):
            shutil.copyfile(source_path, local_path)
        else:
            attachment_id = _attachment_id(att)
            if not attachment_id:
                return None
            raw = await self._run_signal_cli(
                "getAttachment", "--id", attachment_id, "-g", group_id,
                output_json=False,
            )
            payload = re.sub(r"\s+", "", raw)
            with open(local_path, "wb") as f:
                f.write(base64.b64decode(payload))

        return AttachmentInfo(
            local_path=local_path,
            filename=filename,
            kind=_attachment_kind(att),
            caption=caption,
        )

    def _ctx(
        self,
        uid: str,
        group_id: str,
        *,
        text: str = "",
        attachment: AttachmentInfo | None = None,
    ) -> BotContext:
        command: str | None = None
        args = ""
        if text.startswith("/"):
            parts = text[1:].split(None, 1)
            command = parts[0].split("@", 1)[0].lower() if parts else ""
            args = parts[1] if len(parts) > 1 else ""
            text = ""
        return BotContext(
            user_id=uid,
            chat_id=group_id,
            text=text,
            command=command,
            args=args,
            attachment=attachment,
            platform=self,
        )

    # ----- signal-cli helpers --------------------------------------------

    async def _resolve_group_ids(self) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for group_url in self.group_urls:
            resolved[group_url] = await self._resolve_group_id(group_url)
        return resolved

    async def _resolve_group_id(self, group_url: str) -> str:
        group_id = await self._find_group_id_by_url(group_url)
        if group_id:
            return group_id

        try:
            joined_raw = await self._run_signal_cli(
                "joinGroup", "--uri", group_url,
            )
            group_id = _first_group_id(joined_raw)
            if group_id:
                return group_id
        except SignalCliError as e:
            if "already" not in str(e).casefold():
                raise

        group_id = await self._find_group_id_by_url(group_url)
        if group_id:
            return group_id

        raise RuntimeError(
            "Signal group URL could not be resolved after joinGroup: "
            f"{group_url!r}"
        )

    async def _find_group_id_by_url(self, group_url: str) -> str | None:
        raw = await self._run_signal_cli("listGroups")
        wanted = _normalize_group_url(group_url)
        for group in _extract_groups(raw):
            group_id = _group_id(group)
            if not group_id:
                continue
            for url in _group_invite_urls(group):
                if _normalize_group_url(url) == wanted:
                    return group_id
        return None

    async def _run_signal_cli(
        self,
        *args: str | None,
        input_text: str | None = None,
        timeout: int | None = None,
        output_json: bool = True,
    ) -> str:
        argv = [self.signal_cli_path, "-a", self.phone_number]
        if output_json:
            argv.extend(["-o", "json"])
        argv.extend(str(arg) for arg in args if arg is not None)

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if input_text is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(
                    input_text.encode("utf-8") if input_text is not None
                    else None
                ),
                timeout=timeout,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            proc.kill()
            try:
                await proc.wait()
            finally:
                pass
            raise

        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            message = err.strip() or out.strip() or f"exit {proc.returncode}"
            raise SignalCliError(message)
        if err.strip():
            logger.debug("signal-cli stderr: %s", err.strip())
        return out

    def _group_id_for_chat(self, chat_id: str) -> str:
        group_id = str(chat_id)
        if group_id not in self._group_ids:
            raise RuntimeError(f"Signal group is not configured: {chat_id}")
        return group_id


def _split_text(text: str, limit: int = _SIGNAL_TEXT_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


def _dedupe_group_urls(group_urls: list[str]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for value in group_urls:
        url = value.strip()
        if not url:
            continue
        normalized = _normalize_group_url(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        urls.append(url)
    return urls


def _parse_json_items(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        items: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Ignoring non-JSON signal-cli line: %s", line)
                continue
            items.extend(_coerce_json_items(value))
        return items
    return _coerce_json_items(data)


def _coerce_json_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, dict):
        for key in ("envelopes", "messages", "results", "groups"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [x for x in nested if isinstance(x, dict)]
        return [value]
    return []


def _extract_groups(raw: str) -> list[dict[str, Any]]:
    return _parse_json_items(raw)


def _first_group_id(raw: str) -> str:
    for item in _parse_json_items(raw):
        group_id = _group_id(item)
        if group_id:
            return group_id
    return ""


def _group_id(group: dict[str, Any]) -> str:
    for key in ("id", "groupId", "groupID"):
        value = group.get(key)
        normalized = _normalize_group_id(value)
        if normalized:
            return normalized
    return ""


def _group_invite_urls(group: dict[str, Any]) -> list[str]:
    urls: list[str] = []

    def visit(value: Any, key_hint: str = "") -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(nested, key)
            return
        if isinstance(value, list):
            for nested in value:
                visit(nested, key_hint)
            return
        if not isinstance(value, str):
            return

        lowered_key = key_hint.casefold()
        text = value.strip()
        if (
            text.startswith("https://signal.group/")
            or text.startswith("sgnl://")
            or ("invite" in lowered_key and "signal.group" in text)
        ):
            urls.append(text)

    visit(group)
    return urls


def _normalize_group_url(value: str) -> str:
    return value.strip().rstrip("/")


def _extract_message_group_id(data: dict[str, Any]) -> str:
    for key in ("groupInfo", "groupV2", "group"):
        info = data.get(key)
        if isinstance(info, dict):
            group_id = _group_id(info)
            if group_id:
                return group_id
    return ""


def _normalize_group_id(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and all(isinstance(x, int) for x in value):
        try:
            return base64.b64encode(bytes(value)).decode("ascii")
        except ValueError:
            return ""
    return ""


def _extract_sender_id(envelope: dict[str, Any]) -> str:
    for key in ("sourceNumber", "source", "sourceUuid", "sourceName"):
        value = envelope.get(key)
        if value and not isinstance(value, (dict, list)):
            return str(value)
    source = envelope.get("sourceAddress")
    if isinstance(source, dict):
        for key in ("number", "uuid", "name"):
            value = source.get(key)
            if value:
                return str(value)
    return ""


def _extract_timestamp(raw: str) -> str | None:
    items = _parse_json_items(raw)
    for item in items:
        timestamp = _find_key(item, "timestamp")
        if timestamp is not None:
            return str(timestamp)
    match = re.search(r"\b(\d{12,})\b", raw)
    return match.group(1) if match else None


def _find_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for nested in value.values():
            found = _find_key(nested, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_key(nested, key)
            if found is not None:
                return found
    return None


def _attachment_filename(att: dict[str, Any]) -> str:
    for key in ("fileName", "filename", "name"):
        value = att.get(key)
        if isinstance(value, str) and value.strip():
            return os.path.basename(value.strip())
    attachment_id = _attachment_id(att) or "attachment"
    ext = _extension_for_content_type(att.get("contentType"))
    return f"signal_{attachment_id}{ext}"


def _attachment_id(att: dict[str, Any]) -> str:
    for key in ("id", "attachmentId", "attachmentPointerId"):
        value = att.get(key)
        if value:
            return str(value)
    return ""


def _attachment_local_path(att: dict[str, Any]) -> str:
    for key in ("path", "localPath", "storedFilename"):
        value = att.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _attachment_kind(att: dict[str, Any]) -> str:
    content_type = str(att.get("contentType") or "").lower()
    if content_type.startswith("image/"):
        return "photo"
    if content_type.startswith("audio/"):
        return "audio"
    if content_type.startswith("video/"):
        return "video"
    return "document"


def _extension_for_content_type(value: Any) -> str:
    content_type = str(value or "").lower()
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "video/mp4": ".mp4",
        "text/plain": ".txt",
        "application/pdf": ".pdf",
    }
    return mapping.get(content_type, "")
