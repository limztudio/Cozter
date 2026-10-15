"""Signal adapter: wires signal-cli group messages to BotPlatform.

This backend talks to Signal through a shared signal-cli daemon JSON-RPC
Unix or TCP socket. The daemon lifecycle is owned outside Cozter.
"""

from __future__ import annotations

import asyncio
from bisect import bisect_left, bisect_right
import base64
import contextlib
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import deque
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

from ..config import (
    DEFAULT_MAX_UPLOAD_BYTES,
    DEFAULT_MESSAGE_QUEUE_SIZE,
    DEFAULT_RECENT_WORKSPACE_LIMIT,
)
from .. import schedules, session, workspace
from ..utils import create_background_task, save_json_object, split_text_chunks, text_chunk_ranges
from .base import (
    AttachmentInfo,
    BotContext,
    BotPlatform,
    _run_durable_file_operation,
    MessageHandle,
    UploadTooLargeError,
    attachment_kind_from_mime,
    copy_file_with_limit_async,
    reserve_upload_path,
    upload_limit_message,
    upload_size_exceeds_limit,
    write_bytes_atomically,
)
from .formatting import iter_fenced_markdown

logger = logging.getLogger(__name__)
performance_logger = logging.getLogger("Cozter.performance")

_SIGNAL_TEXT_LIMIT = 4_000
_OUTGOING_ECHO_TTL = 7 * 24 * 60 * 60
_OUTGOING_ECHO_LIMIT = 8192
_OUTGOING_JOURNAL_MAX_BYTES = 4 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_QUEUE_ACK_RE = re.compile(r"Queued \([0-9]+/[0-9]+\)\.")
_DIRECTORY_REPLY_PREFIX = "Directory does not exist:\n"
_DIRECTORY_REPLY_SUFFIXES = (
    "\n\nPlease enter a valid directory (or /cancel):",
    "\n\nUse /open to choose from recent workspaces.",
)
_INCOMING_DEDUPE_TTL = 120.0
_INCOMING_DEDUPE_LIMIT = 500
_LEGACY_SIGNAL_PLATFORM_PREFIX = "signal:"
_JSONRPC_STREAM_LIMIT = 128 * 1024 * 1024
_SIGNAL_STYLE_BOLD = "BOLD"
_SIGNAL_STYLE_ITALIC = "ITALIC"
_SIGNAL_STYLE_MONOSPACE = "MONOSPACE"
_SIGNAL_STYLE_STRIKETHROUGH = "STRIKETHROUGH"
_SIGNAL_STYLE_SPOILER = "SPOILER"
_SIGNAL_SEND_MAX_ATTEMPTS = 4
_SIGNAL_SEND_BASE_DELAY_SEC = 1.0
_SIGNAL_SEND_MAX_DELAY_SEC = 30.0
_SIGNAL_SEND_MIN_INTERVAL_SEC = 1.2
_SIGNAL_RETRY_AFTER_RE = re.compile(
    r"retry after (\d+(?:\.\d+)?)\s*seconds?", re.IGNORECASE,
)
_SIGNAL_GROUP_URL_RE = re.compile(r"https://signal\.group/#[^\s]+")
_SIGNAL_GROUP_SCHEME_RE = re.compile(r"sgnl://[^\s]+")
_SIGNAL_ATTACH_WHITESPACE_RE = re.compile(r"\s+")
_SIGNAL_ATTACH_ASCII_WHITESPACE = tuple(
    chr(code) for code in range(128) if chr(code).isspace()
)
_SIGNAL_ATTACH_ASCII_WHITESPACE_DELETE = {
    ord(character): None for character in _SIGNAL_ATTACH_ASCII_WHITESPACE
}
_SIGNAL_MD_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$")
_SIGNAL_WORD_CHAR_RE = re.compile(r"\w")
_SIGNAL_INLINE_START_RE = re.compile(r"[`*_~|]")


def _signal_rate_limit_delay(exc: BaseException) -> float | None:
    """Return the server-requested wait when signal-cli reports a throttle.

    Matches ``RetryLaterException: Retry after N seconds`` (and any other
    ``Retry after N seconds`` wording). Returns None when the error is not
    a rate-limit so callers only sleep-and-retry on real throttles.
    """
    match = _SIGNAL_RETRY_AFTER_RE.search(str(exc))
    if match is None:
        return None
    try:
        delay = float(match.group(1))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(delay) or delay < 0:
        return None
    return min(delay, _SIGNAL_SEND_MAX_DELAY_SEC)

# Send-methods aren't retried: a duplicate is worse than a lost reply.
_NON_IDEMPOTENT_RPC_METHODS = frozenset({"send"})


class SignalCliError(RuntimeError):
    """Raised when the signal-cli JSON-RPC socket is unavailable."""


def _echo_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_generated_control_message(text: str) -> bool:
    """Reject exact bot protocol replies before an input-state handler sees them."""
    while True:
        if _QUEUE_ACK_RE.fullmatch(text):
            return True
        if not text.startswith(_DIRECTORY_REPLY_PREFIX):
            return False
        for suffix in _DIRECTORY_REPLY_SUFFIXES:
            if text.endswith(suffix):
                text = text[len(_DIRECTORY_REPLY_PREFIX):-len(suffix)].strip()
                break
        else:
            return False


class _SignalOutboundJournal:
    """Bounded sent-message provenance; never retain plaintext chat content."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._loaded = False
        self._records: dict[tuple[str, str, str], float] = {}

    @staticmethod
    def _path() -> str:
        return os.path.join(workspace.CONFIG_DIR, "signal_outbound.json")

    def _read(self) -> dict[tuple[str, str, str], float]:
        try:
            with open(self._path(), "rb") as source:
                raw = source.read(_OUTGOING_JOURNAL_MAX_BYTES + 1)
        except FileNotFoundError:
            return {}
        if len(raw) > _OUTGOING_JOURNAL_MAX_BYTES:
            raise SignalCliError("Signal outbound journal exceeds its size limit")
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise SignalCliError("Signal outbound journal is unreadable") from exc
        if (not isinstance(payload, dict) or payload.get("version") != 1
                or not isinstance(payload.get("records"), list)):
            raise SignalCliError("Invalid Signal outbound journal format")
        records: dict[tuple[str, str, str], float] = {}
        now = time.time()
        for row in payload["records"]:
            if not isinstance(row, dict):
                continue
            group, kind, value, created = (
                row.get("group"), row.get("kind"), row.get("value"), row.get("created"),
            )
            if (not isinstance(group, str) or not _SHA256_RE.fullmatch(group)
                    or not isinstance(kind, str) or kind not in ("text", "timestamp")
                    or not isinstance(value, str)
                    or not isinstance(created, (int, float)) or isinstance(created, bool)
                    or not now - _OUTGOING_ECHO_TTL <= created <= now + 300):
                continue
            if ((kind == "text" and _SHA256_RE.fullmatch(value))
                    or (kind == "timestamp" and value.isascii() and value.isdigit()
                        and len(value) <= 20)):
                key = (group, kind, value)
                records[key] = max(records.get(key, 0), float(created))
        return self._bounded(records, now)

    @staticmethod
    def _bounded(
        records: dict[tuple[str, str, str], float], now: float,
    ) -> dict[tuple[str, str, str], float]:
        fresh = ((key, created) for key, created in records.items()
                 if now - _OUTGOING_ECHO_TTL <= created <= now + 300)
        return dict(sorted(fresh, key=lambda item: item[1])[-_OUTGOING_ECHO_LIMIT:])

    def _adopt(self, records: dict[tuple[str, str, str], float]) -> None:
        self._records = records
        self._loaded = True

    async def load(self) -> None:
        if self._loaded:
            return
        await _run_durable_file_operation(
            self._lock,
            lambda: self._records if self._loaded else self._read(),
            on_complete=self._adopt,
        )

    async def remember(self, entries: Iterable[tuple[str, str, str]]) -> None:
        keys = [(_echo_hash(group), kind, value) for group, kind, value in entries]
        if not keys:
            return

        def write() -> dict[tuple[str, str, str], float]:
            now = time.time()
            records = dict(self._records) if self._loaded else self._read()
            records.update((key, now) for key in keys)
            records = self._bounded(records, now)
            save_json_object(self._path(), {
                "version": 1,
                "records": [{"group": group, "kind": kind, "value": value,
                             "created": created}
                            for (group, kind, value), created in records.items()],
            })
            return records

        await _run_durable_file_operation(self._lock, write, on_complete=self._adopt)

    def contains(self, group_id: str, *, text: str, timestamp: str | None) -> bool:
        group = _echo_hash(group_id)
        cutoff = time.time() - _OUTGOING_ECHO_TTL
        return any(created is not None and created >= cutoff
                   for kind, value in (("text", _echo_hash(text)),
                                       ("timestamp", timestamp or ""))
                   if value and (kind != "text" or text)
                   for created in (self._records.get((group, kind, value)),))


async def seed_signal_outbound_echoes(messages: Iterable[tuple[str, str]]) -> None:
    """Recover proven bot output from a quarantined queue without storing its text."""
    journal = _SignalOutboundJournal()
    await journal.remember((group, "text", _echo_hash(text.strip()))
                           for group, text in messages if text.strip())


def _signal_tcp_address(endpoint: str) -> tuple[str, int] | None:
    """Validate TCP URLs while leaving existing Unix socket paths intact."""
    if not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", endpoint):
        if endpoint.lower().startswith("tcp:"):
            raise ValueError(
                "signal_jsonrpc_socket must use tcp://HOST:PORT for TCP"
            )
        return None
    try:
        parsed = urlsplit(endpoint)
        if parsed.scheme != "tcp":
            raise ValueError("unsupported protocol")
        host = parsed.hostname
        port = parsed.port
        if (
            not host
            or port is None
            or not 1 <= port <= 65535
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or "?" in endpoint
            or "#" in endpoint
            or any(char.isspace() for char in endpoint)
            or not endpoint.isprintable()
            or "\\" in host
            or "%" in host
            or (
                parsed.netloc.startswith("[")
                and not re.fullmatch(r"\[[^\]]+\]:[0-9]+", parsed.netloc)
            )
        ):
            raise ValueError("invalid TCP address")
    except ValueError as exc:
        raise ValueError(
            "signal_jsonrpc_socket must be a Unix socket path or "
            "tcp://HOST:PORT (port 1-65535, without credentials, path, "
            "query, or fragment)"
        ) from exc
    return host, port


class SignalBot(BotPlatform):
    """Signal group adapter backed by signal-cli."""

    status_update_interval_sec = 60.0
    status_update_limit = 10
    status_fallback_messages = False
    status_initial_timeout_sec = 30.0
    status_recovery_interval_sec = 300.0

    def __init__(
        self,
        group_urls: list[str],
        *,
        recent_limit: int = DEFAULT_RECENT_WORKSPACE_LIMIT,
        max_queue_size: int = DEFAULT_MESSAGE_QUEUE_SIZE,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
        jsonrpc_socket: str = "",
        own_device_id: int | None = None,
    ):
        super().__init__(
            group_urls,
            recent_limit=recent_limit,
            max_queue_size=max_queue_size,
            max_upload_bytes=max_upload_bytes,
        )
        self.group_urls = _dedupe_group_urls(group_urls)
        self.jsonrpc_socket = jsonrpc_socket.strip() if jsonrpc_socket else ""
        if own_device_id is not None and (
            not isinstance(own_device_id, int) or isinstance(own_device_id, bool)
            or own_device_id < 0
        ):
            raise ValueError("signal_own_device_id must be a non-negative integer or null")
        self.own_device_id = own_device_id or None
        if not self.jsonrpc_socket:
            raise ValueError("signal_jsonrpc_socket is required for Signal")
        self._jsonrpc_tcp_address = _signal_tcp_address(self.jsonrpc_socket)
        self._group_ids_by_url: dict[str, str] = {}
        self._group_ids: set[str] = set()
        self._jsonrpc_reader: asyncio.StreamReader | None = None
        self._jsonrpc_writer: asyncio.StreamWriter | None = None
        self._jsonrpc_reader_task: asyncio.Task | None = None
        self._jsonrpc_write_lock = asyncio.Lock()
        self._jsonrpc_reconnect_lock = asyncio.Lock()
        self._jsonrpc_reconnect_task: asyncio.Task | None = None
        self._jsonrpc_pending: dict[str, asyncio.Future] = {}
        self._jsonrpc_next_id = 0
        self._receive_subscription: int | None = None
        self._receive_started = False
        self._receive_subscribed = False
        self._notification_tasks: set[asyncio.Task] = set()
        self._stop_requested = asyncio.Event()
        self._own_sent_timestamps: set[tuple[str, str]] = set()
        self._own_sent_timestamp_order: deque[tuple[str, str]] = deque()
        self._outbound_journal = _SignalOutboundJournal()
        self._outbound_journal_tasks: set[asyncio.Task] = set()
        self._recent_incoming_keys: set[str] = set()
        self._recent_incoming_key_order: deque[tuple[float, str]] = deque()
        self._send_lock = asyncio.Lock()
        self._last_send_monotonic = 0.0

    @property
    def platform_id(self) -> str:
        return "signal"

    def authorized(self, user_id: str, chat_id: str) -> bool:
        return str(chat_id) in self._group_ids

    async def send_text(
        self, chat_id: str, text: str, *, rich: bool = False,
    ) -> MessageHandle | None:
        if not text:
            return None
        group_id = self._group_id_for_chat(chat_id)
        last: MessageHandle | None = None
        chunks = _signal_rich_text_chunks(text) if rich else [
            (chunk, []) for chunk in split_text_chunks(text, _SIGNAL_TEXT_LIMIT)
        ]
        for chunk, styles in chunks:
            if not chunk:
                continue
            params: dict[str, Any] = {"groupId": group_id, "message": chunk}
            _add_signal_text_styles(params, styles)
            result = await self._send_rpc_and_remember("send", params)
            timestamp = _extract_timestamp_from_value(result)
            if timestamp:
                last = MessageHandle(
                    chat_id=group_id,
                    message_id=timestamp,
                )
        return last

    async def edit_text(
        self, handle: MessageHandle, text: str, *, rich: bool = False,
    ) -> None:
        group_id = self._group_id_for_chat(handle.chat_id)
        message, styles = _signal_rich_text_payload(text) if rich else (
            text, [],
        )
        if not message:
            return
        params: dict[str, Any] = {
            "groupId": group_id,
            "editTimestamp": _timestamp_rpc_param(handle.message_id),
            "message": message,
        }
        _add_signal_text_styles(params, styles)
        await self._send_rpc_and_remember("send", params)

    async def delete_message(self, handle: MessageHandle) -> None:
        group_id = self._group_id_for_chat(handle.chat_id)
        await self._rpc_request(
            "remoteDelete",
            {
                "groupId": group_id,
                "targetTimestamp": _timestamp_rpc_param(handle.message_id),
            },
        )

    async def send_file(self, chat_id: str, path: str) -> None:
        self._check_upload_path(path)
        group_id = self._group_id_for_chat(chat_id)
        await self._send_rpc_and_remember(
            "send", {"groupId": group_id, "attachment": path},
        )

    async def start(self) -> None:
        self._stop_requested.clear()
        await self._outbound_journal.load()
        await self._connect_jsonrpc(resubscribe=False)
        try:
            self._group_ids_by_url = await self._resolve_group_ids()
            self._group_ids = set(self._group_ids_by_url.values())
            self.notify_targets = list(
                dict.fromkeys(self._group_ids_by_url.values())
            )
            migrated = workspace.migrate_current_workspace_platform_keys(
                _LEGACY_SIGNAL_PLATFORM_PREFIX,
                self.platform_id,
            )
            if migrated:
                logger.info(
                    "Migrated %d legacy Signal workspace selection(s).",
                    migrated,
                )
            self._migrate_group_schedules_from_current_workspaces()
            await self._migrate_group_queue_state()
            await self._start_daemon_services()
            await self._start_receive_subscription()
        except Exception:
            await self._stop_jsonrpc()
            raise
        logger.info(
            "Signal JSON-RPC bot started for %d group URL(s) via %s.",
            len(self._group_ids),
            self._jsonrpc_endpoint_name(),
        )

    async def stop(self) -> None:
        await self._stop_daemon_services()
        self._stop_requested.set()
        subscription = self._receive_subscription
        self._receive_subscription = None
        self._receive_started = False
        self._receive_subscribed = False
        if subscription is not None:
            with contextlib.suppress(Exception):
                await self._rpc_request(
                    "unsubscribeReceive",
                    {"subscription": subscription},
                    timeout=5,
                )
        for task in list(self._notification_tasks):
            task.cancel()
        if self._notification_tasks:
            await asyncio.gather(
                *self._notification_tasks, return_exceptions=True,
            )
        await self._flush_outbound_journal()
        await self._stop_jsonrpc()
        logger.info("Signal bot stopped.")

    async def _open_jsonrpc_socket(
        self, *, timeout: float = 15.0,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        unix_connector = None
        if self._jsonrpc_tcp_address is None:
            try:
                unix_connector = asyncio.open_unix_connection
            except AttributeError as exc:
                raise SignalCliError(
                    "Unix signal-cli sockets are unavailable on this platform; "
                    "set signal_jsonrpc_socket to tcp://HOST:PORT"
                ) from exc
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last_error: BaseException | None = None
        while True:
            try:
                if self._jsonrpc_tcp_address is not None:
                    connection = asyncio.open_connection(
                        *self._jsonrpc_tcp_address,
                        limit=_JSONRPC_STREAM_LIMIT,
                    )
                else:
                    assert unix_connector is not None
                    connection = unix_connector(
                        self.jsonrpc_socket,
                        limit=_JSONRPC_STREAM_LIMIT,
                    )
                return await asyncio.wait_for(
                    connection,
                    timeout=min(1.0, max(0.1, deadline - loop.time())),
                )
            except NotImplementedError as exc:
                if self._jsonrpc_tcp_address is not None:
                    raise
                raise SignalCliError(
                    "Unix signal-cli sockets are unavailable on this platform; "
                    "set signal_jsonrpc_socket to tcp://HOST:PORT"
                ) from exc
            except (OSError, TimeoutError) as exc:
                last_error = exc
                if loop.time() >= deadline:
                    raise SignalCliError(
                        "signal-cli JSON-RPC socket is not ready: "
                        f"{self.jsonrpc_socket}"
                    ) from last_error
                await asyncio.sleep(0.25)

    def _jsonrpc_connected(self) -> bool:
        return (
            self._jsonrpc_writer is not None
            and self._jsonrpc_reader_task is not None
            and not self._jsonrpc_reader_task.done()
        )

    async def _connect_jsonrpc(self, *, resubscribe: bool) -> None:
        async with self._jsonrpc_reconnect_lock:
            if not self._jsonrpc_connected():
                await self._close_jsonrpc_transport()
                reader, writer = await self._open_jsonrpc_socket()
                self._jsonrpc_reader = reader
                self._jsonrpc_writer = writer
                self._jsonrpc_reader_task = asyncio.create_task(
                    self._jsonrpc_reader_loop(reader, writer)
                )
            if (
                resubscribe
                and self._receive_started
                and not self._receive_subscribed
            ):
                self._receive_subscription = (
                    await self._subscribe_receive_once()
                )
                self._receive_subscribed = True
                logger.info(
                    "Signal JSON-RPC receive subscription restored."
                )

    async def _close_jsonrpc_transport(self) -> None:
        writer = self._jsonrpc_writer
        reader_task = self._jsonrpc_reader_task
        self._jsonrpc_reader = None
        self._jsonrpc_writer = None
        self._jsonrpc_reader_task = None
        self._receive_subscription = None
        self._receive_subscribed = False
        for request_id, fut in list(self._jsonrpc_pending.items()):
            self._jsonrpc_pending.pop(request_id, None)
            if not fut.done():
                fut.set_exception(SignalCliError("signal-cli JSON-RPC stopped"))
        if (
            reader_task is not None
            and reader_task is not asyncio.current_task()
            and not reader_task.done()
        ):
            reader_task.cancel()
        if writer is not None:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
        if (
            reader_task is not None
            and reader_task is not asyncio.current_task()
        ):
            with contextlib.suppress(asyncio.CancelledError):
                await reader_task

    async def _stop_jsonrpc(self) -> None:
        reconnect_task = self._jsonrpc_reconnect_task
        self._jsonrpc_reconnect_task = None
        if reconnect_task is not None and not reconnect_task.done():
            reconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reconnect_task
        await self._close_jsonrpc_transport()

    async def _jsonrpc_reader_loop(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while not self._stop_requested.is_set():
                line = await reader.readline()
                if not line:
                    break
                try:
                    payload = json.loads(line.decode("utf-8", errors="replace"))
                except (ValueError, RecursionError):
                    logger.debug(
                        "Ignoring non-JSON signal-cli JSON-RPC line: %r",
                        line,
                    )
                    continue
                if not isinstance(payload, dict):
                    continue
                if "id" in payload:
                    request_id = str(payload.get("id"))
                    fut = self._jsonrpc_pending.pop(request_id, None)
                    if fut is not None and not fut.done():
                        fut.set_result(payload)
                    continue
                if payload.get("method") == "receive":
                    task = asyncio.create_task(
                        self._handle_jsonrpc_receive(payload)
                    )
                    self._notification_tasks.add(task)
                    task.add_done_callback(self._notification_tasks.discard)
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self._stop_requested.is_set():
                logger.exception(
                    "signal-cli JSON-RPC %s failed",
                    self._jsonrpc_endpoint_name(),
                )
        finally:
            if self._jsonrpc_reader is reader:
                self._jsonrpc_reader = None
            if self._jsonrpc_writer is writer:
                self._jsonrpc_writer = None
            if self._jsonrpc_reader_task is asyncio.current_task():
                self._jsonrpc_reader_task = None
            self._receive_subscription = None
            self._receive_subscribed = False
            if not self._stop_requested.is_set():
                logger.warning(
                    "signal-cli JSON-RPC %s closed",
                    self._jsonrpc_endpoint_name(),
                )
            for request_id, fut in list(self._jsonrpc_pending.items()):
                self._jsonrpc_pending.pop(request_id, None)
                if not fut.done():
                    fut.set_exception(
                        SignalCliError("signal-cli JSON-RPC stdout closed")
                    )
            self._schedule_jsonrpc_reconnect()
            writer.close()

    async def _handle_jsonrpc_receive(self, payload: dict[str, Any]) -> None:
        try:
            params = payload.get("params")
            if not isinstance(params, dict):
                return
            item = params.get("result") if "result" in params else params
            if isinstance(item, dict):
                await self._handle_received_item(item)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Signal JSON-RPC receive notification failed")

    async def _subscribe_receive(self) -> int | None:
        result = await self._rpc_request("subscribeReceive", timeout=30)
        if isinstance(result, bool) or not isinstance(result, int):
            return None
        return result

    async def _subscribe_receive_once(self) -> int | None:
        result = await self._rpc_request_once("subscribeReceive", timeout=30)
        if isinstance(result, bool) or not isinstance(result, int):
            return None
        return result

    async def _start_receive_subscription(self) -> None:
        """Subscribe during startup without losing an immediate socket EOF.

        The reader task can observe EOF after ``subscribeReceive`` returns but
        before :meth:`start` records that receive handling is active.  Its
        normal reconnect path intentionally does nothing until receive has
        started, so that ordering would leave the bot marked subscribed on a
        dead transport.  Record the completed subscription and receive state
        together, then recover explicitly if the reader already went away.
        Once ``_receive_started`` is true, any later EOF schedules the normal
        reconnect path.
        """
        subscription = await self._subscribe_receive()

        self._receive_subscription = subscription
        self._receive_started = True
        self._receive_subscribed = True
        if self._jsonrpc_connected():
            return

        self._receive_subscription = None
        self._receive_subscribed = False
        try:
            await self._connect_jsonrpc(resubscribe=True)
        except BaseException:
            self._receive_started = False
            raise

    def _schedule_jsonrpc_reconnect(self) -> None:
        if self._stop_requested.is_set() or not self._receive_started:
            return
        task = self._jsonrpc_reconnect_task
        if task is not None and not task.done():
            return
        task = asyncio.create_task(self._jsonrpc_reconnect_loop())
        self._jsonrpc_reconnect_task = task
        task.add_done_callback(self._clear_jsonrpc_reconnect_task)

    def _clear_jsonrpc_reconnect_task(self, task: asyncio.Task) -> None:
        if self._jsonrpc_reconnect_task is task:
            self._jsonrpc_reconnect_task = None

    async def _jsonrpc_reconnect_loop(self) -> None:
        delay = 1.0
        while not self._stop_requested.is_set():
            try:
                await self._connect_jsonrpc(resubscribe=True)
                logger.info("Signal JSON-RPC reconnected.")
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Signal JSON-RPC reconnect failed; retrying in %.0fs: %s",
                    delay,
                    _safe_error_message(exc),
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

    async def _rpc_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: int | float | None = 60,
    ) -> Any:
        for attempt in range(2):
            await self._connect_jsonrpc(resubscribe=True)
            try:
                return await self._rpc_request_once(
                    method, params, timeout=timeout,
                )
            except Exception as exc:
                if attempt == 0 and _is_jsonrpc_transport_error(exc):
                    # Close so the next request reconnects.
                    await self._close_jsonrpc_transport()
                    if method in _NON_IDEMPOTENT_RPC_METHODS:
                        raise
                    logger.warning(
                        "Signal JSON-RPC request %s failed; reconnecting: %s",
                        method,
                        _safe_error_message(exc),
                    )
                    continue
                raise

    async def _rpc_request_once(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: int | float | None = 60,
    ) -> Any:
        writer = self._jsonrpc_writer
        if writer is None or not self._jsonrpc_connected():
            raise SignalCliError("signal-cli JSON-RPC is not running")
        loop = asyncio.get_running_loop()
        self._jsonrpc_next_id += 1
        request_id = str(self._jsonrpc_next_id)
        fut: asyncio.Future = loop.create_future()
        self._jsonrpc_pending[request_id] = fut
        request: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id,
        }
        if params:
            request["params"] = params
        line = json.dumps(request, separators=(",", ":")) + "\n"
        try:
            async with asyncio.timeout(timeout):
                async with self._jsonrpc_write_lock:
                    writer.write(line.encode("utf-8"))
                    await writer.drain()
                response = await fut
        except BaseException:
            self._jsonrpc_pending.pop(request_id, None)
            if not fut.done():
                fut.cancel()
            elif not fut.cancelled():
                fut.exception()
            raise
        if not isinstance(response, dict):
            raise SignalCliError("invalid JSON-RPC response")
        error = response.get("error")
        if error:
            if isinstance(error, dict):
                message = error.get("message") or json.dumps(error)
            else:
                message = str(error)
            raise SignalCliError(message)
        return response.get("result")

    def _jsonrpc_endpoint_name(self) -> str:
        return f"socket {self.jsonrpc_socket}"

    async def _handle_received_item(self, item: dict[str, Any]) -> None:
        receive_error = _format_signal_cli_receive_exception(item)
        if receive_error:
            logger.warning(
                "signal-cli dropped incoming Signal receive before dispatch: %s",
                receive_error,
            )
            return

        envelope = item.get("envelope") if isinstance(item, dict) else None
        if not isinstance(envelope, dict):
            envelope = item
        data, is_sent_sync = _extract_message_data(envelope)
        if not isinstance(data, dict):
            return

        group_id = _extract_message_group_id(data)
        if group_id not in self._group_ids:
            return

        account_id = _extract_account_id(item)
        sender_id = _extract_sender_id(envelope) or account_id
        if not sender_id:
            return

        text = str(data.get("message") or data.get("body") or "").strip()
        attachments = data.get("attachments") or []
        if _is_generated_control_message(text):
            return
        from_local_account = (
            is_sent_sync or _same_signal_id(sender_id, account_id)
        )
        source_device = envelope.get("sourceDevice")
        if not isinstance(source_device, int) or isinstance(source_device, bool) or source_device <= 0:
            source_device = None
        if from_local_account and self.own_device_id and source_device == self.own_device_id:
            return
        known_other_device = bool(self.own_device_id and source_device
                                  and source_device != self.own_device_id)
        if not known_other_device:
            await self._outbound_journal.load()
            if self._is_own_sent_echo(data, group_id, text, from_local_account):
                return
        incoming_key = _incoming_dedupe_key(
            data, group_id, sender_id, text, attachments,
        )
        if self._is_duplicate_incoming(incoming_key):
            logger.debug(
                "Skipping duplicate Signal message in group %s.",
                _short_id(group_id),
            )
            return
        self._migrate_group_workspace_state(sender_id, group_id, account_id)
        await self._migrate_group_workspace_files(
            sender_id, group_id, account_id,
        )

        if text.startswith("/"):
            await self.dispatch_command(
                self._ctx(group_id, text=text),
            )
            return

        if isinstance(attachments, list) and attachments:
            await self._handle_attachments(
                group_id, text, attachments,
            )
            return

        if text:
            await self.dispatch_text(
                self._ctx(group_id, text=text),
            )

    async def _handle_attachments(
        self,
        group_id: str,
        caption: str,
        attachments: list[Any],
    ) -> None:
        ctx_for_reply = self._ctx(group_id, text=caption)
        upload_dir = await self._attachment_upload_dir(ctx_for_reply)
        if upload_dir is None:
            return

        for att in attachments:
            if not isinstance(att, dict):
                continue
            try:
                info = await self._materialize_attachment(
                    att, group_id, upload_dir, caption,
                )
            except UploadTooLargeError:
                try:
                    await ctx_for_reply.reply_text(
                        upload_limit_message(self.max_upload_bytes),
                    )
                except Exception as reply_error:
                    logger.warning(
                        "Failed to report Signal attachment error: %s",
                        reply_error,
                    )
                continue
            except Exception as e:
                logger.warning("Failed to import Signal attachment: %s", e)
                try:
                    await ctx_for_reply.reply_text(
                        f"Failed to download attachment: {e}"
                    )
                except Exception as reply_error:
                    logger.warning(
                        "Failed to report Signal attachment error: %s",
                        reply_error,
                    )
                continue
            if info is None:
                continue
            await self.dispatch_file(
                self._ctx(group_id, attachment=info),
            )

    async def _materialize_attachment(
        self,
        att: dict[str, Any],
        group_id: str,
        upload_dir: str,
        caption: str,
    ) -> AttachmentInfo | None:
        filename = _attachment_filename(att)
        self._check_upload_size(_attachment_declared_size(att))
        source_path = _resolve_attachment_local_path(att)
        attachment_id = "" if source_path else _attachment_id(att)
        if not source_path and not attachment_id:
            return None

        with reserve_upload_path(upload_dir, filename) as local_path:
            if source_path:
                await copy_file_with_limit_async(
                    source_path,
                    local_path,
                    self.max_upload_bytes,
                )
            else:
                result = await self._rpc_request(
                    "getAttachment", {"id": attachment_id, "groupId": group_id},
                )
                payload = _strip_attachment_whitespace(_attachment_payload(result))
                data = _decode_attachment_payload(payload, self.max_upload_bytes)
                write_bytes_atomically(local_path, data)

        return AttachmentInfo(
            local_path,
            filename,
            attachment_kind_from_mime(att.get("contentType")),
            caption,
        )

    def _ctx(
        self,
        group_id: str,
        *,
        text: str = "",
        attachment: AttachmentInfo | None = None,
    ) -> BotContext:
        command: str | None = None
        args = ""
        if text.startswith("/"):
            parts = text[1:].split(None, 1)
            command = parts[0].split("@", 1)[0] if parts else ""
            args = parts[1] if len(parts) > 1 else ""
            text = ""
        return self.make_context(
            self._state_user_id(group_id),
            group_id,
            text=text,
            command=command,
            args=args,
            attachment=attachment,
        )

    async def _resolve_group_ids(self) -> dict[str, str]:
        resolved: dict[str, str] = {}
        total = len(self.group_urls)
        for index, group_url in enumerate(self.group_urls, 1):
            try:
                group_id = await self._resolve_group_id(group_url)
            except Exception as e:
                logger.warning(
                    "Skipping Signal group URL %d/%d: %s",
                    index, total, _safe_error_message(e),
                )
                continue
            resolved[group_url] = group_id
        if not resolved:
            raise RuntimeError("No configured Signal group URLs resolved")
        return resolved

    async def _resolve_group_id(self, group_url: str) -> str:
        group = await self._find_group_by_url(group_url)
        if group:
            group_id = _group_id(group)
            if _signal_group_is_member(group):
                return group_id

        joined_group_id = ""
        try:
            joined = await self._rpc_request("joinGroup", {"uri": group_url})
            joined_group_id = _first_group_id_from_value(joined)
        except SignalCliError as e:
            if "already" not in str(e).casefold():
                raise

        group = await self._find_group_by_url(group_url)
        if group:
            group_id = _group_id(group)
            if _signal_group_is_member(group):
                return group_id
            raise RuntimeError(
                "Signal group URL resolved but the bot is not a member: "
                f"{group_url!r}"
            )
        if joined_group_id:
            return joined_group_id

        raise RuntimeError(
            "Signal group URL could not be resolved after joinGroup: "
            f"{group_url!r}"
        )

    async def _find_group_by_url(
        self, group_url: str,
    ) -> dict[str, Any] | None:
        groups = await self._rpc_request("listGroups")
        wanted = _normalize_group_url(group_url)
        for group in _coerce_json_items(groups):
            group_id = _group_id(group)
            if not group_id:
                continue
            for url in _group_invite_urls(group):
                if _normalize_group_url(url) == wanted:
                    return group
        return None

    def _group_id_for_chat(self, chat_id: str) -> str:
        group_id = str(chat_id)
        if group_id not in self._group_ids:
            raise RuntimeError(f"Signal group is not configured: {chat_id}")
        return group_id

    def _state_user_id(self, group_id: str) -> str:
        return f"signal-group:{group_id}"

    @staticmethod
    def _legacy_source_ids(
        sender_id: str, group_id: str, account_id: str,
    ) -> list[str]:
        return list(
            dict.fromkeys(source_id for source_id in (sender_id, account_id, group_id) if source_id)
        )

    def _group_workspace(self, group_id: str) -> tuple[str, str] | None:
        target_user_id = self._state_user_id(group_id)
        ws = workspace.get_current(target_user_id, self.platform_id)
        if not ws or not os.path.isdir(ws):
            return None
        return target_user_id, ws

    def _legacy_group_state(
        self,
        sender_id: str,
        group_id: str,
        account_id: str,
    ) -> tuple[str, str, list[str]] | None:
        resolved = self._group_workspace(group_id)
        if resolved is None:
            return None
        target_user_id, ws = resolved
        source_ids = self._legacy_source_ids(sender_id, group_id, account_id)
        return target_user_id, ws, source_ids

    def _migrate_current_workspace_to_group(
        self, source_user_id: str, group_id: str,
    ) -> str:
        target_user_id = self._state_user_id(group_id)
        workspace.migrate_current_workspace(
            source_user_id,
            target_user_id,
            self.platform_id,
            source_bot_ids=(self.platform_id,),
            source_bot_prefixes=(_LEGACY_SIGNAL_PLATFORM_PREFIX,),
        )
        return target_user_id

    def _migrate_group_workspace_state(
        self,
        sender_id: str,
        group_id: str,
        account_id: str,
    ) -> None:
        target_user_id = self._state_user_id(group_id)
        if workspace.get_current(target_user_id, self.platform_id):
            return
        for source_id in self._legacy_source_ids(
            sender_id, group_id, account_id,
        ):
            if workspace.migrate_current_workspace(
                source_id,
                target_user_id,
                self.platform_id,
                source_bot_ids=(self.platform_id,),
                source_bot_prefixes=(_LEGACY_SIGNAL_PLATFORM_PREFIX,),
            ):
                logger.info(
                    "Migrated legacy Signal workspace state into group %s.",
                    _short_id(group_id),
                )
                return

    def _migrate_group_session_state(
        self,
        sender_id: str,
        group_id: str,
        account_id: str,
    ) -> None:
        state = self._legacy_group_state(sender_id, group_id, account_id)
        if state is None:
            return
        target_user_id, ws, source_ids = state
        if session.migrate_last_session(ws, source_ids, target_user_id):
            logger.info(
                "Migrated legacy Signal last-session state into group %s.",
                _short_id(group_id),
            )

    async def _migrate_group_workspace_files(
        self,
        sender_id: str,
        group_id: str,
        account_id: str,
    ) -> None:
        resolved = self._group_workspace(group_id)
        if resolved is None:
            return
        _target_user_id, ws = resolved
        async with workspace.get_lock(ws):
            self._migrate_group_session_state(sender_id, group_id, account_id)
            self._migrate_group_schedule_state(sender_id, group_id, account_id)

    def _migrate_group_schedules_from_current_workspaces(self) -> None:
        migrated = 0
        for uid, ws in workspace.iter_current_workspaces(self.platform_id):
            if uid.startswith("signal-group:") or not os.path.isdir(ws):
                continue
            for sched in schedules.list_schedules(ws, uid):
                chat_id = str(sched.get("chat_id") or "")
                if chat_id not in self._group_ids:
                    continue
                target_uid = self._migrate_current_workspace_to_group(
                    uid, chat_id,
                )
                migrated += schedules.migrate_schedules(
                    ws,
                    (uid,),
                    target_uid,
                    source_chat_id=chat_id,
                    target_chat_id=chat_id,
                    source_platform_id=self.platform_id,
                )
        if migrated:
            logger.info("Migrated %d legacy Signal schedule(s).", migrated)

        adopted = 0
        for uid, ws in workspace.iter_current_workspaces(self.platform_id):
            if not uid.startswith("signal-group:") or not os.path.isdir(ws):
                continue
            group_id = uid.removeprefix("signal-group:")
            if group_id not in self._group_ids:
                continue
            adopted += self._migrate_legacy_workspace_schedules(
                ws, uid, group_id,
            )
        if adopted:
            logger.info(
                "Adopted %d pre-Signal schedule(s) into Signal groups.",
                adopted,
            )

    def _migrate_group_schedule_state(
        self,
        sender_id: str,
        group_id: str,
        account_id: str,
    ) -> None:
        state = self._legacy_group_state(sender_id, group_id, account_id)
        if state is None:
            return
        target_user_id, ws, source_ids = state
        migrated = schedules.migrate_schedules(
            ws,
            source_ids,
            target_user_id,
            source_chat_id=group_id,
            target_chat_id=group_id,
            source_platform_id=self.platform_id,
        )
        migrated += self._migrate_legacy_workspace_schedules(
            ws, target_user_id, group_id,
        )
        if migrated:
            logger.info(
                "Migrated %d legacy Signal schedule(s) into group %s.",
                migrated,
                _short_id(group_id),
            )

    def _migrate_legacy_workspace_schedules(
        self,
        ws: str,
        target_user_id: str,
        group_id: str,
    ) -> int:
        targets_for_workspace = [
            self._state_user_id(candidate_group_id)
            for candidate_group_id in self._group_ids
            if workspace.get_current(
                self._state_user_id(candidate_group_id),
                self.platform_id,
            ) == workspace.canonicalize_workspace_path(ws)
        ]
        if targets_for_workspace != [target_user_id]:
            return 0

        source_ids = [
            uid for uid in schedules.list_schedule_user_ids(ws)
            if uid != target_user_id and not uid.startswith("signal-group:")
        ]
        if not source_ids:
            return 0
        return schedules.migrate_schedules(
            ws,
            source_ids,
            target_user_id,
            target_chat_id=group_id,
            source_platform_id=self.platform_id,
        )

    async def _migrate_group_queue_state(self) -> None:
        async with self._queue_file_lock:
            data = self._read_queue_file()
            if not data:
                return

            moved = 0
            changed = False
            for uid, entries in list(data.items()):
                if not isinstance(entries, list) or uid.startswith(
                    "signal-group:",
                ):
                    continue
                remaining: list[dict] = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        remaining.append(entry)
                        continue
                    chat_id = str(entry.get("chat_id") or "")
                    if chat_id not in self._group_ids:
                        remaining.append(entry)
                        continue
                    target_uid = self._migrate_current_workspace_to_group(
                        uid, chat_id,
                    )
                    ws = workspace.get_current(target_uid, self.platform_id)
                    if ws and os.path.isdir(ws):
                        session.migrate_last_session(ws, (uid,), target_uid)
                    target_entries = data.setdefault(target_uid, [])
                    if not isinstance(target_entries, list):
                        target_entries = []
                        data[target_uid] = target_entries
                    entry_id = entry.get("id")
                    duplicate = (
                        bool(entry_id)
                        and any(
                            isinstance(entry, dict) and entry.get("id") == entry_id
                            for entry in target_entries
                        )
                    )
                    if not duplicate:
                        target_entries.append(entry)
                        moved += 1
                    changed = True

                if remaining:
                    data[uid] = remaining
                else:
                    data.pop(uid, None)

            if changed:
                self._write_queue_file(data)
            if moved:
                logger.info("Migrated %d legacy Signal queued message(s).", moved)

    def _remember_own_sent_timestamp(self, group_id: str, timestamp: str) -> None:
        if not timestamp:
            return
        key = (group_id, timestamp)
        if key in self._own_sent_timestamps:
            return
        self._own_sent_timestamps.add(key)
        self._own_sent_timestamp_order.append(key)
        while len(self._own_sent_timestamp_order) > _OUTGOING_ECHO_LIMIT:
            old = self._own_sent_timestamp_order.popleft()
            self._own_sent_timestamps.discard(old)

    async def _send_rpc_and_remember(
        self, method: str, params: dict[str, Any],
    ) -> Any:
        """Persist text provenance before sending; retain acknowledged timestamps."""
        group_id = str(params["groupId"])
        text = params.get("message")
        text_hash = _echo_hash(text.strip()) if isinstance(text, str) and text.strip() else "-"
        group_hash = _echo_hash(group_id)
        kind = "edit" if "editTimestamp" in params else "text" if isinstance(text, str) else "attachment"
        started = time.monotonic()
        phase = "journal"
        try:
            if text_hash != "-":
                await self._outbound_journal.remember([(group_id, "text", text_hash)])
            phase = "rpc"
            result = await self._paced_rpc_request(method, params)
        except BaseException as exc:
            performance_logger.info(
                "stage=signal_send outcome=%s phase=%s kind=%s group_hash=%s text_hash=%s "
                "elapsed_seconds=%.3f error_type=%s",
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "error",
                phase, kind, group_hash, text_hash, time.monotonic() - started, type(exc).__name__,
            )
            raise
        timestamp = _extract_timestamp_from_value(result)
        outcomes = result.get("results") if isinstance(result, dict) else None
        successes = failures = unclassified = 0
        if isinstance(outcomes, list):
            for item in outcomes:
                outcome = item.get("type") if isinstance(item, dict) else None
                if outcome == "SUCCESS":
                    successes += 1
                elif isinstance(outcome, str) and outcome.endswith("_FAILURE"):
                    failures += 1
                else:
                    unclassified += 1
        outcome_name = "acknowledged" if timestamp else "no_timestamp"
        if failures:
            outcome_name += "_with_failures"
        performance_logger.info(
            "stage=signal_send outcome=%s kind=%s group_hash=%s text_hash=%s timestamp=%s "
            "elapsed_seconds=%.3f result_successes=%d result_failures=%d result_unclassified=%d",
            outcome_name, kind, group_hash, text_hash, timestamp or "-",
            time.monotonic() - started, successes, failures, unclassified,
        )
        if timestamp:
            self._remember_own_sent_timestamp(group_id, timestamp)
            task = create_background_task(
                self._persist_acknowledged_timestamp(group_id, timestamp),
                name="signal:outbound-journal", log=logger,
            )
            self._outbound_journal_tasks.add(task)
            task.add_done_callback(self._outbound_journal_tasks.discard)
        return result

    async def _persist_acknowledged_timestamp(self, group_id: str, timestamp: str) -> None:
        try:
            await self._outbound_journal.remember([(group_id, "timestamp", timestamp)])
        except Exception:
            logger.warning("Could not persist acknowledged Signal timestamp", exc_info=True)

    async def _flush_outbound_journal(self) -> None:
        if self._outbound_journal_tasks:
            await asyncio.gather(*self._outbound_journal_tasks, return_exceptions=True)

    async def _paced_rpc_request(
        self, method: str, params: dict[str, Any],
    ) -> Any:
        """Serialize sends, pace attachments, and honor Retry-After throttles.

        Signal's attachment CDN throttles rapid-fire uploads with
        ``RetryLaterException: Retry after N seconds``. Without pacing and
        a wait-and-retry, attaching many files fails most of them. The
        lock keeps back-to-back ``send`` calls (text chunks + attachments
        from one agent reply) at least ``_SIGNAL_SEND_MIN_INTERVAL_SEC``
        apart, and a server-requested ``Retry after N seconds`` delay is
        honored with a bounded retry instead of surfacing an error.
        """
        last_exc: BaseException | None = None
        for attempt in range(1, _SIGNAL_SEND_MAX_ATTEMPTS + 1):
            async with self._send_lock:
                now = time.monotonic()
                wait = (
                    self._last_send_monotonic
                    + _SIGNAL_SEND_MIN_INTERVAL_SEC
                    - now
                )
                if wait > 0:
                    await asyncio.sleep(wait)
                try:
                    return await self._rpc_request(method, params)
                except Exception as exc:
                    last_exc = exc
                    parsed = _signal_rate_limit_delay(exc)
                    if parsed is None:
                        raise
                    if attempt >= _SIGNAL_SEND_MAX_ATTEMPTS:
                        break
                    delay = max(
                        parsed,
                        _SIGNAL_SEND_BASE_DELAY_SEC * (2 ** (attempt - 1)),
                    )
                    logger.warning(
                        "Signal throttled send (attempt %d/%d); "
                        "retrying in %.1fs: %s",
                        attempt,
                        _SIGNAL_SEND_MAX_ATTEMPTS,
                        delay,
                        _safe_error_message(exc),
                    )
                finally:
                    self._last_send_monotonic = time.monotonic()
            await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _is_own_sent_echo(
        self,
        data: dict[str, Any],
        group_id: str,
        text: str,
        from_local_account: bool,
    ) -> bool:
        if not from_local_account:
            return False
        timestamp = _extract_timestamp_from_value(data)
        if timestamp and (group_id, timestamp) in self._own_sent_timestamps:
            return True
        return self._outbound_journal.contains(group_id, text=text, timestamp=timestamp)

    def _is_duplicate_incoming(self, key: str) -> bool:
        if not key:
            return False
        self._prune_incoming_keys()
        if key in self._recent_incoming_keys:
            return True
        self._recent_incoming_keys.add(key)
        self._recent_incoming_key_order.append((time.monotonic(), key))
        while len(self._recent_incoming_key_order) > _INCOMING_DEDUPE_LIMIT:
            _created_at, old_key = self._recent_incoming_key_order.popleft()
            self._recent_incoming_keys.discard(old_key)
        return False

    def _prune_incoming_keys(self) -> None:
        cutoff = time.monotonic() - _INCOMING_DEDUPE_TTL
        while (
            self._recent_incoming_key_order
            and self._recent_incoming_key_order[0][0] < cutoff
        ):
            _created_at, old_key = self._recent_incoming_key_order.popleft()
            self._recent_incoming_keys.discard(old_key)


def _safe_error_message(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    text = _SIGNAL_GROUP_URL_RE.sub("<signal-group-url>", text)
    return _SIGNAL_GROUP_SCHEME_RE.sub("<signal-group-url>", text)


def _is_jsonrpc_transport_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    if not isinstance(exc, SignalCliError):
        return False
    text = str(exc).casefold()
    return any(
        marker in text
        for marker in (
            "json-rpc stopped",
            "json-rpc stdout closed",
            "json-rpc is not running",
            "socket is not ready",
        )
    )


def _signal_rich_text_chunks(
    text: str,
    limit: int = _SIGNAL_TEXT_LIMIT,
) -> list[tuple[str, list[str]]]:
    body, spans = _md_to_signal_body_and_spans(text)
    if not body:
        return []
    ranges = text_chunk_ranges(body, limit)
    if len(ranges) == 1:
        return [(body, _signal_style_strings_for_chunk(body, spans, 0, len(body)))]
    # Preserve the renderer's original (including nested/postorder) span order
    # while visiting only chunks each span actually overlaps. Scanning every
    # span again for every chunk was quadratic for large formatted replies.
    chunk_starts = [start for start, _end in ranges]
    chunk_ends = [end for _start, end in ranges]
    chunk_spans: list[list[tuple[int, int, str]]] = [[] for _ in ranges]
    for span in spans:
        start, length, _style = span
        first_chunk = bisect_right(chunk_ends, start)
        after_last_chunk = bisect_left(chunk_starts, start + length)
        for chunk_index in range(first_chunk, after_last_chunk):
            chunk_spans[chunk_index].append(span)
    return [
        (
            chunk,
            _signal_style_strings_for_chunk(body, chunk_spans[index], start, end),
        )
        for index, (start, end) in enumerate(ranges)
        if (chunk := body[start:end])
    ]


def _signal_rich_text_payload(text: str) -> tuple[str, list[str]]:
    body, spans = _md_to_signal_body_and_spans(text)
    if not body:
        return "", []
    return body, _signal_style_strings_for_chunk(body, spans, 0, len(body))


def _md_to_signal_body_and_spans(
    text: str,
) -> tuple[str, list[tuple[int, int, str]]]:
    """Convert common Markdown to Signal text plus code-point spans."""
    parts: list[str] = []
    spans: list[tuple[int, int, str]] = []
    pos = 0

    def append(value: str) -> None:
        nonlocal pos
        if not value:
            return
        parts.append(value)
        pos += len(value)

    def add_span(start: int, style: str) -> None:
        length = pos - start
        if length > 0:
            spans.append((start, length, style))

    def parse_inline(inline_src: str) -> None:
        offset = 0
        # Each parse scope advances monotonically. Retain the next valid
        # single-character closer (or a known exhausted search) so unmatched
        # identifiers cannot rescan every later underscore/star at each opener.
        # Nested scopes have their own source string and their own tiny cache.
        single_closes: dict[str, int] = {}
        while offset < len(inline_src):
            marker = _signal_inline_marker_at(inline_src, offset)
            if marker is None:
                # Ordinary prose contains no inline control markers. Append
                # that whole run instead of testing/storing each character;
                # every potential marker still uses the existing parser.
                next_marker = _SIGNAL_INLINE_START_RE.search(inline_src, offset + 1)
                end = next_marker.start() if next_marker is not None else len(inline_src)
                append(inline_src[offset:end])
                offset = end
                continue

            open_marker, close_marker, style, parse_nested = marker
            if close_marker in ("*", "_"):
                cached = single_closes.get(close_marker)
                if cached is not None and (cached < 0 or cached >= offset + len(open_marker)):
                    close_at = cached
                else:
                    close_at = _find_signal_inline_close(inline_src, close_marker, offset)
                    single_closes[close_marker] = close_at
            else:
                close_at = _find_signal_inline_close(inline_src, close_marker, offset)
            if close_at < 0:
                append(inline_src[offset])
                offset += 1
                continue

            start = pos
            content = inline_src[offset + len(open_marker):close_at]
            if parse_nested:
                parse_inline(content)
            else:
                append(content)
            add_span(start, style)
            offset = close_at + len(close_marker)

    need_newline = False

    def append_line(line: str, style: str | None = None) -> None:
        nonlocal need_newline
        if need_newline:
            append("\n")
        need_newline = True
        start = pos
        if style == _SIGNAL_STYLE_MONOSPACE:
            append(line)
        else:
            parse_inline(line)
        if style:
            add_span(start, style)

    for is_code, lines in iter_fenced_markdown(text):
        if is_code:
            if lines:
                append_line("\n".join(lines), _SIGNAL_STYLE_MONOSPACE)
            continue

        line = lines[0]
        header = _SIGNAL_MD_HEADING_RE.match(line)
        if header:
            append_line(header.group(1), _SIGNAL_STYLE_BOLD)
        else:
            append_line(line)

    return "".join(parts), spans


def _signal_inline_marker_at(
    text: str,
    index: int,
) -> tuple[str, str, str, bool] | None:
    if text.startswith("`", index):
        return ("`", "`", _SIGNAL_STYLE_MONOSPACE, False)
    if text.startswith("**", index):
        return ("**", "**", _SIGNAL_STYLE_BOLD, True)
    if text.startswith("__", index):
        return ("__", "__", _SIGNAL_STYLE_BOLD, True)
    if text.startswith("~~", index):
        return ("~~", "~~", _SIGNAL_STYLE_STRIKETHROUGH, True)
    if text.startswith("||", index):
        return ("||", "||", _SIGNAL_STYLE_SPOILER, True)
    if text[index] == "*" and _single_marker_can_open(text, index):
        return ("*", "*", _SIGNAL_STYLE_ITALIC, True)
    if text[index] == "_" and _single_marker_can_open(text, index):
        return ("_", "_", _SIGNAL_STYLE_ITALIC, True)
    return None


def _find_signal_inline_close(
    text: str,
    marker: str,
    open_index: int,
) -> int:
    start = open_index + len(marker)
    if marker in ("*", "_"):
        found = text.find(marker, start)
        while found >= 0:
            if _single_marker_can_close(text, found):
                return found
            found = text.find(marker, found + 1)
        return -1
    return text.find(marker, start)


def _single_marker_can_open(text: str, index: int) -> bool:
    previous = text[index - 1] if index > 0 else ""
    next_char = text[index + 1] if index + 1 < len(text) else ""
    return not _is_word_char(previous) and bool(next_char.strip())


def _single_marker_can_close(text: str, index: int) -> bool:
    previous = text[index - 1] if index > 0 else ""
    next_char = text[index + 1] if index + 1 < len(text) else ""
    return bool(previous.strip()) and not _is_word_char(next_char)


def _is_word_char(value: str) -> bool:
    return bool(value and _SIGNAL_WORD_CHAR_RE.match(value))


def _signal_style_strings_for_chunk(
    body: str,
    spans: list[tuple[int, int, str]],
    chunk_start: int,
    chunk_end: int,
) -> list[str]:
    relevant = [
        (max(start, chunk_start), min(start + length, chunk_end), style)
        for start, length, style in spans
        if max(start, chunk_start) < min(start + length, chunk_end)
    ]
    if not relevant:
        return []
    # One UTF-16 prefix sum (astral = 2).
    max_span_end = max(end for _, end, _ in relevant)
    prefix: list[int] = [0] * (max_span_end - chunk_start + 1)
    total = 0
    for body_index in range(chunk_start, max_span_end):
        total += 2 if ord(body[body_index]) > 0xFFFF else 1
        prefix[body_index - chunk_start + 1] = total
    styles: list[str] = []
    for overlap_start, overlap_end, style in relevant:
        utf16_start = prefix[overlap_start - chunk_start]
        utf16_length = (
            prefix[overlap_end - chunk_start] - prefix[overlap_start - chunk_start]
        )
        if utf16_length > 0:
            styles.append(f"{utf16_start}:{utf16_length}:{style}")
    return styles


def _add_signal_text_styles(
    params: dict[str, Any],
    styles: list[str],
) -> None:
    if len(styles) == 1:
        params["textStyle"] = styles[0]
    elif styles:
        params["textStyle"] = styles


def _format_signal_cli_receive_exception(item: dict[str, Any]) -> str:
    exc = item.get("exception")
    if not isinstance(exc, dict):
        return ""

    exc_type = str(exc.get("type") or "").strip()
    message = str(exc.get("message") or "").strip()
    if exc_type and message:
        detail = f"{exc_type}: {message}"
    else:
        detail = exc_type or message or json.dumps(exc, default=str)

    envelope = item.get("envelope")
    timestamp = _extract_timestamp_from_value(envelope)
    if timestamp:
        return f"{detail} (timestamp {timestamp})"
    return detail


def _incoming_dedupe_key(
    data: dict[str, Any],
    group_id: str,
    sender_id: str,
    text: str,
    attachments: Any,
) -> str:
    timestamp = _extract_timestamp_from_value(data)
    attachment_keys: list[str] = []
    if isinstance(attachments, list):
        for attachment in attachments:
            if isinstance(attachment, dict):
                attachment_keys.append(
                    _attachment_id(attachment)
                    or _attachment_filename(attachment)
                    or json.dumps(attachment, sort_keys=True, default=str)
                )
            else:
                attachment_keys.append(str(attachment))
    if not timestamp and not text and not attachment_keys:
        return ""
    return json.dumps(
        {
            "group_id": group_id,
            "sender_id": sender_id,
            "timestamp": timestamp,
            "text": text if not timestamp else "",
            "attachments": attachment_keys,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _short_id(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return value if len(value) <= 12 else value[:11] + "…"


def _dedupe_group_urls(group_urls: list[str]) -> list[str]:
    if isinstance(group_urls, str):
        group_urls = [group_urls]
    if not isinstance(group_urls, list):
        return []
    urls: list[str] = []
    seen: set[str] = set()
    for value in group_urls:
        if not isinstance(value, str):
            continue
        url = value.strip()
        if not url:
            continue
        normalized = _normalize_group_url(url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        urls.append(url)
    return urls


def _coerce_json_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("envelopes", "messages", "results", "groups"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [value]
    return []


def _first_group_id_from_value(value: Any) -> str:
    for item in _coerce_json_items(value):
        group_id = _group_id(item)
        if group_id:
            return group_id
    return ""


def _extract_message_data(
    envelope: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    data = envelope.get("dataMessage")
    if isinstance(data, dict):
        return data, False

    sync = envelope.get("syncMessage")
    if isinstance(sync, dict):
        sent = sync.get("sentMessage")
        if isinstance(sent, dict):
            return sent, True

    return None, False


def _group_id(group: dict[str, Any]) -> str:
    for key in ("id", "groupId", "groupID"):
        value = group.get(key)
        normalized = _normalize_group_id(value)
        if normalized:
            return normalized
    return ""


def _signal_group_is_member(group: dict[str, Any]) -> bool:
    return group.get("isMember") is not False


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
            text.startswith(("https://signal.group/", "sgnl://"))
            or ("invite" in lowered_key and "signal.group" in text)
        ):
            urls.append(text)

    visit(group)
    return urls


def _normalize_group_url(value: str) -> str:
    if not isinstance(value, str):
        return ""
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
    if (
        isinstance(value, list)
        and value
        and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )
    ):
        try:
            return base64.b64encode(bytes(value)).decode("ascii")
        except ValueError:
            return ""
    return ""


def _signal_id_text(value: Any) -> str:
    """Stringify a Signal identifier without coercing junk types.

    Booleans, dicts, lists, floats, and None must not become "True" or
    "{...}" sender/account ids. Only non-empty strings (stripped) and
    plain ints (excluding bool) are usable ids.
    """
    if isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int):
        return str(value)
    return ""


def _extract_sender_id(envelope: dict[str, Any]) -> str:
    for key in ("sourceNumber", "source", "sourceUuid", "sourceName"):
        text = _signal_id_text(envelope.get(key))
        if text:
            return text
    source = envelope.get("sourceAddress")
    if isinstance(source, dict):
        for key in ("number", "uuid", "name"):
            text = _signal_id_text(source.get(key))
            if text:
                return text
    return ""


def _extract_account_id(item: dict[str, Any]) -> str:
    return _signal_id_text(item.get("account"))


def _same_signal_id(left: str, right: str) -> bool:
    return bool(left and right and left.strip() == right.strip())


def _extract_timestamp_from_value(value: Any) -> str | None:
    timestamp = _find_key(value, "timestamp")
    if timestamp is None or isinstance(timestamp, bool):
        return None
    if isinstance(timestamp, int):
        return str(timestamp) if timestamp >= 0 else None
    if isinstance(timestamp, float):
        if not math.isfinite(timestamp) or timestamp < 0:
            return None
        as_int = int(timestamp)
        return str(as_int if float(as_int) == timestamp else timestamp)
    if isinstance(timestamp, str):
        text = timestamp.strip()
        if not text:
            return None
        try:
            number = float(text)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number) or number < 0:
            return None
        return text
    return None


def _timestamp_rpc_param(value: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError, OverflowError) as exc:
        raise SignalCliError(
            f"signal-cli returned a non-numeric timestamp: {value!r}",
        ) from exc


def _attachment_payload(value: Any) -> str:
    if isinstance(value, str):
        return value
    payload = _find_key(value, "data")
    if isinstance(payload, str):
        return payload
    payload = _find_key(value, "attachment")
    if isinstance(payload, str):
        return payload
    raise SignalCliError("signal-cli getAttachment returned no payload")


def _attachment_declared_size(att: dict[str, Any]) -> int | None:
    """Return a usable signal-cli attachment byte count, when provided."""
    for key in ("size", "fileSize", "length"):
        value = att.get(key)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        ):
            return value
    return None


def _strip_attachment_whitespace(payload: str) -> str:
    """Normalize base64 whitespace without a regex scan for ordinary payloads."""
    if payload.isascii():
        # Signal normally sends unbroken ASCII base64. Reuse that string
        # without another attachment-sized allocation when nothing is removed.
        if not any(character in payload for character in _SIGNAL_ATTACH_ASCII_WHITESPACE):
            return payload
        return payload.translate(_SIGNAL_ATTACH_ASCII_WHITESPACE_DELETE)
    # Keep the Unicode whitespace accepted by the existing import path.
    return _SIGNAL_ATTACH_WHITESPACE_RE.sub("", payload)


def _decode_attachment_payload(payload: str, max_upload_bytes: int) -> bytes:
    """Decode a bounded base64 Signal attachment payload."""
    max_encoded_bytes = ((max_upload_bytes + 2) // 3) * 4
    if len(payload) > max_encoded_bytes:
        raise UploadTooLargeError(max_upload_bytes)
    data = base64.b64decode(payload, validate=True)
    if upload_size_exceeds_limit(len(data), max_upload_bytes):
        raise UploadTooLargeError(max_upload_bytes)
    return data


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
    for key in (
        "fileName",
        "filename",
        "name",
        "storedFilename",
        "localPath",
        "path",
    ):
        value = att.get(key)
        if isinstance(value, str) and value.strip():
            return os.path.basename(value.strip())
    attachment_id = _attachment_id(att) or "attachment"
    ext = _extension_for_content_type(att.get("contentType"))
    return f"signal_{attachment_id}{ext}"


def _attachment_id(att: dict[str, Any]) -> str:
    for key in ("id", "attachmentId", "attachmentPointerId", "storedFilename"):
        text = _signal_id_text(att.get(key))
        if text:
            if key == "storedFilename":
                text = os.path.basename(text)
            return text
    return ""


def _attachment_local_path(att: dict[str, Any]) -> str:
    for key in ("path", "localPath", "storedFilename"):
        value = att.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _resolve_attachment_local_path(att: dict[str, Any]) -> str:
    raw_path = _attachment_local_path(att)
    if not raw_path:
        return ""

    for candidate in _attachment_path_candidates(raw_path):
        if os.path.isfile(candidate):
            return candidate
    return ""


def _attachment_path_candidates(path: str) -> list[str]:
    if os.path.isabs(path):
        return [path]

    candidates = [path]
    basename = os.path.basename(path)
    for attachments_dir in _signal_cli_attachment_dirs():
        candidates.append(os.path.join(attachments_dir, path))
        if basename != path:
            candidates.append(os.path.join(attachments_dir, basename))
    return list(dict.fromkeys(candidates))


def _signal_cli_attachment_dirs() -> list[str]:
    data_dirs: list[str] = []
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        data_dirs.append(os.path.join(xdg_data_home, "signal-cli"))
    data_dirs.append(os.path.expanduser("~/.local/share/signal-cli"))
    data_dirs.append(os.path.expanduser("~/.config/signal-cli"))

    return [
        os.path.join(data_dir, "attachments")
        for data_dir in dict.fromkeys(data_dirs)
    ]


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
