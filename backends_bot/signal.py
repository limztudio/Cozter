"""Signal adapter: wires signal-cli group messages to BotPlatform.

This backend talks to Signal through a shared signal-cli daemon JSON-RPC
Unix socket. The daemon lifecycle is owned outside Cozter.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import shutil
import time
from collections import deque
from typing import Any

from .. import schedules, session, workspace
from .base import AttachmentInfo, BotContext, BotPlatform, MessageHandle

logger = logging.getLogger(__name__)

_SIGNAL_TEXT_LIMIT = 4_000
_OUTGOING_ECHO_TTL = 30.0
_OUTGOING_ECHO_LIMIT = 200
_INCOMING_DEDUPE_TTL = 120.0
_INCOMING_DEDUPE_LIMIT = 500
_LEGACY_SIGNAL_PLATFORM_PREFIX = "signal:"
_JSONRPC_STREAM_LIMIT = 128 * 1024 * 1024


class SignalCliError(RuntimeError):
    """Raised when the signal-cli JSON-RPC socket is unavailable."""


class SignalBot(BotPlatform):
    """Signal group adapter backed by signal-cli."""

    def __init__(
        self,
        group_urls: list[str],
        *,
        recent_limit: int = 10,
        max_queue_size: int = 50,
        jsonrpc_socket: str = "",
    ):
        super().__init__(
            group_urls,
            recent_limit=recent_limit,
            max_queue_size=max_queue_size,
        )
        self.group_urls = _dedupe_group_urls(group_urls)
        self.jsonrpc_socket = jsonrpc_socket.strip() if jsonrpc_socket else ""
        if not self.jsonrpc_socket:
            raise ValueError("signal_jsonrpc_socket is required for Signal")
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
        self._own_sent_timestamps: set[str] = set()
        self._own_sent_timestamp_order: deque[str] = deque()
        self._recent_outgoing_texts: deque[tuple[float, str, str]] = deque()
        self._recent_incoming_keys: set[str] = set()
        self._recent_incoming_key_order: deque[tuple[float, str]] = deque()

    @property
    def platform_id(self) -> str:
        return "signal"

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
            self._remember_outgoing_text(group_id, chunk)
            result = await self._rpc_request(
                "send", {"groupId": group_id, "message": chunk},
            )
            timestamp = _extract_timestamp_from_value(result)
            if timestamp:
                self._remember_own_sent_timestamp(timestamp)
                last = MessageHandle(
                    chat_id=group_id,
                    message_id=timestamp,
                )
        return last

    async def edit_text(self, handle: MessageHandle, text: str) -> None:
        group_id = self._group_id_for_chat(handle.chat_id)
        self._remember_outgoing_text(group_id, text)
        result = await self._rpc_request(
            "send",
            {
                "groupId": group_id,
                "editTimestamp": handle.message_id,
                "message": text,
            },
        )
        timestamp = _extract_timestamp_from_value(result)
        if timestamp:
            self._remember_own_sent_timestamp(timestamp)

    async def delete_message(self, handle: MessageHandle) -> None:
        group_id = self._group_id_for_chat(handle.chat_id)
        await self._rpc_request(
            "remoteDelete",
            {"groupId": group_id, "targetTimestamp": handle.message_id},
        )

    async def send_file(self, chat_id: str, path: str) -> None:
        group_id = self._group_id_for_chat(chat_id)
        result = await self._rpc_request(
            "send", {"groupId": group_id, "attachment": path},
        )
        timestamp = _extract_timestamp_from_value(result)
        if timestamp:
            self._remember_own_sent_timestamp(timestamp)

    async def send_status(self, chat_id: str, text: str) -> None:
        # Signal has no cheap, reliable transient status surface if
        # message timestamps are unavailable, so avoid spamming the group
        # with every tool event. The final reply still arrives normally.
        return None

    # ----- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        self._stop_requested.clear()
        await self._start_jsonrpc()
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
            await self.restore_queues()
            await self.start_scheduler()
            self._receive_subscription = await self._subscribe_receive()
            self._receive_started = True
            self._receive_subscribed = True
        except Exception:
            await self._stop_jsonrpc()
            raise
        logger.info(
            "Signal JSON-RPC bot started for %d group URL(s) via %s.",
            len(self._group_ids),
            self._jsonrpc_endpoint_name(),
        )

    async def stop(self) -> None:
        await self.stop_scheduler()
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
        await self._stop_jsonrpc()
        logger.info("Signal bot stopped.")

    async def send_startup_messages(
        self, version: str, commit_date: str,
    ) -> None:
        msg = (
            f"Cozter started.\nVersion: {version}\nUpdated: {commit_date}"
        )
        await self.notify_users(msg)

    # ----- JSON-RPC transport --------------------------------------------

    async def _start_jsonrpc(self) -> None:
        await self._connect_jsonrpc(resubscribe=False)

    async def _open_jsonrpc_socket(
        self, *, timeout: float = 15.0,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last_error: BaseException | None = None
        while True:
            try:
                return await asyncio.wait_for(
                    asyncio.open_unix_connection(
                        self.jsonrpc_socket,
                        limit=_JSONRPC_STREAM_LIMIT,
                    ),
                    timeout=min(1.0, max(0.1, deadline - loop.time())),
                )
            except (OSError, asyncio.TimeoutError) as exc:
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
                except json.JSONDecodeError:
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
        return int(result) if isinstance(result, int) else None

    async def _subscribe_receive_once(self) -> int | None:
        result = await self._rpc_request_once("subscribeReceive", timeout=30)
        return int(result) if isinstance(result, int) else None

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
                    logger.warning(
                        "Signal JSON-RPC request %s failed; reconnecting: %s",
                        method,
                        _safe_error_message(exc),
                    )
                    await self._close_jsonrpc_transport()
                    continue
                raise
        raise SignalCliError("signal-cli JSON-RPC request failed")

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
            async with self._jsonrpc_write_lock:
                writer.write(line.encode("utf-8"))
                await writer.drain()
            response = await asyncio.wait_for(fut, timeout=timeout)
        except Exception:
            self._jsonrpc_pending.pop(request_id, None)
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
        from_local_account = (
            is_sent_sync or _same_signal_id(sender_id, account_id)
        )
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
                self._ctx(sender_id, group_id, text=text),
            )
            return

        if isinstance(attachments, list) and attachments:
            await self._handle_attachments(
                sender_id, group_id, text, attachments,
            )
            return

        if text:
            await self.dispatch_text(
                self._ctx(sender_id, group_id, text=text),
            )

    async def _handle_attachments(
        self,
        sender_id: str,
        group_id: str,
        caption: str,
        attachments: list[Any],
    ) -> None:
        ctx_for_reply = self._ctx(sender_id, group_id, text=caption)
        ws = workspace.get_current(ctx_for_reply.user_id, self.platform_id)
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
                self._ctx(sender_id, group_id, attachment=info),
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
        source_path = _resolve_attachment_local_path(att)

        if source_path:
            shutil.copyfile(source_path, local_path)
        else:
            attachment_id = _attachment_id(att)
            if not attachment_id:
                return None
            result = await self._rpc_request(
                "getAttachment", {"id": attachment_id, "groupId": group_id},
            )
            payload = re.sub(r"\s+", "", _attachment_payload(result))
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
        _sender_id: str,
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
        return BotContext(
            user_id=self._state_user_id(group_id),
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
        group_id = await self._find_group_id_by_url(group_url)
        if group_id:
            return group_id

        try:
            joined = await self._rpc_request("joinGroup", {"uri": group_url})
            group_id = _first_group_id_from_value(joined)
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
        groups = await self._rpc_request("listGroups")
        wanted = _normalize_group_url(group_url)
        for group in _extract_groups_from_value(groups):
            group_id = _group_id(group)
            if not group_id:
                continue
            for url in _group_invite_urls(group):
                if _normalize_group_url(url) == wanted:
                    return group_id
        return None

    def _group_id_for_chat(self, chat_id: str) -> str:
        group_id = str(chat_id)
        if group_id not in self._group_ids:
            raise RuntimeError(f"Signal group is not configured: {chat_id}")
        return group_id

    def _state_user_id(self, group_id: str) -> str:
        return f"signal-group:{group_id}"

    def _migrate_group_workspace_state(
        self,
        sender_id: str,
        group_id: str,
        account_id: str,
    ) -> None:
        target_user_id = self._state_user_id(group_id)
        if workspace.get_current(target_user_id, self.platform_id):
            return
        source_ids = dict.fromkeys(
            x for x in (sender_id, account_id, group_id) if x
        )
        for source_id in source_ids:
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
        target_user_id = self._state_user_id(group_id)
        ws = workspace.get_current(target_user_id, self.platform_id)
        if not ws or not os.path.isdir(ws):
            return
        source_ids = list(
            dict.fromkeys(x for x in (sender_id, account_id, group_id) if x)
        )
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
        target_user_id = self._state_user_id(group_id)
        ws = workspace.get_current(target_user_id, self.platform_id)
        if not ws or not os.path.isdir(ws):
            return
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
                target_uid = self._state_user_id(chat_id)
                workspace.migrate_current_workspace(
                    uid,
                    target_uid,
                    self.platform_id,
                    source_bot_ids=(self.platform_id,),
                    source_bot_prefixes=(_LEGACY_SIGNAL_PLATFORM_PREFIX,),
                )
                migrated += schedules.migrate_schedules(
                    ws,
                    (uid,),
                    target_uid,
                    source_chat_id=chat_id,
                    target_chat_id=chat_id,
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
        target_user_id = self._state_user_id(group_id)
        ws = workspace.get_current(target_user_id, self.platform_id)
        if not ws or not os.path.isdir(ws):
            return
        source_ids = list(
            dict.fromkeys(x for x in (sender_id, account_id, group_id) if x)
        )
        migrated = schedules.migrate_schedules(
            ws,
            source_ids,
            target_user_id,
            source_chat_id=group_id,
            target_chat_id=group_id,
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
            ) == ws
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
                    target_uid = self._state_user_id(chat_id)
                    workspace.migrate_current_workspace(
                        uid,
                        target_uid,
                        self.platform_id,
                        source_bot_ids=(self.platform_id,),
                        source_bot_prefixes=(_LEGACY_SIGNAL_PLATFORM_PREFIX,),
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
                            isinstance(e, dict) and e.get("id") == entry_id
                            for e in target_entries
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

    def _remember_outgoing_text(self, group_id: str, text: str) -> None:
        self._prune_outgoing_texts()
        self._recent_outgoing_texts.append((time.monotonic(), group_id, text))
        while len(self._recent_outgoing_texts) > _OUTGOING_ECHO_LIMIT:
            self._recent_outgoing_texts.popleft()

    def _remember_own_sent_timestamp(self, timestamp: str) -> None:
        if not timestamp:
            return
        self._own_sent_timestamps.add(timestamp)
        self._own_sent_timestamp_order.append(timestamp)
        while len(self._own_sent_timestamp_order) > _OUTGOING_ECHO_LIMIT:
            old = self._own_sent_timestamp_order.popleft()
            self._own_sent_timestamps.discard(old)

    def _is_own_sent_echo(
        self,
        data: dict[str, Any],
        group_id: str,
        text: str,
        from_local_account: bool,
    ) -> bool:
        timestamp = _extract_timestamp_from_value(data)
        if timestamp and timestamp in self._own_sent_timestamps:
            return True
        if not from_local_account or not text:
            return False
        return self._consume_recent_outgoing_text(group_id, text)

    def _consume_recent_outgoing_text(self, group_id: str, text: str) -> bool:
        self._prune_outgoing_texts()
        kept: deque[tuple[float, str, str]] = deque()
        matched = False
        while self._recent_outgoing_texts:
            item = self._recent_outgoing_texts.popleft()
            _created_at, item_group_id, item_text = item
            if not matched and item_group_id == group_id and item_text == text:
                matched = True
                continue
            kept.append(item)
        self._recent_outgoing_texts = kept
        return matched

    def _prune_outgoing_texts(self) -> None:
        cutoff = time.monotonic() - _OUTGOING_ECHO_TTL
        while (
            self._recent_outgoing_texts
            and self._recent_outgoing_texts[0][0] < cutoff
        ):
            self._recent_outgoing_texts.popleft()

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


def _safe_error_message(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    text = re.sub(r"https://signal\.group/#[^\s]+", "<signal-group-url>", text)
    text = re.sub(r"sgnl://[^\s]+", "<signal-group-url>", text)
    return text


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
    return value if len(value) <= 12 else f"{value[:8]}..."


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


def _extract_groups_from_value(value: Any) -> list[dict[str, Any]]:
    return _coerce_json_items(value)


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


def _extract_account_id(item: dict[str, Any]) -> str:
    value = item.get("account")
    if value and not isinstance(value, (dict, list)):
        return str(value)
    return ""


def _same_signal_id(left: str, right: str) -> bool:
    return bool(left and right and left.strip() == right.strip())


def _extract_timestamp_from_value(value: Any) -> str | None:
    timestamp = _find_key(value, "timestamp")
    return str(timestamp) if timestamp is not None else None


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
        value = att.get(key)
        if value:
            value = str(value)
            if key == "storedFilename":
                value = os.path.basename(value)
            return value
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
