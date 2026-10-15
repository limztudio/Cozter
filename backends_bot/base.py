"""Chat-platform abstraction and shared command handlers.

Each concrete BotPlatform (telegram, slack, ...) is responsible for:
  1. Starting/stopping its framework and dispatching platform events.
  2. Building a BotContext per event and calling into the shared
     ``dispatch_command`` / ``dispatch_text`` / ``dispatch_file`` hooks.
  3. Providing the send/edit/delete/file primitives through the abstract
     methods at the bottom of BotPlatform.

All command logic lives here on BotPlatform itself, so the per-platform
adapters only deal with framework plumbing.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from typing import BinaryIO, ClassVar, TypeVar
from urllib.parse import urlparse

import aiohttp

from .. import (
    agent, backends_agent, colony, config, schedules, session, updater,
    workspace,
)
from ..agent_tools.owned_worker import run_owned_worker
from ..utils import await_cancelled
from ..utils import create_background_task
from ..utils import drain_queue as _drain_queue
from ..utils import ensure_lock
from ..utils import load_json_object
from ..utils import IMAGE_EXTENSIONS
from ..utils import parse_decimal_int
from ..utils import probe_image_dimensions
from ..utils import save_json_object
from ..utils import try_parse_int
from ..backends_agent.base import DetachedTaskRef, DetachedTaskRequest

logger = logging.getLogger(__name__)
performance_logger = logging.getLogger("Cozter.performance")

_FileResult = TypeVar("_FileResult")


async def _run_owned_poll_tasks(tasks: list[asyncio.Task[None]]) -> None:
    """Keep worker cleanup owned when shutdown is cancelled repeatedly."""
    if not tasks:
        return
    group = asyncio.gather(*tasks)
    interrupted: asyncio.CancelledError | None = None
    try:
        # wait() prevents parent cancellation from repeatedly cancelling a
        # provider's already-running process/RPC cleanup through gather().
        await asyncio.wait((group,))
        group.result()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        pending = {task for task in tasks if not task.done()}
        while pending:
            try:
                _, pending = await asyncio.wait(pending)
            except asyncio.CancelledError as error:
                interrupted = error
        # A child cancellation can finish gather before its siblings; all
        # siblings have now settled and its exception must still be consumed.
        with suppress(asyncio.CancelledError, Exception):
            group.result()
        if interrupted is not None:
            raise interrupted


async def _run_durable_file_operation(
    lock: asyncio.Lock, operation: Callable[[], _FileResult], *,
    on_complete: Callable[[_FileResult], None] | None = None,
    required_cleanup: bool = False,
) -> _FileResult:
    """Run file I/O off-loop, retaining lock ownership through cancellation.

    A queued operation can be cancelled before it acquires its file lock,
    except required cleanup of a session which has already been created.
    Once submitted, its executor future must physically finish before this
    coroutine unwinds any caller-owned admission or delivery locks. Apply
    in-memory commit bookkeeping on the event loop before replaying a cancel.
    """
    cancelled: asyncio.CancelledError | None = None
    while True:
        try:
            await lock.acquire()
            break
        except asyncio.CancelledError as error:
            if not required_cleanup:
                raise
            cancelled = error
    try:
        context = contextvars.copy_context()
        future = asyncio.get_running_loop().run_in_executor(
            None, context.run, operation,
        )
        while not future.done():
            try:
                # Cancelling asyncio.wait does not cancel its input future.
                await asyncio.wait((future,))
            except asyncio.CancelledError as error:
                cancelled = error
        try:
            result = future.result()
            if on_complete is not None:
                on_complete(result)
        except Exception:
            if cancelled is not None:
                logger.error(
                    "Durable file operation failed during cancellation",
                    exc_info=True,
                )
                raise cancelled
            raise
        if cancelled is not None:
            raise cancelled
        return result
    finally:
        lock.release()


_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".sh", ".bash", ".zsh", ".fish",
    ".html", ".htm", ".css", ".scss", ".xml", ".csv",
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".swift", ".kt", ".r", ".m",
    ".sql", ".graphql", ".proto",
    ".dockerfile", ".gitignore", ".env",
    ".log", ".diff", ".patch",
})
_INLINE_SIZE_LIMIT = 50_000


def _read_inline_text_attachment(path: str) -> str:
    """Read only enough text to decide whether an attachment can be inlined."""
    with open(path, encoding="utf-8", errors="replace") as file_handle:
        return file_handle.read(_INLINE_SIZE_LIMIT + 1)


UPLOADS_DIR = "uploads"
# Status previews must never block the stream or answer.
_STATUS_OPERATION_TIMEOUT_SEC = 4.0
_STATUS_EDIT_INTERVAL_SEC = 1.5
# Ack is cosmetic; bound it.
_INJECT_ACK_TIMEOUT_SEC = 8.0
_ATTACHMENT_TAIL_WAIT_SEC = 5.0
_ATTACHMENT_UPLOAD_CONCURRENCY = 3
_DETACHED_TASK_POLL_INTERVAL_SEC = 5.0
# A stalled provider or chat must leave capacity for unrelated owners.
_DURABLE_POLL_CONCURRENCY = 4
_DETACHED_TERMINAL_STATES = frozenset({"done", "failed", "stopped"})
# Precompiled for state save/load.
_UNSAFE_FILENAME_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
NO_WORKSPACE_TEXT = (
    "No workspace selected (or it was deleted). Use /new or /open."
)


def _record_with_required_string_fields(
    value: object, required: tuple[str, ...],
) -> dict | None:
    """Return a durable record only when all identity fields are strings."""
    if not isinstance(value, dict) or not all(
        isinstance(value.get(key), str) and value[key] for key in required
    ):
        return None
    return value


def ensure_upload_dir(workspace_path: str) -> str:
    """Return the workspace upload directory, creating it if needed."""
    return workspace.ensure_workspace_state_dir(workspace_path, UPLOADS_DIR)


@contextmanager
def reserve_upload_path(upload_dir: str, filename: str) -> Iterator[str]:
    """Reserve a collision-free upload path until its transfer completes.

    The placeholder is created exclusively so concurrent inbound transfers
    cannot pick the same destination.  A successful atomic writer replaces
    it; a failed transfer removes it along with its temporary output.
    """
    safe_name = os.path.basename(filename) or "file"
    stem, suffix = os.path.splitext(safe_name)
    index = 1

    while True:
        candidate = (
            safe_name if index == 1 else f"{stem} ({index}){suffix}"
        )
        local_path = os.path.join(upload_dir, candidate)
        try:
            descriptor = os.open(
                local_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600,
            )
        except FileExistsError:
            index += 1
            continue
        else:
            os.close(descriptor)
            break

    completed = False
    try:
        yield local_path
        completed = True
    finally:
        if not completed:
            try:
                os.unlink(local_path)
            except FileNotFoundError:
                pass


def attachment_kind_from_mime(value: object) -> str:
    """Map a transport MIME value to Cozter's coarse attachment labels."""
    mime = str(value or "").lower()
    if mime.startswith("image/"):
        return "photo"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"
    return "document"


class UploadTooLargeError(ValueError):
    """Raised before an attachment can exceed the configured size cap."""

    def __init__(self, max_upload_bytes: int):
        self.max_upload_bytes = max_upload_bytes
        super().__init__(upload_limit_message(max_upload_bytes))


def upload_size_exceeds_limit(size: object, max_upload_bytes: int) -> bool:
    """Whether a trustworthy byte count is above the configured cap.

    Transport metadata is optional, so an unknown or malformed value cannot
    be rejected here. Download/copy code still measures streamed data where
    a transport does not provide a usable size upfront.
    """
    if (
        not isinstance(max_upload_bytes, int)
        or isinstance(max_upload_bytes, bool)
    ):
        return False
    return (
        isinstance(size, int)
        and not isinstance(size, bool)
        and size > max_upload_bytes
    )


def upload_limit_message(max_upload_bytes: int) -> str:
    """Short user-facing explanation for a rejected attachment."""
    if (
        not isinstance(max_upload_bytes, int)
        or isinstance(max_upload_bytes, bool)
    ):
        return "File is too large."
    return (
        "File is too large "
        f"(maximum upload size: {max_upload_bytes:,} bytes)."
    )


@contextmanager
def atomic_output_file(local_path: str) -> Iterator[BinaryIO]:
    """Yield a temporary binary output file, then atomically replace *local_path*.

    Attachment adapters all need the same guarantee: a failed download or
    copy must leave neither a partial destination nor a temporary file
    behind.  Keeping that lifecycle here prevents the platform-specific
    transports from drifting apart.
    """
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=os.path.dirname(os.path.abspath(local_path)),
            prefix=f".{os.path.basename(local_path)}.",
            delete=False,
        ) as output:
            temp_path = output.name
            yield output
        os.replace(temp_path, local_path)
        temp_path = ""
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


async def write_limited_async_stream(
    chunks: AsyncIterator[bytes],
    local_path: str,
    max_upload_bytes: int,
) -> None:
    """Atomically write an async byte stream without exceeding a byte cap."""
    written = 0
    with atomic_output_file(local_path) as output:
        async for chunk in chunks:
            written += len(chunk)
            if upload_size_exceeds_limit(written, max_upload_bytes):
                raise UploadTooLargeError(max_upload_bytes)
            output.write(chunk)


def is_http_url(url: str) -> bool:
    """Whether *url* uses one of the network schemes attachment adapters allow."""
    return urlparse(url).scheme in ("http", "https")


async def download_http_file(
    url: str,
    local_path: str,
    max_upload_bytes: int,
    *,
    headers: dict[str, str] | None = None,
) -> None:
    """Download an HTTP(S) attachment atomically while enforcing its cap.

    Platform adapters supply any provider-specific request headers, while
    this shared transport path consistently validates the scheme, handles
    HTTP errors, rejects known oversize files, and cleans up partial streams.
    """
    if not is_http_url(url):
        raise ValueError(f"Refusing to fetch non-http url: {url!r}")

    async with (
        aiohttp.ClientSession() as http_session,
        http_session.get(url, headers=headers) as response,
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


def copy_file_with_limit(
    source_path: str,
    local_path: str,
    max_upload_bytes: int,
) -> None:
    """Atomically copy a local file without exceeding a byte cap."""
    if upload_size_exceeds_limit(
        os.path.getsize(source_path), max_upload_bytes,
    ):
        raise UploadTooLargeError(max_upload_bytes)

    with (
        open(source_path, "rb") as source,
        atomic_output_file(local_path) as output,
    ):
        copied = 0
        while chunk := source.read(64 * 1024):
            copied += len(chunk)
            if upload_size_exceeds_limit(copied, max_upload_bytes):
                raise UploadTooLargeError(max_upload_bytes)
            output.write(chunk)


async def copy_file_with_limit_async(
    source_path: str,
    local_path: str,
    max_upload_bytes: int,
) -> None:
    """Copy off-loop, settling the writer before caller-owned path cleanup.

    Cancelling ``to_thread`` would leave its worker running after an upload
    reservation removed the destination. The atomic writer could then recreate
    the cancelled file. Delay cancellation until the copy physically finishes
    so the reservation also removes any completed output.
    """
    await _run_durable_file_operation(
        asyncio.Lock(),
        functools.partial(
            copy_file_with_limit, source_path, local_path, max_upload_bytes,
        ),
    )


def write_bytes_atomically(local_path: str, data: bytes) -> None:
    """Atomically write already-bounded binary attachment data."""
    with atomic_output_file(local_path) as output:
        output.write(data)


# Message handle + attachments

@dataclass
class MessageHandle:
    """A reference to a sent message that can be edited or deleted later."""
    chat_id: str
    message_id: str


class _RecoverableTurnStatus:
    """Own an editable status without letting slow creation block agent work."""

    def __init__(self, platform: BotPlatform, uid: str, chat_id: str) -> None:
        self.platform = platform
        self.uid = uid
        self.chat_id = chat_id
        self.handle: MessageHandle | None = None
        self.pending: str | None = None
        self.lines: list[str] = []
        self.latest_text = ""
        self.updates = 0
        self.last_edit = 0.0
        self.next_post = 0.0
        self.closed = False
        self.wake = asyncio.Event()
        self.initial: asyncio.Future[MessageHandle | None] = asyncio.get_running_loop().create_future()
        self.worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        started = time.monotonic()
        self.worker = create_background_task(
            self._run(), name=f"{self.platform.platform_id}:status-owner:{self.uid}", log=logger,
        )
        # wait() leaves creation owned by this turn after its short foreground
        # allowance. A late acknowledgement can still supply the edit handle.
        await asyncio.wait((self.initial,), timeout=_STATUS_OPERATION_TIMEOUT_SEC)
        outcome = "pending"
        if self.initial.done():
            outcome = "posted" if self.initial.result() is not None else "no_handle"
        performance_logger.info(
            "stage=thinking_status platform=%s outcome=%s elapsed_ms=%.3f",
            self.platform.platform_id, outcome, (time.monotonic() - started) * 1_000,
        )

    def on_event(self, ev: agent.ChatEvent) -> None:
        if self.closed:
            return
        if ev.kind == "text":
            if ev.content.strip():
                self.latest_text = ev.content
        elif ev.kind in ("tool", "file"):
            content = ev.content.split("\n")[0] if ev.kind == "tool" else ev.content
            clipped = (content[:80 - len("… [clipped]")] + "… [clipped]"
                       if len(content) > 80 else content)
            self.lines.append(f"» {clipped}")
        else:
            return
        self.pending = self.platform._compose_thinking_display(self.lines, self.latest_text)
        self.wake.set()

    async def _wait_until(self, deadline: float) -> None:
        loop = asyncio.get_running_loop()
        while (delay := deadline - loop.time()) > 0:
            await asyncio.sleep(max(delay, 2 * time.get_clock_info("monotonic").resolution))

    async def _post(self, text: str, reason: str) -> bool:
        started = time.monotonic()
        result = await self.platform._run_status_operation(
            self.platform.send_text(self.chat_id, text),
            action="posting a recoverable progress status",
            timeout=self.platform.status_initial_timeout_sec,
        )
        loop = asyncio.get_running_loop()
        assert self.platform.status_recovery_interval_sec is not None
        self.next_post = loop.time() + self.platform.status_recovery_interval_sec
        success = isinstance(result, MessageHandle)
        performance_logger.info(
            "stage=status_post platform=%s reason=%s outcome=%s elapsed_ms=%.3f",
            self.platform.platform_id, reason, "posted" if success else "no_handle",
            (time.monotonic() - started) * 1_000,
        )
        if isinstance(result, MessageHandle):
            old_handle = self.handle
            self.handle = result
            self.updates = 0
            self.last_edit = loop.time()
            if old_handle is not None and old_handle != result:
                self._delete(old_handle)
        return success

    def _delete(self, handle: MessageHandle) -> None:
        create_background_task(
            self.platform._run_status_operation(
                self.platform.delete_message(handle), action="deleting the Thinking status",
            ),
            name=f"{self.platform.platform_id}:status-delete:{self.uid}", log=logger,
        )

    async def _run(self) -> None:
        try:
            await self._post("Thinking...", "initial")
            self.initial.set_result(self.handle)
            while not self.closed:
                if self.pending is None:
                    await self.wake.wait()
                    self.wake.clear()
                    continue
                limit = self.platform.status_update_limit
                if self.handle is None or (limit is not None and self.updates >= limit):
                    await self._wait_until(self.next_post)
                    display, self.pending = self.pending, None
                    if not await self._post(display, "recovery" if self.handle is None else "rotation"):
                        # Retry the latest real activity even if the provider
                        # becomes quiet while status transport is unavailable.
                        if self.pending is None:
                            self.pending = display
                    continue
                await self._wait_until(self.last_edit + max(
                    _STATUS_EDIT_INTERVAL_SEC, self.platform.status_update_interval_sec,
                ))
                display, self.pending = self.pending, None
                self.updates += 1  # Ambiguous failures consume the per-message cap.

                async def edit() -> bool:
                    assert self.handle is not None
                    await self.platform.edit_text(self.handle, display)
                    return True

                succeeded = await self.platform._run_status_operation(
                    edit(), action="updating the progress status",
                )
                self.last_edit = asyncio.get_running_loop().time()
                performance_logger.info(
                    "stage=status_edit platform=%s outcome=%s attempt=%d",
                    self.platform.platform_id, "ok" if succeeded else "unavailable", self.updates,
                )
                if not succeeded and self.pending is None:
                    self.pending = display
        finally:
            if not self.initial.done():
                self.initial.set_result(self.handle)

    async def close(self) -> None:
        self.closed = True  # Close before yielding, so no event can reopen it.
        self.pending = None
        interrupted = False
        if self.worker is not None:
            self.worker.cancel()
            while not self.worker.done():
                try:
                    await asyncio.wait((self.worker,))
                except asyncio.CancelledError:
                    # Settle the owned send/edit even under repeated /stop.
                    interrupted = True
        if not self.initial.done():
            self.initial.set_result(self.handle)
        if self.handle is not None:
            self._delete(self.handle)
        if interrupted:
            raise asyncio.CancelledError


@dataclass
class AttachmentInfo:
    """An incoming file upload already downloaded to the workspace."""
    local_path: str
    filename: str
    kind: str  # "document", "photo", "audio", "video", "voice", "video note"
    caption: str = ""


# Per-event facade for command handlers.

@dataclass
class BotContext:
    """Per-event context passed to command handlers.

    All identifiers are stringified so the same dicts/logic work on both
    Telegram (int ids) and Slack (U... string ids).
    """
    user_id: str
    chat_id: str
    text: str
    command: str | None
    args: str
    attachment: AttachmentInfo | None
    platform: BotPlatform
    # Set when this command cleared a pending input flow.
    had_pending: bool = False

    async def reply_text(
        self, text: str, *, rich: bool = False,
    ) -> MessageHandle | None:
        return await self.platform.send_text(
            self.chat_id, text, rich=rich,
        )


# Handler: callback taking a BotContext.
Handler = Callable[[BotContext], Awaitable[None]]
QueueEntry = tuple[str, str, str, bool]


# Cap for the redacted per-turn progress label kept for stuck diagnostics.
_TURN_PROGRESS_LABEL_LIMIT = 80


def _redact_progress_label(value: object, *, limit: int = _TURN_PROGRESS_LABEL_LIMIT) -> str:
    """Return a non-sensitive one-line progress label for stuck diagnostics.

    Only the first line is kept (multi-line output bodies are dropped),
    clipped to *limit* chars with a visible marker. Callers must pass a
    label they constructed from non-sensitive parts (tool/file *kind*
    plus a tool *name*, never prompts, arguments, command text, file
    contents, or credentials).
    """
    text = value if isinstance(value, str) else str(value)
    first_line = text.split("\n")[0].strip()
    if len(first_line) <= limit:
        return first_line
    marker = "… [clipped]"
    if limit <= len(marker):
        return first_line[:max(0, limit - 1)] + "…" if limit > 1 else "…"[:limit]
    return first_line[:limit - len(marker)] + marker


@dataclass
class _TurnProgress:
    """Non-sensitive attribution for one in-flight agent turn.

    ``backend``/``model`` name the agent driving the turn; ``started``
    is a monotonic timestamp for the turn start; ``last_kind`` is the
    last observed event kind (``"tool"`` / ``"file"`` / ...); ``label``
    is a redacted first-line label built from non-sensitive parts only
    (tool name, never arguments or command text); ``event_count`` counts
    observed progress events; ``last_at`` is the monotonic timestamp of
    the most recent one; ``pid`` is the backend subprocess id when the
    backend runs as an OS process (``None`` for in-process HTTP agents
    and for phases that have not launched a process).
    """

    backend: str = ""
    model: str = ""
    started: float = 0.0
    last_kind: str = ""
    label: str = ""
    event_count: int = 0
    last_at: float = 0.0
    pid: int | None = None


class _InjectQueue(asyncio.Queue[str]):
    """A turn-owned queue that can atomically stop accepting injections.

    ``/inject`` must never acknowledge text after the agent has produced its
    final answer.  Both ``put_if_active`` and ``close`` are synchronous, so
    on the event loop there is no await-point between checking the window and
    enqueueing (or closing) it.
    """
    def __init__(self, maxsize: int) -> None:
        super().__init__(maxsize=maxsize)
        self._accepting = True

    def put_if_active(self, text: str) -> str:
        """Queue *text* or return ``finished`` / ``full`` without waiting."""
        if not self._accepting:
            return "finished"
        if self.full():
            return "full"
        self.put_nowait(text)
        return "accepted"

    def close(self) -> None:
        """Prevent any later ``/inject`` from being acknowledged."""
        self._accepting = False


# Shared state and command logic.

class BotPlatform(ABC):
    """Base class: holds all command logic, leaves I/O to subclasses."""

    supports_transient_status: ClassVar[bool] = True
    # Platform-specific limits supplement the shared preview throttle.
    status_update_interval_sec: ClassVar[float] = 0.0
    status_update_limit: ClassVar[int | None] = None
    status_fallback_messages: ClassVar[bool] = True
    # Opt in to owned late creation, recovery and per-message edit rotation.
    status_initial_timeout_sec: ClassVar[float | None] = None
    status_recovery_interval_sec: ClassVar[float | None] = None

    def __init__(
        self,
        notify_targets: list[str],
        *,
        recent_limit: int = config.DEFAULT_RECENT_WORKSPACE_LIMIT,
        max_queue_size: int = config.DEFAULT_MESSAGE_QUEUE_SIZE,
        max_upload_bytes: int = config.DEFAULT_MAX_UPLOAD_BYTES,
    ):
        # Startup-greet + default auth set.
        self.notify_targets: list[str] = [str(target) for target in notify_targets]
        self.recent_limit = recent_limit
        self.max_queue_size = max_queue_size
        self.max_upload_bytes = (
            max_upload_bytes
            if isinstance(max_upload_bytes, int)
            and not isinstance(max_upload_bytes, bool)
            and max_upload_bytes > 0
            else config.DEFAULT_MAX_UPLOAD_BYTES
        )

        # Per-user runtime state.
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._task_locks: dict[str, asyncio.Lock] = {}
        self._message_queues: dict[str, asyncio.Queue] = {}
        self._inject_queues: dict[str, _InjectQueue] = {}
        # Users awaiting an answer: queue drain paused.
        self._awaiting_answer: set[str] = set()
        # An unfinished job needing input blocks every later entry, including
        # scheduled work. Its checkpoint is also retained in the queue file.
        self._paused_jobs: set[str] = set()
        # Multi-step flow state.
        self._pending_input: dict[str, Handler] = {}
        # Scheduler state.
        self._scheduler_task: asyncio.Task | None = None
        # Update pulled; waiting for active replies before restart.
        self._update_restart_pending = False
        # Serializes persistent-queue read-modify-write.
        self._queue_file_lock: asyncio.Lock = asyncio.Lock()
        # Staged final text for retry without rerunning.
        self._reply_delivery_file_lock: asyncio.Lock = asyncio.Lock()
        # Per-user delivery-vs-cancel lock.
        self._reply_delivery_locks: dict[str, asyncio.Lock] = {}
        # Recovery poll classes run independently; outbound chunks for one
        # owner still form one ordered delivery rather than interleaving.
        self._recovery_delivery_locks: dict[str, asyncio.Lock] = {}
        self._reply_delivery_tasks: dict[str, set[asyncio.Task]] = {}
        self._pending_reply_delivery_users: set[str] = set()
        # Stop/cancel can overlap while durable file operations settle.
        self._cancel_work_depths: dict[str, int] = {}
        # Pipelined attachment uploads: per-chat tails.
        self._attachment_tasks: dict[str, set[asyncio.Task]] = {}
        self._attachment_tails: dict[str, asyncio.Task] = {}
        # Detached tasks: results awaiting delivery, not prompts.
        self._detached_task_file_lock: asyncio.Lock = asyncio.Lock()
        self._detached_task_watcher: asyncio.Task | None = None
        self._detached_delivery_locks: dict[str, asyncio.Lock] = {}
        self._detached_delivery_tasks: dict[str, set[asyncio.Task]] = {}
        # Pending explicit launches remain independently owned during foreground turns.
        self._detached_launch_tasks: dict[str, set[asyncio.Task]] = {}
        # Non-sensitive per-turn progress for stuck-wait attribution
        # (backend/model, timestamps, tool/file kinds, redacted label, PIDs).
        self._turn_progress: dict[str, _TurnProgress] = {}
        # Users already acknowledged by /cancel or /stop.
        self._cancel_acknowledged: set[str] = set()
        # Per-user epochs.
        self._cancel_generations: dict[str, int] = {}
        # Per-user admission lock.
        self._dispatch_admission_locks: dict[str, asyncio.Lock] = {}

    def _check_upload_size(self, size: object) -> None:
        """Raise before accepting or sending a known-oversized file."""
        if upload_size_exceeds_limit(size, self.max_upload_bytes):
            raise UploadTooLargeError(self.max_upload_bytes)

    async def _attachment_upload_dir(self, ctx: BotContext) -> str | None:
        """Return a safe upload directory, replying when no workspace exists."""
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws or not os.path.isdir(ws):
            await ctx.reply_text(NO_WORKSPACE_TEXT)
            return None
        try:
            return ensure_upload_dir(ws)
        except (OSError, ValueError) as exc:
            await ctx.reply_text(f"Workspace state is unsafe: {exc}")
            return None

    def _check_upload_path(self, path: str) -> None:
        """Raise before an outbound adapter hands a large file to its API."""
        self._check_upload_size(os.path.getsize(path))

    def make_context(
        self,
        user_id: str | int,
        chat_id: str | int,
        *,
        text: str = "",
        command: str | None = None,
        args: str = "",
        attachment: AttachmentInfo | None = None,
    ) -> BotContext:
        """Build the common command context used by platform adapters."""
        return BotContext(
            user_id=str(user_id),
            chat_id=str(chat_id),
            text=text,
            command=command,
            args=args,
            attachment=attachment,
            platform=self,
        )

    # identity + I/O primitives

    @property
    @abstractmethod
    def platform_id(self) -> str:
        """Stable id used to scope workspace/session state to this bot."""

    @abstractmethod
    async def start(self) -> None:
        """Start listening for events."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop listening and shut down."""

    @abstractmethod
    async def send_text(
        self, chat_id: str, text: str, *, rich: bool = False,
    ) -> MessageHandle | None:
        """Send a message; return a handle for later edit/delete if supported."""

    @abstractmethod
    async def edit_text(
        self, handle: MessageHandle, text: str, *, rich: bool = False,
    ) -> None:
        """Edit a previously sent message."""

    @abstractmethod
    async def delete_message(self, handle: MessageHandle) -> None:
        """Delete a previously sent message."""

    @abstractmethod
    async def send_file(self, chat_id: str, path: str) -> None:
        """Upload a file to the chat."""

    async def notify_users(self, message: str) -> None:
        """Best-effort broadcast of *message* to every notify target."""
        for target in self.notify_targets:
            try:
                await self.send_text(target, message)
            except Exception as e:
                logger.warning("Failed to notify %s: %s", target, e)

    async def send_status(self, chat_id: str, text: str) -> None:
        """Emit a transient progress/status message ("» <tool>", etc.).

        Default: delegate to ``send_text``. Platforms that can render
        these visually distinct from the final reply (e.g. CLI mode
        with ANSI dim grey) should override.
        """
        await self.send_text(chat_id, text)

    async def _start_daemon_services(self) -> None:
        """Restore durable state and start services used by daemon bots.

        Telegram, Slack, and Signal all persist queues, schedule recurring
        prompts, and poll detached provider tasks. CLI mode deliberately
        skips the first two, so its lifecycle remains separate.
        """
        await self.restore_queues()
        self.start_detached_task_watcher()
        await self.start_scheduler()

    async def _stop_daemon_services(self) -> None:
        """Stop the background services started by _start_daemon_services.

        Owned durable-reply retries keep the staged ledger durable, so a
        blocked send must be cancelled (not awaited) here: restart
        delivery resumes from the retained record via
        ``restore_reply_deliveries``.
        """
        for owner in list(self._detached_launch_tasks):
            await self._cancel_owned_deliveries(owner, self._detached_launch_tasks)
        await self.stop_detached_task_watcher()
        await self.stop_scheduler()
        for owner in list(self._attachment_tasks):
            await self._cancel_attachment_uploads(owner)
        for owner in list(self._reply_delivery_tasks):
            await self._cancel_owned_deliveries(owner, self._reply_delivery_tasks)

    async def _send_text_best_effort(
        self, chat_id: str, text: str, *, rich: bool = False,
    ) -> bool:
        """Send text and return whether it succeeded, swallowing I/O errors."""
        try:
            await self.send_text(chat_id, text, rich=rich)
        except Exception:
            return False
        return True

    async def _run_status_operation(
        self, operation: Awaitable[object], *, action: str, timeout: float | None = None,
    ) -> object | None:
        """Run noncritical status I/O without delaying a chat reply.

        Thinking previews can be posted, edited, or deleted independently of
        the answer. A transient platform API stall should therefore be bounded
        and best-effort rather than backpressure the agent event stream.
        """
        limit = _STATUS_OPERATION_TIMEOUT_SEC if timeout is None else timeout
        try:
            return await asyncio.wait_for(
                operation, timeout=limit,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            logger.debug(
                "Timed out %s after %.1fs",
                action,
                limit,
            )
        except Exception:
            logger.debug("Failed %s", action, exc_info=True)
        return None

    def _start_queue_drain(self, uid: str) -> None:
        if uid in self._paused_jobs:
            return
        create_background_task(
            self._drain_message_queue(uid),
            name=f"{self.platform_id}:drain:{uid}",
            log=logger,
        )

    async def send_startup_messages(
        self, version: str, commit_date: str,
    ) -> None:
        """Announce startup to each notify target.

        Default: generic "Cozter started" broadcast. Platforms where the
        notify target is a specific user (Telegram) may override to
        include per-user workspace info.
        """
        msg = (
            f"Cozter started.\nVersion: {version}\nUpdated: {commit_date}"
        )
        await self.notify_users(msg)

    async def begin_update_restart(self) -> None:
        """Pause new AI turns while an auto-update restart is pending."""
        self._update_restart_pending = True
        await self.stop_scheduler()

    async def cancel_update_restart(self) -> None:
        """Resume normal processing if an update restart is aborted."""
        self._update_restart_pending = False
        await self.start_scheduler()
        for uid in list(self._message_queues):
            self._start_queue_drain(uid)

    def has_active_turns(self) -> bool:
        """Return True while a reply or cancellation cleanup is in progress.

        Standalone durable-reply retries (the watcher/restart path that
        re-sends an already staged reply without holding the per-user
        turn lock) own their sends in ``_reply_delivery_tasks``. They
        count here so an update restart waits for the in-flight send
        instead of racing ahead of it; the staged ledger stays durable
        either way. Explicit ``/cancel``/``/stop`` still interrupts
        them via ``_clear_reply_deliveries``.
        Cancellation also stays active while its durable cleanup settles.
        """
        if self._cancel_work_depths:
            return True
        if any(not task.done() for task in self._running_tasks.values()):
            return True
        if any(not task.done() for owned in self._detached_launch_tasks.values() for task in owned):
            return True
        tasks = self._attachment_tasks
        if any(
            not task.done() for owned in tasks.values() for task in owned
        ):
            return True
        deliveries = self._reply_delivery_tasks
        if any(
            not task.done() for owned in deliveries.values() for task in owned
        ):
            return True
        return any(lock.locked() for lock in self._task_locks.values())

    def stuck_turn_diagnostics(self) -> str:
        """Human-readable dump of turn-tracking state, for stuck waits.

        Called when the update loop's idle wait blows past its ceiling
        (``update_idle_timeout``) so the critical log names the exact
        uid(s) still holding state, plus whether it's a not-yet-done
        task, a held lock, or both — the key evidence for diagnosing a
        wedged ``has_active_turns()`` that would otherwise only surface
        as repeated ``Delaying update check`` lines.

        Active turns also report their non-sensitive attribution
        (backend/model, backend PID when the backend runs as an OS
        process, elapsed seconds since the turn started, seconds since
        the last progress event, and the redacted last-progress label),
        so a historical command-output wait can be attributed to the
        tool/backend/PID it was waiting on without logging prompts,
        arguments, or credentials.
        """
        try:
            now = asyncio.get_running_loop().time()
        except RuntimeError:
            now = time.monotonic()
        tasks = {
            uid: task for uid, task in self._running_tasks.items()
            if not task.done()
        }
        locks = {
            uid for uid, lock in self._task_locks.items() if lock.locked()
        }
        parts: list[str] = []
        if tasks:
            # Report uid + task state + non-sensitive turn attribution.
            names = [
                self._describe_running_task(uid, task, now)
                for uid, task in tasks.items()
            ]
            parts.append("running_tasks=[" + ", ".join(names) + "]")
        pending_uploads = {
            owner: sum(1 for task in owned if not task.done())
            for owner, owned in self._attachment_tasks.items()
        }
        pending_uploads = {
            owner: count for owner, count in pending_uploads.items()
            if count
        }
        if pending_uploads:
            uploads = [
                f"{owner}({count})"
                for owner, count in sorted(pending_uploads.items())
            ]
            parts.append("pending_uploads=[" + ", ".join(uploads) + "]")
        pending_deliveries = {
            owner: sum(1 for task in owned if not task.done())
            for owner, owned in self._reply_delivery_tasks.items()
        }
        pending_deliveries = {
            owner: count for owner, count in pending_deliveries.items()
            if count
        }
        if pending_deliveries:
            deliveries = [
                f"{owner}({count})"
                for owner, count in sorted(pending_deliveries.items())
            ]
            parts.append("pending_deliveries=[" + ", ".join(deliveries) + "]")
        if locks:
            parts.append("held_locks=[" + ", ".join(sorted(locks)) + "]")
        return "; ".join(parts) or "<no stuck state found>"

    def _describe_running_task(
        self, uid: str, task: asyncio.Task, now: float,
    ) -> str:
        """Render one running turn with its non-sensitive attribution.

        Falls back to the legacy ``uid(cancelled=...)`` shape when no
        progress record exists (turns started before tracking, or tests
        that register ``_running_tasks`` directly).
        """
        base = f"{uid}(cancelled={task.cancelled()})"
        progress = self._turn_progress.get(uid)
        if progress is None:
            return base
        elapsed = max(0.0, now - progress.started) if progress.started else 0.0
        detail = f"backend={progress.backend or '?'}"
        detail += f" model={progress.model or '?'}"
        detail += f" pid={progress.pid if progress.pid is not None else '?'}"
        detail += f" elapsed={elapsed:.1f}s"
        if progress.event_count:
            idle = max(0.0, now - progress.last_at) if progress.last_at else 0.0
            detail += (
                f" last={progress.last_kind or '?'}"
                f"({progress.event_count} events,"
                f" idle={idle:.1f}s)"
            )
            if progress.label:
                detail += f" {progress.label}"
        else:
            detail += " last=<no progress events yet>"
        return f"{base}[{detail}]"

    def _begin_turn_progress(
        self, uid: str, *, backend: str, model: str,
    ) -> None:
        """Start (or restart) the non-sensitive progress record for *uid*."""
        try:
            started = asyncio.get_running_loop().time()
        except RuntimeError:
            started = time.monotonic()
        self._turn_progress[uid] = _TurnProgress(
            backend=backend if isinstance(backend, str) else str(backend),
            model=model if isinstance(model, str) else str(model),
            started=started,
            last_at=started,
        )

    def _note_turn_process(self, uid: str, pid: object) -> None:
        """Record the backend subprocess id for *uid* when it is numeric.

        In-process HTTP agents report ``pid=-1`` (``HttpAgentProcess``);
        that sentinel — and any non-integer value — is stored as ``None``
        so diagnostics render ``pid=?`` instead of a bogus id.
        """
        progress = self._turn_progress.get(uid)
        if progress is None:
            return
        if isinstance(pid, bool):
            return
        if isinstance(pid, int) and pid > 0:
            progress.pid = pid
        elif pid is None:
            progress.pid = None

    def record_turn_progress(self, uid: str, *, kind: str, label: str) -> None:
        """Record one non-sensitive progress event for *uid*'s turn.

        *kind* is the event kind (``"tool"`` / ``"file"`` / ...);
        *label* must already be redacted by the caller — a tool/file
        *name*, never prompts, arguments, command text, file contents,
        or credentials. Unknown/empty kinds are ignored so a malformed
        event can never poison the attribution record.
        """
        progress = self._turn_progress.get(uid)
        if progress is None:
            return
        if not isinstance(kind, str) or not kind.strip():
            return
        try:
            now = asyncio.get_running_loop().time()
        except RuntimeError:
            now = time.monotonic()
        progress.last_kind = kind.strip()
        progress.label = _redact_progress_label(label)
        progress.event_count += 1
        progress.last_at = now

    def _clear_turn_progress(self, uid: str) -> None:
        """Drop the attribution record for *uid* (turn finished/cancelled)."""
        self._turn_progress.pop(uid, None)

    def _record_turn_event_progress(self, uid: str, ev: object) -> None:
        """Fold one streamed event into *uid*'s attribution record.

        Only the event *kind* plus a redacted label are kept. The label
        is the leading tool-name token of the event's first line
        (``"read_file: ..."`` → ``"tool: read_file"``;
        ``"$ ..."`` shell lines → ``"tool: shell"``), so arguments,
        command text, paths, prompts, file contents, and credentials
        are never stored. ``file`` events record ``"file: change"``
        without any path; every other kind records just its kind.
        """
        kind = getattr(ev, "kind", "")
        content = getattr(ev, "content", "")
        if not isinstance(kind, str) or not kind.strip():
            return
        kind = kind.strip()
        label = kind
        if kind == "tool" and isinstance(content, str) and content.strip():
            first_line = content.strip().split("\n")[0].strip()
            # Status-preview lines carry a "» " marker; drop it.
            if first_line.startswith("»"):
                first_line = first_line[1:].strip()
            if first_line.startswith("$"):
                label = "tool: shell"
            else:
                token = first_line.split(":")[0].split(" ")[0].split("(")[0].strip()
                # Keep only plausible bare tool/status names.
                if token and len(token) <= 64 and token.replace(
                    "_", "").replace("-", "").isalnum():
                    label = f"tool: {token}"
                else:
                    label = "tool"
        elif kind == "file":
            label = "file: change"
        self.record_turn_progress(uid, kind=kind, label=label)

    def authorized(self, user_id: str, _chat_id: str) -> bool:
        """Return True if an event from *(user_id, chat_id)* is allowed.

        Default scopes by user_id (Telegram semantics). Platforms that
        scope by channel instead (Slack) override to check *chat_id*.
        """
        return str(user_id) in self.notify_targets

    async def dispatch_command(self, ctx: BotContext) -> None:
        """Entry point for slash commands."""
        if not self.authorized(ctx.user_id, ctx.chat_id):
            logger.warning(
                "Unauthorized command from user=%s chat=%s",
                ctx.user_id, ctx.chat_id,
            )
            return
        handler = self._COMMANDS.get((ctx.command or "").lower())
        if handler is None:
            pending = self._pending_input.pop(ctx.user_id, None)
            if pending is not None:
                ctx.text = "/" + (ctx.command or "")
                if ctx.args:
                    ctx.text += f" {ctx.args}"
                ctx.command = None
                ctx.args = ""
                await pending(ctx)
                return
            await ctx.reply_text(
                f"Unknown command: /{ctx.command}."
                f" Available: /{' /'.join(sorted(self._COMMANDS))}"
            )
            return
        # Any new command cancels a pending flow.
        ctx.had_pending = (
            self._pending_input.pop(ctx.user_id, None) is not None
        )
        await handler(self, ctx)

    async def dispatch_text(self, ctx: BotContext) -> None:
        """Entry point for plain text and backslash command aliases."""
        if not self.authorized(ctx.user_id, ctx.chat_id):
            return
        if not ctx.text or not ctx.text.strip():
            return

        # Accept ``\open`` etc. as aliases (known names only).
        text = ctx.text.strip()
        if text.startswith("\\"):
            parts = text[1:].split(None, 1)
            command = parts[0].split("@", 1)[0] if parts else ""
            if command.lower() in self._COMMANDS:
                ctx.text = ""
                ctx.command = command
                ctx.args = parts[1] if len(parts) > 1 else ""
                await self.dispatch_command(ctx)
                return

        pending = self._pending_input.pop(ctx.user_id, None)
        if pending is not None:
            # Multi-step flow continuation.
            await pending(ctx)
            return
        await self._ai_chat(ctx)

    async def dispatch_file(self, ctx: BotContext) -> None:
        """Entry point for file/attachment uploads."""
        if not self.authorized(ctx.user_id, ctx.chat_id):
            return
        if ctx.attachment is None:
            return
        # File uploads cancel pending flows.
        self._pending_input.pop(ctx.user_id, None)
        await self._ai_file(ctx)

    def _expect_input(self, user_id: str, callback: Handler) -> None:
        """Arm a one-shot text-input handler for *user_id*."""
        self._pending_input[user_id] = callback

    def _ensure_task_lock(self, uid: str) -> asyncio.Lock:
        return ensure_lock(self._task_locks, uid)

    def _ensure_dispatch_admission_lock(self, uid: str) -> asyncio.Lock:
        """Return the per-user barrier for durable inbound queue writes."""
        return ensure_lock(self._dispatch_admission_locks, uid)

    def _ensure_message_queue(
        self, uid: str, *, min_size: int = 0,
    ) -> asyncio.Queue:
        msg_queue = self._message_queues.get(uid)
        maxsize = max(self.max_queue_size, min_size)
        if msg_queue is None:
            msg_queue = asyncio.Queue(maxsize=maxsize)
            self._message_queues[uid] = msg_queue
        elif msg_queue.maxsize != maxsize:
            # Keep queue object and entries: asyncio.Queue has no public
            # resize API; a restored backlog may exceed today's cap.
            msg_queue._maxsize = maxsize
            # A growth can make space without a get(), so wake blocked putters
            # to recheck full(). Each still checks capacity before admission.
            if not msg_queue.full():
                while msg_queue._putters:
                    msg_queue._wakeup_next(msg_queue._putters)
        return msg_queue

    @staticmethod
    def _has_pending_normal_entries(msg_queue: asyncio.Queue) -> bool:
        """True if *msg_queue* holds at least one non-ephemeral queued entry.

        ``QueueEntry`` is ``(text, chat_id, entry_id, ephemeral)``.
        Used by :meth:`_send_result` to avoid arming a ``[[await]]``
        pause when the user already sent follow-up messages during the
        turn, which would strand that backlog forever.
        """
        for entry in msg_queue._queue:  # type: ignore[attr-defined]
            # Only well-formed non-ephemeral entries count.
            if (
                isinstance(entry, tuple)
                and len(entry) == 4
                and entry[3] is False
            ):
                return True
        return False

    @staticmethod
    def _select_queue_entry(
        msg_queue: asyncio.Queue,
        predicate: Callable[[QueueEntry], bool],
        *,
        requeue_selected: bool = False,
    ) -> QueueEntry | None:
        """Select the first matching entry while preserving FIFO order.

        When *requeue_selected* is true, restore the match ahead of every
        remaining entry. This makes the same stable queue partition usable
        for both selecting a runnable entry and promoting a user's answer.
        """
        selected: QueueEntry | None = None
        buffered: list[QueueEntry] = []
        while not msg_queue.empty():
            entry = msg_queue.get_nowait()
            if selected is None and predicate(entry):
                selected = entry
            else:
                buffered.append(entry)

        if selected is not None and requeue_selected:
            msg_queue.put_nowait(selected)
        for entry in buffered:
            msg_queue.put_nowait(entry)
        return selected

    @staticmethod
    def _pop_next_queue_entry(
        msg_queue: asyncio.Queue, *, ephemeral_only: bool = False,
    ) -> QueueEntry | None:
        """Pop the next runnable queue entry.

        When a collaborative turn is awaiting an answer, normal chat
        entries must remain paused, but scheduled ``/reserve`` entries
        are independent ephemeral runs and may continue.
        """
        if not ephemeral_only:
            return msg_queue.get_nowait()
        # Guard the shape: skip malformed entries.
        return BotPlatform._select_queue_entry(
            msg_queue,
            lambda entry: (
                isinstance(entry, tuple)
                and len(entry) == 4
                and entry[3] is True
            ),
        )

    @staticmethod
    def _promote_queue_entry(
        msg_queue: asyncio.Queue, entry_id: str,
    ) -> None:
        """Move a queued entry to the front, preserving all others."""
        BotPlatform._select_queue_entry(
            msg_queue,
            lambda entry: (
                isinstance(entry, tuple)
                and len(entry) == 4
                and isinstance(entry[2], str)
                and entry[2] == entry_id
            ),
            requeue_selected=True,
        )

    async def _persist_promote(
        self, uid: str, entry_id: str,
    ) -> None:
        """Move a persisted queue entry to the front for restart parity."""
        def promote() -> None:
            data = self._read_queue_file()
            entries = self._queue_entries(data.get(uid))
            selected: dict | None = None
            remaining: list[dict] = []
            for entry in entries:
                if selected is None and entry.get("id") == entry_id:
                    selected = entry
                else:
                    remaining.append(entry)
            if selected is None:
                return
            data[uid] = [selected, *remaining]
            self._write_queue_file(data)

        await _run_durable_file_operation(self._queue_file_lock, promote)

    async def _resume_awaiting_answer(
        self, uid: str, msg_queue: asyncio.Queue, entry_id: str, *,
        reason: str,
    ) -> bool:
        """Clear an answer pause and promote the answer before backlog."""
        if uid not in self._awaiting_answer:
            return False
        self._awaiting_answer.discard(uid)
        self._promote_queue_entry(msg_queue, entry_id)
        await self._persist_promote(uid, entry_id)
        logger.info("User %s answered while %s; queue resumed", uid, reason)
        return True

    async def cmd_start(self, ctx: BotContext) -> None:
        await ctx.reply_text("Cozter bot is running.")

    async def cmd_version(self, ctx: BotContext) -> None:
        ver, date = await asyncio.gather(
            asyncio.to_thread(updater.get_current_version),
            asyncio.to_thread(updater.get_last_commit_date),
        )
        await ctx.reply_text(f"Version: {ver}\nUpdated: {date}")

    async def cmd_doctor(self, ctx: BotContext) -> None:
        """Report readiness of every backend (CLI on PATH / server up)."""
        lines = ["Backend readiness:"]
        # Only direct backends have anything to probe.
        for name in backends_agent.DIRECT_BACKENDS:
            backend = backends_agent.get_backend(name)
            ok, detail = await asyncio.to_thread(backend.health_check)
            lines.append(f"  {'ok' if ok else 'XX'} {name}: {detail}")

        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if ws and os.path.isdir(ws):
            chat_backend = workspace.get_backend_name(ws)
            summary_backend = workspace.get_summary_backend_name(ws)
            lines.append("")
            lines.append(
                f"This workspace: chat={chat_backend},"
                f" summary={summary_backend}"
            )
            if chat_backend == workspace.FLEXIBLE_BACKEND:
                for tier, (name, model) in workspace.get_flexible_run_config(
                    ws,
                ).items():
                    lines.append(f"  flexible {tier:<4} = {name}/{model}")
        await ctx.reply_text("\n".join(lines))

    async def cmd_cancel(self, ctx: BotContext) -> None:
        if ctx.had_pending:
            await ctx.reply_text("Cancelled.")
            return

        task_running, cancelled_work = await self._cancel_user_work(ctx.user_id)
        if task_running or cancelled_work:
            await ctx.reply_text("Cancelled.")
            return
        await ctx.reply_text("Nothing to cancel.")

    async def _cancel_user_work(self, uid: str) -> tuple[bool, bool]:
        """Cancel a user's foreground turn and discard its queued work.

        Return whether a foreground task was running and whether any paused,
        queued, delayed, or detached work was removed. Both ``/cancel`` and
        ``/stop`` use the same cleanup; only their no-work reply differs.
        """
        # Advance pre-await so suspended dispatches notice /stop.
        self._cancel_generations[uid] = (
            self._cancel_generations.get(uid, 0) + 1
        )
        self._cancel_work_depths[uid] = self._cancel_work_depths.get(uid, 0) + 1
        try:
            # Signal foreground agents first.
            task = self._running_tasks.get(uid)
            task_running = task is not None and not task.done()
            if task_running:
                self._cancel_acknowledged.add(uid)
                task.cancel()

            launches = tuple(task for task in self._detached_launch_tasks.get(uid, ())
                             if not task.done())
            for launch in launches:
                launch.cancel()

            # Signal outbound work before durable cleanup yields to the loop.
            for owned_tasks in (self._reply_delivery_tasks, self._detached_delivery_tasks):
                for delivery in tuple(owned_tasks.get(uid, ())):
                    delivery.cancel()
            # A queued upload must not begin while a file commit is settling.
            uploads = tuple(
                task for task in self._attachment_tasks.get(uid, ())
                if not task.done()
            )
            for upload in uploads:
                upload.cancel()

            # Settle provider handoffs before clearing their possible ledger rows.
            await self._cancel_owned_deliveries(uid, self._detached_launch_tasks)

            # Order the bump + clear against persists.
            admission_lock = self._ensure_dispatch_admission_lock(uid)
            async with admission_lock:
                cancelled_uploads = max(
                    len(uploads), await self._cancel_attachment_uploads(uid),
                )
                was_awaiting = uid in self._awaiting_answer
                self._awaiting_answer.discard(uid)
                drained: list = []
                _drain_queue(self._message_queues.get(uid), collect=drained)
                # Dropped work must not return after restart.
                persisted = await self._clear_persistent_queue(uid)
                # Admit the next prompt only after its predecessor's entire
                # cancellation is finished, including delayed outbound work.
                delayed_replies = await self._clear_reply_deliveries(uid)
                detached_cancelled = await self._cancel_detached_tasks(uid)
            cleared = max(len(drained), persisted, delayed_replies)
            return task_running, bool(
                was_awaiting or cleared or detached_cancelled
                or cancelled_uploads or launches
            )
        finally:
            depth = self._cancel_work_depths[uid] - 1
            if depth:
                self._cancel_work_depths[uid] = depth
            else:
                self._cancel_work_depths.pop(uid, None)

    # /new (dir-input flow)

    async def cmd_new(self, ctx: BotContext) -> None:
        current = workspace.get_current(ctx.user_id, self.platform_id)
        await ctx.reply_text(
            f"Current workspace: {current or '(none)'}\n\n"
            "Enter the full path for the new workspace directory"
            " (or /cancel):"
        )
        self._expect_input(ctx.user_id, self._receive_new_dir)

    async def _receive_new_dir(self, ctx: BotContext) -> None:
        path = ctx.text.strip()
        if os.path.exists(path):
            await ctx.reply_text(
                f"Directory already exists:\n{path}\n\n"
                "Please choose a different path (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_new_dir)
            return
        try:
            os.makedirs(path)
        except OSError as e:
            await ctx.reply_text(
                f"Failed to create directory: {e}\n\n"
                "Please try again (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_new_dir)
            return
        try:
            workspace.ensure_cozter_dir(path)
        except ValueError as exc:
            await ctx.reply_text(
                f"Failed to initialize secure workspace state: {exc}\n\n"
                "Please choose a different directory (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_new_dir)
            return
        workspace.select_workspace(ctx.user_id, path, self.platform_id)
        await ctx.reply_text(f"Workspace created and selected:\n{path}")

    # /open

    async def cmd_open(self, ctx: BotContext) -> None:
        if ctx.args.strip():
            await self._open_workspace_from_text(
                ctx, ctx.args.strip(), rearm_on_error=False,
            )
            return

        current = workspace.get_current(ctx.user_id, self.platform_id)
        recent = workspace.get_recent(ctx.user_id, self.recent_limit)
        lines = [f"Current workspace: {current or '(none)'}"]
        if recent:
            lines.append("\nRecent workspaces:")
            for recent_index, recent_path in enumerate(recent, 1):
                lines.append(f"  {recent_index}. {recent_path}")
        else:
            lines.append("\nNo recent workspaces.")
        lines.append(
            "\nEnter a directory path or number from the list (or /cancel):"
        )
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_open_dir)

    async def _receive_open_dir(self, ctx: BotContext) -> None:
        await self._open_workspace_from_text(
            ctx, ctx.text.strip(), rearm_on_error=True,
        )

    async def _open_workspace_from_text(
        self, ctx: BotContext, text: str, *, rearm_on_error: bool,
    ) -> None:
        recent = workspace.get_recent(ctx.user_id, self.recent_limit)
        if text.isdecimal():
            number = parse_decimal_int(text)
            idx = -1 if number is None else number - 1
            if 0 <= idx < len(recent):
                path = recent[idx]
            else:
                if rearm_on_error:
                    await ctx.reply_text(
                        "Invalid number. Please try again (or /cancel):"
                    )
                    self._expect_input(ctx.user_id, self._receive_open_dir)
                else:
                    await ctx.reply_text(
                        "Invalid number. Use /open to choose from recent"
                        " workspaces."
                    )
                return
        else:
            path = text
        if not os.path.isdir(path):
            if rearm_on_error:
                await ctx.reply_text(
                    f"Directory does not exist:\n{path}\n\n"
                    "Please enter a valid directory (or /cancel):"
                )
                self._expect_input(ctx.user_id, self._receive_open_dir)
            else:
                await ctx.reply_text(
                    f"Directory does not exist:\n{path}\n\n"
                    "Use /open to choose from recent workspaces."
                )
            return
        try:
            workspace.ensure_cozter_dir(path)
        except ValueError as exc:
            message = f"Workspace state is unsafe: {exc}"
            if rearm_on_error:
                message += "\n\nPlease enter a different directory (or /cancel):"
                self._expect_input(ctx.user_id, self._receive_open_dir)
            else:
                message += "\n\nUse /open to choose a different workspace."
            await ctx.reply_text(message)
            return
        workspace.select_workspace(ctx.user_id, path, self.platform_id)
        await ctx.reply_text(f"Workspace selected:\n{path}")

    # /model

    async def cmd_model(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        backend_name = workspace.get_backend_name(ws)
        # Flexible has per-tier models.
        if backend_name == workspace.FLEXIBLE_BACKEND:
            await ctx.reply_text(self._flexible_summary(ws))
            return
        current = workspace.get_model(ws)
        options = await asyncio.to_thread(workspace.get_available_models, ws)
        await ctx.reply_text("\n".join([
            f"Current model: {current} (backend: {backend_name})\n",
            *self._picker_list(options, current, noun="model"),
        ]))
        self._expect_input(ctx.user_id, self._receive_model)

    async def _receive_model(
        self, ctx: BotContext,
    ) -> None:
        await self._apply_model_choice(
            ctx,
            fetch_options=workspace.get_available_models,
            retry_handler=self._receive_model,
            set_model=workspace.set_model,
            success_text="Model set to: {model}",
        )

    # /summarymodel

    async def cmd_summary_model(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_summary_model(ws)
        summary_backend = workspace.get_summary_backend_name(ws)
        options = await asyncio.to_thread(
            workspace.get_available_summary_models, ws,
        )
        await ctx.reply_text("\n".join([
            f"Current summary model: {current}"
            f" (summary agent: {summary_backend})\n",
            *self._picker_list(options, current, noun="model"),
        ]))
        self._expect_input(ctx.user_id, self._receive_summary_model)

    async def _receive_summary_model(
        self, ctx: BotContext,
    ) -> None:
        await self._apply_model_choice(
            ctx,
            fetch_options=workspace.get_available_summary_models,
            retry_handler=self._receive_summary_model,
            set_model=workspace.set_summary_model,
            success_text="Summary model set to: {model}",
        )

    # /agent

    async def cmd_agent(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_backend_name(ws)
        options = workspace.AVAILABLE_BACKENDS
        await ctx.reply_text("\n".join([
            f"Current agent: {current}\n",
            *self._picker_list(
                options, current, noun="agent", first_number=0,
            ),
            f"\n{workspace.FLEXIBLE_BACKEND} splits each request by"
            " difficulty and routes the parts to its low/mid/high agents.",
        ]))
        self._expect_input(ctx.user_id, self._receive_agent)

    async def _receive_agent(self, ctx: BotContext) -> None:
        selected = await self._receive_backend_choice(
            ctx,
            options=workspace.AVAILABLE_BACKENDS,
            retry_handler=self._receive_agent,
            first_number=0,
        )
        if selected is None:
            return
        ws, name = selected
        workspace.set_backend_name(ws, name)
        if name == workspace.FLEXIBLE_BACKEND:
            await ctx.reply_text(
                f"Agent set to: {name}\n\n{self._flexible_summary(ws)}"
            )
            return
        _, model, summary_model, _, summary_backend = (
            workspace.get_run_config(ws)
        )
        # Always show the summary agent.
        await ctx.reply_text(
            f"Agent set to: {name}\n"
            f"Model: {model}\n"
            f"Summary agent: {summary_backend}\n"
            f"Summary model: {summary_model}"
        )

    async def _receive_backend_choice(
        self,
        ctx: BotContext,
        *,
        options: list[str],
        retry_handler: Callable[[BotContext], Awaitable[None]],
        first_number: int = 1,
    ) -> tuple[str, str] | None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return None
        text = ctx.text.strip()
        name = self._pick_option(text, options, first_number=first_number)
        if name is None:
            await ctx.reply_text(
                f"Unknown agent: {text}\nTry again (or /cancel):"
            )
            self._expect_input(ctx.user_id, retry_handler)
            return None
        return ws, name

    async def _receive_model_choice(
        self,
        ctx: BotContext,
        *,
        fetch_options: Callable[[str], list[str]],
        retry_handler: Callable[[BotContext], Awaitable[None]],
    ) -> tuple[str, str] | None:
        """Shared picker-input handler for /model and /summarymodel.

        Resolves the workspace off-thread via *fetch_options*, matches the
        user's text against the available models, and re-prompts through
        *retry_handler* on an unknown name. Returns ``(ws, model)`` or
        ``None`` when the workspace is gone or the input did not match.
        """
        ws = await self._require_ws(ctx)
        if ws is None:
            return None
        text = ctx.text.strip()
        options = await asyncio.to_thread(fetch_options, ws)
        model = self._pick_option(text, options)
        if model is None:
            await ctx.reply_text(
                f"Unknown model: {text}\nTry again (or /cancel):"
            )
            self._expect_input(ctx.user_id, retry_handler)
            return None
        return ws, model

    async def _apply_model_choice(
        self,
        ctx: BotContext,
        *,
        fetch_options: Callable[[str], list[str]],
        retry_handler: Callable[[BotContext], Awaitable[None]],
        set_model: Callable[[str, str], None],
        success_text: str,
    ) -> None:
        """Resolve a model-picker input, save it, and confirm the choice."""
        selected = await self._receive_model_choice(
            ctx,
            fetch_options=fetch_options,
            retry_handler=retry_handler,
        )
        if selected is None:
            return
        workspace_path, model = selected
        set_model(workspace_path, model)
        await ctx.reply_text(success_text.format(model=model))

    @staticmethod
    def _option_lines(
        options: list[str], current: str, *, first_number: int = 1,
    ) -> list[str]:
        """Number a picker's options and mark the active one."""
        return [
            f"  {option_index}. {name}{' <-' if name == current else ''}"
            for option_index, name in enumerate(options, first_number)
        ]

    @staticmethod
    def _picker_list(
        options: list[str], current: str, *, noun: str,
        first_number: int = 1,
    ) -> list[str]:
        """The shared body of a model/agent picker: the "Available …" list,
        the numbered options, and the trailing input prompt."""
        return [
            f"Available {noun}s:",
            *BotPlatform._option_lines(
                options, current, first_number=first_number,
            ),
            f"\nEnter a number or {noun} name (or /cancel):",
        ]

    @staticmethod
    def _flexible_summary(ws: str) -> str:
        """Describe how the flexible agent is currently wired up."""
        lines = [
            "The flexible agent has no single model — it splits your"
            " request and routes each part to a difficulty tier:",
            "",
        ]
        for tier, (name, model) in workspace.get_flexible_run_config(
            ws,
        ).items():
            lines.append(
                f"  {tier:<4} {name}/{model}"
                f" — {workspace.FLEXIBLE_TIER_DESCRIPTIONS[tier]}"
            )
        lines.append(
            "\nPlanning and merging run on the summary agent:"
            f" {workspace.get_summary_backend_name(ws)}"
            f"/{workspace.get_summary_model(ws)}"
        )
        lines.append(
            "\nRebind a tier with /agent_flexible_<tier> or"
            " /model_flexible_<tier>, where <tier> is "
            + "|".join(workspace.FLEXIBLE_TIERS)
            + "."
        )
        return "\n".join(lines)

    # /agent_flexible_<tier>, /model_flexible_<tier>

    async def _show_flexible_agent(self, ctx: BotContext, tier: str) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_flexible_backend_name(ws, tier)
        # Exclude flexible: a tier loop recurses.
        options = workspace.DIRECT_BACKENDS
        await ctx.reply_text("\n".join([
            f"Flexible {tier} tier —"
            f" {workspace.FLEXIBLE_TIER_DESCRIPTIONS[tier]}",
            f"Current agent: {current}\n",
            *self._picker_list(options, current, noun="agent"),
        ]))
        self._expect_input(
            ctx.user_id,
            functools.partial(self._receive_flexible_agent, tier=tier),
        )

    async def _receive_flexible_agent(
        self, ctx: BotContext, *, tier: str,
    ) -> None:
        selected = await self._receive_backend_choice(
            ctx,
            options=workspace.DIRECT_BACKENDS,
            retry_handler=functools.partial(
                self._receive_flexible_agent, tier=tier,
            ),
        )
        if selected is None:
            return
        ws, name = selected
        workspace.set_flexible_backend_name(ws, tier, name)
        await ctx.reply_text(
            f"Flexible {tier} agent set to: {name}\n"
            f"Flexible {tier} model: {workspace.get_flexible_model(ws, tier)}"
        )

    async def _show_flexible_model(self, ctx: BotContext, tier: str) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_flexible_model(ws, tier)
        backend_name = workspace.get_flexible_backend_name(ws, tier)
        options = await asyncio.to_thread(
            workspace.get_available_flexible_models, ws, tier,
        )
        await ctx.reply_text("\n".join([
            f"Flexible {tier} tier —"
            f" {workspace.FLEXIBLE_TIER_DESCRIPTIONS[tier]}",
            f"Current model: {current} (agent: {backend_name})\n",
            *self._picker_list(options, current, noun="model"),
        ]))
        self._expect_input(
            ctx.user_id,
            functools.partial(self._receive_flexible_model, tier=tier),
        )

    async def _receive_flexible_model(
        self, ctx: BotContext, *, tier: str,
    ) -> None:
        def set_flexible_model(workspace_path: str, model: str) -> None:
            workspace.set_flexible_model(workspace_path, tier, model)

        await self._apply_model_choice(
            ctx,
            fetch_options=functools.partial(
                workspace.get_available_flexible_models, tier=tier,
            ),
            retry_handler=functools.partial(
                self._receive_flexible_model, tier=tier,
            ),
            set_model=set_flexible_model,
            success_text=f"Flexible {tier} model set to: {{model}}",
        )

    # /summaryagent

    async def cmd_summary_agent(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_summary_backend_name(ws)
        # The summary agent runs compaction/titling/plan/merge.
        options = workspace.DIRECT_BACKENDS
        await ctx.reply_text("\n".join([
            f"Current summary agent: {current}\n",
            *self._picker_list(options, current, noun="agent"),
        ]))
        self._expect_input(ctx.user_id, self._receive_summary_agent)

    async def _receive_summary_agent(self, ctx: BotContext) -> None:
        selected = await self._receive_backend_choice(
            ctx,
            options=workspace.DIRECT_BACKENDS,
            retry_handler=self._receive_summary_agent,
        )
        if selected is None:
            return
        ws, name = selected
        workspace.set_summary_backend_name(ws, name)
        summary_model = workspace.get_summary_model(ws)
        await ctx.reply_text(
            f"Summary agent set to: {name}\n"
            f"Summary model: {summary_model}"
        )

    # /permission

    async def cmd_permission(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_permission(ws)
        options = workspace.AVAILABLE_PERMISSIONS
        lines = [f"Current permission: {current}\n", "Available modes:"]
        for option_index, option in enumerate(options, 1):
            marker = " <-" if option == current else ""
            desc = workspace.PERMISSION_DESCRIPTIONS[option]
            lines.append(f"  {option_index}. {option} - {desc}{marker}")
        ceiling = workspace.permission_ceiling()
        if ceiling != "full":
            lines.append(
                f"\n(Capped at '{ceiling}' by config max_permission;"
                " higher modes are rejected.)"
            )
        lines.append(
            "\nTip: chat surfaces can't show a per-tool approval dialog, so"
            " 'confirm' is best-effort. For ask-before-acting behavior on"
            " any backend use /style collaborative — it pauses the turn for"
            " your reply."
        )
        lines.append("\nEnter a number or mode name (or /cancel):")
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_permission)

    async def _receive_permission(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        text = ctx.text.strip().lower()
        options = workspace.AVAILABLE_PERMISSIONS
        perm = self._pick_option(text, options)
        if perm is None:
            await ctx.reply_text(
                f"Unknown mode: {text}\nTry again (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_permission)
            return
        try:
            workspace.set_permission(ws, perm)
        except ValueError as e:
            await ctx.reply_text(f"{e}\nTry again (or /cancel):")
            self._expect_input(ctx.user_id, self._receive_permission)
            return
        desc = workspace.PERMISSION_DESCRIPTIONS[perm]
        await ctx.reply_text(f"Permission set to: {perm}\n{desc}")

    # /style

    async def cmd_style(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_interaction_style(ws)
        options = workspace.AVAILABLE_STYLES
        lines = [
            f"Current interaction style: {current}\n",
            "Available styles:",
        ]
        for option_index, option in enumerate(options, 1):
            marker = " <-" if option == current else ""
            desc = workspace.STYLE_DESCRIPTIONS[option]
            lines.append(f"  {option_index}. {option} - {desc}{marker}")
        lines.append(
            "\nCollaborative makes the agent ask before big or ambiguous"
            " actions and pause for your reply; autonomous makes it decide"
            " and proceed. Applies to chat turns (scheduled runs are always"
            " autonomous)."
        )
        lines.append("\nEnter a number or style name (or /cancel):")
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_style)

    async def _receive_style(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        text = ctx.text.strip().lower()
        options = workspace.AVAILABLE_STYLES
        style = self._pick_option(text, options)
        if style is None:
            await ctx.reply_text(
                f"Unknown style: {text}\nTry again (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_style)
            return
        workspace.set_interaction_style(ws, style)
        desc = workspace.STYLE_DESCRIPTIONS[style]
        await ctx.reply_text(f"Interaction style set to: {style}\n{desc}")

    # /effort

    async def cmd_effort(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        current = workspace.get_reasoning_effort(ws)
        await ctx.reply_text(
            f"Current reasoning effort: {current}%\n\n"
            "Pick a percentage 0-100. Higher = more reasoning,"
            " slower & costlier. 0 = no override (server picks).\n\n"
            "Each agent maps to its native scale:\n"
            "  codex:       model-aware; current models start at low and"
            " range up to xhigh, max, or ultra\n"
            "  llama:       1-24 minimal / 25-49 low / 50-74 medium"
            " / 75-100 high\n"
            "  claude_code: model-aware; current models range from low to"
            " max, while Haiku and legacy pins use their default\n"
            "  copilot:     1-16 minimal / 17-33 low / 34-49 medium"
            " / 50-66 high / 67-83 xhigh / 84-100 max"
            " for an explicit model; auto uses Copilot's default\n"
            "  grok:        model-aware; 4.7/4.6 range from low to xhigh,"
            " while 4.5 and unknown models range from low to high\n"
            "  meta:        standard Muse Spark 1.3 ranges from minimal"
            " to max; older and Contributor models stop at xhigh\n"
            "  zai:         GLM-5.3/Flash/FlashX use low / high / max;"
            " GLM-5.2 uses seven bands; compulsory-thinking models stay"
            " enabled, and other GLMs toggle at 50%\n\n"
            "Applies to your chat turns (not compaction/routing)."
            "\nEnter a number 0-100 (or /cancel):"
        )
        self._expect_input(ctx.user_id, self._receive_effort)

    async def _receive_effort(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        text = ctx.text.strip().rstrip("%").strip()
        value = try_parse_int(text)
        if value is None:
            await ctx.reply_text(
                f"Not a number: {ctx.text!r}\nEnter 0-100 (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_effort)
            return
        if not 0 <= value <= 100:
            await ctx.reply_text(
                f"Out of range: {value}\nEnter 0-100 (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_effort)
            return
        workspace.set_reasoning_effort(ws, value)
        await ctx.reply_text(
            f"Reasoning effort set to: {value}%"
            + ("\n(0 = no override - each backend uses its default)"
               if value == 0 else "")
        )

    # /refresh

    async def cmd_refresh(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        codex_dir = os.path.join(ws, ".codex")
        cleared = False
        if os.path.isdir(codex_dir):
            shutil.rmtree(codex_dir, ignore_errors=True)
            logger.info("Cleared codex session dir: %s", codex_dir)
            cleared = True
        backend_name = workspace.get_backend_name(ws)
        if cleared:
            msg = (
                f"{backend_name} CLI session refreshed."
                " Your conversation history is preserved."
            )
        else:
            msg = (
                f"{backend_name} has no workspace-local state to clear."
                " Your conversation history is preserved."
            )
        await ctx.reply_text(msg)

    # /compact

    @staticmethod
    def _first_command_argument(ctx: BotContext) -> str:
        """Return the normalized first argument supplied to a command."""
        arguments = ctx.args.strip().lower().split()
        return arguments[0] if arguments else ""

    @staticmethod
    async def _set_decimal_workspace_setting(
        ctx: BotContext,
        workspace_path: str,
        argument: str,
        *,
        setter: Callable[[str, int], None],
        success_message: str,
    ) -> bool:
        """Set a decimal workspace value, returning whether it was handled."""
        if not argument.isdecimal():
            return False
        value = parse_decimal_int(argument)
        if value is None:
            await ctx.reply_text("Error: number is too large.")
            return True
        try:
            setter(workspace_path, value)
        except ValueError as exc:
            await ctx.reply_text(f"Error: {exc}")
            return True
        await ctx.reply_text(success_message.format(value=value))
        return True

    async def _workspace_for_decimal_setting(
        self,
        ctx: BotContext,
        *,
        setter: Callable[[str, int], None],
        success_message: str,
    ) -> str | None:
        """Return the workspace when a decimal setting was not handled."""
        ws = await self._require_ws(ctx)
        if ws is None:
            return None
        if await self._set_decimal_workspace_setting(
            ctx,
            ws,
            self._first_command_argument(ctx),
            setter=setter,
            success_message=success_message,
        ):
            return None
        return ws

    async def cmd_compact(self, ctx: BotContext) -> None:
        ws = await self._workspace_for_decimal_setting(
            ctx,
            setter=workspace.set_compact_interval,
            success_message="Fallback compact interval set to {value} messages.",
        )
        if ws is None:
            return

        current = workspace.get_compact_interval(ws)
        sessions = session.list_sessions(ws)
        total_msgs = sum(session.get("message_count", 0) for session in sessions)
        lines = [
            f"Fallback compact interval: {current} messages",
            f"Sessions: {len(sessions)} ({total_msgs} total messages)",
            "",
            "Compaction is automatic. When every selected recipient model "
            "has a known context window, Cozter compacts from its estimated "
            "token use (while respecting /context). Otherwise it falls back "
            f"to every {current} stored messages.",
            "",
            "Usage:",
            "  /compact <number> - set fallback interval",
        ]
        await ctx.reply_text("\n".join(lines))

    # /context

    async def cmd_context(self, ctx: BotContext) -> None:
        ws = await self._workspace_for_decimal_setting(
            ctx,
            setter=workspace.set_history_budget,
            success_message="Context budget set to {value} characters.",
        )
        if ws is None:
            return

        current = workspace.get_history_budget(ws)
        lines = [
            f"Context budget: {current} characters",
            "",
            "Colony memory, long-term memory, the session summary, and"
            " recent messages are prepended to each turn up to this"
            " budget; the oldest recent messages are dropped first to fit."
            " Measured in characters as a provider-agnostic proxy for"
            " tokens — raise it for large-context models, lower it for"
            " small local ones.",
            "",
            "Usage:",
            "  /context <number> - set the budget (characters)",
        ]
        await ctx.reply_text("\n".join(lines))

    # /newsession

    async def cmd_new_session(self, ctx: BotContext) -> None:
        """Start a fresh session in the current workspace.

        Normally every turn continues the user's last-active session
        (persisted in ``.cozter/last_session.json``, so a bot restart
        resumes where the user left off). This command creates a new
        session and pins it as the new last_session, so the next
        message starts a clean conversation without affecting any
        existing session's history.
        """
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        data = session.create_session(ws)
        session.set_last_session(ws, ctx.user_id, data["id"])
        await ctx.reply_text(
            f"Started new session: {data['name']}\n"
            "Your next message goes into this fresh session.",
        )

    # /sessions

    @staticmethod
    def _pick_session(choice: str, sessions: list[dict]) -> dict | None:
        """Resolve a /sessions selection: 1-based number, exact, or substring."""
        if not isinstance(choice, str) or not isinstance(sessions, list):
            return None
        choice = choice.strip()
        if choice.isdecimal():
            number = parse_decimal_int(choice)
            if number is None:
                return None
            idx = number - 1
            if 0 <= idx < len(sessions) and isinstance(sessions[idx], dict):
                return sessions[idx]
            return None
        low = choice.lower()
        named = [
            session for session in sessions
            if isinstance(session, dict) and isinstance(session.get("name"), str)
        ]
        for named_session in named:
            if named_session["name"].lower() == low:
                return named_session
        for named_session in named:
            if low and low in named_session["name"].lower():
                return named_session
        return None

    async def cmd_sessions(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        arg = ctx.args.strip()
        if arg:
            await self._switch_session(ctx, ws, arg)
            return
        sessions = session.list_sessions(ws)
        if not sessions:
            await ctx.reply_text(
                "No sessions yet — send a message to start one.",
            )
            return
        current = session.get_last_session(ws, ctx.user_id)
        lines = ["Sessions (newest first):"]
        for session_index, existing in enumerate(sessions, 1):
            marker = " <-" if existing["id"] == current else ""
            lines.append(
                f"  {session_index}. {existing['name']} ({existing['message_count']} msgs){marker}"
            )
        lines.append(
            "\nReply with a number or name to switch, /newsession for a"
            " fresh one (or /cancel):"
        )
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_session_switch)

    async def _receive_session_switch(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        await self._switch_session(ctx, ws, ctx.text.strip())

    async def _switch_session(
        self, ctx: BotContext, ws: str, choice: str,
    ) -> None:
        sessions = session.list_sessions(ws)
        target = self._pick_session(choice, sessions)
        if target is None:
            await ctx.reply_text(
                f"No session matches {choice!r}. Use /sessions to list them.",
            )
            return
        session.set_last_session(ws, ctx.user_id, target["id"])
        await ctx.reply_text(
            f"Switched to session: {target['name']}\n"
            "Your next message continues this conversation.",
        )

    # /colony

    async def cmd_colony(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        first = self._first_command_argument(ctx)

        if first == "now":
            await ctx.reply_text("Consolidating colony...")
            summary_model = workspace.get_summary_model(ws)
            summary_backend = workspace.get_summary_backend_name(ws)
            ok = await colony.consolidate(
                ws, summary_model, backend_name=summary_backend,
            )
            if ok:
                items = colony.get_items(ws)
                await ctx.reply_text(
                    f"Colony consolidated ({len(items)} item(s))."
                )
            else:
                await ctx.reply_text(
                    "Colony pass produced no changes (see logs)."
                )
            return

        if await self._set_decimal_workspace_setting(
            ctx,
            ws,
            first,
            setter=workspace.set_colony_interval,
            success_message="Colony interval set to {value} compaction(s).",
        ):
            return

        items = colony.get_items(ws)
        count = colony.get_compact_count(ws)
        interval = workspace.get_colony_interval(ws)
        # Countdown in [1, interval].
        until = interval - (count % interval) if interval > 0 else 0
        lines = [
            f"Colony items: {len(items)}",
            f"Compactions since last pass: {count % interval}/{interval}",
            f"Compactions until next pass: {until}",
            "",
        ]
        if items:
            lines.append("Items:")
            for item_index, it in enumerate(items, 1):
                lines.append(f"  {item_index}. {it}")
            lines.append("")
        lines.extend([
            "Usage:",
            "  /colony <number> - set interval (compactions per pass)",
            "  /colony now - run consolidation immediately",
        ])
        await ctx.reply_text("\n".join(lines))

    # /stop

    async def cmd_stop(self, ctx: BotContext) -> None:
        # /stop abandons questions and queued prompts.
        task_running, cancelled_work = await self._cancel_user_work(ctx.user_id)
        if task_running or cancelled_work:
            await ctx.reply_text("Cancelled.")
            return
        await ctx.reply_text("Nothing is running.")

    # /bg, /background

    async def cmd_background(self, ctx: BotContext) -> None:
        """Launch a provider-owned detached task and arrange its callback."""
        task_prompt = ctx.args.strip()
        if not task_prompt:
            await ctx.reply_text("Usage: /bg <task>")
            return
        if self._cancel_work_depths.get(ctx.user_id, 0):
            await ctx.reply_text("Cancellation is still finishing. Please retry /bg afterward.")
            return
        generation = self._cancel_generations.get(ctx.user_id, 0)
        task = asyncio.create_task(self._run_background_command(ctx, task_prompt, generation))
        tasks = self._detached_launch_tasks.setdefault(ctx.user_id, set())
        tasks.add(task)
        try:
            await task
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.wait((task,))
                except asyncio.CancelledError:
                    continue
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        finally:
            tasks.discard(task)
            if not tasks:
                self._detached_launch_tasks.pop(ctx.user_id, None)

    async def _run_background_command(
        self, ctx: BotContext, task_prompt: str, generation: int,
    ) -> None:
        ws = await self._require_ws(ctx)
        if ws is None or self._cancel_generations.get(ctx.user_id, 0) != generation:
            return
        backend_name, model, summary_model, permission, summary_backend = (
            workspace.get_run_config(ws)
        )
        try:
            launch = await agent.launch_detached(
                task_prompt,
                ws,
                ctx.user_id,
                model=model,
                summary_model=summary_model,
                approval=permission,
                backend_name=backend_name,
                summary_backend_name=summary_backend,
            )
        except Exception as exc:
            await ctx.reply_text(f"Could not start background task: {exc}")
            return
        try:
            if self._cancel_generations.get(ctx.user_id, 0) != generation:
                raise asyncio.CancelledError
            tracked = await self._register_detached_task(
                uid=ctx.user_id,
                chat_id=ctx.chat_id,
                workspace_path=ws,
                session_id=launch.session_id,
                backend_name=launch.backend_name,
                task_id=launch.task_id,
                validate=True,
                generation=generation,
            )
            if self._cancel_generations.get(ctx.user_id, 0) != generation:
                raise asyncio.CancelledError
        except asyncio.CancelledError:
            await self._cleanup_cancelled_detached_launch(ctx.user_id, ws, launch)
            raise
        except Exception:
            logger.exception(
                "Could not persist detached task %s", launch.task_id,
            )
            tracked = False
        try:
            await self._acknowledge_background_launch(ctx, launch, tracked, generation)
        except asyncio.CancelledError:
            await self._cleanup_cancelled_detached_launch(ctx.user_id, ws, launch)
            raise

    async def _acknowledge_background_launch(
        self, ctx: BotContext, launch: agent.DetachedTaskLaunch, tracked: bool, generation: int,
    ) -> None:
        if self._cancel_generations.get(ctx.user_id, 0) != generation:
            raise asyncio.CancelledError
        if tracked:
            await ctx.reply_text(
                f"Started background task {launch.task_id}. "
                "I’ll post its final result here.",
            )
            return
        await ctx.reply_text(
            f"Started background task {launch.task_id}, but Cozter could "
            "not verify it for completion delivery. Check it with "
            f"`claude agents` or `claude logs {launch.task_id}`.",
        )

    async def _cleanup_cancelled_detached_launch(
        self, uid: str, ws: str, launch: agent.DetachedTaskLaunch,
    ) -> None:
        cleanup = asyncio.create_task(self._stop_cancelled_detached_launch(uid, ws, launch))
        while not cleanup.done():
            try:
                await asyncio.wait((cleanup,))
            except asyncio.CancelledError:
                continue
        cleanup.result()

    async def _stop_cancelled_detached_launch(
        self, uid: str, ws: str, launch: agent.DetachedTaskLaunch,
    ) -> None:
        """Stop a known provider task cancelled during validation or persistence."""
        try:
            backend = backends_agent.get_backend(launch.backend_name)
            stopped = await backend.stop_detached_task(ws, launch.task_id)
            if stopped:
                for record in await self._list_detached_task_records():
                    if (record["user_id"] == uid and record["backend_name"] == launch.backend_name
                            and record["task_id"] == launch.task_id):
                        await self._remove_detached_task_record(record["id"])
        except Exception:
            logger.warning("Could not stop cancelled detached launch %s", launch.task_id, exc_info=True)

    # /inject

    async def cmd_inject(self, ctx: BotContext) -> None:
        text = ctx.args.strip()
        if not text:
            await ctx.reply_text("Usage: /inject <message>")
            return
        inject_q = self._inject_queues.get(ctx.user_id)
        if inject_q is not None:
            outcome = inject_q.put_if_active(text)
            if outcome == "accepted":
                # Ack is cosmetic; bound it.
                try:
                    await asyncio.wait_for(
                        ctx.reply_text("Injected."),
                        timeout=_INJECT_ACK_TIMEOUT_SEC,
                    )
                except TimeoutError:  # == asyncio.TimeoutError on 3.11+
                    logger.warning(
                        "Inject ack send timed out after %.1fs (message queued)",
                        _INJECT_ACK_TIMEOUT_SEC,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("Inject ack send failed", exc_info=True)
                return
            if outcome == "full":
                await ctx.reply_text("Inject queue full.")
                return
            # "finished": delivery may hold the lock; queue a follow-up, don't drop.
        uid = ctx.user_id
        lock = self._task_locks.get(uid)
        running = self._running_tasks.get(uid)
        turn_active = (
            inject_q is not None
            or (lock is not None and lock.locked())
            or (running is not None and not running.done())
            or uid in self._pending_reply_delivery_users
        )
        if not turn_active:
            await ctx.reply_text("No task is running.")
            return
        generation = self._cancel_generations.get(uid, 0)
        msg_queue = self._ensure_message_queue(uid)
        if msg_queue.full():
            await ctx.reply_text("Queue full. Wait or /stop first.")
            return
        was_awaiting = uid in self._awaiting_answer
        entry_id = await self._persist_and_queue_dispatch(
            uid, text, ctx.chat_id, generation, msg_queue,
        )
        if entry_id is None:
            return
        if was_awaiting:
            await self._resume_awaiting_answer(
                uid, msg_queue, entry_id,
                reason="inject arrived after turn finished",
            )
        await ctx.reply_text(
            f"Queued ({msg_queue.qsize()}/{self.max_queue_size})."
            " Running next.",
        )
        # Race guard: kick a drain so the entry isn't orphaned.
        self._start_queue_drain(uid)

    # /reserve (recurring schedule wizard)

    async def cmd_reserve(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        await ctx.reply_text(
            "Select days for this schedule.\n"
            "Use comma-separated names or numbers, or 'all'.\n"
            "1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun\n\n"
            "Examples: '1,3,5' or 'mon,wed,fri' or 'all'\n\n"
            "Enter days (or /cancel):"
        )
        self._expect_input(ctx.user_id, self._receive_reserve_days)

    async def _receive_reserve_days(self, ctx: BotContext) -> None:
        text = ctx.text.strip().lower()
        days = schedules.parse_days(text)
        if not days:
            await ctx.reply_text(
                "Could not parse days. Try again (e.g., 'mon,wed,fri'"
                " or '1,3,5') (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_reserve_days)
            return
        await ctx.reply_text(
            f"Days: {', '.join(days)}\n\n"
            "Enter time in HH:MM (24-hour) format (or /cancel):"
        )

        self._expect_input(
            ctx.user_id,
            functools.partial(self._receive_reserve_time, days=days),
        )

    async def _receive_reserve_time(
        self, ctx: BotContext, days: list[str],
    ) -> None:
        time_str = schedules.parse_time(ctx.text.strip())
        if time_str is None:
            await ctx.reply_text(
                "Invalid time. Enter HH:MM (24-hour, e.g., 09:30)"
                " (or /cancel):"
            )

            self._expect_input(
                ctx.user_id,
                functools.partial(self._receive_reserve_time, days=days),
            )
            return
        await ctx.reply_text(
            f"Time: {time_str}\n\n"
            "Enter the command to run at that time (or /cancel):"
        )

        self._expect_input(
            ctx.user_id,
            functools.partial(
                self._receive_reserve_command,
                days=days,
                time_str=time_str,
            ),
        )

    async def _receive_reserve_command(
        self, ctx: BotContext, days: list[str], time_str: str,
    ) -> None:
        command = ctx.text.strip()
        if not command:
            await ctx.reply_text(
                "Command cannot be empty. Try again (or /cancel):"
            )

            self._expect_input(
                ctx.user_id,
                functools.partial(
                    self._receive_reserve_command,
                    days=days,
                    time_str=time_str,
                ),
            )
            return

        ws = await self._require_ws(ctx)
        if ws is None:
            return

        schedule = {
            "id": uuid.uuid4().hex[:12],
            "days": days,
            "time": time_str,
            "command": command,
            "created": datetime.now().isoformat(),
            "chat_id": ctx.chat_id,
            "user_id": ctx.user_id,
            "platform_id": self.platform_id,
        }
        # Hold the lock against scheduler ticks.
        async with workspace.get_lock(ws):
            schedules.add_schedule(ws, ctx.user_id, schedule)
        await ctx.reply_text(
            "Schedule created:\n"
            f"  Days: {', '.join(days)}\n"
            f"  Time: {time_str}\n"
            f"  Command: {command}"
        )

    # /schedules (list/delete)

    def _platform_schedules(self, ws: str, uid: str) -> list[dict]:
        """Keep legacy schedules and records created by this platform."""
        return [
            record for record in schedules.list_schedules(ws, uid)
            if record.get("platform_id") in (None, "", self.platform_id)
        ]

    async def cmd_schedules(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        user_schedules = self._platform_schedules(ws, ctx.user_id)
        if not user_schedules:
            await ctx.reply_text("No schedules.")
            return
        lines = ["Schedules:"]
        for schedule_index, schedule in enumerate(user_schedules, 1):
            raw_days = schedule.get("days", [])
            days = raw_days if isinstance(raw_days, list) else []
            days_str = ",".join(day for day in days if isinstance(day, str))
            lines.append(
                f"  {schedule_index}. [{days_str}] {schedule.get('time', '?')}"
                f" — {schedule.get('command', '')}"
            )
        lines.append(
            "\nEnter a number to delete, or /cancel to exit:"
        )
        await ctx.reply_text("\n".join(lines))
        self._expect_input(ctx.user_id, self._receive_schedules)

    async def _receive_schedules(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        user_schedules = self._platform_schedules(ws, ctx.user_id)
        text = ctx.text.strip()
        if not text.isdecimal():
            await ctx.reply_text(
                "Invalid input. Enter a number or /cancel:"
            )
            self._expect_input(ctx.user_id, self._receive_schedules)
            return
        number = parse_decimal_int(text)
        idx = -1 if number is None else number - 1
        if not (0 <= idx < len(user_schedules)):
            await ctx.reply_text(
                "Invalid number. Try again (or /cancel):"
            )
            self._expect_input(ctx.user_id, self._receive_schedules)
            return
        removed = user_schedules[idx]
        schedule_id = removed.get("id")
        if not isinstance(schedule_id, str) or not schedule_id:
            await ctx.reply_text(
                "That schedule entry is malformed and cannot be deleted "
                "by id. Fix or remove it from .cozter/schedules.json."
            )
            return
        async with workspace.get_lock(ws):
            schedules.remove_schedule(ws, ctx.user_id, schedule_id)
        raw_days = removed.get("days", [])
        days = raw_days if isinstance(raw_days, list) else []
        await ctx.reply_text(
            f"Removed: [{','.join(day for day in days if isinstance(day, str))}]"
            f" {removed.get('time', '?')} — {removed.get('command', '')}"
        )

    def _platform_state_file_path(self, stem: str) -> str:
        """Return a durable per-platform state path with a safe file name."""
        os.makedirs(workspace.CONFIG_DIR, exist_ok=True)
        # Sanitize platform_id for files.
        safe = _UNSAFE_FILENAME_CHARS_RE.sub('_', self.platform_id)
        return os.path.join(workspace.CONFIG_DIR, f"{stem}_{safe}.json")

    def _read_platform_state_file(self, path: str, label: str) -> dict:
        """Load one platform-scoped durable state file."""
        return load_json_object(path, label, logger)

    def _write_platform_state_file(self, path: str, data: dict) -> None:
        """Atomically save one platform-scoped durable state file."""
        save_json_object(path, data)

    def _queue_file_path(self) -> str:
        return self._platform_state_file_path("queue")

    def _read_queue_file(self) -> dict:
        return self._read_platform_state_file(
            self._queue_file_path(), "queue file",
        )

    @staticmethod
    def _queue_entries(value: object) -> list[dict]:
        if not isinstance(value, list):
            return []
        return [entry for entry in value if isinstance(entry, dict)]

    def _write_queue_file(self, data: dict) -> None:
        self._write_platform_state_file(self._queue_file_path(), data)

    async def _load_job_checkpoint(self, uid: str, entry_id: str) -> dict:
        def read() -> dict:
            for entry in self._queue_entries(self._read_queue_file().get(uid)):
                if entry.get("id") == entry_id:
                    job = entry.get("job")
                    return job if isinstance(job, dict) else {}
            return {}

        return await _run_durable_file_operation(self._queue_file_lock, read)

    async def _save_job_checkpoint(
        self, uid: str, entry_id: str, job: dict,
    ) -> bool:
        """Checkpoint an existing job without resurrecting a cancelled entry."""
        def save() -> bool:
            data = self._read_queue_file()
            entries = self._queue_entries(data.get(uid))
            for entry in entries:
                if entry.get("id") == entry_id:
                    entry["job"] = job
                    # The running request must win over later arrivals even
                    # if it originally entered through the direct path.
                    data[uid] = [entry, *[item for item in entries if item is not entry]]
                    self._write_queue_file(data)
                    return True
            return False

        return await _run_durable_file_operation(self._queue_file_lock, save)

    async def _retain_job_entry(self, uid: str, entry_id: str) -> None:
        """Put unfinished work first in memory, retaining durable FIFO order."""
        def read() -> dict | None:
            return next((entry for entry in self._queue_entries(
                self._read_queue_file().get(uid),
            ) if entry.get("id") == entry_id), None)

        entry = await _run_durable_file_operation(self._queue_file_lock, read)
        if entry is None:
            return  # /stop already removed it
        existing = self._message_queues.get(uid)
        queue = self._ensure_message_queue(
            uid, min_size=(existing.qsize() if existing else 0) + 1,
        )
        if not any(isinstance(item, tuple) and len(item) == 4
                   and item[2] == entry_id for item in queue._queue):
            queue.put_nowait((entry.get("text", ""), entry.get("chat_id", ""),
                              entry_id, entry.get("ephemeral") is True))
        self._promote_queue_entry(queue, entry_id)

    async def _pause_job_after_failure(
        self, uid: str, entry_id: str, chat_id: str, error: Exception,
    ) -> None:
        self._paused_jobs.add(uid)
        try:
            job = await self._load_job_checkpoint(uid, entry_id)
            job.update({"state": "paused", "instruction": f"Resolve this failure and finish the original job: {error}"})
            await self._save_job_checkpoint(uid, entry_id, job)
            await self._retain_job_entry(uid, entry_id)
        except Exception:
            # The original inbound entry is already durable. A failed
            # checkpoint write must never turn it into a completed request.
            logger.exception("Could not save failure checkpoint for unfinished job")
        await self._send_text_best_effort(
            chat_id, f"Current job remains unfinished: {error}. "
            "Later jobs are held; reply with guidance or use /stop to cancel.",
        )

    async def _resume_paused_job(
        self, uid: str, chat_id: str, guidance: str, generation: int,
    ) -> bool:
        if uid not in self._paused_jobs:
            return False

        def resume() -> bool:
            if self._cancel_generations.get(uid, 0) != generation:
                return False
            data = self._read_queue_file()
            entries = self._queue_entries(data.get(uid))
            if not entries or entries[0].get("chat_id") != chat_id:
                return False
            job = entries[0].get("job")
            if not isinstance(job, dict) or job.get("state") != "paused":
                return False
            request = job.get("request") or entries[0].get("text", "")
            job["request"] = request + "\n\n[User guidance for the unfinished job]:\n" + guidance
            job["state"] = "running"
            for key in ("next_retry_at", "retry_count", "retry_reason"):
                job.pop(key, None)
            self._write_queue_file(data)
            return True

        async with self._ensure_dispatch_admission_lock(uid):
            resumed = await _run_durable_file_operation(self._queue_file_lock, resume)
        if self._cancel_generations.get(uid, 0) != generation:
            return False
        if resumed:
            self._paused_jobs.discard(uid)
            self._awaiting_answer.discard(uid)
            self._start_queue_drain(uid)
        return resumed

    async def _persist_enqueue(
        self, uid: str, text: str, chat_id: str,
        *, ephemeral: bool = False,
    ) -> str:
        entry_id = uuid.uuid4().hex[:12]
        entry = {
            "id": entry_id, "text": text, "chat_id": chat_id,
        }
        if ephemeral:
            entry["ephemeral"] = True
        def enqueue() -> None:
            data = self._read_queue_file()
            entries = self._queue_entries(data.get(uid))
            entries.append(entry)
            data[uid] = entries
            self._write_queue_file(data)

        await _run_durable_file_operation(self._queue_file_lock, enqueue)
        return entry_id

    async def _persist_admitted_dispatch_entry(
        self,
        uid: str,
        text: str,
        chat_id: str,
        generation: int,
        *,
        ephemeral: bool = False,
    ) -> str | None:
        """Persist a pre-cancellation dispatch, or reject it as stale.

        ``/stop`` advances the generation before it waits for this admission
        barrier and clears the durable queue while holding it. Consequently a
        dispatch that captured an older generation either writes before that
        clear or is rejected before it can write after it.
        """
        admission_lock = self._ensure_dispatch_admission_lock(uid)
        async with admission_lock:
            if self._cancel_generations.get(uid, 0) != generation:
                return None
            return await self._persist_enqueue(
                uid, text, chat_id, ephemeral=ephemeral,
            )

    async def _persist_complete(
        self, uid: str, entry_id: str,
    ) -> None:
        def complete() -> str | None:
            data = self._read_queue_file()
            entries = self._queue_entries(data.get(uid))
            if not entries:
                if uid in data:
                    data.pop(uid, None)
                    self._write_queue_file(data)
                return
            completed = next((entry for entry in entries if entry.get("id") == entry_id), {})
            job = completed.get("job", {})
            workspace_path = job.get("workspace_path") if isinstance(job, dict) else None
            remaining = [entry for entry in entries if entry.get("id") != entry_id]
            if len(remaining) == len(entries):
                return  # entry already gone
            if remaining:
                data[uid] = remaining
            else:
                data.pop(uid, None)
            self._write_queue_file(data)
            return workspace_path if isinstance(workspace_path, str) else None

        completed_workspace = await _run_durable_file_operation(self._queue_file_lock, complete)
        if completed_workspace:
            from .. import execution_state
            try:
                await execution_state.discard_execution(completed_workspace, entry_id)
            except (OSError, ValueError):
                logger.warning("Could not clean completed backend continuation", exc_info=True)

    async def _clear_persistent_queue(self, uid: str) -> int:
        running = self._running_tasks.get(uid)
        active = running is not None and not running.done()

        def clear() -> int:
            data = self._read_queue_file()
            entries = data.pop(uid, None)
            if entries is not None:
                self._write_queue_file(data)
                for entry in self._queue_entries(entries):
                    job = entry.get("job")
                    if (entry.get("ephemeral") is True and isinstance(job, dict)
                            and (not active or job.get("state") == "paused")
                            and isinstance(job.get("workspace_path"), str)
                            and isinstance(job.get("session_id"), str)):
                        try:
                            session.delete_session(job["workspace_path"], job["session_id"])
                        except Exception:
                            logger.warning("Failed to clean a cancelled scheduled-job session", exc_info=True)
                return len(self._queue_entries(entries)) or 1
            return 0

        return await _run_durable_file_operation(
            self._queue_file_lock, clear,
            on_complete=lambda _: self._paused_jobs.discard(uid),
        )

    async def restore_queues(self) -> None:
        """Rehydrate in-memory queues from disk and resume processing.

        Called once after the platform has started. For each user with
        persisted entries, refill the in-memory queue and spawn a drain
        task so the oldest entry runs first.
        """
        # Park prompts with staged replies.
        await self.restore_reply_deliveries()
        data = await _run_durable_file_operation(
            self._queue_file_lock, self._read_queue_file,
        )
        if not data:
            return

        drained_users: list[str] = []
        for uid, raw_entries in data.items():
            entries = self._queue_entries(raw_entries)
            if not entries:
                continue
            self._ensure_task_lock(uid)
            existing_q = self._message_queues.get(uid)
            existing_size = existing_q.qsize() if existing_q else 0
            msg_queue = self._ensure_message_queue(
                uid, min_size=existing_size + len(entries),
            )
            for entry in entries:
                # Oldest-first preserves ordering.
                msg_queue.put_nowait((
                    entry.get("text", ""),
                    entry.get("chat_id", ""),
                    entry.get("id", ""),
                    # Malformed flags must not make ephemeral turns.
                    entry.get("ephemeral") is True,
                ))
            first_job = entries[0].get("job")
            if isinstance(first_job, dict) and first_job.get("state") == "paused":
                self._paused_jobs.add(uid)
            drained_users.append(uid)

        for uid in drained_users:
            self._start_queue_drain(uid)

    def _reply_deliveries_file_path(self) -> str:
        """Return the per-platform ledger for completed text replies."""
        return self._platform_state_file_path("reply_deliveries")

    def _read_reply_deliveries_file(self) -> dict:
        return self._read_platform_state_file(
            self._reply_deliveries_file_path(), "reply delivery ledger",
        )

    def _write_reply_deliveries_file(self, data: dict) -> None:
        self._write_platform_state_file(
            self._reply_deliveries_file_path(), data,
        )

    @staticmethod
    def _reply_delivery_records(value: object) -> list[dict]:
        """Normalize staged text payloads from durable user-editable state."""
        if not isinstance(value, list):
            return []
        records: list[dict] = []
        required = ("id", "user_id", "chat_id")
        for value_item in value:
            record = _record_with_required_string_fields(value_item, required)
            if record is None:
                continue
            messages = record.get("messages")
            if not isinstance(messages, list) or not all(
                isinstance(message, str) for message in messages
            ):
                continue
            records.append({
                "id": record["id"],
                "user_id": record["user_id"],
                "chat_id": record["chat_id"],
                "messages": list(messages),
                "awaiting": record.get("awaiting") is True,
            })
        return records

    async def _list_reply_delivery_records(self) -> list[dict]:
        def read() -> list[dict]:
            data = self._read_reply_deliveries_file()
            return self._reply_delivery_records(data.get("deliveries"))

        return await _run_durable_file_operation(
            self._reply_delivery_file_lock, read,
        )

    def _reply_delivery_lock(self, uid: str) -> asyncio.Lock:
        """Return the per-user lock for delivery/cancellation ordering."""
        return ensure_lock(self._reply_delivery_locks, uid)

    def _recovery_delivery_lock(self, uid: str) -> asyncio.Lock:
        return ensure_lock(self._recovery_delivery_locks, uid)

    async def _get_reply_delivery_record(
        self, record_id: str,
    ) -> dict | None:
        """Load the live delivery record, if it has not been cancelled."""
        def read() -> dict | None:
            data = self._read_reply_deliveries_file()
            return next(
                (
                    record
                    for record in self._reply_delivery_records(
                        data.get("deliveries"),
                    )
                    if record["id"] == record_id
                ),
                None,
            )

        return await _run_durable_file_operation(
            self._reply_delivery_file_lock, read,
        )

    async def restore_reply_deliveries(self) -> None:
        """Block later prompts until any prior completed reply is delivered."""
        records = await self._list_reply_delivery_records()
        users = {record["user_id"] for record in records}
        self._pending_reply_delivery_users.intersection_update(users)
        self._pending_reply_delivery_users.update(users)

    def _reply_delivery_payload(
        self, ws: str, result: agent.AgentResult,
    ) -> tuple[list[str], bool]:
        """Extract the retry-safe text part of an agent result.

        Attachments deliberately remain on the normal best-effort path. A
        platform upload has no stable idempotency key, so retrying it after a
        lost acknowledgement could duplicate the file. Text has the same
        unavoidable at-least-once delivery boundary, but retaining it avoids
        rerunning the completed agent and its tools.
        """
        messages: list[str] = []
        awaiting = False
        for event in result.events:
            if event.kind != "text":
                continue
            text, _ = agent.extract_attachment_sources(event.content, ws)
            text, event_awaiting = agent.extract_await(text)
            awaiting = awaiting or event_awaiting
            if text:
                messages.append(text)
        if result.usage and config.get_show_usage():
            footer = agent.format_usage(result.usage)
            if footer:
                messages.append(footer)
        return messages, awaiting

    async def _stage_reply_delivery(
        self,
        uid: str,
        entry_id: str,
        chat_id: str,
        ws: str,
        result: agent.AgentResult,
        *,
        allow_await: bool,
    ) -> dict:
        """Persist a final text reply before any non-idempotent send."""
        messages, awaiting = self._reply_delivery_payload(ws, result)
        awaiting = awaiting and allow_await
        def stage() -> tuple[dict, bool]:
            data = self._read_reply_deliveries_file()
            records = self._reply_delivery_records(data.get("deliveries"))
            for record in records:
                if record["id"] == entry_id:
                    return record, False
            record = {
                "id": entry_id,
                "user_id": uid,
                "chat_id": chat_id,
                "messages": messages,
                "awaiting": awaiting,
            }
            records.append(record)
            data["deliveries"] = records
            self._write_reply_deliveries_file(data)
            return record, True

        def staged(outcome: tuple[dict, bool]) -> None:
            if outcome[1]:
                self._pending_reply_delivery_users.add(uid)
                # Starting is idempotent.
                self.start_detached_task_watcher()

        record, _ = await _run_durable_file_operation(
            self._reply_delivery_file_lock, stage, on_complete=staged,
        )
        return record

    async def _complete_reply_input(self, record: dict) -> bool:
        """Consume a staged reply's original prompt without running it.

        The two ledgers are separate files, so a crash can leave both an
        inbound prompt and its completed reply on disk. Complete the prompt
        before delivery, and remove a restored in-memory copy too, so the
        reply ledger remains the sole recovery source.
        """
        try:
            await self._persist_complete(record["user_id"], record["id"])
        except Exception:
            logger.warning(
                "Could not complete input for staged reply %s",
                record["id"], exc_info=True,
            )
            return False
        msg_queue = self._message_queues.get(record["user_id"])
        if msg_queue is not None:
            self._select_queue_entry(
                msg_queue,
                lambda entry: (
                    isinstance(entry, tuple)
                    and len(entry) == 4
                    and entry[2] == record["id"]
                ),
            )
        return True

    async def _remove_reply_delivery_record(self, record_id: str) -> None:
        def remove() -> str | None:
            data = self._read_reply_deliveries_file()
            records = self._reply_delivery_records(data.get("deliveries"))
            removed = next(
                (record for record in records if record["id"] == record_id),
                None,
            )
            if removed is None:
                return
            remaining = [record for record in records if record["id"] != record_id]
            if remaining:
                data["deliveries"] = remaining
            else:
                data.pop("deliveries", None)
            self._write_reply_deliveries_file(data)
            uid = removed["user_id"]
            if not any(record["user_id"] == uid for record in remaining):
                return uid
            return None

        def removed(cleared_user: str | None) -> None:
            if cleared_user is not None:
                self._pending_reply_delivery_users.discard(cleared_user)

        await _run_durable_file_operation(
            self._reply_delivery_file_lock, remove, on_complete=removed,
        )

    async def _clear_reply_deliveries(self, uid: str) -> int:
        """Discard delayed replies the user explicitly cancelled.

        A delivery which was already accepted by the platform cannot be
        recalled, but this lock guarantees a reply which has not begun its
        outbound send cannot race past this explicit cancellation.
        """
        await self._cancel_owned_deliveries(uid, self._reply_delivery_tasks)
        def clear() -> int:
            data = self._read_reply_deliveries_file()
            records = self._reply_delivery_records(data.get("deliveries"))
            remaining = [
                record for record in records if record["user_id"] != uid
            ]
            removed = len(records) - len(remaining)
            if removed:
                if remaining:
                    data["deliveries"] = remaining
                else:
                    data.pop("deliveries", None)
                self._write_reply_deliveries_file(data)
            return removed

        async with self._reply_delivery_lock(uid):
            return await _run_durable_file_operation(
                self._reply_delivery_file_lock, clear,
                on_complete=lambda _: self._pending_reply_delivery_users.discard(uid),
            )

    async def _deliver_staged_reply(
        self, record: dict, *, ws: str | None = None,
        result: agent.AgentResult | None = None,
        awaiting_uid: str | None = None,
    ) -> bool:
        """Own each delivery so explicit cancellation can interrupt its I/O."""
        uid = record["user_id"]
        if self._cancel_work_depths.get(uid, 0):
            return False
        task = asyncio.create_task(self._deliver_current_staged_reply(
            record, ws=ws, result=result, awaiting_uid=awaiting_uid,
        ))
        tasks = self._reply_delivery_tasks.setdefault(uid, set())
        tasks.add(task)
        try:
            return await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            return False
        finally:
            tasks.discard(task)
            if not tasks:
                self._reply_delivery_tasks.pop(uid, None)

    async def _deliver_current_staged_reply(
        self, record: dict, *, ws: str | None = None,
        result: agent.AgentResult | None = None,
        awaiting_uid: str | None = None,
    ) -> bool:
        """Deliver a staged reply once, retaining it after a send failure.

        The foreground path supplies ``result`` so attachments retain their
        existing best-effort behavior. Restart retries send only the already
        staged text payload and never invoke an agent again.
        """
        record_id = record["id"]
        uid = record["user_id"]
        # Re-check under the lock before sending.
        async with self._recovery_delivery_lock(uid), self._reply_delivery_lock(uid):
            record = await self._get_reply_delivery_record(record_id)
            if record is None:
                return False
            if not self.authorized(record["user_id"], record["chat_id"]):
                logger.warning(
                    "Discarding undeliverable reply for unauthorized "
                    "user=%s chat=%s",
                    record["user_id"], record["chat_id"],
                )
                if not await self._complete_reply_input(record):
                    return False
                await self._remove_reply_delivery_record(record_id)
                return True

            try:
                if not await self._complete_reply_input(record):
                    return False
                if result is not None:
                    assert ws is not None
                    await self._send_result(
                        record["chat_id"], ws, result, uid=awaiting_uid,
                    )
                else:
                    for text in record["messages"]:
                        await self.send_text(
                            record["chat_id"], text, rich=True,
                        )
                    if record["awaiting"]:
                        self._arm_awaiting_answer(record["user_id"])
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Failed to deliver completed reply %s; retaining it for "
                    "retry", record_id, exc_info=True,
                )
                return False

            try:
                await self._remove_reply_delivery_record(record_id)
            except Exception:
                # Keep the ledger for at-least-once recovery.
                logger.warning(
                    "Delivered completed reply %s but could not clear its "
                    "ledger", record_id, exc_info=True,
                )
                return False
        self._start_queue_drain(uid)
        return True

    async def _check_reply_deliveries(self) -> None:
        await self._poll_records_by_owner(
            await self._list_reply_delivery_records(),
            self._deliver_staged_reply,
            label="completed reply",
        )

    async def _poll_records_by_owner(
        self, records: list[dict], operation: Callable[[dict], Awaitable[object]],
        *, label: str,
    ) -> None:
        """Poll independent owners concurrently, preserving each owner's order.

        Fixed workers bound both live operations and task creation. A failed
        staged reply returns False, holding later replies for that same owner
        until recovery, while other owners can continue immediately.

        This is one bounded parallel pass. Its next pass waits for the slowest
        owner; existing provider/platform operation deadlines bound that wait.
        The three recovery classes have independent polling cycles.
        """
        grouped: dict[str, list[dict]] = {}
        for record in records:
            grouped.setdefault(record["user_id"], []).append(record)
        owners = iter(grouped.values())

        async def worker() -> None:
            for owner_records in owners:
                for record in owner_records:
                    try:
                        if await operation(record) is False:
                            break
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.warning(
                            "Could not poll %s %s", label, record["id"], exc_info=True,
                        )
                        break

        workers = [asyncio.create_task(worker()) for _ in range(
            min(_DURABLE_POLL_CONCURRENCY, len(grouped)),
        )]
        await _run_owned_poll_tasks(workers)

    def _detached_tasks_file_path(self) -> str:
        """Return this platform's durable detached-task ledger path."""
        return self._platform_state_file_path("detached_tasks")

    def _read_detached_tasks_file(self) -> dict:
        return self._read_platform_state_file(
            self._detached_tasks_file_path(), "detached task ledger",
        )

    def _write_detached_tasks_file(self, data: dict) -> None:
        self._write_platform_state_file(
            self._detached_tasks_file_path(), data,
        )

    @staticmethod
    def _detached_task_records(value: object) -> list[dict]:
        """Normalize durable records, ignoring stale/malformed entries."""
        if not isinstance(value, list):
            return []
        records: list[dict] = []
        required = (
            "id", "backend_name", "task_id", "workspace_path",
            "user_id", "chat_id",
        )
        for value_item in value:
            record = _record_with_required_string_fields(value_item, required)
            if record is None:
                continue
            record = dict(record)
            if not isinstance(record.get("session_id"), str):
                record["session_id"] = None
            for key in ("delivery_text", "last_reported_state"):
                if not isinstance(record.get(key), str):
                    record.pop(key, None)
            records.append(record)
        return records

    async def _list_detached_task_records(self) -> list[dict]:
        def read() -> list[dict]:
            data = self._read_detached_tasks_file()
            return self._detached_task_records(data.get("tasks"))

        return await _run_durable_file_operation(
            self._detached_task_file_lock, read,
        )

    async def _get_detached_task_record(self, record_id: str) -> dict | None:
        """Re-read a callback's identity after acquiring its delivery lock."""
        def read() -> dict | None:
            data = self._read_detached_tasks_file()
            return next(
                (record for record in self._detached_task_records(data.get("tasks"))
                 if record["id"] == record_id),
                None,
            )

        return await _run_durable_file_operation(
            self._detached_task_file_lock, read,
        )

    async def _register_detached_task(
        self,
        *,
        uid: str,
        chat_id: str,
        workspace_path: str,
        session_id: str | None,
        backend_name: str,
        task_id: str,
        validate: bool,
        generation: int | None = None,
    ) -> bool:
        """Persist a task before any completion callback can be sent.

        Parser-discovered task ids are validated against the provider's task
        list first. That prevents a model/tool transcript from forging a
        ``backgrounded`` line that attaches Cozter to an unrelated session.
        """
        if generation is None:
            generation = self._cancel_generations.get(uid, 0)
        if self._cancel_generations.get(uid, 0) != generation:
            return False
        if validate:
            try:
                backend = backends_agent.get_backend(backend_name)
                status = None
                for delay in (0.0, 0.25, 0.75):
                    if delay:
                        await asyncio.sleep(delay)
                    status = await backend.get_detached_task_status(
                        workspace_path, task_id,
                    )
                    if status is not None and status.state != "unknown":
                        break
                if status is None or status.state == "unknown":
                    logger.warning(
                        "Ignoring unverified detached task %s from %s",
                        task_id, backend_name,
                    )
                    return False
            except Exception:
                logger.warning(
                    "Could not validate detached task %s from %s",
                    task_id, backend_name, exc_info=True,
                )
                return False

        def register() -> bool:
            if self._cancel_generations.get(uid, 0) != generation:
                return False
            data = self._read_detached_tasks_file()
            records = self._detached_task_records(data.get("tasks"))
            if any(
                item["backend_name"] == backend_name
                and item["task_id"] == task_id
                and item["user_id"] == uid
                and item["chat_id"] == chat_id
                for item in records
            ):
                return False
            records.append({
                "id": uuid.uuid4().hex,
                "backend_name": backend_name,
                "task_id": task_id,
                "workspace_path": workspace_path,
                "user_id": uid,
                "chat_id": chat_id,
                "session_id": session_id,
            })
            data["tasks"] = records
            self._write_detached_tasks_file(data)
            return True

        def registered(created: bool) -> None:
            if created and self._cancel_generations.get(uid, 0) == generation:
                self.start_detached_task_watcher()

        async with self._ensure_dispatch_admission_lock(uid):
            if self._cancel_generations.get(uid, 0) != generation:
                return False
            await _run_durable_file_operation(
                self._detached_task_file_lock, register, on_complete=registered,
            )
        return self._cancel_generations.get(uid, 0) == generation

    async def _update_detached_task_record(
        self, record_id: str, **changes: object,
    ) -> None:
        def update() -> None:
            data = self._read_detached_tasks_file()
            records = self._detached_task_records(data.get("tasks"))
            changed = False
            for record in records:
                if record["id"] != record_id:
                    continue
                record.update(changes)
                changed = True
                break
            if changed:
                data["tasks"] = records
                self._write_detached_tasks_file(data)

        await _run_durable_file_operation(self._detached_task_file_lock, update)

    async def _remove_detached_task_record(self, record_id: str) -> None:
        def remove() -> None:
            data = self._read_detached_tasks_file()
            records = self._detached_task_records(data.get("tasks"))
            remaining = [record for record in records if record["id"] != record_id]
            if len(remaining) == len(records):
                return
            if remaining:
                data["tasks"] = remaining
            else:
                data.pop("tasks", None)
            self._write_detached_tasks_file(data)

        await _run_durable_file_operation(self._detached_task_file_lock, remove)

    def start_detached_task_watcher(self) -> None:
        """Start the durable completion watcher once per platform."""
        if (
            self._detached_task_watcher is not None
            and not self._detached_task_watcher.done()
        ):
            return
        self._detached_task_watcher = create_background_task(
            self._detached_task_loop(),
            name=f"{self.platform_id}:detached-tasks",
            log=logger,
        )

    async def stop_detached_task_watcher(self) -> None:
        task = self._detached_task_watcher
        self._detached_task_watcher = None
        if task is not None and not task.done():
            task.cancel()
            await await_cancelled(task)

    async def _detached_task_loop(self) -> None:
        async def poll(check: Callable[[], Awaitable[None]], label: str) -> None:
            while True:
                try:
                    await check()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Durable %s poll failed", label)
                await asyncio.sleep(_DETACHED_TASK_POLL_INTERVAL_SEC)

        # A slow provider status or reply send must not postpone due job
        # retries, nor stall another recovery class on every later cycle.
        tasks = [asyncio.create_task(poll(check, label), name=f"{self.platform_id}:{label}")
                 for check, label in (
                     (self._check_job_retries, "job-retries"),
                     (self._check_reply_deliveries, "reply-deliveries"),
                     (self._check_detached_tasks, "detached-completions"),
                 )]
        await _run_owned_poll_tasks(tasks)

    async def _check_job_retries(self) -> None:
        """Wake due prerequisite recovery without occupying a foreground turn.

        Only the durable head can be claimed. Admission and queue locks keep
        a concurrent /stop or user update from resurrecting cancelled work.
        Waiting is persisted, so a daemon restart preserves the retry time.
        """
        if self._update_restart_pending:
            return
        data = await _run_durable_file_operation(
            self._queue_file_lock, self._read_queue_file,
        )
        now = time.time()
        for uid, raw_entries in data.items():
            if uid in self._pending_reply_delivery_users:
                continue
            entries = self._queue_entries(raw_entries)
            if not entries:
                continue
            entry_id = entries[0].get("id")
            if not isinstance(entry_id, str) or not entry_id or not self._job_retry_due(entries[0], now):
                continue
            generation = self._cancel_generations.get(uid, 0)
            task_lock = self._ensure_task_lock(uid)
            async with self._ensure_dispatch_admission_lock(uid):
                def claim() -> bool:
                    if (self._update_restart_pending
                            or self._cancel_work_depths.get(uid, 0)
                            or uid in self._pending_reply_delivery_users
                            or self._cancel_generations.get(uid, 0) != generation
                            or task_lock.locked()):
                        return False
                    current = self._read_queue_file()
                    rows = self._queue_entries(current.get(uid))
                    if (not rows or rows[0].get("id") != entry_id
                            or not self._job_retry_due(rows[0], time.time())):
                        return False
                    job = rows[0]["job"]
                    job["state"] = "running"
                    job.pop("next_retry_at", None)
                    self._write_queue_file(current)
                    return True

                def claimed(ready: bool) -> None:
                    if ready and self._cancel_generations.get(uid, 0) == generation:
                        self._paused_jobs.discard(uid)

                ready = await _run_durable_file_operation(
                    self._queue_file_lock, claim, on_complete=claimed,
                )
                if (ready and self._cancel_generations.get(uid, 0) == generation
                        and not self._cancel_work_depths.get(uid, 0)):
                    await self._retain_job_entry(uid, entry_id)
            if (ready and self._cancel_generations.get(uid, 0) == generation
                    and not self._update_restart_pending
                    and not self._cancel_work_depths.get(uid, 0)):
                self._start_queue_drain(uid)

    @staticmethod
    def _job_retry_due(entry: dict, now: float) -> bool:
        job = entry.get("job")
        if not isinstance(job, dict) or job.get("state") != "paused":
            return False
        retry_at = job.get("next_retry_at")
        if not isinstance(retry_at, (int, float)) or isinstance(retry_at, bool):
            return False
        try:
            return math.isfinite(retry_at) and 0 <= retry_at <= now
        except OverflowError:
            return False

    async def _check_detached_tasks(self) -> None:
        await self._poll_records_by_owner(
            await self._list_detached_task_records(), self._check_detached_task,
            label="detached task",
        )

    async def _check_detached_task(self, record: dict) -> None:
        """Poll/deliver one record. A final payload is persisted before I/O."""
        if not self.authorized(record["user_id"], record["chat_id"]):
            logger.warning(
                "Discarding detached task for unauthorized user=%s chat=%s",
                record["user_id"], record["chat_id"],
            )
            await self._remove_detached_task_record(record["id"])
            return

        delivery_text = record.get("delivery_text")
        if isinstance(delivery_text, str) and delivery_text:
            await self._deliver_detached_task(record, delivery_text)
            return

        try:
            backend = backends_agent.get_backend(record["backend_name"])
            status = await backend.get_detached_task_status(
                record["workspace_path"], record["task_id"],
            )
        except Exception:
            logger.warning(
                "Could not poll detached task %s", record["task_id"],
                exc_info=True,
            )
            return

        if status is None:
            await self._stage_detached_task_delivery(
                record,
                f"Background task {record['task_id']} is no longer available.",
            )
            return
        if status.state == "unknown":
            return
        if status.state == "working":
            if record.get("last_reported_state"):
                await self._update_detached_task_record(
                    record["id"], last_reported_state="",
                )
            return
        if status.state == "blocked":
            if record.get("last_reported_state") == "blocked":
                return
            try:
                output = await backend.get_detached_task_output(
                    record["workspace_path"], record["task_id"],
                )
            except Exception:
                output = ""
            reason = f" ({status.waiting_for})" if status.waiting_for else ""
            message = (
                f"Background task {record['task_id']} needs input{reason}.\n\n"
                "Use Claude Code's agent view or `claude attach` to answer it."
            )
            if output:
                message += f"\n\nRecent output:\n{output}"
            await self._report_blocked_detached_task(record, message)
            return
        if status.state not in _DETACHED_TERMINAL_STATES:
            logger.warning(
                "Detached task %s has unrecognized state %r",
                record["task_id"], status.state,
            )
            return
        try:
            output = await backend.get_detached_task_output(
                record["workspace_path"], record["task_id"],
            )
        except Exception:
            logger.warning(
                "Could not collect completed detached task %s",
                record["task_id"], exc_info=True,
            )
            return
        label = {
            "done": "completed",
            "failed": "failed",
            "stopped": "stopped",
        }[status.state]
        message = f"Background task {record['task_id']} {label}."
        if output:
            message += f"\n\n{output}"
        await self._stage_detached_task_delivery(record, message)

    async def _report_blocked_detached_task(self, record: dict, message: str) -> None:
        """Serialize a needs-input notification against explicit cancellation."""
        await self._run_detached_delivery(
            record["user_id"], self._report_live_blocked_detached_task(record, message),
        )

    def _detached_task_delivery_lock(self, uid: str) -> asyncio.Lock:
        return ensure_lock(self._detached_delivery_locks, uid)

    async def _run_detached_delivery(self, uid: str, delivery: Coroutine[None, None, None]) -> None:
        """Own each outbound callback so cancellation can interrupt stalled I/O."""
        if self._cancel_work_depths.get(uid, 0):
            delivery.close()
            return
        task = asyncio.create_task(delivery)
        tasks = self._detached_delivery_tasks.setdefault(uid, set())
        tasks.add(task)
        try:
            await task
        except asyncio.CancelledError:
            # A user cancelled this child send; keep the shared watcher alive.
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        finally:
            tasks.discard(task)
            if not tasks:
                self._detached_delivery_tasks.pop(uid, None)

    async def _report_live_blocked_detached_task(self, record: dict, message: str) -> None:
        uid = record["user_id"]
        async with self._recovery_delivery_lock(uid), self._detached_task_delivery_lock(uid):
            if uid in self._pending_reply_delivery_users:
                return  # Keep the notification eligible after the owed reply.
            record = await self._get_detached_task_record(record["id"])
            if record is None or record.get("last_reported_state") == "blocked":
                return
            if not self.authorized(record["user_id"], record["chat_id"]):
                return
            try:
                await self.send_text(record["chat_id"], message, rich=True)
            except Exception:
                logger.warning(
                    "Failed to report blocked detached task %s",
                    record["task_id"], exc_info=True,
                )
                return
            await self._update_detached_task_record(
                record["id"], last_reported_state="blocked",
            )

    async def _stage_detached_task_delivery(
        self, record: dict, message: str,
    ) -> None:
        """Durably save a completion payload before platform delivery."""
        await self._update_detached_task_record(
            record["id"], delivery_text=message,
        )
        staged = dict(record)
        staged["delivery_text"] = message
        await self._deliver_detached_task(staged, message)

    async def _deliver_detached_task(self, record: dict, message: str) -> None:
        """Deliver an already-persisted completion, then remove its ledger row."""
        await self._run_detached_delivery(
            record["user_id"], self._deliver_current_detached_task(record, message),
        )

    async def _deliver_current_detached_task(self, record: dict, message: str) -> None:
        uid = record["user_id"]
        async with self._recovery_delivery_lock(uid), self._detached_task_delivery_lock(uid):
            if uid in self._pending_reply_delivery_users:
                return  # Its staged payload remains durable for the next poll.
            live = await self._get_detached_task_record(record["id"])
            if live is None:
                return
            await self._deliver_live_detached_task(live, message)

    async def _deliver_live_detached_task(self, record: dict, message: str) -> None:
        """Send a live callback while holding the user's delivery lock."""
        if not self.authorized(record["user_id"], record["chat_id"]):
            await self._remove_detached_task_record(record["id"])
            return
        result = agent.AgentResult()
        agent.append_text_result(result, message)
        try:
            # Uploads stay user-owned.
            await self._send_detached_result(
                record["chat_id"], record["workspace_path"], result,
                record["user_id"],
            )
        except Exception:
            # Keep the staged payload.
            logger.warning(
                "Failed to deliver detached task %s",
                record["task_id"], exc_info=True,
            )
            return
        await agent.log_detached_task_completion(
            record["workspace_path"], record.get("session_id"), message,
        )
        await self._remove_detached_task_record(record["id"])

    async def _register_result_detached_tasks(
        self,
        uid: str,
        chat_id: str,
        workspace_path: str,
        result: agent.AgentResult,
        *,
        session_id: str | None,
    ) -> None:
        for task in result.detached_tasks:
            await self._register_detached_task(
                uid=uid,
                chat_id=chat_id,
                workspace_path=workspace_path,
                session_id=session_id,
                backend_name=task.backend_name,
                task_id=task.task_id,
                validate=True,
            )

    async def _launch_result_detached_tasks(
        self,
        uid: str,
        chat_id: str,
        workspace_path: str,
        result: agent.AgentResult,
        *,
        model: str | None,
        summary_model: str | None,
        approval: str,
        backend_name: str,
        summary_backend_name: str,
    ) -> None:
        """Start agent-requested durable tasks and queue their callbacks."""
        requests = result.detached_task_requests
        if not requests:
            return
        if result.error:
            agent.append_text_result(
                result,
                "I did not start the requested background task because the "
                "foreground agent run failed.",
            )
            return
        backend = backends_agent.get_backend(backend_name)
        if not getattr(backend, "supports_detached_tasks", False):
            agent.append_text_result(
                result,
                f"{backend.name} does not support restart-safe background "
                "tasks on this Cozter installation.",
            )
            return

        for request in requests:
            generation = self._cancel_generations.get(uid, 0)
            try:
                launch = await agent.launch_detached(
                    request.prompt,
                    workspace_path,
                    uid,
                    model=model,
                    summary_model=summary_model,
                    approval=approval,
                    backend_name=backend_name,
                    summary_backend_name=summary_backend_name,
                )
            except Exception as exc:
                logger.exception("Could not start agent-requested background task")
                agent.append_text_result(
                    result, f"Could not start background task: {exc}",
                )
                continue
            try:
                tracked = await self._register_detached_task(
                    uid=uid,
                    chat_id=chat_id,
                    workspace_path=workspace_path,
                    session_id=launch.session_id,
                    backend_name=launch.backend_name,
                    task_id=launch.task_id,
                    validate=True,
                    generation=generation,
                )
                if self._cancel_generations.get(uid, 0) != generation:
                    raise asyncio.CancelledError
            except asyncio.CancelledError:
                await self._cleanup_cancelled_detached_launch(uid, workspace_path, launch)
                raise
            except Exception:
                logger.exception(
                    "Could not persist agent-requested detached task %s",
                    launch.task_id,
                )
                tracked = False
            if tracked:
                agent.append_text_result(
                    result,
                    f"Started background task {launch.task_id}. "
                    "I’ll post its final result here.",
                )
            else:
                agent.append_text_result(
                    result,
                    f"Started background task {launch.task_id}, but Cozter "
                    "could not verify it for completion delivery. Check it "
                    f"with `claude agents` or `claude logs {launch.task_id}`.",
                )

    async def _cancel_detached_tasks(self, uid: str) -> int:
        """Cancel/dismiss every tracked detached task owned by one user."""
        await self._cancel_owned_deliveries(uid, self._detached_delivery_tasks)
        async with self._detached_task_delivery_lock(uid):
            return await self._cancel_detached_tasks_locked(uid)

    async def _cancel_owned_deliveries(
        self, uid: str, owned_tasks: dict[str, set[asyncio.Task]],
    ) -> None:
        tasks = tuple(owned_tasks.get(uid, ()))
        for task in tasks:
            task.cancel()
        for task in tasks:
            await await_cancelled(task)

    async def _cancel_detached_tasks_locked(self, uid: str) -> int:
        """Remove detached callbacks while excluding sends for this user."""
        cancelled = 0
        for record in await self._list_detached_task_records():
            if record["user_id"] != uid:
                continue
            # A staged terminal payload: /cancel drops the delayed message.
            if isinstance(record.get("delivery_text"), str):
                await self._remove_detached_task_record(record["id"])
                cancelled += 1
                continue
            try:
                backend = backends_agent.get_backend(record["backend_name"])
                stopped = await backend.stop_detached_task(
                    record["workspace_path"], record["task_id"],
                )
            except Exception:
                logger.warning(
                    "Could not cancel detached task %s",
                    record["task_id"], exc_info=True,
                )
                stopped = False
            if stopped:
                await self._remove_detached_task_record(record["id"])
                cancelled += 1
        return cancelled

    async def start_scheduler(self) -> None:
        """Kick off the background scheduler task. Idempotent."""
        if self._scheduler_task and not self._scheduler_task.done():
            return
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop_scheduler(self) -> None:
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            await await_cancelled(self._scheduler_task)
        self._scheduler_task = None

    async def _scheduler_loop(self) -> None:
        """Poll every 30s; fire schedules whose day+time match 'now'."""
        # Let the platform start.
        await asyncio.sleep(5)
        while True:
            try:
                await self._scheduler_tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler tick failed")
            await asyncio.sleep(30)

    async def _scheduler_tick(self) -> None:
        """Fire every schedule whose most-recent matching slot hasn't run.

        A schedule's "slot" is the most recent (day, time) match at or
        before ``now``. If that slot is newer than the schedule's
        persisted ``last_fired`` (or the schedule has never fired and
        the slot is newer than ``created``), fire it.

        Because ``last_fired`` lives in the session JSON on disk, this
        correctly catches up after a bot restart: any slot that should
        have run while the bot was down will fire once on the next tick.
        """
        now = datetime.now()

        # Same-slot fires in creation order.
        to_fire: list[tuple[str, str, dict, datetime]] = []

        # Iterate workspace state.
        for uid, ws in workspace.iter_current_workspaces(self.platform_id):
            if not os.path.isdir(ws):
                continue
            for sched in self._platform_schedules(ws, uid):
                schedule_id = sched.get("id")
                if not isinstance(schedule_id, str) or not schedule_id:
                    continue
                slot = schedules.most_recent_slot(sched, now)
                if slot is None:
                    continue
                last_fired = schedules.parse_iso(sched.get("last_fired"))
                baseline = last_fired or schedules.parse_iso(
                    sched.get("created"),
                )
                # Never fire at/before creation.
                if baseline is not None and slot <= baseline:
                    continue
                to_fire.append((uid, ws, sched, slot))

        to_fire.sort(key=lambda item: str(item[2].get("created") or ""))

        for uid, ws, sched, slot in to_fire:
            schedule_id = sched.get("id")
            if not isinstance(schedule_id, str) or not schedule_id:
                continue
            # Persist last_fired first.
            async with workspace.get_lock(ws):
                current = next(
                    (record for record in self._platform_schedules(ws, uid)
                     if record.get("id") == schedule_id),
                    None,
                )
                if current is None:
                    continue
                # Days/time can change while this tick waits for the lock.
                # Claim only the current schedule's slot, never its snapshot.
                slot = schedules.most_recent_slot(current, now)
                if slot is None:
                    continue
                baseline = schedules.parse_iso(current.get("last_fired")) or schedules.parse_iso(
                    current.get("created"),
                )
                if baseline is not None and slot <= baseline:
                    # Lost the lock race; another tick claimed it.
                    continue
                claimed_schedule = schedules.update_schedule_fired(
                    ws, uid, schedule_id, slot.isoformat(),
                )
            # Re-check under the lock; snapshots can go stale.
            if claimed_schedule is not None:
                await self._fire_schedule(uid, claimed_schedule)

    async def _fire_schedule(self, uid: str, sched: dict) -> None:
        """Push a scheduled command onto the user's message queue.

        The queue entry is flagged ``ephemeral=True`` so the drain loop
        runs it in a job-owned session kept until completion or cancellation,
        rather than appending to whichever session was current at the
        time the schedule fires.
        """
        command = sched.get("command", "")
        chat_id = sched.get("chat_id", "")
        generation = self._cancel_generations.get(uid, 0)
        if sched.get("platform_id") not in (None, "", self.platform_id):
            return
        if not command or not chat_id:
            return

        # Re-check auth: creators may be deauthorized.
        if not self.authorized(uid, chat_id):
            logger.warning(
                "Skipping schedule for unauthorized user=%s chat=%s",
                uid, chat_id,
            )
            return

        # Scheduler can fire before first input.
        self._ensure_task_lock(uid)
        msg_queue = self._ensure_message_queue(uid)

        # Check capacity first.
        if msg_queue.full():
            await self._run_status_operation(
                self.send_text(
                    chat_id, f"Queue full — dropped scheduled command: {command}",
                ),
                action="announcing a dropped scheduled task",
            )
            return

        entry_id = await self._persist_admitted_dispatch_entry(
            uid, command, chat_id, generation, ephemeral=True,
        )
        if entry_id is None:
            return
        if self._cancel_generations.get(uid, 0) != generation:
            await self._discard_cancelled_dispatch_entry(uid, entry_id)
            return
        await msg_queue.put((command, chat_id, entry_id, True))
        if self._cancel_generations.get(uid, 0) != generation:
            await self._discard_cancelled_dispatch_entry(uid, entry_id, msg_queue)
            return

        try:
            await self._run_status_operation(
                self.send_text(chat_id, f"⏰ Queued scheduled task: {command}"),
                action="announcing a queued scheduled task",
            )
        finally:
            # Admitted work must wake even if its acknowledgement is cancelled.
            if self._cancel_generations.get(uid, 0) == generation:
                self._start_queue_drain(uid)

    # AI chat + file

    async def _require_ws(self, ctx: BotContext) -> str | None:
        ws = workspace.get_current(ctx.user_id, self.platform_id)
        if not ws or not os.path.isdir(ws):
            await ctx.reply_text(NO_WORKSPACE_TEXT)
            return None
        try:
            workspace.ensure_cozter_dir(ws)
        except ValueError:
            await ctx.reply_text(
                "Workspace state is unsafe. Choose another workspace with /open."
            )
            return None
        return ws

    async def _ai_chat(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        await self._dispatch_ai(ctx, ctx.text.strip())

    async def _ai_file(self, ctx: BotContext) -> None:
        ws = await self._require_ws(ctx)
        if ws is None:
            return
        att = ctx.attachment
        assert att is not None  # dispatch_file already checked

        rel_path = os.path.relpath(att.local_path, ws)
        parts: list[str] = []
        if att.caption:
            parts.append(att.caption)
        parts.append(
            f"[{att.kind.capitalize()} attachment saved to: {rel_path}]"
        )

        ext = os.path.splitext(att.filename)[1].lower()
        if (
            ext in IMAGE_EXTENSIONS or att.kind in ("photo", "video")
        ) and ext not in _TEXT_EXTENSIONS:
            try:
                dims = await asyncio.to_thread(
                    probe_image_dimensions, att.local_path,
                )
                size = 0
                try:
                    size = os.path.getsize(att.local_path)
                except OSError:
                    size = 0
                if dims is not None:
                    width, height, fmt = dims
                    parts.append(
                        f"[Image: {att.filename} is a {width}x{height} {fmt}"
                        f" ({size:,} bytes) at {rel_path}."
                        " Vision-capable backends receive these pixels"
                        " natively: describe what is actually in the image."
                        " If your backend cannot see pixels, say what you"
                        " verified (dimensions, format, size) and ask the"
                        " user to describe the content.]"
                    )
                elif size:
                    parts.append(
                        f"[Image: {att.filename} ({size:,} bytes) at"
                        f" {rel_path}. Vision-capable backends receive its"
                        " pixels natively; otherwise say what you verified"
                        " and ask the user to describe the"
                        " content if needed.]"
                    )
            except OSError:
                pass
        if ext in _TEXT_EXTENSIONS:
            try:
                # Read the decision boundary off-loop: never decode megabytes to decline.
                content = await asyncio.to_thread(
                    _read_inline_text_attachment, att.local_path,
                )
                if len(content) <= _INLINE_SIZE_LIMIT:
                    parts.append(
                        f"[File contents of {att.filename}]\n"
                        f"{content}\n"
                        "[End of file]"
                    )
                else:
                    parts.append(
                        "[File too large to inline (over "
                        f"{_INLINE_SIZE_LIMIT:,} chars);"
                        f" read it from {rel_path}]"
                    )
            except OSError:
                pass

        await self._dispatch_ai(ctx, "\n".join(parts))

    # dispatch + AI turn

    def _cleanup_turn(self, uid: str, lock: asyncio.Lock) -> None:
        self._running_tasks.pop(uid, None)
        self._inject_queues.pop(uid, None)
        self._cancel_acknowledged.discard(uid)
        self._clear_turn_progress(uid)
        lock.release()

    async def _discard_cancelled_dispatch_entry(
        self, uid: str, entry_id: str, msg_queue: asyncio.Queue | None = None,
    ) -> None:
        """Remove an entry whose dispatch began before /cancel or /stop.

        The queue file may already have been cleared by the command.  Still
        complete by id because a dispatch can persist after that clear while
        it was suspended waiting on the queue-file lock.
        """
        if msg_queue is not None:
            self._select_queue_entry(
                msg_queue,
                lambda entry: (
                    isinstance(entry, tuple)
                    and len(entry) == 4
                    and entry[2] == entry_id
                ),
            )
        try:
            await self._persist_complete(uid, entry_id)
        except Exception:
            logger.warning(
                "Failed to discard cancelled queue entry %s for user %s",
                entry_id, uid, exc_info=True,
            )

    async def _persist_and_queue_dispatch(
        self,
        uid: str,
        text: str,
        chat_id: str,
        generation: int,
        msg_queue: asyncio.Queue,
    ) -> str | None:
        """Persist and enqueue a dispatch unless cancellation made it stale."""
        entry_id = await self._persist_admitted_dispatch_entry(
            uid, text, chat_id, generation,
        )
        if entry_id is None:
            return None
        if self._cancel_generations.get(uid, 0) != generation:
            await self._discard_cancelled_dispatch_entry(uid, entry_id)
            return None
        await msg_queue.put((text, chat_id, entry_id, False))
        if self._cancel_generations.get(uid, 0) != generation:
            await self._discard_cancelled_dispatch_entry(uid, entry_id, msg_queue)
            return None
        return entry_id

    async def _dispatch_ai(self, ctx: BotContext, text: str) -> None:
        uid = ctx.user_id
        chat_id = ctx.chat_id
        generation = self._cancel_generations.get(uid, 0)
        if await self._resume_paused_job(uid, chat_id, text, generation):
            await ctx.reply_text("Resuming the unfinished job with your guidance.")
            return
        if self._update_restart_pending:
            self._ensure_task_lock(uid)
            msg_queue = self._ensure_message_queue(uid)
            if msg_queue.full():
                await ctx.reply_text("Queue full. Try again after restart.")
            else:
                entry_id = await self._persist_and_queue_dispatch(
                    uid, text, chat_id, generation, msg_queue,
                )
                if entry_id is None:
                    return
                await self._resume_awaiting_answer(
                    uid, msg_queue, entry_id, reason="update was pending",
                )
                await ctx.reply_text(
                    f"Queued ({msg_queue.qsize()}/{self.max_queue_size})."
                )
            return

        lock = self._ensure_task_lock(uid)

        if (lock.locked() or uid in self._pending_reply_delivery_users
                or uid in self._paused_jobs):
            msg_queue = self._ensure_message_queue(uid)
            if msg_queue.full():
                await ctx.reply_text("Queue full. Wait or /stop first.")
            else:
                was_awaiting = uid in self._awaiting_answer
                entry_id = await self._persist_and_queue_dispatch(
                    uid, text, chat_id, generation, msg_queue,
                )
                if entry_id is None:
                    return
                if was_awaiting:
                    await self._resume_awaiting_answer(
                        uid, msg_queue, entry_id, reason="turn was finishing",
                    )
                await ctx.reply_text(
                    f"Queued ({msg_queue.qsize()}/{self.max_queue_size})."
                )
                # Race guard: kick a drain so the entry isn't orphaned.
                self._start_queue_drain(uid)
            return

        # Persist before locking.
        entry_id = await self._persist_admitted_dispatch_entry(
            uid, text, chat_id, generation,
        )
        if entry_id is None:
            return
        await lock.acquire()
        if self._cancel_generations.get(uid, 0) != generation:
            try:
                await self._discard_cancelled_dispatch_entry(uid, entry_id)
            finally:
                lock.release()
            return
        # Park in RAM too so an aborted update can resume it without restart.
        if self._update_restart_pending:
            lock.release()
            queued = self._message_queues.get(uid)
            msg_queue = self._ensure_message_queue(
                uid,
                min_size=(queued.qsize() if queued is not None else 0) + 1,
            )
            msg_queue.put_nowait((text, chat_id, entry_id, False))
            await self._resume_awaiting_answer(
                uid, msg_queue, entry_id, reason="update became pending",
            )
            await ctx.reply_text(
                f"Queued ({msg_queue.qsize()}/{max(self.max_queue_size, msg_queue.qsize())})."
            )
            # Start anyway: covers the no-update/cancel race after placement.
            self._start_queue_drain(uid)
            return
        # [[await]]: clear only after lock.
        self._awaiting_answer.discard(uid)
        # Register early so /stop finds the task.
        self._running_tasks[uid] = asyncio.current_task()
        try:
            completed = await self._run_turn(
                uid, chat_id, text, queue_entry_id=entry_id,
            )
        except asyncio.CancelledError:
            # Shutdown keeps the entry for resume; don't mask the clean cancel.
            if uid not in self._cancel_acknowledged:
                await self._send_text_best_effort(chat_id, "Cancelled.")
            return
        except Exception as e:
            logger.exception("AI turn failed")
            await self._pause_job_after_failure(uid, entry_id, chat_id, e)
        else:
            if completed is False:
                await self._retain_job_entry(uid, entry_id)
            else:
                await self._persist_complete(uid, entry_id)
        finally:
            self._cleanup_turn(uid, lock)

        await self._drain_message_queue(uid)

    @staticmethod
    def _compose_thinking_display(
        status_lines: list[str], latest_text: str,
    ) -> str:
        """Build the live 'Thinking...' preview.

        Recent tool/file activity plus the latest streamed answer text
        (tail-truncated so the newest content is kept). Rendered into the
        editable status message during a turn. The final reply is sent
        independently, so a slow status cleanup cannot delay it.
        """
        parts = ["Thinking..."]
        if status_lines:
            if len(status_lines) > 5:
                parts.append(
                    f"… [{len(status_lines) - 5} older update(s) omitted"
                    " — preview only]",
                )
            parts.append("\n".join(status_lines[-5:]))
        text = latest_text.strip()
        if text:
            if len(text) > 600:
                _marker = "… [preview] "
                text = _marker + text[-(600 - len(_marker)):]
            parts.append(text)
        return "\n\n".join(parts)

    async def _run_turn(
        self, uid: str, chat_id: str, text: str,
        *, session_id: str | None = None,
        queue_entry_id: str | None = None,
    ) -> bool:
        """Keep one job current until its full scope is complete.

        Each execution allowance is a checkpoint boundary, not a queue
        completion. The original request, progress and pinned session remain
        durable before a restart can yield to anything else.
        """
        job = await self._load_job_checkpoint(uid, queue_entry_id) if queue_entry_id else {}
        ws = job.get("workspace_path")
        if not isinstance(ws, str) or not ws:
            ws = await self._current_workspace_for_turn(uid, chat_id)
        if ws is None or not os.path.isdir(ws):
            if queue_entry_id:
                job.update({"state": "paused", "request": text,
                            "instruction": "Restore the job's workspace and finish the original scope."})
                await self._save_job_checkpoint(uid, queue_entry_id, job)
            self._paused_jobs.add(uid)
            return False
        request = job.get("request")
        if not isinstance(request, str) or not request:
            request = text
        pinned_session = job.get("session_id")
        if not isinstance(pinned_session, str) or not pinned_session:
            pinned_session = session_id
        if job.get("state") == "paused":
            self._paused_jobs.add(uid)
            return False

        while True:
            job.update({"workspace_path": ws, "session_id": pinned_session,
                        "request": request, "state": "running"})
            if queue_entry_id and not await self._save_job_checkpoint(uid, queue_entry_id, job):
                return False
            prompt = request
            checkpoint = job.get("checkpoint")
            instruction = job.get("instruction")
            if checkpoint or instruction:
                prompt += (
                    "\n\n[Resume the SAME unfinished job; later queued jobs remain held.]\n"
                    "Keep the original full scope above. Re-check saved artifacts and "
                    "working-tree changes before doing more work. Use prior worker "
                    "evidence to avoid repeating verified completed steps. Reports may "
                    "be partial or truncated; inspect their artifacts to verify them. "
                    "Complete every remaining obligation, including final verification "
                    "and requested commit/push. Return DONE only when the full original "
                    "request is complete; otherwise state the exact remainder.\n"
                    + (instruction if isinstance(instruction, str) else "")
                    + "\n[Saved progress checkpoint]:\n"
                    + json.dumps(checkpoint or {}, ensure_ascii=False)
                )
            previous = self._job_output(job.get("output"))

            async def session_ready(selected_session: str) -> None:
                nonlocal pinned_session
                pinned_session = selected_session
                job["session_id"] = selected_session
                if queue_entry_id and not await self._save_job_checkpoint(uid, queue_entry_id, job):
                    raise asyncio.CancelledError

            with agent.turn_budget(fresh=True):
                result = await self._run_turn_with_budget(
                    uid, chat_id, prompt, session_id=pinned_session,
                    queue_entry_id=queue_entry_id,
                    workspace_path=ws, previous_result=previous,
                    on_session=session_ready,
                    original_request=request,
                )
            if result is None:
                self._paused_jobs.discard(uid)
                return True

            checkpoint = result.continuation_checkpoint or {}
            additions = checkpoint.get("accepted_additions", [])
            if checkpoint.get("accepted_additions_truncated"):
                checkpoint_session = result.session_id or pinned_session
                data = (await asyncio.to_thread(session.load_session, ws, checkpoint_session)
                        if checkpoint_session else None)
                messages = data.get("messages", []) if isinstance(data, dict) else []
                full_request = next((message.get("content") for message in reversed(messages)
                                     if isinstance(message, dict) and message.get("role") == "user"
                                     and isinstance(message.get("content"), str)), "")
                if full_request.startswith(prompt):
                    request += full_request[len(prompt):]
                else:
                    # Never silently accept completion with known omitted
                    # user requirements when the full source is unavailable.
                    result.continuation_reason = None
                    result.continue_instruction = ""
                    result.completion_verified = False
                    result.error = "Cannot recover the full accepted job updates from its saved session."
            elif isinstance(additions, list):
                for addition in additions:
                    if isinstance(addition, str) and addition:
                        request += "\n\n[Added while working]:\n" + addition
            pinned_session = result.session_id or pinned_session
            resumable = bool(result.continuation_reason or (
                result.continue_instruction and not result.error
            ) or (not result.error and result.completion_verified is not False
                  and result.text.lstrip().upper().startswith("PARTIAL")))
            job.update({
                "request": request, "session_id": pinned_session,
                "instruction": result.continue_instruction or (
                    "Finish the remaining original scope and verify completion.\n"
                    + (result.text or result.error or "No completed answer was produced.")
                ),
                "checkpoint": self._merge_job_checkpoint(job.get("checkpoint"), checkpoint, result),
                "attempts": (job.get("attempts", 0) if isinstance(job.get("attempts", 0), int)
                             and not isinstance(job.get("attempts", 0), bool) else 0) + 1,
                "state": "running" if resumable else "paused",
                "output": self._job_output_record(result, ws),
            })
            prerequisite_retry = result.continuation_reason == "prerequisite"
            if prerequisite_retry:
                previous_retries = job.get("retry_count", 0)
                if (not isinstance(previous_retries, int) or isinstance(previous_retries, bool)
                        or previous_retries < 0):
                    previous_retries = 0
                retries = min(previous_retries + 1, 1_000_000)
                delay = min(30 * 2 ** min(retries - 1, 4), 300)
                job.update({
                    "state": "paused", "next_retry_at": time.time() + delay,
                    "retry_count": retries,
                    "retry_reason": result.blocked_reason or result.continue_instruction,
                })
                self._paused_jobs.add(uid)
            if queue_entry_id and not await self._save_job_checkpoint(uid, queue_entry_id, job):
                return False
            if prerequisite_retry:
                self.start_detached_task_watcher()
                await self._run_status_operation(
                    self.send_status(
                        chat_id, "Completed work is saved. Retrying prerequisite recovery "
                        f"automatically in {delay} seconds; the current job keeps its place.",
                    ),
                    action="announcing automatic prerequisite recovery",
                )
                return False
            if not resumable:
                self._paused_jobs.add(uid)
                explanation = ("Completion could not be verified.\n\nLatest draft:\n"
                               if result.completion_verified is False else "")
                paused_text, _ = agent.extract_await(result.text or result.error or
                                                     "No completed answer was produced.")
                await self._send_text_best_effort(
                    chat_id,
                    explanation + paused_text
                    + "\n\nThe current job is still unfinished. Later jobs are held. "
                    "Reply with guidance to resume it, or use /stop to cancel.",
                )
                return False
            await self._run_status_operation(
                self.send_status(chat_id, "Saved progress; continuing the current "
                                 "job before starting any later queued job."),
                action="announcing continued work",
            )
            if self._update_restart_pending:
                return False  # queue checkpoint survives the updater's restart
            await asyncio.sleep(0)

    @staticmethod
    def _merge_job_checkpoint(previous: object, current: dict, result: agent.AgentResult) -> dict:
        """Retain worker evidence across attempts without nesting resume prompts."""
        checkpoint = {key: value for key, value in current.items()
                      if key not in {"request", "accepted_additions", "reports"}}
        if not current:
            checkpoint["pending"] = [result.continue_instruction] if result.continue_instruction else []
            checkpoint["latest_draft"] = result.text[:12_000]
        reports = []
        seen = set()
        for source in (previous, current):
            rows = source.get("reports", []) if isinstance(source, dict) else []
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                instruction, report = row.get("instruction"), row.get("report")
                if not isinstance(instruction, str) or not isinstance(report, str):
                    continue
                key = (instruction, report)
                if key not in seen:
                    seen.add(key)
                    reports.append({"instruction": instruction, "report": report})
        truncated = bool(checkpoint.get("truncated") or (
            isinstance(previous, dict) and previous.get("truncated")
        ))
        if len(reports) > 128:
            reports = reports[:64] + reports[-64:]
            truncated = True
        available = max(0, agent.CONTINUATION_CHECKPOINT_CHARS
                        - len(json.dumps(checkpoint, ensure_ascii=False)) - 8192)
        quota = available // max(1, len(reports))
        for row in reports:
            instruction_limit = min(2000, quota // 3)
            report_limit = max(0, quota - instruction_limit)
            for key, limit in (("instruction", instruction_limit), ("report", report_limit)):
                value = row[key]
                if len(value) > limit:
                    marker = "… [saved evidence excerpt truncated; verify its artifacts]"
                    row[key] = value[:max(0, limit - len(marker))] + marker[:limit]
                    truncated = True
        checkpoint.update({"reports": reports, "truncated": truncated})
        return checkpoint

    @staticmethod
    def _job_output_record(result: agent.AgentResult, ws: str) -> dict:
        """Keep cumulative artifacts/usage without replaying old partial replies."""
        retained = agent.AgentResult()
        agent.carry_forward_turn_output(result, retained, ws)
        return {
            "attachments": list(dict.fromkeys(event.content for event in retained.events
                                               if event.kind == "attachment")),
            "usage": result.usage,
            "detached_tasks": [{"backend_name": task.backend_name, "task_id": task.task_id}
                               for task in result.detached_tasks],
            "detached_task_requests": [request.prompt for request in result.detached_task_requests],
        }

    @staticmethod
    def _job_output(payload: object) -> agent.AgentResult | None:
        if not isinstance(payload, dict):
            return None
        result = agent.AgentResult()
        attachments = payload.get("attachments")
        if isinstance(attachments, list):
            result.events = [agent.ChatEvent(kind="attachment", content=path)
                             for path in attachments if isinstance(path, str)]
        usage = payload.get("usage")
        result.usage = usage if isinstance(usage, dict) else None
        tasks = payload.get("detached_tasks")
        if isinstance(tasks, list):
            result.detached_tasks = [DetachedTaskRef(task["backend_name"], task["task_id"])
                                     for task in tasks if isinstance(task, dict)
                                     and isinstance(task.get("backend_name"), str)
                                     and isinstance(task.get("task_id"), str)]
        requests = payload.get("detached_task_requests")
        if isinstance(requests, list):
            result.detached_task_requests = [DetachedTaskRequest(prompt)
                                            for prompt in requests if isinstance(prompt, str)]
        return result

    @staticmethod
    def _job_result_is_incomplete(result: agent.AgentResult) -> bool:
        text = result.text or "\n\n".join(event.content for event in result.events
                                           if event.kind == "text")
        _, awaiting = agent.extract_await(text)
        return bool(result.continuation_reason or result.continue_instruction
                    or result.error or awaiting or result.completion_verified is False
                    or (not text.strip()
                        and not any(event.kind == "attachment" for event in result.events)
                        and not result.detached_tasks)
                    or text.lstrip().upper().startswith("PARTIAL"))

    async def _run_turn_with_budget(
        self, uid: str, chat_id: str, text: str,
        *, session_id: str | None = None,
        queue_entry_id: str | None = None,
        workspace_path: str | None = None,
        previous_result: agent.AgentResult | None = None,
        on_session: Callable[[str], Awaitable[None]] | None = None,
        original_request: str | None = None,
    ) -> agent.AgentResult | None:
        """Send a "Thinking..." status, run the agent, then post the reply.

        ``session_id`` pins the run to a specific session — used by
        ``_run_ephemeral_turn`` to route a scheduled command into a
        throwaway session. When None, ``agent.run`` routes the prompt
        to the best-matching session via ``router.select_or_create_session``.

        ``_running_tasks`` is already populated by the caller
        (``_dispatch_ai`` or ``_drain_message_queue``) right after
        lock.acquire(), so /stop can cancel this turn during any of
        its await points.
        """
        ws = workspace_path or await self._current_workspace_for_turn(uid, chat_id)
        if ws is None:
            return
        backend_name, model, summary_model, perm, summary_backend = (
            workspace.get_run_config(ws)
        )
        # Start attribution before the first yield so a stuck wait can
        # name backend/model/elapsed even without a first progress event.
        self._begin_turn_progress(uid, backend=backend_name, model=model)

        inject_q = _InjectQueue(maxsize=self.max_queue_size)
        self._inject_queues[uid] = inject_q

        recovering_status = (
            _RecoverableTurnStatus(self, uid, chat_id)
            if self.supports_transient_status and self.status_recovery_interval_sec is not None
            else None
        )
        thinking_result = None
        if self.supports_transient_status and recovering_status is None:
            status_started = time.monotonic()
            thinking_result = await self._run_status_operation(
                self.send_text(chat_id, "Thinking..."),
                action="posting the Thinking status",
            )
            performance_logger.info(
                "stage=thinking_status platform=%s outcome=%s elapsed_ms=%.3f",
                self.platform_id,
                "posted" if isinstance(thinking_result, MessageHandle) else "no_handle",
                (time.monotonic() - status_started) * 1_000,
            )
        thinking_handle = (
            thinking_result
            if isinstance(thinking_result, MessageHandle) else None
        )

        status_lines: list[str] = []
        latest_text = ""
        last_edit = (
            asyncio.get_running_loop().time()
            if self.status_update_interval_sec > 0 else 0.0
        )
        status_updates = 0
        pending_status: str | None = None
        status_edit_task: asyncio.Task[object] | None = None

        async def flush_status_edits() -> None:
            """Rate-limit one worker, keeping the latest pending preview."""
            nonlocal pending_status, last_edit, status_updates
            while pending_status is not None:
                if (
                    self.status_update_limit is not None
                    and status_updates >= self.status_update_limit
                ):
                    pending_status = None
                    return
                loop = asyncio.get_running_loop()
                next_edit = last_edit + max(
                    _STATUS_EDIT_INTERVAL_SEC, self.status_update_interval_sec,
                )
                delay = next_edit - loop.time()
                while delay > 0:
                    # Windows timers can wake one clock tick early. Recheck
                    # the deadline without spinning on sub-tick sleeps.
                    await asyncio.sleep(max(
                        delay, 2 * time.get_clock_info("monotonic").resolution,
                    ))
                    delay = next_edit - loop.time()
                # Select after sleeping so superseded previews are discarded.
                display = pending_status
                pending_status = None
                # Count attempts before I/O: ambiguous failures/timeouts must
                # not cause us to exceed a platform's per-message edit cap.
                status_updates += 1
                operation = (
                    self.edit_text(thinking_handle, display)
                    if thinking_handle is not None
                    else self.send_status(chat_id, display)
                )
                await self._run_status_operation(
                    operation, action="updating the progress status",
                )
                # Start the next interval after the op settles so edits
                # cannot bunch behind a delayed dispatch.
                last_edit = loop.time()

        async def on_event(ev: agent.ChatEvent) -> None:
            nonlocal latest_text
            nonlocal pending_status, status_edit_task
            self._record_turn_event_progress(uid, ev)
            if recovering_status is not None:
                recovering_status.on_event(ev)
                return
            if not self.supports_transient_status:
                return
            if (
                (thinking_handle is None and not self.status_fallback_messages)
                or (
                    self.status_update_limit is not None
                    and status_updates >= self.status_update_limit
                )
            ):
                return
            if ev.kind == "text":
                if ev.content.strip():
                    latest_text = ev.content
            elif ev.kind in ("tool", "file"):
                content = (
                    ev.content.split(chr(10))[0]
                    if ev.kind == "tool" else ev.content
                )
                clipped = (
                    content[:80 - len("… [clipped]")] + "… [clipped]"
                    if len(content) > 80 else content
                )
                status_lines.append(f"» {clipped}")
            else:
                return
            if thinking_handle is None:
                # No editable status: progress as fresh messages; skip answer text.
                if ev.kind == "text":
                    return
                pending_status = status_lines[-1]
            else:
                # One Thinking message, edited in place.
                pending_status = self._compose_thinking_display(
                    status_lines, latest_text,
                )
            if status_edit_task is None or status_edit_task.done():
                status_edit_task = create_background_task(
                    flush_status_edits(),
                    name=f"{self.platform_id}:status-edit:{uid}",
                    log=logger,
                )

        try:
            if recovering_status is not None:
                await recovering_status.start()
            result = await agent.run(
                text, ws, user_id=uid,
                model=model, summary_model=summary_model,
                approval=perm,
                on_event=on_event, inject_queue=inject_q,
                backend_name=backend_name,
                summary_backend_name=summary_backend,
                session_id=session_id,
                on_session=on_session,
                execution_id=queue_entry_id,
                original_request=original_request,
                on_process=lambda proc, uid=uid: self._note_turn_process(
                    uid, getattr(proc, "pid", None),
                ),
            )
            if previous_result is not None:
                agent.carry_forward_turn_output(previous_result, result, ws)
                seen_attachments: set[str] = set()
                events = []
                for event in result.events:
                    if event.kind == "attachment":
                        if event.content in seen_attachments:
                            continue
                        seen_attachments.add(event.content)
                    events.append(event)
                result.events = events
            if self._job_result_is_incomplete(result):
                return result  # checkpoint first; never enter the final reply ledger

            # Start durable provider jobs here (interactive turns only).
            if session_id is None:
                try:
                    await self._launch_result_detached_tasks(
                        uid,
                        chat_id,
                        ws,
                        result,
                        model=model,
                        summary_model=summary_model,
                        approval=perm,
                        backend_name=backend_name,
                        summary_backend_name=summary_backend,
                    )
                except Exception:
                    # The foreground reply is still owed on handoff failure.
                    logger.exception("Could not launch detached task request")
                    agent.append_text_result(
                        result, "Could not start the requested background task.",
                    )
            # Register Claude bg ids before the parent reply.
            try:
                await self._register_result_detached_tasks(
                    uid,
                    chat_id,
                    ws,
                    result,
                    session_id=(
                        None if session_id is not None else result.session_id
                    ),
                )
            except Exception:
                # Ledger failures must never hide the owed foreground answer.
                logger.exception("Could not register detached task callback")
        finally:
            # Idempotent window close before I/O can yield.
            inject_q.close()
            # Attribution record ends with the agent phase; delivery/post-turn
            # work no longer counts as a command-output wait.
            self._clear_turn_progress(uid)
            if recovering_status is not None:
                await recovering_status.close()
            if status_edit_task is not None and not status_edit_task.done():
                status_edit_task.cancel()
            if thinking_handle is not None:
                # Stale-status cleanup must not hold the answer hostage.
                create_background_task(
                    self._run_status_operation(
                        self.delete_message(thinking_handle),
                        action="deleting the Thinking status",
                    ),
                    name=f"{self.platform_id}:status-delete:{uid}",
                    log=logger,
                )

        # [[await]] pause is interactive-only.
        if queue_entry_id is None:
            await self._send_result(
                chat_id, ws, result,
                uid=uid if session_id is None else None,
            )
            return

        # Preserve the reply: a failed send must not rerun the agent.
        record = await self._stage_reply_delivery(
            uid, queue_entry_id, chat_id, ws, result,
            allow_await=session_id is None,
        )
        await self._deliver_staged_reply(
            record,
            ws=ws,
            result=result,
            awaiting_uid=uid if session_id is None else None,
        )

    async def _run_ephemeral_turn(
        self, uid: str, chat_id: str, text: str,
        *, queue_entry_id: str | None = None,
    ) -> bool:
        """Keep a scheduled session until its job completes or is cancelled."""
        job = await self._load_job_checkpoint(uid, queue_entry_id) if queue_entry_id else {}
        ws = job.get("workspace_path")
        if not isinstance(ws, str) or not ws:
            ws = await self._current_workspace_for_turn(uid, chat_id)
        if ws is None:
            if queue_entry_id:
                job.update({"state": "paused", "request": text,
                            "instruction": "Restore the scheduled job's workspace and finish its original scope."})
                await self._save_job_checkpoint(uid, queue_entry_id, job)
            self._paused_jobs.add(uid)
            return False

        # Distinctive name: avoids the auto-title race.
        stripped = text.strip()
        label = stripped.splitlines()[0] if stripped else "scheduled"
        if len(label) > 40:
            label = label[:40 - len("… [clipped]")] + "… [clipped]"
        sid = job.get("session_id")
        if not isinstance(sid, str) or session.load_session(ws, sid) is None:
            sid = None
        completed = False

        def created(sess_data: dict) -> None:
            nonlocal sid
            sid = sess_data["id"]

        try:
            if sid is None:
                await _run_durable_file_operation(
                    workspace.get_lock(ws),
                    lambda: session.create_session(ws, name=f"⏰ {label}"),
                    on_complete=created,
                )
            if queue_entry_id:
                job.update({"workspace_path": ws, "session_id": sid})
                if not await self._save_job_checkpoint(uid, queue_entry_id, job):
                    return False
            await self._run_status_operation(
                self.send_text(chat_id, f"⏰ Starting scheduled task: {text}"),
                action="announcing a starting scheduled task",
            )
            outcome = await self._run_turn(
                uid, chat_id, text, session_id=sid,
                queue_entry_id=queue_entry_id,
            )
            completed = outcome is not False
            return completed
        finally:
            # A restarted daemon must resume the same job/session, not rerun
            # its completed workers in a fresh scheduled conversation.
            try:
                pending = await self._load_job_checkpoint(uid, queue_entry_id) if queue_entry_id else {}
                retain_session = (not completed and pending.get("session_id") == sid
                                  and uid not in self._cancel_acknowledged
                                  and not self._cancel_work_depths.get(uid, 0))
                if sid is not None and not retain_session:
                    await _run_durable_file_operation(
                        workspace.get_lock(ws), lambda: session.delete_session(ws, sid),
                        required_cleanup=True,
                    )
            except Exception:
                logger.warning(
                    "Failed to delete ephemeral session %s", sid,
                    exc_info=True,
                )

    async def _current_workspace_for_turn(
        self, uid: str, chat_id: str,
    ) -> str | None:
        ws = workspace.get_current(uid, self.platform_id)
        if not ws or not os.path.isdir(ws):
            await self.send_text(
                chat_id,
                "Workspace not available (deleted?). Use /new or /open.",
            )
            return None
        return ws

    async def _await_attachment_tail(self, chat_id: str) -> None:
        """Give earlier pictures a short head start without stranding text.

        The upload chain retains its own ordering and cancellation owner.
        Timing out or cancelling this wait must never cancel that chain.
        """
        tail = self._attachment_tails.get(chat_id)
        if tail is None or tail.done():
            return
        loop = asyncio.get_running_loop()
        started = loop.time()
        settled, _ = await asyncio.wait({tail}, timeout=_ATTACHMENT_TAIL_WAIT_SEC)
        # asyncio.wait does not propagate an older upload's exception or
        # cancel it when this foreground wait is cancelled or times out.
        performance_logger.info(
            "stage=attachment_tail_wait outcome=%s elapsed_seconds=%.3f",
            "settled" if settled else "timeout", loop.time() - started,
        )

    def _track_attachment_upload(
        self, owner: str, chat_id: str, paths: list[str],
    ) -> asyncio.Task[None]:
        """Own a bounded upload pool, preserving per-chat batch order."""
        tail = self._attachment_tails.get(chat_id)
        if tail is not None and tail.done():
            tail = None

        async def _send_one(abs_path: str) -> None:
            try:
                await self.send_file(chat_id, abs_path)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "Failed to send attachment %s: %s", abs_path, e,
                )
                await self._send_text_best_effort(
                    chat_id,
                    f"Failed to attach {os.path.basename(abs_path)}: {e}",
                )

        async def _run() -> None:
            if tail is not None and not tail.done():
                try:
                    await asyncio.shield(tail)
                except asyncio.CancelledError:
                    # Cancelling this owner must stop its batch. Cancellation
                    # of an earlier owner's tail merely releases the chain.
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
                except Exception:
                    pass
            remaining = iter(paths)

            async def worker() -> None:
                for path in remaining:
                    await _send_one(path)

            workers = [asyncio.create_task(
                worker(), name=f"{self.platform_id}:upload-file:{index}",
            ) for index in range(min(_ATTACHMENT_UPLOAD_CONCURRENCY, len(paths)))]
            try:
                await asyncio.gather(*workers)
            finally:
                for worker_task in workers:
                    if not worker_task.done():
                        worker_task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

        task: asyncio.Task[None] = create_background_task(
            _run(),
            name=f"{self.platform_id}:upload:{owner}",
            log=logger,
        )
        owned = self._attachment_tasks.setdefault(owner, set())
        owned.add(task)
        self._attachment_tails[chat_id] = task

        def _done(done: asyncio.Task[None]) -> None:
            owned.discard(done)
            if not owned:
                self._attachment_tasks.pop(owner, None)
            if self._attachment_tails.get(chat_id) is done:
                self._attachment_tails.pop(chat_id, None)

        task.add_done_callback(_done)
        return task

    async def _cancel_attachment_uploads(self, owner: str) -> int:
        """Cancel background uploads owned by *owner*; return count."""
        owned = self._attachment_tasks.get(owner)
        if not owned:
            return 0
        tasks = [task for task in owned if not task.done()]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await asyncio.gather(task, return_exceptions=True)
            except asyncio.CancelledError:
                pass
        return len(tasks)

    async def _send_detached_result(
        self, chat_id: str, ws: str, result: agent.AgentResult, uid: str,
    ) -> None:
        """Send a detached callback without arming the [[await]] pause.

        Uploads stay owned by *uid* so /cancel still finds them; only
        the queue-pause side effect is suppressed, since a provider's
        textual ``[[await]]`` marker must not park the normal queue.
        A prior interactive question keeps its existing pause.
        """
        await self._send_result(chat_id, ws, result, uid=uid, allow_await=False)

    async def _send_result(
        self, chat_id: str, ws: str, result: agent.AgentResult,
        *, uid: str | None = None,
        allow_await: bool = True,
    ) -> None:
        """Send the agent's reply (text + attachments) and honor [[await]].

        ``uid`` has two jobs: it opts the user's queue into the await
        pause, and it owns the background picture uploads so /cancel can
        find them. Detached callbacks pass the user id with awaiting
        disabled with ``allow_await=False`` (their textual ``[[await]]``
        marker must not pause the normal queue); ephemeral schedule turns
        omit ``uid`` because their session is deleted right after, so
        ownership falls back to the chat.

        Latency shape: texts (and the usage footer) send synchronously
        first, then pictures upload in a per-chat background chain. The
        turn returns as soon as texts are out, so the per-user lock
        releases and the next queued turn's agent work overlaps the
        uploads. The next reply's texts give the earlier tail at most five
        seconds to finish; attachment batches retain their own order.
        Attachments stay best-effort: a crash mid-upload loses only the
        pictures, never the staged text payload.
        """
        await self._await_attachment_tail(chat_id)
        awaiting = False
        sent_sources: set[str] = set()
        pending_uploads: list[str] = []

        def collect_attachment(path: str) -> None:
            source_path = agent.attachment_source_path(path, ws)
            if source_path is None:
                logger.warning(
                    "Refusing to send missing or unsafe attachment: %s", path,
                )
                return
            if source_path in sent_sources:
                return
            abs_path = agent.prepare_attachment_path(path, ws)
            if abs_path is None:
                logger.warning(
                    "Failed to prepare attachment for sending: %s", path,
                )
                return
            sent_sources.add(source_path)
            pending_uploads.append(abs_path)

        for ev in result.events:
            if ev.kind == "attachment":
                # Explicit generated images can require a large atomic copy.
                # Keep that I/O off-loop and owned until it physically settles,
                # so cancellation cannot acknowledge while a copy still writes.
                await run_owned_worker(collect_attachment, ev.content)
                continue
            if ev.kind != "text":
                continue
            text, attach_paths = agent.extract_attachment_sources(
                ev.content, ws,
            )
            text, ev_awaiting = agent.extract_await(text)
            if ev_awaiting:
                awaiting = True
            if text:
                await self.send_text(chat_id, text, rich=True)
            for path in attach_paths:
                await run_owned_worker(collect_attachment, path)

        # Compact per-turn token/cost footer.
        if result.usage and config.get_show_usage():
            footer = agent.format_usage(result.usage)
            if footer:
                await self._send_text_best_effort(
                    chat_id, footer, rich=True,
                )

        if pending_uploads:
            owner = uid if uid is not None else f"chat:{chat_id}"
            self._track_attachment_upload(owner, chat_id, pending_uploads)

        if awaiting and uid is not None and allow_await:
            self._arm_awaiting_answer(uid)

    def _arm_awaiting_answer(self, uid: str) -> None:
        """Honor a delivered ``[[await]]`` reply without stranding backlog."""
        # [[await]] parks new messages; queued turns still drain.
        msg_queue = self._message_queues.get(uid)
        if msg_queue is not None and self._has_pending_normal_entries(msg_queue):
            logger.info(
                "User %s reply ended with [[await]] but messages sent "
                "earlier are queued; skipping answer pause",
                uid,
            )
            return
        self._awaiting_answer.add(uid)
        logger.info(
            "User %s awaiting answer; queue paused until next message", uid,
        )

    async def _drain_message_queue(self, uid: str) -> None:
        msg_queue = self._message_queues.get(uid)
        if not msg_queue:
            return
        # Retry the unaccepted answer before a new turn.
        if uid in self._pending_reply_delivery_users:
            return
        # Defensive: create a missing lock on demand.
        lock = self._ensure_task_lock(uid)

        while not msg_queue.empty():
            if uid in self._pending_reply_delivery_users:
                break
            if self._update_restart_pending:
                break
            if uid in self._paused_jobs:
                break
            if lock.locked():
                break
            await lock.acquire()
            try:
                entry = self._pop_next_queue_entry(
                    msg_queue,
                    ephemeral_only=uid in self._awaiting_answer,
                )
            except asyncio.QueueEmpty:
                lock.release()
                break
            if entry is None:
                # [[await]] pause: chat waits; ephemeral entries still drain.
                lock.release()
                break
            # Drop malformed shapes; never wedge the drain loop.
            if (
                not isinstance(entry, tuple)
                or len(entry) != 4
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], str)
                or not isinstance(entry[2], str)
                or not isinstance(entry[3], bool)
            ):
                logger.warning(
                    "Dropping malformed queue entry for user=%s: %r",
                    uid, entry,
                )
                lock.release()
                continue
            text, msg_chat_id, entry_id, ephemeral = entry

            # Drop de-authorized entries; completion is best-effort.
            if not self.authorized(uid, msg_chat_id):
                logger.warning(
                    "Dropping queued entry for unauthorized user=%s chat=%s",
                    uid, msg_chat_id,
                )
                try:
                    await self._persist_complete(uid, entry_id)
                except Exception:
                    logger.exception(
                        "Failed to persist completion of unauthorized entry",
                    )
                finally:
                    lock.release()
                continue

            # Register before first yield so /stop cancels early.
            self._running_tasks[uid] = asyncio.current_task()
            try:
                if ephemeral:
                    completed = await self._run_ephemeral_turn(
                        uid, msg_chat_id, text, queue_entry_id=entry_id,
                    )
                else:
                    completed = await self._run_turn(
                        uid, msg_chat_id, text, queue_entry_id=entry_id,
                    )
            except asyncio.CancelledError:
                # Keep on disk for restart resume.
                if uid not in self._cancel_acknowledged:
                    await self._send_text_best_effort(
                        msg_chat_id, "Cancelled.",
                    )
                break
            except Exception as e:
                logger.exception("Queued AI chat failed")
                await self._pause_job_after_failure(uid, entry_id, msg_chat_id, e)
                if self._update_restart_pending or uid in self._paused_jobs:
                    break
            else:
                if completed is False:
                    await self._retain_job_entry(uid, entry_id)
                    if self._update_restart_pending or uid in self._paused_jobs:
                        break
                    continue  # guidance arrived while the pause notice was sending
                await self._persist_complete(uid, entry_id)
            finally:
                self._cleanup_turn(uid, lock)

    @staticmethod
    def _pick_option(
        text: str, options: list[str], *, first_number: int = 1,
    ) -> str | None:
        if not isinstance(text, str) or not isinstance(options, list):
            return None
        text = text.strip()
        if text.isdecimal():
            number = parse_decimal_int(text)
            if number is None:
                return None
            idx = number - first_number
            if (
                0 <= idx < len(options)
                and isinstance(options[idx], str)
            ):
                return options[idx]
            return None
        if text in options:
            return text
        return None

    # Command registry (populated at class body end).
    _COMMANDS: ClassVar[dict[str, Handler]] = {}


def _flexible_tier_commands() -> dict[str, Handler]:
    """One /agent_flexible_<tier> and /model_flexible_<tier> per tier.

    Generated rather than written out so a new difficulty tier is a
    one-line change in :mod:`Cozter.flexible` instead of six here.
    """
    commands: dict[str, Handler] = {}
    for tier in workspace.FLEXIBLE_TIERS:
        commands[f"agent_flexible_{tier}"] = _flexible_command(
            BotPlatform._show_flexible_agent, tier,
        )
        commands[f"model_flexible_{tier}"] = _flexible_command(
            BotPlatform._show_flexible_model, tier,
        )
    return commands


def _flexible_command(show: Callable, tier: str) -> Handler:
    async def command(self: BotPlatform, ctx: BotContext) -> None:
        await show(self, ctx, tier)
    return command


def _build_command_registry() -> dict[str, Handler]:
    return {
        "start":        BotPlatform.cmd_start,
        "version":      BotPlatform.cmd_version,
        "doctor":       BotPlatform.cmd_doctor,
        "cancel":       BotPlatform.cmd_cancel,
        "new":          BotPlatform.cmd_new,
        "open":         BotPlatform.cmd_open,
        "model":        BotPlatform.cmd_model,
        "summarymodel": BotPlatform.cmd_summary_model,
        "agent":        BotPlatform.cmd_agent,
        "summaryagent": BotPlatform.cmd_summary_agent,
        **_flexible_tier_commands(),
        "permission":   BotPlatform.cmd_permission,
        "style":        BotPlatform.cmd_style,
        "effort":       BotPlatform.cmd_effort,
        "refresh":      BotPlatform.cmd_refresh,
        "compact":      BotPlatform.cmd_compact,
        "context":      BotPlatform.cmd_context,
        "newsession":   BotPlatform.cmd_new_session,
        "sessions":     BotPlatform.cmd_sessions,
        "colony":       BotPlatform.cmd_colony,
        "stop":         BotPlatform.cmd_stop,
        "bg":           BotPlatform.cmd_background,
        "background":   BotPlatform.cmd_background,
        "inject":       BotPlatform.cmd_inject,
        "reserve":      BotPlatform.cmd_reserve,
        "schedules":    BotPlatform.cmd_schedules,
    }


BotPlatform._COMMANDS = _build_command_registry()

COMMAND_NAMES: tuple[str, ...] = tuple(BotPlatform._COMMANDS.keys())
