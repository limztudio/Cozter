"""Durable job continuation and conversation-owned native CLI sessions.

Only actual execution workers bind a backend record. Internal routing, title,
summary and planner calls must remain stateless. Native provider identifiers
never cross a workspace, conversation, execution lane, model, permission or reasoning
boundary. Jobs retain exact native-generation pins independently of the
conversation's latest reusable generation.
"""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import math
import os
import re
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any
import uuid
import weakref

from .utils import fsync_directory, json_digest, validate_json_nesting

logger = logging.getLogger(__name__)
_MAX_STATE_BYTES = 64 * 1024 * 1024
_VERSION = 1
_NAMESPACES = frozenset({"executions", "native_sessions"})


@dataclass
class _ConversationLifetime:
    lock: Any = field(default_factory=threading.RLock)
    heads_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    lanes: dict[str, asyncio.Lock] = field(default_factory=dict)
    discarded: bool = False


_LIFETIMES: weakref.WeakValueDictionary[tuple[str, str], _ConversationLifetime] = (
    weakref.WeakValueDictionary()
)
_LIFETIMES_LOCK = threading.RLock()
_BACKEND_LOCKS: weakref.WeakValueDictionary[tuple[str, str, str, str], asyncio.Lock] = (
    weakref.WeakValueDictionary()
)
_BACKEND_LOCKS_LOCK = threading.RLock()


def _conversation_lifetime(workspace_path: str, session_id: str) -> _ConversationLifetime:
    key = (os.path.normcase(os.path.realpath(workspace_path)), json_digest(session_id))
    with _LIFETIMES_LOCK:
        lifetime = _LIFETIMES.get(key)
        if lifetime is None:
            lifetime = _ConversationLifetime()
            _LIFETIMES[key] = lifetime
        return lifetime


def _session_id(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("native conversation requires a nonempty session id")
    return value


def _worker_lane(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("native worker lane requires a nonempty name")
    return value


def _backend_lock(execution: "Execution", identity: object) -> asyncio.Lock:
    """Serialize one job record even across independent Execution objects."""
    key = (
        os.path.normcase(os.path.realpath(execution.workspace_path)),
        execution.namespace, execution.directory_id, json_digest(identity),
    )
    with _BACKEND_LOCKS_LOCK:
        lock = _BACKEND_LOCKS.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _BACKEND_LOCKS[key] = lock
        return lock


def _generation(value: object) -> str | None:
    if not isinstance(value, str) or len(value) != 32:
        return None
    try:
        return value if uuid.UUID(value).hex == value else None
    except ValueError:
        return None


def native_session_id(value: object) -> str | None:
    """Only a full UUID can identify a saved native conversation."""
    if not isinstance(value, str) or len(value) != 36:
        return None
    try:
        canonical = str(uuid.UUID(value))
    except ValueError:
        return None
    return canonical if canonical == value.lower() else None


def _finite_json_float(value: str) -> float:
    """Reject both nonstandard constants and overflowing numeric literals."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("backend continuation contains a nonfinite number")
    return number


async def _settle(task: asyncio.Task) -> Any:
    """Finish an owned atomic write even if cancellation is repeated."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


def _copy_json_values(values: dict[str, Any]) -> dict[str, Any]:
    """Normalize and own caller values without retaining mutable references."""
    return json.loads(json.dumps(values, ensure_ascii=False, allow_nan=False))


def _write_snapshot(path: str, snapshot: dict[str, Any]) -> None:
    """Serialize once off the event loop, then atomically install bounded bytes."""
    # Match the existing indented/escaped on-disk format and Windows newline
    # translation. Check the actual bytes before opening a temporary file.
    encoded = json.dumps(snapshot, indent=2, allow_nan=False)
    payload = encoded.replace("\n", os.linesep).encode("utf-8")
    if len(payload) > _MAX_STATE_BYTES:
        raise ValueError("backend continuation exceeds the 64 MiB limit")
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        fsync_directory(directory)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


@dataclass
class ExecutionRecord:
    """One isolated JSON record, with cancellation-safe serialized writes."""

    execution: "Execution"
    key: str
    _values: dict[str, Any] = field(default_factory=dict)
    _revision: int = 0
    _written: int = 0
    _operation: int = 0
    _applied_operations: dict[str, int] = field(default_factory=dict)
    _writer: asyncio.Task | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def directory(self) -> Path:
        from . import workspace
        self.execution.ensure_live()
        return Path(workspace.ensure_workspace_state_dir(
            self.execution.workspace_path, self.execution.namespace, self.execution.directory_id,
            self.key,
        ))

    @property
    def path(self) -> Path:
        from . import workspace
        self.execution.ensure_live()
        # Revalidate containment on every read/write, including existing links.
        return Path(workspace.workspace_state_path(
            self.execution.workspace_path, self.execution.namespace, self.execution.directory_id,
            self.key, "state.json",
        ))

    def _read_values(self) -> dict:
        path = self.path
        if not path.exists():
            return {}
        if path.stat().st_size > _MAX_STATE_BYTES:
            raise RuntimeError("Saved backend continuation is oversized; preserve and inspect it before resuming")
        try:
            data = json.loads(
                path.read_bytes(), parse_constant=_finite_json_float,
                parse_float=_finite_json_float,
            )
            validate_json_nesting(data)
        except (OSError, ValueError, RecursionError) as exc:
            raise RuntimeError("Saved backend continuation is unreadable; preserve and inspect it before resuming") from exc
        if (not isinstance(data, dict) or data.get("version") != _VERSION
                or data.get("identity") != self.key
                or not isinstance(data.get("values"), dict)):
            raise RuntimeError("Saved backend continuation has an invalid identity or format; inspect it before resuming")
        return data["values"]

    def get(self, key: str, default: Any = None) -> Any:
        return deepcopy(self._values.get(key, default))

    def set(self, **values: Any) -> None:
        # Reject non-JSON values synchronously, before a caller starts a tool.
        self._set_copied(_copy_json_values(values))

    def _reserve_operation(self) -> int:
        self._operation += 1
        return self._operation

    def _set_copied(self, copied: dict[str, Any], operation: int | None = None) -> None:
        if operation is None:
            operation = self._reserve_operation()
        # Snaps race: an old save never beats a new one.
        # Failed validation never marks a key applied, retaining earlier valid
        # updates if a later invalid update is rejected.
        accepted = {
            key: value for key, value in copied.items()
            if self._applied_operations.get(key, 0) < operation
        }
        if not accepted:
            return
        self._values.update(accepted)
        for key in accepted:
            self._applied_operations[key] = operation
        self._revision += 1
        if self._writer is None or self._writer.done():
            self._writer = asyncio.create_task(
                self._write_pending(), name=f"execution-state:{self.key[:12]}",
            )

    async def save(self, **values: Any) -> None:
        if not values:
            await self.flush()
            return
        operation = self._reserve_operation()

        async def snapshot_and_write() -> None:
            # HTTP histories can be several MiB. Keep their JSON validation
            # and copy out of event parsing/status dispatch. The owned task
            # also ensures cancellation cannot drop a completed tool result
            # while its snapshot is being prepared.
            copied = await asyncio.to_thread(_copy_json_values, values)
            self._set_copied(copied, operation)
            await self.flush()

        task = asyncio.create_task(snapshot_and_write(), name="save-execution-state")
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await _settle(task)
            raise

    async def flush(self) -> None:
        writer = self._writer
        if writer is None:
            return
        try:
            await asyncio.shield(writer)
        except asyncio.CancelledError:
            await _settle(writer)
            raise

    async def reload(self) -> None:
        """Refresh a flushed record while its caller owns the record lease."""
        await self.flush()
        self._values = await asyncio.to_thread(self._read_values)

    async def _write_pending(self) -> None:
        async with self._lock:
            while self._written < self._revision:
                revision = self._revision
                snapshot = {
                    "version": _VERSION,
                    "identity": self.key,
                    # Values are already privately copied on entry and get()
                    # returns copies. Updates replace top-level values, so a
                    # shallow snapshot remains immutable during thread I/O.
                    "values": dict(self._values),
                }
                def write() -> None:
                    lifetime = self.execution._lifetime
                    if lifetime is None:
                        self.directory
                        _write_snapshot(str(self.path), snapshot)
                    else:
                        # Deletion and final replacement share a lock so a
                        # queued writer cannot resurrect a deleted session.
                        with lifetime.lock:
                            if lifetime.discarded:
                                return
                            self.directory
                            _write_snapshot(str(self.path), snapshot)

                await asyncio.to_thread(write)
                self._written = revision


@dataclass
class Execution:
    workspace_path: str
    execution_id: str
    original_request: str
    records: dict[str, ExecutionRecord] = field(default_factory=dict)
    namespace: str = "executions"
    conversation: "Execution | None" = field(default=None, init=False)
    native_main_record: ExecutionRecord | None = field(default=None, init=False)
    _lifetime: _ConversationLifetime | None = field(default=None, init=False)
    _native_requests: dict[tuple[str, str], tuple[ExecutionRecord, set[str]]] = field(
        default_factory=dict, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        if self.namespace not in _NAMESPACES:
            raise ValueError("invalid backend continuation namespace")
        if self.namespace == "native_sessions":
            self._lifetime = _conversation_lifetime(
                self.workspace_path, _session_id(self.execution_id),
            )

    def ensure_live(self) -> None:
        if self.namespace not in _NAMESPACES:
            raise ValueError("invalid backend continuation namespace")
        if self._lifetime is not None and self._lifetime.discarded:
            raise RuntimeError("native conversation was deleted")

    def bind_conversation(self, session_id: str) -> None:
        """Bind the resolved Cozter conversation without changing job evidence."""
        _session_id(session_id)
        if self.namespace != "executions":
            raise ValueError("only a job execution can bind a native conversation")
        if self.conversation is not None:
            if self.conversation.execution_id != session_id:
                raise ValueError("an execution cannot change its native conversation")
            self.conversation.ensure_live()
            return
        self.conversation = Execution(
            self.workspace_path, session_id, "", namespace="native_sessions",
        )
        self.conversation.ensure_live()

    def remember_native_request(self, record: ExecutionRecord, request: str) -> bool:
        """Remember an already acknowledged request only for this job's UUID.

        Callers invoke this after the native prompt's durable acknowledgment.
        Request bodies stay in memory; they are never execution-state values.
        """
        if not isinstance(request, str):
            raise ValueError("native request body must be a string")
        if (not isinstance(record, ExecutionRecord)
                or record.execution is not self and record.execution is not self.conversation):
            raise ValueError("native request record belongs to another execution")
        native_id = self._request_record_session(record)
        if native_id is None or record.get("native_tools_started") is not True:
            return False
        directory = os.path.normcase(str(record.path.parent))
        # A fallback can replace the provider session in the same generation.
        # Discard the previous UUID's requests instead of relabeling them.
        for key in list(self._native_requests):
            if key[0] == directory and key[1] != native_id:
                del self._native_requests[key]
        _, requests = self._native_requests.setdefault((directory, native_id), (record, set()))
        requests.add(request)
        return True

    @staticmethod
    def _request_record_session(record: ExecutionRecord) -> str | None:
        lifetime = record.execution._lifetime
        if lifetime is not None and lifetime.discarded:
            return None
        native_id = record.get("native_session_id")
        if native_session_id(native_id) is None:
            return None
        assert isinstance(native_id, str)
        native_id = native_id.lower()
        ledger = record.get("native_prompt_ledger")
        if not isinstance(ledger, dict) or ledger.get("session_id") != native_id:
            return None
        return native_id

    def native_request_deliveries(self) -> list[tuple[ExecutionRecord, str, frozenset[str]]]:
        """Snapshot exact accepted bodies, excluding replaced or deleted UUIDs."""
        return [
            (record, native_id, frozenset(requests))
            for (_, native_id), (record, requests) in self._native_requests.items()
            if self._request_record_session(record) == native_id
        ]

    @property
    def directory_id(self) -> str:
        return json_digest(self.execution_id)

    async def record(self, identity: object, *, reload: bool = False) -> ExecutionRecord:
        # The queue entry owns identity. Accepted guidance changes the request
        # while retaining the same native conversation and tool evidence.
        # Flexible plans separately fingerprint the latest full requirements.
        self.ensure_live()
        key = json_digest(identity)
        cached = self.records.get(key)
        if cached is not None:
            if reload:
                await cached.reload()
            return cached
        result = ExecutionRecord(self, key)

        result._values = await asyncio.to_thread(result._read_values)
        # The workspace run lock normally serializes this, but independent
        # workers may request the same record concurrently.
        return self.records.setdefault(key, result)

    async def flush(self) -> None:
        async def finish() -> None:
            results = await asyncio.gather(
                *(item.flush() for item in self.records.values()),
                *((self.conversation.flush(),) if self.conversation is not None else ()),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result

        task = asyncio.create_task(finish(), name="flush-execution-state")
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await _settle(task)
            raise

    async def native_record(
        self, identity: dict[str, Any], job_record: ExecutionRecord,
    ) -> tuple[ExecutionRecord, ExecutionRecord | None, str | None]:
        """Select a conversation generation while pinning this job before work."""
        conversation = self.conversation
        if conversation is None:
            return job_record, None, None
        lifetime = conversation._lifetime
        assert lifetime is not None
        async with lifetime.heads_lock:
            conversation.ensure_live()
            pinned = job_record.get("native_record_id")
            # Preserve unfinished pre-conversation jobs in place, including
            # their owned Copilot home. Never copy unrelated job evidence.
            if pinned is None and native_session_id(job_record.get("native_session_id")) is not None:
                return job_record, None, None
            if pinned is not None and _generation(pinned) is None:
                raise RuntimeError("saved job has an invalid native generation pin")
            head = await conversation.record({"native_head": identity}, reload=True)
            head_generation = head.get("native_record_id")
            if head_generation is not None and _generation(head_generation) is None:
                raise RuntimeError("saved conversation has an invalid native generation head")
            selected = pinned
            state = None
            if selected is None and head_generation is not None:
                candidate = await conversation.record({
                    "native_generation": head_generation, "backend": identity,
                }, reload=True)
                if (candidate.get("native_record_id") == head_generation
                        and native_session_id(candidate.get("native_session_id")) is not None):
                    selected = head_generation
                    state = candidate
            if selected is None:
                selected = uuid.uuid4().hex
                state = await conversation.record({
                    "native_generation": selected, "backend": identity,
                })
                await state.save(native_record_id=selected)
            elif state is None:
                state = await conversation.record({
                    "native_generation": selected, "backend": identity,
                }, reload=True)
                if state.get("native_record_id") != selected:
                    raise RuntimeError("saved native generation is missing or has an invalid identity")
            assert state is not None
            if pinned is None:
                await job_record.save(
                    native_record_id=selected, native_worker=identity["worker"],
                )
            # A resumed old job must not replace a newer conversation head.
            expected = head_generation if pinned is None else selected
            return state, head, expected

    async def publish_native_head(
        self, state: ExecutionRecord, head: ExecutionRecord | None,
        expected: str | None,
    ) -> None:
        if head is None or native_session_id(state.get("native_session_id")) is None:
            return
        conversation = self.conversation
        assert conversation is not None and conversation._lifetime is not None
        lifetime = conversation._lifetime
        async with lifetime.heads_lock:
            if lifetime.discarded:
                return
            # Different job Execution objects can have stale cached heads.
            # Read the persisted head again before promoting this generation.
            def current_generation() -> object:
                with lifetime.lock:
                    if lifetime.discarded:
                        return object()
                    return head._read_values().get("native_record_id")

            current = await asyncio.to_thread(current_generation)
            selected = state.get("native_record_id")
            if _generation(selected) is None:
                raise RuntimeError("saved native generation has an invalid identity")
            if current == expected or current == selected or current is None and expected == selected:
                await head.save(native_record_id=selected)


_EXECUTION: ContextVar[Execution | None] = ContextVar("cozter_execution", default=None)
_WORKER: ContextVar[str] = ContextVar("cozter_worker", default="main")
_RUN: ContextVar[ExecutionRecord | None] = ContextVar("cozter_backend_run", default=None)


def current_execution() -> Execution | None:
    return _EXECUTION.get()


def current_run() -> ExecutionRecord | None:
    return _RUN.get()


def adopt_native_session_id(value: object) -> None:
    """Save the CLI session id unless this run already owns another one."""
    native_id = native_session_id(value)
    record = current_run()
    if (record is not None and native_id is not None
            and record.get("native_session_id") in (None, native_id)):
        record.set(native_session_id=native_id)


def record_missing_native_session(message: object) -> None:
    """Allow recovery only for an exact missing ID before activity."""
    record = current_run()
    if (record is None or not record.get("native_resume_attempted")
            or record.get("native_tools_started") or not isinstance(message, str)):
        return
    native_id = native_session_id(record.get("native_session_id"))
    if native_id is None:
        return
    escaped = re.escape(native_id)
    missing = re.fullmatch(
        rf"\s*(?:No (?:saved )?(?:session|conversation) found (?:with |for )?"
        rf"(?:(?:session )?ID[: ]+)?['\"]?{escaped}['\"]?"
        rf"|(?:Session|Conversation)(?: ID)?[: ]+['\"]?{escaped}['\"]? (?:was )?not found"
        rf"|Session not found[: ]+['\"]?{escaped}['\"]?)\.?\s*",
        message, re.IGNORECASE,
    )
    if missing:
        record.set(native_session_id=None, native_resume_attempted=False,
                   native_resume_unavailable=True, native_unavailable_session_id=native_id)


def bind_conversation(session_id: str) -> None:
    """Bind routing's resolved conversation to the current job, when present."""
    execution = current_execution()
    if execution is not None:
        execution.bind_conversation(session_id)


@contextmanager
def suspended_run():
    """Keep nested routing/summary calls out of a worker's native session."""
    token = _RUN.set(None)
    try:
        yield
    finally:
        _RUN.reset(token)


async def record(name: str) -> ExecutionRecord | None:
    execution = current_execution()
    if execution is None:
        return None
    return await execution.record({"record": name, "worker": _WORKER.get()})


def _discard_owned_directory(workspace_path: str, namespace: str, identity: str) -> None:
    from . import workspace

    if namespace not in _NAMESPACES:
        raise ValueError("invalid backend continuation namespace")
    parent = workspace.workspace_state_path(workspace_path, namespace)
    target = workspace.workspace_state_path(workspace_path, namespace, json_digest(identity))
    # Require the literal directory layout, not even an in-workspace link
    # to another job. shutil.rmtree does not follow child symlink entries.
    if (os.path.normcase(os.path.abspath(parent)) != os.path.normcase(os.path.realpath(parent))
            or os.path.normcase(os.path.abspath(target)) != os.path.normcase(os.path.realpath(target))
            or os.path.dirname(os.path.abspath(target)) != os.path.abspath(parent)):
        raise ValueError("unsafe backend continuation cleanup path")
    if os.path.isdir(target):
        shutil.rmtree(target)


def discard_native_conversation(workspace_path: str, session_id: str) -> None:
    """Delete only this conversation's native state and invalidate late writes."""
    lifetime = _conversation_lifetime(workspace_path, _session_id(session_id))
    with lifetime.lock:
        lifetime.discarded = True
        _discard_owned_directory(workspace_path, "native_sessions", session_id)


async def discard_execution(workspace_path: str, execution_id: str) -> None:
    """Remove only a completed queue entry's owned continuation directory."""

    task = asyncio.create_task(asyncio.to_thread(
        _discard_owned_directory, workspace_path, "executions", execution_id,
    ), name="discard-execution-state")
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await _settle(task)
        raise


@contextmanager
def worker_scope(scope: str):
    token = _WORKER.set(scope)
    try:
        yield
    finally:
        _WORKER.reset(token)


@asynccontextmanager
async def execution_scope(workspace_path: str, execution_id: str | None,
                          original_request: str):
    execution = Execution(
        os.path.normcase(os.path.realpath(workspace_path)),
        execution_id or uuid.uuid4().hex, original_request,
    )
    token = _EXECUTION.set(execution)
    try:
        yield execution
    finally:
        try:
            await execution.flush()
            if execution_id is None:
                # Callers without a durable queue identity cannot resume this
                # directory later. Avoid retaining one private home per turn.
                await discard_execution(execution.workspace_path, execution.execution_id)
        finally:
            _EXECUTION.reset(token)


@asynccontextmanager
async def backend_scope(
    backend: str, model: str | None, approval: str, effort: int, *, native: bool = False,
    native_worker: str | None = None,
):
    """Bind a job record and optionally reuse a separately named native lane.

    The detailed worker identity still owns job evidence and its exact pin.
    Only a new native pin may choose the requested reusable conversation lane.
    Leases cover selection, execution and durable publication so parallel
    workers can never drive the same native session at the same time.
    """
    if native_worker is not None:
        _worker_lane(native_worker)
    execution = current_execution()
    state = None
    head = None
    expected = None
    async with AsyncExitStack() as leases:
        if execution is not None:
            identity = {
                "backend": backend, "model": model, "approval": approval,
                "effort": effort, "worker": _WORKER.get(),
            }
            if native:
                await leases.enter_async_context(_backend_lock(execution, identity))
            state = await execution.record(identity, reload=native)
            if native and execution.conversation is not None:
                # Existing pins are immutable. Their generation's path was
                # computed from the original worker unless a lane was saved.
                # Preserve pre-conversation native jobs in their job home too.
                if state.get("native_record_id") is not None:
                    try:
                        lane = _worker_lane(state.get("native_worker", _WORKER.get()))
                    except ValueError as exc:
                        raise RuntimeError("saved job has an invalid native worker lane") from exc
                elif native_session_id(state.get("native_session_id")) is not None:
                    lane = _WORKER.get()
                else:
                    lane = native_worker if native_worker is not None else _WORKER.get()
                native_identity = {**identity, "worker": lane}
                lifetime = execution.conversation._lifetime
                assert lifetime is not None
                lane_lease = lifetime.lanes.setdefault(json_digest(native_identity), asyncio.Lock())
                await leases.enter_async_context(lane_lease)
                execution.conversation.ensure_live()
                state, head, expected = await execution.native_record(native_identity, state)
                if _WORKER.get() == "main":
                    execution.native_main_record = state
        token = _RUN.set(state)
        try:
            yield state
        finally:
            try:
                if state is not None:
                    async def finish() -> None:
                        assert state is not None
                        await state.flush()
                        if execution is not None:
                            await execution.publish_native_head(state, head, expected)

                    task = asyncio.create_task(finish(), name="flush-backend-state")
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        await _settle(task)
                        raise
            finally:
                _RUN.reset(token)
