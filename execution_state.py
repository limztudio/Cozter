"""Durable, job-scoped backend continuation, independent of schedules.

Only actual execution workers bind a backend record. Internal routing, title,
summary and planner calls must remain stateless. Native provider identifiers
never cross a workspace, job, worker, model, permission or reasoning boundary.
"""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

from .utils import _fsync_directory, validate_json_nesting

logger = logging.getLogger(__name__)
_MAX_STATE_BYTES = 64 * 1024 * 1024
_VERSION = 1


def _finite_json_float(value: str) -> float:
    """Reject both nonstandard constants and overflowing numeric literals."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("backend continuation contains a nonfinite number")
    return number


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


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
        _fsync_directory(directory)
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
        return Path(workspace.ensure_workspace_state_dir(
            self.execution.workspace_path, "executions", self.execution.directory_id,
            self.key,
        ))

    @property
    def path(self) -> Path:
        from . import workspace
        # Revalidate containment on every read/write, including existing links.
        return Path(workspace.workspace_state_path(
            self.execution.workspace_path, "executions", self.execution.directory_id,
            self.key, "state.json",
        ))

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
        # Independent snapshot threads can finish out of order. A slow older
        # save must not overwrite a newer save or a synchronous native event.
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
                self.directory
                path = str(self.path)
                await asyncio.to_thread(_write_snapshot, path, snapshot)
                self._written = revision


@dataclass
class Execution:
    workspace_path: str
    execution_id: str
    original_request: str
    records: dict[str, ExecutionRecord] = field(default_factory=dict)

    @property
    def directory_id(self) -> str:
        return _digest(self.execution_id)

    async def record(self, identity: object) -> ExecutionRecord:
        # The queue entry owns identity. Accepted guidance changes the request
        # while retaining the same native conversation and tool evidence.
        # Flexible plans separately fingerprint the latest full requirements.
        key = _digest(identity)
        cached = self.records.get(key)
        if cached is not None:
            return cached
        result = ExecutionRecord(self, key)

        def load() -> dict:
            path = result.path
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
                    or data.get("identity") != key
                    or not isinstance(data.get("values"), dict)):
                raise RuntimeError("Saved backend continuation has an invalid identity or format; inspect it before resuming")
            return data["values"]

        result._values = await asyncio.to_thread(load)
        # The workspace run lock normally serializes this, but independent
        # workers may request the same record concurrently.
        return self.records.setdefault(key, result)

    async def flush(self) -> None:
        results = await asyncio.gather(
            *(item.flush() for item in self.records.values()), return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result


_EXECUTION: ContextVar[Execution | None] = ContextVar("cozter_execution", default=None)
_WORKER: ContextVar[str] = ContextVar("cozter_worker", default="main")
_RUN: ContextVar[ExecutionRecord | None] = ContextVar("cozter_backend_run", default=None)


def current_execution() -> Execution | None:
    return _EXECUTION.get()


def current_run() -> ExecutionRecord | None:
    return _RUN.get()


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


async def discard_execution(workspace_path: str, execution_id: str) -> None:
    """Remove only a completed queue entry's owned continuation directory."""
    from . import workspace

    def remove() -> None:
        parent = workspace.workspace_state_path(workspace_path, "executions")
        target = workspace.workspace_state_path(workspace_path, "executions", _digest(execution_id))
        # Require the literal directory layout, not even an in-workspace link
        # to another job. shutil.rmtree does not follow child symlink entries.
        if (os.path.normcase(os.path.abspath(parent)) != os.path.normcase(os.path.realpath(parent))
                or os.path.normcase(os.path.abspath(target)) != os.path.normcase(os.path.realpath(target))
                or os.path.dirname(os.path.abspath(target)) != os.path.abspath(parent)):
            raise ValueError("unsafe backend continuation cleanup path")
        if os.path.isdir(target):
            shutil.rmtree(target)

    task = asyncio.create_task(asyncio.to_thread(remove), name="discard-execution-state")
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
async def backend_scope(backend: str, model: str | None, approval: str, effort: int):
    execution = current_execution()
    state = None
    if execution is not None:
        state = await execution.record({
            "backend": backend, "model": model, "approval": approval,
            "effort": effort, "worker": _WORKER.get(),
        })
    token = _RUN.set(state)
    try:
        yield state
    finally:
        try:
            if state is not None:
                await state.flush()
        finally:
            _RUN.reset(token)
