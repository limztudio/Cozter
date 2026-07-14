"""Shared low-level utilities."""

import asyncio
import contextlib
import inspect
import json
import logging
import os
import signal
import subprocess
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

COZTER_DIR = ".cozter"  # name of the per-workspace dotfile directory
CONFIG_DIR = os.path.join(  # package-wide config dir (config.json, queues, etc.)
    os.path.dirname(os.path.abspath(__file__)), ".config",
)
_STDERR_CAPTURE_BYTES = 64 * 1024
_BackgroundResult = TypeVar("_BackgroundResult")
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def drain_queue(
    q: asyncio.Queue | None, collect: list | None = None,
) -> None:
    """Empty a queue non-blocking. If collect is given, append items to it."""
    if q is None:
        return
    while not q.empty():
        try:
            msg = q.get_nowait()
        except asyncio.QueueEmpty:
            break
        if collect is not None:
            collect.append(msg)


def atomic_write(target: str, data: dict, tmp_dir: str) -> None:
    """Write data as JSON to target atomically via a temp file + os.replace.

    A crash during the write leaves the temp file orphaned but the target
    untouched, so the file is never left in a half-written corrupt state.
    """
    fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            # fsync before the rename so the data is durably on disk. Without
            # it, a power loss can land the rename while the file's blocks are
            # still zero, leaving a truncated/empty target - which readers
            # treat as "absent" and silently reset to defaults (e.g. a "deny"
            # permission would revert to the more permissive default).
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)  # atomic on same filesystem
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


async def await_cancelled(task: Awaitable[object]) -> None:
    """Await a task after cancellation, ignoring the expected cancel."""
    with contextlib.suppress(asyncio.CancelledError):
        await task


def create_background_task(
    coro: Coroutine[Any, Any, _BackgroundResult],
    *,
    name: str,
    log: logging.Logger | None = None,
) -> asyncio.Task[_BackgroundResult]:
    """Start a background task and log unhandled exceptions when it exits."""
    task = asyncio.create_task(coro, name=name)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(
        lambda done: _finalize_background_task(done, name, log or logger),
    )
    return task


def _finalize_background_task(
    task: asyncio.Task[Any],
    name: str,
    log: logging.Logger,
) -> None:
    _BACKGROUND_TASKS.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        log.exception("Background task %s failed", name)


def terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    """Force-stop a subprocess and, where possible, all of its children.

    CLI backends are spawned with ``start_new_session`` so each leads a new
    process group. Killing only the parent PID orphans the grandchildren the
    CLI spawns (builds, test runs, MCP servers via its own bash tool): they
    are reparented to init and keep running - and mutating the workspace -
    after /stop or an inject-restart. POSIX can signal the new process group;
    Windows instead uses ``taskkill /T`` to terminate the process tree rooted
    at the backend process.

    Guarded so we never signal our own group (which would kill the bot):
    a process that isn't a group leader (its pgid equals ours) or the
    fake HttpAgentProcess (pid <= 0, whose ``kill()`` just cancels a task)
    falls back to a single-target ``proc.kill()``.
    """
    pid = getattr(proc, "pid", None)
    if os.name == "nt" and isinstance(pid, int) and pid > 0:
        # ``asyncio`` has no Windows equivalent of POSIX process groups.
        # ``taskkill /T`` follows the child-process tree, which matters when
        # a .cmd shim launches Node or an agent invokes a build/test command.
        # A brief, best-effort synchronous wait makes the caller's subsequent
        # ``proc.wait()`` safe to treat as complete teardown.  If taskkill is
        # unavailable or rejects an already-exited PID, retain the existing
        # single-process kill as a fallback.
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode == 0:
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
    elif isinstance(pid, int) and pid > 0:
        try:
            pgid = os.getpgid(pid)
            if pgid != os.getpgid(0):
                os.killpg(pgid, signal.SIGKILL)
                return
        except (ProcessLookupError, PermissionError, OSError):
            pass  # already gone or no permission - fall back below
    with contextlib.suppress(OSError):
        proc.kill()


async def kill_and_wait(proc: asyncio.subprocess.Process) -> None:
    """Kill a subprocess (its group if it leads one) and reap it."""
    terminate_process_group(proc)
    with contextlib.suppress(OSError):
        await proc.wait()


async def cleanup_backend_process(
    backend: object,
    proc: asyncio.subprocess.Process,
    *,
    log: logging.Logger = logger,
) -> None:
    """Run an optional backend cleanup hook after its process is reaped."""
    cleanup = getattr(backend, "cleanup_process", None)
    if not callable(cleanup):
        return
    try:
        result = cleanup(proc)
        if inspect.isawaitable(result):
            await result
    except Exception:
        name = getattr(backend, "name", type(backend).__name__)
        log.warning("%s backend cleanup failed", name, exc_info=True)


def save_json_object(path: str, data: dict) -> None:
    """Create *path*'s parent directory and atomically write JSON data."""
    target_dir = os.path.dirname(path) or "."
    os.makedirs(target_dir, exist_ok=True)
    atomic_write(path, data, target_dir)


def normalize_string_list(
    value: object,
    *,
    allow_scalar: bool = False,
    strip: bool = True,
) -> list[str]:
    """Return non-empty strings from a list, optionally accepting one string."""
    if isinstance(value, str) and allow_scalar:
        value = [value]
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip() if strip else item
        if text:
            items.append(text)
    return items


def load_json_object(
    path: str,
    label: str,
    log: logging.Logger | None = None,
) -> dict:
    """Load a JSON object from *path*, returning {} on missing/invalid data."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        (log or logger).warning(
            "Corrupt or unreadable %s (%s): %s", label, path, e,
        )
        return {}
    if isinstance(data, dict):
        return data
    (log or logger).warning("Ignoring non-object %s (%s)", label, path)
    return {}


async def iter_stream_lines(
    stream: asyncio.StreamReader, chunk_size: int = 64 * 1024,
) -> AsyncIterator[str]:
    """Yield decoded stdout lines without StreamReader.readline() limits."""
    buffer = bytearray()

    while True:
        chunk = await stream.read(chunk_size)
        if not chunk:
            if buffer:
                yield buffer.decode("utf-8", errors="replace")
            return

        buffer.extend(chunk)
        parts = buffer.split(b"\n")
        buffer = bytearray(parts.pop())

        for part in parts:
            yield part.decode("utf-8", errors="replace")


async def iter_json_events(
    stream: asyncio.StreamReader,
    *,
    on_invalid: Callable[[str], None] | None = None,
) -> AsyncIterator[dict]:
    """Yield non-empty JSON objects from a line-oriented byte stream."""
    async for line in iter_stream_lines(stream):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            if on_invalid:
                on_invalid(stripped)
            continue
        if isinstance(event, dict):
            yield event
        elif on_invalid:
            on_invalid(stripped)


async def drain_text_stream(
    stream: asyncio.StreamReader | None,
    *,
    limit: int = _STDERR_CAPTURE_BYTES,
) -> str:
    """Drain a byte stream and return decoded text capped to *limit* bytes."""
    if stream is None:
        return ""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            break
        if total < limit:
            remaining = limit - total
            chunks.append(chunk[:remaining])
        total += len(chunk)

    text = b"".join(chunks).decode("utf-8", errors="replace").strip()
    if total > limit:
        suffix = f"... [stderr truncated, {total} bytes total]"
        text = f"{text}\n{suffix}" if text else suffix
    return text


def extract_marker_block(text: str, tag: str) -> str | None:
    """Return the body of ``[TAG]...[/TAG]`` (stripped), or None if absent."""
    open_tag = f"[{tag}]"
    close_tag = f"[/{tag}]"
    i = text.find(open_tag)
    if i == -1:
        return None
    j = text.find(close_tag, i + len(open_tag))
    if j == -1:
        return None
    return text[i + len(open_tag):j].strip()


def strip_marker_block(text: str, tag: str) -> str:
    """Return *text* with the first ``[TAG]...[/TAG]`` block removed."""
    open_tag = f"[{tag}]"
    close_tag = f"[/{tag}]"
    i = text.find(open_tag)
    if i == -1:
        return text
    j = text.find(close_tag, i + len(open_tag))
    if j == -1:
        return text
    return text[:i] + text[j + len(close_tag):]


def take_recent_lines(
    items: list,
    budget: int,
    formatter,
) -> list[str]:
    """Format the most recent items that fit in *budget* chars.

    Iterates ``items`` newest-first, calls ``formatter(item)`` on each,
    accumulates lines until the next one would exceed ``budget``, then
    reverses back into chronological order. Newlines that join the
    output count toward the budget.
    """
    used = 0
    out: list[str] = []
    for item in reversed(items):
        line = formatter(item)
        if used + len(line) > budget:
            break
        out.append(line)
        used += len(line) + 1  # +1 for the joining newline
    out.reverse()
    return out


def parse_bullets(block: str | None) -> list[str]:
    """Parse a block into list items. Accepts ``- `` or ``* `` bullet prefixes
    and skips blank lines. Returns ``[]`` for an empty/None block.
    """
    if not block:
        return []
    items: list[str] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ")):
            line = line[2:].strip()
        if line:
            items.append(line)
    return items


def split_text_chunks(text: str, limit: int) -> list[str]:
    """Split text into <=limit chunks, preferring newline boundaries."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


async def drain_llm_subprocess(
    proc: asyncio.subprocess.Process,
    backend,
    timeout: float,
    label: str,
    *,
    log: logging.Logger | None = None,
) -> str:
    """Drain JSON event lines from an internal LLM subprocess and return
    the last agent text emitted, or an empty string on timeout/no output.

    The subprocess is *always* killed and reaped on exit — including on
    cancellation — so /stop or any other exception path can't leak a
    running subprocess past the cancelled task.
    """
    active_log = log or logger
    raw = ""
    finished = False
    stderr_task = asyncio.create_task(drain_text_stream(proc.stderr))

    def _capture_bare_text(line: str) -> None:
        nonlocal raw
        if not raw:
            raw = line

    assert proc.stdout is not None  # spawned with stdout=PIPE
    try:
        async with asyncio.timeout(timeout):
            async for event in iter_json_events(
                proc.stdout, on_invalid=_capture_bare_text,
            ):
                text = backend.extract_agent_text(event)
                if text:
                    raw = text
            await proc.wait()
            finished = True
    except TimeoutError:
        finished = True
        active_log.warning("%s timed out after %ds", label, timeout)
    finally:
        if proc.returncode is None:
            await kill_and_wait(proc)
        try:
            stderr = await stderr_task
        finally:
            await cleanup_backend_process(backend, proc, log=active_log)
        if finished and not raw:
            suffix = f": {stderr}" if stderr else ""
            active_log.warning(
                "%s produced no output (exit %s)%s",
                label,
                proc.returncode,
                suffix,
            )
        elif stderr:
            active_log.debug("%s stderr: %s", label, stderr)
    return raw


async def run_internal_backend(
    backend,
    workspace_path: str,
    prompt: str,
    model: str | None,
    *,
    timeout: float,
    label: str,
    log: logging.Logger,
    missing_executable_message: str,
    missing_level: int = logging.ERROR,
) -> str | None:
    """Launch and drain an internal no-tools backend call.

    Return ``None`` when the backend executable is missing and an empty
    string when it runs without producing an agent response.
    """
    try:
        proc = await backend.launch(
            workspace_path, prompt, model, approval="full", compaction=True,
        )
    except FileNotFoundError:
        log.log(missing_level, missing_executable_message, backend.executable)
        return None
    return await drain_llm_subprocess(
        proc,
        backend,
        timeout,
        label,
        log=log,
    )
