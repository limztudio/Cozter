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
# A malformed CLI can emit an unbroken stdout line forever. JSONL normally
# has short event lines, so keep a generous cap while preventing the decoder
# buffer from growing until it exhausts the bot process.
_MAX_STREAM_LINE_BYTES = 4 * 1024 * 1024
# A CLI spawned with ``start_new_session=True`` leads this process group on
# POSIX.  Keep the identifier on the process object at spawn time: by the
# time a leaked descendant is discovered, its direct parent can already have
# exited and ``os.getpgid(parent_pid)`` then cannot recover the group.
_PROCESS_GROUP_ID_ATTR = "_cozter_process_group_id"
# A parent that has exited while its stdout pipe remains open has left that
# descriptor in a child (or another inherited descendant).  After signalling
# the owned group, give pipe shutdown a short grace period before abandoning
# the reader so cancellation and internal LLM calls cannot wedge forever on a
# deliberately detached descendant.
_POST_EXIT_STREAM_DRAIN_TIMEOUT = 1.0
# Once a parent has exited, stdout is no longer trustworthy as a backend
# stream: an ordinary child can keep the descriptor open and emit syntactically
# valid JSON forever.  Keep a short grace for finite parent-authored backlog,
# but cap the amount parsed during that grace so a child cannot drive memory
# growth before the tree cleanup takes effect.
_POST_EXIT_STREAM_DRAIN_BYTES = 8 * 1024 * 1024
_PROCESS_EXIT_POLL_INTERVAL = 0.05
_BackgroundResult = TypeVar("_BackgroundResult")
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def is_path_within(path: object, root: object) -> bool:
    """Return whether *path* resolves inside *root*, including *root*.

    This is the common filesystem-boundary check for workspace and provider
    state paths. Resolving both values rejects ``..`` traversal and symlink
    escapes; ``commonpath`` avoids sibling-prefix false positives such as
    treating ``/work2`` as part of ``/work``. Invalid values and paths on
    different drives are safely treated as outside.
    """
    if not isinstance(path, str) or not isinstance(root, str):
        return False
    if not path or not root or "\x00" in path or "\x00" in root:
        return False
    try:
        resolved_path = os.path.realpath(path)
        resolved_root = os.path.realpath(root)
        return os.path.commonpath((resolved_path, resolved_root)) == resolved_root
    except (OSError, ValueError):
        return False


def try_parse_int(value: str) -> int | None:
    """Return *value* as an integer, or ``None`` when it is invalid.

    Python limits conversion of exceptionally long decimal strings to avoid
    denial-of-service inputs.  User-facing parsers should turn that
    ``ValueError`` into an ordinary invalid value rather than let it escape a
    command or scheduler loop.
    """
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def parse_decimal_int(value: object) -> int | None:
    """Return a decimal text input as an integer, or ``None`` if invalid."""
    if not isinstance(value, str) or not value.isdecimal():
        return None
    return try_parse_int(value)


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
    Both the new file and its containing directory are synced on POSIX: the
    directory sync makes the rename itself durable across a power loss.
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
        _fsync_directory(os.path.dirname(os.path.abspath(target)) or ".")
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _fsync_directory(path: str) -> None:
    """Sync a POSIX directory so a preceding rename survives a crash.

    ``os.replace`` makes a new file visible atomically, but its directory
    entry can still be lost after a sudden power failure unless the parent
    directory is synced too. Windows does not expose a portable directory
    file descriptor through :func:`os.open`, so its existing replace behavior
    is retained there.
    """
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


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


def terminate_windows_process_tree(pid: object) -> bool:
    """Best-effort terminate the Windows process tree rooted at *pid*.

    ``asyncio`` has no Windows equivalent of POSIX process groups.  Calling
    ``taskkill /T`` is therefore the common teardown primitive for both
    long-lived backend processes and short-lived CLI probes.  The boolean
    result lets callers choose their appropriate single-process fallback.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def mark_process_group_leader(proc: asyncio.subprocess.Process) -> None:
    """Record the POSIX process group owned by a newly spawned process.

    Call this immediately after creating a process with
    ``start_new_session=True``.  The process's PID is then also the process
    group ID, and remains useful after the leader has exited while one of its
    children still owns a captured stdout/stderr descriptor.
    """
    if os.name == "nt":
        return
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return
    # asyncio's Process permits custom attributes.  Keep this defensive for
    # process-like adapters supplied by tests or future in-process backends.
    with contextlib.suppress(AttributeError, TypeError):
        setattr(proc, _PROCESS_GROUP_ID_ATTR, pid)


def has_managed_process_group(proc: object) -> bool:
    """Whether *proc* has a POSIX group Cozter created and owns."""
    group_id = getattr(proc, _PROCESS_GROUP_ID_ATTR, None)
    return os.name != "nt" and isinstance(group_id, int) and group_id > 0


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
    if os.name == "nt":
        # ``asyncio`` has no Windows equivalent of POSIX process groups.
        # ``taskkill /T`` follows the child-process tree, which matters when
        # a .cmd shim launches Node or an agent invokes a build/test command.
        # A brief, best-effort synchronous wait makes the caller's subsequent
        # ``proc.wait()`` safe to treat as complete teardown.  If taskkill is
        # unavailable or rejects an already-exited PID, retain the existing
        # single-process kill as a fallback.
        if terminate_windows_process_tree(pid):
            return
    elif has_managed_process_group(proc):
        # Do not ask the OS for the parent's current group here.  It is common
        # for a CLI wrapper to exit after spawning a child that inherited its
        # pipes; at that point getpgid(parent_pid) raises ProcessLookupError,
        # but the original group (and its descendants) still exists.
        pgid = getattr(proc, _PROCESS_GROUP_ID_ATTR)
        try:
            if pgid != os.getpgid(0):
                os.killpg(pgid, signal.SIGKILL)
                return
        except (ProcessLookupError, PermissionError, OSError):
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


def close_subprocess_pipe(proc: object, fd: int) -> None:
    """Best-effort close one captured subprocess pipe transport.

    ``StreamReader`` has no public close operation.  This is only used after
    Cozter has abandoned a reader because a descendant outside the owned tree
    retained the descriptor; closing the private asyncio pipe transport then
    releases the otherwise-leaked file descriptor and lets the subprocess
    transport finish.
    """
    transport = getattr(proc, "_transport", None)
    get_pipe_transport = getattr(transport, "get_pipe_transport", None)
    if not callable(get_pipe_transport):
        return
    with contextlib.suppress(Exception):
        pipe_transport = get_pipe_transport(fd)
        if pipe_transport is not None:
            pipe_transport.close()


async def kill_and_wait(proc: asyncio.subprocess.Process) -> None:
    """Kill a subprocess (its group if it leads one) and await its exit.

    Do not await ``Process.wait()`` here: asyncio can defer that waiter until
    every captured pipe closes, including one still held by a descendant that
    escaped the process group.  The child watcher has already reaped the
    parent once ``returncode`` is populated.
    """
    terminate_process_group(proc)
    try:
        await asyncio.wait_for(
            wait_for_process_exit(proc),
            timeout=_POST_EXIT_STREAM_DRAIN_TIMEOUT,
        )
    except TimeoutError:
        logger.warning(
            "Process %s did not exit after tree teardown",
            getattr(proc, "pid", "?"),
        )


async def wait_for_process_exit(proc: asyncio.subprocess.Process) -> int:
    """Wait until a process has exited without waiting for inherited pipes.

    ``asyncio.subprocess.Process.wait()`` can wait for its captured pipe
    transports to close as well as for the child itself.  A descendant that
    inherited stdout or stderr therefore makes it unsuitable for deciding
    whether the parent has exited.  ``returncode`` is populated by the child
    watcher independently of those pipes.
    """
    while proc.returncode is None:
        await asyncio.sleep(_PROCESS_EXIT_POLL_INTERVAL)
    return proc.returncode


async def finish_process_stderr(
    proc: asyncio.subprocess.Process,
    stderr_task: asyncio.Task[str],
    *,
    preserve_process_tree: Callable[[], bool] | None = None,
) -> str:
    """Return captured stderr without letting an inherited pipe block cleanup.

    The normal drain remains lossless.  If an exited parent leaves stderr open
    in a child, signal the owned group and give the reader a bounded grace
    period.  A descendant that deliberately escaped that group cannot be
    forced down portably, so cancel its reader rather than holding a cancelled
    turn (or daemon shutdown) forever.  A caller that has already recognized a
    provider-owned detached task can instead preserve its tree by supplying a
    predicate; its inherited pipe is still closed after the same bounded grace.
    """
    try:
        if stderr_task.done():
            return await stderr_task

        if proc.returncode is None:
            await kill_and_wait(proc)

        # Give an ordinary reader one final scheduling turn to consume any
        # buffered stderr before deciding an inherited descriptor is still
        # live.  This keeps normal short-lived CLI output unchanged.
        await asyncio.sleep(0)
        if stderr_task.done():
            return await stderr_task

        preserve_tree = bool(
            preserve_process_tree and preserve_process_tree()
        )
        if has_managed_process_group(proc) and not preserve_tree:
            logger.warning(
                "Backend process %s exited with stderr still open; "
                "terminating its process tree",
                getattr(proc, "pid", "?"),
            )
            terminate_process_group(proc)
        elif preserve_tree:
            logger.warning(
                "Backend process %s exited with stderr still open; "
                "preserving its reported detached task",
                getattr(proc, "pid", "?"),
            )

        try:
            return await asyncio.wait_for(
                asyncio.shield(stderr_task),
                timeout=_POST_EXIT_STREAM_DRAIN_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(
                "Backend process %s stderr remained open after cleanup grace; "
                "abandoning stream drain",
                getattr(proc, "pid", "?"),
            )
            close_subprocess_pipe(proc, 2)
            stderr_task.cancel()
            await await_cancelled(stderr_task)
            return ""
    except asyncio.CancelledError:
        # A second cancellation during a turn's finally block must not leave
        # a background reader attached to a leaked pipe.
        if not stderr_task.done():
            close_subprocess_pipe(proc, 2)
            stderr_task.cancel()
            await await_cancelled(stderr_task)
        raise


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
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
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
    """Yield decoded stdout lines without ``readline()`` limits.

    Individual JSONL lines are bounded so a broken child that omits a newline
    cannot make the in-memory decoder buffer grow without limit. An oversized
    line is discarded and the next complete line is still processed.
    """
    async def chunks() -> AsyncIterator[bytes]:
        while chunk := await stream.read(chunk_size):
            yield chunk

    async for line in iter_bounded_lines(
        chunks(),
        max_line_bytes=_MAX_STREAM_LINE_BYTES,
        source="backend stdout",
    ):
        yield line


async def iter_bounded_lines(
    chunks: AsyncIterator[bytes],
    *,
    max_line_bytes: int,
    source: str,
    log: logging.Logger | None = None,
) -> AsyncIterator[str]:
    """Decode newline-delimited *chunks* while bounding each retained line.

    Both subprocess JSONL and HTTP SSE transports can receive arbitrarily
    chunked data. Keeping the framing logic here makes their memory cap and
    recovery behavior identical: discard an oversized line, then resume at
    the next newline instead of retaining an unbounded malformed stream.
    """
    buffer = bytearray()
    discarding_long_line = False

    async for chunk in chunks:
        cursor = 0
        while cursor < len(chunk):
            if discarding_long_line:
                newline = chunk.find(b"\n", cursor)
                if newline == -1:
                    break
                discarding_long_line = False
                cursor = newline + 1
                continue

            newline = chunk.find(b"\n", cursor)
            end = newline if newline != -1 else len(chunk)
            segment = chunk[cursor:end]
            remaining = max_line_bytes - len(buffer)
            if len(segment) > remaining:
                (log or logger).warning(
                    "Discarding %s line larger than %d bytes",
                    source, max_line_bytes,
                )
                buffer.clear()
                if newline == -1:
                    discarding_long_line = True
                    break
                cursor = newline + 1
                continue

            buffer.extend(segment)
            if newline == -1:
                break
            yield buffer.decode("utf-8", errors="replace")
            buffer.clear()
            cursor = newline + 1

    if buffer and not discarding_long_line:
        yield buffer.decode("utf-8", errors="replace")


async def iter_json_events(
    stream: asyncio.StreamReader,
    *,
    on_invalid: Callable[[str], None] | None = None,
    on_event_line: Callable[[str], None] | None = None,
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
            if on_event_line:
                on_event_line(line)
            yield event
        elif on_invalid:
            on_invalid(stripped)


async def iter_process_json_events(
    proc: asyncio.subprocess.Process,
    *,
    on_invalid: Callable[[str], None] | None = None,
    preserve_process_tree: Callable[[], bool] | None = None,
) -> AsyncIterator[dict]:
    """Yield JSON events while detecting descendants that keep stdout open.

    A CLI parent can emit its final JSON line, spawn a background child that
    inherits stdout/stderr, and exit.  Waiting only for ``stdout`` EOF then
    hangs forever: EOF arrives only after that child exits.  Race the next
    stream event against the parent process wait; once the parent has exited
    and no buffered event/EOF is immediately available, terminate the owned
    process group and bound the final pipe drain.  A caller that has already
    recognized a provider-owned detached task can supply a predicate to keep
    that tree alive; the inherited stdout reader is still abandoned after the
    same bounded grace.

    The caller still owns normal process teardown.  This helper only handles
    the special parent-exited/stream-still-open case that would otherwise
    prevent the caller from reaching its ``finally`` block.
    """
    stream = proc.stdout
    assert stream is not None
    parent_exited = False
    post_exit_bytes = 0

    def _account_event_line(line: str) -> None:
        nonlocal post_exit_bytes
        if parent_exited:
            # ``line`` is decoded with replacement, so its re-encoded size
            # is a safe close approximation for a post-exit byte budget.
            post_exit_bytes += len(line.encode("utf-8", errors="replace"))

    events = iter_json_events(
        stream,
        on_invalid=on_invalid,
        on_event_line=_account_event_line,
    ).__aiter__()

    async def _next_event() -> dict:
        return await anext(events)

    next_event_task: asyncio.Task[dict] = asyncio.create_task(_next_event())

    exit_task = asyncio.create_task(wait_for_process_exit(proc))
    loop = asyncio.get_running_loop()
    post_exit_deadline: float | None = None

    async def _abandon_post_exit_stream() -> None:
        """Stop reading an exited parent's inherited stdout safely."""
        preserve_tree = bool(
            preserve_process_tree and preserve_process_tree()
        )
        if preserve_tree:
            logger.warning(
                "Backend process %s exited with stdout still open; "
                "preserving its reported detached task",
                getattr(proc, "pid", "?"),
            )
        else:
            logger.warning(
                "Backend process %s exited with stdout still open; "
                "terminating its process tree",
                getattr(proc, "pid", "?"),
            )
            terminate_process_group(proc)
        # A preserved provider task may intentionally outlive this process,
        # but Cozter must not retain its inherited read end forever.
        close_subprocess_pipe(proc, 1)
        if not next_event_task.done():
            next_event_task.cancel()
            await await_cancelled(next_event_task)

    try:
        while True:
            if parent_exited:
                # Let a finite parent-authored backlog reach the parser, but
                # never let an inherited child continue the turn forever.
                assert post_exit_deadline is not None
                if not next_event_task.done():
                    remaining = post_exit_deadline - loop.time()
                    if remaining <= 0:
                        await _abandon_post_exit_stream()
                        return
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(next_event_task),
                            timeout=remaining,
                        )
                    except StopAsyncIteration:
                        return
                    except TimeoutError:
                        await _abandon_post_exit_stream()
                        return
                if post_exit_bytes >= _POST_EXIT_STREAM_DRAIN_BYTES:
                    # Preserve EOF if it is already next, but discard a
                    # further descendant-authored event instead of parsing
                    # past the bounded post-exit allowance.
                    try:
                        await next_event_task
                    except StopAsyncIteration:
                        return
                    await _abandon_post_exit_stream()
                    return
            else:
                done, _ = await asyncio.wait(
                    (next_event_task, exit_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if exit_task in done:
                    await exit_task
                    parent_exited = True
                    post_exit_deadline = (
                        loop.time() + _POST_EXIT_STREAM_DRAIN_TIMEOUT
                    )
                    continue

            try:
                event = await next_event_task
            except StopAsyncIteration:
                return
            yield event
            next_event_task = asyncio.create_task(_next_event())
    finally:
        if not next_event_task.done():
            close_subprocess_pipe(proc, 1)
            next_event_task.cancel()
            await await_cancelled(next_event_task)
        if not exit_task.done():
            exit_task.cancel()
            await await_cancelled(exit_task)


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


def _marker_block_slices(text: str, tag: str) -> tuple[slice, slice] | None:
    """Return whole-block and body slices for the first ``[TAG]`` block."""
    open_tag = f"[{tag}]"
    close_tag = f"[/{tag}]"
    i = text.find(open_tag)
    if i == -1:
        return None
    body_start = i + len(open_tag)
    j = text.find(close_tag, body_start)
    if j == -1:
        return None
    return slice(i, j + len(close_tag)), slice(body_start, j)


def extract_marker_block(text: str, tag: str) -> str | None:
    """Return the body of ``[TAG]...[/TAG]`` (stripped), or None if absent."""
    slices = _marker_block_slices(text, tag)
    if slices is None:
        return None
    _, body = slices
    return text[body].strip()


def strip_marker_block(text: str, tag: str) -> str:
    """Return *text* with the first ``[TAG]...[/TAG]`` block removed."""
    slices = _marker_block_slices(text, tag)
    if slices is None:
        return text
    block, _ = slices
    return text[:block.start] + text[block.stop:]


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


def text_chunk_ranges(text: str, limit: int) -> list[tuple[int, int]]:
    """Return contiguous <=limit text ranges, preferring newline boundaries.

    The ranges always cover *text* exactly. When a newline is used as a
    boundary it stays at the end of the preceding range, rather than being
    silently discarded with any adjacent blank lines. Keeping offsets also
    lets rich-text senders preserve style ranges after splitting.
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if len(text) <= limit:
        return [(0, len(text))]
    ranges: list[tuple[int, int]] = []
    start = 0
    text_len = len(text)
    while text_len - start > limit:
        split_at = text.rfind("\n", start, start + limit)
        if split_at >= 0:
            end = split_at + 1
        else:
            end = start + limit
        ranges.append((start, end))
        start = end
    ranges.append((start, text_len))
    return ranges


def split_text_chunks(text: str, limit: int) -> list[str]:
    """Split text into <=limit chunks, preferring newline boundaries.

    Concatenating the returned chunks always reconstructs *text* exactly.
    """
    return [
        text[start:end] for start, end in text_chunk_ranges(text, limit)
    ]


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
            async for event in iter_process_json_events(
                proc, on_invalid=_capture_bare_text,
            ):
                text = backend.extract_agent_text(event)
                if text:
                    raw = text
            await wait_for_process_exit(proc)
            finished = True
    except TimeoutError:
        finished = True
        active_log.warning("%s timed out after %ds", label, timeout)
    finally:
        try:
            if proc.returncode is None:
                await kill_and_wait(proc)
            stderr = await finish_process_stderr(proc, stderr_task)
            # Internal LLM calls never own provider-detached work.  Their
            # managed group must therefore end with the foreground response,
            # even when a child closed both captured descriptors first.
            if has_managed_process_group(proc):
                terminate_process_group(proc)
        finally:
            if not stderr_task.done():
                close_subprocess_pipe(proc, 2)
                stderr_task.cancel()
                await await_cancelled(stderr_task)
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

    Internal prompts include user-controlled conversation content.  Keep them
    at the least-privileged ``deny`` level instead of treating summarization,
    titling, and routing as a reason to bypass backend safety controls.

    Return ``None`` when the backend executable is missing and an empty
    string when it runs without producing an agent response.
    """
    try:
        proc = await backend.launch(
            workspace_path, prompt, model, approval="deny", compaction=True,
        )
    except FileNotFoundError:
        log.log(missing_level, missing_executable_message, backend.executable)
        return None
    except (OSError, RuntimeError) as exc:
        log.error("%s could not start: %s", label, exc)
        return ""
    return await drain_llm_subprocess(
        proc,
        backend,
        timeout,
        label,
        log=log,
    )
