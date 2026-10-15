"""Lightweight subprocess launch and cancellation ownership.

Native backends and standalone tools use the same ownership helpers without
initializing the provider registry.
"""

import asyncio
import os
import tempfile
from collections.abc import Awaitable, Callable

from .utils import (
    close_subprocess_pipe,
    has_managed_process_group,
    kill_and_wait,
    mark_process_group_leader,
)

ProcessCleanup = Callable[[asyncio.subprocess.Process], Awaitable[None]]


async def create_prompt_subprocess(
    cmd: list[str],
    prompt: str,
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> asyncio.subprocess.Process:
    """Spawn a JSONL CLI with the exact UTF-8 prompt available on stdin.

    File-backed stdin lets launch return before the child consumes its input.
    Waiting for a prompt pipe to flush first can deadlock when the child fills
    stdout or stderr before reading stdin, because our output readers start
    only after launch returns. The CLI still reads the same bytes and EOF;
    it must support redirected stdin rather than requiring a terminal.

    TemporaryFile leaves no application-managed prompt path. The child's
    inherited handle remains valid after we close ours, and the OS removes
    the backing file when the last handle closes (immediately unlinked on
    POSIX, delete-on-close on Windows).
    """
    with tempfile.TemporaryFile(mode="w+b") as prompt_input:
        prompt_input.write(prompt.encode("utf-8"))
        prompt_input.seek(0)
        launch = asyncio.create_task(create_piped_subprocess(
            cmd,
            stdin=prompt_input.fileno(),
            cwd=cwd,
            env=env,
            # New session so /stop or /inject kills the group.
            start_new_session=os.name != "nt",
        ))
        # Keep stdin alive until creation settles, even on cancellation.
        return await await_process_launch(launch, reap_failed_subprocess)


async def create_captured_subprocess(
    cmd: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    start_new_session: bool = False,
    cancel_cleanup: ProcessCleanup | None = None,
) -> asyncio.subprocess.Process:
    """Start a captured command, settling creation and cleanup on cancellation."""
    launch = asyncio.create_task(create_piped_subprocess(
        cmd,
        stdin=asyncio.subprocess.DEVNULL,
        cwd=cwd,
        env=env,
        start_new_session=start_new_session,
    ))
    return await await_process_launch(
        launch, cancel_cleanup or reap_failed_subprocess,
    )


async def await_process_launch(
    launch: asyncio.Task[asyncio.subprocess.Process],
    cancel_cleanup: ProcessCleanup,
) -> asyncio.subprocess.Process:
    """Transfer a started process, or finish ownership cleanup on cancel."""
    try:
        # wait() keeps the owned launch running when cancelled here.
        await asyncio.wait((launch,))
        return launch.result()
    except asyncio.CancelledError:
        cleanup = asyncio.create_task(reap_cancelled_process_launch(launch, cancel_cleanup))
        while not cleanup.done():
            try:
                await asyncio.wait((cleanup,))
            except asyncio.CancelledError:
                # A second /stop must not interrupt ownership cleanup.
                continue
        cleanup.result()
        raise


async def create_piped_subprocess(
    cmd: list[str],
    *,
    stdin: int,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    start_new_session: bool = False,
) -> asyncio.subprocess.Process:
    """Spawn a process whose stdout and stderr are always captured."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=stdin,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
        start_new_session=start_new_session,
    )
    if start_new_session:
        mark_process_group_leader(proc)
    return proc


async def reap_failed_subprocess(
    proc: asyncio.subprocess.Process,
) -> None:
    """Kill/reap a failed launch and close both owned output pipes."""
    try:
        if proc.returncode is None or has_managed_process_group(proc):
            await kill_and_wait(proc)
        # An exited parent can still have inherited output pipes open.
    finally:
        close_subprocess_pipe(proc, 1)
        close_subprocess_pipe(proc, 2)


async def reap_cancelled_process_launch(
    launch: asyncio.Task[asyncio.subprocess.Process],
    cancel_cleanup: ProcessCleanup,
) -> None:
    """Reap a cancelled launch once subprocess creation has settled."""
    try:
        proc = await launch
    except (Exception, asyncio.CancelledError):
        # Failed process creation has no subprocess to transfer to a caller.
        return
    await cancel_cleanup(proc)
