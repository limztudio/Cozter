"""Lightweight subprocess launch and cancellation ownership.

Git plugins use this module without initializing the provider registry. Backend
adapters retain wrappers around the same helpers for their existing patch hooks.
"""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from .utils import (
    close_subprocess_pipe,
    has_managed_process_group,
    kill_and_wait,
    mark_process_group_leader,
)

ProcessCleanup = Callable[[asyncio.subprocess.Process], Awaitable[None]]
CancelledLaunchCleanup = Callable[
    [asyncio.Task[asyncio.subprocess.Process], ProcessCleanup], Coroutine[Any, Any, None],
]


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
    *,
    reap_cancelled: CancelledLaunchCleanup | None = None,
) -> asyncio.subprocess.Process:
    """Transfer a started process, or finish ownership cleanup on cancel."""
    if reap_cancelled is None:
        reap_cancelled = reap_cancelled_process_launch
    try:
        # wait() keeps the owned launch running when cancelled here.
        await asyncio.wait((launch,))
        return launch.result()
    except asyncio.CancelledError:
        cleanup = asyncio.create_task(reap_cancelled(launch, cancel_cleanup))
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
    mark_group: Callable[[asyncio.subprocess.Process], None] | None = None,
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
        if mark_group is None:
            mark_group = mark_process_group_leader
        mark_group(proc)
    return proc


async def reap_failed_subprocess(
    proc: asyncio.subprocess.Process,
    *,
    has_group: Callable[[asyncio.subprocess.Process], bool] | None = None,
    kill: ProcessCleanup | None = None,
    close_pipe: Callable[[asyncio.subprocess.Process, int], None] | None = None,
) -> None:
    """Kill/reap a failed launch and close both owned output pipes.

    Optional callbacks keep the backend wrapper's existing patch hooks local
    while sharing the launch-ownership implementation with standalone tools.
    """
    if has_group is None:
        has_group = has_managed_process_group
    if kill is None:
        kill = kill_and_wait
    if close_pipe is None:
        close_pipe = close_subprocess_pipe
    try:
        if proc.returncode is None or has_group(proc):
            await kill(proc)
        # An exited parent can still have inherited output pipes open.
    finally:
        close_pipe(proc, 1)
        close_pipe(proc, 2)


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
