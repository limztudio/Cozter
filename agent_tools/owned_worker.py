"""Move blocking mutations off the loop without abandoning their ownership."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextvars import copy_context
from functools import partial
from typing import ParamSpec, TypeVar


_P = ParamSpec("_P")
_T = TypeVar("_T")


async def run_owned_worker(
    operation: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs,
) -> _T:
    """Finish an owned thread operation before returning or propagating cancel.

    Copying the caller's ContextVars preserves the permission scope. Unlike a
    plain await of the executor future, waiting on it does not forward
    cancellation. A cancelled or timed-out caller cannot permit its next
    mutation while this worker is still writing. Repeated cancellation cannot
    interrupt settlement; a worker failure is consumed when cancellation wins.
    """
    caller = asyncio.current_task()
    if caller is not None and caller.cancelling():
        raise asyncio.CancelledError
    # A direct executor Future also survives asyncio.run's blanket cancellation
    # of child Tasks during shutdown; cancelling a to_thread Task would lose
    # ownership of its still-running executor operation.
    worker = asyncio.get_running_loop().run_in_executor(
        None, copy_context().run, partial(operation, *args, **kwargs),
    )
    try:
        await asyncio.wait((worker,))
        return worker.result()
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.wait((worker,))
            except asyncio.CancelledError:
                continue
        try:
            worker.result()
        except BaseException:
            # The original cancellation remains the visible outcome, including
            # when a late write error arrives after timeout or repeated /stop.
            pass
        raise
