"""Shared adapters for in-process HTTP-agent backends.

The orchestrator in :mod:`agent` expects ``backend.launch()`` to return
an :class:`asyncio.subprocess.Process` - the CLI backends (codex,
copilot, claude_code) provide that natively, but HTTP backends like
llama don't spawn anything. :class:`HttpAgentProcess` fakes the
process surface (stdout reader, kill, wait, returncode) so the
orchestrator stays backend-agnostic.

:func:`http_error_translator` is an async context manager that
converts aiohttp's exception hierarchy into ``RuntimeError`` with
bot-user-facing messages for any HTTP agent's request path.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
from collections.abc import AsyncIterator, Awaitable

import aiohttp

from ..utils import HTTP_EVENT_LINE_MAX_BYTES, await_cancelled, mark_http_event_stream

logger = logging.getLogger(__name__)


class HttpAgentProcess:
    """Process-like adapter for in-process HTTP agent loops.

    Duck-types :class:`asyncio.subprocess.Process` so the orchestrator
    in :mod:`agent` consumes events identically to CLI-backed agents:
    ``stdout``/``stderr`` are async streams, ``kill`` cancels the
    underlying task, ``wait`` blocks until it settles, and
    ``returncode`` reports success / cancel / error.
    """

    pid: int = -1

    def __init__(self, label: str) -> None:
        """*label* is used only in the crash log message."""
        self._label = label
        self.stdout: asyncio.StreamReader = asyncio.StreamReader()
        mark_http_event_stream(self.stdout)
        self.stderr: asyncio.StreamReader = asyncio.StreamReader()
        # No stderr on HTTP: EOF readers.
        self.stderr.feed_eof()
        self.returncode: int | None = None
        self._task: asyncio.Task | None = None

    def emit(self, event: dict) -> None:
        """Emit complete bounded JSON, preserving Unicode and escaped controls."""
        try:
            line = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        except UnicodeEncodeError:
            # Lone JSON surrogates are accepted by the existing decoder. Keep
            # their escaped spelling rather than replacing or losing text.
            line = (json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
        if len(line) > HTTP_EVENT_LINE_MAX_BYTES:
            # Fail fast so no reader drops bytes in silence.
            raise RuntimeError("HTTP agent event exceeded the 32 MiB transport limit")
        self.stdout.feed_data(line)

    def kill(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def wait(self) -> int:
        if self._task is not None:
            await await_cancelled(self._task)
        return self.returncode if self.returncode is not None else 0

    def start(self, coro: Awaitable[None]) -> None:
        async def _driver() -> None:
            try:
                await coro
                self.returncode = 0
            except asyncio.CancelledError:
                self.returncode = 130
                raise
            except RuntimeError as exc:
                # User-facing failure: log briefly.
                logger.warning("%s: %s", self._label, exc)
                self.emit({"type": "error", "message": str(exc)})
                self.returncode = 1
            except Exception as exc:
                # Unexpected type = bug: keep traceback.
                logger.exception("%s loop crashed", self._label)
                self.emit({"type": "error", "message": str(exc)})
                self.returncode = 1

        def _finalize(task: asyncio.Task[None]) -> None:
            # A launch cancelled before _driver starts still needs EOF here.
            if task.cancelled():
                if inspect.iscoroutine(coro):
                    coro.close()
                elif isinstance(coro, asyncio.Future):
                    coro.cancel()
                if self.returncode is None:
                    self.returncode = 130
            self.stdout.feed_eof()

        self._task = asyncio.create_task(_driver())
        self._task.add_done_callback(_finalize)


@contextlib.asynccontextmanager
async def http_error_translator(
    label: str,
    timeout_seconds: float | None = None,
) -> AsyncIterator[None]:
    """Map aiohttp client errors to user-facing ``RuntimeError`` messages.

    Wrap the request lifecycle (post + body read) to get clear bot-side
    error messages instead of raw aiohttp exception strings::

        async with http_error_translator(f"X at {url}", timeout_seconds):
            async with sess.post(...) as resp:
                ...

    *label* names the service. When the caller explicitly requested an
    operation timeout, its duration is included in timeout errors.
    """
    try:
        yield
    except aiohttp.ClientConnectorError as exc:
        raise RuntimeError(
            f"{label} is unreachable - check the service is up and"
            " the network/VPN is reachable"
        ) from exc
    except (
        aiohttp.ServerDisconnectedError,
        aiohttp.ClientPayloadError,
    ) as exc:
        raise RuntimeError(
            f"{label} dropped the connection mid-response"
        ) from exc
    except TimeoutError as exc:
        if timeout_seconds is not None and timeout_seconds > 0:
            raise RuntimeError(
                f"{label} request timed out after {timeout_seconds:g}s: {exc}"
            ) from exc
        raise RuntimeError(
            f"{label} request timed out: {exc}"
        ) from exc
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"{label} request failed: {exc}") from exc
