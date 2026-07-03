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
import json
import logging
from collections.abc import AsyncIterator, Awaitable

import aiohttp

from ..utils import await_cancelled

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
        self.stderr: asyncio.StreamReader = asyncio.StreamReader()
        # No separate stderr channel from the HTTP path; close it now
        # so any reader sees EOF immediately.
        self.stderr.feed_eof()
        self.returncode: int | None = None
        self._task: asyncio.Task | None = None

    def emit(self, event: dict) -> None:
        """Push an event line into stdout for the orchestrator to read."""
        line = (json.dumps(event) + "\n").encode("utf-8")
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
                # User-facing backend failure - network timeout, auth
                # rejection, server-side tool error, etc. The message
                # is already actionable; log briefly without a stack
                # trace so the bot log doesn't fill with noise on
                # transient upstream issues. agent.run will read
                # result.error from the emitted event and may trigger
                # the file-convert retry.
                logger.warning("%s: %s", self._label, exc)
                self.emit({"type": "error", "message": str(exc)})
                self.returncode = 1
            except Exception as exc:
                # Unexpected exception type = real bug somewhere in
                # the backend. Keep the traceback for debugging.
                logger.exception("%s loop crashed", self._label)
                self.emit({"type": "error", "message": str(exc)})
                self.returncode = 1
            finally:
                self.stdout.feed_eof()

        self._task = asyncio.create_task(_driver())


@contextlib.asynccontextmanager
async def http_error_translator(
    label: str,
    sock_read_timeout: int,
) -> AsyncIterator[None]:
    """Map aiohttp client errors to user-facing ``RuntimeError`` messages.

    Wrap the request lifecycle (post + body read) to get clear bot-side
    error messages instead of raw aiohttp exception strings::

        async with http_error_translator(f"X at {url}", sock_read):
            async with sess.post(...) as resp:
                ...

    *label* names the service in each error message. *sock_read_timeout*
    is the configured timeout value (printed in the timeout error so the
    user knows what to raise). All HTTP backends currently share the
    ``llama_socket_timeout`` config knob, so the message references it.
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
        raise RuntimeError(
            f"{label} did not respond within {sock_read_timeout}s"
            " (raise llama_socket_timeout in config.json if your"
            " server is slow, not stuck)"
        ) from exc
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"{label} request failed: {exc}") from exc
