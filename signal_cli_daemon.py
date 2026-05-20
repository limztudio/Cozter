"""Shared signal-cli daemon helpers.

The daemon is deliberately managed as a singleton per Signal account and
Unix socket. signal-cli should not have several independent receivers for
the same account, so every local script should reuse the same socket and
only fall back to launching the daemon through this guarded path.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import logging
import os
import shutil
import stat
from typing import ClassVar

logger = logging.getLogger(__name__)


class SignalCliDaemon:
    """Process-safe singleton gate for ``signal-cli daemon --socket``."""

    _instances: ClassVar[dict[tuple[str, str, str], "SignalCliDaemon"]] = {}

    def __init__(
        self,
        phone_number: str,
        socket_path: str,
        *,
        signal_cli_path: str = "signal-cli",
    ) -> None:
        self.phone_number = phone_number
        self.socket_path = socket_path
        self.signal_cli_path = signal_cli_path
        self._process: asyncio.subprocess.Process | None = None
        self._start_lock = asyncio.Lock()

    @classmethod
    def get(
        cls,
        phone_number: str,
        socket_path: str,
        *,
        signal_cli_path: str = "signal-cli",
    ) -> "SignalCliDaemon":
        key = (phone_number, socket_path, signal_cli_path)
        daemon = cls._instances.get(key)
        if daemon is None:
            daemon = cls(
                phone_number,
                socket_path,
                signal_cli_path=signal_cli_path,
            )
            cls._instances[key] = daemon
        return daemon

    async def ensure_running(self) -> None:
        """Connect if possible; otherwise let exactly one process start it."""
        if await self._can_connect():
            return

        async with self._start_lock:
            if await self._can_connect():
                return

            socket_dir = os.path.dirname(self.socket_path)
            if socket_dir:
                os.makedirs(socket_dir, exist_ok=True)
            # This file lock is the cross-process part of the singleton:
            # Cozter, scripts, and shells using the same pattern all wait here
            # so only one of them can create the shared signal-cli daemon.
            with _exclusive_file_lock(self._lock_path()):
                if await self._can_connect():
                    return
                self._remove_stale_socket()
                await self._spawn_daemon()
                await self._wait_until_ready()

    async def _can_connect(self, *, timeout: float = 1.0) -> bool:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self.socket_path),
                timeout=timeout,
            )
        except (OSError, asyncio.TimeoutError):
            return False
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()
        del reader
        return True

    async def _spawn_daemon(self) -> None:
        executable = shutil.which(self.signal_cli_path)
        if executable is None:
            raise RuntimeError(
                f"signal-cli executable not found: {self.signal_cli_path}"
            )

        self._process = await asyncio.create_subprocess_exec(
            executable,
            "-a",
            self.phone_number,
            "daemon",
            "--socket",
            self.socket_path,
            "--receive-mode",
            "manual",
            "--ignore-stories",
            "--no-receive-stdout",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info("Started shared signal-cli daemon at %s", self.socket_path)

    async def _wait_until_ready(self, *, timeout: float = 15.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if await self._can_connect(timeout=0.25):
                return
            if self._process is not None and self._process.returncode is not None:
                raise RuntimeError("signal-cli daemon exited before it was ready")
            await asyncio.sleep(0.25)
        raise TimeoutError(
            f"signal-cli daemon socket did not become ready: {self.socket_path}"
        )

    def _remove_stale_socket(self) -> None:
        try:
            mode = os.stat(self.socket_path).st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(mode):
            raise RuntimeError(
                f"signal-cli socket path exists but is not a socket:"
                f" {self.socket_path}"
            )
        os.unlink(self.socket_path)

    def _lock_path(self) -> str:
        return f"{self.socket_path}.lock"


@contextlib.contextmanager
def _exclusive_file_lock(path: str):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
