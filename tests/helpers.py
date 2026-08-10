"""Shared fixtures and helpers for tests."""

from collections.abc import Iterator
from contextlib import contextmanager
import asyncio
import json
import os
import signal
import sys
import tempfile
import time

from Cozter import config
from Cozter.backends_bot.base import BotPlatform, MessageHandle


class TestBot(BotPlatform):
    """Concrete no-op platform base for focused bot-logic tests.

    Individual tests override only the I/O behavior relevant to the scenario,
    keeping the abstract-platform plumbing in one place.
    """

    # This is a reusable fake rather than a pytest test case.  Its historical
    # name starts with ``Test``, so mark it explicitly to keep pytest's
    # collection warnings from obscuring real test failures.
    __test__ = False

    @property
    def platform_id(self) -> str:
        return "test:bot"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send_text(
        self, _chat_id: str, _text: str, *, rich: bool = False,
    ) -> MessageHandle | None:
        return None

    async def edit_text(
        self, _handle: MessageHandle, _text: str, *, rich: bool = False,
    ) -> None:
        return None

    async def delete_message(self, _handle: MessageHandle) -> None:
        return None

    async def send_file(self, _chat_id: str, _path: str) -> None:
        return None


@contextmanager
def temporary_config(body: object) -> Iterator[str]:
    """Point the runtime config reader at a temporary JSON configuration."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(body, f)
        old_path = config.CONFIG_PATH
        config.CONFIG_PATH = path
        try:
            yield path
        finally:
            config.CONFIG_PATH = old_path


async def create_python_script_process(script: str) -> asyncio.subprocess.Process:
    """Start a Python child process with both output streams captured."""
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


def process_is_running(pid: int) -> bool:
    """Return whether a process is live, treating Linux zombies as exited."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A reparented child may briefly remain as a zombie after SIGKILL; it
    # cannot keep a pipe open or mutate a workspace. Other POSIX systems may
    # not mount Linux's /proc, where a successful kill(0) is sufficient.
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            parts = f.read().split()
            return len(parts) <= 2 or parts[2] != "Z"
    except OSError:
        return True


def wait_for_process_exit(pid: int, timeout: float = 2.0) -> bool:
    """Wait briefly for a child process to exit and report its final state."""
    deadline = time.monotonic() + timeout
    while process_is_running(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not process_is_running(pid)


def kill_process(pid: int) -> None:
    """Best-effort test cleanup for a process which may already be gone."""
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
