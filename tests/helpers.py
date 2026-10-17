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
import uuid

from Cozter import config
from Cozter.agent_tools.base import _glob_segments, _match_glob_parts
from Cozter.backends_bot.base import BotPlatform, MessageHandle


class BytesContent:
    """In-memory HTTP body with the bounded asynchronous read interface."""

    def __init__(self, body: bytes) -> None:
        self.body = body

    async def read(self, limit: int) -> bytes:
        chunk, self.body = self.body[:limit], self.body[limit:]
        return chunk


class TestBot(BotPlatform):
    """Concrete no-op platform base for focused bot-logic tests.

    Individual tests override only the I/O behavior relevant to the scenario,
    keeping the abstract-platform plumbing in one place.
    """

    # Reusable fake: mark to silence collection warnings.
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


async def persist_queue_entry(
    bot: BotPlatform, uid: str, text: str, chat_id: str,
    *, ephemeral: bool = False, workspace_path: str | None = None,
) -> str:
    """Stage an existing durable job without simulating a new inbound admission.

    Checkpoint and restart tests need a persisted row independently of the
    in-memory dispatch cache. Admission itself is tested through the platform's
    canonical admission method instead of this fixture.
    """
    if ephemeral and workspace_path is None:
        workspace_path = await bot._current_workspace_for_turn(uid, chat_id)
    entry_id = uuid.uuid4().hex
    entry: dict = {"id": entry_id, "text": text, "chat_id": chat_id}
    if ephemeral:
        entry["ephemeral"] = True
    if workspace_path is not None:
        entry["job"] = {"workspace_path": workspace_path}

    def stage() -> None:
        data = bot._read_queue_file()
        data.setdefault(uid, []).append(entry)
        bot._write_queue_file(data)

    async with bot._queue_file_lock:
        await asyncio.to_thread(stage)
    return entry_id


@contextmanager
def temporary_config(body: object) -> Iterator[str]:
    """Point the runtime config reader at a temporary JSON configuration."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        with open(path, "w", encoding="utf-8") as file_handle:
            json.dump(body, file_handle)
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


def path_matches_glob(rel_path: str, pattern: str) -> bool:
    """Apply the discovery matcher to a path in test reference scans."""
    return _match_glob_parts(
        _glob_segments(pattern or "**/*"), _glob_segments(rel_path),
    )


async def cancel_after_delay_and_read_pid(
    task: asyncio.Task[str], pid_path: str,
) -> int:
    """Cancel a running tool after its child has had time to record its PID."""
    await asyncio.sleep(1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    with open(pid_path, encoding="utf-8") as file_handle:
        return int(file_handle.read().strip())


def process_is_running(pid: int) -> bool:
    """Return whether a process is live, treating Linux zombies as exited."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A reparented zombie can't mutate; kill(0) suffices off Linux.
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as file_handle:
            parts = file_handle.read().split()
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
