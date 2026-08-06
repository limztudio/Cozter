"""Shared fixtures for tests that exercise :class:`BotPlatform` behavior."""

from collections.abc import Iterator
from contextlib import contextmanager
import asyncio
import json
import os
import sys
import tempfile

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
