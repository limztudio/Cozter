"""Shared low-level utilities."""

import asyncio
import json
import logging
import os
import tempfile
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

COZTER_DIR = ".cozter"


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
    """
    fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, target)  # atomic on same filesystem
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


async def iter_stream_lines(
    stream: asyncio.StreamReader, chunk_size: int = 64 * 1024,
) -> AsyncIterator[str]:
    """Yield decoded stdout lines without StreamReader.readline() limits."""
    buffer = bytearray()

    while True:
        chunk = await stream.read(chunk_size)
        if not chunk:
            if buffer:
                yield buffer.decode("utf-8", errors="replace")
            return

        buffer.extend(chunk)
        parts = buffer.split(b"\n")
        buffer = bytearray(parts.pop())

        for part in parts:
            yield part.decode("utf-8", errors="replace")


def extract_marker_block(text: str, tag: str) -> str | None:
    """Return the body of ``[TAG]...[/TAG]`` (stripped), or None if absent."""
    open_tag = f"[{tag}]"
    close_tag = f"[/{tag}]"
    i = text.find(open_tag)
    if i == -1:
        return None
    j = text.find(close_tag, i + len(open_tag))
    if j == -1:
        return None
    return text[i + len(open_tag):j].strip()


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


async def drain_llm_subprocess(
    proc: asyncio.subprocess.Process,
    backend,
    timeout: float,
    label: str,
) -> str:
    """Drain JSON event lines from an internal LLM subprocess and return
    the last agent text emitted, or an empty string on timeout/no output.

    The subprocess is *always* killed and reaped on exit — including on
    cancellation — so /stop or any other exception path can't leak a
    running subprocess past the cancelled task.
    """
    raw = ""
    try:
        async with asyncio.timeout(timeout):
            async for line in iter_stream_lines(proc.stdout):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    if not raw:
                        raw = line  # bare-text fallback
                    continue
                text = backend.extract_agent_text(event)
                if text:
                    raw = text
            await proc.wait()
    except TimeoutError:
        logger.warning("%s timed out after %ds", label, timeout)
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except OSError:
                pass
    return raw
