"""Shared low-level utilities."""

import asyncio
import json
import os
import tempfile


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
