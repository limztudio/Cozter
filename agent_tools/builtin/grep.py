"""grep: regex search across workspace file contents."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import re
import stat
from typing import Any

from ... import supervision
from ..base import (
    AgentTool,
    coerce_int_arg,
    get_tool_permission,
    git_discovery_files,
    iter_workspace_files,
    object_parameters,
    require_nonempty_string_arg,
    resolve_tool_path,
    summarize_arg,
    tool_permission_scope,
)
from ..discovery import render_discovery_page

_GREP_MAX_FILE_BYTES = 1_000_000

_GREP_MAX_LINE_CHARS = 200
_GREP_WORKER_JOIN_SECONDS = 0.5


class GrepTool(AgentTool):
    name = "grep"
    description = (
        "Regex-search files (`path:lineno: line`). Use the returned offset to"
        " page remaining matches; restart at offset=0 if files change. Broad"
        " workspace searches use the root repository's Git files; set a"
        " specific path or directory glob for ignored files or nested repositories."
    )
    parameters = object_parameters(
        {
            "pattern": {
                "type": "string",
            },
            "path": {"type": "string"},
            "glob": {"type": "string"},
            "max_results": {
                "type": "integer",
                "description": "max 200.",
            },
            "offset": {
                "type": "integer", "minimum": 0,
                "description": "Matches to skip; use the next offset returned by the tool.",
            },
        },
        ["pattern"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        pattern_str, error = require_nonempty_string_arg(args, "pattern")
        if error:
            return error
        assert pattern_str is not None
        try:
            regex = re.compile(pattern_str)
        except re.error as exc:
            return f"Invalid regex: {exc}"

        raw_path = args.get("path") or "."
        search_root, raw_path, path_error = resolve_tool_path(
            workspace_path, raw_path,
        )
        if path_error:
            return path_error
        assert search_root is not None
        if not os.path.isdir(search_root):
            return f"Not a directory: {raw_path}"

        file_glob = args.get("glob") or "**/*"
        if not isinstance(file_glob, str) or not file_glob:
            file_glob = "**/*"

        max_results = coerce_int_arg(
            args.get("max_results", 50),
            default=50,
            minimum=1,
            maximum=200,
        )
        offset = coerce_int_arg(args.get("offset", 0), default=0, minimum=0)
        supervision.mark_replannable_read()
        details = (
            f"grep pattern={pattern_str!r} root={search_root!r}"
            f" glob={file_glob!r} max_results={max_results} offset={offset}"
        )
        supervision.note_command(details)

        # CPU-bound regex: killable process, 0.1s polls.
        try:
            source_files = await git_discovery_files(
                workspace_path, search_root, file_glob,
            )
            results = await _scan_in_subprocess_async(
                workspace_path, search_root, file_glob, regex, max_results,
                offset, source_files, details,
            )
        except Exception as exc:
            return f"Grep failed: {exc}"

        if not results:
            if offset:
                return f"No more matches at offset {offset} for pattern: {pattern_str}"
            return f"No matches for pattern: {pattern_str}"

        return render_discovery_page(results, offset=offset, max_results=max_results)

    @staticmethod
    def _scan(
        workspace_path: str,
        search_root: str,
        file_glob: str,
        regex: re.Pattern[str],
        max_results: int,
        offset: int = 0,
        *, sort_entries: bool = False,
        source_files: list[str] | None = None,
    ) -> list[str]:
        results: list[str] = []
        skipped = 0
        for fpath, rel, _root_rel in iter_workspace_files(
            workspace_path, search_root, file_glob,
            sort_entries=sort_entries,
            source_files=source_files,
        ):
            try:
                metadata = os.stat(fpath)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_size > _GREP_MAX_FILE_BYTES
                ):
                    continue
                with open(fpath, "rb") as file_handle:
                    # A file may grow after stat(); bound the read.
                    raw = file_handle.read(_GREP_MAX_FILE_BYTES + 1)
            except OSError:
                continue
            if len(raw) > _GREP_MAX_FILE_BYTES:
                continue
            if b"\x00" in raw[:8192]:
                continue
            content = raw.decode("utf-8", errors="replace")
            for lineno, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    if skipped < offset:
                        skipped += 1
                        continue
                    display_line = line
                    if len(line) > _GREP_MAX_LINE_CHARS:
                        display_line = (
                            line[:_GREP_MAX_LINE_CHARS - len("… [line clipped]")]
                            + "… [line clipped]"
                        )
                    results.append(f"{rel}:{lineno}: {display_line}")
                    if len(results) >= max_results:
                        return results
        return results

    def summarize(self, args: dict) -> str:
        return summarize_arg("grep", args, "pattern", default="?")


def _scan_worker(
    result_conn: Any,
    workspace_path: str,
    search_root: str,
    file_glob: str,
    regex: re.Pattern[str],
    max_results: int,
    approval: str = "auto",
    offset: int = 0,
    source_files: list[str] | None = None,
) -> None:
    """Run one scan in a child process and return its result."""
    try:
        with tool_permission_scope(approval):
            result_conn.send((True, GrepTool._scan(
                workspace_path, search_root, file_glob, regex, max_results,
                offset, sort_entries=True,
                source_files=source_files,
            )))
    except BaseException as exc:
        # Report scan failures, not false empties.
        try:
            result_conn.send((False, f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        result_conn.close()


def _stop_scan_worker(proc: Any) -> None:
    """Join a completed worker or forcibly stop a cancelled one."""
    proc.join(_GREP_WORKER_JOIN_SECONDS)
    if not proc.is_alive():
        return
    proc.terminate()
    proc.join(_GREP_WORKER_JOIN_SECONDS)
    if proc.is_alive():
        proc.kill()
        proc.join(_GREP_WORKER_JOIN_SECONDS)


def _start_scan_worker(
    workspace_path: str,
    search_root: str,
    file_glob: str,
    regex: re.Pattern[str],
    max_results: int,
    offset: int = 0,
    source_files: list[str] | None = None,
) -> tuple[Any, Any]:
    """Spawn the grep worker; return ``(proc, receive_conn)``."""
    context = multiprocessing.get_context("spawn")
    receive_conn, send_conn = context.Pipe(duplex=False)
    proc = context.Process(
        target=_scan_worker,
        args=(
            send_conn, workspace_path, search_root, file_glob, regex,
            max_results, get_tool_permission(),
            offset,
            source_files,
        ),
        daemon=True,
    )
    try:
        proc.start()
    except Exception:
        receive_conn.close()
        send_conn.close()
        raise
    send_conn.close()
    return proc, receive_conn


def _read_scan_payload(receive_conn: Any) -> list[str]:
    """Read and validate one worker result after ``poll()`` said ready."""
    ok, payload = receive_conn.recv()
    if not ok:
        raise RuntimeError(str(payload))
    if (
        not isinstance(payload, list)
        or not all(isinstance(line, str) for line in payload)
    ):
        raise RuntimeError("grep worker returned an invalid result")
    return payload


async def _scan_in_subprocess_async(
    workspace_path: str,
    search_root: str,
    file_glob: str,
    regex: re.Pattern[str],
    max_results: int,
    offset: int = 0,
    source_files: list[str] | None = None,
    details: str = "",
) -> list[str]:
    """Return up to ``max_results + 1`` matches, including page lookahead.

    ``poll()`` waits in 0.1s slices on a helper thread so /stop cancels
    promptly and the ``finally`` below still reaps the worker.
    """
    proc, receive_conn = _start_scan_worker(
        workspace_path, search_root, file_glob, regex, max_results + 1, offset,
        source_files,
    )
    try:
        supervision.attach_process(proc, details)
        while True:
            if await asyncio.to_thread(receive_conn.poll, 0.1):
                return _read_scan_payload(receive_conn)
            if not proc.is_alive():
                # Silent child exit is a scan failure.
                raise RuntimeError("grep worker exited without a result")
    finally:
        receive_conn.close()
        # Reap off-loop so no worker outlives a dead turn.
        await asyncio.to_thread(_stop_scan_worker, proc)
