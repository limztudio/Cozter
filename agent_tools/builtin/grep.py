"""grep: regex search across workspace file contents."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import re
import stat
from typing import Any

from ..base import (
    AgentTool,
    capped_list_tail,
    coerce_int_arg,
    iter_workspace_files,
    object_parameters,
    require_nonempty_string_arg,
    resolve_tool_path,
    summarize_arg,
)

# Skip grep on files bigger than this - usually binary or generated.
_GREP_MAX_FILE_BYTES = 1_000_000  # 1 MB

# Per-match-line truncation so one giant minified line can't blow past
# the agent's tool-result cap and hide every other match.
_GREP_MAX_LINE_CHARS = 200
# Python's built-in regex engine has no per-match deadline. Run scans in a
# killable process instead of a thread so cancel (/stop, new message) always
# reaps the worker instead of leaving an abandoned executor thread behind.
# No wall-clock timeout: the scan runs until it finishes or the turn is
# cancelled.
_GREP_WORKER_JOIN_SECONDS = 0.5


class GrepTool(AgentTool):
    name = "grep"
    description = "Regex-search files (`path:lineno: line`; page remainder when truncated)."
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
        },
        ["pattern"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        pattern_str, error = require_nonempty_string_arg(args, "pattern")
        if error:
            return error
        assert pattern_str is not None  # non-None once error is None
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
        assert search_root is not None  # non-None once error is empty
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

        # regex.search on adversarial input (catastrophic backtracking) is
        # CPU-bound and cannot be interrupted at an await point. A thread
        # would keep running after asyncio cancels its await, so isolate the
        # whole scan in a killable process instead.
        # No timeout: the scan runs until it finishes or the turn is
        # cancelled; cancel reaps the worker (see _scan_in_subprocess).
        try:
            results = await asyncio.to_thread(
                _scan_in_subprocess,
                workspace_path, search_root, file_glob, regex, max_results,
            )
        except Exception as exc:
            return f"Grep failed: {exc}"

        if not results:
            return f"No matches for pattern: {pattern_str}"

        summary = "\n".join(results)
        if len(results) >= max_results:
            summary += capped_list_tail(
                max_results, "max_results", "narrow path/glob",
            )
        return summary

    @staticmethod
    def _scan(
        workspace_path: str,
        search_root: str,
        file_glob: str,
        regex: re.Pattern[str],
        max_results: int,
    ) -> list[str]:
        results: list[str] = []
        for fpath, rel, _root_rel in iter_workspace_files(
            workspace_path, search_root, file_glob,
        ):
            try:
                metadata = os.stat(fpath)
                # os.walk also yields FIFOs, sockets, and device files. A
                # blocking open of one of those can strand this worker thread
                # long after the tool coroutine times out, so grep only reads
                # regular files.
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_size > _GREP_MAX_FILE_BYTES
                ):
                    continue
                with open(fpath, "rb") as f:
                    raw = f.read()
            except OSError:
                continue
            if b"\x00" in raw[:8192]:
                continue  # likely binary
            content = raw.decode("utf-8", errors="replace")
            for lineno, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
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
) -> None:
    """Run one scan in a child process and return its result."""
    try:
        result_conn.send((True, GrepTool._scan(
            workspace_path, search_root, file_glob, regex, max_results,
        )))
    except BaseException as exc:
        # This isolated child is always reaped by the parent. Preserve a
        # concise failure for the tool caller instead of silently returning
        # an empty match set if the filesystem scan itself broke.
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


def _scan_in_subprocess(
    workspace_path: str,
    search_root: str,
    file_glob: str,
    regex: re.Pattern[str],
    max_results: int,
) -> list[str]:
    """Run grep work in a process that cancel always reaps (no timeout)."""
    context = multiprocessing.get_context("spawn")
    receive_conn, send_conn = context.Pipe(duplex=False)
    proc = context.Process(
        target=_scan_worker,
        args=(
            send_conn, workspace_path, search_root, file_glob, regex,
            max_results,
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
    try:
        while True:
            if receive_conn.poll(0.1):
                ok, payload = receive_conn.recv()
                if not ok:
                    raise RuntimeError(str(payload))
                if (
                    not isinstance(payload, list)
                    or not all(isinstance(line, str) for line in payload)
                ):
                    raise RuntimeError("grep worker returned an invalid result")
                return payload
            if not proc.is_alive():
                # A child that exits without writing a result is a real scan
                # failure, not a no-match result.
                raise RuntimeError("grep worker exited without a result")
    finally:
        receive_conn.close()
        _stop_scan_worker(proc)
