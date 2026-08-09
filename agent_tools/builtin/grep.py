"""grep: regex search across workspace file contents."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import re
import stat
import time
from typing import Any

from ..base import (
    AgentTool,
    coerce_int_arg,
    iter_workspace_files,
    object_parameters,
    require_nonempty_string_arg,
    resolve_inside_workspace,
)

# Skip grep on files bigger than this - usually binary or generated.
_GREP_MAX_FILE_BYTES = 1_000_000  # 1 MB

# Per-match-line truncation so one giant minified line can't blow past
# the agent's tool-result cap and hide every other match.
_GREP_MAX_LINE_CHARS = 200
# Python's built-in regex engine has no per-match deadline. Run scans in a
# short-lived process and cap its lifetime so a catastrophic pattern cannot
# leave an abandoned executor thread consuming CPU after the tool timeout.
_GREP_MAX_SCAN_SECONDS = 30.0
_GREP_WORKER_JOIN_SECONDS = 0.5


class _GrepScanTimeout(RuntimeError):
    """Raised internally when the isolated grep worker exceeded its budget."""


class GrepTool(AgentTool):
    name = "grep"
    description = (
        "Search file contents in the workspace for a regex pattern."
        " Returns matching lines as 'path:lineno: line'. Binary files"
        " and files larger than 1 MB are skipped."
    )
    parameters = object_parameters(
        {
            "pattern": {
                "type": "string",
                "description": "Python regex to search for.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Directory to search in. Defaults to workspace root."
                ),
            },
            "glob": {
                "type": "string",
                "description": (
                    "Glob restricting which files to search, e.g."
                    " '**/*.py'. Defaults to '**/*'."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": (
                    "Maximum matching lines, default 50, max 200."
                ),
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
        if not isinstance(raw_path, str):
            raw_path = "."
        search_root = resolve_inside_workspace(workspace_path, raw_path)
        if not os.path.isdir(search_root):
            return f"Not a directory: {raw_path}"

        file_glob = args.get("glob") or "**/*"
        if not isinstance(file_glob, str) or not file_glob:
            file_glob = "**/*"

        max_results = coerce_int_arg(
            args.get("max_results") or 50,
            default=50,
            minimum=1,
            maximum=200,
        )

        # regex.search on adversarial input (catastrophic backtracking) is
        # CPU-bound and cannot be interrupted at an await point. A thread
        # would keep running after asyncio cancels its await, so isolate the
        # whole scan in a killable process instead.
        timeout = _grep_scan_timeout()
        try:
            results = await asyncio.to_thread(
                _scan_in_subprocess,
                workspace_path, search_root, file_glob, regex, max_results,
                timeout,
            )
        except _GrepScanTimeout:
            return (
                f"Grep timed out after {timeout:g}s and was stopped. "
                "Simplify the regex or narrow the search path."
            )
        except Exception as exc:
            return f"Grep failed: {exc}"

        if not results:
            return f"No matches for pattern: {pattern_str}"

        summary = "\n".join(results)
        if len(results) >= max_results:
            summary += f"\n(stopped at {max_results} matches)"
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
                        display_line = line[:_GREP_MAX_LINE_CHARS] + "..."
                    results.append(f"{rel}:{lineno}: {display_line}")
                    if len(results) >= max_results:
                        return results
        return results

    def summarize(self, args: dict) -> str:
        return f"grep: {args.get('pattern', '?')}"


def _grep_scan_timeout() -> float:
    """Return a finite worker deadline no greater than tool_timeout."""
    # Import lazily because this builtin is discovered while agent_tools is
    # initializing. The generic tool wrapper reads the same setting, so an
    # isolated worker always exits no later than that outer timeout.
    from ... import config

    return min(float(config.get_tool_timeout()), _GREP_MAX_SCAN_SECONDS)


def _scan_worker(
    result_conn: Any,
    workspace_path: str,
    search_root: str,
    file_glob: str,
    regex: re.Pattern[str],
    max_results: int,
) -> None:
    """Run one scan in a child process and return its bounded result."""
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
    """Join a completed worker or forcibly stop an overdue one."""
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
    timeout: float,
) -> list[str]:
    """Run grep work in a process that is always reaped by its deadline."""
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
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if receive_conn.poll(max(0.0, min(remaining, 0.1))):
                ok, payload = receive_conn.recv()
                if not ok:
                    raise RuntimeError(str(payload))
                if (
                    not isinstance(payload, list)
                    or not all(isinstance(line, str) for line in payload)
                ):
                    raise RuntimeError("grep worker returned an invalid result")
                return payload
            if remaining <= 0:
                raise _GrepScanTimeout
            if not proc.is_alive():
                # A child that exits without writing a result is a real scan
                # failure, not a no-match result.
                raise RuntimeError("grep worker exited without a result")
    finally:
        receive_conn.close()
        _stop_scan_worker(proc)
