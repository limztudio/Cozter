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

# Skip files above this size (usually binary/generated).
_GREP_MAX_FILE_BYTES = 1_000_000  # 1 MB

# Truncate match lines so one minified line can't hide the other matches.
_GREP_MAX_LINE_CHARS = 200
# No per-match deadline in ``re``: scan in a killable process so cancel
# always reaps the worker (real-work cap: up to tool_timeout).
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

        # Regex is CPU-bound and uninterruptable: isolate in a killable
        # process (real-work cap; cancel reaps it). Poll in 0.1s slices so
        # the loop stays responsive without an extra thread hop.
        try:
            results = await _scan_in_subprocess_async(
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
                # Skip non-regular files (FIFOs/sockets/devices block on open).
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
        # Report scan failures instead of returning a false empty match set.
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
) -> tuple[Any, Any]:
    """Spawn the grep worker; return ``(proc, receive_conn)``."""
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
) -> list[str]:
    """Wait for the grep worker without stalling the event loop.

    ``poll()`` waits in 0.1s slices on a helper thread so /stop cancels
    promptly and the ``finally`` below still reaps the worker.
    """
    proc, receive_conn = _start_scan_worker(
        workspace_path, search_root, file_glob, regex, max_results,
    )
    try:
        while True:
            if await asyncio.to_thread(receive_conn.poll, 0.1):
                return _read_scan_payload(receive_conn)
            if not proc.is_alive():
                # Silent child exit is a scan failure, not "no matches".
                raise RuntimeError("grep worker exited without a result")
    finally:
        receive_conn.close()
        # Reap on a thread so the worker never survives a cancelled turn.
        await asyncio.to_thread(_stop_scan_worker, proc)
