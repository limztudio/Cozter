"""Shared plumbing for the git agent-tool plugins.

``git_info``/``git_ops``/``git_sync`` share ref validation, one
cancellation-safe runner, and one output clipper; per-tool tables, argv
builders, and permission levels stay in their own modules.
"""

from __future__ import annotations

import asyncio
import codecs
import os
import re
from dataclasses import dataclass, field

from ..base import truncate_with_marker
from ...process_launch import create_captured_subprocess
from ...utils import (
    close_subprocess_pipe, has_managed_process_group, kill_and_wait,
    wait_for_process_exit,
)

MAX_OUTPUT_CHARS = 12_000
MAX_GIT_ERROR_CHARS = 500
TRUNCATION_MARKER = "say PARTIAL + remainder when coverage is unclear"
_GIT_POST_EXIT_DRAIN_SECONDS = 1.0
# Keep draining so long hooks still end.
_GIT_STDOUT_BYTES = MAX_OUTPUT_CHARS * 4
_GIT_STDERR_BYTES = MAX_GIT_ERROR_CHARS * 4


@dataclass
class _GitCapture:
    limit: int
    data: bytearray = field(default_factory=bytearray)
    truncated: bool = False

    async def drain(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(64 * 1024):
            remaining = self.limit - len(self.data)
            self.data.extend(chunk[:remaining])
            self.truncated |= len(chunk) > remaining

    def text(self) -> str:
        text = self.data.decode("utf-8", errors="replace")
        if self.truncated:
            text += f"\n… [git output truncated; {TRUNCATION_MARKER}]"
        return text


@dataclass
class GitTextWindow:
    """Retain a character window while draining and counting all UTF-8 output.

    Incremental decoding keeps offsets stable when a multibyte character spans
    pipe reads. Only the requested window, one read chunk, and decoder state
    remain in memory, including when the requested offset is far into a patch.
    """

    offset: int
    limit: int
    parts: list[str] = field(default_factory=list)
    total_chars: int = 0
    retained_chars: int = 0
    incomplete: bool = False

    def reset(self) -> None:
        self.parts.clear()
        self.total_chars = 0
        self.retained_chars = 0
        self.incomplete = False

    def _accept_text(self, text: str) -> None:
        chunk_start = self.total_chars
        self.total_chars += len(text)
        remaining = self.limit - self.retained_chars
        if remaining <= 0 or self.total_chars <= self.offset:
            return
        start = max(0, self.offset - chunk_start)
        selected = text[start:start + remaining]
        if selected:
            self.parts.append(selected)
            self.retained_chars += len(selected)

    async def drain(self, stream: asyncio.StreamReader) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while chunk := await stream.read(64 * 1024):
            self._accept_text(decoder.decode(chunk))
        self._accept_text(decoder.decode(b"", final=True))

    def text(self) -> str:
        return "".join(self.parts)

# Allowed ref chars; markers/edge checks below reject git-special cases.
REF_RE = re.compile(r"^[A-Za-z0-9._\-/]+$")
_REF_MARKERS = (
    "..", "@{", "~", "^", ":", "?", "*", "[", "\\",
    "'", '"', "`", "$", "(", ")", "|", ";", "&", "<", ">",
    "!", " ", "\t", "\n",
)


class GitFailedError(Exception):
    """Git exited non-zero; carries the model-facing stderr excerpt."""


def has_invalid_ref_chars(name: str) -> bool:
    """Return True when *name* carries characters git treats specially."""
    return any(marker in name for marker in _REF_MARKERS)


def check_ref(name: str) -> str:
    """Validate a revision token so no model flag can enter the argv."""
    if len(name) > 200 or not REF_RE.match(name):
        raise ValueError(f"invalid ref {name!r}")
    if has_invalid_ref_chars(name):
        raise ValueError(f"invalid ref {name!r}")
    if name.startswith(("-", "/", ".")) or name.endswith(("/", ".", ".lock")):
        raise ValueError(f"invalid ref {name!r}")
    return name


def clean_ref(value: object, *, what: str = "ref") -> tuple[str | None, str | None]:
    """Validate a branch/tag/ref name; return (value, error).

    Tuple form for builders that return model-facing ``"Error: ..."`` strings.
    """
    if not isinstance(value, str) or not value.strip():
        return None, f"Error: '{what}' must be a non-empty string"
    name = value.strip()
    try:
        check_ref(name)
    except ValueError:
        return None, f"Error: invalid {what} {name!r}"
    return name, None


def first_stderr_line(stderr: str) -> str:
    """Clip git's first stderr line for the model-facing error detail."""
    first_line = stderr.strip().splitlines()
    raw_detail = first_line[0] if first_line else "?"
    if len(raw_detail) > MAX_GIT_ERROR_CHARS:
        raw_detail = (
            raw_detail[:MAX_GIT_ERROR_CHARS - len("… [clipped]")]
            + "… [clipped]"
        )
    return raw_detail or "?"


def add_common_git_flags(argv: list[str]) -> list[str]:
    """Splice the shared ``-c`` switches into a ``git`` argv."""
    return [
        argv[0],
        "-c", "core.fsmonitor=false",
        "-c", "core.quotepath=false",
        *argv[1:],
    ]


async def git_once(
    argv: list[str],
    workspace: str,
    env: dict[str, str],
    *,
    stdout_window: GitTextWindow | None = None,
) -> tuple[str, str, int]:
    """Run Git with bounded capture and an exited parent's drain grace."""
    proc = await _spawn_git(argv, workspace, env)
    assert proc.stdout is not None and proc.stderr is not None
    stdout: _GitCapture | GitTextWindow
    if stdout_window is None:
        stdout = _GitCapture(_GIT_STDOUT_BYTES)
    else:
        stdout_window.reset()
        stdout = stdout_window
    stderr = _GitCapture(_GIT_STDERR_BYTES)
    readers = (
        asyncio.create_task(stdout.drain(proc.stdout)),
        asyncio.create_task(stderr.drain(proc.stderr)),
    )
    exited = asyncio.create_task(wait_for_process_exit(proc))
    incomplete = False
    try:
        pending_tasks: set[asyncio.Task] = {*readers, exited}
        while not exited.done():
            completed, pending_tasks = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED,
            )
            for reader in completed:
                # Surface a broken pipe reader immediately while the parent
                # is alive; successful EOF alone must still await its exit.
                reader.result()
        await exited
        _, pending = await asyncio.wait(
            readers, timeout=_GIT_POST_EXIT_DRAIN_SECONDS,
        )
        incomplete = bool(pending)
        if incomplete and has_managed_process_group(proc):
            await kill_and_wait(proc)
        for reader in readers:
            if reader.done():
                reader.result()
    except BaseException:
        if proc.returncode is None or has_managed_process_group(proc):
            await kill_and_wait(proc)
        raise
    finally:
        close_subprocess_pipe(proc, 1)
        close_subprocess_pipe(proc, 2)
        for reader in (*readers, exited):
            if not reader.done():
                reader.cancel()
        await asyncio.gather(*readers, exited, return_exceptions=True)
    output = stdout.text()
    if incomplete:
        if stdout_window is not None:
            stdout_window.incomplete = True
        else:
            output += (
                "\n… [git exited while a child kept an output pipe open;"
                " output may be incomplete]"
            )
    return (
        output,
        stderr.text(),
        proc.returncode if proc.returncode is not None else -1,
    )


async def _spawn_git(
    argv: list[str], workspace: str, env: dict[str, str],
) -> asyncio.subprocess.Process:
    """Retain ownership when cancellation races process creation."""
    launch = asyncio.create_task(create_captured_subprocess(
        argv, cwd=workspace, env=env, start_new_session=os.name != "nt",
    ))
    try:
        # wait() retains the owned launch on cancellation without creating
        # a shield future that logs an already-consumed failure on 3.14.
        await asyncio.wait((launch,))
        return launch.result()
    except asyncio.CancelledError:
        cleanup = asyncio.create_task(_reap_cancelled_git_launch(launch))
        while not cleanup.done():
            try:
                await asyncio.wait((cleanup,))
            except asyncio.CancelledError:
                continue
        cleanup.result()
        raise


async def _reap_cancelled_git_launch(
    launch: asyncio.Task[asyncio.subprocess.Process],
) -> None:
    try:
        proc = await launch
    except (Exception, asyncio.CancelledError):
        return
    try:
        if proc.returncode is None or has_managed_process_group(proc):
            await kill_and_wait(proc)
    finally:
        close_subprocess_pipe(proc, 1)
        close_subprocess_pipe(proc, 2)


def bounded(text: str) -> str:
    """Clip oversized tool output with the shared truncation marker."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return truncate_with_marker(
        text, MAX_OUTPUT_CHARS, TRUNCATION_MARKER,
    )


def append_stderr_note(text: str, stderr: str) -> str:
    """Append git's stderr excerpt to *text* with the shared clip marker."""
    if not stderr.strip():
        return text
    clipped_err = stderr.strip()
    if len(clipped_err) > MAX_GIT_ERROR_CHARS:
        clipped_err = (
            clipped_err[:MAX_GIT_ERROR_CHARS - len("… [stderr clipped]")]
            + "… [stderr clipped]"
        )
    return f"{text}\n\ngit said:\n{clipped_err}"


def finish_git_output(
    stdout: str, stderr: str, *, empty_text: str = "OK",
) -> str:
    """Clip stdout/stderr into the model-facing result with empty handling."""
    text = stdout.strip() or empty_text
    return bounded(append_stderr_note(text, stderr))


def finish_git_run_error(action: object, exc: BaseException) -> str:
    """Map a git runner failure to the model-facing ``Error: ...`` string."""
    if isinstance(exc, FileNotFoundError):
        return "Error: git is not installed or not on PATH"
    if isinstance(exc, TimeoutError):
        return f"Error: git {action} timed out"
    return f"Error: git {action}: {exc}"


async def run_git(argv: list[str], workspace: str) -> tuple[str, str]:
    """Run a fixed git argv with prompts disabled; return decoded output."""
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "0",
        "GIT_OPTIONAL_LOCKS": "1",
    }
    argv = add_common_git_flags(argv)
    stdout, stderr, returncode = await git_once(argv, workspace, env)
    if returncode != 0:
        raise GitFailedError(first_stderr_line(stderr))
    return stdout, stderr
