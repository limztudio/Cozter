"""Shared plumbing for the git agent-tool plugins.

``git_info`` (read-only inspection), ``git_ops`` (local writes), and
``git_sync`` (network sync) each build their own fixed argv, but all
three validate model-supplied ref names the same way, run one git
process with the same cancellation-safe reaping, and clip oversized
output with the same marker. This module holds that shared half so a
fix to validation, process cleanup, or truncation lands once instead
of three times. The per-tool action tables, argv builders, and
permission levels stay in their own modules.
"""

from __future__ import annotations

import asyncio
import re
from contextlib import suppress

from ..base import truncate_with_marker

# Output/error clipping shared by every git tool: large diffs and logs
# are truncated with a marker, first-stderr-line diagnostics are capped.
MAX_OUTPUT_CHARS = 12_000
MAX_GIT_ERROR_CHARS = 500
TRUNCATION_MARKER = "say PARTIAL + remainder when coverage is unclear"

# Model-supplied names may only use these characters; the marker and
# edge checks below reject everything git would treat specially.
REF_RE = re.compile(r"^[A-Za-z0-9._\-/]+$")
_REF_MARKERS = (
    "..", "@{", "~", "^", ":", "?", "*", "[", "\\",
    "'", '"', "`", "$", "(", ")", "|", ";", "&", "<", ">",
    "!", " ", "\t", "\n",
)


class GitFailed(Exception):
    """Git exited non-zero; carries the model-facing stderr excerpt."""


def has_invalid_ref_chars(name: str) -> bool:
    """Return True when *name* carries characters git treats specially."""
    for marker in _REF_MARKERS:
        if marker in name:
            return True
    return False


def check_ref(name: str) -> str:
    """Validate a revision token so no model flag can enter the argv.

    Raises ValueError for bad input (used by git_info's argv builder).
    """
    if len(name) > 200 or not REF_RE.match(name):
        raise ValueError(f"invalid ref {name!r}")
    if has_invalid_ref_chars(name):
        raise ValueError(f"invalid ref {name!r}")
    if name.startswith(("-", "/", ".")) or name.endswith(("/", ".", ".lock")):
        raise ValueError(f"invalid ref {name!r}")
    return name


def clean_ref(value: object, *, what: str = "ref") -> tuple[str | None, str | None]:
    """Validate a branch/tag/ref name; return (value, error).

    Tuple form (used by git_ops/git_sync builders that return
    model-facing ``"Error: ..."`` strings instead of raising).
    """
    if not isinstance(value, str) or not value.strip():
        return None, f"Error: '{what}' must be a non-empty string"
    name = value.strip()
    if len(name) > 200 or not REF_RE.match(name):
        return None, f"Error: invalid {what} {name!r}"
    if has_invalid_ref_chars(name):
        return None, f"Error: invalid {what} {name!r}"
    if name.startswith(("-", "/", ".")) or name.endswith(("/", ".", ".lock")):
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
) -> tuple[str, str, int]:
    """Run one git argv, reaping the process on cancellation."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workspace,
        env=env,
    )
    try:
        stdout_b, stderr_b = await proc.communicate()
    except BaseException:
        # A turn stop cancelled us; never orphan
        # the git process inside Cozter's process group.
        if proc.returncode is None:
            proc.kill()
            with suppress(ProcessLookupError):
                await proc.wait()
        raise
    return (
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
        proc.returncode if proc.returncode is not None else -1,
    )


def bounded(text: str) -> str:
    """Clip oversized tool output with the shared truncation marker."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return truncate_with_marker(
        text, MAX_OUTPUT_CHARS, "say PARTIAL + remainder when coverage is unclear",
    )
