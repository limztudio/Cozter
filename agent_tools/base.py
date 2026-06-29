"""Base interface for agent tools and helpers shared across tools."""

from __future__ import annotations

import html
import fnmatch
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import aiohttp


# Hard cap on raw HTTP body bytes per web tool call so a pathological
# URL can't OOM the bot.
_MAX_FETCH_BYTES = 5 * 1024 * 1024  # 5 MB
HTTP_USER_AGENT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 compatible; CozterAgent/1.0; +https://local"
    )
}
DISCOVERY_SKIP_DIRS = frozenset({
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".cozter",
    ".codex",
})


class AgentTool(ABC):
    """One tool an agent backend can invoke.

    Backend-agnostic: any agent loop (llama-server, OpenAI, Mistral,
    Gemini, Claude API, etc.) can drive these tools. The tools only
    need a workspace path and an args dict; nothing about how the
    model emitted the call leaks into them.

    Subclasses must define:
      - ``name``: identifier the model uses to call the tool.
      - ``description``: model-facing tool description.
      - ``parameters``: JSON-Schema dict for tool arguments.
      - ``run(workspace_path, args)``: async, returns the result string
        the model will read back.

    Subclasses may optionally set:
      - ``file_action``: one of ``"write"``, ``"edit"``, ``"delete"`` if
        the call should surface as a file-status event in the UI.
      - ``order``: integer for the tool-list ordering sent to the model.
        Lower comes first; ties broken alphabetically by ``name``.
        Defaults to 100.
      - ``summarize(args)``: one-line status-display formatter. The
        default returns just the tool name.

    Every concrete subclass auto-registers itself in
    ``AgentTool.registry`` at class-definition time, so adding a new
    tool only requires dropping a new file in this package - no edits
    to ``__init__.py`` or any backend module are needed.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    file_action: str | None = None
    order: int = 100

    # Whether this tool was loaded from ``agent_tools/plugins/`` (True)
    # vs ``agent_tools/builtin/`` (False). Set by the package loader
    # on each registered instance, not on the class. CLI backends use
    # the flag to enumerate plugins in their bash prelude; HTTP backends
    # see plugins as ordinary typed tools in the schema either way.
    is_plugin: bool = False

    # Populated by __init_subclass__. Read by the package's __init__.
    registry: list["AgentTool"] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip intermediate abstract classes that don't implement run().
        # ABCMeta sets cls.__abstractmethods__ AFTER __init_subclass__
        # runs (it's done in ABCMeta._abc_init, called from __new__
        # after super().__new__ returns). So we check the run method's
        # own __isabstractmethod__ flag, which IS set at definition
        # time and survives inheritance: True for an intermediate
        # subclass that hasn't overridden run, False for a concrete
        # implementation.
        if getattr(cls.run, "__isabstractmethod__", False):
            return
        instance = cls()
        # Idempotent: replace any prior registration with the same name
        # so a hot-reload doesn't accumulate duplicates.
        AgentTool.registry[:] = [
            t for t in AgentTool.registry if t.name != instance.name
        ]
        AgentTool.registry.append(instance)

    @property
    def schema(self) -> dict[str, Any]:
        """Return the inner ``function`` dict for OpenAI tool definitions."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @abstractmethod
    async def run(self, workspace_path: str, args: dict) -> str:
        """Execute the tool and return a string the model can read."""

    def summarize(self, args: dict) -> str:
        """One-line summary for the agent's status display."""
        return self.name

    @classmethod
    def run_as_script(cls) -> None:
        """Entry point for bash-mode invocation (CLI-backend plugins).

        Reads JSON args from ``sys.argv[1]`` (defaults to ``"{}"``),
        runs the tool against the current working directory (which the
        CLI subprocess already sets to the workspace via ``cwd=`` or
        ``-C``), and prints the result to stdout. Errors go to stderr
        with a non-zero exit code.

        A plugin file at ``agent_tools/plugins/<name>.py`` becomes
        invocable as ``python -m Cozter.agent_tools.plugins.<name>``
        by ending the file with::

            if __name__ == "__main__":
                MyTool.run_as_script()
        """
        import asyncio
        import json
        import os
        import sys

        raw = sys.argv[1] if len(sys.argv) > 1 else "{}"
        try:
            args = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"Error: invalid JSON args: {exc}", file=sys.stderr)
            sys.exit(2)
        if not isinstance(args, dict):
            print(
                "Error: JSON args must be an object",
                file=sys.stderr,
            )
            sys.exit(2)
        tool = cls()
        try:
            result = asyncio.run(tool.run(os.getcwd(), args))
        except Exception as exc:  # noqa: BLE001
            print(f"Error: tool {cls.__name__} failed: {exc}", file=sys.stderr)
            sys.exit(1)
        print(result)


# ---------------------------------------------------------------------------
# Helpers shared across tools
# ---------------------------------------------------------------------------


def resolve_inside_workspace(workspace: str, path: str) -> str:
    """Return absolute path; raise if it escapes the workspace.

    ``path`` may be relative to the workspace root or an absolute path
    inside it. Symlinks are followed via ``os.path.realpath`` and the
    resolved target must stay under the workspace root.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    abs_ws = os.path.realpath(workspace)
    candidate = (
        path if os.path.isabs(path) else os.path.join(workspace, path)
    )
    abs_path = os.path.realpath(candidate)
    if not (abs_path == abs_ws or abs_path.startswith(abs_ws + os.sep)):
        raise ValueError(f"path escapes workspace: {path}")
    return abs_path


def coerce_int_arg(
    value: object,
    *,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Return an int argument clamped to the provided bounds."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    number = max(minimum, number)
    if maximum is not None:
        number = min(number, maximum)
    return number


def ensure_parent_dir(path: str) -> None:
    """Create the containing directory for *path*, if any."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def source_destination_parameters() -> dict[str, Any]:
    """Return the common schema for tools that move data between paths."""
    return {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "destination": {"type": "string"},
        },
        "required": ["source", "destination"],
    }


def resolve_source_destination(
    workspace: str, args: dict,
) -> tuple[str, str, str, str]:
    """Return raw and resolved source/destination paths from tool args."""
    raw_src = args.get("source", "")
    raw_dst = args.get("destination", "")
    src = resolve_inside_workspace(workspace, raw_src)
    dst = resolve_inside_workspace(workspace, raw_dst)
    return raw_src, raw_dst, src, dst


def summarize_path(action: str, args: dict, default: str = "?") -> str:
    return f"{action}: {args.get('path', default)}"


def summarize_path_pair(action: str, args: dict) -> str:
    return (
        f"{action}: {args.get('source', '?')}"
        f" -> {args.get('destination', '?')}"
    )


def iter_workspace_files(
    workspace: str,
    root: str,
    pattern: str = "**/*",
) -> Iterator[tuple[str, str, str]]:
    """Yield files under *root* that match *pattern*.

    Returns ``(absolute_path, workspace_relative_path, root_relative_path)``.
    Common generated/cache directories are pruned unless the pattern
    explicitly names that directory segment.
    """
    abs_ws = os.path.realpath(workspace)
    abs_root = os.path.realpath(root)
    skip_dirs = _discovery_skip_dirs(pattern)

    for dirpath, dirnames, filenames in os.walk(abs_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            fpath = os.path.join(dirpath, filename)
            real = os.path.realpath(fpath)
            if not (real == abs_ws or real.startswith(abs_ws + os.sep)):
                continue
            root_rel = os.path.relpath(fpath, abs_root)
            if not _path_matches_glob(root_rel, pattern):
                continue
            ws_rel = os.path.relpath(fpath, abs_ws)
            yield fpath, ws_rel, root_rel


def _discovery_skip_dirs(pattern: str) -> set[str]:
    explicit_segments = set(_glob_segments(pattern))
    return {
        d for d in DISCOVERY_SKIP_DIRS
        if d not in explicit_segments
    }


def _glob_segments(pattern: str) -> list[str]:
    normalized = pattern.replace("\\", "/")
    return [
        part for part in normalized.split("/")
        if part not in ("", ".")
    ]


def _path_matches_glob(rel_path: str, pattern: str) -> bool:
    path_parts = _glob_segments(rel_path)
    pattern_parts = _glob_segments(pattern or "**/*")
    return _match_glob_parts(pattern_parts, path_parts)


def _match_glob_parts(pattern_parts: list[str], path_parts: list[str]) -> bool:
    if not pattern_parts:
        return not path_parts
    part = pattern_parts[0]
    if part == "**":
        return (
            _match_glob_parts(pattern_parts[1:], path_parts)
            or (
                bool(path_parts)
                and _match_glob_parts(pattern_parts, path_parts[1:])
            )
        )
    return (
        bool(path_parts)
        and fnmatch.fnmatchcase(path_parts[0], part)
        and _match_glob_parts(pattern_parts[1:], path_parts[1:])
    )


def validate_replacement_strings(
    old: object, new: object,
) -> tuple[str, str] | str:
    """Validate edit-tool replacement args.

    Returns ``(old, new)`` on success, otherwise a model-facing error
    message without a leading ``Error:`` prefix.
    """
    if not isinstance(old, str) or not isinstance(new, str):
        return "old_string and new_string must be strings"
    if old == "":
        return "'old_string' must not be empty"
    return old, new


def apply_string_replacement(
    content: str,
    old: str,
    new: str,
    *,
    replace_all: bool,
) -> tuple[str, int, int]:
    """Apply a validated string replacement and return updated/count/done."""
    count = content.count(old)
    if count == 0 or (count > 1 and not replace_all):
        return content, count, 0
    replacements = count if replace_all else 1
    return content.replace(old, new, replacements), count, replacements


async def read_bounded_text(resp: aiohttp.ClientResponse) -> str:
    """Read up to MAX_FETCH_BYTES from *resp* and decode with its charset."""
    body_bytes = await resp.content.read(_MAX_FETCH_BYTES + 1)
    encoding = resp.charset or "utf-8"
    return body_bytes.decode(encoding, errors="replace")


def html_to_text(value: str) -> str:
    """Strip script/style blocks and remaining tags; collapse whitespace."""
    value = re.sub(
        r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()
