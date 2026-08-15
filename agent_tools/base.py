"""Base interface for agent tools and helpers shared across tools."""

from __future__ import annotations

import errno
import html
import fnmatch
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, suppress
from functools import lru_cache
from typing import Any, ClassVar

import aiohttp

from ..utils import is_path_within


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
      - ``requires_full_permission``: whether an HTTP agent must be in
        ``full`` mode before this tool can be exposed or executed. Use this
        for tools that can escape Cozter's workspace-bounded safety model
        (for example, a direct host shell). Defaults to ``False``.
      - ``summarize(args)``: one-line status-display formatter. The
        default returns just the tool name.

    Every concrete subclass auto-registers itself in
    ``AgentTool.registry`` at class-definition time, so adding a new
    tool only requires dropping a new file in this package - no edits
    to ``__init__.py`` or any backend module are needed.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    parameters: ClassVar[dict[str, Any]] = {}
    file_action: ClassVar[str | None] = None
    order: ClassVar[int] = 100
    requires_full_permission: ClassVar[bool] = False

    # Whether this tool was loaded from ``agent_tools/plugins/`` (True)
    # vs ``agent_tools/builtin/`` (False). Set by the package loader
    # on each registered instance, not on the class. CLI backends use
    # the flag to enumerate plugins in their bash prelude; HTTP backends
    # see plugins as ordinary typed tools in the schema either way.
    is_plugin: bool = False

    # Populated by __init_subclass__. Read by the package's __init__.
    registry: ClassVar[list[AgentTool]] = []

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

    def summarize(self, _args: dict) -> str:
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
        except Exception as exc:
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
    if not is_path_within(abs_path, abs_ws):
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
        # value is an arbitrary tool arg; the except handles non-numerics.
        number = int(value)  # type: ignore[arg-type, call-overload]
    except (TypeError, ValueError, OverflowError):
        number = default
    number = max(minimum, number)
    if maximum is not None:
        number = min(number, maximum)
    return number


def require_nonempty_string_arg(
    args: dict,
    key: str,
    *,
    strip: bool = False,
) -> tuple[str | None, str | None]:
    """Return a required string argument, or a model-facing error."""
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        return None, f"Error: '{key}' must be a non-empty string"
    return value.strip() if strip else value, None


def ensure_parent_dir(path: str) -> None:
    """Create the containing directory for *path*, if any."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def object_parameters(
    properties: dict[str, Any],
    required: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Return a JSON-Schema object parameter declaration."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
    }


def source_destination_parameters() -> dict[str, Any]:
    """Return the common schema for tools that move data between paths."""
    return object_parameters(
        {
            "source": {"type": "string"},
            "destination": {"type": "string"},
        },
        ["source", "destination"],
    )


def path_parameters() -> dict[str, Any]:
    """Return the common schema for tools that operate on one path."""
    return object_parameters(
        {
            "path": path_property(),
        },
        ["path"],
    )


def path_property(description: str | None = None) -> dict[str, Any]:
    """Return a fresh JSON-Schema property for one workspace path."""
    schema: dict[str, Any] = {"type": "string"}
    if description:
        schema["description"] = description
    return schema


def replacement_properties() -> dict[str, Any]:
    """Return the common replacement fields for the edit tools.

    A fresh mapping on every call keeps each tool's JSON schema independent:
    tool loaders or plugin code can safely inspect or augment one without
    altering another.
    """
    return {
        "old_string": {"type": "string"},
        "new_string": {"type": "string"},
        "replace_all": {
            "type": "boolean",
            "description": (
                "When true, replace every occurrence. When false"
                " (default), require a unique match."
            ),
        },
    }


def prepare_source_destination(
    workspace: str,
    args: dict,
    *,
    file_action: str | None = None,
) -> tuple[str, str, str, str] | str:
    """Return validated transfer paths or a model-facing preflight error."""
    raw_src = args.get("source", "")
    raw_dst = args.get("destination", "")
    src = resolve_inside_workspace(workspace, raw_src)
    dst = resolve_inside_workspace(workspace, raw_dst)
    if not os.path.exists(src):
        return f"Source not found: {raw_src}"
    if file_action is not None and not os.path.isfile(src):
        return f"Not a file (refusing to {file_action}): {raw_src}"
    if os.path.exists(dst):
        return f"Destination already exists: {raw_dst}"
    return raw_src, raw_dst, src, dst


def summarize_path(action: str, args: dict, default: str = "?") -> str:
    return f"{action}: {args.get('path', default)}"


def summarize_path_pair(action: str, args: dict) -> str:
    return (
        f"{action}: {args.get('source', '?')}"
        f" -> {args.get('destination', '?')}"
    )


def summarize_arg(
    action: str,
    args: dict,
    key: str,
    *,
    max_chars: int = 200,
) -> str:
    value = args.get(key, "")
    if not isinstance(value, str):
        value = str(value)
    return f"{action}: {value[:max_chars]}" + (
        "..." if len(value) > max_chars else ""
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
            if not is_path_within(real, abs_ws):
                continue
            root_rel = os.path.relpath(fpath, abs_root).replace(os.sep, "/")
            if not _path_matches_glob(root_rel, pattern):
                continue
            ws_rel = os.path.relpath(fpath, abs_ws).replace(os.sep, "/")
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
    """Match glob path segments without re-exploring ``**`` branches.

    Adjacent (or otherwise repeated) ``**`` segments have many equivalent
    ways to consume the same path components.  Caching the pair of indexes
    keeps an agent-supplied pattern such as ``**/**/**/...`` polynomial
    instead of exponential while preserving the existing matching rules.
    """
    @lru_cache(maxsize=None)
    def matches(pattern_index: int, path_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        part = pattern_parts[pattern_index]
        if part == "**":
            return (
                matches(pattern_index + 1, path_index)
                or (
                    path_index < len(path_parts)
                    and matches(pattern_index, path_index + 1)
                )
            )
        return (
            path_index < len(path_parts)
            and fnmatch.fnmatchcase(path_parts[path_index], part)
            and matches(pattern_index + 1, path_index + 1)
        )

    return matches(0, 0)


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
    if old == new:
        return "old_string and new_string are identical; nothing to change"
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


def read_text_for_edit(path: str) -> tuple[str, bool] | str:
    """Read a UTF-8 text file for in-place editing.

    Returns ``(text, uses_crlf)``: *text* has any ``\\r\\n`` normalized to
    ``\\n`` so newline-based match strings still match a CRLF file, and
    *uses_crlf* records the file's convention so :func:`write_text_after_edit`
    can restore it byte-for-byte. Returns a model-facing error string (no
    ``Error:`` prefix) if the file is not valid UTF-8 - a read-modify-write
    edit would otherwise replace every non-UTF-8 byte in the whole file with
    U+FFFD, silently corrupting content the edit never touched, so we refuse.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        return f"could not read file: {exc}"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return (
            "file is not valid UTF-8 (binary or another encoding); refusing"
            " to edit it because rewriting would corrupt its bytes"
        )
    uses_crlf = "\r\n" in text
    if uses_crlf:
        text = text.replace("\r\n", "\n")
    return text, uses_crlf


def write_text_after_edit(path: str, text: str, *, uses_crlf: bool) -> None:
    """Atomically replace text at *path*, restoring its newline convention.

    ``newline=""`` disables the platform newline translation open() would
    otherwise apply on write (which turns every ``\\n`` into ``\\r\\n`` on
    Windows), so only the bytes the edit actually changed differ on disk.
    Write into the target's directory and replace only after the full file is
    flushed: a write failure can then leave the old source intact instead of
    truncating it midway through an edit, patch application, or overwrite.
    Existing files retain their mode; a missing target is created through the
    same atomic replacement path.
    """
    if uses_crlf:
        text = text.replace("\n", "\r\n")
    parent = os.path.dirname(path) or "."
    try:
        original_mode = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        original_mode = None
    fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if original_mode is not None:
            os.chmod(tmp_path, original_mode)
        os.replace(tmp_path, path)
    except Exception:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise


def create_text_file_atomically(
    path: str, text: str, *, uses_crlf: bool,
) -> bool:
    """Create *path* only when absent, without exposing partial contents.

    The temporary file and target live in the same directory.  Linking the
    completed temporary inode to the target is an atomic no-clobber create:
    unlike ``os.replace``, it fails with :class:`FileExistsError` if another
    writer created the target after the caller's preflight check.  On a
    writable filesystem without hard-link support, an exclusive target
    reservation preserves no-clobber behavior while degrading only the
    crash-atomicity of a brand-new file. Return ``True`` when this call
    created the target and ``False`` when it already existed.
    """
    if uses_crlf:
        text = text.replace("\n", "\r\n")
    parent = os.path.dirname(path) or "."
    fd, tmp_path = _new_text_temp_file(parent, path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            return False
        except OSError as exc:
            if not _hard_link_unsupported(exc):
                raise
            return _copy_to_exclusive_new_path(tmp_path, path)
        return True
    finally:
        # After a successful link the target retains the completed inode, so
        # removing the temporary name cannot affect it.  Best-effort cleanup
        # also handles failures before the target existed.
        with suppress(OSError):
            os.unlink(tmp_path)


def copy_file_atomically(source_path: str, target_path: str) -> bool:
    """Copy one file to a new path without ever overwriting a concurrent file.

    A preflight ``exists()`` check alone cannot uphold a no-clobber contract:
    another writer can create a file (or a symlink) before ``shutil.copy2``
    opens the destination, causing that operation to overwrite it or follow
    the link.  Copy into a completed same-directory temporary file first,
    then publish it with a hard link, whose target creation is atomic and
    fails when the destination already exists.  The no-hard-link fallback
    retains the same no-clobber guarantee through an exclusive reservation.

    Return ``True`` only when this call created *target_path*; return
    ``False`` if it already existed when publishing the completed copy.
    """
    parent = os.path.dirname(target_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
    os.close(fd)
    try:
        # copy2 preserves the documented file metadata on the normal,
        # hard-link publication path. Sync the completed temporary contents
        # before making the destination name visible.
        shutil.copy2(source_path, tmp_path)
        with open(tmp_path, "rb") as tmp_file:
            os.fsync(tmp_file.fileno())
        try:
            os.link(tmp_path, target_path)
        except FileExistsError:
            return False
        except OSError as exc:
            if not _hard_link_unsupported(exc):
                raise
            return _copy_to_exclusive_new_path(
                tmp_path, target_path, preserve_metadata=True,
            )
        return True
    finally:
        # A successful hard link owns the same inode independently, and the
        # fallback has copied it. In every case this temporary name is ours.
        with suppress(OSError):
            os.unlink(tmp_path)


def _new_text_temp_file(parent: str, target_path: str) -> tuple[int, str]:
    """Open a same-directory temp file with normal file-creation mode.

    ``tempfile.mkstemp`` intentionally forces 0600. Source files created by
    a patch historically followed ``open(..., "w")`` and therefore used
    0666 masked by the process umask, so reserve our random temporary name
    with that same mode before it becomes the final hard-linked file.
    """
    prefix = f".{os.path.basename(target_path) or 'new'}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for _ in range(100):
        tmp_path = os.path.join(parent, f"{prefix}.{uuid.uuid4().hex}.tmp")
        try:
            return os.open(tmp_path, flags, 0o666), tmp_path
        except FileExistsError:
            continue
    raise FileExistsError("could not reserve a unique temporary file")


def _hard_link_unsupported(exc: OSError) -> bool:
    """Whether a failed link warrants the exclusive-create fallback."""
    unsupported_errnos = {
        errno.EPERM,
        errno.EXDEV,
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
        errno.EOPNOTSUPP,
    }
    if exc.errno in unsupported_errnos:
        return True
    # Windows reports these when the volume or share does not support hard
    # links. Do not fall back for arbitrary access or I/O failures.
    return getattr(exc, "winerror", None) in {1, 50}


def _copy_to_exclusive_new_path(
    tmp_path: str, path: str, *, preserve_metadata: bool = False,
) -> bool:
    """Copy a completed temp file through an atomically reserved new path.

    This fallback is for filesystems without hard links. ``O_EXCL`` keeps
    the no-clobber contract even when another process races us. A crash while
    copying can leave a partial *new* target, but can never replace another
    writer's target; an ordinary write error removes our reservation when it
    still names the same inode.
    """
    fd: int | None = None
    reserved_stat: os.stat_result | None = None
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError:
        return False
    try:
        reserved_stat = os.fstat(fd)
        with (
            open(tmp_path, "rb") as source,
            os.fdopen(fd, "wb", closefd=True) as output,
        ):
            fd = None
            while chunk := source.read(64 * 1024):
                output.write(chunk)
            output.flush()
            if preserve_metadata:
                _copy_file_metadata_to_fd(tmp_path, output.fileno())
            os.fsync(output.fileno())
        return True
    except BaseException:
        if fd is not None:
            with suppress(OSError):
                os.close(fd)
        if reserved_stat is not None:
            _unlink_if_same_file(path, reserved_stat)
        raise


def _copy_file_metadata_to_fd(source_path: str, target_fd: int) -> None:
    """Best-effort copy of standard file metadata through an open target fd.

    The no-hard-link fallback must not re-open the just-reserved destination
    by pathname: another actor could replace that name between the exclusive
    create and metadata copy. File-descriptor operations preserve mode and
    timestamps without that race. Extended metadata remains platform-specific
    in the fallback, while the normal ``copy2`` + link path retains it.
    """
    source_stat = os.stat(source_path, follow_symlinks=False)
    try:
        os.fchmod(target_fd, stat.S_IMODE(source_stat.st_mode))
        os.utime(
            target_fd,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
        )
    except (AttributeError, OSError):
        # Some Windows filesystems do not support descriptor-based metadata
        # operations. The completed bytes and no-clobber guarantee matter
        # more than metadata on that fallback path.
        return


def _unlink_if_same_file(path: str, expected: os.stat_result) -> None:
    """Remove a failed exclusive reservation only if nobody replaced it."""
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError:
        return
    if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
        return
    with suppress(OSError):
        os.unlink(path)


def move_path_no_clobber(source_path: str, target_path: str) -> bool:
    """Move a path only when *target_path* does not already exist.

    ``os.rename`` replaces an existing target on POSIX, so the ordinary
    absence preflight performed by ``move_file`` cannot uphold its documented
    no-clobber contract when another writer creates the target in between.
    Regular files use an atomic hard-link publication (with the existing
    no-clobber copy fallback across filesystems); symlinks use exclusive
    symlink creation.  Directories and special files need the OS's native
    no-replace rename primitive, because they cannot be safely represented by
    a temporary file/link pair.

    Return ``True`` after a completed move and ``False`` only when another
    path already owns the target name.  Other errors leave the source in
    place whenever this helper can do so safely.
    """
    if os.path.islink(source_path):
        return _move_symlink_no_clobber(source_path, target_path)
    if os.path.isfile(source_path):
        return _move_regular_file_no_clobber(source_path, target_path)
    return _rename_path_no_clobber(source_path, target_path)


def _move_regular_file_no_clobber(source_path: str, target_path: str) -> bool:
    """Publish a regular file at a new name before unlinking its old name."""
    source_stat = os.stat(source_path, follow_symlinks=False)
    try:
        # A hard link is an atomic create: unlike rename, it refuses to
        # replace a destination which appeared after preflight.  Unlinking the
        # original afterwards is a true rename on the usual same-filesystem
        # path and preserves all file metadata.
        os.link(source_path, target_path, follow_symlinks=False)
        # The target is the same inode as the source at publication time, so
        # this identity remains safe even if another writer later changes the
        # source path.
        target_stat = source_stat
    except FileExistsError:
        return False
    except OSError as exc:
        if not _hard_link_unsupported(exc):
            raise
        # A workspace may cross a mount boundary or live on a filesystem
        # without hard links.  The shared helper publishes a finished copy
        # with the same no-clobber guarantee; remove the source only after it
        # succeeds.
        if not copy_file_atomically(source_path, target_path):
            return False
        target_stat = os.stat(target_path, follow_symlinks=False)

    try:
        current_source = os.stat(source_path, follow_symlinks=False)
    except OSError as exc:
        _remove_owned_target(target_path, target_stat)
        raise OSError("source changed while move was in progress") from exc
    if (current_source.st_dev, current_source.st_ino) != (
        source_stat.st_dev, source_stat.st_ino,
    ):
        _remove_owned_target(target_path, target_stat)
        raise OSError("source changed while move was in progress")
    try:
        os.unlink(source_path)
    except OSError:
        # Preserve all-or-nothing behavior when the target was ours.  If a
        # different actor replaced it, _unlink_if_same_file leaves that newer
        # destination untouched.
        _remove_owned_target(target_path, target_stat)
        raise
    return True


def _move_symlink_no_clobber(source_path: str, target_path: str) -> bool:
    """Move a symlink without following or overwriting either endpoint."""
    source_stat = os.stat(source_path, follow_symlinks=False)
    link_target = os.readlink(source_path)
    try:
        os.symlink(
            link_target,
            target_path,
            target_is_directory=os.path.isdir(source_path),
        )
    except FileExistsError:
        return False
    target_stat = os.stat(target_path, follow_symlinks=False)
    try:
        current_source = os.stat(source_path, follow_symlinks=False)
    except OSError as exc:
        _remove_owned_target(target_path, target_stat)
        raise OSError("source changed while move was in progress") from exc
    if (current_source.st_dev, current_source.st_ino) != (
        source_stat.st_dev, source_stat.st_ino,
    ):
        _remove_owned_target(target_path, target_stat)
        raise OSError("source changed while move was in progress")
    try:
        os.unlink(source_path)
    except OSError:
        _remove_owned_target(target_path, target_stat)
        raise
    return True


def _remove_owned_target(target_path: str, expected: os.stat_result) -> None:
    """Remove the target only when it still names the file we just created."""
    _unlink_if_same_file(target_path, expected)


def _rename_path_no_clobber(source_path: str, target_path: str) -> bool:
    """Use a platform-native no-replace rename for non-file paths."""
    if os.name == "nt":
        # Python documents Windows os.rename as failing when the destination
        # exists, unlike POSIX's replacement behavior.
        try:
            os.rename(source_path, target_path)
        except FileExistsError:
            return False
        return True
    if sys.platform.startswith("linux"):
        return _linux_rename_no_replace(source_path, target_path)
    raise OSError(
        getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
        "atomic no-clobber rename is unavailable for this path type",
    )


def _linux_rename_no_replace(source_path: str, target_path: str) -> bool:
    """Call Linux ``renameat2(..., RENAME_NOREPLACE)`` through libc."""
    import ctypes

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as exc:
        raise OSError(
            getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
            "atomic no-clobber rename is unavailable on this system",
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(
        -100, os.fsencode(source_path), -100, os.fsencode(target_path), 1,
    ) == 0:
        return True
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        return False
    if error in {
        errno.EINVAL,
        errno.ENOSYS,
        errno.EOPNOTSUPP,
        getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
    }:
        raise OSError(
            getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
            "atomic no-clobber rename is unavailable for this filesystem",
        )
    raise OSError(error, os.strerror(error), target_path)


async def read_bounded_text(resp: aiohttp.ClientResponse) -> str:
    """Read up to MAX_FETCH_BYTES from *resp* and decode with its charset."""
    chunks: list[bytes] = []
    remaining = _MAX_FETCH_BYTES
    while remaining:
        # StreamReader.read(n) may return fewer than n bytes before EOF, so a
        # single large read can silently truncate a chunked/slow response.
        chunk = await resp.content.read(min(64 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)

    body_bytes = b"".join(chunks)
    encoding = resp.charset or "utf-8"
    return body_bytes.decode(encoding, errors="replace")


@asynccontextmanager
async def open_http_response(
    url: str,
    *,
    timeout: int,
    allow_redirects: bool = True,
) -> AsyncIterator[aiohttp.ClientResponse]:
    """Open one HTTP request with the shared web-tool client settings."""
    async with (
        aiohttp.ClientSession(headers=HTTP_USER_AGENT_HEADERS) as session,
        session.get(
            url,
            allow_redirects=allow_redirects,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response,
    ):
        yield response


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
