"""Claude Code CLI backend.

Claude Code's non-interactive flow (``claude --print --output-format
stream-json --verbose``) reads the prompt from stdin and streams a
JSONL event sequence:

  - ``{"type": "system", ...}`` - init / metadata
  - ``{"type": "assistant", "message": {...}}`` - assistant turn with
    ``content`` blocks (``text`` / ``tool_use``)
  - ``{"type": "user", "message": {...}}`` - tool-result deliveries
  - ``{"type": "result", "subtype": "success", "result": "...", ...}``
    - terminal event carrying the final aggregated assistant text

Workspace access uses the subprocess ``cwd`` (Claude Code has no -C flag).
Permission modes map to Claude's ``--permission-mode`` choices. Internal
text-only calls use ``deny``; ``compaction=True`` never broadens approval.
"""

import asyncio
import glob
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field

from .base import (
    AgentResult, Backend, ChatEvent, DetachedTaskStatus,
    append_detached_task, append_text_result, create_captured_subprocess,
    create_prompt_subprocess, executable_command, record_error_event,
    set_error_result, truncate_status_text,
)
from ..utils import (
    close_subprocess_pipe, is_path_within, wait_for_process_exit,
)

logger = logging.getLogger(__name__)

# Claude Code's effort support is model-specific.  Keep these pinned-model
# exceptions separate from the aliases below: aliases intentionally follow
# whichever current model the installed Claude Code resolves for the account.
_FOUR_LEVEL_EFFORT_MODELS = frozenset({
    "claude-opus-4-6",
    "claude-sonnet-4-6",
})
_NO_EFFORT_MODELS = frozenset({
    "haiku",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-1",
    "claude-opus-4-5",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
})
_FOUR_LEVEL_EFFORTS = ("low", "medium", "high", "max")

# Claude Code does not expose a safe, non-interactive account model catalog
# with numeric capacities. Keep only explicit, provider-documented 1M pins
# and the CLI's explicit ``[1m]`` selections; aliases, default selection, and
# arbitrary/private IDs stay unknown so compaction keeps the message-interval
# safeguard. An operator can configure a private deployment through
# model_context_windows in Cozter's config.json.
_LONG_CONTEXT_WINDOW_TOKENS = 1_000_000
_ONE_MILLION_CONTEXT_MODELS = frozenset({
    # The current Fable/Opus/Sonnet 5 pins have a 1M window by default.
    # Do not infer 1M capacity for mutable aliases or bare 4.x pins: Claude
    # Code can select their 200K variant based on account/provider settings.
    "claude-fable-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "fable[1m]",
    "sonnet[1m]",
    "opus[1m]",
    "opusplan[1m]",
    "claude-opus-4-7[1m]",
    "claude-opus-5[1m]",
    "claude-opus-4-8[1m]",
    "claude-opus-4-6[1m]",
    "claude-sonnet-4-6[1m]",
})

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_BACKGROUND_ID_RE = re.compile(
    r"(?im)^[ \t]*backgrounded[ \t]*(?:·|\*)[ \t]*"
    r"(?P<id>[A-Za-z0-9][A-Za-z0-9_-]{0,127})[ \t]*$",
)
_BACKGROUND_BASH_RE = re.compile(
    r"(?:^|[\s;&|])claude\b[^\n]*--(?:bg|background)\b",
    re.IGNORECASE,
)
_SAFE_BACKGROUND_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_DETACHED_COMMAND_TIMEOUT_SEC = 30
_BACKGROUND_GUARD_TIMEOUT_SEC = 5
# A normal control launcher closes its inherited pipes with the parent.  If a
# Claude supervisor keeps them open instead, retain any already-drained
# prefix but release those pipe transports promptly rather than waiting for
# the worker itself to finish.
_DETACHED_COMMAND_STREAM_DRAIN_TIMEOUT_SEC = 1.0
_DETACHED_COMMAND_EXIT_CLEANUP_TIMEOUT_SEC = 1.0
# Detached-task controls are small metadata commands.  Keep each stream well
# below the size of a normal chat reply, while still leaving plenty of room
# for a busy ``claude agents --json`` response.  The limit is per stream so a
# verbose diagnostic cannot starve the stdout reader and deadlock the child.
_MAX_DETACHED_COMMAND_OUTPUT_BYTES = 1 * 1024 * 1024
_DETACHED_COMMAND_READ_BYTES = 64 * 1024
# Claude's durable JSONL can include tool payloads much larger than the
# assistant text we need to deliver.  Bound an individual physical line
# before calling ``json.loads`` and separately bound the visible text kept
# across the whole transcript.
# JSON framing and metadata add overhead around a valid 4 MiB visible result,
# so permit a larger (still finite) record before deciding it is malformed.
_MAX_DETACHED_TRANSCRIPT_LINE_BYTES = 8 * 1024 * 1024
_MAX_DETACHED_OUTPUT_TEXT_BYTES = 4 * 1024 * 1024
# ``state.json`` carries the fallback result too.  Loading it with
# ``json.load`` would otherwise materialize an arbitrarily large provider
# file before the result-text cap below can take effect.
_MAX_DETACHED_STATE_BYTES = 8 * 1024 * 1024
_DETACHED_OUTPUT_TRUNCATION_MARKER = "\n[truncated]"

def _background_guard_settings() -> str:
    """Return the session-only Claude hook that blocks orphaned Bash jobs."""
    guard_path = os.path.join(
        os.path.dirname(__file__), "claude_background_guard.py",
    )
    return json.dumps({
        # CLI settings take precedence over workspace/user settings.  A
        # lower-priority ``disableAllHooks`` must not turn off Cozter's
        # callback-safety guard for this one session.
        "disableAllHooks": False,
        "hooks": {
            "PreToolUse": [{
                "matcher": "Bash",
                "hooks": [{
                    "type": "command",
                    # ``args`` selects Claude Code's exec form: both the
                    # interpreter and path are passed verbatim, including on
                    # Windows and in workspaces with spaces in their names.
                    "command": sys.executable,
                    "args": [guard_path],
                    "timeout": _BACKGROUND_GUARD_TIMEOUT_SEC,
                }],
            }],
        },
    }, separators=(",", ":"))


def _decode_cli_output(value: bytes | None) -> str:
    """Decode one short Claude CLI command stream defensively."""
    return (value or b"").decode("utf-8", errors="replace").strip()


def _truncate_utf8_text(
    value: str,
    limit: int,
    *,
    marker: str = "",
) -> tuple[str, bool]:
    """Return at most *limit* UTF-8 bytes without splitting a character.

    When it fits, *marker* makes a clipped detached result unambiguous while
    staying inside the same cap.  Extremely small test/configured limits may
    be too short even for the marker; in that case the bounded prefix is the
    only possible safe result.
    """
    limit = max(0, limit)
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return value, False
    marker_bytes = marker.encode("utf-8", errors="replace")
    prefix_limit = limit
    if marker_bytes and len(marker_bytes) <= limit:
        prefix_limit -= len(marker_bytes)
    # ``value`` can contain lone surrogates.  Encode with replacement above,
    # then decode only a complete UTF-8 prefix so the resulting text remains
    # safe for JSON/state delivery and fits the byte budget.
    return (
        encoded[:prefix_limit].decode("utf-8", errors="ignore")
        + (marker if prefix_limit != limit else ""),
        True,
    )


def _append_bounded_transcript_text(
    parts: list[str],
    used_bytes: int,
    text: str,
) -> tuple[int, bool]:
    """Append one visible transcript message within the aggregate cap.

    The separator is counted too: otherwise many tiny messages could evade
    the cap during the final ``join``.  Returning ``True`` means the retained
    output reached its limit and the caller can stop parsing the transcript.
    """
    value = text if not parts else "\n\n" + text
    retained, truncated = _truncate_utf8_text(
        value,
        _MAX_DETACHED_OUTPUT_TEXT_BYTES - used_bytes,
        marker=_DETACHED_OUTPUT_TRUNCATION_MARKER,
    )
    if retained:
        parts.append(retained)
        used_bytes += len(retained.encode("utf-8", errors="replace"))
    return used_bytes, truncated


def _background_task_ids(text: str) -> list[str]:
    """Extract only Claude's dedicated background-launch output lines."""
    normalized = _ANSI_ESCAPE_RE.sub("", text).replace("\r\n", "\n")
    ids: list[str] = []
    for match in _BACKGROUND_ID_RE.finditer(normalized):
        task_id = match.group("id")
        if task_id not in ids:
            ids.append(task_id)
    return ids


def _claude_home() -> str:
    """Return Claude Code's user-level persistence root."""
    return os.path.join(os.path.expanduser("~"), ".claude")


def _workspace_contains(workspace_path: str, candidate: object) -> bool:
    """Return whether a provider-reported directory belongs to this workspace."""
    return is_path_within(candidate, workspace_path)


def _local_background_state(
    workspace_path: str, task_id: str,
) -> dict | None:
    """Read a Claude background job's durable local state, if it is trusted."""
    if not _SAFE_BACKGROUND_ID_RE.fullmatch(task_id):
        return None
    path = os.path.join(_claude_home(), "jobs", task_id, "state.json")
    try:
        with open(path, "rb") as f:
            raw_state = f.read(_MAX_DETACHED_STATE_BYTES + 1)
        if len(raw_state) > _MAX_DETACHED_STATE_BYTES:
            return None
        state = json.loads(raw_state)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict) or not _workspace_contains(
        workspace_path, state.get("cwd"),
    ):
        return None
    return state


def _local_background_status(
    workspace_path: str, task_id: str,
) -> DetachedTaskStatus | None:
    state = _local_background_state(workspace_path, task_id)
    if state is None:
        return None
    value = state.get("state")
    if not isinstance(value, str) or not value:
        return None
    return DetachedTaskStatus(state=value)


def _transcript_text_from_content(content: object) -> str:
    """Flatten one persisted Claude assistant message's visible text blocks.

    A transcript line is capped before this runs, but one message can still
    contain many text blocks.  Keep this intermediate result bounded as well
    instead of building an arbitrarily large ``parts`` list before the outer
    transcript accumulator gets a chance to enforce its limit.
    """
    if isinstance(content, str):
        return _truncate_utf8_text(
            content.strip(),
            _MAX_DETACHED_OUTPUT_TEXT_BYTES,
            marker=_DETACHED_OUTPUT_TRUNCATION_MARKER,
        )[0]
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    used_bytes = 0
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            value = text.strip()
            if parts:
                value = "\n" + value
            retained, truncated = _truncate_utf8_text(
                value,
                _MAX_DETACHED_OUTPUT_TEXT_BYTES - used_bytes,
                marker=_DETACHED_OUTPUT_TRUNCATION_MARKER,
            )
            if retained:
                parts.append(retained)
                used_bytes += len(retained.encode("utf-8", errors="replace"))
            if truncated:
                break
    return "".join(parts)


def _iter_bounded_transcript_lines(path: str):
    """Yield complete JSONL records without retaining an unbounded line."""
    with open(path, "rb") as f:
        while line := f.readline(_MAX_DETACHED_TRANSCRIPT_LINE_BYTES + 1):
            if len(line) <= _MAX_DETACHED_TRANSCRIPT_LINE_BYTES:
                yield line
                continue

            # ``readline(limit)`` returns a prefix of an oversized physical
            # line.  Discard the rest in equally bounded chunks so a malformed
            # no-newline record cannot make the next fragment look like JSON.
            while not line.endswith(b"\n"):
                line = f.readline(_MAX_DETACHED_TRANSCRIPT_LINE_BYTES + 1)
                if not line:
                    break


def _local_background_output(workspace_path: str, task_id: str) -> str:
    """Read the visible result from Claude's durable background transcript.

    ``claude logs`` renders an interactive terminal transcript on current CLI
    versions, which is unsuitable for a chat callback. Claude's persisted
    JSONL transcript is the same durable source the supervisor resumes from
    and preserves just the assistant's actual response text.
    """
    state = _local_background_state(workspace_path, task_id)
    if state is None:
        return ""
    session_id = state.get("sessionId")
    if not isinstance(session_id, str) or not session_id:
        return ""
    if os.path.basename(session_id) != session_id:
        return ""
    project_root = os.path.realpath(os.path.join(_claude_home(), "projects"))
    candidates: list[str] = []
    linked_path = state.get("linkScanPath")
    if isinstance(linked_path, str):
        candidates.append(linked_path)
    pattern = os.path.join(_claude_home(), "projects", "*", f"{session_id}.jsonl")
    candidates.extend(glob.glob(pattern))

    # ``glob`` does not resolve intermediate symlinks.  A project entry may
    # therefore look like a child of ``projects`` while resolving elsewhere;
    # normalize every candidate before opening it, just like linkScanPath.
    paths: list[str] = []
    for candidate in candidates:
        real_path = os.path.realpath(candidate)
        if (
            is_path_within(real_path, project_root)
            and os.path.basename(real_path) == f"{session_id}.jsonl"
            and real_path not in paths
        ):
            paths.append(real_path)
    for path in paths:
        texts: list[str] = []
        text_bytes = 0
        try:
            for line in _iter_bounded_transcript_lines(path):
                try:
                    item = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(item, dict):
                    continue
                message = item.get("message")
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                text = _transcript_text_from_content(message.get("content"))
                if text:
                    text_bytes, truncated = _append_bounded_transcript_text(
                        texts, text_bytes, text,
                    )
                    if truncated:
                        break
        except OSError:
            continue
        if texts:
            return "".join(texts)

    output = state.get("output")
    if isinstance(output, dict):
        summary = output.get("result")
        if isinstance(summary, str):
            return _truncate_utf8_text(
                summary.strip(),
                _MAX_DETACHED_OUTPUT_TEXT_BYTES,
                marker=_DETACHED_OUTPUT_TRUNCATION_MARKER,
            )[0]
    return ""


@dataclass
class _CappedCommandOutput:
    """Mutable bounded stream state, retained if a reader is abandoned."""

    data: bytearray = field(default_factory=bytearray)
    truncated: bool = False


async def _read_capped_command_stream(
    stream: asyncio.StreamReader | None,
    output: _CappedCommandOutput,
) -> None:
    """Drain one control-command stream while retaining only its prefix.

    Draining continues after the cap so a noisy stdout or stderr cannot block
    the sibling stream or leave the command wedged on a full pipe.  Callers
    reject the command rather than parsing this partial prefix as JSON or a
    background-launch acknowledgement.
    """
    if stream is None:
        return
    while chunk := await stream.read(_DETACHED_COMMAND_READ_BYTES):
        remaining = _MAX_DETACHED_COMMAND_OUTPUT_BYTES - len(output.data)
        if remaining > 0:
            output.data.extend(chunk[:remaining])
        if len(chunk) > remaining:
            output.truncated = True


async def _abandon_command_readers(
    proc: asyncio.subprocess.Process,
    readers: tuple[asyncio.Task[None], asyncio.Task[None]],
) -> None:
    """Release output readers that a supervisor descendant kept open."""
    close_subprocess_pipe(proc, 1)
    close_subprocess_pipe(proc, 2)
    for reader in readers:
        if not reader.done():
            reader.cancel()
    await asyncio.gather(*readers, return_exceptions=True)


async def _capture_claude_command_output(
    proc: asyncio.subprocess.Process,
) -> tuple[tuple[bytes, bool], tuple[bytes, bool]]:
    """Drain both command streams without waiting on inherited pipes."""
    stdout = _CappedCommandOutput()
    stderr = _CappedCommandOutput()
    stdout_task = asyncio.create_task(
        _read_capped_command_stream(proc.stdout, stdout),
    )
    stderr_task = asyncio.create_task(
        _read_capped_command_stream(proc.stderr, stderr),
    )
    readers = (stdout_task, stderr_task)
    try:
        # Unlike ``Process.wait()``, this observes the child watcher return
        # code without waiting for pipe EOF from a detached supervisor.
        await wait_for_process_exit(proc)
        _done, pending = await asyncio.wait(
            readers, timeout=_DETACHED_COMMAND_STREAM_DRAIN_TIMEOUT_SEC,
        )
        if pending:
            logger.warning(
                "Claude Code control launcher exited with output pipes "
                "still open; using bounded partial output",
            )
            await _abandon_command_readers(proc, readers)
        else:
            # Propagate a genuine read failure rather than silently parsing a
            # partial response as an agent list or launch acknowledgement.
            await asyncio.gather(*readers)
        return (
            (bytes(stdout.data), stdout.truncated),
            (bytes(stderr.data), stderr.truncated),
        )
    except BaseException:
        # ``wait_for`` cancels this coroutine on timeout.  A background
        # descendant may keep its inherited pipe open after the launcher has
        # exited, so do not leave either reader behind waiting for EOF.
        await _abandon_command_readers(proc, readers)
        raise


async def _run_claude_command(
    cmd: list[str], *, cwd: str,
) -> tuple[int, str, str]:
    """Run a short Claude control command without owning its worker tree."""
    proc = await create_captured_subprocess(cmd, cwd=cwd)
    try:
        stdout_result, stderr_result = await asyncio.wait_for(
            _capture_claude_command_output(proc),
            timeout=_DETACHED_COMMAND_TIMEOUT_SEC,
        )
    except TimeoutError as exc:
        # Do not kill the command's process group here. ``claude --bg``
        # hands a worker to Claude's supervisor; timing out while the
        # launcher is slow must not terminate the detached session itself.
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                # The launcher can exit in the small race between the
                # returncode check and ``kill``; its child watcher will still
                # make the bounded cleanup below observe that exit.
                pass
        close_subprocess_pipe(proc, 1)
        close_subprocess_pipe(proc, 2)
        try:
            await asyncio.wait_for(
                wait_for_process_exit(proc),
                timeout=_DETACHED_COMMAND_EXIT_CLEANUP_TIMEOUT_SEC,
            )
        except TimeoutError:
            logger.warning(
                "Claude Code detached-task control launcher did not exit "
                "after timeout",
            )
        raise RuntimeError("Claude Code detached-task command timed out") from exc
    stdout, stdout_truncated = stdout_result
    stderr, stderr_truncated = stderr_result
    if stdout_truncated or stderr_truncated:
        raise RuntimeError(
            "Claude Code detached-task command output exceeded the "
            f"{_MAX_DETACHED_COMMAND_OUTPUT_BYTES} byte limit",
        )
    return proc.returncode or 0, _decode_cli_output(stdout), _decode_cli_output(stderr)


class ClaudeCodeBackend(Backend):
    name = "claude_code"
    executable = "claude"
    # ``acceptEdits`` preserves normal checks outside workspace edits; plan
    # permits inspection while blocking edits, the safest non-interactive
    # fallback for confirm and deny.
    permission_arg_sets = {
        "full": ("--dangerously-skip-permissions",),
        "auto": ("--permission-mode", "acceptEdits"),
        "restricted": ("--permission-mode", "plan"),
    }
    supports_detached_tasks = True
    # Claude Code has no safe non-interactive catalog command.  In
    # particular, a managed Bedrock/Vertex/Foundry login cannot be enumerated
    # through Anthropic's public API, and probing candidate IDs can make a
    # billable request.  Keep this curated fallback plus config.extra_models
    # until the CLI exposes an account-aware model-list interface.
    # Mirrors the model registry embedded in the Claude Code CLI. Aliases
    # resolve to the current default for each tier; ``default`` clears a pin
    # and lets Claude Code choose the account-tier default. Full IDs pin a
    # specific version. Mythos stays out of the picker: it ships only to
    # Project Glasswing participants. Users can still add gateway or local IDs
    # through config.extra_models.
    #
    # Three rules the CLI enforces, each of which this tuple has gotten wrong
    # before - check them before adding an entry:
    #   - A dated snapshot exists only where the API publishes one (Opus 4.5,
    #     Sonnet 4.5, Haiku 4.5). From Opus/Sonnet 4.6 on, the ID is undated
    #     and inventing a date suffix 404s.
    #   - ``[1m]`` is only valid on aliases/models whose registry exposes a
    #     long-context variant. Current 1M aliases and variants appear below,
    #     but Sonnet 4.5 remains a 200K model. Keep full
    #     Fable/Sonnet suffixes out of this curated picker unless the CLI
    #     exposes them as picker entries.
    #   - Fast mode is a session toggle (``/fast``) on Opus 5/4.8/4.7, not a
    #     model ID. The ``claude-opus-4-*-fast`` strings are retired API IDs:
    #     4.6-fast silently degrades to standard Opus 4.6, and 4.7-fast errors
    #     once removed.
    available_models = (
        "default",
        "sonnet",
        "opus",
        "fable",
        "haiku",
        "best",
        "opusplan",
        "fable[1m]",
        "sonnet[1m]",
        "opus[1m]",
        "opusplan[1m]",
        "claude-fable-5",
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-7[1m]",
        "claude-opus-5[1m]",
        "claude-opus-4-8[1m]",
        "claude-opus-4-6[1m]",
        "claude-sonnet-4-6[1m]",
    )
    default_model = "default"
    default_summary_model = "haiku"
    tier_models = {"low": "haiku", "mid": "sonnet", "high": "opus"}
    effort_levels = ("low", "medium", "high", "xhigh", "max")

    def effort_levels_for_model(self, model: str | None) -> tuple[str, ...]:
        """Return only the effort values accepted by a selected Claude model.

        Claude Code clamps unsupported values, but omitting the flag for
        models without adaptive reasoning avoids presenting a misleading
        workspace setting.  ``[1m]`` only changes context length, not effort
        support, so normalize it before looking up a pinned model.
        """
        selected = (model or self.default_model).strip().casefold()
        selected = selected.removesuffix("[1m]")
        if selected in _NO_EFFORT_MODELS:
            return ()
        if selected in _FOUR_LEVEL_EFFORT_MODELS:
            return _FOUR_LEVEL_EFFORTS
        # Keep existing gateway/private model behavior: unknown IDs receive
        # the current full scale rather than being silently downgraded.
        return self.effort_levels

    def context_window_tokens(self, model: str | None) -> int | None:
        """Return a verified capacity for one explicit Claude variant."""
        selected = (model or self.default_model).strip().casefold()
        if selected in _ONE_MILLION_CONTEXT_MODELS:
            return _LONG_CONTEXT_WINDOW_TOKENS
        return None

    # File-editing tools whose tool_use blocks we surface as kind="file"
    # ChatEvents (the rest of the tool name is kept as the action label).
    _FILE_TOOLS = frozenset({
        "Write", "Edit", "MultiEdit", "NotebookEdit",
    })

    async def launch(
        self,
        workspace_path: str,
        prompt: str,
        model: str | None,
        approval: str,
        *,
        compaction: bool = False,
        effort: int = 0,
    ) -> asyncio.subprocess.Process:
        prefix = executable_command(self.executable)
        cmd: list[str] = [
            *prefix,
            "--print",
            "--output-format", "stream-json",
            "--verbose",  # required by claude when stream-json is set
            "--no-session-persistence",  # we manage sessions ourselves
            # Claude's Bash tool is outside Cozter's process tree.  Install
            # a session-scoped PreToolUse hook so it cannot leave an
            # untracked ordinary background job behind.
            "--settings", _background_guard_settings(),
        ]
        self.append_launch_options(cmd, model, effort, approval)

        return await create_prompt_subprocess(cmd, prompt, cwd=workspace_path)

    async def launch_detached(
        self,
        workspace_path: str,
        prompt: str,
        model: str | None,
        approval: str,
        *,
        effort: int = 0,
    ) -> str:
        """Start a whole Claude Code session through ``claude --bg``.

        Background sessions are intentionally *not* print/stream-json runs:
        Claude Code rejects that combination because its supervisor needs a
        persistent interactive-session transcript to host the detached worker.
        """
        cmd: list[str] = [
            *executable_command(self.executable),
            "--bg",
            "--settings", _background_guard_settings(),
        ]
        self.append_launch_options(cmd, model, effort, approval)
        # ``--bg`` takes a positional prompt, not ``--print``/stdin.
        cmd.append(prompt)

        returncode, stdout, stderr = await _run_claude_command(
            cmd, cwd=workspace_path,
        )
        combined = "\n".join(part for part in (stdout, stderr) if part)
        if returncode != 0:
            raise RuntimeError(
                "Claude Code could not start a background session"
                + (f": {combined}" if combined else ""),
            )
        task_ids = _background_task_ids(combined)
        if len(task_ids) != 1:
            raise RuntimeError(
                "Claude Code started a background session but did not report "
                "one unambiguous task id",
            )
        return task_ids[0]

    async def get_detached_task_status(
        self,
        workspace_path: str,
        task_id: str,
    ) -> DetachedTaskStatus | None:
        """Inspect one Claude supervisor job through its JSON task list."""
        cmd = [
            *executable_command(self.executable),
            "agents", "--json", "--all", "--cwd", workspace_path,
        ]
        returncode, stdout, stderr = await _run_claude_command(
            cmd, cwd=workspace_path,
        )
        if returncode != 0:
            logger.warning(
                "Claude Code background task listing failed for %s: %s",
                task_id, stderr or stdout,
            )
            # A transient supervisor failure must not be mistaken for a task
            # disappearing. A completed task may have already retired from
            # the daemon, though, so consult its durable local state first.
            local = _local_background_status(workspace_path, task_id)
            if local is not None:
                return local
            return DetachedTaskStatus("unknown")
        try:
            sessions = json.loads(stdout)
        except json.JSONDecodeError:
            logger.warning(
                "Claude Code background task listing was not JSON: %s",
                stdout[:200],
            )
            return DetachedTaskStatus("unknown")
        if not isinstance(sessions, list):
            return DetachedTaskStatus("unknown")
        for item in sessions:
            if (
                not isinstance(item, dict)
                or item.get("kind") != "background"
                or item.get("id") != task_id
            ):
                continue
            if not _workspace_contains(workspace_path, item.get("cwd")):
                logger.warning(
                    "Claude Code task %s belongs to a different workspace",
                    task_id,
                )
                return None
            state = item.get("state")
            if not isinstance(state, str) or not state:
                state = "unknown"
            waiting_for = item.get("waitingFor")
            return DetachedTaskStatus(
                state=state,
                waiting_for=waiting_for if isinstance(waiting_for, str) else "",
            )
        # Claude's daemon can retire a completed worker before the next poll.
        # The persisted state keeps the callback restart-safe across that gap.
        return _local_background_status(workspace_path, task_id)

    async def get_detached_task_output(
        self,
        workspace_path: str,
        task_id: str,
    ) -> str:
        """Retrieve a detached task's visible result without terminal ANSI."""
        local = _local_background_output(workspace_path, task_id)
        if local:
            return local

        # Compatibility fallback for older Claude Code versions that do not
        # persist the current job/transcript layout. Newer versions render a
        # full terminal screen here, hence the durable transcript above.
        cmd = [*executable_command(self.executable), "logs", task_id]
        returncode, stdout, stderr = await _run_claude_command(
            cmd, cwd=workspace_path,
        )
        if returncode != 0:
            raise RuntimeError(
                "Claude Code could not read background task output"
                + (f": {stderr or stdout}" if stderr or stdout else ""),
            )
        return stdout

    async def stop_detached_task(
        self,
        workspace_path: str,
        task_id: str,
    ) -> bool:
        """Ask Claude's supervisor to stop a detached session."""
        cmd = [*executable_command(self.executable), "stop", task_id]
        returncode, stdout, stderr = await _run_claude_command(
            cmd, cwd=workspace_path,
        )
        if returncode != 0:
            logger.warning(
                "Claude Code could not stop background task %s: %s",
                task_id, stderr or stdout,
            )
            return False
        return True

    def parse_event(self, event: dict, result: AgentResult) -> None:
        etype = event.get("type", "")

        if etype == "assistant":
            msg = event.get("message", {}) or {}
            if not isinstance(msg, dict):
                return
            content = msg.get("content")
            if isinstance(content, str):
                if content:
                    append_text_result(result, content)
                return
            if not isinstance(content, list):
                return
            for block in content:
                self._handle_assistant_block(block, result)
            return

        if etype == "result":
            # The terminal event. If the assistant streamed text blocks
            # above, we already captured them; otherwise fall back to
            # the cumulative 'result' field.
            usage = event.get("usage")
            if isinstance(usage, dict):
                result.usage = dict(usage)
                cost = event.get("total_cost_usd")
                if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                    result.usage["total_cost_usd"] = cost
            if event.get("is_error"):
                err = (
                    event.get("error")
                    or event.get("result")
                    or "Unknown error"
                )
                set_error_result(result, err)
                return
            text = event.get("result", "")
            if (
                isinstance(text, str)
                and text
                and not any(e.kind == "text" for e in result.events)
            ):
                append_text_result(result, text)
            return

        if etype == "user":
            # Tool results normally stay out of the status display. The one
            # exception is a paired Bash result from ``claude --bg``: it
            # contains the short supervisor task id that Cozter can later
            # validate and monitor after this foreground stream exits.
            self._handle_user_tool_results(event, result)
            return

        # System/init events are noisy for the status display and don't
        # contribute new info; skip.
        if etype == "system":
            return

        if record_error_event(event, result):
            return

        logger.debug(
            "Claude Code: unhandled event type=%r keys=%r",
            etype, list(event.keys()),
        )

    def extract_agent_text(self, event: dict) -> str | None:
        # Compaction prefers the terminal result.result field since it's
        # the aggregated, fully-rendered final reply. Streaming assistant
        # text blocks are partials and may not include the full answer.
        etype = event.get("type", "")
        if etype == "result" and not event.get("is_error"):
            text = event.get("result", "")
            return text if isinstance(text, str) and text else None
        if etype == "assistant":
            msg = event.get("message", {}) or {}
            if not isinstance(msg, dict):
                return None
            content = msg.get("content")
            if isinstance(content, str):
                return content or None
            if not isinstance(content, list):
                return None
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if isinstance(text, str) and text:
                        return text
        return None

    # -- helpers ----------------------------------------------------------

    def _handle_assistant_block(
        self, block: object, result: AgentResult,
    ) -> None:
        # Anthropic allows ``content`` to be either a list of typed blocks
        # or, for plain-text messages, a bare string. Iterating yields
        # dicts normally, but a non-dict entry must not raise here and
        # crash the turn.
        if not isinstance(block, dict):
            return
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if isinstance(text, str) and text:
                append_text_result(result, text)
            return
        if btype == "tool_use":
            self._emit_tool_event(block, result)

    def _handle_user_tool_results(
        self, event: dict, result: AgentResult,
    ) -> None:
        """Pick up a background-id only from its matching Bash result."""
        message = event.get("message") or {}
        if not isinstance(message, dict):
            return
        blocks = message.get("content") or []
        if not isinstance(blocks, list):
            return
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            if (
                not isinstance(tool_use_id, str)
                or tool_use_id not in result.detached_task_tool_use_ids
            ):
                continue
            for task_id in _background_task_ids(
                self._tool_result_text(block.get("content")),
            ):
                append_detached_task(result, self.name, task_id)

    @staticmethod
    def _tool_result_text(content: object) -> str:
        """Flatten Anthropic's string/list shaped Bash tool-result content."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    @staticmethod
    def _emit_tool_event(block: dict, result: AgentResult) -> None:
        tool = block.get("name", "?")
        inp = block.get("input") or {}
        if not isinstance(inp, dict):
            inp = {}

        # File-editing tools: emit kind="file" with the touched path so
        # the bot's "Thinking..." status renders them under the file UX.
        if tool in ClaudeCodeBackend._FILE_TOOLS:
            path = inp.get("file_path") or inp.get("notebook_path") or "?"
            action = "write" if tool == "Write" else "edit"
            result.events.append(ChatEvent(
                kind="file",
                content=f"📄 {action}: {os.path.basename(path)}",
            ))
            return

        # Bash gets the command itself; other tools get just their name.
        if tool == "Bash":
            cmd = inp.get("command") or "?"
            tool_use_id = block.get("id")
            if (
                isinstance(cmd, str)
                and isinstance(tool_use_id, str)
                and _BACKGROUND_BASH_RE.search(cmd)
            ):
                result.detached_task_tool_use_ids.add(tool_use_id)
            result.events.append(ChatEvent(
                kind="tool",
                content=f"$ {truncate_status_text(cmd)}",
            ))
            return

        # Best-effort summary for generic tools.
        result.events.append(ChatEvent(kind="tool", content=tool))
