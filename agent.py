"""Agent runtime — runs a single user turn: routes to a session,
prepends colony + session context to the prompt, invokes the backend
CLI, streams events, logs the turn, and triggers compaction and
auto-titling background tasks.
"""

import asyncio
import logging
import os
import re
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from . import (
    agent_tools, backends_agent, colony, compaction, flexible, router,
    session, titling,
)
from . import workspace as workspace_mod
from .backends_agent.base import (
    NO_RESPONSE_TEXT, AgentResult, ChatEvent, append_detached_task_request,
    append_text_result, set_error_result,
)
from .utils import (
    COZTER_DIR,
    await_cancelled,
    cleanup_backend_process,
    create_background_task,
    drain_text_stream,
    iter_json_events,
    kill_and_wait,
    run_internal_backend,
    terminate_process_group,
)
from .utils import drain_queue as _drain_queue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetachedTaskLaunch:
    """A restart-safe provider task started for one Cozter session."""
    backend_name: str
    task_id: str
    session_id: str

# Shared preamble prepended to every backend's prompt. It documents the
# out-of-band markers Cozter understands and sets the agent's working
# disposition. Because it rides on top of whatever the underlying CLI
# does, it is the one lever that steers every backend (codex/copilot/
# claude_code/llama) the same way.
#
# Two variants: the collaboration policy (the Claude-Code-style
# disposition — ask a short question and pause via [[await]] rather than
# guessing) and an autonomous "decide and proceed" policy. Interactive
# turns pick between them via the workspace's interaction-style setting
# (see workspace.get_interaction_style); scheduled / ephemeral turns run
# unattended and cannot pause on [[await]], so they are always autonomous.
_ATTACH_HINT = (
    "To attach a file in your reply, include \"[[attach: PATH]]\" on its "
    "own line. PATH is relative to the workspace root, or absolute. If "
    "you create or generate an image or file for the user to view, make "
    "sure it is attached."
)

_COLLABORATION_POLICY = (
    "Work collaboratively — this is a live conversation, not an "
    "unattended batch job. When the request is ambiguous, underspecified, "
    "or open to more than one reasonable interpretation, or before any "
    "large-scope, destructive, or hard-to-reverse action, ask the user "
    "one short, specific question instead of guessing, and end that reply "
    "with \"[[await]]\". The bot will pause normal queued chat work until "
    "the user's next message, which you should treat as the answer. "
    "Scheduled /reserve tasks are autonomous and may still run while the "
    "chat queue is paused. For small, reversible, clearly-scoped choices, "
    "pick a sensible option, state it in one line, and keep going; don't "
    "ask about things that wouldn't change what the user does next."
)

_AUTONOMY_POLICY = (
    "You are running unattended on a scheduled task — nobody is watching "
    "in real time to answer questions mid-run. Do not ask for "
    "confirmation; make reasonable, well-scoped decisions and finish the "
    "task. If you are genuinely blocked, state plainly in your reply what "
    "is missing."
)


_DETACHED_TASK_POLICY = (
    "For durable background work, do not start a shell background process "
    "(`&`, `nohup`, Bash run_in_background, or a nested `claude --bg`) and "
    "do not promise a later callback yourself. Instead, finish your visible "
    "reply and place one self-contained task request on its own line as "
    "`[[background: <task>]]`. Cozter will launch, track, and later post the "
    "provider task's final result. Use this only when the remaining work can "
    "continue without user input."
)


def _capability_hint(
    collaborative: bool, *, allow_detached_requests: bool,
) -> str:
    """Build the per-turn preamble (collaboration vs autonomy policy)."""
    policy = _COLLABORATION_POLICY if collaborative else _AUTONOMY_POLICY
    parts = [_ATTACH_HINT, policy]
    if allow_detached_requests:
        parts.append(_DETACHED_TASK_POLICY)
    return "[System: " + "\n\n".join(parts) + "]"


def _is_collaborative_turn(
    workspace_path: str, *, explicit_session: bool,
) -> bool:
    """Return whether this turn may pause for a live user answer."""
    return (
        not explicit_session
        and workspace_mod.get_interaction_style(workspace_path)
        == "collaborative"
    )


MAX_HISTORY_CHARS = 50_000

_ATTACH_RE = re.compile(
    r"\[\[attach:\s*([^\]\n]+?)\s*\]\]", re.IGNORECASE,
)
_AWAIT_RE = re.compile(r"\[\[await\]\]", re.IGNORECASE)
_DETACHED_TASK_REQUEST_RE = re.compile(
    r"(?ims)^[ \t]*\[\[background:\s*(?P<prompt>.*?)\s*\]\]"
    r"[ \t]*(?:\r?\n|$)",
)
_MAX_DETACHED_TASK_REQUESTS = 3
_MAX_DETACHED_TASK_PROMPT_CHARS = 12_000

_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
}

_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
)
_ATTACHMENT_SCAN_SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
}
_EXTERNAL_ATTACHMENT_ROOTS_ENV = "COZTER_ATTACHMENT_ROOTS"


def _path_inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def _workspace_candidate_path(path: str, workspace_path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(workspace_path, path)


def extract_attachment_sources(text: str, ws: str) -> tuple[str, list[str]]:
    """Parse attachment markers without copying external generated images."""
    paths: list[str] = []
    seen_sources: set[str] = set()

    def _sub(m: re.Match) -> str:
        rel = m.group(1).strip()
        if not rel:
            return ""
        source = _resolve_attachment_source(rel, ws)
        if source is None:
            return ""
        source_path, _ = source
        if source_path in seen_sources:
            return ""
        seen_sources.add(source_path)
        paths.append(source_path)
        return ""

    cleaned = _ATTACH_RE.sub(_sub, text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, paths


def _explicit_attachment_sources(
    events: list[ChatEvent], workspace_path: str,
) -> set[str]:
    """Return source paths already referenced by agent output."""
    sources: set[str] = set()
    for ev in events:
        if ev.kind == "attachment":
            source_path = attachment_source_path(ev.content, workspace_path)
            if source_path is not None:
                sources.add(source_path)
        elif ev.kind == "text":
            _, source_paths = extract_attachment_sources(
                ev.content, workspace_path,
            )
            sources.update(source_paths)
    return sources


def extract_await(text: str) -> tuple[str, bool]:
    """Detect and strip ``[[await]]`` markers from agent-emitted text.

    Pairs with :func:`_capability_hint` (whose interactive variant
    instructs the model to ask and end with the marker when it needs a
    decision before continuing).
    Returns ``(cleaned_text, awaiting)``.
    """
    if not _AWAIT_RE.search(text):
        return text, False
    cleaned = _AWAIT_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, True


def extract_detached_task_requests(text: str) -> tuple[str, list[str]]:
    """Strip agent ``[[background: ...]]`` markers and return their prompts.

    The marker is intentionally a final-response protocol rather than a CLI
    command the foreground agent launches itself. Cozter can therefore
    persist the provider task id before it acknowledges the request, and can
    still deliver completion after a restart.
    """
    prompts: list[str] = []

    def _take(match: re.Match) -> str:
        prompt = match.group("prompt").strip()
        if len(prompt) > _MAX_DETACHED_TASK_PROMPT_CHARS:
            logger.warning("Ignoring oversized detached task request")
        elif prompt and prompt not in prompts:
            if len(prompts) < _MAX_DETACHED_TASK_REQUESTS:
                prompts.append(prompt)
            else:
                logger.warning("Ignoring excess detached task request")
        return ""

    cleaned = _DETACHED_TASK_REQUEST_RE.sub(_take, text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, prompts


def _consume_detached_task_requests(result: AgentResult) -> None:
    """Move response markers into ``result`` before logging or delivery."""
    events: list[ChatEvent] = []
    for ev in result.events:
        if ev.kind != "text":
            events.append(ev)
            continue
        cleaned, prompts = extract_detached_task_requests(ev.content)
        for prompt in prompts:
            append_detached_task_request(result, prompt)
        if cleaned:
            ev.content = cleaned
            events.append(ev)
    result.events = events

    cleaned_text, prompts = extract_detached_task_requests(result.text)
    for prompt in prompts:
        append_detached_task_request(result, prompt)
    result.text = cleaned_text


def _compact_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def format_usage(usage: dict | None) -> str | None:
    """Compact one-line token/cost footer from a backend's usage dict.

    Returns None when there's nothing worth showing. Understands the
    fields codex (``turn.completed``) and claude_code (``result``) expose
    (``input_tokens`` / ``output_tokens`` / ``total_cost_usd``).
    """
    if not isinstance(usage, dict):
        return None
    parts: list[str] = []
    inp = usage.get("input_tokens")
    if isinstance(inp, int) and not isinstance(inp, bool):
        parts.append(f"{_compact_tokens(inp)} in")
    out = usage.get("output_tokens")
    if isinstance(out, int) and not isinstance(out, bool):
        parts.append(f"{_compact_tokens(out)} out")
    cost = usage.get("total_cost_usd")
    if (
        isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and cost > 0
    ):
        cost_str = f"{cost:.4f}".rstrip("0").rstrip(".")
        parts.append(f"${cost_str}")
    if not parts:
        return None
    return "📊 " + " · ".join(parts)


def _codex_generated_images_dir() -> str:
    codex_home = os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    return os.path.join(codex_home, "generated_images")


def _external_attachment_roots() -> list[str]:
    """Roots where trusted agents may save generated preview images."""
    roots = [_codex_generated_images_dir()]
    extra = os.environ.get(_EXTERNAL_ATTACHMENT_ROOTS_ENV, "")
    roots.extend(p for p in extra.split(os.pathsep) if p.strip())

    resolved: list[str] = []
    seen: set[str] = set()
    for root in roots:
        try:
            real = os.path.realpath(os.path.expanduser(root.strip()))
        except (ValueError, OSError):
            continue
        if real and real not in seen:
            seen.add(real)
            resolved.append(real)
    return resolved


def _image_extension(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return None
    for magic, ext in _IMAGE_MAGIC:
        if head.startswith(magic):
            return ext
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return ".webp"
    ext = os.path.splitext(path)[1].lower()
    return ext if ext in _IMAGE_EXTENSIONS else None


def _safe_generated_image_name(src: str, ext: str) -> str:
    parent = os.path.basename(os.path.dirname(src))
    stem = os.path.splitext(os.path.basename(src))[0]
    raw = f"{parent}-{stem}" if parent else stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip(".-")
    return f"{safe or 'generated-image'}{ext}"


def _unique_path(directory: str, filename: str) -> str:
    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    i = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem}-{i}{ext}")
        i += 1
    return candidate


def _copy_generated_image_into_workspace(
    src: str, workspace_path: str,
) -> str | None:
    ext = _image_extension(src)
    if ext is None:
        logger.warning("Ignoring generated non-image attachment: %s", src)
        return None

    try:
        src_real = os.path.realpath(src)
        ws_real = os.path.realpath(workspace_path)
    except OSError:
        return None

    if src_real == ws_real or src_real.startswith(ws_real + os.sep):
        return src_real

    dest_dir = os.path.join(workspace_path, COZTER_DIR, "generated_images")
    try:
        os.makedirs(dest_dir, exist_ok=True)
        dest = _unique_path(dest_dir, _safe_generated_image_name(src_real, ext))
        shutil.copy2(src_real, dest)
        return os.path.realpath(dest)
    except OSError:
        logger.warning(
            "Failed to copy generated image into workspace: %s", src,
            exc_info=True,
        )
        return None


def _resolve_attachment_source(
    path: str, workspace_path: str,
) -> tuple[str, bool] | None:
    """Return ``(real_path, needs_copy)`` for an attachable source file."""
    if not path:
        return None
    try:
        candidate = _workspace_candidate_path(path, workspace_path)
        real = os.path.realpath(candidate)
        ws_real = os.path.realpath(workspace_path)
    except (ValueError, OSError):
        return None
    if not os.path.isfile(real):
        return None
    if _path_inside(real, ws_real):
        return real, False
    if _image_extension(real) is None:
        return None
    for root in _external_attachment_roots():
        if _path_inside(real, root):
            return real, True
    return None


def prepare_attachment_path(path: str, workspace_path: str) -> str | None:
    """Resolve an agent attachment path into a sendable workspace file.

    Explicit workspace files are returned directly. Generated image files
    from trusted external artifact roots are copied into
    ``.cozter/generated_images`` first so chat platforms can upload them
    without gaining access to arbitrary files outside the workspace.
    """
    source = _resolve_attachment_source(path, workspace_path)
    if source is None:
        return None
    return _prepare_resolved_attachment(source, workspace_path)


def attachment_source_path(path: str, workspace_path: str) -> str | None:
    """Return the canonical source path for a valid attachment."""
    source = _resolve_attachment_source(path, workspace_path)
    if source is None:
        return None
    return source[0]


def _prepare_resolved_attachment(
    source: tuple[str, bool], workspace_path: str,
) -> str | None:
    """Turn a resolved attachment source into a sendable path."""
    real, needs_copy = source
    if not needs_copy:
        return real
    return _copy_generated_image_into_workspace(real, workspace_path)


def _iter_image_files(root: str, *, skip_dirs: bool) -> list[str]:
    paths: list[str] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            if skip_dirs:
                dirnames[:] = [
                    d for d in dirnames if d not in _ATTACHMENT_SCAN_SKIP_DIRS
                ]
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in _IMAGE_EXTENSIONS:
                    continue
                path = os.path.realpath(os.path.join(dirpath, filename))
                if os.path.isfile(path):
                    paths.append(path)
    except OSError:
        logger.warning("Failed to scan image artifacts under %s", root,
                       exc_info=True)
    return paths


def _snapshot_attachment_images(
    workspace_path: str,
) -> dict[str, tuple[int, int]]:
    """Return image artifact state for workspace and trusted external roots."""
    snapshot: dict[str, tuple[int, int]] = {}
    ws_real = os.path.realpath(workspace_path)
    roots = [ws_real]
    roots.extend(_external_attachment_roots())

    seen_roots: set[str] = set()
    for root in roots:
        if root in seen_roots or not os.path.isdir(root):
            continue
        seen_roots.add(root)
        skip_dirs = root == ws_real
        for path in _iter_image_files(root, skip_dirs=skip_dirs):
            try:
                st = os.stat(path)
            except OSError:
                continue
            snapshot[path] = (st.st_mtime_ns, st.st_size)
    return snapshot


def _collect_new_attachment_images(
    before: dict[str, tuple[int, int]],
    workspace_path: str,
    *,
    exclude_sources: set[str] | None = None,
) -> list[str]:
    """Return images created or modified during this run as attachments."""
    after = _snapshot_attachment_images(workspace_path)
    changed = [
        path for path, stamp in after.items()
        if before.get(path) != stamp
    ]
    ordered = sorted(changed, key=lambda p: (after[p][0], p))

    copied: list[str] = []
    excluded = exclude_sources or set()
    seen_sources: set[str] = set()
    seen_destinations: set[str] = set()
    for src in ordered:
        if src in excluded or src in seen_sources:
            continue
        seen_sources.add(src)
        dest = prepare_attachment_path(src, workspace_path)
        if not dest or dest in seen_destinations:
            continue
        seen_destinations.add(dest)
        copied.append(dest)
    return copied


# ------------------------------------------------------------------
# Contextual prompt building
# ------------------------------------------------------------------

def _build_contextual_prompt(
    prompt: str,
    session_data: dict | None,
    colony_items: list[str] | None = None,
    budget: int = MAX_HISTORY_CHARS,
) -> str:
    """Prepend colony + session history to the prompt for full context.

    Block order: [Colony] (workspace-shared) → [Long-term Memory]
    (session-scoped) → [Session Summary] → [Recent Messages] → user prompt.
    """
    data = session_data
    if data is None:
        data = {}
    summary: str | None = data.get("summary")
    long_term: list[str] = data.get("long_term") or []
    messages: list[dict] = data.get("messages", [])
    colony_list: list[str] = colony_items or []

    if not summary and not messages and not long_term and not colony_list:
        return prompt

    parts: list[str] = []

    if colony_list:
        parts.append("[Colony]")
        parts.extend(f"- {item}" for item in colony_list)
        parts.append("[End of Colony]\n")

    if long_term:
        parts.append("[Long-term Memory]")
        parts.extend(f"- {item}" for item in long_term)
        parts.append("[End of Long-term Memory]\n")

    if summary:
        parts.append("[Session Summary]")
        parts.append(summary)
        parts.append("[End of Session Summary]\n")

    if messages:
        parts.append("[Recent Messages]")
        parts.extend(session.format_msg_line(msg) for msg in messages)
        parts.append("[End of Recent Messages]\n")

    parts.append(
        "Continue the conversation. The user's new message follows.\n"
    )
    parts.append(prompt)

    full = "\n".join(parts)

    # Truncate if too long - drop oldest messages; colony, long-term and
    # summary are durable so they're preserved at the expense of recent msgs.
    if len(full) > budget:
        colony_block = ""
        if colony_list:
            colony_block = (
                "[Colony]\n"
                + "\n".join(f"- {item}" for item in colony_list)
                + "\n[End of Colony]\n"
            )
        lt_block = ""
        if long_term:
            lt_block = (
                "[Long-term Memory]\n"
                + "\n".join(f"- {item}" for item in long_term)
                + "\n[End of Long-term Memory]\n"
            )
        summary_block = (
            f"[Session Summary]\n{summary}\n[End of Session Summary]\n"
            if summary else ""
        )
        overhead = (
            len(prompt) + len(colony_block) + len(lt_block)
            + len(summary_block) + 500
        )
        msg_budget = max(0, budget - overhead)
        if msg_budget == 0 and messages:
            logger.warning(
                "History truncation: colony/long-term/summary fill budget; "
                "dropping all %d recent messages", len(messages),
            )

        history_parts: list[str] = []
        if colony_block:
            history_parts.append(colony_block)
        if lt_block:
            history_parts.append(lt_block)
        if summary_block:
            history_parts.append(summary_block)

        # Add messages newest-to-oldest until the budget is exhausted.
        # Content is already capped at session.MSG_CONTENT_MAX so budget arithmetic
        # is predictable.
        msg_lines = (
            session.take_recent_messages(messages, msg_budget)
            if msg_budget > 0 else []
        )

        if msg_lines:
            history_parts.append("[Recent Messages]")
            history_parts.extend(msg_lines)
            history_parts.append("[End of Recent Messages]\n")

        history_parts.append(
            "Continue the conversation. The user's new message follows.\n"
        )
        history_parts.append(prompt)
        full = "\n".join(history_parts)

    return full


# ------------------------------------------------------------------
# Backend execution
# ------------------------------------------------------------------

class BackendUnavailable(Exception):
    """A backend's CLI is not installed on this machine."""

    def __init__(self, backend) -> None:
        super().__init__(f"{backend.executable} CLI not found on PATH.")


def _take_pending_injections(
    inject_queue: asyncio.Queue[str] | None,
    injected: list[str] | None,
) -> bool:
    """Collect injections that landed while a phase was winding down.

    A backend watcher is cancelled during normal teardown.  A user message
    can arrive in that small handoff window, after the watcher stopped but
    before the next phase (or final return) takes over.  Treat it as an
    interrupt rather than silently dropping an already acknowledged
    ``/inject``.
    """
    pending: list[str] = []
    _drain_queue(inject_queue, collect=pending)
    if not pending:
        return False
    if injected is not None:
        injected.extend(pending)
    return True


def _close_inject_queue(inject_queue: asyncio.Queue[str] | None) -> None:
    """Stop a bot-owned injection channel from accepting late messages.

    The runtime also accepts ordinary :class:`asyncio.Queue` instances in
    tests and programmatic callers.  Its private queue subclass exposes an
    idempotent synchronous ``close`` method; duck-typing keeps that optional
    lifecycle hook out of the public agent API.
    """
    if inject_queue is None:
        return
    close = getattr(inject_queue, "close", None)
    if callable(close):
        close()


def _build_backend_prompt(
    backend,
    contextual_prompt: str,
    *,
    collaborative: bool,
    allow_detached_requests: bool = True,
) -> str:
    """Wrap a prompt in the preamble and plugin list *backend* needs.

    For backends that can't be handed typed tool definitions (CLI
    subprocess agents whose toolset is fixed by the CLI), user plugins are
    enumerated in the prompt so the model can invoke them via its own
    bash/shell tool. HTTP backends with typed tools see plugins via
    TOOL_SCHEMA. Chat-only HTTP backends opt out via
    supports_plugin_prelude=False, since they have no shell to invoke the
    prelude'd commands either.
    """
    parts = [_capability_hint(
        collaborative=collaborative,
        allow_detached_requests=(
            allow_detached_requests
            and bool(getattr(backend, "supports_detached_tasks", False))
        ),
    )]
    if not backend.supports_typed_plugins and backend.supports_plugin_prelude:
        prelude = agent_tools.cli_plugin_prelude()
        if prelude:
            parts.append(prelude)
    parts.append(contextual_prompt)
    return "\n\n".join(parts)


async def _drive_backend(
    backend,
    workspace_path: str,
    full_prompt: str,
    model: str | None,
    approval: str,
    *,
    effort: int,
    on_event: Callable[[ChatEvent], Awaitable[None]] | None = None,
    inject_queue: asyncio.Queue[str] | None = None,
    injected: list[str] | None = None,
    close_inject_on_completion: bool = False,
) -> tuple[AgentResult, bool]:
    """Launch *backend*, stream its events, and collect the result.

    Returns ``(result, restarting)``. *restarting* is True when a message
    arrived on *inject_queue* mid-run: the subprocess was killed, the
    message appended to *injected*, and the caller should rebuild the
    prompt and drive the backend again.

    Raises :exc:`BackendUnavailable` when the CLI isn't installed.
    """
    try:
        proc = await backend.launch(
            workspace_path, full_prompt, model, approval, effort=effort,
        )
    except FileNotFoundError as e:
        raise BackendUnavailable(backend) from e

    result = AgentResult()
    restarting = False
    stderr_task = asyncio.create_task(drain_text_stream(proc.stderr))

    def _log_non_json_line(line: str) -> None:
        logger.debug("Non-JSON line: %s", line)

    # Watch inject_queue - kill subprocess when a message arrives
    async def _watch_inject(
        active_proc: asyncio.subprocess.Process = proc,
    ) -> None:
        nonlocal restarting
        assert inject_queue is not None  # only scheduled when set
        msg = await inject_queue.get()
        if injected is not None:
            injected.append(msg)
        restarting = True
        # Kill the whole process group (not just the parent PID) so a CLI
        # backend's grandchildren - a build/test command it shelled out to -
        # don't keep running and mutating the workspace after the restart.
        terminate_process_group(active_proc)

    inject_task: asyncio.Task | None = None
    if inject_queue is not None:
        inject_task = asyncio.create_task(_watch_inject())

    assert proc.stdout is not None  # spawned with stdout=PIPE
    try:
        async for event in iter_json_events(
            proc.stdout, on_invalid=_log_non_json_line,
        ):
            prev_count = len(result.events)
            backend.parse_event(event, result)

            if on_event:
                for ev in result.events[prev_count:]:
                    await on_event(ev)

        await proc.wait()
    except asyncio.CancelledError:
        logger.info(
            "%s run cancelled, killing subprocess %d",
            backend.name, proc.pid,
        )
        raise
    finally:
        # Event parsing and chat-platform callbacks can fail just like
        # cancellation can. Never let any exceptional stream exit leave
        # the backend (or its stderr drain task) running in the
        # background.
        if proc.returncode is None:
            await kill_and_wait(proc)
        if inject_task and not inject_task.done():
            inject_task.cancel()
            await await_cancelled(inject_task)
        try:
            stderr = await stderr_task
        finally:
            await cleanup_backend_process(backend, proc, log=logger)
    if stderr:
        logger.debug("%s stderr: %s", backend.name, stderr)

    if restarting:
        return result, True

    # ``inject_task`` is cancelled as part of normal teardown.  Do one
    # last synchronous drain before handing off to another flexible phase
    # (or finalizing a direct turn), so an accepted message cannot land in
    # the watcher teardown gap and disappear.
    if _take_pending_injections(inject_queue, injected):
        return result, True
    if close_inject_on_completion:
        _close_inject_queue(inject_queue)

    # Keyed on the *text*, not on the events: a backend that streamed tool
    # calls and then died still owes the caller an answer, and leaving it
    # textless would hand the flexible merge step an empty worker report
    # with no hint of what went wrong.
    if proc.returncode != 0 and not result.text:
        msg = f"{backend.name} exited with code {proc.returncode}"
        if stderr:
            msg += f"\n{stderr}"
        set_error_result(result, msg, display_text=msg)

    return result, False


async def _run_with_inject_watch(
    coro: Awaitable[str | None],
    inject_queue: asyncio.Queue[str] | None,
    injected: list[str] | None,
    *,
    close_inject_on_completion: bool = False,
) -> tuple[str | None, bool]:
    """Run *coro* (an internal-backend call) but abandon it if a message
    arrives on *inject_queue* first.

    The flexible planner and merge steps call the summary backend directly
    rather than through :func:`_drive_backend`, so they don't watch the inject
    queue. Without this, an ``/inject`` sent during those windows (the merge in
    particular can run for minutes) would sit in the queue unread and then be
    discarded when the turn completes - even though ``cmd_inject`` already told
    the user "Injected." Racing the call against the queue lets the inject
    abandon the pipeline and replan, matching how the worker steps behave.

    Returns ``(text, restarting)``; *text* is None when *restarting* is True.
    When *inject_queue* is None the call runs unwatched (unchanged behavior).
    """
    call_task: asyncio.Task[str | None] = asyncio.ensure_future(coro)
    if inject_queue is None:
        return await call_task, False
    watch_task: asyncio.Task[str] = asyncio.ensure_future(inject_queue.get())
    try:
        await asyncio.wait(
            {call_task, watch_task}, return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError:
        call_task.cancel()
        watch_task.cancel()
        await await_cancelled(call_task)
        await await_cancelled(watch_task)
        raise
    if watch_task.done() and not watch_task.cancelled():
        # Inject arrived first (or in the same tick): abandon the call and
        # signal a restart so the new message is folded into a replanned turn.
        msg = watch_task.result()
        if injected is not None:
            injected.append(msg)
        call_task.cancel()
        await await_cancelled(call_task)
        return None, True
    # The call finished first; stop watching without consuming a message.
    watch_task.cancel()
    await await_cancelled(watch_task)
    # A message can arrive after the watcher is cancelled but before this
    # phase returns to its caller.  Collect it here and restart, instead of
    # letting an acknowledged /inject fall through a phase boundary.
    if _take_pending_injections(inject_queue, injected):
        # Retrieve the completed result so an exception is not left
        # unobserved; propagate it normally if the call itself failed.
        call_task.result()
        return None, True
    if close_inject_on_completion:
        # This is the terminal planner/merge phase.  Closing happens before
        # returning (and therefore before any reply/session I/O awaits), so
        # late /inject commands are rejected instead of being dropped.
        _close_inject_queue(inject_queue)
    return call_task.result(), False


# ------------------------------------------------------------------
# Flexible agent — plan, route by difficulty, merge
# ------------------------------------------------------------------

# Usage fields format_usage knows how to display; summed across the
# workers so a flexible turn reports one total rather than N partials.
_USAGE_TOTAL_FIELDS = ("input_tokens", "output_tokens", "total_cost_usd")


def _accumulate_usage(totals: dict, usage: dict | None) -> None:
    if not isinstance(usage, dict):
        return
    for field in _USAGE_TOTAL_FIELDS:
        value = usage.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            totals[field] = totals.get(field, 0) + value


def _split_attach_markers(text: str) -> tuple[str, list[str]]:
    """Pull ``[[attach: ...]]`` markers out of a worker's report.

    Workers' text never reaches the user - only the merged answer does -
    so their attachment markers have to be carried over by hand or the
    files they meant to send would be dropped along with the text.
    """
    markers = [m.group(0) for m in _ATTACH_RE.finditer(text)]
    if not markers:
        return text, []
    cleaned = _ATTACH_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, markers


async def _run_flexible(
    contextual_prompt: str,
    request: str,
    workspace_path: str,
    *,
    approval: str,
    effort: int,
    collaborative: bool,
    summary_backend_name: str,
    summary_model: str | None,
    on_event: Callable[[ChatEvent], Awaitable[None]] | None,
    inject_queue: asyncio.Queue[str] | None,
    injected: list[str],
) -> tuple[AgentResult, bool]:
    """Run one turn of the ``flexible`` meta-agent.

    Three phases: the summary agent understands the request and splits it
    into difficulty-graded sub-tasks; each sub-task runs as a full agent
    turn on the agent+model bound to its tier; the summary agent merges
    the workers' reports into the one reply the user sees.

    Returns ``(result, restarting)`` like :func:`_drive_backend` - an
    inject mid-pipeline abandons the run and replans with the new context.
    """
    summary_backend = backends_agent.get_backend(summary_backend_name)
    tiers = workspace_mod.get_flexible_run_config(workspace_path)

    async def status(text: str) -> None:
        if on_event:
            await on_event(ChatEvent(kind="tool", content=text))

    # 1. Understand the request and split it by difficulty.
    await status(
        f"flexible: planning with {summary_backend.name}/{summary_model}"
    )
    raw_plan, restarting = await _run_with_inject_watch(
        run_internal_backend(
            summary_backend,
            workspace_path,
            flexible.build_plan_prompt(
                contextual_prompt, collaborative=collaborative,
            ),
            summary_model,
            timeout=flexible.PLAN_TIMEOUT,
            label="Flexible planner",
            log=logger,
            missing_executable_message=(
                "%s CLI not found - flexible falling back to a single task"
            ),
            missing_level=logging.WARNING,
        ),
        inject_queue,
        injected,
    )
    if restarting:
        return AgentResult(), True
    plan = flexible.parse_plan(raw_plan or "", request)

    # The planner is the only step allowed to stop the turn and ask: the
    # workers run mid-pipeline, where nobody is reading their questions.
    if plan.question:
        if collaborative:
            # A question ends the Flexible turn without reaching the merge
            # phase.  It therefore owns the same terminal injection boundary
            # as a normal merge result.
            if _take_pending_injections(inject_queue, injected):
                return AgentResult(), True
            _close_inject_queue(inject_queue)
            result = AgentResult()
            append_text_result(result, f"{plan.question}\n\n[[await]]")
            return result, False
        plan = flexible.fallback_plan(request)

    logger.info(
        "Flexible plan: %s",
        ", ".join(f"[{t.tier}] {t.instruction[:60]}" for t in plan.subtasks),
    )

    # 2. Route each sub-task to the agent+model bound to its tier.
    result = AgentResult()
    reports: list[str] = []
    attach_markers: list[str] = []
    usage_totals: dict = {}
    blocked: list[int] = []
    total = len(plan.subtasks)

    for i, task in enumerate(plan.subtasks):
        tier_backend_name, tier_model = tiers[task.tier]
        tier_backend = backends_agent.get_backend(tier_backend_name)
        await status(
            f"flexible [{i + 1}/{total}] {task.tier} ·"
            f" {tier_backend_name}/{tier_model}: {task.instruction}"
        )
        sub_result, restarting = await _drive_backend(
            tier_backend,
            workspace_path,
            _build_backend_prompt(
                tier_backend,
                flexible.build_subtask_prompt(
                    contextual_prompt, plan, i, reports,
                ),
                collaborative=False,
                allow_detached_requests=False,
            ),
            tier_model,
            approval,
            effort=effort,
            on_event=on_event,
            inject_queue=inject_queue,
            injected=injected,
        )
        if restarting:
            return result, True

        # The workers' tool/file events are the visible trace of the turn
        # and stream through as usual. Their *text* is internal - it goes
        # to the merge step, not to the user - so it is kept out of the
        # events the bot renders as chat messages.
        result.events.extend(
            ev for ev in sub_result.events if ev.kind != "text"
        )
        _accumulate_usage(usage_totals, sub_result.usage)

        if sub_result.error:
            logger.warning(
                "Flexible sub-task %d/%d (%s/%s) failed: %s",
                i + 1, total, tier_backend_name, tier_model, sub_result.error,
            )

        report, worker_awaiting = extract_await(sub_result.text)
        report, markers = _split_attach_markers(report)
        attach_markers.extend(markers)
        report = report.strip()

        # A worker that says nothing at all has failed, whatever its exit
        # code claimed. Saying nothing is a legitimate outcome for a direct
        # turn, but never for a sub-task: an empty report tells the merge
        # step nothing, and if the merge is broken too - it usually is, since
        # a misconfigured agent tends to break every step at once - it leaves
        # the user with a blank answer and no idea which tier went quiet.
        # Name the tier instead, so it reaches the user either way.
        if not report:
            report = (
                f"(no output: {tier_backend_name}/{tier_model} ended the"
                " sub-task without a reply"
                + (f" - {sub_result.error}" if sub_result.error else "")
                + ")"
            )
        elif sub_result.error:
            # Half an answer and a failure. The merge step has to be told,
            # or it presents the half it got as the finished job.
            report += f"\n\n(the agent then failed: {sub_result.error})"
        reports.append(report)

        # Workers run under the autonomy policy, so one that asks anyway is
        # genuinely stuck. Remember that: the merge step below has to end
        # the turn on that question *and* pause the queue, or the user's
        # answer lands as an unrelated new turn.
        if worker_awaiting:
            blocked.append(i)

    # 3. Merge the reports into the single reply the user sees.
    await status(
        f"flexible: merging with {summary_backend.name}/{summary_model}"
    )
    merged, restarting = await _run_with_inject_watch(
        run_internal_backend(
            summary_backend,
            workspace_path,
            flexible.build_merge_prompt(
                contextual_prompt, plan, reports,
                collaborative=collaborative, blocked=blocked,
            ),
            summary_model,
            timeout=flexible.MERGE_TIMEOUT,
            label="Flexible merge",
            log=logger,
            missing_executable_message=(
                "%s CLI not found - flexible returning the raw worker reports"
            ),
            missing_level=logging.WARNING,
        ),
        inject_queue,
        injected,
        close_inject_on_completion=True,
    )
    if restarting:
        return AgentResult(), True
    final = (merged or "").strip() or flexible.merge_fallback(plan, reports)

    # The merge writes the reply the user reads, so it is the one step
    # downstream of the planner allowed to end the turn on a question and
    # pause the queue. Pull the marker off wherever the merge put it and
    # re-add it last, so the pause still happens when a worker blocked and
    # the merge relayed its question without one. An unattended turn has
    # nobody to answer, so it never pauses - a marker there would strand
    # the run.
    final, merge_awaiting = extract_await(final)
    final = final.strip()

    # Every worker came back empty *and* the merge model had nothing to
    # add. Report that as the failure it is - a bare placeholder here
    # reads as an answer and tells the user nothing about which tier
    # broke.
    if not final:
        logger.warning(
            "Flexible produced no text: %d worker report(s) all empty",
            len(reports),
        )
        final = (
            "The workers produced no output for this turn. Check the agent"
            " and model bound to each difficulty tier with"
            " /agent_flexible_low, _mid, and _high."
        )

    for marker in attach_markers:
        if marker not in final:
            final += f"\n\n{marker}"

    if collaborative and (merge_awaiting or blocked):
        final += "\n\n[[await]]"

    append_text_result(result, final)
    result.usage = usage_totals or None
    return result, False


# ------------------------------------------------------------------
# Main run function
# ------------------------------------------------------------------

async def run(
    prompt: str,
    workspace_path: str,
    user_id: int,
    model: str | None = None,
    summary_model: str | None = None,
    approval: str = "auto",
    on_event: Callable[[ChatEvent], Awaitable[None]] | None = None,
    inject_queue: asyncio.Queue[str] | None = None,
    backend_name: str | None = None,
    summary_backend_name: str | None = None,
    session_id: str | None = None,
) -> AgentResult:
    """Run a turn, serialized per workspace.

    Concurrent turns in the same workspace - a user message racing a
    scheduled ``/reserve`` run, or two allow-listed users - would otherwise
    interleave the agent's file edits. A dedicated per-workspace run lock
    (separate from the file lock used for session logging/compaction, so no
    reentrancy deadlock) forces turns to run one at a time. The lock is
    released as soon as the turn finishes; background titling runs after.
    """
    async with workspace_mod.get_run_lock(workspace_path):
        return await _run_turn(
            prompt,
            workspace_path,
            user_id,
            model=model,
            summary_model=summary_model,
            approval=approval,
            on_event=on_event,
            inject_queue=inject_queue,
            backend_name=backend_name,
            summary_backend_name=summary_backend_name,
            session_id=session_id,
        )


async def launch_detached(
    prompt: str,
    workspace_path: str,
    user_id: int | str,
    model: str | None = None,
    summary_model: str | None = None,
    approval: str = "auto",
    *,
    backend_name: str | None = None,
    summary_backend_name: str | None = None,
) -> DetachedTaskLaunch:
    """Start a provider-owned task whose result can arrive after this turn.

    Unlike :func:`run`, this does not keep a foreground CLI process open.
    The backend must expose a durable task id; the bot persists that id before
    acknowledging the launch and later polls it through the same adapter.
    """
    async with workspace_mod.get_run_lock(workspace_path):
        backend = backends_agent.get_backend(backend_name)
        if not getattr(backend, "supports_detached_tasks", False):
            raise RuntimeError(
                f"{backend.name} does not support restart-safe detached tasks.",
            )

        summary_backend = summary_backend_name or backend.name
        last_sid = session.get_last_session(workspace_path, user_id)
        last_data = (
            session.load_session(workspace_path, last_sid)
            if last_sid else None
        )
        if last_data is not None:
            session_id, session_data = last_sid, last_data
        else:
            session_id, session_data = await router.select_or_create_session(
                prompt, workspace_path, summary_model,
                backend_name=summary_backend,
            )
        assert isinstance(session_id, str) and session_id
        session.set_last_session(workspace_path, user_id, session_id)

        contextual_prompt = _build_contextual_prompt(
            prompt,
            session_data,
            colony.get_items(workspace_path),
            budget=workspace_mod.get_history_budget(workspace_path),
        )
        task_id = await backend.launch_detached(
            workspace_path,
            _build_backend_prompt(
                backend,
                contextual_prompt,
                collaborative=False,
                allow_detached_requests=False,
            ),
            model,
            approval,
            effort=workspace_mod.get_reasoning_effort(workspace_path),
        )

        started = AgentResult(session_id=session_id)
        append_text_result(
            started,
            f"Started detached {backend.name} task {task_id}. "
            "Cozter will deliver its completion here.",
        )
        async with workspace_mod.get_lock(workspace_path):
            _log_to_session(workspace_path, session_id, prompt, started)
        return DetachedTaskLaunch(
            backend_name=backend.name,
            task_id=task_id,
            session_id=session_id,
        )


async def log_detached_task_completion(
    workspace_path: str,
    session_id: str | None,
    text: str,
) -> None:
    """Append a detached task's delivered outcome to its original session."""
    if not session_id:
        return
    try:
        async with workspace_mod.get_lock(workspace_path):
            session.append_messages(workspace_path, session_id, [{
                "role": "assistant",
                "content": text,
            }])
    except Exception:
        logger.error("Failed to log detached task completion", exc_info=True)


async def _run_post_turn_maintenance(
    workspace_path: str,
    session_id: str,
    summary_model: str | None,
    summary_backend: str,
) -> None:
    """Run non-critical session maintenance after the user reply is ready.

    Compaction can require another full model call. Keeping it out of the
    foreground turn means chat platforms can deliver the completed answer
    immediately. Titling follows compaction so a title emitted by the
    compactor wins over the lightweight auto-title fallback.
    """
    await compaction.maybe_compact(
        workspace_path, session_id, summary_model,
        backend_name=summary_backend,
    )
    await titling.maybe_auto_title(
        workspace_path, session_id, summary_model,
        backend_name=summary_backend,
    )


async def _run_turn(
    prompt: str,
    workspace_path: str,
    user_id: int,
    model: str | None = None,
    summary_model: str | None = None,
    approval: str = "auto",
    on_event: Callable[[ChatEvent], Awaitable[None]] | None = None,
    inject_queue: asyncio.Queue[str] | None = None,
    backend_name: str | None = None,
    summary_backend_name: str | None = None,
    session_id: str | None = None,
) -> AgentResult:
    """Run the selected agent CLI with session history prepended.

    backend_name selects the CLI adapter (codex/copilot). When None, the
    default backend is used. The workspace's configured backend should be
    passed in by the caller.

    session_id pins the run to a specific session (used for ephemeral
    schedule sessions). When None, the prompt is routed to the
    best-matching existing session via ``select_or_create_session`` —
    or a new session is created when no session is a good fit.

    on_event  - called for each parsed event as it arrives (streaming).
    inject_queue - when a message is put, the running subprocess is killed
                   and restarted with the injected context appended.
    """
    backend = backends_agent.get_backend(backend_name)
    is_flexible = backend.name == flexible.BACKEND_NAME

    # The session router, compaction, auto-titling, and (on a flexible
    # turn) the planner and merge steps all run on the summary backend -
    # which may differ from the chat backend, and is the backend the
    # caller's summary_model was resolved against. Flexible is a
    # meta-agent with no CLI of its own, so it can never fill that role:
    # fall back to a real backend rather than recursing into itself.
    summary_backend = summary_backend_name or (
        backends_agent.DEFAULT_DIRECT_BACKEND if is_flexible else backend.name
    )

    # Track whether the caller pinned a specific session: when True
    # (ephemeral schedule runs), we do NOT update the user's
    # last_session - that would clobber whatever they were actually
    # working on with a throwaway scheduler session.
    explicit_session = session_id is not None

    # session_data is reused on every inject restart so the session file
    # is not re-read for each iteration of the restart loop.
    if explicit_session:
        assert session_id is not None  # explicit_session == (session_id set)
        session_data = session.load_session(workspace_path, session_id)
        if session_data is None:
            # The pinned session was deleted out from under us; bail
            # rather than silently writing into a fresh one.
            result = AgentResult()
            set_error_result(
                result,
                f"session {session_id} not found in {workspace_path}.",
            )
            return result
    else:
        # Resume whatever session the user was last writing into.
        # Falls back to the router only when there's no last_session
        # pointer (first turn in this workspace, or /newsession reset
        # it) or the pointed-to session has been deleted.
        last_sid = session.get_last_session(workspace_path, user_id)
        last_data = (
            session.load_session(workspace_path, last_sid)
            if last_sid else None
        )
        if last_data is not None:
            session_id, session_data = last_sid, last_data
        else:
            session_id, session_data = await router.select_or_create_session(
                prompt, workspace_path, summary_model,
                backend_name=summary_backend,
            )

    # session_id is set by both resolution branches by this point.
    assert session_id is not None
    if not explicit_session:
        # Persist for the next turn - including the next bot restart.
        session.set_last_session(workspace_path, user_id, session_id)

    # Workspace-shared memory is loaded once and reused on every inject
    # restart, just like session_data.
    colony_items = colony.get_items(workspace_path)

    # Interactive turns honor the workspace's interaction style; scheduled/
    # ephemeral turns (explicit_session) can't pause on [[await]], so they
    # always run under the autonomous policy. Resolved once and reused
    # across inject restarts.
    collaborative = _is_collaborative_turn(
        workspace_path, explicit_session=explicit_session,
    )

    # Character budget for the prepended context block; configurable per
    # workspace so large-context models can keep more history.
    history_budget = workspace_mod.get_history_budget(workspace_path)

    injected: list[str] = []
    effort = workspace_mod.get_reasoning_effort(workspace_path)

    while True:  # restart loop for inject
        effective_prompt = prompt
        if injected:
            effective_prompt += (
                "\n\n[Additional context from user while you were thinking]:\n"
                + "\n".join(injected)
            )

        contextual_prompt = _build_contextual_prompt(
            effective_prompt, session_data, colony_items,
            budget=history_budget,
        )

        attachment_images_before = _snapshot_attachment_images(workspace_path)

        try:
            if is_flexible:
                result, restarting = await _run_flexible(
                    contextual_prompt, effective_prompt, workspace_path,
                    approval=approval,
                    effort=effort,
                    collaborative=collaborative,
                    summary_backend_name=summary_backend,
                    summary_model=summary_model,
                    on_event=on_event,
                    inject_queue=inject_queue,
                    injected=injected,
                )
            else:
                logger.info(
                    "Running %s (prompt %d chars, context %d chars)",
                    backend.name, len(prompt), len(contextual_prompt),
                )
                result, restarting = await _drive_backend(
                    backend, workspace_path,
                    _build_backend_prompt(
                        backend, contextual_prompt,
                        collaborative=collaborative,
                        allow_detached_requests=not explicit_session,
                    ),
                    model, approval,
                    effort=effort,
                    on_event=on_event,
                    inject_queue=inject_queue,
                    injected=injected,
                    close_inject_on_completion=True,
                )
        except BackendUnavailable as e:
            result = AgentResult()
            set_error_result(result, str(e))
            result.session_id = session_id
            _close_inject_queue(inject_queue)
            _drain_queue(inject_queue)
            return result

        # If we're restarting due to inject, drain pipes and any extra
        # injects that arrived while we were shutting down.
        if restarting:
            _drain_queue(inject_queue, collect=injected)
            logger.info(
                "Restarting %s with %d injected message(s)",
                backend.name, len(injected),
            )
            if on_event:
                await on_event(ChatEvent(
                    kind="tool",
                    content="Restarting with injected context...",
                ))
            continue  # restart loop

        explicit_attachment_sources = _explicit_attachment_sources(
            result.events, workspace_path,
        )
        for path in _collect_new_attachment_images(
            attachment_images_before, workspace_path,
            exclude_sources=explicit_attachment_sources,
        ):
            result.events.append(ChatEvent(kind="attachment", content=path))

        break  # normal completion

    # The terminal backend phase normally closes this first.  Keep a
    # defensive close here for direct callers and queues that exited through
    # an unusual but non-error path, before session/reply work yields.
    _close_inject_queue(inject_queue)
    # Discard any messages from a plain programmatic Queue after the final
    # answer. Bot-owned queues reject them before this point.
    _drain_queue(inject_queue)

    # A supported foreground agent may request that Cozter own the remaining
    # work as a durable provider task. Consume the control marker before its
    # text reaches session history or the chat platform.
    _consume_detached_task_requests(result)

    # Log the original prompt (including injected context) to session.
    async with workspace_mod.get_lock(workspace_path):
        _log_to_session(workspace_path, session_id, effective_prompt, result)

    # Compaction may take another model round-trip. The answer is already
    # complete, so keep it out of the foreground turn; Slack can post the
    # result as soon as ``agent.run`` returns. The helper preserves the old
    # compaction-before-title ordering.
    create_background_task(
        _run_post_turn_maintenance(
            workspace_path, session_id, summary_model, summary_backend,
        ),
        name=f"post-turn-maintenance:{session_id}",
        log=logger,
    )

    if (
        not any(e.kind == "text" for e in result.events)
        and not any(e.kind == "attachment" for e in result.events)
        and not result.detached_task_requests
    ):
        append_text_result(result, result.text or NO_RESPONSE_TEXT)

    result.session_id = session_id
    return result


# ------------------------------------------------------------------
# Session logging
# ------------------------------------------------------------------

def _log_to_session(
    workspace_path: str, session_id: str, prompt: str, result: AgentResult,
) -> None:
    """Append the user prompt and AI response in a single read+write."""
    try:
        session.append_messages(workspace_path, session_id, [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": _format_session_response(result, workspace_path),
            },
        ])
    except Exception:
        logger.error("Failed to log session", exc_info=True)


def _format_session_response(
    result: AgentResult, workspace_path: str,
) -> str:
    """Return the assistant's final text reply for session logging.

    Tool and file events are intermediate 'thinking' — the text reply
    already summarizes what was done, and skipping them keeps the saved
    history (and the context fed to future turns) compact.

    ``[[await]]`` is stripped: it is a control marker the bot consumes,
    not something the assistant said. Logging it would replay it as
    conversation on every later turn — and into compaction summaries and
    auto-titles — teaching the model to emit it when nothing is blocked.
    """
    text_parts: list[str] = []
    for ev in result.events:
        if ev.kind != "text":
            continue
        cleaned, _ = extract_await(ev.content)
        cleaned = cleaned.strip()
        if cleaned:
            text_parts.append(cleaned)
    attachment_parts: list[str] = []
    ws_real = os.path.realpath(workspace_path)
    for ev in result.events:
        if ev.kind != "attachment":
            continue
        try:
            path = os.path.realpath(
                _workspace_candidate_path(ev.content, workspace_path),
            )
            if path == ws_real or path.startswith(ws_real + os.sep):
                path = os.path.relpath(path, workspace_path)
        except OSError:
            path = ev.content
        attachment_parts.append(f"[Attachment: {path}]")
    if text_parts or attachment_parts:
        return "\n\n".join([*text_parts, *attachment_parts])
    cleaned, _ = extract_await(result.text)
    return cleaned.strip() or result.text
