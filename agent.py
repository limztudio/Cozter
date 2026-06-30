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

from . import (
    agent_tools, backends_agent, colony, compaction, router, session,
    titling,
)
from . import workspace as workspace_mod
from .backends_agent.base import AgentResult, ChatEvent, set_error_result
from .utils import drain_text_stream, iter_json_events
from .utils import drain_queue as _drain_queue

logger = logging.getLogger(__name__)

CAPABILITY_HINT = (
    "[System: To attach a file in your reply, include "
    "\"[[attach: PATH]]\" on its own line. PATH is relative to the "
    "workspace root, or absolute. "
    "If you create or generate an image or file for the user to view, "
    "make sure it is attached. "
    "If you need a decision from the user before you can continue, "
    "ask the question in your reply and end with \"[[await]]\". The "
    "bot will pause — including any queued messages or scheduled "
    "commands — until the user's next message, which you should "
    "treat as the answer.]"
)

MAX_HISTORY_CHARS = 50_000

_ATTACH_RE = re.compile(
    r"\[\[attach:\s*([^\]\n]+?)\s*\]\]", re.IGNORECASE,
)
_AWAIT_RE = re.compile(r"\[\[await\]\]", re.IGNORECASE)

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

    Pairs with :data:`CAPABILITY_HINT` (which instructs the model to
    use the marker when it needs a decision before continuing).
    Returns ``(cleaned_text, awaiting)``.
    """
    if not _AWAIT_RE.search(text):
        return text, False
    cleaned = _AWAIT_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, True


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

    dest_dir = os.path.join(workspace_path, ".cozter", "generated_images")
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
        for item in colony_list:
            parts.append(f"- {item}")
        parts.append("[End of Colony]\n")

    if long_term:
        parts.append("[Long-term Memory]")
        for item in long_term:
            parts.append(f"- {item}")
        parts.append("[End of Long-term Memory]\n")

    if summary:
        parts.append("[Session Summary]")
        parts.append(summary)
        parts.append("[End of Session Summary]\n")

    if messages:
        parts.append("[Recent Messages]")
        for msg in messages:
            parts.append(session.format_msg_line(msg))
        parts.append("[End of Recent Messages]\n")

    parts.append(
        "Continue the conversation. The user's new message follows.\n"
    )
    parts.append(prompt)

    full = "\n".join(parts)

    # Truncate if too long - drop oldest messages; colony, long-term and
    # summary are durable so they're preserved at the expense of recent msgs.
    if len(full) > MAX_HISTORY_CHARS:
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
        msg_budget = max(0, MAX_HISTORY_CHARS - overhead)
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

    # Track whether the caller pinned a specific session: when True
    # (ephemeral schedule runs), we do NOT update the user's
    # last_session - that would clobber whatever they were actually
    # working on with a throwaway scheduler session.
    explicit_session = session_id is not None

    # session_data is reused on every inject restart so the session file
    # is not re-read for each iteration of the restart loop.
    if explicit_session:
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
                backend_name=backend.name,
            )

    if not explicit_session:
        # Persist for the next turn - including the next bot restart.
        session.set_last_session(workspace_path, user_id, session_id)

    # Workspace-shared memory is loaded once and reused on every inject
    # restart, just like session_data.
    colony_items = colony.get_items(workspace_path)

    injected: list[str] = []

    while True:  # restart loop for inject
        effective_prompt = prompt
        if injected:
            effective_prompt += (
                "\n\n[Additional context from user while you were thinking]:\n"
                + "\n".join(injected)
            )

        contextual_prompt = _build_contextual_prompt(
            effective_prompt, session_data, colony_items,
        )
        parts = [CAPABILITY_HINT]
        # For backends that can't be handed typed tool definitions
        # (CLI subprocess agents whose toolset is fixed by the CLI),
        # enumerate user plugins in the prompt so the model can invoke
        # them via its own bash/shell tool. HTTP backends with typed
        # tools see plugins via TOOL_SCHEMA. Chat-only HTTP backends
        # opt out via supports_plugin_prelude=False, since they have
        # no shell to invoke the prelude'd commands either.
        if (
            not backend.supports_typed_plugins
            and backend.supports_plugin_prelude
        ):
            prelude = agent_tools.cli_plugin_prelude()
            if prelude:
                parts.append(prelude)
        parts.append(contextual_prompt)
        full_prompt = "\n\n".join(parts)
        logger.info(
            "Running %s (prompt %d chars, context %d chars)",
            backend.name, len(prompt), len(contextual_prompt),
        )

        attachment_images_before = _snapshot_attachment_images(workspace_path)

        try:
            proc = await backend.launch(
                workspace_path, full_prompt, model, approval,
                effort=workspace_mod.get_reasoning_effort(workspace_path),
            )
        except FileNotFoundError:
            err = f"{backend.executable} CLI not found on PATH."
            result = AgentResult()
            set_error_result(result, err)
            return result

        result = AgentResult()
        restarting = False
        stderr_task = asyncio.create_task(drain_text_stream(proc.stderr))

        def _log_non_json_line(line: str) -> None:
            logger.debug("Non-JSON line: %s", line)

        # Watch inject_queue - kill subprocess when a message arrives
        async def _watch_inject() -> None:
            nonlocal restarting
            msg = await inject_queue.get()
            injected.append(msg)
            restarting = True
            try:
                proc.kill()
            except OSError:
                # ProcessLookupError on Unix, other OSError on Windows
                # when TerminateProcess fails (e.g., already exited).
                pass

        inject_task: asyncio.Task | None = None
        if inject_queue is not None:
            inject_task = asyncio.create_task(_watch_inject())

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
            try:
                proc.kill()
                await proc.wait()
            except OSError:
                pass
            stderr = await stderr_task
            if stderr:
                logger.debug("%s stderr: %s", backend.name, stderr)
            raise
        finally:
            if inject_task and not inject_task.done():
                inject_task.cancel()
                try:
                    await inject_task
                except asyncio.CancelledError:
                    pass

        # If we're restarting due to inject, drain pipes and any extra
        # injects that arrived while we were shutting down.
        stderr = await stderr_task
        if stderr:
            logger.debug("%s stderr: %s", backend.name, stderr)
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

    # Discard any inject messages that arrived after the final answer.
    _drain_queue(inject_queue)

    if proc.returncode != 0 and not result.events:
        msg = f"{backend.name} exited with code {proc.returncode}"
        if stderr:
            msg += f"\n{stderr}"
        set_error_result(result, msg, display_text=msg)

    # Log the original prompt (including injected context) to session.
    async with workspace_mod.get_lock(workspace_path):
        _log_to_session(workspace_path, session_id, effective_prompt, result)

    # Compaction + titling intentionally use the summary backend, which
    # may differ from the chat backend (e.g. chat=llama, summary=codex).
    summary_backend = summary_backend_name or backend.name
    await compaction.maybe_compact(
        workspace_path, session_id, summary_model,
        backend_name=summary_backend,
    )

    # Auto-title sessions whose name still matches the default
    # "Session YYYY-MM-DD" pattern. The in-memory snapshot reflects
    # the name as it was at run start; a session with a custom name
    # is no longer a candidate for renaming, so skip the spawn entirely.
    # compaction above could have set a fresh title via [TITLE] —
    # in that case spawning is harmless (the task just bails on its
    # own is_default_name check after a fresh load).
    if session.is_default_name(session_data.get("name")):
        asyncio.create_task(titling.maybe_auto_title(
            workspace_path, session_id, summary_model,
            backend_name=summary_backend,
        ))

    if (
        not any(e.kind == "text" for e in result.events)
        and not any(e.kind == "attachment" for e in result.events)
    ):
        result.events.append(ChatEvent(kind="text", content=result.text))

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
    """
    text_parts = [ev.content for ev in result.events if ev.kind == "text"]
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
    return result.text
