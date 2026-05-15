"""Agent runtime — runs a single user turn: routes to a session,
prepends colony + session context to the prompt, invokes the backend
CLI, streams events, logs the turn, and triggers compaction and
auto-titling background tasks.
"""

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable

from . import (
    agent_tools, backends_agent, colony, compaction, router, session,
    titling,
)
from . import workspace as workspace_mod
from .backends_agent.base import AgentResult, ChatEvent
from .utils import iter_stream_lines
from .utils import drain_queue as _drain_queue

logger = logging.getLogger(__name__)

CAPABILITY_HINT = (
    "[System: To attach a file in your reply, include "
    "\"[[attach: PATH]]\" on its own line. PATH is relative to the "
    "workspace root, or absolute. "
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


def extract_attachments(text: str, ws: str) -> tuple[str, list[str]]:
    """Parse ``[[attach: PATH]]`` markers from agent-emitted text.

    Pairs with :data:`CAPABILITY_HINT` (which instructs the model to
    use these markers). Returns ``(cleaned_text, [absolute_paths])``;
    only paths that resolve inside the workspace and exist as files
    are included.
    """
    ws_real = os.path.realpath(ws)
    paths: list[str] = []

    def _sub(m: re.Match) -> str:
        rel = m.group(1).strip()
        if not rel:
            return ""
        try:
            abs_path = rel if os.path.isabs(rel) else os.path.join(ws, rel)
            abs_path = os.path.realpath(abs_path)
            inside = (
                abs_path == ws_real
                or abs_path.startswith(ws_real + os.sep)
            )
            if inside and os.path.isfile(abs_path):
                paths.append(abs_path)
        except (ValueError, OSError):
            pass
        return ""

    cleaned = _ATTACH_RE.sub(_sub, text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, paths


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
            result.text = (
                f"Error: session {session_id} not found in {workspace_path}."
            )
            result.events.append(ChatEvent(kind="text", content=result.text))
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

        try:
            proc = await backend.launch(
                workspace_path, full_prompt, model, approval,
                effort=workspace_mod.get_reasoning_effort(workspace_path),
            )
        except FileNotFoundError:
            err = f"{backend.executable} CLI not found on PATH."
            result = AgentResult()
            result.error = err
            result.text = f"Error: {err}"
            result.events.append(ChatEvent(kind="text", content=result.text))
            return result

        result = AgentResult()
        restarting = False

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
            async for line in iter_stream_lines(proc.stdout):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON line: %s", line)
                    continue

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
        if restarting:
            try:
                await proc.stderr.read()
            except Exception:
                pass
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

        break  # normal completion

    # Discard any inject messages that arrived after the final answer.
    _drain_queue(inject_queue)

    stderr = (
        await proc.stderr.read()
    ).decode("utf-8", errors="replace").strip()
    if stderr:
        logger.debug("%s stderr: %s", backend.name, stderr)

    if proc.returncode != 0 and not result.events:
        msg = f"{backend.name} exited with code {proc.returncode}"
        if stderr:
            msg += f"\n{stderr}"
        result.error = msg
        result.text = msg
        result.events.append(ChatEvent(kind="text", content=result.text))

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

    if not any(e.kind == "text" for e in result.events):
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
            {"role": "assistant", "content": _format_session_response(result)},
        ])
    except Exception:
        logger.error("Failed to log session", exc_info=True)


def _format_session_response(result: AgentResult) -> str:
    """Return the assistant's final text reply for session logging.

    Tool and file events are intermediate 'thinking' — the text reply
    already summarizes what was done, and skipping them keeps the saved
    history (and the context fed to future turns) compact.
    """
    text_parts = [ev.content for ev in result.events if ev.kind == "text"]
    if text_parts:
        return "\n\n".join(text_parts)
    return result.text


