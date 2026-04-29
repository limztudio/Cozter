"""Agent orchestrator - session management, context building, compaction.

The actual CLI invocation and event parsing are delegated to backend
adapters in the ``backends_agent`` package (currently codex and copilot).
The backend is chosen per workspace via ``workspace.get_backend_name``.
"""

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import AsyncIterator

from . import backends_agent, colony, session, workspace as workspace_mod
from .backends_agent.base import AgentResult, ChatEvent
from .utils import drain_queue as _drain_queue

logger = logging.getLogger(__name__)

CAPABILITY_HINT = (
    "[System: To attach a file in your reply, include "
    "\"[[attach: PATH]]\" on its own line. PATH is relative to the "
    "workspace root, or absolute.]"
)

MAX_HISTORY_CHARS = 50_000
# Cap each individual message's content when building context so a single
# long AI response cannot consume the entire message budget.
MSG_CONTENT_MAX = 800

# Per-workspace lock to prevent concurrent session file corruption
_workspace_locks: dict[str, asyncio.Lock] = {}


def get_workspace_lock(workspace: str) -> asyncio.Lock:
    if workspace not in _workspace_locks:
        _workspace_locks[workspace] = asyncio.Lock()
    return _workspace_locks[workspace]


KEEP_RECENT_AFTER_COMPACT = 5
MAX_SUMMARY_CHARS = 80_000  # ~20K tokens - safe for most models
COMPACT_TIMEOUT = 120  # seconds

SUMMARY_PROMPT = (
    "You are compacting a conversation into two memory layers: a SCRATCH "
    "summary (rewritten each compaction) and LONG-TERM memory (persistent "
    "facts that survive future compactions). You will also produce a short "
    "TITLE that names the session.\n\n"
    "=== SCRATCH SUMMARY ===\n"
    "Produce a concise abstract of the conversation below. This abstract "
    "REPLACES the raw history, so it must be thorough enough to continue "
    "work seamlessly.\n"
    "The reader will always see the [Long-term Memory] list alongside this "
    "summary, so do NOT repeat facts already captured there. Focus only on "
    "ephemeral context: recent work, current state, open commitments, "
    "in-progress decisions, and errors encountered.\n"
    "Capture: what was just done and why, file paths touched and the nature "
    "of each change, concrete tool results and errors, current "
    "done/in-progress/blocked state, and open commitments not yet in "
    "long-term memory.\n"
    "Drop: greetings, filler, repeated restatements, exploratory turns the "
    "user redirected, superseded tool output, and anything already in the "
    "long-term list.\n"
    "Prose paragraphs grouped by topic. Aim for 80-200 words. "
    "Prefer specificity (names, paths, values) over generic phrasing.\n\n"
    "=== LONG-TERM MEMORY ===\n"
    "Long-term items are durable facts: explicit preferences, architectural "
    "decisions, invariants, stable project facts, hard constraints. "
    "Each item is ONE self-contained sentence.\n"
    "Output the COMPLETE new list in [LONG_TERM] markers - this REPLACES "
    "the existing list entirely. Rules:\n"
    "- Carry forward items that are still true\n"
    "- Merge similar items into one sentence\n"
    "- Remove items that are now wrong or redundant with the summary\n"
    "- Add new durable facts from this conversation\n"
    "Keep the list SHORT: aim for 5-15 items, hard max 30. "
    "Fewer precise items beat many vague ones.\n\n"
    "=== TITLE ===\n"
    "A short descriptive name for this session: 3-7 words, Title Case, "
    "no trailing punctuation, no quotes. Pick the dominant topic of the "
    "conversation as a whole. Example: 'Refactor Schedule Storage'.\n\n"
    "=== OUTPUT FORMAT ===\n"
    "Emit all three sections. Omit a body if empty, but always include "
    "the markers.\n\n"
    "[SUMMARY]\n"
    "<scratch summary prose here>\n"
    "[/SUMMARY]\n\n"
    "[LONG_TERM]\n"
    "- <item 1>\n"
    "- <item 2>\n"
    "[/LONG_TERM]\n\n"
    "[TITLE]\n"
    "<short title>\n"
    "[/TITLE]"
)

# A standalone prompt used to title a session early (after the first
# turn) — before there's enough material for a full compaction.
TITLE_PROMPT = (
    "You are titling a chat session for a list view. Read the recent "
    "messages and produce a short descriptive name: 3-7 words, "
    "Title Case, no trailing punctuation, no quotes, no commentary. "
    "Pick the dominant topic, not the most recent line. Output only "
    "the title."
)
TITLE_TIMEOUT = 30  # seconds
TITLE_MAX_CHARS = 60
TITLE_CONTEXT_CHARS = 8_000


# ------------------------------------------------------------------
# Session router — picks the best-matching existing session for a new
# user message, or creates a new one when no session is a good fit.
# ------------------------------------------------------------------

ROUTER_PROMPT = (
    "You are a session router for a multi-session chat assistant.\n"
    "The user is about to send a new message. Pick the existing "
    "session whose ongoing topic best fits the message — or output "
    "NEW if the user has switched to a topic none of the sessions "
    "match.\n\n"
    "Rules:\n"
    "- Prefer to continue an existing session when there is a clear "
    "topical match.\n"
    "- Choose NEW for genuinely new topics, not minor digressions.\n"
    "- Output exactly one line: either the bare session id "
    "(no quotes, no commentary), or the literal word NEW.\n"
)
ROUTER_TIMEOUT = 30  # seconds
ROUTER_MAX_SESSIONS = 20  # cap input size; sessions are listed newest-first
ROUTER_PER_SESSION_CHARS = 600
ROUTER_PROMPT_PREVIEW_CHARS = 1_000

_ROUTER_LINE_RE = re.compile(r"^[A-Za-z0-9]+$")


def _build_router_prompt(prompt: str, sessions_meta: list[dict]) -> str:
    """Assemble the router prompt body. Caller prepends ROUTER_PROMPT."""
    parts: list[str] = ["User message:"]
    preview = prompt.strip()
    if len(preview) > ROUTER_PROMPT_PREVIEW_CHARS:
        preview = preview[:ROUTER_PROMPT_PREVIEW_CHARS] + "…"
    parts.append(preview)
    parts.append("")
    parts.append(f"Existing sessions ({len(sessions_meta)}, newest first):")
    parts.append("")
    for s in sessions_meta:
        block = [
            f"id: {s['id']}",
            f"name: {s['name']}",
        ]
        if s.get("summary"):
            sm = s["summary"]
            if len(sm) > ROUTER_PER_SESSION_CHARS:
                sm = sm[:ROUTER_PER_SESSION_CHARS] + "…"
            block.append(f"summary: {sm}")
        if s.get("long_term"):
            lt = "; ".join(s["long_term"][:5])
            block.append(f"long-term: {lt}")
        block.append("")
        parts.extend(block)
    parts.append(
        "Output exactly one line: a session id from the list above, or NEW."
    )
    return "\n".join(parts)


def _parse_router_output(raw: str, valid_ids: set[str]) -> str | None:
    """Return a session id, "NEW", or None if the output is unparseable.

    The model is instructed to output exactly one line, but in practice
    it sometimes adds a trailing period, code-fences, or explanatory
    text. We scan lines and take the first that's either "NEW" or a
    known session id.
    """
    for line in raw.splitlines():
        token = line.strip().strip("`'\"., ")
        if not token:
            continue
        if token.upper() == "NEW":
            return "NEW"
        if _ROUTER_LINE_RE.match(token) and token in valid_ids:
            return token
    return None


async def select_or_create_session(
    prompt: str,
    workspace_path: str,
    summary_model: str | None = None,
    *,
    backend_name: str | None = None,
) -> tuple[str, dict]:
    """Pick the session whose topic best matches *prompt*, else create one.

    Shortcuts an LLM call for the trivial cases (zero sessions →
    create new). Returns (session_id, loaded session data).
    """
    backend = backends_agent.get_backend(backend_name)

    metas = session.list_sessions(workspace_path)
    metas = metas[:ROUTER_MAX_SESSIONS]

    if not metas:
        data = session.create_session(workspace_path)
        logger.info(
            "Router: no existing sessions, created %s", data["id"],
        )
        return (data["id"], data)

    # Hydrate the lightweight metadata with each session's summary +
    # top long-term items so the router has topical signal.
    enriched: list[dict] = []
    for meta in metas:
        data = session.load_session(workspace_path, meta["id"])
        if data is None:
            continue
        enriched.append({
            "id": meta["id"],
            "name": meta.get("name") or meta["id"][:8],
            "summary": data.get("summary"),
            "long_term": data.get("long_term") or [],
        })
    if not enriched:
        data = session.create_session(workspace_path)
        return (data["id"], data)

    body = _build_router_prompt(prompt, enriched)
    full_prompt = f"{ROUTER_PROMPT}\n\n{body}"

    try:
        proc = await backend.launch(
            workspace_path, full_prompt, summary_model, approval="full",
            compaction=True,
        )
    except FileNotFoundError:
        logger.warning(
            "%s CLI not found - router falling back to NEW",
            backend.executable,
        )
        data = session.create_session(workspace_path)
        return (data["id"], data)

    raw = await _drain_internal_proc(proc, backend, ROUTER_TIMEOUT, "Router")

    valid_ids = {e["id"] for e in enriched}
    decision = _parse_router_output(raw, valid_ids) if raw else None
    if decision and decision != "NEW":
        loaded = session.load_session(workspace_path, decision)
        if loaded is not None:
            logger.info("Router: continuing session %s", decision)
            return (decision, loaded)

    if decision is None:
        logger.warning("Router output unparseable; defaulting to NEW: %r", raw)
    fresh = session.create_session(workspace_path)
    logger.info("Router: created new session %s", fresh["id"])
    return (fresh["id"], fresh)


# ------------------------------------------------------------------
# Colony (workspace-shared memory) — consolidates recurring long-term
# items across all sessions in a workspace into a single canonical list.
# ------------------------------------------------------------------

COLONY_PROMPT = (
    "You are consolidating a workspace's shared memory ('colony') from "
    "the long-term memory of every session in the workspace.\n\n"
    "Goal: maintain a canonical list of facts that recur across "
    "sessions, AND retire colony items that are no longer used by "
    "any session.\n\n"
    "=== INPUT ===\n"
    "You receive: the current colony list, then for each session, a "
    "'Session: <name>' header line followed by a [SESSION:<id>] "
    "block of that session's long-term items. The session name tells "
    "you the session's overall topic; use it together with the "
    "long-term items to judge whether each colony item is still "
    "relevant.\n\n"
    "=== TASK ===\n"
    "PROMOTE to colony:\n"
    "- An item that appears (verbatim or as a paraphrase) in 2+ "
    "sessions.\n"
    "- An existing colony item whose topic is still represented by "
    "at least one current session (its name or its long-term items).\n\n"
    "KEEP in a session's own list:\n"
    "- An item clearly specific to that one session.\n\n"
    "MERGE near-duplicates into one canonical sentence.\n\n"
    "PRUNE (drop entirely):\n"
    "- Colony items whose topic is no longer represented in any "
    "session — the colony exists for cross-session knowledge, so "
    "once a topic disappears from the workspace, retire the entry.\n"
    "- Items that are wrong or contradicted by current input.\n\n"
    "Each item must be ONE self-contained sentence.\n\n"
    "=== OUTPUT FORMAT ===\n"
    "Emit one [COLONY] block, then one [SESSION:<id>] block per "
    "input session (even if the new list is empty). Use the same "
    "<id> values you saw in the input. One bullet per item. Do NOT "
    "emit 'Session:' header lines in your output.\n\n"
    "[COLONY]\n"
    "- <colony item 1>\n"
    "- <colony item 2>\n"
    "[/COLONY]\n\n"
    "[SESSION:<session_id>]\n"
    "- <remaining session-specific item>\n"
    "[/SESSION]\n"
)
COLONY_TIMEOUT = 180  # seconds; consolidation can be heavier than compaction
COLONY_MAX_INPUT_CHARS = 100_000

_SESSION_BLOCK_RE = re.compile(
    r"\[SESSION:([^\]\s]+)\](.*?)\[/SESSION\]", re.DOTALL,
)


def _parse_colony_output(
    text: str,
) -> tuple[list[str] | None, dict[str, list[str]]]:
    """Extract (colony, {session_id: long_term}) from the consolidation output.

    colony is None when the [COLONY] markers are absent; sessions absent
    from the output are simply not in the dict (caller leaves them as-is).
    """
    def _bullets(block: str) -> list[str]:
        out: list[str] = []
        for raw in block.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("- ", "* ")):
                line = line[2:].strip()
            if line:
                out.append(line)
        return out

    colony_block: str | None = None
    open_tag, close_tag = "[COLONY]", "[/COLONY]"
    i = text.find(open_tag)
    if i != -1:
        j = text.find(close_tag, i + len(open_tag))
        if j != -1:
            colony_block = text[i + len(open_tag):j].strip()
    new_colony: list[str] | None = (
        _bullets(colony_block) if colony_block is not None else None
    )

    per_session: dict[str, list[str]] = {}
    for m in _SESSION_BLOCK_RE.finditer(text):
        sid = m.group(1).strip()
        body = m.group(2)
        per_session[sid] = _bullets(body)

    return new_colony, per_session


def _parse_compaction_output(
    text: str,
) -> tuple[str, list[str] | None, str | None]:
    """Extract (summary, long_term, title) from model output.

    long_term is None if [LONG_TERM] markers are absent (no rewrite).
    title is None if [TITLE] markers are absent or empty.
    Falls back to treating the entire text as the summary if [SUMMARY]
    markers are absent, so misbehaving models still produce a usable result.
    """
    def _extract(tag: str) -> str | None:
        open_tag = f"[{tag}]"
        close_tag = f"[/{tag}]"
        i = text.find(open_tag)
        if i == -1:
            return None
        j = text.find(close_tag, i + len(open_tag))
        if j == -1:
            return None
        return text[i + len(open_tag):j].strip()

    def _bullets(block: str | None) -> list[str]:
        if not block:
            return []
        items: list[str] = []
        for raw in block.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("- ", "* ")):
                line = line[2:].strip()
            if line:
                items.append(line)
        return items

    summary = _extract("SUMMARY")
    lt_block = _extract("LONG_TERM")
    long_term: list[str] | None = (
        _bullets(lt_block) if lt_block is not None else None
    )
    title_block = _extract("TITLE")
    title: str | None = None
    if title_block:
        title = _clean_title(title_block)
    if summary is None:
        # Fallback: treat full text as summary, stripping the other blocks.
        fallback = text
        for tag in ("LONG_TERM", "TITLE"):
            open_tag, close_tag = f"[{tag}]", f"[/{tag}]"
            i = fallback.find(open_tag)
            if i != -1:
                j = fallback.find(close_tag, i + len(open_tag))
                if j != -1:
                    fallback = (
                        fallback[:i] + fallback[j + len(close_tag):]
                    )
        summary = fallback.strip()
    return summary, long_term, title


def _clean_title(raw: str) -> str | None:
    """Trim a model-emitted title to a single short line."""
    stripped = raw.strip()
    if not stripped:
        return None
    line = stripped.splitlines()[0].strip(" \t.\"'`*_")
    if not line:
        return None
    if len(line) > TITLE_MAX_CHARS:
        line = line[:TITLE_MAX_CHARS].rstrip()
    return line or None


# ------------------------------------------------------------------
# Contextual prompt building
# ------------------------------------------------------------------

def _format_msg_line(msg: dict) -> str:
    """Format a session message as 'Role: content', capped at MSG_CONTENT_MAX."""
    role = msg.get("role", "?").capitalize()
    content = msg.get("content", "")
    if len(content) > MSG_CONTENT_MAX:
        content = content[:MSG_CONTENT_MAX] + "…"
    return f"{role}: {content}"


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
            parts.append(_format_msg_line(msg))
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
        # Content is already capped at MSG_CONTENT_MAX so budget arithmetic
        # is predictable.
        msg_lines: list[str] = []
        used = 0
        if msg_budget > 0:
            for msg in reversed(messages):
                line = _format_msg_line(msg)
                if used + len(line) > msg_budget:
                    break
                msg_lines.append(line)
                used += len(line) + 1  # +1 for the joining newline
            msg_lines.reverse()

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


async def _iter_stream_lines(
    stream: asyncio.StreamReader, chunk_size: int = 64 * 1024,
) -> AsyncIterator[str]:
    """Yield decoded stdout lines without StreamReader.readline() limits."""
    buffer = bytearray()

    while True:
        chunk = await stream.read(chunk_size)
        if not chunk:
            if buffer:
                yield buffer.decode("utf-8", errors="replace")
            return

        buffer.extend(chunk)
        parts = buffer.split(b"\n")
        buffer = bytearray(parts.pop())

        for part in parts:
            yield part.decode("utf-8", errors="replace")


async def _drain_internal_proc(
    proc: asyncio.subprocess.Process,
    backend,
    timeout: float,
    label: str,
) -> str:
    """Drain JSON event lines from an internal LLM subprocess and return
    the last agent text emitted, or an empty string on timeout/no output.

    The subprocess is *always* killed and reaped on exit — including on
    cancellation — so /stop or any other exception path can't leak a
    running subprocess past the cancelled task.
    """
    raw = ""
    try:
        async with asyncio.timeout(timeout):
            async for line in _iter_stream_lines(proc.stdout):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    if not raw:
                        raw = line  # bare-text fallback
                    continue
                text = backend.extract_agent_text(event)
                if text:
                    raw = text
            await proc.wait()
    except TimeoutError:
        logger.warning("%s timed out after %ds", label, timeout)
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except OSError:
                pass
    return raw


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

    # session_data is reused on every inject restart so the session file
    # is not re-read for each iteration of the restart loop.
    if session_id is not None:
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
        session_id, session_data = await select_or_create_session(
            prompt, workspace_path, summary_model,
            backend_name=backend.name,
        )

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
        full_prompt = CAPABILITY_HINT + "\n\n" + contextual_prompt
        logger.info(
            "Running %s (prompt %d chars, context %d chars)",
            backend.name, len(prompt), len(contextual_prompt),
        )

        try:
            proc = await backend.launch(
                workspace_path, full_prompt, model, approval,
            )
        except FileNotFoundError:
            result = AgentResult()
            result.text = f"Error: {backend.executable} CLI not found on PATH."
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
            async for line in _iter_stream_lines(proc.stdout):
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
        result.text = f"{backend.name} exited with code {proc.returncode}"
        if stderr:
            result.text += f"\n{stderr}"
        result.events.append(ChatEvent(kind="text", content=result.text))

    # Log the original prompt (including injected context) to session.
    async with get_workspace_lock(workspace_path):
        _log_to_session(workspace_path, session_id, effective_prompt, result)

    await _maybe_compact(
        workspace_path, session_id, summary_model, backend_name=backend.name,
    )

    # Auto-title sessions whose name still matches the default
    # "Session YYYY-MM-DD" pattern. Fire-and-forget so the user-visible
    # reply isn't gated on a second backend call.
    asyncio.create_task(_maybe_auto_title(
        workspace_path, session_id, summary_model,
        backend_name=backend.name,
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


# ------------------------------------------------------------------
# Auto-compaction
# ------------------------------------------------------------------

async def _maybe_compact(
    workspace_path: str, session_id: str, summary_model: str | None = None,
    *, backend_name: str | None = None,
) -> None:
    """Compact session if uncompacted messages reach the compact interval.

    The trigger is len(messages) >= interval, checked with a single load.
    Compaction runs outside the workspace lock so other requests
    aren't stalled.
    """
    try:
        data = session.load_session(workspace_path, session_id)
        if data is None:
            return
        msgs = data.get("messages", [])
        interval = workspace_mod.get_compact_interval(workspace_path)
        if interval <= 0 or len(msgs) < interval:
            return

        logger.info(
            "Auto-compact triggered (msgs=%d, interval=%d)",
            len(msgs), interval,
        )
        existing_summary = data.get("summary") or ""
        new_summary, new_long_term, new_title = await compact_session(
            workspace_path, session_id, summary_model,
            backend_name=backend_name,
            _preloaded_data=data,
        )
        if not new_summary:
            logger.error(
                "Compaction produced empty summary for session %s",
                session_id,
            )
            return
        # Reject summaries that are suspiciously short compared to the
        # existing one - a sign of a truncated or failed backend response.
        min_len = max(100, len(existing_summary) // 2)
        if len(new_summary) < min_len:
            logger.error(
                "Compaction summary too short (%d chars, min %d) "
                "for session %s - keeping existing",
                len(new_summary), min_len, session_id,
            )
            return
        async with get_workspace_lock(workspace_path):
            session.set_summary(
                workspace_path, session_id, new_summary,
                keep_recent=KEEP_RECENT_AFTER_COMPACT,
                long_term_rewrite=new_long_term,
                title=new_title,
            )
            colony_count = colony.bump_compact_count(workspace_path)
        lt_count = len(new_long_term) if new_long_term is not None else "?"
        logger.info(
            "Session %s compacted, summary %d chars, long_term %s items",
            session_id, len(new_summary), lt_count,
        )
        maybe_trigger_colony(
            workspace_path, colony_count, summary_model,
            backend_name=backend_name,
        )
    except Exception:
        logger.error("Compaction check failed", exc_info=True)


def maybe_trigger_colony(
    workspace_path: str,
    compact_count: int,
    summary_model: str | None,
    *,
    backend_name: str | None,
) -> None:
    """Fire a colony consolidation task when the interval is hit.

    Should be called after a successful compaction (auto or manual)
    has bumped the workspace-wide compaction counter. Fire-and-forget:
    the user-visible reply isn't gated on it, and a shutdown mid-pass
    just leaves the colony unchanged for the next interval hit.
    """
    interval = workspace_mod.get_colony_interval(workspace_path)
    if interval <= 0 or compact_count % interval != 0:
        return
    logger.info(
        "Colony pass triggered (count=%d, interval=%d)",
        compact_count, interval,
    )
    asyncio.create_task(colony_consolidate(
        workspace_path, summary_model, backend_name=backend_name,
    ))


# ------------------------------------------------------------------
# Colony consolidation
# ------------------------------------------------------------------

# Per-workspace lock so two compactions hitting the same interval mark
# don't both spawn a colony pass that would race against each other.
_colony_in_flight: set[str] = set()


async def colony_consolidate(
    workspace_path: str,
    summary_model: str | None = None,
    *,
    backend_name: str | None = None,
) -> bool:
    """Promote recurring long-term items into the workspace-shared colony.

    Reads every session's ``long_term`` list, asks the backend to
    identify items that recur across sessions, writes the new colony
    list, and rewrites each session's ``long_term`` with the promoted
    items removed. Returns True on a successful apply.
    """
    if workspace_path in _colony_in_flight:
        logger.info(
            "Colony pass already in flight for %s, skipping",
            workspace_path,
        )
        return False
    _colony_in_flight.add(workspace_path)
    try:
        return await _colony_consolidate_inner(
            workspace_path, summary_model, backend_name=backend_name,
        )
    finally:
        _colony_in_flight.discard(workspace_path)


async def _colony_consolidate_inner(
    workspace_path: str,
    summary_model: str | None,
    *,
    backend_name: str | None,
) -> bool:
    backend = backends_agent.get_backend(backend_name)

    # Collect non-empty long_term lists from every session in the workspace.
    # The session name accompanies each list so the model can judge whether
    # a colony item's topic is still represented in the workspace.
    inputs: list[tuple[str, str, list[str]]] = []
    for meta in session.list_sessions(workspace_path):
        sid = meta["id"]
        data = session.load_session(workspace_path, sid)
        if data is None:
            continue
        lt = data.get("long_term") or []
        if not lt:
            continue
        name = data.get("name") or sid[:8]
        inputs.append((sid, name, lt))

    if not inputs:
        logger.info(
            "Colony pass: no sessions with long-term items in %s",
            workspace_path,
        )
        return False

    existing_colony = colony.get_items(workspace_path)

    parts: list[str] = ["Current colony list:"]
    if existing_colony:
        for it in existing_colony:
            parts.append(f"- {it}")
    else:
        parts.append("(empty)")
    parts.append("")

    # Greedy build: if a session can't fit in the budget, skip the rest.
    # Sessions are listed newest-first so older sessions are dropped
    # first under tight budgets.
    used = sum(len(p) + 1 for p in parts)
    included: list[str] = []
    for sid, name, lt in inputs:
        block_lines = [
            f"Session: {name}",
            f"[SESSION:{sid}]",
        ]
        for it in lt:
            block_lines.append(f"- {it}")
        block_lines.append("[/SESSION]")
        block_lines.append("")
        block = "\n".join(block_lines)
        if used + len(block) > COLONY_MAX_INPUT_CHARS:
            logger.warning(
                "Colony input over budget; dropping %d session(s) from "
                "consolidation pass",
                len(inputs) - len(included),
            )
            break
        parts.append(block)
        used += len(block) + 1
        included.append(sid)

    if not included:
        logger.info("Colony pass: no sessions fit in budget, skipping")
        return False

    full_prompt = f"{COLONY_PROMPT}\n\n" + "\n".join(parts)

    try:
        proc = await backend.launch(
            workspace_path, full_prompt, summary_model, approval="full",
            compaction=True,
        )
    except FileNotFoundError:
        logger.error(
            "%s CLI not found on PATH - cannot consolidate colony",
            backend.executable,
        )
        return False

    output = await _drain_internal_proc(
        proc, backend, COLONY_TIMEOUT, f"Colony ({workspace_path})",
    )

    if not output:
        stderr = (
            await proc.stderr.read()
        ).decode("utf-8", errors="replace").strip()
        logger.warning(
            "Colony consolidation produced no output (exit %d): %s",
            proc.returncode, stderr,
        )
        return False

    new_colony, per_session = _parse_colony_output(output)
    if new_colony is None:
        logger.warning("Colony output missing [COLONY] block; aborting")
        return False

    # Drop output for sessions we didn't actually send (the model
    # invented them) so we don't accidentally rewrite anything else.
    included_set = set(included)
    per_session = {
        sid: lt for sid, lt in per_session.items() if sid in included_set
    }

    # Apply atomically: write colony, then re-read+rewrite each session
    # so concurrent message appends aren't clobbered.
    async with get_workspace_lock(workspace_path):
        colony.set_items(workspace_path, new_colony)
        for sid, new_lt in per_session.items():
            data = session.load_session(workspace_path, sid)
            if data is None:
                continue
            cleaned = [item for item in new_lt if item][:session.LONG_TERM_CAP]
            data["long_term"] = cleaned
            session.save_session(workspace_path, sid, data)

    logger.info(
        "Colony consolidated for %s: %d colony item(s); %d session(s) rewritten",
        workspace_path, len(new_colony), len(per_session),
    )
    return True


# ------------------------------------------------------------------
# Auto-titling
# ------------------------------------------------------------------

# Per-session lock so two concurrent run() calls (in theory) on the
# same session don't both spawn a title pass. The bot's per-user lock
# already serializes turns, but we don't rely on that here.
_title_in_flight: set[tuple[str, str]] = set()


async def _maybe_auto_title(
    workspace_path: str, session_id: str, summary_model: str | None,
    *, backend_name: str | None,
) -> None:
    """Generate a title once a session has its first assistant reply.

    Skipped when the session already has a custom name (anything other
    than the auto-generated ``Session YYYY-MM-DD``) so we don't
    overwrite a meaningful title with a freshly-generated one. The
    compaction path refreshes the title separately.
    """
    key = (workspace_path, session_id)
    if key in _title_in_flight:
        return
    try:
        data = session.load_session(workspace_path, session_id)
        if data is None:
            return
        if not session.is_default_name(data.get("name")):
            return
        # Need at least one assistant reply before titling makes sense.
        msgs = data.get("messages", [])
        if not any(m.get("role") == "assistant" for m in msgs):
            return
        _title_in_flight.add(key)
        title = await generate_session_title(
            workspace_path, session_id, summary_model,
            backend_name=backend_name, _preloaded_data=data,
        )
        if not title:
            return
        async with get_workspace_lock(workspace_path):
            session.set_session_name(workspace_path, session_id, title)
        logger.info("Session %s auto-titled: %s", session_id, title)
    except Exception:
        logger.warning("Auto-title failed", exc_info=True)
    finally:
        _title_in_flight.discard(key)


async def generate_session_title(
    workspace_path: str,
    session_id: str,
    summary_model: str | None = None,
    *,
    backend_name: str | None = None,
    _preloaded_data: dict | None = None,
) -> str | None:
    """Run a small backend call to title a session. Returns None on failure."""
    backend = backends_agent.get_backend(backend_name)
    data = _preloaded_data or session.load_session(workspace_path, session_id)
    if data is None:
        return None

    parts: list[str] = []
    summary = data.get("summary")
    if summary:
        parts.append(f"Previous summary:\n{summary}\n")
    parts.append("Recent messages:")

    # Newest-first under a tight char budget; the title only needs the
    # gist, so we don't pull the whole history.
    used = 0
    msg_lines: list[str] = []
    for msg in reversed(data.get("messages", [])):
        role = msg.get("role", "?").capitalize()
        content = msg.get("content", "")
        if len(content) > MSG_CONTENT_MAX:
            content = content[:MSG_CONTENT_MAX] + "…"
        line = f"{role}: {content}"
        if used + len(line) > TITLE_CONTEXT_CHARS:
            break
        msg_lines.append(line)
        used += len(line) + 1
    msg_lines.reverse()
    if not msg_lines:
        return None
    parts.extend(msg_lines)

    full_prompt = f"{TITLE_PROMPT}\n\n" + "\n".join(parts)

    try:
        proc = await backend.launch(
            workspace_path, full_prompt, summary_model, approval="full",
            compaction=True,
        )
    except FileNotFoundError:
        logger.warning(
            "%s CLI not found on PATH - cannot title session",
            backend.executable,
        )
        return None

    raw = await _drain_internal_proc(
        proc, backend, TITLE_TIMEOUT, f"Title (session {session_id})",
    )
    return _clean_title(raw) if raw else None


async def compact_session(
    workspace_path: str,
    session_id: str,
    summary_model: str | None = None,
    *,
    backend_name: str | None = None,
    _preloaded_data: dict | None = None,
) -> tuple[str, list[str] | None, str | None]:
    """Run the selected backend to compact a session.

    Returns (summary, long_term, title). long_term is the new complete
    list, or None if the model did not emit a [LONG_TERM] block
    (existing list kept). title is None if the model did not emit a
    [TITLE] block. On failure returns ("", None, None). Does NOT write
    to disk - caller takes the workspace lock and calls set_summary.

    _preloaded_data: pass already-loaded session dict to skip a disk read
    (used by _maybe_compact which loads the data to check the interval).
    """
    backend = backends_agent.get_backend(backend_name)
    data = _preloaded_data or session.load_session(workspace_path, session_id)
    if data is None:
        return ("", None, None)
    messages = data.get("messages", [])
    existing_summary = data.get("summary")
    existing_long_term: list[str] = data.get("long_term") or []

    if not messages:
        return ("", None, None)

    # Build the content to summarize, staying within a token budget.
    # Large prompts cause the summary model to return truncated/empty output.
    parts: list[str] = []
    if existing_long_term:
        # Show up to 15% of the budget for the existing list so the model
        # knows what to rewrite. With a target of <=30 items this is plenty.
        lt_max = int(MAX_SUMMARY_CHARS * 0.15)
        lt_lines: list[str] = []
        lt_used = 0
        for item in reversed(existing_long_term):
            line = f"- {item}"
            if lt_used + len(line) + 1 > lt_max:
                break
            lt_lines.append(line)
            lt_used += len(line) + 1
        lt_lines.reverse()
        if lt_lines:
            parts.append(
                "Existing long-term items (rewrite this list per the "
                "instructions above):"
            )
            parts.extend(lt_lines)
            parts.append("")
    if existing_summary:
        parts.append(f"Previous summary:\n{existing_summary}\n")
    parts.append("Conversation to summarize:")

    # Add messages newest-first until we hit the budget, then reverse
    overhead = len(SUMMARY_PROMPT) + sum(len(p) for p in parts) + 200
    budget = max(0, MAX_SUMMARY_CHARS - overhead)
    msg_lines: list[str] = []
    used = 0
    for msg in reversed(messages):
        role = msg.get("role", "?").capitalize()
        content = msg.get("content", "")
        line = f"{role}: {content}"
        if used + len(line) > budget:
            break
        msg_lines.append(line)
        used += len(line) + 1
    msg_lines.reverse()
    parts.extend(msg_lines)

    if not msg_lines:
        logger.warning(
            "Session %s messages too large even for a single entry",
            session_id,
        )
        return ("", None, None)

    full_prompt = f"{SUMMARY_PROMPT}\n\n" + "\n".join(parts)

    logger.info(
        "Running %s compaction for session %s", backend.name, session_id,
    )

    try:
        proc = await backend.launch(
            workspace_path, full_prompt, summary_model, approval="full",
            compaction=True,
        )
    except FileNotFoundError:
        logger.error(
            "%s CLI not found on PATH - cannot compact session",
            backend.executable,
        )
        return ("", None, None)

    new_summary = await _drain_internal_proc(
        proc, backend, COMPACT_TIMEOUT, f"Compaction (session {session_id})",
    )

    if not new_summary:
        stderr = (
            await proc.stderr.read()
        ).decode("utf-8", errors="replace").strip()
        logger.warning(
            "Compaction produced no summary (exit %d): %s",
            proc.returncode, stderr,
        )
        return ("", None, None)

    summary, long_term, title = _parse_compaction_output(new_summary)
    return (summary, long_term, title)
