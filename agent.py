"""Agent orchestrator - session management, context building, compaction.

The actual CLI invocation and event parsing are delegated to backend
adapters in the ``backends_agent`` package (currently codex and copilot).
The backend is chosen per workspace via ``workspace.get_backend_name``.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import AsyncIterator

from . import backends_agent, session
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
    "facts that survive future compactions).\n\n"
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
    "=== OUTPUT FORMAT ===\n"
    "Emit exactly two sections. Omit a section's body if empty, "
    "but always include the markers.\n\n"
    "[SUMMARY]\n"
    "<scratch summary prose here>\n"
    "[/SUMMARY]\n\n"
    "[LONG_TERM]\n"
    "- <item 1>\n"
    "- <item 2>\n"
    "[/LONG_TERM]"
)


def _parse_compaction_output(
    text: str,
) -> tuple[str, list[str] | None]:
    """Extract (summary, long_term) from model output.

    long_term is None if [LONG_TERM] markers are absent (no rewrite).
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
    if summary is None:
        # Fallback: treat full text as summary, stripping the long-term block.
        fallback = text
        open_tag, close_tag = "[LONG_TERM]", "[/LONG_TERM]"
        i = fallback.find(open_tag)
        if i != -1:
            j = fallback.find(close_tag, i + len(open_tag))
            if j != -1:
                fallback = fallback[:i] + fallback[j + len(close_tag):]
        summary = fallback.strip()
    return summary, long_term


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
    prompt: str, session_data: dict | None,
) -> str:
    """Prepend session history to the prompt so the agent has full context."""
    data = session_data
    if data is None:
        return prompt
    summary: str | None = data.get("summary")
    long_term: list[str] = data.get("long_term") or []
    messages: list[dict] = data.get("messages", [])

    if not summary and not messages and not long_term:
        return prompt

    parts: list[str] = []

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

    # Truncate if too long - drop oldest messages, keep long-term + summary
    if len(full) > MAX_HISTORY_CHARS:
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
        overhead = len(prompt) + len(lt_block) + len(summary_block) + 500
        msg_budget = max(0, MAX_HISTORY_CHARS - overhead)
        if msg_budget == 0 and messages:
            logger.warning(
                "History truncation: long-term/summary fill budget; "
                "dropping all %d recent messages", len(messages),
            )

        history_parts: list[str] = []
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
) -> AgentResult:
    """Run the selected agent CLI with session history prepended.

    backend_name selects the CLI adapter (codex/copilot). When None, the
    default backend is used. The workspace's configured backend should be
    passed in by the caller.

    on_event  - called for each parsed event as it arrives (streaming).
    inject_queue - when a message is put, the running subprocess is killed
                   and restarted with the injected context appended.
    """
    backend = backends_agent.get_backend(backend_name)

    # ensure_session_with_data gives us (id, data) from a single load.
    # session_data is reused on every inject restart so the session file
    # is not re-read for each iteration of the restart loop.
    session_id, session_data = session.ensure_session_with_data(
        workspace_path, user_id,
    )

    injected: list[str] = []

    while True:  # restart loop for inject
        effective_prompt = prompt
        if injected:
            effective_prompt += (
                "\n\n[Additional context from user while you were thinking]:\n"
                + "\n".join(injected)
            )

        contextual_prompt = _build_contextual_prompt(
            effective_prompt, session_data,
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
        interval = data.get(
            "compact_interval", session.DEFAULT_COMPACT_INTERVAL,
        )
        if interval <= 0 or len(msgs) < interval:
            return

        logger.info(
            "Auto-compact triggered (msgs=%d, interval=%d)",
            len(msgs), interval,
        )
        existing_summary = data.get("summary") or ""
        new_summary, new_long_term = await compact_session(
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
            )
        lt_count = len(new_long_term) if new_long_term is not None else "?"
        logger.info(
            "Session %s compacted, summary %d chars, long_term %s items",
            session_id, len(new_summary), lt_count,
        )
    except Exception:
        logger.error("Compaction check failed", exc_info=True)


async def compact_session(
    workspace_path: str,
    session_id: str,
    summary_model: str | None = None,
    *,
    backend_name: str | None = None,
    _preloaded_data: dict | None = None,
) -> tuple[str, list[str] | None]:
    """Run the selected backend to compact a session.

    Returns (summary, long_term). long_term is the new complete list, or
    None if the model did not emit a [LONG_TERM] block (existing list kept).
    On failure returns ("", None). Does NOT write to disk - caller takes
    the workspace lock and calls set_summary.

    _preloaded_data: pass already-loaded session dict to skip a disk read
    (used by _maybe_compact which loads the data to check the interval).
    """
    backend = backends_agent.get_backend(backend_name)
    data = _preloaded_data or session.load_session(workspace_path, session_id)
    if data is None:
        return ("", None)
    messages = data.get("messages", [])
    existing_summary = data.get("summary")
    existing_long_term: list[str] = data.get("long_term") or []

    if not messages:
        return ("", None)

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
        return ("", None)

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
        return ("", None)

    # Collect the last agent text from the JSON event stream.
    new_summary = ""
    try:
        async with asyncio.timeout(COMPACT_TIMEOUT):
            async for line in _iter_stream_lines(proc.stdout):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    if not new_summary:
                        new_summary = line  # bare-text fallback
                    continue
                text = backend.extract_agent_text(event)
                if text:
                    new_summary = text  # keep updating - last one wins

            await proc.wait()
    except TimeoutError:
        logger.error(
            "Compaction timed out after %ds for session %s",
            COMPACT_TIMEOUT, session_id,
        )
        try:
            proc.kill()
            await proc.wait()
        except OSError:
            pass
        return ("", None)

    if not new_summary:
        stderr = (
            await proc.stderr.read()
        ).decode("utf-8", errors="replace").strip()
        logger.warning(
            "Compaction produced no summary (exit %d): %s",
            proc.returncode, stderr,
        )
        return ("", None)

    summary, long_term = _parse_compaction_output(new_summary)
    return (summary, long_term)
