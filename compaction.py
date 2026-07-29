"""Per-session compaction.

Rolls a session's message history into a SCRATCH summary plus a
LONG-TERM list, replacing the raw messages so the conversation can
continue without exhausting the context window. Triggered automatically
when ``len(messages) >= compact_interval`` after a turn. The interval is
set via ``/compact <number>``; there is no one-shot manual compaction.

A successful compaction also bumps the workspace-wide colony counter
and may fire a colony consolidation pass via ``colony.maybe_trigger``.
"""

import logging

from . import backends_agent, colony, session, titling
from . import workspace as workspace_mod
from .utils import (
    extract_marker_block, parse_bullets, run_internal_backend,
    strip_marker_block, take_recent_lines,
)

logger = logging.getLogger(__name__)

# A reply can finish while a previous post-turn compaction is still running.
# Do not let both snapshots race to summarize and rewrite the same session.
_in_flight: set[tuple[str, str]] = set()


KEEP_RECENT_AFTER_COMPACT = 5
MAX_SUMMARY_CHARS = 80_000  # ~20K tokens - safe for most models
COMPACT_TIMEOUT = 240  # seconds; large sessions with rich long-term lists need headroom

SUMMARY_PROMPT = (
    "You are compacting a conversation into two memory layers: a SCRATCH "
    "summary (rewritten each compaction) and LONG-TERM memory (persistent "
    "facts that survive future compactions). You will also produce a short "
    "TITLE that names the session.\n\n"
    "IMPORTANT: This is a pure text-summarization task. Do NOT call any "
    "tools, shell commands, file reads, or web fetches. The conversation "
    "below is everything you need — work from it directly and emit the "
    "output as plain text.\n\n"
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


def _parse_output(
    text: str,
) -> tuple[str, list[str] | None, str | None]:
    """Extract (summary, long_term, title) from the compaction output.

    long_term is None if [LONG_TERM] markers are absent (no rewrite).
    title is None if [TITLE] markers are absent or empty.
    Falls back to treating the entire text as the summary if [SUMMARY]
    markers are absent, so misbehaving models still produce a usable result.
    """
    summary = extract_marker_block(text, "SUMMARY")
    lt_block = extract_marker_block(text, "LONG_TERM")
    long_term: list[str] | None = (
        parse_bullets(lt_block) if lt_block is not None else None
    )
    title_block = extract_marker_block(text, "TITLE")
    title: str | None = (
        titling.clean_title(title_block) if title_block else None
    )
    if summary is None:
        # Fallback: treat full text as summary, stripping the other blocks.
        fallback = text
        for tag in ("LONG_TERM", "TITLE"):
            fallback = strip_marker_block(fallback, tag)
        summary = fallback.strip()
    return summary, long_term, title


async def maybe_compact(
    workspace_path: str, session_id: str, summary_model: str | None = None,
    *, backend_name: str | None = None,
) -> None:
    """Compact session if uncompacted messages reach the compact interval.

    The trigger is len(messages) >= interval, checked with a single load.
    Compaction runs outside the workspace lock so other requests
    aren't stalled.
    """
    key = (
        workspace_mod.canonicalize_workspace_path(workspace_path), session_id,
    )
    if key in _in_flight:
        logger.debug("Compaction already in progress for session %s", session_id)
        return

    _in_flight.add(key)
    try:
        async with workspace_mod.get_memory_maintenance_lock(workspace_path):
            await _maybe_compact_under_maintenance_lock(
                workspace_path, session_id, summary_model,
                backend_name=backend_name,
            )
    except Exception:
        logger.error("Compaction check failed", exc_info=True)
    finally:
        _in_flight.discard(key)


async def _maybe_compact_under_maintenance_lock(
    workspace_path: str,
    session_id: str,
    summary_model: str | None,
    *,
    backend_name: str | None,
) -> None:
    """Run one compaction while no colony rewrite can race its snapshot."""
    data = session.load_session(workspace_path, session_id)
    if data is None:
        return
    msgs = data.get("messages", [])
    interval = workspace_mod.get_compact_interval(workspace_path)
    if len(msgs) < interval:
        return

    logger.info(
        "Auto-compact triggered (msgs=%d, interval=%d)",
        len(msgs), interval,
    )
    existing_summary = data.get("summary") or ""
    new_summary, new_long_term, new_title, covered_count = await compact_session(
        workspace_path, session_id, summary_model,
        backend_name=backend_name,
        _preloaded_data=data,
    )
    if not new_summary or covered_count <= KEEP_RECENT_AFTER_COMPACT:
        logger.error(
            "Compaction did not cover enough messages for session %s "
            "(covered=%d, keep_recent=%d)",
            session_id, covered_count, KEEP_RECENT_AFTER_COMPACT,
        )
        return
    # Reject summaries that are suspiciously short compared to the existing
    # one - a sign of a truncated or failed backend response.
    min_len = max(100, len(existing_summary) // 2)
    if len(new_summary) < min_len:
        logger.error(
            "Compaction summary too short (%d chars, min %d) "
            "for session %s - keeping existing",
            len(new_summary), min_len, session_id,
        )
        return
    async with workspace_mod.get_lock(workspace_path):
        session.set_summary(
            workspace_path, session_id, new_summary,
            keep_recent=KEEP_RECENT_AFTER_COMPACT,
            long_term_rewrite=new_long_term,
            title=new_title,
            # Only trim the contiguous prefix actually sent to the summary
            # backend. A foreground turn can append more while the summary
            # runs, and oversized histories intentionally leave their later
            # messages raw for the next pass.
            summarized_count=covered_count,
        )
        colony_count = colony.bump_compact_count(workspace_path)
    lt_count = len(new_long_term) if new_long_term is not None else "?"
    logger.info(
        "Session %s compacted, summary %d chars, long_term %s items",
        session_id, len(new_summary), lt_count,
    )
    colony.maybe_trigger(
        workspace_path, colony_count, summary_model,
        backend_name=backend_name,
    )


def _take_oldest_message_lines(messages: list[dict], budget: int) -> list[str]:
    """Return the largest budget-fitting contiguous prefix of *messages*.

    ``session.set_summary`` can only safely discard a prefix of the raw
    history. Selecting a newest suffix made it possible to mark older,
    omitted messages as summarized and then delete them. Keep the prefix
    contiguous and let a later pass compact the remaining suffix once the
    raw overlap has advanced.
    """
    lines: list[str] = []
    used = 0
    for message in messages:
        line = session.format_msg_line(message, cap=None)
        if used + len(line) > budget:
            break
        lines.append(line)
        # Match utils.take_recent_lines' conservative accounting for the
        # newline that joins adjacent lines.
        used += len(line) + 1
    return lines


async def compact_session(
    workspace_path: str,
    session_id: str,
    summary_model: str | None = None,
    *,
    backend_name: str | None = None,
    _preloaded_data: dict | None = None,
) -> tuple[str, list[str] | None, str | None, int]:
    """Run the selected backend to compact a session.

    Returns ``(summary, long_term, title, covered_count)``. ``covered_count``
    is the number of consecutive oldest raw messages included in the prompt;
    callers must only trim against that count. ``long_term`` is the new
    complete list, or None if the model did not emit a [LONG_TERM] block
    (existing list kept). ``title`` is None if the model did not emit a
    [TITLE] block. On failure returns ``("", None, None, 0)``. Does NOT write
    to disk - caller takes the workspace lock and calls ``set_summary``.

    _preloaded_data: pass already-loaded session dict to skip a disk read
    (used by maybe_compact which loads the data to check the interval).
    """
    backend = backends_agent.get_backend(backend_name)
    data = _preloaded_data or session.load_session(workspace_path, session_id)
    if data is None:
        return ("", None, None, 0)
    messages = data.get("messages", [])
    existing_summary = data.get("summary")
    existing_long_term: list[str] = data.get("long_term") or []

    if not messages:
        return ("", None, None, 0)

    # Build the content to summarize, staying within a token budget.
    # Large prompts cause the summary model to return truncated/empty output.
    parts: list[str] = []
    if existing_long_term:
        # Show up to 15% of the budget for the existing list so the model
        # knows what to rewrite. With a target of <=30 items this is plenty.
        lt_max = int(MAX_SUMMARY_CHARS * 0.15)
        lt_lines = take_recent_lines(
            existing_long_term, lt_max, lambda x: f"- {x}",
        )
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

    # Add a contiguous oldest prefix until we hit the budget. cap=None so
    # the model sees full message content (compaction's budget is generous
    # enough to afford it). The caller only removes this exact prefix, which
    # prevents unsent messages from being mistaken for summarized history.
    overhead = len(SUMMARY_PROMPT) + sum(len(p) for p in parts) + 200
    budget = max(0, MAX_SUMMARY_CHARS - overhead)
    msg_lines = _take_oldest_message_lines(messages, budget)
    parts.extend(msg_lines)

    if not msg_lines:
        logger.warning(
            "Session %s messages too large even for a single entry",
            session_id,
        )
        return ("", None, None, 0)

    full_prompt = f"{SUMMARY_PROMPT}\n\n" + "\n".join(parts)

    logger.info(
        "Running %s compaction for session %s", backend.name, session_id,
    )

    new_summary = await run_internal_backend(
        backend,
        workspace_path,
        full_prompt,
        summary_model,
        timeout=COMPACT_TIMEOUT,
        label=f"Compaction (session {session_id})",
        log=logger,
        missing_executable_message=(
            "%s CLI not found on PATH - cannot compact session"
        ),
    )
    if not new_summary:
        return ("", None, None, 0)

    summary, long_term, title = _parse_output(new_summary)
    return (summary, long_term, title, len(msg_lines))
