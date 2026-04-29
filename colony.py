"""Workspace-level shared memory ("colony").

Colony is a workspace-shared layer that sits above each session's
``long_term`` list. Every Nth per-session compaction (``colony_interval``,
default 3, configurable in workspace settings), the agent looks across
all sessions in the workspace, finds long-term items that recur or
overlap, and promotes them into the colony — removing them from each
session's own list so there's a single canonical place for them.

When building the prompt for any turn, the colony is prepended to the
session's long-term list and summary so the agent sees both
workspace-shared and session-scoped knowledge.

File shape (``.cozter/colony.json``):
    {
        "items": ["fact 1", "fact 2", ...],
        "compact_count": <int>
    }

``compact_count`` is bumped after every successful per-session
compaction; when ``compact_count % colony_interval == 0`` we run the
consolidation pass.
"""

import asyncio
import json
import logging
import os
import re

from . import backends_agent, session
from . import workspace as workspace_mod
from .utils import atomic_write as _atomic_write
from .utils import drain_llm_subprocess

logger = logging.getLogger(__name__)

COZTER_DIR = ".cozter"
COLONY_FILE = "colony.json"

DEFAULT_COLONY_INTERVAL = 3
COLONY_CAP = 100  # hard cap on items so the prompt context stays bounded


def _path(workspace: str) -> str:
    return os.path.join(workspace, COZTER_DIR, COLONY_FILE)


def _load(workspace: str) -> dict:
    path = _path(workspace)
    if not os.path.exists(path):
        return {"items": [], "compact_count": 0}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupt colony file, ignoring: %s", path)
        return {"items": [], "compact_count": 0}


def _save(workspace: str, data: dict) -> None:
    target_dir = os.path.join(workspace, COZTER_DIR)
    os.makedirs(target_dir, exist_ok=True)
    _atomic_write(_path(workspace), data, tmp_dir=target_dir)


def get_items(workspace: str) -> list[str]:
    return _load(workspace)["items"]


def set_items(workspace: str, items: list[str]) -> None:
    data = _load(workspace)
    cleaned = [s for s in (i.strip() for i in items if i) if s]
    if len(cleaned) > COLONY_CAP:
        dropped = len(cleaned) - COLONY_CAP
        logger.warning(
            "Colony rewrite exceeded cap (%d); dropping %d oldest item(s)",
            COLONY_CAP, dropped,
        )
        cleaned = cleaned[-COLONY_CAP:]
    data["items"] = cleaned
    _save(workspace, data)


def get_compact_count(workspace: str) -> int:
    return _load(workspace)["compact_count"]


def bump_compact_count(workspace: str) -> int:
    """Increment the workspace-wide compaction counter and return its new value.

    Caller is expected to hold the workspace lock so two concurrent
    compactions don't both observe the same count.
    """
    data = _load(workspace)
    data["compact_count"] = int(data.get("compact_count") or 0) + 1
    _save(workspace, data)
    return data["compact_count"]


# ---------------------------------------------------------------------------
# Consolidation — promote recurring long_term items into the colony,
# rewrite each session's long_term with promoted items removed, and
# retire colony entries whose topic no longer appears in any session.
# ---------------------------------------------------------------------------

CONSOLIDATE_PROMPT = (
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
CONSOLIDATE_TIMEOUT = 180  # heavier than per-session compaction
CONSOLIDATE_MAX_INPUT_CHARS = 100_000

_SESSION_BLOCK_RE = re.compile(
    r"\[SESSION:([^\]\s]+)\](.*?)\[/SESSION\]", re.DOTALL,
)

# Per-workspace guard so two compactions hitting the same interval mark
# don't both spawn a colony pass that races against itself.
_consolidate_in_flight: set[str] = set()


def _parse_consolidate_output(
    text: str,
) -> tuple[list[str] | None, dict[str, list[str]]]:
    """Extract (colony, {session_id: long_term}) from the model output.

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


def maybe_trigger(
    workspace_path: str,
    compact_count: int,
    summary_model: str | None,
    *,
    backend_name: str | None,
) -> None:
    """Fire a consolidation task when the configured interval is hit.

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
    asyncio.create_task(consolidate(
        workspace_path, summary_model, backend_name=backend_name,
    ))


async def consolidate(
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
    if workspace_path in _consolidate_in_flight:
        logger.info(
            "Colony pass already in flight for %s, skipping",
            workspace_path,
        )
        return False
    _consolidate_in_flight.add(workspace_path)
    try:
        return await _consolidate_inner(
            workspace_path, summary_model, backend_name=backend_name,
        )
    finally:
        _consolidate_in_flight.discard(workspace_path)


async def _consolidate_inner(
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
    for data in session.list_sessions_with_data(workspace_path):
        lt = data.get("long_term") or []
        if not lt:
            continue
        sid = data["id"]
        name = data.get("name") or sid[:8]
        inputs.append((sid, name, lt))

    if not inputs:
        logger.info(
            "Colony pass: no sessions with long-term items in %s",
            workspace_path,
        )
        return False

    existing_colony = get_items(workspace_path)

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
        if used + len(block) > CONSOLIDATE_MAX_INPUT_CHARS:
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

    full_prompt = f"{CONSOLIDATE_PROMPT}\n\n" + "\n".join(parts)

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

    output = await drain_llm_subprocess(
        proc, backend, CONSOLIDATE_TIMEOUT, f"Colony ({workspace_path})",
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

    new_colony, per_session = _parse_consolidate_output(output)
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
    async with workspace_mod.get_lock(workspace_path):
        set_items(workspace_path, new_colony)
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
