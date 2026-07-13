"""Flexible agent — understand, split by difficulty, route, merge.

The ``flexible`` agent is not a CLI of its own: it is a meta-agent that
spends a cheap summary-model call to understand the user's request and
split it into sub-tasks, routes each sub-task to the agent+model
configured for its difficulty tier (low/mid/high), then spends a second
summary-model call to merge the workers' reports into the single reply
the user sees.

This module holds the side-effect-free half — prompt construction and
output parsing. The orchestration loop lives in :mod:`agent`, which owns
the backend subprocess driver the workers run on.
"""

import re
from collections.abc import Collection
from dataclasses import dataclass

from .utils import extract_marker_block

BACKEND_NAME = "flexible"

# Difficulty tiers, cheapest first. Each is bound to its own agent+model
# pair via /agent_flexible_<tier> and /model_flexible_<tier>.
TIERS: tuple[str, ...] = ("low", "mid", "high")

# Tier used when planning fails or emits nothing parseable. Deliberately
# the strongest one: a botched split should under-spend nobody's time by
# handing a hard task to a weak model.
FALLBACK_TIER = "high"

TIER_DESCRIPTIONS = {
    "low": "Straightforward, well-scoped work with clear intent",
    "mid": "Some reasoning needed, but the problem stays bounded",
    "high": (
        "Only for ambiguity, complex logic, or deeper system understanding"
    ),
}

# Bounds the fan-out: each sub-task is a full agent turn, so an
# over-eager planner would otherwise turn one message into a dozen runs.
MAX_SUBTASKS = 6

PLAN_TIMEOUT = 120  # seconds; on timeout the planner falls back to one task
MERGE_TIMEOUT = 180  # seconds; on timeout the worker reports are concatenated

# The user-facing rubric the planner grades each sub-task against.
_RUBRIC = (
    "low  - straightforward, well-scoped work with clear intent.\n"
    "       Example: add a small validation check, or extend an existing\n"
    "       function with clearly defined behavior.\n"
    "mid  - some reasoning is required, but the problem stays bounded.\n"
    "       Example: write unit tests for an existing method with known\n"
    "       inputs and outputs.\n"
    "high - ONLY when the task involves ambiguity, complex logic, or\n"
    "       deeper system understanding.\n"
    "       Example: refactor a system with unclear dependencies, or debug\n"
    "       a non-obvious issue."
)

_PLANNER_RULES = (
    "You are the planner for a multi-agent assistant.\n\n"
    "Read the conversation below, understand what the user is asking for, "
    "and split it into the smallest set of sub-tasks that fully covers the "
    "request. Grade each sub-task by difficulty so it can be routed to an "
    "appropriately sized model:\n\n"
    f"{_RUBRIC}\n\n"
    "Rules:\n"
    "- Split only where it helps. A simple request is ONE sub-task; do not "
    "invent busywork.\n"
    f"- At most {MAX_SUBTASKS} sub-tasks.\n"
    "- Order them so that each can be done with only the previous ones' "
    "results in hand. They run one at a time, in order.\n"
    "- Grade honestly. Over-grading burns the expensive model; under-grading "
    "sends a hard problem to a weak one.\n"
    "- Each sub-task must be a self-contained instruction to an agent that "
    "can read and edit files, run commands, and search the web.\n"
    "- Do NOT call any tools or read any files yourself; plan from the text "
    "below.\n"
)

_PLANNER_FORMAT = (
    "Reply in exactly this format and nothing else:\n\n"
    "[UNDERSTANDING]\n"
    "One or two sentences restating what the user actually wants.\n"
    "[/UNDERSTANDING]\n"
    "[PLAN]\n"
    "1. [low|mid|high] first sub-task instruction\n"
    "2. [low|mid|high] second sub-task instruction\n"
    "[/PLAN]\n"
)

_PLANNER_QUESTION_RULE = (
    "If — and only if — the request is too ambiguous to plan against and "
    "guessing wrong would waste real work, skip the plan and ask the user "
    "one short, specific question instead, in this format:\n\n"
    "[QUESTION]\n"
    "your one question\n"
    "[/QUESTION]\n\n"
    "Prefer planning. Only ask when you genuinely cannot proceed.\n"
)

_MERGE_RULES = (
    "You are the voice of a multi-agent assistant. Several worker agents "
    "just carried out the plan below, each reporting back what it did or "
    "found. Write the single reply the user sees.\n\n"
    "Rules:\n"
    "- Answer the user's request directly. Lead with the outcome.\n"
    "- Write as one assistant who did the work, not as an editor stitching "
    "reports together. Never mention the plan, the workers, the sub-tasks, "
    "or their difficulty tiers.\n"
    "- Keep every concrete result the workers produced: code, file paths, "
    "commands, numbers, and errors. Do not re-summarize them into vagueness.\n"
    "- If a worker reported a failure or a blocker, say so plainly.\n"
    "- Reply in the language the user wrote in.\n"
    "- Do NOT call any tools or read any files; the work is already done.\n"
)

# The merge step writes the reply the user actually reads, so it is the
# only step downstream of the planner that can stop the turn and wait.
# Without this the pipeline can end on a blocking question and still let
# the queue drain straight past it, leaving the user's answer to land as
# an unrelated new turn.
_MERGE_QUESTION_RULE = (
    "- If your reply ends by asking the user something you genuinely need "
    "answered before the work can continue, end it with \"[[await]]\" on its "
    "own line. The bot then pauses the chat queue and treats the user's next "
    "message as the answer. Use it only for questions that actually block "
    "progress — not for optional offers or suggested next steps, which should "
    "just be stated without the marker.\n"
)

# Workers run under the autonomy policy, so one that stops to ask has
# already established the turn cannot finish without the user. Tell the
# merge outright instead of leaving it to infer that from the report text.
_MERGE_BLOCKED_RULE = (
    "- A worker stopped because it needs an answer from the user before the "
    "work can continue; its report is marked BLOCKED below. End your reply "
    "with its question, followed by \"[[await]]\" on its own line.\n"
)


@dataclass(frozen=True)
class Subtask:
    """One planned unit of work and the difficulty tier it routes to."""
    tier: str
    instruction: str


@dataclass(frozen=True)
class Plan:
    """A parsed planner reply.

    Exactly one of *question* and *subtasks* drives the turn: a question
    short-circuits into a user-facing ask, otherwise the sub-tasks run.
    """
    understanding: str
    subtasks: tuple[Subtask, ...]
    question: str | None = None


_TIER_ALIASES = {"medium": "mid", "med": "mid", "middle": "mid"}

_PLAN_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?"      # optional bullet / "3." numbering
    r"\[\s*(?P<tier>[A-Za-z]+)\s*\]\s*"     # [high]
    r"(?P<instruction>.+?)\s*$",
    re.IGNORECASE,
)


def normalize_tier(value: str) -> str | None:
    """Return the canonical tier name for *value*, or None if unknown."""
    tier = value.strip().lower()
    tier = _TIER_ALIASES.get(tier, tier)
    return tier if tier in TIERS else None


def fallback_plan(request: str) -> Plan:
    """The whole request as one sub-task on the strongest tier."""
    return Plan(
        understanding="",
        subtasks=(Subtask(tier=FALLBACK_TIER, instruction=request.strip()),),
    )


def parse_plan(raw: str, request: str) -> Plan:
    """Parse a planner reply, falling back to a single high-tier task.

    Tolerant by design: the planner is a cheap model, so a missing
    ``[PLAN]`` wrapper, stray prose, or a ``[medium]`` tier should not
    cost the user their turn.
    """
    if not raw or not raw.strip():
        return fallback_plan(request)

    question = extract_marker_block(raw, "QUESTION")
    if question:
        return Plan(
            understanding=extract_marker_block(raw, "UNDERSTANDING") or "",
            subtasks=(),
            question=question,
        )

    # Scan the whole reply when the planner forgot the wrapper - the
    # numbered "[tier] instruction" lines are distinctive enough.
    block = extract_marker_block(raw, "PLAN") or raw
    subtasks: list[Subtask] = []
    for line in block.splitlines():
        match = _PLAN_LINE_RE.match(line)
        if match is None:
            continue
        tier = normalize_tier(match.group("tier"))
        instruction = match.group("instruction").strip()
        if tier is None or not instruction:
            continue
        subtasks.append(Subtask(tier=tier, instruction=instruction))
        if len(subtasks) == MAX_SUBTASKS:
            break

    if not subtasks:
        return fallback_plan(request)
    return Plan(
        understanding=extract_marker_block(raw, "UNDERSTANDING") or "",
        subtasks=tuple(subtasks),
    )


def build_plan_prompt(context: str, *, collaborative: bool) -> str:
    """Prompt the summary agent to understand and split the request."""
    parts = [_PLANNER_RULES]
    if collaborative:
        parts.append(_PLANNER_QUESTION_RULE)
    parts.append(_PLANNER_FORMAT)
    parts.append("--- conversation ---")
    parts.append(context)
    return "\n".join(parts)


def _render_plan(plan: Plan, *, current: int | None = None) -> str:
    lines = []
    for i, task in enumerate(plan.subtasks):
        marker = "  <- your task" if i == current else ""
        lines.append(
            f"  {i + 1}. [{task.tier}] {task.instruction}{marker}"
        )
    return "\n".join(lines)


def build_subtask_prompt(
    context: str, plan: Plan, index: int, results: list[str],
) -> str:
    """Prompt for the worker running sub-task *index* of *plan*."""
    task = plan.subtasks[index]
    parts = [context, ""]
    parts.append(
        f"[Sub-task {index + 1} of {len(plan.subtasks)}"
        f" — difficulty: {task.tier}]"
    )
    parts.append(
        "You are one worker in a multi-agent pipeline answering the user's"
        " message above. The request was split into the sub-tasks below and"
        " they run one at a time, in order."
    )
    if plan.understanding:
        parts.append(f"\nWhat the user wants: {plan.understanding}")
    parts.append("\nThe full plan:")
    parts.append(_render_plan(plan, current=index))

    if results:
        parts.append("\nWhat the earlier workers reported:")
        for i, text in enumerate(results):
            parts.append(f"\n--- sub-task {i + 1} result ---\n{text}")

    parts.append(
        f"\nDo ONLY sub-task {index + 1}: {task.instruction}\n"
        "Leave the other sub-tasks to their own workers — another agent"
        " handles each one, and duplicated work gets thrown away. Use your"
        " tools to actually carry it out; do not merely describe what you"
        " would do. Then report back concisely: what you did, what you"
        " found, and anything the next worker needs to know. Your report is"
        " read by the agent that writes the user's final reply, not by the"
        " user, so include concrete details (paths, commands, results) and"
        " skip the pleasantries."
    )
    return "\n".join(parts)


def build_merge_prompt(
    context: str, plan: Plan, results: list[str], *, collaborative: bool,
    blocked: Collection[int] = (),
) -> str:
    """Prompt the summary agent to write the user-facing reply.

    *blocked* holds the indices of sub-tasks whose worker stopped to ask
    the user something.
    """
    rules = _MERGE_RULES
    if collaborative:
        rules += _MERGE_QUESTION_RULE
        if blocked:
            rules += _MERGE_BLOCKED_RULE
    parts = [rules, "--- conversation ---", context, ""]
    if plan.understanding:
        parts.append(f"What the user wants: {plan.understanding}")
    parts.append("\nThe plan the workers carried out:")
    parts.append(_render_plan(plan))
    parts.append("\n--- worker reports ---")
    for i, (task, text) in enumerate(zip(plan.subtasks, results)):
        tag = " — BLOCKED, needs a user answer" if i in blocked else ""
        parts.append(
            f"\n--- sub-task {i + 1} [{task.tier}]:"
            f" {task.instruction}{tag} ---\n"
            f"{text or '(no report)'}"
        )
    parts.append(
        "\n--- end of reports ---\n\n"
        "Now write the user's reply."
    )
    return "\n".join(parts)


def merge_fallback(plan: Plan, results: list[str]) -> str:
    """Concatenate worker reports when the merge model is unavailable.

    Returns an empty string when no worker had anything to say; the
    caller turns that into a user-facing failure. Minting a placeholder
    here would let "nobody answered" reach the user dressed as an answer.
    """
    if len(results) == 1:
        return results[0]
    parts: list[str] = []
    for i, (task, text) in enumerate(zip(plan.subtasks, results)):
        if not text:
            continue
        parts.append(f"**{i + 1}. {task.instruction}**\n\n{text}")
    return "\n\n".join(parts)
