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
# over-eager planner would otherwise turn one message into unbounded runs.
# Twelve leaves room for genuinely broad work without making an accidental
# plan prohibitively slow or expensive.
MAX_SUBTASKS = 12

PLAN_TIMEOUT = 120  # seconds; on timeout the planner falls back to one task
MERGE_TIMEOUT = 180  # seconds; on timeout the worker reports are concatenated

# The user-facing rubric the planner grades each sub-task against.
_RUBRIC = (
    "low  - straightforward, well-scoped work with clear intent.\n"
    "mid  - some reasoning is required, but the problem stays bounded.\n"
    "high - ONLY for ambiguity, complex logic, or deeper system "
    "understanding."
)

_PLANNER_RULES = (
    "Planner for a multi-agent assistant. Split the request below into"
    " the fewest sub-tasks covering it; grade each low/mid/high so it"
    " routes to a right-sized model:\n\n"
    f"{_RUBRIC}\n\n"
    "Rules:\n"
    "- Simple request = ONE sub-task; no busywork.\n"
    f"- At most {MAX_SUBTASKS} sub-tasks, ordered (each needs only earlier"
    " results; they run in order).\n"
    "- Grade honestly: over-grading wastes the strong model, under-grading"
    " strands hard work on a weak one.\n"
    "- Each sub-task self-contained: the worker sees only the user message,"
    " this plan, and earlier reports — never full history.\n"
    "- No tools/file reads; plan from the text below.\n"
)

_PLANNER_FORMAT = (
    "Reply in exactly this format, nothing else:\n\n"
    "[UNDERSTANDING]\n"
    "1-2 sentences: what the user wants.\n"
    "[/UNDERSTANDING]\n"
    "[PLAN]\n"
    "1. [low|mid|high] first sub-task instruction\n"
    "2. [low|mid|high] second sub-task instruction\n"
    "[/PLAN]\n"
)

_PLANNER_QUESTION_RULE = (
    "Too ambiguous to plan (guessing wastes real work)? Skip the plan,"
    " ask one short question instead:\n\n"
    "[QUESTION]\n"
    "your one question\n"
    "[/QUESTION]\n\n"
    "Prefer planning; ask only when stuck.\n"
)

_MERGE_RULES = (
    "Merge the worker reports below into the single reply the user sees.\n\n"
    "Rules:\n"
    "- Answer directly, outcome first, as the assistant who did the work"
    " (never mention plans/workers/tiers).\n"
    "- Keep concrete results (code, paths, commands, numbers, errors);"
    " report failures plainly.\n"
    "- User's language. No tool calls; work is done.\n"
)

# The merge step writes the reply the user actually reads, so it is the
# only step downstream of the planner that can stop the turn and wait.
# Without this the pipeline can end on a blocking question and still let
# the queue drain straight past it, leaving the user's answer to land as
# an unrelated new turn.
_MERGE_QUESTION_RULE = (
    "- \"[[await]]\" on its own line only for a blocking question (next"
    " message = the answer). Optional offers: no marker.\n"
)

# Workers run under the autonomy policy, so one that stops to ask has
# already established the turn cannot finish without the user. Tell the
# merge outright instead of leaving it to infer that from the report text.
_MERGE_BLOCKED_RULE = (
    "- A BLOCKED-marked report needs a user answer: end with its question"
    " plus \"[[await]]\" on its own line.\n"
)


# Worker reports are unbounded model output, and each one is re-sent to
# every later worker plus the merge step. Cap each so N workers cost at
# most N * cap instead of N * anything.
_REPORT_MAX_CHARS = 6_000
_REPORT_TRUNCATION_MARKER = "\n… [report truncated]"


def _truncate_report(text: str) -> str:
    """Bound one worker report for re-prompting downstream."""
    if len(text) <= _REPORT_MAX_CHARS:
        return text
    return text[:_REPORT_MAX_CHARS] + _REPORT_TRUNCATION_MARKER


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
    """Prompt for the worker running sub-task *index* of *plan*.

    *context* is the bare user request, not the full history — the
    planner saw the history and wrote self-contained instructions.
    """
    task = plan.subtasks[index]
    parts = [context, ""]
    parts.append(
        f"[Sub-task {index + 1}/{len(plan.subtasks)}: {task.tier}]"
    )
    parts.append(
        "One worker in a pipeline answering the message above; sub-tasks"
        " run in order."
    )
    if plan.understanding:
        parts.append(f"\nGoal: {plan.understanding}")
    parts.append("\nPlan:")
    parts.append(_render_plan(plan, current=index))

    if results:
        parts.append("\nEarlier workers reported:")
        for i, text in enumerate(results):
            parts.append(
                f"\n--- sub-task {i + 1} result ---\n"
                f"{_truncate_report(text)}"
            )

    parts.append(
        f"\nDo ONLY sub-task {index + 1}: {task.instruction}\n"
        "Others have their own workers; duplicated work is discarded."
        " Use tools to do it, not describe it. Report concisely for the"
        " agent writing the final reply: what you did/found, what the"
        " next worker needs (paths, commands, results, no pleasantries)."
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
        parts.append(f"Goal: {plan.understanding}")
    parts.append("\nPlan carried out:")
    parts.append(_render_plan(plan))
    parts.append("\n--- worker reports ---")
    for i, (task, text) in enumerate(zip(plan.subtasks, results)):
        tag = " — BLOCKED, needs a user answer" if i in blocked else ""
        parts.append(
            f"\n--- sub-task {i + 1} [{task.tier}]:"
            f" {task.instruction}{tag} ---\n"
            f"{_truncate_report(text) if text else '(no report)'}"
        )
    parts.append("\n--- end of reports ---\n\nWrite the user's reply.")
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
