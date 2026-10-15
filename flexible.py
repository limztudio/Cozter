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

import hashlib
import json
import re
from collections.abc import Collection
from dataclasses import dataclass

from .utils import extract_marker_block

BACKEND_NAME = "flexible"

# Difficulty tiers, cheapest first.
TIERS: tuple[str, ...] = ("low", "mid", "high")

# Plan-failure tier: strongest.
FALLBACK_TIER = "high"

TIER_DESCRIPTIONS = {
    "low": "Straightforward, well-scoped work with clear intent",
    "mid": "Some reasoning needed, but the problem stays bounded",
    "high": (
        "Only for ambiguity, complex logic, or deeper system understanding"
    ),
}

# Cap fan-out: each sub-task is a turn.
MAX_SUBTASKS = 12
MAX_FOLLOWUP_TASKS_PER_REPORT = len(TIERS)

# Work cap: 3600s.
PLAN_TIMEOUT: float | None = 3600.0
MERGE_TIMEOUT: float | None = 3600.0

# Continue-judge bounds per backend per turn.
JUDGE_MAX_CONTINUES = 3
JUDGE_TIMEOUT: float | None = 3600.0

# Auto-chain bounds for judge-exhausted CONTINUE.
MAX_AUTO_CHAIN_TURNS = 3

# The rubric the planner grades each sub-task against.
_RUBRIC = (
    "low  - straightforward, well-scoped work.\n"
    "mid  - some reasoning, but bounded.\n"
    "high - ONLY ambiguity, complex logic, or deep system needs."
)

_PLANNER_RULES = (
    "Planner for a multi-agent assistant. Split the request below into"
    " the fewest sub-tasks covering it; grade each low/mid/high:\n\n"
    f"{_RUBRIC}\n\n"
    "Rules:\n"
    "- Simple request = ONE sub-task.\n"
    "- For an audit followed by repairs, group independent inspection scopes first,"
    " then use one cohesive repair-and-verification task for the related findings."
    " Do not split enumeration, implementation, tests, and documentation into"
    " separate workers when one worker can complete the related change.\n"
    "- Mark a task [parallel-read] ONLY when it is read-only inspection AND"
    " independent of every other consecutive task with that marker. Marked tasks"
    " may use completed earlier reports, but cannot depend on another marked"
    " task in the same consecutive group. Unmarked tasks run in order. Never"
    " mark edits, builds, tests, process launches, repairs, commits, or pushes.\n"
    "- Saved checkpoints and earlier reports identify completed work. Exclude"
    " completed actions from the new plan; revalidate only concrete evidence"
    " needed for the remaining requested scope, without repeating whole audits.\n"
    "- Plan only the checkpoint's unfinished obligations. For example, when"
    " builds and tests are verified and only commit/push remains, plan the"
    " remaining commit/push and necessary targeted checks, not another audit"
    " or rebuild. Reuse verified evidence when relevant inputs and environment"
    " are unchanged; final verification still covers the full original scope.\n"
    f"- At most {MAX_SUBTASKS} sub-tasks, ordered (each needs only earlier"
    " results). If more scope remains, the plan still lists it as an"
    " explicit PARTIAL + remainder note after the numbered tasks.\n"
    "- Progress is Thinking... status only (done/total may grow as"
    " follow-ups join); never a chat reply; DONE ping only at the end.\n"
    "- Grade honestly: over-grading wastes the strong model, under-grading"
    " strands hard work on a weak one.\n"
    "- Self-contained: the worker sees only the user message, the plan,"
    " and earlier reports.\n"
    "- Cover everything: when the request says all/entire/every/whole, the"
    " plan must cover every item with no gaps; prefer concrete enumerated"
    " targets over one vague 'handle all' task.\n"
    "- Doc-following: request pointing at a file/doc/standard = split so"
    " every rule/section is covered; never one vague 'follow the doc' task"
    " hiding dozens of rules.\n"
    "- Build/verify requests (buildable, no error/warning, no runtime"
    " warning): keep enumeration, canonical build/test, log inspection,"
    " repair, re-run, and runtime verification in one cohesive task for"
    " each target or related target group. Split only when distinct scope"
    " or difficulty requires another worker, not merely for workflow stages."
    " Require the 3600s cap (cancel still stops instantly; capture to file),"
    " inspect full logs, classify actual unexpected failures, fix and verify them,"
    " then runtime launch/exercise/log-check. Each task states evidence"
    " required (commands + exit codes + actual failures), never one vague 'check build' task.\n"
    "- Test names, expected negative-path diagnostics, platform skips, and historical"
    " log entries are not active defects. Missing unconfigured providers or other"
    " unavailable environments are stated limitations; do not add provisioning or"
    " cross-platform verification unless the user explicitly requested it.\n"
    "- Each instruction states its done-criteria and requires the worker to"
    " verify no leftovers remain with evidence.\n"
    "- No tools; plan from the text.\n"
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
    "For independent read-only inspection only, put [parallel-read] immediately"
    " after the tier, for example: 1. [low] [parallel-read] inspect existing logs."
    " Omit that marker for every task that changes or executes anything.\n"
)

_PLANNER_QUESTION_RULE = (
    "Too ambiguous (guessing wastes real work)? Skip the plan, ask one"
    " short question instead:\n\n"
    "[QUESTION]\n"
    "your one question\n"
    "[/QUESTION]\n\n"
    "Prefer planning; ask only when stuck.\n"
)

_MERGE_RULES = (
    "Merge the worker reports below into the single reply the user sees.\n\n"
    "Rules:\n"
    "- Answer directly, outcome first, as the assistant who did the work"
    " (never mention plans/workers/tiers; never emit"
    " 'Next turn should:' or any other handoff/planning note"
    " - carry leftover work inside this reply only).\n"
    "- Keep concrete results (code, paths, commands, numbers, errors).\n"
    "- For all/entire/every requests, check every plan item is covered by"
    " the reports before claiming done.\n"
    "- Never claim full completion when any report is partial, missing, or"
    " failed: say what finished, what did not, and what remains"
    " (final reply only; progress stays as Thinking... status).\n"
    "- Doc/standard requests: check every rule/section in the plan is"
    " covered by the reports; never accept first/last-few as full — say"
    " PARTIAL + remainder when coverage is unclear.\n"
    "- A bounded report excerpt may omit its middle; never treat this preview"
    " as full content. Omission alone does not imply unfinished work. Claim"
    " completion only when the retained scope and verification evidence"
    " support it; missing evidence = PARTIAL + remainder.\n"
    "- Build/verify requests: claim clean only when every report lists its"
    " commands + exit codes + error/warning counts from full (not preview)"
    " logs with no unexpected failures; distinguish test names, expected negative-path"
    " diagnostics, platform skips, and historical entries from active defects.\n"
    "- User's language. No tool calls; work is done.\n"
)

# The merge writes the reply; only it can stop and wait.
_MERGE_QUESTION_RULE = (
    "- \"[[await]]\" on its own line only for a blocking question."
    " Optional offers: no marker.\n"
)

# An unavailable prerequisite retains work and schedules recovery, not approval.
_MERGE_BLOCKED_RULE = (
    "- A BLOCKED report leaves the job unfinished. Preserve completed evidence"
    " and name the exact missing prerequisite plus an autonomous recovery step"
    " or availability check. Do not ask permission merely to continue routine"
    " work already authorized by the request and configured permissions."
    " Use \"[[await]]\" only for specific information the user must supply,"
    " after finishing everything else possible; never use it for routine approval.\n"
)

# Continue-judge: finish, continue actionable work, or retain a recoverable prerequisite.
_JUDGE_RULES = (
    "Continue-judge: decide whether the draft below fully answers the"
    " user's request.\n\n"
    "Rules:\n"
    "- Read the original request, then the draft. DONE only when the requested"
    " work is complete with evidence. Incidental expected skips and unavailable"
    " checks outside the requested scope may be reported as limitations without"
    " preventing DONE.\n"
    "- CONTINUE for concrete missing requested work achievable in the current"
    " environment. Finish every remaining independently actionable item before"
    " declaring an external blocker.\n"
    "- Use existing authorization and configured permissions for routine recovery"
    " within the requested scope. When full permission already authorizes the"
    " necessary action, continue without another approval. Never ask 'can I"
    " continue' merely to resume authorized work.\n"
    "- BLOCKED only when a necessary part of the explicit original scope is"
    " currently unavailable after all independently actionable work is complete."
    " State the evidence and exact missing prerequisite in MISSING; NEXT should"
    " give an autonomous recovery step or a precise availability condition to"
    " recheck. Try authorized remediation rather than asking the user to approve"
    " continuation. For example, required desktop interaction cannot run in a"
    " non-interactive Session 0; recover the authorized interactive environment"
    " if possible, otherwise recheck its availability. BLOCKED leaves the job"
    " unfinished; do not mark it done.\n"
    "- BLOCKED retains progress for automatic recovery and availability rechecks"
    " with backoff. Do not repeat completed audits, builds, or expensive work"
    " merely because the prerequisite is still unavailable.\n"
    "- Missing credentials or genuinely unknown user information must be named"
    " honestly; never invent credentials or claim access that is absent. Finish"
    " all possible work and recheck availability. An explicit [[await]] asks for"
    " necessary information, not permission to simply continue.\n"
    "- A bounded draft excerpt may omit its middle. Omission alone is not"
    " evidence of unfinished work. CONTINUE requires a concrete missing"
    " requested item, not a demand to regenerate omitted text.\n"
    "- Never mention plans/workers/tiers; never emit 'Next turn should:'.\n"
    "- No tools; decide from the text.\n"
    "- Do not CONTINUE for expected negative-path test logs, test names, platform"
    " skips, historical failures already repaired, or missing unconfigured providers."
    " Report these limitations without expanding scope, unless explicitly requested.\n"
    "- Reply in exactly this format, nothing else:\n\n"
    "[VERDICT]\n"
    "DONE\n"
    "[/VERDICT]\n\n"
    "or:\n\n"
    "[VERDICT]\n"
    "CONTINUE\n"
    "[/VERDICT]\n"
    "[MISSING]\n"
    "one checklist line still uncovered\n"
    "[/MISSING]\n"
    "[NEXT]\n"
    "one self-contained follow-up instruction for the worker\n"
    "[/NEXT]\n\n"
    "or:\n\n"
    "[VERDICT]\n"
    "BLOCKED\n"
    "[/VERDICT]\n"
    "[MISSING]\n"
    "necessary requested scope, evidence it cannot proceed, and exact unavailable prerequisite\n"
    "[/MISSING]\n"
    "[NEXT]\n"
    "autonomous recovery step or exact availability condition to recheck (this block is optional)\n"
    "[/NEXT]\n"
)


# Reports are re-sent downstream; cap each.
_REPORT_MAX_CHARS = 6_000
_REPORT_TRUNCATION_MARKER = (
    "\n… [middle omitted from bounded report excerpt — never treat this preview"
    " as full content; omission alone does not imply unfinished work]\n"
)

# Judge and continuation excerpts preserve final evidence as well as context.
_DRAFT_MAX_CHARS = 12_000
_DRAFT_OMISSION_MARKER = (
    "\n… [middle omitted from bounded draft excerpt; omission alone does not"
    " imply unfinished work]\n"
)


def _clip_excerpt_middle(text: str, limit: int, marker: str) -> str:
    """Retain both ends of an excerpt around its omission marker."""
    remaining = limit - len(marker)
    head = remaining // 2
    tail = remaining - head
    return text[:head] + marker + text[-tail:]


def _bounded_draft_excerpt(text: str, *, limit: int = _DRAFT_MAX_CHARS) -> str:
    """Keep a draft's opening and final evidence without adding new work."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(_DRAFT_OMISSION_MARKER):
        return "…" + text[-(limit - 1):] if limit > 1 else "…"
    return _clip_excerpt_middle(text, limit, _DRAFT_OMISSION_MARKER)


def _truncate_report(text: str) -> str:
    """Bound a report while retaining its opening and final evidence."""
    if len(text) <= _REPORT_MAX_CHARS:
        return text
    return _clip_excerpt_middle(text, _REPORT_MAX_CHARS, _REPORT_TRUNCATION_MARKER)


@dataclass(frozen=True)
class Subtask:
    """One planned unit of work and the difficulty tier it routes to."""
    tier: str
    instruction: str
    read_only: bool = False
    independent: bool = False


@dataclass(frozen=True)
class Plan:
    """A parsed planner reply.

    Exactly one of *question* and *subtasks* drives the turn: a question
    short-circuits into a user-facing ask, otherwise the sub-tasks run.
    """
    understanding: str
    subtasks: tuple[Subtask, ...]
    question: str | None = None


_FOLLOWUP_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?"
    r"\[\s*followup\s*:\s*(?P<tier>[A-Za-z]+)\s*\]"
    r"\s*(?P<instruction>.+?)\s*$",
    re.IGNORECASE,
)

_TIER_ALIASES = {"medium": "mid", "med": "mid", "middle": "mid"}

_PLAN_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?"      # optional bullet / "3." numbering
    r"\[\s*(?P<tier>[A-Za-z]+)\s*\]\s*"     # [high]
    r"(?P<instruction>.+?)\s*$",
    re.IGNORECASE,
)

_PARALLEL_READ_RE = re.compile(r"^\[\s*parallel-read\s*\]\s*", re.IGNORECASE)

_REMAINDER_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?(?:PARTIAL\b|remainder\b)", re.IGNORECASE)


def normalize_tier(value: str) -> str | None:
    """Return the canonical tier name for *value*, or None if unknown."""
    if not isinstance(value, str):
        return None
    tier = value.strip().lower()
    tier = _TIER_ALIASES.get(tier, tier)
    return tier if tier in TIERS else None


def _parse_subtask_line(
    line: str, pattern: re.Pattern[str], *, allow_parallel_read: bool = False,
) -> Subtask | None:
    """Parse one tier/instruction line shared by plans and follow-ups."""
    match = pattern.match(line)
    if match is None:
        return None
    tier = normalize_tier(match.group("tier"))
    instruction = match.group("instruction").strip()
    marker = _PARALLEL_READ_RE.match(instruction)
    if marker is not None:
        instruction = instruction[marker.end():].strip()
    if tier is None or not instruction:
        return None
    marked = allow_parallel_read and marker is not None
    return Subtask(tier=tier, instruction=instruction, read_only=marked, independent=marked)


def parse_followups(report: str) -> list[Subtask]:
    """Parse worker-reported ``[followup:<tier>] <instruction>`` lines."""
    found: list[Subtask] = []
    seen: set[Subtask] = set()
    if not isinstance(report, str) or not report:
        return found
    for line in report.splitlines():
        entry = _parse_subtask_line(line, _FOLLOWUP_LINE_RE)
        if entry is None:
            continue
        if entry not in seen:
            seen.add(entry)
            found.append(entry)
    return found


def instruction_identity(instruction: str) -> str:
    """Normalize formatting, preserving case and whitespace in quoted targets."""
    pieces = re.split(r"(`[^`]*`|'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")", instruction)
    normalized = "".join(piece if index % 2 else re.sub(r"\s+", " ", piece)
                         for index, piece in enumerate(pieces)).strip()
    # A polite request prefix does not create another obligation. Everything
    # identifying a target or action remains case-sensitive and unchanged.
    return re.sub(r"^Please\s+", "", normalized)


def state_signature(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def task_identity(task: Subtask) -> str:
    return task.tier + ":" + instruction_identity(task.instruction)


def task_record(task: Subtask) -> dict:
    return {"tier": task.tier, "instruction": task.instruction,
            "read_only": task.read_only, "independent": task.independent}


def restore_tasks(value: object, *, limit: int | None = MAX_SUBTASKS * 2) -> list[Subtask] | None:
    if not isinstance(value, list) or (limit is not None and len(value) > limit):
        return None
    tasks = []
    for row in value:
        if (not isinstance(row, dict) or row.get("tier") not in TIERS
                or not isinstance(row.get("instruction"), str) or not row["instruction"].strip()
                or not isinstance(row.get("read_only", False), bool)
                or not isinstance(row.get("independent", False), bool)):
            return None
        tasks.append(Subtask(row["tier"], row["instruction"],
                             row.get("read_only", False), row.get("independent", False)))
    return tasks


def coalesce_followups(report: str, known: set[str]) -> tuple[list[Subtask], list[Subtask], dict[str, list[str]]]:
    """Bound new workers, grouping consecutive obligations of the same tier.

    Only identical instructions (apart from harmless formatting/politeness)
    are deduplicated. Grouped obligations remain verbatim and retain their
    requested tier, so consolidation neither changes providers nor loses scope.
    """
    grouped: list[list[Subtask]] = []
    added: set[str] = set()
    for task in parse_followups(report):
        identity = task_identity(task)
        if identity in known or identity in added:
            continue
        added.add(identity)
        if grouped and grouped[-1][0].tier == task.tier:
            grouped[-1].append(task)
        else:
            grouped.append([task])
    tasks = []
    members = {}
    for obligations in grouped:
        tier = obligations[0].tier
        if len(obligations) == 1:
            tasks.append(obligations[0])
        else:
            instruction = (
                "Finish these remaining obligations together as one cohesive task. "
                "Retain completed repairs and evidence; do not restart the whole audit. "
                "Complete and verify every item below:\n"
                + "\n".join(f"{index}. {task.instruction}"
                            for index, task in enumerate(obligations, 1))
            )
            tasks.append(Subtask(tier, instruction))
        members[task_identity(tasks[-1])] = [task_identity(task) for task in obligations]
    return tasks[:MAX_FOLLOWUP_TASKS_PER_REPORT], tasks[MAX_FOLLOWUP_TASKS_PER_REPORT:], members


def fallback_plan(request: str) -> Plan:
    """The whole request as one sub-task on the strongest tier."""
    if isinstance(request, str):
        instruction = request.strip() or "(empty request)"
    else:
        instruction = str(request) or "(empty request)"
    return Plan(
        understanding="",
        subtasks=(Subtask(tier=FALLBACK_TIER, instruction=instruction),),
    )


def parse_plan(raw: str, request: str) -> Plan:
    """Parse a planner reply, falling back to a single high-tier task.

    Tolerant by design: the planner is a cheap model, so a missing
    ``[PLAN]`` wrapper, stray prose, or a ``[medium]`` tier should not
    cost the user their turn.
    """
    if not isinstance(raw, str):
        return fallback_plan(request)
    if not raw or not raw.strip():
        return fallback_plan(request)

    question = extract_marker_block(raw, "QUESTION")
    if question:
        return Plan(
            understanding=extract_marker_block(raw, "UNDERSTANDING") or "",
            subtasks=(),
            question=question,
        )

    # Scan the whole reply when the planner forgot the wrapper.
    block = extract_marker_block(raw, "PLAN") or raw
    subtasks: list[Subtask] = []
    seen: set[tuple[str, str]] = set()
    capped = False
    for line in block.splitlines():
        subtask = _parse_subtask_line(line, _PLAN_LINE_RE, allow_parallel_read=True)
        if subtask is None:
            continue
        key = (subtask.tier, subtask.instruction)
        if key in seen:
            continue
        if len(subtasks) == MAX_SUBTASKS:
            capped = True
            break
        seen.add(key)
        subtasks.append(subtask)

    if not subtasks:
        return fallback_plan(request)
    understanding = extract_marker_block(raw, "UNDERSTANDING") or ""
    # The planner may put its explicit remainder note after the numbered
    # tasks or outside [PLAN]; keep that obligation in downstream prompts.
    for line in raw.splitlines():
        if _REMAINDER_LINE_RE.match(line) and line.strip() not in understanding:
            understanding = "\n".join(filter(None, (understanding, line.strip())))
    if capped and "PARTIAL" not in understanding:
        note = (
            f"PARTIAL: plan capped at {MAX_SUBTASKS} sub-tasks;"
            " remainder beyond the cap is still uncovered."
        )
        understanding = f"{understanding}\n{note}".strip() if understanding else note
    return Plan(
        understanding=understanding,
        subtasks=tuple(subtasks),
    )


def build_plan_prompt(context: str, *, collaborative: bool) -> str:
    """Prompt the summary agent to understand and split the request."""
    parts = [_PLANNER_RULES]
    if collaborative:
        parts.append(_PLANNER_QUESTION_RULE)
    parts.append(_PLANNER_FORMAT)
    parts.append("--- input ---")
    parts.append(context)
    return "\n".join(parts)


def _render_plan(plan: Plan, *, current: int | None = None) -> str:
    lines = []
    for task_index, task in enumerate(plan.subtasks):
        marker = "  <- your task" if task_index == current else ""
        instruction = task.instruction
        # The actionable current instruction is repeated after the reports.
        # Keep it there in full without also amplifying the plan overview.
        placeholder = "(full instruction below)"
        if task_index == current and len(instruction) > len(placeholder):
            instruction = placeholder
        lines.append(
            f"  {task_index + 1}. [{task.tier}]"
            + (" [parallel-read]" if task.read_only and task.independent else "")
            + f" {instruction}{marker}"
        )
    return "\n".join(lines)


def build_subtask_prompt(
    context: str, plan: Plan, index: int, results: list[str],
    *, previous_draft: str | None = None,
) -> str:
    """Prompt for the worker running sub-task *index* of *plan*.

    *context* is the bare user request, not the full history — the
    planner saw the history and wrote self-contained instructions.
    When the request carries "[... attachment saved to: ...]" markers
    they ride along verbatim so a vision-capable tier backend still
    resolves the image bytes natively. A continuation also carries a bounded
    *previous_draft* excerpt so workers retain already-completed work and its
    final evidence.
    """
    task = plan.subtasks[index]
    parts = [context, ""]
    parts.append(
        f"[Sub-task {index + 1}/{len(plan.subtasks)}: {task.tier}]"
        " — do ONLY this one; others have their own workers."
    )
    if plan.understanding:
        parts.append(f"Goal: {plan.understanding}")
    parts.append("\nPlan:")
    parts.append(_render_plan(plan, current=index))

    if results:
        parts.append("\nEarlier reports:")
        for result_index, text in enumerate(results):
            parts.append(
                f"\n--- report {result_index + 1} ---\n"
                f"{_truncate_report(text)}"
            )

    if previous_draft:
        parts.append(
            "\nPrevious draft and evidence from this turn:\n"
            + _bounded_draft_excerpt(previous_draft)
            + "\nRetain completed work and verify its current state. Address"
            " the remaining instructions without repeating completed actions."
        )

    if task.read_only:
        parts.append(
            f"\nYour task: {task.instruction}\n"
            "Inspection only: read existing files, logs, and metadata within this"
            " task's scope. Do not write, edit, delete, apply fixes, create files,"
            " build, run tests, launch or stop processes, commit, or push."
            " Do not expand into the full original request. Use only inspection"
            " tools; do not run commands that can change workspace or runtime state.\n"
            "Consecutive [parallel-read] tasks may run together. You can use the"
            " earlier reports above, but cannot rely on another concurrent task's"
            " findings or file changes.\n"
            "Inspect every item in your assigned scope, then report exact findings"
            " and existing evidence, including file paths and relevant locations."
            " Give precise repair recommendations and unresolved questions to the"
            " later writer; leave implementation, execution, and verification to"
            " that writer. Do not request redundant workflow stages or repeat"
            " completed work from earlier reports or saved checkpoints.\n"
            "If requested inspection is unfinished, say PARTIAL and identify the"
            " exact remainder. Expected test diagnostics, platform skips, historical"
            " failures, and unavailable services are not new defects. Progress is"
            " Thinking... status only."
        )
        return "\n".join(parts)

    parts.append(
        f"\nYour task: {task.instruction}\n"
        "Finish ALL of it - never a sample. If it says all/entire/every,"
        " find every item first (glob/grep/list), handle each one, then"
        " re-check for leftovers.\n"
        "If it points at a doc/standard, enumerate every rule/section first"
        " (read fully via offset chunks + grep), apply each, never stop"
        " after first/last few.\n"
        "Build/verify, when needed and not already covered by still-valid evidence:"
        " enumerate targets, run canonical build/test with 3600s cap"
        " (cancel still stops instantly; capture output to file), grep FULL logs for"
        " unexpected errors/warnings/exceptions (never eyeball tail only), classify"
        " each hit, repair actual failures and verify; runtime means launch +"
        " exercise paths + check logs, not just compile.\n"
        "For independent compilation or test targets, use the project's supported"
        " jobs/parallel flag with a resource bound appropriate to available CPU"
        " and memory. Never overlap configure/build commands that share a generated"
        " tree or dependency metadata, including different configurations that use"
        " the same generated state.\n"
        "Reuse verified completed build/test evidence when relevant inputs and"
        " environment are unchanged; rerun only invalidated checks. A handoff or"
        " resume alone is not a reason to repeat whole audits or builds. Complete"
        " only the checkpoint's remaining obligations, such as commit/push after"
        " verified builds. Final verification must still cover the original scope;"
        " distinguish valid retained evidence from anything unverified.\n"
        "Test names, expected negative-path diagnostics, platform skips, and historical"
        " log entries are not current defects. State unavailable environment or"
        " unconfigured-provider verification as limitations; do not provision services"
        " or create cross-platform follow-ups unless explicitly requested.\n"
        "Use tools, not description. Report per-item: what you did/found,"
        " what you skipped or could not do, what the next worker needs"
        " (paths, commands, exit codes, error/warning counts).\n"
        "Be honest: say PARTIAL and list what remains if anything is"
        " unfinished or unverified; never report done when work remains"
        " or evidence is missing. Progress is Thinking... status only; notify"
        " DONE only at the very end.\n"
        "New work discovered while doing this task: list each item on its own"
        " line as [followup:<low|mid|high>] <instruction> (self-contained, one"
        " line each). Only genuine leftovers found inside this task - never"
        " restate the plan or completed/queued scope. Prefer finishing related"
        " repairs and their verification together; do not create speculative"
        " audits or separate follow-ups for each workflow step. Related leftovers"
        " of one consecutive tier group will run in one cohesive worker, preserving every item."
        " Never write 'Next turn should:' or any other"
        " handoff/planning note; the followup lines above are the only"
        " carry-forward mechanism."
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
    parts = [rules, "--- input ---", context, ""]
    if plan.understanding:
        parts.append(f"Goal: {plan.understanding}")
    parts.append("\nPlan:")
    parts.append(_render_plan(plan))
    parts.append("\n--- reports ---")
    # Lengths may differ; index both sides instead of zip().
    report_count = max(len(plan.subtasks), len(results))
    for report_index in range(report_count):
        task = plan.subtasks[report_index] if report_index < len(plan.subtasks) else None
        text = results[report_index] if report_index < len(results) else ""
        tag = " [BLOCKED]" if report_index in blocked else ""
        header = (
            f"\n--- {report_index + 1} [{task.tier}]:"
            f" plan item {report_index + 1}{tag} ---\n"
            if task is not None
            else f"\n--- {report_index + 1} [unknown]: (no sub-task){tag} ---\n"
        )
        parts.append(
            header + (_truncate_report(text) if text else "(no report)"),
        )
    parts.append("\n--- end ---\n\nWrite the user's reply.")
    return "\n".join(parts)


def merge_fallback(plan: Plan, results: list[str]) -> str:
    """Concatenate worker reports when the merge model is unavailable.

    Returns an empty string when no worker had anything to say; the
    caller turns that into a user-facing failure. Minting a placeholder
    here would let "nobody answered" reach the user dressed as an answer.
    """
    if not any(results):
        return ""
    if len(results) == 1 and len(plan.subtasks) <= 1:
        return results[0]
    parts: list[str] = []
    missing: list[str] = []
    # Index both sides so mismatches can't drop reports.
    for coverage_index in range(max(len(plan.subtasks), len(results))):
        text = results[coverage_index] if coverage_index < len(results) else ""
        if not text:
            if coverage_index < len(plan.subtasks):
                missing.append(
                    f"{coverage_index + 1}. {plan.subtasks[coverage_index].instruction}"
                )
            continue
        if coverage_index < len(plan.subtasks):
            instruction = plan.subtasks[coverage_index].instruction
        else:
            instruction = "(no sub-task)"
        parts.append(f"**{coverage_index + 1}. {instruction}**\n\n{text}")
    if missing:
        parts.append(
            "PARTIAL: no report for the following task(s); "
            "their completion remains unverified:\n" + "\n".join(missing)
        )
    return "\n\n".join(parts)


@dataclass(frozen=True)
class JudgeVerdict:
    """A judge decision separating completion, actionable work and external blockers."""

    should_continue: bool
    missing: str = ""
    next_instruction: str = ""
    verified: bool = True
    blocked: bool = False


def build_judge_prompt(request: str, draft: str) -> str:
    """Prompt the summary agent to judge one draft answer.

    *request* is the bare user request; *draft* is the candidate reply.
    The judge never does the work itself: it returns DONE, a CONTINUE
    step, or BLOCKED with the prerequisite for automatic recovery/rechecking.
    """
    parts = [
        _JUDGE_RULES,
        "--- request ---",
        request if isinstance(request, str) and request else "(empty)",
        "",
        "--- draft ---",
        _bounded_draft_excerpt(draft) if draft else "(no draft)",
        "",
        "--- end ---",
    ]
    return "\n".join(parts)


def parse_judge(raw: str) -> JudgeVerdict:
    """Recognize explicit decisions; malformed replies never verify completion."""
    if not isinstance(raw, str) or not raw.strip():
        return JudgeVerdict(should_continue=False, verified=False)
    verdict = (extract_marker_block(raw, "VERDICT") or "").strip().upper()
    if verdict == "DONE":
        return JudgeVerdict(should_continue=False)
    if verdict not in {"CONTINUE", "BLOCKED"}:
        return JudgeVerdict(should_continue=False, verified=False)
    missing = (extract_marker_block(raw, "MISSING") or "").strip()
    followup = (extract_marker_block(raw, "NEXT") or "").strip()
    if verdict == "BLOCKED":
        return JudgeVerdict(
            should_continue=False, missing=missing, next_instruction=followup,
            verified=False, blocked=bool(missing),
        )
    if not followup:
        return JudgeVerdict(should_continue=False, verified=False)
    return JudgeVerdict(
        should_continue=True, missing=missing, next_instruction=followup,
    )
