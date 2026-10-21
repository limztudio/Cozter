"""Flexible agent — choose an owner, consult, complete.

The ``flexible`` agent is not a CLI of its own: it is a meta-agent that
spends a cheap summary-model call to choose one whole-job owner and any
useful initial read-only specialists. The owner uses the configured
low/mid/high backend and retains implementation, verification, and completion.
Older saved multi-worker plans retain their ordered execution and merge.

This module holds the side-effect-free half — prompt construction and
output parsing. The orchestration loop lives in :mod:`agent`, which owns
the backend subprocess driver the workers run on.
"""

import json
import re
from collections.abc import Collection
from dataclasses import dataclass

from .utils import extract_marker_block

BACKEND_NAME = "flexible"

# Difficulty tiers, cheapest first.
TIERS: tuple[str, ...] = ("low", "mid", "high")

# Start with scoped inspection when difficulty is unknown.
FALLBACK_TIER = "low"

TIER_DESCRIPTIONS = {
    "low": "Default for bounded coding, debugging, refactoring, and verification",
    "mid": "Involved implementation with an understood approach and several constraints",
    "high": "New design, ambiguous causes, coupled invariants, or performance/correctness tradeoffs",
}

# Cap fan-out: each sub-task is a turn.
MAX_SUBTASKS = 12
MAX_FOLLOWUP_TASKS_PER_REPORT = len(TIERS)
MAX_TIER_LEASES = 3
MAX_INITIAL_SPECIALISTS = 2

# Continue-judge bounds per backend per turn.
JUDGE_MAX_CONTINUES = 3

# Shared by the planner and workers choosing focused help.
_RUBRIC = (
    "Choose the cheapest capable owner or specialist for the reasoning required."
    " Low is the broad default; the request itself can justify mid/high upfront:\n"
    "low  - default for bounded implementation, debugging, refactoring, test repairs,"
    " configuration, documentation, inspection, and build/runtime verification."
    " A capable low worker can inspect, implement, and verify a complete change"
    " across several files using existing patterns; low is not limited to mechanical edits.\n"
    "mid  - involved non-routine implementation or diagnosis with an understood"
    " approach: carry several clear constraints through the change and its checks."
    " Ordinary choices following existing patterns stay low.\n"
    "high - substantive new design or approach selection, ambiguous multi-cause"
    " diagnosis, refactors with coupled invariants, difficult algorithms, subtle"
    " concurrency, or performance/correctness tradeoffs. Choose high directly when"
    " this reasoning is central and tightly coupled to implementation and verification."
    " A failed lower-tier attempt or a known root cause is not required.\n"
    "File count, duration, broad wording, or the words all/fix/audit/optimise"
    " alone do not justify mid/high. Unfamiliar files alone can start with low"
    " inspection and continue in that owner when the path is clear. When the request"
    " already presents design uncertainty, multiple plausible interacting causes,"
    " or competing invariants, use high reasoning upfront rather than requiring"
    " lower-tier failures. A low/mid owner can consult high early for a focused"
    " decision, then retain the straightforward implementation. Do not force a mix of tiers."
)

_ESCALATION_RULES = (
    "Try the appropriate same-tier diagnosis, repair, and verification first."
    " If a reasoning limit prevents finishing this SAME obligation after inspecting"
    " evidence and attempting the relevant verification, request the next tier"
    " by ending your report with exactly one block, using one nonempty line per field:\n"
    "[ESCALATE]\n"
    "reason: the concrete reasoning difficulty this attempt could not resolve\n"
    "evidence: work attempted, checks performed, and the unresolved result\n"
    "remaining: the self-contained unfinished scope, retaining original constraints\n"
    "[/ESCALATE]\n"
    "Do not wrap the block in a Markdown code fence. Do not mark that obligation"
    " complete or duplicate it as a followup. Scoped inspection or a targeted check"
    " can establish the reasoning limit; do not make speculative edits just to qualify. The"
    " coordinator preserves completed evidence and advances only one tier at a"
    " time. At high, the block records unfinished reasoning; no higher tier exists."
    " Do not escalate for quota/auth/network errors,"
    " missing prerequisites, unavailable tools, permissions, or slow commands."
    " Report those conditions and their recovery evidence normally. Use existing"
    " authorization; do not ask permission merely to retry authorized work.\n"
)

_LEASE_RULES = (
    "For a focused analysis or read-only inspection question, you may borrow a"
    " helper from any OTHER tier without handing over your whole obligation."
    " For example, high can use low for routine inventory, and low/mid can consult"
    " high early for approach selection, competing causes, or an invariant before"
    " making speculative changes. No failed attempt is required for consultation."
    " Choose the cheapest capable helper"
    " when the saved work warrants the handoff; finish trivial queries directly."
    " Retain ownership of edits, verification, and completion. To yield, end your"
    " report with exactly this block, using one nonempty line per field:\n"
    "[LEASE]\n"
    "tier: low|mid|high\n"
    "reason: why this focused question benefits from the chosen helper\n"
    "task: a self-contained read-only question with exact scope and needed evidence\n"
    "[/LEASE]\n"
    "Do not wrap the block in a Markdown code fence. Stop issuing tools after"
    " the block. The coordinator runs the helper, then"
    " resumes you automatically with its result; no permission question is needed."
    " The helper never writes, edits, builds, runs tests, launches/stops processes,"
    " commits, or pushes. It reports evidence for you to assess and apply."
    f" At most {MAX_TIER_LEASES} lease requests are allowed per logical task; helpers cannot"
    " request their own leases or escalations. Use LEASE for a question while"
    " retaining the parent task, and ESCALATE only when the remaining whole"
    " obligation needs a stronger worker. Never emit both blocks in one report.\n"
)

_WORKER_ROUTING_RULES = (
    "Use the same tier policy for newly discovered work:\n"
    f"{_RUBRIC}\n"
    "An ordinary [followup:<low|mid|high>] is NEW scope discovered inside this task,"
    " not a stronger retry of unfinished work. Prefer finishing related work here;"
    " a previous high-tier worker does not make routine leftovers high.\n"
    f"{_ESCALATION_RULES}"
    f"{_LEASE_RULES}"
)

_OWNER_ROUTING_RULES = (
    "You remain the completion owner of the whole user request, including related"
    " obligations discovered during inspection, implementation, and verification."
    " Your brief and advisory reports never narrow the user's original scope."
    " Keep design, edits, tests, documentation, and authorized publication in this"
    " owner conversation; do not assign workflow stages to new writers. Finish"
    " discovered work here. If it must carry forward, list exact remaining items"
    " as [followup:<low|mid|high>] lines; they remain with this owner, and the tier"
    " label does not create a replacement writer. Use LEASE for focused advice or"
    " ESCALATE for a demonstrated whole-obligation reasoning limit.\n"
    f"{_RUBRIC}\n"
    f"{_ESCALATION_RULES}"
    f"{_LEASE_RULES}"
)

_PLANNER_RULES = (
    "Planner for a multi-agent assistant. Choose ONE persistent owner for the"
    " entire unfinished job, plus optional focused read-only specialists:\n\n"
    f"{_RUBRIC}\n\n"
    "Rules:\n"
    "- The owner retains the whole job through inspection, design, implementation,"
    " tests, documentation, publication when authorized, and newly discovered"
    " obligations. Never split workflow stages into different writers.\n"
    "- Add no specialist for routine work. When focused advice would materially"
    " improve an approach, ask high early about design, ambiguous causes, coupled"
    " invariants, or performance/correctness tradeoffs; do not wait for failed"
    " edits. A high owner can ask low for a useful routine inventory instead."
    " This is optional advice, not a mandatory high-to-low pipeline.\n"
    f"- At most {MAX_INITIAL_SPECIALISTS} initial specialists, each at a distinct"
    " tier different from the owner's. Their questions must be self-contained,"
    " read-only, and independent of each other. They run before the owner's first"
    " turn and count toward its bounded helper allowance. They cannot edit, build,"
    " test, launch/stop processes, commit, push, or request other helpers. The"
    " owner assesses their evidence and performs all changes and verification.\n"
    "- Saved checkpoints and earlier reports identify completed work. Exclude"
    " completed actions from the new plan; revalidate only concrete evidence"
    " needed for the remaining requested scope, without repeating whole audits.\n"
    "- Plan only the checkpoint's unfinished obligations. For example, when"
    " builds and tests are verified and only commit/push remains, plan the"
    " remaining commit/push and necessary targeted checks, not another audit"
    " or rebuild. Reuse verified evidence when relevant inputs and environment"
    " are unchanged; final verification still covers the full original scope.\n"
    "- The OWNER task is a self-contained whole-job brief, with every unfinished"
    " obligation and constraint. Advice questions never replace requested scope."
    " Do not drop difficult work merely because a specialist is unavailable.\n"
    "- Progress is Thinking... status only (done/total may grow as"
    " follow-ups join); never a chat reply; DONE ping only at the end.\n"
    "- Grade the actual reasoning needed. Low owners can complete coding and"
    " verification; mid handles involved work with a known approach; high can"
    " own design or ambiguous diagnosis directly, without prior lower-tier attempts.\n"
    "- On a continuation, grade only the unfinished obligation. Reuse earlier"
    " findings and verified work; a remaining known test command or commit/push"
    " normally stays low, even after a difficult original request. An owner can request"
    " a bounded next-tier escalation if actual evidence reveals harder reasoning.\n"
    "- Self-contained: the owner sees the user message, its whole-job brief,"
    " retained evidence, and specialist findings. It keeps ownership of follow-ups.\n"
    "- Cover everything: when the request says all/entire/every/whole, the"
    " plan must cover every item with no gaps; prefer concrete enumerated"
    " targets over one vague 'handle all' task.\n"
    "- Doc-following: request pointing at a file/doc/standard = explicitly retain"
    " every rule/section; never one vague 'follow the doc' brief"
    " hiding dozens of rules.\n"
    "- Build/verify requests (buildable, no error/warning, no runtime"
    " warning): keep enumeration, canonical build/test, log inspection,"
    " repair, re-run, and runtime verification as one cohesive task with the same"
    " owner for every target or related target group. Use focused advice when useful without"
    " handing implementation stages to new writers."
    " Run commands to completion (cancel still stops them; capture to file),"
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
    "Reply using these blocks, nothing else. UNDERSTANDING and SPECIALISTS are"
    " optional; OWNER is required, last, and outside Markdown code fences. Use"
    " one nonempty line per OWNER field:\n\n"
    "[UNDERSTANDING]\n"
    "1-2 sentences: what the user wants.\n"
    "[/UNDERSTANDING]\n"
    "[SPECIALISTS]\n"
    '[{"tier":"high","reason":"why focused advice helps",'
    '"task":"self-contained read-only question and required evidence"}]\n'
    "[/SPECIALISTS]\n"
    "[OWNER]\n"
    "tier: low|mid|high\n"
    "reason: why this tier can own the whole remaining job\n"
    "task: complete whole-job brief, retaining all requested scope and constraints\n"
    "[/OWNER]\n"
    "SPECIALISTS must be a JSON array with exact tier/reason/task string fields;"
    " use [] or omit the block when no advice is needed. Do not copy the example"
    " high specialist automatically. Its tier must differ from the owner's.\n"
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

# Continue-judge: finish, continue work, or keep a fixable prerequisite.
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
REPORT_TRUNCATION_MARKER = (
    "\n… [middle omitted from bounded report excerpt — never treat this preview"
    " as full content; omission alone does not imply unfinished work]\n"
)

# Judge and continuation excerpts preserve final evidence as well as context.
_DRAFT_MAX_CHARS = 12_000
DRAFT_OMISSION_MARKER = (
    "\n… [middle omitted from bounded draft excerpt; omission alone does not"
    " imply unfinished work]\n"
)


def _clip_excerpt_middle(text: str, limit: int, marker: str) -> str:
    """Retain both ends of an excerpt around its omission marker."""
    remaining = limit - len(marker)
    head = remaining // 2
    tail = remaining - head
    return text[:head] + marker + text[-tail:]


def bounded_draft_excerpt(text: str, *, limit: int = _DRAFT_MAX_CHARS) -> str:
    """Keep a draft's opening and final evidence without adding new work."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(DRAFT_OMISSION_MARKER):
        return "…" + text[-(limit - 1):] if limit > 1 else "…"
    return _clip_excerpt_middle(text, limit, DRAFT_OMISSION_MARKER)


def truncate_report(text: str) -> str:
    """Bound a report while retaining its opening and final evidence."""
    if len(text) <= _REPORT_MAX_CHARS:
        return text
    return _clip_excerpt_middle(text, _REPORT_MAX_CHARS, REPORT_TRUNCATION_MARKER)


@dataclass(frozen=True)
class Subtask:
    """One planned unit of work and the difficulty tier it routes to."""
    tier: str
    instruction: str
    read_only: bool = False
    independent: bool = False


@dataclass(frozen=True)
class Escalation:
    """Evidence and remaining scope for one capability escalation."""

    reason: str
    evidence: str
    remaining: str


@dataclass(frozen=True)
class Lease:
    """A focused question for a helper while the parent retains its task."""

    tier: str
    reason: str
    task: str


@dataclass(frozen=True)
class Plan:
    """A parsed planner reply.

    Exactly one of *question* and *subtasks* drives the turn: a question
    short-circuits into a user-facing ask, otherwise the sub-tasks run.
    """
    understanding: str
    subtasks: tuple[Subtask, ...]
    question: str | None = None
    owner_mode: bool = False
    specialists: tuple[Lease, ...] = ()
    owner_reason: str = ""


_FOLLOWUP_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?"
    r"\[\s*followup\s*:\s*(?P<tier>[A-Za-z]+)\s*\]"
    r"\s*(?P<instruction>.+?)\s*$",
    re.IGNORECASE,
)

_PLAN_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?"      # optional bullet / "3." numbering
    r"\[\s*(?P<tier>[A-Za-z]+)\s*\]\s*"     # [high]
    r"(?P<instruction>.+?)\s*$",
    re.IGNORECASE,
)

_PARALLEL_READ_RE = re.compile(r"^\[\s*parallel-read\s*\]\s*", re.IGNORECASE)

_REMAINDER_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?(?:PARTIAL\b|remainder\b)", re.IGNORECASE)

_ESCALATION_MARKER_RE = re.compile(
    r"^ {0,3}\[(?P<close>/)?ESCALATE\][ \t]*\r?$", re.IGNORECASE | re.MULTILINE,
)
_LEASE_MARKER_RE = re.compile(
    r"^ {0,3}\[(?P<close>/)?LEASE\][ \t]*\r?$", re.IGNORECASE | re.MULTILINE,
)
_OWNER_MARKER_RE = re.compile(
    r"^ {0,3}\[(?P<close>/)?OWNER\][ \t]*\r?$", re.IGNORECASE | re.MULTILINE,
)
_SPECIALISTS_MARKER_RE = re.compile(
    r"^ {0,3}\[(?P<close>/)?SPECIALISTS\][ \t]*\r?$", re.IGNORECASE | re.MULTILINE,
)
_CODE_FENCE_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>[^\r\n]*)\r?$", re.MULTILINE)
_CONTROL_NONSPACE_RE = re.compile(r"\S")
_CONTROL_MAX_CHARS = 6_000


def _restore_control_fields(value: object, names: tuple[str, ...]) -> dict[str, str] | None:
    """Validate a bounded, unambiguous retained control message."""
    if not isinstance(value, dict) or set(value) != set(names):
        return None
    fields = {}
    size = 0
    for key, text in value.items():
        if not isinstance(text, str):
            return None
        size += len(text)
        if size > _CONTROL_MAX_CHARS or not text.strip() or text.splitlines() != [text]:
            return None
        fields[key] = text.strip()
    return fields


def _control_block(
    report: str, marker: re.Pattern[str], *, terminal: bool = True,
    max_chars: int = _CONTROL_MAX_CHARS,
) -> str | None:
    """Accept one bounded control block outside quoted code."""
    if not isinstance(report, str):
        return None
    markers: list[re.Match[str]] = []
    fences = iter(_CODE_FENCE_RE.finditer(report))
    fence_match = next(fences, None)
    fence_char, fence_size = "", 0
    for match in marker.finditer(report):
        while fence_match is not None and fence_match.start() < match.start():
            fence = fence_match.group("fence")
            info = fence_match.group("info")
            if not fence_char:
                if fence[0] != "`" or "`" not in info:
                    fence_char, fence_size = fence[0], len(fence)
            elif fence[0] == fence_char and len(fence) >= fence_size and not info.strip():
                fence_char, fence_size = "", 0
            fence_match = next(fences, None)
        if fence_char:
            continue
        if len(markers) == 2:
            return None
        markers.append(match)
    if len(markers) != 2:
        return None
    opening, closing = markers
    if opening.group("close") or not closing.group("close"):
        return None
    if terminal and _CONTROL_NONSPACE_RE.search(report, closing.end()):
        return None
    if closing.start() - opening.end() > max_chars:
        return None
    return report[opening.end():closing.start()]


def _parse_control_fields(
    report: str, marker: re.Pattern[str], names: tuple[str, ...],
) -> dict[str, str] | None:
    """Accept one terminal control block outside quoted code."""
    block = _control_block(report, marker)
    if block is None:
        return None
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if not line.strip():
            continue
        label, separator, value = line.partition(":")
        key = label.strip().casefold()
        if not separator or key not in names or key in fields:
            return None
        value = value.strip()
        if not value:
            return None
        fields[key] = value
    return _restore_control_fields(fields, names)


def restore_escalation(value: object) -> Escalation | None:
    """Validate retained escalation fields before resuming the same task."""
    fields = _restore_control_fields(value, ("reason", "evidence", "remaining"))
    return Escalation(**fields) if fields is not None else None


def parse_escalation(report: str) -> Escalation | None:
    """Accept one bounded escalation block; the coordinator chooses transitions."""
    fields = _parse_control_fields(report, _ESCALATION_MARKER_RE, ("reason", "evidence", "remaining"))
    return restore_escalation(fields)


def restore_lease(value: object) -> Lease | None:
    """Validate one retained helper request without changing parent ownership."""
    fields = _restore_control_fields(value, ("tier", "reason", "task"))
    if fields is None:
        return None
    tier = normalize_tier(fields["tier"])
    return Lease(tier, fields["reason"], fields["task"]) if tier is not None else None


def parse_lease(report: str) -> Lease | None:
    """Accept one bounded lease block; the coordinator chooses transitions."""
    fields = _parse_control_fields(report, _LEASE_MARKER_RE, ("tier", "reason", "task"))
    return restore_lease(fields)


def _unique_json_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("duplicate specialist field")
        fields[key] = value
    return fields


def _initial_specialists(raw: str, owner_tier: str) -> tuple[tuple[Lease, ...], bool]:
    """Keep optional advice only when its whole bounded list is valid."""
    if _SPECIALISTS_MARKER_RE.search(raw) is None:
        return (), True
    block = _control_block(
        raw, _SPECIALISTS_MARKER_RE, terminal=False,
        max_chars=MAX_INITIAL_SPECIALISTS * _CONTROL_MAX_CHARS + 256,
    )
    if block is None:
        return (), False
    try:
        rows = json.loads(block, object_pairs_hook=_unique_json_fields)
    except (ValueError, RecursionError):
        return (), False
    if not isinstance(rows, list) or len(rows) > MAX_INITIAL_SPECIALISTS:
        return (), False
    specialists = []
    tiers = {owner_tier}
    for row in rows:
        lease = restore_lease(row)
        if lease is None or lease.tier in tiers:
            return (), False
        specialists.append(lease)
        tiers.add(lease.tier)
    return tuple(specialists), True


def normalize_tier(value: str) -> str | None:
    """Return the canonical tier name for *value*, or None if unknown."""
    if not isinstance(value, str):
        return None
    tier = value.strip().lower()
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


_INSTRUCTION_QUOTE_START_RE = re.compile(r"[`'\"]")
_INSTRUCTION_QUOTE_DELIMITERS = {
    "'": re.compile(r"['\\]"),
    '"': re.compile(r'["\\]'),
}


def instruction_identity(instruction: str) -> str:
    """Normalize formatting, preserving case and whitespace in quoted targets.

    Search for the same first complete quoted spans as the previous split
    regex. An unmatched escaped-quote run used to restart that regex at every
    quote. Remember each failed run's boundary instead, so each quote style
    scans any input character at most once even for malformed instructions.
    """
    pieces = []
    plain_start = search_from = 0
    failed_until = {"'": 0, '"': 0}
    while match := _INSTRUCTION_QUOTE_START_RE.search(instruction, search_from):
        start = match.start()
        quote = match.group()
        search_from = start + 1
        if quote == "`":
            close = instruction.find("`", search_from)
            end = close + 1 if close >= 0 else None
        else:
            if start < failed_until[quote]:
                continue
            end = None
            position = search_from
            while delimiter := _INSTRUCTION_QUOTE_DELIMITERS[quote].search(instruction, position):
                if delimiter.group() == quote:
                    end = delimiter.end()
                    break
                escaped = delimiter.end()
                if escaped == len(instruction) or instruction[escaped] == "\n":
                    # A regex dot cannot consume LF. Every same-style quote
                    # encountered before this invalid escape was itself escaped;
                    # starting there would fail at this identical boundary too.
                    failed_until[quote] = escaped + 1
                    break
                position = escaped + 1
            else:
                failed_until[quote] = len(instruction)
        if end is not None:
            pieces.append(re.sub(r"\s+", " ", instruction[plain_start:start]))
            pieces.append(instruction[start:end])
            plain_start = search_from = end
    pieces.append(re.sub(r"\s+", " ", instruction[plain_start:]))
    normalized = "".join(pieces).strip()
    # A polite request prefix does not create another obligation. Everything
    # identifying a target or action remains case-sensitive and unchanged.
    return re.sub(r"^Please\s+", "", normalized)


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
    """Keep the whole request on low when planning gives no difficulty evidence."""
    if isinstance(request, str):
        instruction = request.strip() or "(empty request)"
    else:
        instruction = str(request) or "(empty request)"
    return Plan(
        understanding="",
        subtasks=(Subtask(tier=FALLBACK_TIER, instruction=instruction),),
        owner_mode=True,
        owner_reason="No valid owner choice was supplied; inspect the full request and retain every obligation.",
    )


def parse_plan(raw: str, request: str) -> Plan:
    """Parse a whole-job owner or a legacy plan, falling back to a low owner.

    A missing ``[PLAN]`` wrapper or stray prose can still yield tasks.
    Unrecognized tiers do not make malformed plans more expensive.
    """
    if not isinstance(raw, str):
        return fallback_plan(request)
    if not raw or not raw.strip():
        return fallback_plan(request)

    if _OWNER_MARKER_RE.search(raw) is not None:
        owner = restore_lease(_parse_control_fields(raw, _OWNER_MARKER_RE, ("tier", "reason", "task")))
        if owner is None:
            return fallback_plan(request)
        specialists, valid = _initial_specialists(raw, owner.tier)
        understanding = extract_marker_block(raw, "UNDERSTANDING") or ""
        if not valid:
            note = (
                "Optional initial advice was omitted because its specification was invalid;"
                " the owner retains every requested obligation and constraint."
            )
            understanding = "\n".join(filter(None, (understanding, note)))
        return Plan(
            understanding=understanding,
            subtasks=(Subtask(owner.tier, owner.task),),
            owner_mode=True,
            specialists=specialists,
            owner_reason=owner.reason,
        )
    if _SPECIALISTS_MARKER_RE.search(raw) is not None:
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


def build_plan_prompt(
    context: str, *, collaborative: bool, skills_catalog: str = "",
) -> str:
    """Prompt the summary agent to choose a whole-job owner and focused advice."""
    parts = [_PLANNER_RULES]
    if collaborative:
        parts.append(_PLANNER_QUESTION_RULE)
    parts.append(_PLANNER_FORMAT)
    parts.append("--- input ---")
    parts.append(context)
    if skills_catalog:
        parts.extend((
            "--- available workspace skills ---",
            skills_catalog,
            "Select relevant skills by name. If a skill-reading interface is available,"
            " read the selected instructions before choosing a detailed workflow."
            " Without read tools, select skills from catalog metadata only; do not"
            " assume hidden instructions. Name the selected skills in the responsible"
            " worker's instruction and explicitly assign that worker to load the full"
            " skill before choosing or executing its detailed workflow. The catalog"
            " describes available skills; do not assume it contains their full instructions.",
        ))
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
    """Render the complete worker prompt, retaining its established format."""
    return "\n".join(build_subtask_sections(
        context, plan, index, results, previous_draft=previous_draft,
    ).values())


def build_subtask_sections(
    context: str, plan: Plan, index: int, results: list[str],
    *, previous_draft: str | None = None,
) -> dict[str, str]:
    """Prompt for the worker running sub-task *index* of *plan*.

    *context* is the bare user request, not the full history — the
    planner saw the history and wrote self-contained instructions.
    When the request carries "[... attachment saved to: ...]" markers
    they ride along verbatim so a vision-capable tier backend still
    resolves the image bytes natively. A continuation also carries a bounded
    *previous_draft* excerpt so workers retain already-completed work and its
    final evidence.

    Sections retain their presentation separators and insertion order, so
    joining their values with a newline produces the complete prompt. Native
    callers may omit ``request`` and send the request separately, comparing
    the other named sections without parsing human-readable prompt markers.
    """
    task = plan.subtasks[index]
    # Fallback holds the whole plea once; reuse needs exact text.
    task_instruction = task.instruction
    if len(task_instruction) >= 128 and task_instruction == context.strip():
        task_instruction = "the complete user request above, including every saved obligation and constraint"
    parts = [""]
    parts.append(
        f"[Sub-task {index + 1}/{len(plan.subtasks)}: {task.tier}]"
        + (" — you own the whole unfinished job through completion."
           if plan.owner_mode else " — do ONLY this one; others have their own workers.")
    )
    if plan.owner_mode and plan.owner_reason:
        parts.append(f"Owner choice: {plan.owner_reason}")
    if plan.understanding:
        parts.append(f"Goal: {plan.understanding}")
    parts.append("\nPlan:")
    parts.append(_render_plan(plan, current=index))
    sections = {"request": context, "plan": "\n".join(parts)}

    retained_reports: list[str] = []
    if results:
        parts = ["\nEarlier reports:"]
        for result_index, text in enumerate(results):
            excerpt = truncate_report(text)
            duplicate = next((earlier + 1 for earlier, retained in enumerate(results[:result_index])
                              if len(excerpt) >= 128 and text == retained), None)
            retained_reports.append(excerpt)
            parts.append(
                f"\n--- report {result_index + 1} ---\n"
                + (f"Same evidence as report {duplicate} above (identical report)."
                   if duplicate is not None else excerpt)
            )
        sections["earlier_reports"] = "\n".join(parts)

    if previous_draft:
        excerpt = bounded_draft_excerpt(previous_draft)
        already_retained = len(previous_draft) >= 128 and (
            previous_draft in context or excerpt in context
            or any(previous_draft in report or excerpt in report for report in retained_reports)
        )
        sections["previous_draft"] = (
            "\nPrevious draft and evidence from this turn:\n"
            + ("The same retained evidence already appears in the user request or earlier reports above."
               if already_retained else excerpt)
            + "\nRetain completed work and verify its current state. Address"
            " the remaining instructions without repeating completed actions."
        )

    if task.read_only:
        sections["worker_instructions"] = (
            f"\nYour task: {task_instruction}\n"
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
            " Thinking... status only.\n"
            f"{_WORKER_ROUTING_RULES}"
            "A reasoning escalation preserves this inspection-only role. Its"
            " remaining field must request analysis only; never edits, commands,"
            " builds, tests, process operations, or other writes."
        )
        return sections

    sections["worker_instructions"] = (
        f"\nYour task: {task_instruction}\n"
        "Finish ALL of it - never a sample. If it says all/entire/every,"
        " find every item first (glob/grep/list), handle each one, then"
        " re-check for leftovers.\n"
        "If it points at a doc/standard, enumerate every rule/section first"
        " (read fully via offset chunks + grep), apply each, never stop"
        " after first/last few.\n"
        "Build/verify, when needed and not already covered by still-valid evidence:"
        " enumerate targets, run canonical build/test to completion"
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
        " what you skipped or could not do, and retained evidence"
        " (paths, commands, exit codes, error/warning counts).\n"
        "Be honest: say PARTIAL and list what remains if anything is"
        " unfinished or unverified; never report done when work remains"
        " or evidence is missing. Progress is Thinking... status only; notify"
        " DONE only at the very end.\n"
        + ("New work discovered while doing this task: list each item on its own"
        " line as [followup:<low|mid|high>] <instruction> (self-contained, one"
        " line each). Only genuine leftovers found inside this task - never"
        " restate the plan or completed/queued scope. Prefer finishing related"
        " repairs and their verification together; do not create speculative"
        " audits or separate follow-ups for each workflow step. Related leftovers"
        " of one consecutive tier group will run in one cohesive worker, preserving every item."
        " Never write 'Next turn should:' or any other"
        " handoff/planning note; the followup lines above are the only"
        " carry-forward mechanism for new scope.\n"
           if not plan.owner_mode else "")
        + (_OWNER_ROUTING_RULES if plan.owner_mode else _WORKER_ROUTING_RULES)
    )
    return sections


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
    retained_reports: list[str] = []
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
        excerpt = truncate_report(text) if text else "(no report)"
        duplicate = next((earlier + 1 for earlier, retained in enumerate(retained_reports)
                          if len(excerpt) >= 128 and text and text == retained), None)
        retained_reports.append(text)
        parts.append(header + (f"Same evidence as report {duplicate} above (identical report)."
                               if duplicate is not None else excerpt))
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
        bounded_draft_excerpt(draft) if draft else "(no draft)",
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
