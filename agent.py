"""Agent runtime — runs a single user turn: routes to a session,
prepends colony + session context to the prompt, invokes the backend
CLI or HTTP backend, streams events, logs the turn, and triggers compaction and
auto-titling background tasks.
"""

import asyncio
import contextvars
import logging
import math
import os
import re
import stat
import weakref
from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import dataclasses
from datetime import datetime
from typing import Any, TypeVar

from . import (
    agent_tools, backends_agent, colony, compaction, config, execution_state, flexible, router,
    native_prompt, session, skills, titling,
)
from . import workspace as workspace_mod
from .agent_tools.base import copy_file_atomically
from .agent_tools.owned_worker import run_owned_worker
from .backends_agent.base import (
    DOC_FOLLOWING_RULE, NO_RESPONSE_TEXT, WHOLE_SCOPE_RULE, AgentResult,
    ChatEvent, ContinuationCheckpoint, ContinuationReason, ContinuationReport,
    NativeSessionUnavailableError,
    append_detached_task_request,
    append_text_result, record_backend_error, set_error_result,
)
from .backends_agent._openai_agent import http_continuation_scope
from .utils import (
    IMAGE_EXTENSIONS,
    abandon_subprocess_stream_task,
    await_cancelled,
    cancel_and_settle,
    cleanup_backend_process,
    create_background_task,
    drain_text_stream,
    finish_process_stderr,
    has_managed_process_group,
    image_header_extension,
    is_path_within,
    iter_process_json_events,
    json_digest,
    kill_and_wait,
    run_internal_backend,
    take_recent_lines,
    terminate_process_group_async,
    wait_for_process_exit,
)
from .utils import drain_queue as _drain_queue

logger = logging.getLogger(__name__)
performance_logger = logging.getLogger("Cozter.performance")
_StageResult = TypeVar("_StageResult")


class _TurnBudgetExhaustedError(Exception):
    """Pause an execution attempt while retaining the request's remaining work."""


CONTINUATION_CHECKPOINT_CHARS = 64_000
_CONTINUATION_CHECKPOINT_ITEMS = flexible.MAX_SUBTASKS * 4
_DEFAULT_PENDING_SCOPE = "Verify completion of the full original request and finish any remaining work."


def _checkpoint_scope(instruction: str, request: str) -> str:
    """Keep a complete transport request out of scope metadata.

    The canonical request is saved separately. A fallback worker can receive
    that entire request, including a previous checkpoint, as its instruction;
    copying it into pending scope or report labels would nest resume payloads.
    Exact equality preserves independently authored instructions verbatim.
    """
    return _DEFAULT_PENDING_SCOPE if instruction == request else instruction


def _current_task_or_none() -> asyncio.Task | None:
    """Capture a task owner when running inside an event loop."""
    try:
        return asyncio.current_task()
    except RuntimeError:
        # Budgets can be constructed before a synchronous caller starts a loop.
        return None


@dataclass
class TurnBudget:
    """Worker allowance and retained evidence shared by immediate continuations."""

    max_workers: int = flexible.MAX_SUBTASKS * 2
    started: bool = False
    workers_started: int = 0
    pending: list[str] = dataclasses.field(default_factory=list)
    reports: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    output: AgentResult = dataclasses.field(default_factory=AgentResult)
    captured: list[weakref.ReferenceType[AgentResult]] = dataclasses.field(default_factory=list)
    latest_draft: str = ""
    owner: asyncio.Task | None = dataclasses.field(default_factory=_current_task_or_none)
    auto_attachment_files: int = 0
    auto_attachment_bytes: int = 0
    auto_attachment_sources: set[str] = dataclasses.field(default_factory=set)
    auto_attachment_omitted_sources: set[str] = dataclasses.field(default_factory=set)

    def start(self) -> None:
        # Start after the workspace run lock.
        if not self.started:
            self.started = True
            performance_logger.info(
                "stage=turn_budget event=start max_workers=%d", self.max_workers,
            )

    def claim_worker(self) -> None:
        if self.workers_started >= self.max_workers:
            raise _TurnBudgetExhaustedError(
                f"the shared limit of {self.max_workers} workers was reached"
            )
        self.workers_started += 1

    def capture(
        self, result: AgentResult, workspace_path: str, *, account_usage: bool = True,
    ) -> None:
        self.captured = [reference for reference in self.captured if reference() is not None]
        newly_captured = not any(result is reference() for reference in self.captured)
        if newly_captured:
            self.captured.append(weakref.ref(result))
        retained = AgentResult()
        carry_forward_turn_output(result, retained, workspace_path)
        for event in retained.events:
            if event.kind == "attachment" and event not in self.output.events:
                self.output.events.append(event)
        for task in retained.detached_tasks:
            if task not in self.output.detached_tasks:
                self.output.detached_tasks.append(task)
        for request in retained.detached_task_requests:
            if request not in self.output.detached_task_requests:
                self.output.detached_task_requests.append(request)
        if newly_captured and account_usage:
            usage = self.output.usage or {}
            _accumulate_usage(usage, result.usage)
            self.output.usage = usage or None

    def checkpoint(self, reason: str, turn: "_InterruptedTurn") -> ContinuationCheckpoint:
        """Keep worker evidence and actionable scope outside the reply's small preview."""
        reports = [(_checkpoint_scope(instruction, turn.prompt), report)
                   for instruction, report in self.reports[-_CONTINUATION_CHECKPOINT_ITEMS:]]
        pending = list(dict.fromkeys(
            _checkpoint_scope(instruction, turn.prompt) for instruction in self.pending
        ))[:_CONTINUATION_CHECKPOINT_ITEMS]
        additions = turn.injected[:_CONTINUATION_CHECKPOINT_ITEMS]
        if not pending:
            pending = [_DEFAULT_PENDING_SCOPE]
        streamed = next((
            event.content for event in reversed(turn.events)
            if event.kind == "text" and event.content.strip()
        ), "")
        if streamed == self.latest_draft:
            streamed = ""
        activity = next((
            event.content for event in reversed(turn.events)
            if event.kind in ("tool", "file") and event.content.strip()
        ), "")
        values = [value for pair in reports for value in pair] + pending + additions + [
            turn.prompt,
            self.latest_draft,
            _truncate_context_text(activity, INTERRUPTED_TURN_ACTIVITY_CHARS),
            flexible.bounded_draft_excerpt(streamed, limit=1200),
        ]
        quotas = _context_quotas([len(value) for value in values], CONTINUATION_CHECKPOINT_CHARS)
        bounded = [
            flexible.bounded_draft_excerpt(value, limit=quota)
            for value, quota in zip(values, quotas, strict=True)
        ]
        report_values: list[ContinuationReport] = [
            {"instruction": bounded[index * 2], "report": bounded[index * 2 + 1]}
            for index in range(len(reports))
        ]
        pending_start = len(reports) * 2
        additions_start = pending_start + len(pending)
        bounded_additions = bounded[additions_start:additions_start + len(additions)]
        return {
            "reason": reason,
            "request": bounded[-4],
            "request_truncated": bounded[-4] != turn.prompt,
            "accepted_additions": bounded_additions,
            "accepted_additions_truncated": turn.injected != bounded_additions,
            "reports": report_values,
            "pending": bounded[pending_start:pending_start + len(pending)],
            "latest_draft": bounded[-3],
            "last_activity": bounded[-2],
            "streamed_output": bounded[-1],
            "truncated": (
                len(self.reports) > len(reports)
                or len(set(self.pending)) > len(pending)
                or len(turn.injected) > len(additions)
                or values != bounded
                or len(activity) > INTERRUPTED_TURN_ACTIVITY_CHARS
                or len(streamed) > 1200
                or flexible.DRAFT_OMISSION_MARKER in self.latest_draft
                or any(flexible.REPORT_TRUNCATION_MARKER in report for _, report in reports)
            ),
        }

    def finish(
        self, reason: str, turn: "_InterruptedTurn",
        *, continuation_reason: ContinuationReason = "worker_budget",
    ) -> AgentResult:
        result = self.output
        result.error = f"Turn budget exhausted: {reason}."
        result.completion_verified = False
        result.continuation_reason = continuation_reason
        result.continuation_checkpoint = self.checkpoint(reason, turn)
        result.continue_instruction = (
            "Continue the same original request from the retained checkpoint. "
            "Re-check saved files and artifacts; preserve verified completed work, "
            "finish every unfinished or unverified requirement, and verify full completion."
            " Resume interrupted long validation from its per-case checkpoint "
            "when inputs and build are unchanged (for example, CTest -F). "
            "Inspect retained full logs for prior failures; a skipped case or "
            "a successful remaining subset alone is not full passing evidence."
        )
        result.session_id = turn.session_id
        sections = [
            f"PARTIAL: {reason}. Work already completed was retained; "
            "this execution attempt paused before full completion and needs continuation.",
        ]
        if self.reports:
            reports = "\n\n".join(
                f"Task: {instruction}\n{flexible.truncate_report(report)}"
                for instruction, report in self.reports
            )
            sections.append("Completed worker reports:\n" +
                            flexible.bounded_draft_excerpt(reports))
        elif self.latest_draft:
            sections.append("Latest completed draft:\n" +
                            flexible.bounded_draft_excerpt(self.latest_draft))
        streamed = next((
            event.content for event in reversed(turn.events)
            if event.kind == "text" and event.content.strip()
        ), "")
        if streamed and streamed != self.latest_draft:
            sections.append("Latest streamed output (unverified):\n" +
                            flexible.bounded_draft_excerpt(streamed, limit=1200))
        activity = next((
            event.content for event in reversed(turn.events)
            if event.kind in ("tool", "file") and event.content.strip()
        ), "")
        if activity:
            sections.append("Last activity before stopping:\n" +
                            _truncate_context_text(activity, INTERRUPTED_TURN_ACTIVITY_CHARS))
        if self.pending:
            pending = "\n".join(
                f"- {_truncate_context_text(instruction, 200)}"
                for instruction in dict.fromkeys(self.pending)
            )
            sections.append("Unfinished or unverified scope:\n" +
                            flexible.bounded_draft_excerpt(pending, limit=6000))
        else:
            sections.append("Final completion verification remains unfinished.")
        append_text_result(result, "\n\n".join(sections))
        return result


_TURN_BUDGET: ContextVar[TurnBudget | None] = ContextVar("cozter_turn_budget", default=None)


@contextmanager
def turn_budget(*, fresh: bool = False):
    """Reuse an attempt's budget, or explicitly start the next allowance."""
    current = _TURN_BUDGET.get()
    if not fresh and current is not None and current.owner is _current_task_or_none():
        yield current
        return
    token = _TURN_BUDGET.set(TurnBudget())
    try:
        yield _TURN_BUDGET.get()
    finally:
        _TURN_BUDGET.reset(token)


_PENDING_SESSION_WRITES: dict[asyncio.Lock, set[asyncio.Task[Any]]] = {}


async def _run_with_stage_timing(
    call: Awaitable[_StageResult], stage: str, backend_name: str,
    model: str | None,
    *, worker_index: int | None = None, worker_count: int | None = None,
    worker_tier: str | None = None,
) -> _StageResult:
    """Record stage starts and durations without private call content."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    worker_metadata = ""
    if worker_index is not None:
        worker_metadata += f" worker_index={worker_index}"
    if worker_count is not None:
        worker_metadata += f" worker_count={worker_count}"
    if worker_tier in flexible.TIERS:
        worker_metadata += f" tier={worker_tier}"
    performance_logger.info(
        "stage=%s backend=%s model=%s event=start%s",
        stage, backend_name, model, worker_metadata,
    )
    outcome = "ok"
    try:
        value = await call
        payload = value[0] if isinstance(value, tuple) and value else value
        if getattr(payload, "error", None):
            outcome = "error"
        if isinstance(value, tuple) and len(value) == 2 and value[1] is True:
            outcome = "cancelled"
        return value
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except Exception:
        outcome = "error"
        raise
    finally:
        performance_logger.info(
            "stage=%s backend=%s model=%s outcome=%s duration_seconds=%.3f%s",
            stage, backend_name, model, outcome, loop.time() - started,
            worker_metadata,
        )


def _terminal_provider_error(error: str) -> bool:
    """Recognize explicit account/auth failures and Cozter's refusal stop."""
    if _account_provider_unavailable(error):
        return True
    if re.search(r"\bHTTP\s*(?:status\s*)?(?:401|403|429)\b", error, re.IGNORECASE):
        return True
    normalized = error.casefold().replace("_", " ").replace("-", " ")
    return any(marker in normalized for marker in (
        "usage limit", "quota limit",
        "consecutive rounds where every tool call was refused",
        "tool budget exhausted",
    ))


def _account_provider_unavailable(error: str) -> bool:
    """Recognize definite account limits, separately from task refusals.

    A bare HTTP 429 may be temporary throttling, and a tool refusal or HTTP
    403 can depend on the task. Only explicit exhausted limits or failed
    authentication justify suppressing another launch on the same model.
    """
    if re.search(r"\bHTTP\s*(?:status\s*)?401\b", error, re.IGNORECASE):
        return True
    normalized = error.casefold().replace("_", " ").replace("-", " ")
    if re.search(
        r"\b(?:hit|reached|exceeded)\s+(?:(?:your|the|a)\s+)?monthly\s+spend(?:ing)?\s+limit\b"
        r"|\bmonthly\s+spend(?:ing)?\s+limit\s+(?:(?:has\s+been|was|is)\s+)?(?:hit|reached|exceeded)\b",
        normalized,
    ):
        return True
    return any(marker in normalized for marker in (
        "usage limit reached", "usage limit exceeded", "reached your usage limit",
        "quota exceeded", "quota exhausted", "quota limit reached", "quota limit exceeded",
        "insufficient quota", "insufficient credits", "out of credits",
        "credit balance exhausted", "invalid api key", "authentication failed",
    ))


@dataclass(frozen=True)
class DetachedTaskLaunch:
    """A restart-safe provider task started for one Cozter session."""
    backend_name: str
    task_id: str
    session_id: str

_ATTACH_HINT = (
    'Send created files via "[[attach: PATH]]" on its own line.'
)

_COLLABORATION_POLICY = (
    "Live chat. Ambiguous/large/destructive/irreversible:"
    " ask ONE short question ending [[await]] (next msg"
    " answers). Small reversible: decide, note in one line, go."
)

_AUTONOMY_POLICY = (
    "Unattended: no mid-task answers. Decide, finish; if"
    " blocked, state what's missing."
)

_COMPLETENESS_RULE = (
    f"{WHOLE_SCOPE_RULE}"
    f" {DOC_FOLLOWING_RULE}"
    " Verify-with-evidence: build/test/verify = enumerate targets, run"
    " canonical commands to completion (cancel still stops instantly; capture"
    " to file), grep full logs for error/warning/exception, fix each,"
    " re-run until zero; runtime = launch + exercise paths + check logs;"
    " report commands + exit codes + counts; never claim clean from"
    " truncated preview or unrun steps — say PARTIAL + remainder."
    " Progress is tool status (Thinking... preview, done/total may grow"
    " as follow-ups join); interim progress is never a chat reply."
    " Notify DONE only when everything is done."
    " Work efficiently: batch independent searches and file ranges into one"
    " tool round trip (read_files when available, or native parallel reads)."
    " Run independent read-only inspections concurrently with bounded fan-out;"
    " serialize dependent edits and builds sharing output directories."
    " Reuse still-valid completed worker evidence and saved validation logs;"
    " recheck changed inputs, failed checks and unfinished scope rather than"
    " repeating completed setup or the full verification after each handoff."
    " Keep all requested obligations and completion evidence."
)


_DETACHED_TASK_POLICY = (
    "No shell backgrounding/callbacks. End with one self-contained"
    " `[[background: <task>]]` line; Cozter runs it, posts result."
    " No-input work only."
)


def _capability_hint(
    collaborative: bool, *, allow_detached_requests: bool,
) -> str:
    """Build the per-turn preamble (collaboration vs autonomy policy)."""
    policy = _COLLABORATION_POLICY if collaborative else _AUTONOMY_POLICY
    parts = [_ATTACH_HINT, policy, _COMPLETENESS_RULE]
    if allow_detached_requests:
        parts.append(_DETACHED_TASK_POLICY)
    return "[System: " + "\n\n".join(parts) + "]"


def _is_collaborative_turn(
    workspace_path: str, *, is_scheduled: bool,
) -> bool:
    """Return whether this turn may pause for a live user answer."""
    return (
        not is_scheduled
        and workspace_mod.get_interaction_style(workspace_path)
        == "collaborative"
    )


_ATTACH_PREFIX_RE = re.compile(r"\[\[attach:", re.IGNORECASE)
_ATTACH_BOUNDARY_RE = re.compile(r"[\]\n]")
_AWAIT_RE = re.compile(r"\[\[await\]\]", re.IGNORECASE)
_DETACHED_TASK_PREFIX_RE = re.compile(r"^[ \t]*\[\[background:", re.IGNORECASE | re.MULTILINE)
_EXTRA_BLANK_LINES_RE = re.compile(r"\n{3,}")
_SAFE_IMAGE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_DETACHED_TASK_REQUESTS = 3
_MAX_DETACHED_TASK_PROMPT_CHARS = 12_000


def _iter_attachment_markers(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield attachment ranges/captures without overlapping suffix scans."""
    cursor = 0
    boundary = -1
    tail_boundary = -1
    tail_end = -1
    while prefix := _ATTACH_PREFIX_RE.search(text, cursor):
        content_start = prefix.end()
        while content_start < len(text) and text[content_start].isspace():
            content_start += 1
        if content_start == len(text):
            return
        if text[content_start] == "]":
            # Optional whitespace can give the required capture one whitespace
            # character, but the capture cannot contain a newline.
            capture_start = content_start - 1
            while capture_start >= prefix.end() and text[capture_start] == "\n":
                capture_start -= 1
            if text.startswith("]]", content_start) and capture_start >= prefix.end():
                end = content_start + 2
                yield prefix.start(), end, text[capture_start:capture_start + 1]
                cursor = end
            else:
                cursor = prefix.end()
            continue

        if boundary < content_start:
            stop = _ATTACH_BOUNDARY_RE.search(text, content_start)
            if stop is None:
                return
            boundary = stop.start()
        closing = boundary
        if text[boundary] == "\n":
            if tail_boundary != boundary:
                tail_boundary = boundary
                tail_end = boundary
                while tail_end < len(text) and text[tail_end].isspace():
                    tail_end += 1
            closing = tail_end
        if not text.startswith("]]", closing):
            cursor = prefix.end()
            continue
        end = closing + 2
        yield prefix.start(), end, text[content_start:boundary].rstrip()
        cursor = end


def _iter_detached_task_markers(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield complete line-anchored background requests with linear searches."""
    cursor = 0
    while prefix := _DETACHED_TASK_PREFIX_RE.search(text, cursor):
        closing = text.find("]]", prefix.end())
        while closing >= 0:
            end = closing + 2
            while end < len(text) and text[end] in " \t":
                end += 1
            if end == len(text):
                break
            if text[end] == "\n":
                end += 1
                break
            if text.startswith("\r\n", end):
                end += 2
                break
            closing = text.find("]]", closing + 1)
        if closing < 0:
            return
        start = prefix.end()
        while start < closing and text[start].isspace():
            start += 1
        yield prefix.start(), end, text[start:closing].rstrip()
        cursor = end


def _remove_protocol_markers(
    text: str, markers: Iterable[tuple[int, int, str]],
    on_capture: Callable[[str], None] | None = None,
) -> str:
    parts: list[str] = []
    cursor = 0
    for start, end, capture in markers:
        parts.append(text[cursor:start])
        cursor = end
        if on_capture is not None:
            on_capture(capture)
    parts.append(text[cursor:])
    return "".join(parts)

_ATTACHMENT_SCAN_SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
}
_AUTO_ATTACHMENT_SKIP_DIRS = {
    "__cmake", "build", "builds", ".build", "_build", "_builds", "cmakefiles",
    "testing", "test-results", "test-artifacts", "test-output", "_deps", ".cache",
}
_AUTO_ATTACHMENT_MAX_FILES = 8
_AUTO_ATTACHMENT_MAX_BYTES = 20 * 1024 * 1024
_EXTERNAL_ATTACHMENT_ROOTS_ENV = "COZTER_ATTACHMENT_ROOTS"


def _workspace_candidate_path(path: str, workspace_path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(workspace_path, path)


def _collapse_extra_blank_lines(text: str) -> str:
    """Normalize leftover blank lines after stripping control markers."""
    if not isinstance(text, str):
        return ""
    return _EXTRA_BLANK_LINES_RE.sub("\n\n", text).strip()


def extract_attachment_sources(text: str, workspace_path: str) -> tuple[str, list[str]]:
    """Parse attachment markers without copying external generated images."""
    if not isinstance(text, str):
        return "", []
    paths: list[str] = []
    seen_sources: set[str] = set()

    def _sub(capture: str) -> None:
        rel = capture.strip()
        if not rel:
            return
        source = _resolve_attachment_source(rel, workspace_path)
        if source is None:
            return
        source_path, _ = source
        if source_path in seen_sources:
            return
        seen_sources.add(source_path)
        paths.append(source_path)

    cleaned = _remove_protocol_markers(text, _iter_attachment_markers(text), _sub)
    return _collapse_extra_blank_lines(cleaned), paths


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

    Pairs with :func:`_capability_hint` (whose interactive variant
    instructs the model to ask and end with the marker when it needs a
    decision before continuing).
    Returns ``(cleaned_text, awaiting)``.
    """
    if not isinstance(text, str):
        return "", False
    if not _AWAIT_RE.search(text):
        return text, False
    cleaned = _AWAIT_RE.sub("", text)
    return _collapse_extra_blank_lines(cleaned), True


def extract_detached_task_requests(text: str) -> tuple[str, list[str]]:
    """Strip agent ``[[background: ...]]`` markers and return their prompts.

    The marker is intentionally a final-response protocol rather than a CLI
    command the foreground agent launches itself. Cozter can therefore
    persist the provider task id before it acknowledges the request, and can
    still deliver completion after a restart.
    """
    if not isinstance(text, str):
        return "", []
    prompts: list[str] = []

    def _take(capture: str) -> None:
        prompt = capture.strip()
        if len(prompt) > _MAX_DETACHED_TASK_PROMPT_CHARS:
            logger.warning("Ignoring oversized detached task request")
        elif prompt and prompt not in prompts:
            if len(prompts) < _MAX_DETACHED_TASK_REQUESTS:
                prompts.append(prompt)
            else:
                logger.warning("Ignoring excess detached task request")

    cleaned = _remove_protocol_markers(text, _iter_detached_task_markers(text), _take)
    return _collapse_extra_blank_lines(cleaned), prompts


def _consume_detached_task_requests(result: AgentResult) -> None:
    """Move response markers into ``result`` before logging or delivery."""
    events: list[ChatEvent] = []
    for ev in result.events:
        if ev.kind != "text":
            events.append(ev)
            continue
        cleaned, prompts = extract_detached_task_requests(ev.content)
        for prompt in prompts:
            append_detached_task_request(result, prompt)
        if cleaned:
            ev.content = cleaned
            events.append(ev)
    result.events = events

    cleaned_text, prompts = extract_detached_task_requests(result.text)
    for prompt in prompts:
        append_detached_task_request(result, prompt)
    result.text = cleaned_text


def _valid_usage_number(value: object) -> bool:
    """Keep malformed provider accounting from breaking reply delivery."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return value >= 0 and math.isfinite(value)
    except OverflowError:
        return False


def _compact_tokens(n: int) -> str:
    if isinstance(n, bool) or n < 0:
        return "0"
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def format_usage(usage: dict | None) -> str | None:
    """Compact one-line token/cost footer from a backend's usage dict.

    Returns None when there's nothing worth showing. Understands the
    fields Codex (``turn.completed``), Claude Code and Grok (``result``) expose
    (``input_tokens`` / ``output_tokens`` / ``total_cost_usd``).
    """
    if not isinstance(usage, dict):
        return None
    parts: list[str] = []
    input_tokens = usage.get("input_tokens")
    if isinstance(input_tokens, int) and _valid_usage_number(input_tokens):
        parts.append(f"{_compact_tokens(input_tokens)} in")
    output_tokens = usage.get("output_tokens")
    if isinstance(output_tokens, int) and _valid_usage_number(output_tokens):
        parts.append(f"{_compact_tokens(output_tokens)} out")
    cost = usage.get("total_cost_usd")
    if (
        isinstance(cost, (int, float)) and _valid_usage_number(cost) and cost > 0
    ):
        if cost < 0.0001:
            # Fixed-point: sub-cent costs must not show as $0.
            cost_str = f"{cost:.6f}".rstrip("0").rstrip(".")
        else:
            cost_str = f"{cost:.4f}".rstrip("0").rstrip(".")
        parts.append(f"${cost_str}")
    if not parts:
        return None
    return "📊 " + " · ".join(parts)


def _codex_generated_images_dir() -> str:
    codex_home = os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    return os.path.join(codex_home, "generated_images")


def _external_attachment_roots() -> list[str]:
    """Roots where trusted agents may save generated preview images."""
    roots = [_codex_generated_images_dir()]
    extra = os.environ.get(_EXTERNAL_ATTACHMENT_ROOTS_ENV, "")
    roots.extend(path_entry for path_entry in extra.split(os.pathsep) if path_entry.strip())

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
        with open(path, "rb") as file_handle:
            head = file_handle.read(32)
    except OSError:
        return None
    sniffed = image_header_extension(head)
    if sniffed is not None:
        return sniffed
    ext = os.path.splitext(path)[1].lower()
    return ext if ext in IMAGE_EXTENSIONS else None


def _safe_generated_image_name(src: str, ext: str) -> str:
    parent = os.path.basename(os.path.dirname(src))
    stem = os.path.splitext(os.path.basename(src))[0]
    raw = f"{parent}-{stem}" if parent else stem
    safe = _SAFE_IMAGE_NAME_RE.sub("-", raw).strip(".-")
    return f"{safe or 'generated-image'}{ext}"


def _copy_to_unique_path(source: str, directory: str, filename: str) -> str | None:
    """Publish *source* under an available filename without clobbering one.

    The generated-image directory may receive concurrent agent turns. Pick a
    readable basename first, but let the atomic no-clobber copy decide whether
    a candidate is actually available: an ``exists()`` check followed by a
    normal copy could overwrite a file that appeared in between.
    """
    stem, ext = os.path.splitext(filename)
    for index in range(1, 1_001):
        candidate = os.path.join(
            directory,
            filename if index == 1 else f"{stem}-{index}{ext}",
        )
        if os.path.lexists(candidate):
            continue
        if copy_file_atomically(source, candidate):
            return os.path.realpath(candidate)
    return None


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

    if is_path_within(src_real, ws_real):
        return src_real

    try:
        # Re-resolve: symlinks must not escape the workspace.
        dest_dir = workspace_mod.ensure_workspace_state_dir(
            workspace_path, "generated_images",
        )
        if not is_path_within(dest_dir, ws_real):
            logger.warning(
                "Refusing generated-image destination outside workspace: %s",
                dest_dir,
            )
            return None
        dest = _copy_to_unique_path(
            src_real, dest_dir, _safe_generated_image_name(src_real, ext),
        )
        if dest is None:
            logger.warning(
                "Could not reserve a generated-image destination after 1,000 attempts",
            )
        return dest
    except (OSError, ValueError):
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
    if is_path_within(real, ws_real):
        return real, False
    if _image_extension(real) is None:
        return None
    for root in _external_attachment_roots():
        if is_path_within(real, root):
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


def _skip_automatic_image_directory(name: str) -> bool:
    normalized = name.casefold()
    return (normalized in _ATTACHMENT_SCAN_SKIP_DIRS
            or normalized in _AUTO_ATTACHMENT_SKIP_DIRS
            or normalized.startswith("cmake-build-"))


def _iter_image_file_metadata(
    root: str, *, resolved_root: str | None = None,
) -> Iterator[tuple[str, os.stat_result]]:
    """Yield contained regular images and reuse each file's metadata.

    Canonicalize each candidate once before comparing its boundary. The
    workspace snapshot already supplies its canonical root; resolving both
    canonical values again for every image adds Windows path-handle calls
    without strengthening the boundary. Send/copy resolution remains separate.
    """
    try:
        root_real = resolved_root if resolved_root is not None else os.path.realpath(root)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                dirname for dirname in dirnames if not _skip_automatic_image_directory(dirname)
            ]
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue
                path = os.path.realpath(os.path.join(dirpath, filename))
                # os.walk follows file symlinks; stay workspace-only.
                try:
                    if os.path.commonpath((path, root_real)) != root_real:
                        continue
                    metadata = os.stat(path)
                except (OSError, ValueError):
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    continue
                relative_parts = os.path.relpath(path, root_real).split(os.sep)[:-1]
                # A link can point into a skipped tree.
                if not any(_skip_automatic_image_directory(part) for part in relative_parts):
                    yield path, metadata
    except OSError:
        logger.warning("Failed to scan image artifacts under %s", root,
                       exc_info=True)


def _snapshot_attachment_images(
    workspace_path: str,
) -> dict[str, tuple[int, int]]:
    """Return image artifact state for the current workspace only.

    Trusted external roots remain available through explicit ``[[attach:]]``
    markers, but must not be auto-scanned: their common default is a global
    Codex output directory shared by concurrent workspaces and sessions.
    Automatically attaching every changed image there can leak one turn's
    artifact into another chat.
    """
    snapshot: dict[str, tuple[int, int]] = {}
    ws_real = os.path.realpath(workspace_path)
    if not os.path.isdir(ws_real):
        return snapshot
    for path, metadata in _iter_image_file_metadata(ws_real, resolved_root=ws_real):
        snapshot[path] = (metadata.st_mtime_ns, metadata.st_size)
    return snapshot


@dataclass
class _AutomaticImageSelection:
    total_bytes: int = 0
    selected_sources: set[str] = dataclasses.field(default_factory=set)
    omitted_sources: set[str] = dataclasses.field(default_factory=set)


def _collect_new_attachment_images(
    before: dict[str, tuple[int, int]],
    workspace_path: str,
    *,
    exclude_sources: set[str] | None = None,
    max_files: int = _AUTO_ATTACHMENT_MAX_FILES,
    max_bytes: int = _AUTO_ATTACHMENT_MAX_BYTES,
    selection: _AutomaticImageSelection | None = None,
) -> list[str]:
    """Select a bounded set of changed output images, excluding generated trees."""
    after = _snapshot_attachment_images(workspace_path)
    changed = [
        path for path, stamp in after.items()
        if before.get(path) != stamp
    ]
    ordered = sorted(changed, key=lambda path: (after[path][0], path))

    copied: list[str] = []
    chosen = selection if selection is not None else _AutomaticImageSelection()
    excluded = exclude_sources or set()
    seen_sources: set[str] = set()
    seen_destinations: set[str] = set()
    for src in ordered:
        if src in excluded or src in seen_sources:
            continue
        seen_sources.add(src)
        size = after[src][1]
        if len(copied) >= max_files or size > max_bytes - chosen.total_bytes:
            chosen.omitted_sources.add(src)
            continue
        dest = prepare_attachment_path(src, workspace_path)
        if not dest or dest in seen_destinations:
            continue
        seen_destinations.add(dest)
        copied.append(dest)
        chosen.selected_sources.add(src)
        chosen.total_bytes += size
    return copied


async def _append_new_attachment_images(
    result: AgentResult,
    before: dict[str, tuple[int, int]],
    workspace_path: str,
) -> None:
    """Bound automatic output images across this attempt; explicit files are separate."""
    explicit_sources = _explicit_attachment_sources(result.events, workspace_path)
    if result.text:
        explicit_sources.update(extract_attachment_sources(result.text, workspace_path)[1])
    budget = _TURN_BUDGET.get()
    excluded = explicit_sources | (budget.auto_attachment_sources if budget else set())
    selection = _AutomaticImageSelection()
    paths = await asyncio.to_thread(
        _collect_new_attachment_images,
        before,
        workspace_path,
        exclude_sources=excluded,
        max_files=max(0, _AUTO_ATTACHMENT_MAX_FILES - (budget.auto_attachment_files if budget else 0)),
        max_bytes=max(0, _AUTO_ATTACHMENT_MAX_BYTES - (budget.auto_attachment_bytes if budget else 0)),
        selection=selection,
    )
    result.events.extend(ChatEvent(kind="attachment", content=path) for path in paths)
    if budget is not None:
        budget.auto_attachment_files += len(paths)
        budget.auto_attachment_bytes += selection.total_bytes
        budget.auto_attachment_sources.update(selection.selected_sources)
        budget.auto_attachment_omitted_sources.update(selection.omitted_sources)
        budget.auto_attachment_omitted_sources.difference_update(
            selection.selected_sources | explicit_sources,
        )
    omitted = len(budget.auto_attachment_omitted_sources if budget else selection.omitted_sources)
    if paths or omitted:
        performance_logger.info(
            "stage=automatic_images selected_files=%d selected_bytes=%d omitted_files=%d",
            len(paths), selection.total_bytes, omitted,
        )
    if omitted:
        note = (
            f"Automatic image attachments are limited to {_AUTO_ATTACHMENT_MAX_FILES} files "
            f"and {_AUTO_ATTACHMENT_MAX_BYTES // (1024 * 1024)} MiB per attempt; "
            f"{omitted} additional image(s) remain in the workspace. "
            "Additional files can be attached explicitly."
        )
        if not any(event.kind == "text" and event.content == note for event in result.events):
            if result.text and not any(event.kind == "text" for event in result.events):
                result.events.append(ChatEvent(kind="text", content=result.text))
            # This is delivery metadata, not another obligation for the judge.
            result.events.append(ChatEvent(kind="text", content=note))


_CONTEXT_TRUNCATION_MARKER = (
    "\n… [truncated to fit budget; older context omitted — never treat"
    " this preview as full content; re-check leftovers via tools;"
    " say PARTIAL + remainder when coverage is unclear]"
)

_PLANNER_OMISSION_MARKER = (
    "\n… [middle context omitted for 12k planner cap — preview only, not"
    " full coverage; plan must still cover every item/rule — prefer"
    " enumerated targets; say PARTIAL + remainder when coverage is"
    " unclear]\n"
)


def _truncate_context_text(text: str, limit: int) -> str:
    """Return *text* clipped to *limit* characters with a clear marker."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(_CONTEXT_TRUNCATION_MARKER):
        # Too tight for the marker: keep a cut indicator.
        if limit <= 1:
            return "…"[:limit]
        return text[:limit - 1] + "…"
    return text[:limit - len(_CONTEXT_TRUNCATION_MARKER)] + (
        _CONTEXT_TRUNCATION_MARKER
    )


def _bounded_context_block(
    label: str,
    body: str,
    limit: int | None = None,
) -> str:
    """Format one labeled memory block, clipping its body when requested."""
    if not body:
        return ""
    header = f"[{label}]\n"
    if limit is None:
        return header + body
    body_limit = limit - len(header)
    if body_limit <= 0:
        return ""
    clipped = _truncate_context_text(body, body_limit)
    return header + clipped if clipped else ""


def _bounded_context_list_block(
    label: str,
    items: list,
    formatter,
    limit: int | None = None,
) -> str:
    """Format a labeled list block, retaining newest entries under a cap."""
    if not items:
        return ""
    header = f"[{label}]\n"
    if limit is None:
        return header + "\n".join(formatter(item) for item in items)
    body_limit = limit - len(header)
    if body_limit <= 0:
        return ""
    lines = take_recent_lines(items, body_limit, formatter)
    if len(lines) < len(items):
        omission = (
            "… [older items omitted — never treat this preview as full"
            " coverage; say PARTIAL + remainder when coverage is unclear]"
        )
        # Drop oldest first (lengths tracked for O(n)).
        line_lengths = [len(line) for line in lines]
        total = sum(line_lengths) + max(0, len(lines) - 1)
        start = 0
        while start < len(lines) and total + len(omission) + 1 > body_limit:
            total -= line_lengths[start] + (1 if start < len(lines) - 1 else 0)
            start += 1
        lines = lines[start:]
        if lines:
            body = omission + "\n" + "\n".join(lines)
        else:
            body = _truncate_context_text(omission, body_limit)
        if not body:
            return ""
        return header + body
    body = "\n".join(lines)
    return header + body if body else ""


def _context_quotas(lengths: list[int], budget: int) -> list[int]:
    """Distribute a context budget fairly without wasting unused capacity."""
    quotas = [0] * len(lengths)
    remaining = max(0, budget)
    active = {index for index, length in enumerate(lengths) if length > 0}
    while remaining and active:
        share = max(1, remaining // len(active))
        progressed = False
        for index in tuple(active):
            grant = min(share, lengths[index] - quotas[index], remaining)
            if grant:
                quotas[index] += grant
                remaining -= grant
                progressed = True
            if quotas[index] >= lengths[index]:
                active.discard(index)
            if not remaining:
                break
        if not progressed:
            break
    return quotas


_KEYWORD_RE = re.compile(r"[a-z0-9]{3,}")

_PLANNER_CONTEXT_CAP = 12_000


def _request_keywords(text: str, limit: int = 64) -> set[str]:
    words = _KEYWORD_RE.findall(text.lower())
    seen: set[str] = set()
    for word in words:
        seen.add(word)
        if len(seen) >= limit:
            break
    return seen


def _relevance_last(items: list, keywords: set[str]) -> list:
    if not keywords:
        return items
    # First-char probe: skip items without a keyword initial.
    keyword_initials = {keyword[0] for keyword in keywords if keyword}
    keyword_initials |= {char.swapcase() for char in keyword_initials}
    scored = []
    for index, item in enumerate(items):
        text = item if isinstance(item, str) else session.format_msg_line(item)
        if not any(char in text for char in keyword_initials):
            scored.append((0, index, item))
            continue
        words = set(_KEYWORD_RE.findall(text.lower()))
        scored.append((len(words & keywords), index, item))
    scored.sort(key=lambda entry: (entry[0], entry[1]))
    return [entry[2] for entry in scored]


def _planner_context(contextual_prompt: str, request: str) -> str:
    """Bound historical context while retaining the current request in full.

    Durable resumes can carry a checkpoint larger than the history allowance.
    That current request is exempt from clipping; prepending another excerpt
    of the same checkpoint only duplicates input without adding evidence.
    """
    if len(request) >= _PLANNER_CONTEXT_CAP:
        return request
    if len(contextual_prompt) <= _PLANNER_CONTEXT_CAP and (
        not request or contextual_prompt.endswith(request)
    ):
        return contextual_prompt
    history = (
        contextual_prompt[:-len(request)]
        if request and contextual_prompt.endswith(request)
        else contextual_prompt + ("\n\n" if request and contextual_prompt else "")
    )
    available = _PLANNER_CONTEXT_CAP - len(request)
    if len(history) <= available:
        return history + request
    excerpt_budget = available - len(_PLANNER_OMISSION_MARKER)
    if excerpt_budget <= 0:
        return request
    tail_reserve = min(200, excerpt_budget // 2)
    head = history[:excerpt_budget - tail_reserve]
    tail = history[-tail_reserve:] if tail_reserve else ""
    return head + _PLANNER_OMISSION_MARKER + tail + request


def _build_continuation_prompt(
    contextual_prompt: str, request: str, draft: str, followup: str, *, draft_limit: int,
) -> str:
    """Retain history and evidence once, with the exact current request last.

    The flexible planner recognizes the request as a suffix when budgeting
    historical context. Inserting the previous draft between the original
    request and its follow-up defeats that recognition and appends the whole
    request a second time. Keep its evidence before the combined request.
    """
    history = (contextual_prompt[:-len(request)]
               if request and contextual_prompt.endswith(request)
               else contextual_prompt + ("\n\n" if contextual_prompt else ""))
    return (
        history
        + "[Previous draft — continue its work and return the completed answer]:\n"
        + flexible.bounded_draft_excerpt(draft, limit=draft_limit)
        + "\n\nNew message:\n"
        + request + followup
    )


def _build_contextual_prompt(
    prompt: str,
    session_data: dict | None,
    colony_items: list[str] | None = None,
    budget: int = workspace_mod.DEFAULT_HISTORY_BUDGET,
) -> str:
    """Prepend colony + session history to the prompt for full context.

    Block order: [Colony] (workspace-shared) → [Long-term Memory]
    (session-scoped) → [Session Summary] → [Recent Messages] → user prompt.
    """
    data = session_data
    if data is None:
        data = {}
    summary: str | None = data.get("summary")
    if not isinstance(summary, str):
        summary = None
    raw_long_term = data.get("long_term") or []
    long_term: list[str] = (
        [item for item in raw_long_term if isinstance(item, str) and item]
        if isinstance(raw_long_term, list) else []
    )
    raw_messages = data.get("messages", [])
    messages: list[dict] = (
        [message for message in raw_messages if isinstance(message, dict)]
        if isinstance(raw_messages, list) else []
    )
    raw_colony = colony_items or []
    colony_list: list[str] = (
        [item for item in raw_colony if isinstance(item, str) and item]
        if isinstance(raw_colony, list) else []
    )

    if not summary and not messages and not long_term and not colony_list:
        return prompt

    parts = session.format_context_blocks(data, colony_list)

    parts.append(
        "New message:"
    )
    parts.append(prompt)

    full = "\n".join(parts)

    if len(full) > budget:
        keywords = _request_keywords(prompt)
        colony_list = _relevance_last(list(colony_list), keywords)
        long_term = _relevance_last(list(long_term), keywords)
        descriptors = [
            (
                lambda limit: _bounded_context_list_block(
                    "Colony", colony_list, lambda item: f"- {item}", limit,
                ),
                _bounded_context_list_block(
                    "Colony", colony_list, lambda item: f"- {item}",
                ),
            ),
            (
                lambda limit: _bounded_context_list_block(
                    "Long-term Memory", long_term, lambda item: f"- {item}",
                    limit,
                ),
                _bounded_context_list_block(
                    "Long-term Memory", long_term, lambda item: f"- {item}",
                ),
            ),
            (
                lambda limit: _bounded_context_block(
                    "Session Summary", summary or "", limit,
                ),
                _bounded_context_block("Session Summary", summary or ""),
            ),
            (
                lambda limit: _bounded_context_list_block(
                    "Recent Messages", messages, session.format_msg_line, limit,
                ),
                _bounded_context_list_block(
                    "Recent Messages", messages, session.format_msg_line,
                ),
            ),
        ]
        descriptors = [
            (render, natural) for render, natural in descriptors if natural
        ]
        continuation = (
            "New message:"
        )
        # Keep the request intact.
        fixed_length = len(continuation) + 1 + len(prompt)
        if fixed_length > budget:
            return prompt
        context_budget = max(
            0,
            budget - fixed_length - len(descriptors),
        )
        quotas = _context_quotas(
            [len(natural) for _render, natural in descriptors], context_budget,
        )
        history_parts = [
            render(quota)
            for (render, _natural), quota in zip(descriptors, quotas, strict=True)
            if quota
        ]
        history_parts = [part for part in history_parts if part]
        history_parts.extend((continuation, prompt))
        full = "\n".join(history_parts)

    return full


class BackendUnavailableError(Exception):
    """A backend's CLI is not installed on this machine."""

    def __init__(self, backend) -> None:
        super().__init__(f"{backend.executable} CLI not found on PATH.")


def _take_pending_injections(
    inject_queue: asyncio.Queue[str] | None,
    injected: list[str] | None,
) -> bool:
    """Collect injections that landed while a phase was winding down.

    A backend watcher is cancelled during normal teardown.  A user message
    can arrive in that small handoff window, after the watcher stopped but
    before the next phase (or final return) takes over.  Treat it as an
    interrupt rather than silently dropping an already acknowledged
    ``/inject``.
    """
    pending: list[str] = []
    _drain_queue(inject_queue, collect=pending)
    if not pending:
        return False
    if injected is not None:
        injected.extend(pending)
    return True


def _close_inject_queue(inject_queue: asyncio.Queue[str] | None) -> None:
    """Stop a bot-owned injection channel from accepting late messages.

    The runtime also accepts ordinary :class:`asyncio.Queue` instances in
    tests and programmatic callers.  Its private queue subclass exposes an
    idempotent synchronous ``close`` method; duck-typing keeps that optional
    lifecycle hook out of the public agent API.
    """
    if inject_queue is None:
        return
    close = getattr(inject_queue, "close", None)
    if callable(close):
        close()


def _error_result(
    session_id: str | None,
    message: str,
    inject_queue: asyncio.Queue[str] | None,
) -> AgentResult:
    """Build a terminal BackendUnavailableError result with the queue drained."""
    result = AgentResult()
    set_error_result(result, message)
    result.session_id = session_id
    _close_inject_queue(inject_queue)
    _drain_queue(inject_queue)
    return result


async def _announce_restart(
    backend_name: str,
    injected: list[str],
    inject_queue: asyncio.Queue[str] | None,
    on_event: Callable[[ChatEvent], Awaitable[None]] | None,
) -> None:
    """Drain a restart's pending injects and announce the rebuild."""
    _drain_queue(inject_queue, collect=injected)
    logger.info(
        "Restarting %s with %d injected message(s)",
        backend_name, len(injected),
    )
    if on_event is not None:
        await on_event(ChatEvent(
            kind="tool",
            content="Restarting with injected context...",
        ))


def _build_backend_prompt(
    backend,
    contextual_prompt: str,
    *,
    collaborative: bool,
    allow_detached_requests: bool = True,
    skills_catalog: str = "",
) -> str:
    """Wrap a prompt in the preamble and plugin list *backend* needs.

    For backends that can't be handed typed tool definitions (CLI
    subprocess agents whose toolset is fixed by the CLI), user plugins are
    enumerated in the prompt so the model can invoke them via its own
    bash/shell tool. HTTP backends with typed tools see plugins via
    TOOL_SCHEMA. Chat-only HTTP backends opt out via
    supports_plugin_prelude=False, since they have no shell to invoke the
    prelude'd commands either.
    """
    parts = [_capability_hint(
        collaborative=collaborative,
        allow_detached_requests=(
            allow_detached_requests
            and bool(getattr(backend, "supports_detached_tasks", False))
        ),
    )]
    if not backend.supports_typed_plugins and backend.supports_plugin_prelude:
        prelude = agent_tools.cli_plugin_prelude()
        if prelude:
            parts.append(prelude)
    if skills_catalog and getattr(backend, "name", None) in _NATIVE_BACKENDS:
        parts.append(skills_catalog)
    if contextual_prompt and getattr(backend, "name", None) in _NATIVE_BACKENDS:
        parts.append(skills.refresh_instructions(cli=True))
    parts.append(contextual_prompt)
    return "\n\n".join(parts)


_NATIVE_BACKENDS = frozenset({"codex", "claude_code", "copilot", "grok", "gemini_cli"})


class _NativeHangRecovered(Exception):
    """The owned CLI stopped after a confirmed hang; its native state survives."""

    def __init__(self, diagnosis: str, partial: AgentResult):
        super().__init__(diagnosis)
        self.diagnosis = diagnosis
        self.partial = partial


class _NativeRecoveryDeferred(Exception):
    """A worker keeps its plan position while repeated recovery backs off."""

    def __init__(self, result: AgentResult):
        super().__init__(result.error)
        self.result = result


def _note_native_progress(proc: Any, event: dict) -> None:
    """Observe actual native work independently of the status presentation."""
    from . import supervision

    cursors = getattr(proc, "_cozter_progress_cursors", None)
    if not isinstance(cursors, dict):
        cursors = {}
        setattr(proc, "_cozter_progress_cursors", cursors)

    def note(key: str, value: object) -> None:
        if value is None:
            return
        text = value if isinstance(value, str) else str(value)
        if not text.strip():
            return
        digest = json_digest(text)
        if cursors.get(key) == digest:
            return
        if len(cursors) >= 128:
            cursors.pop(next(iter(cursors)))
        cursors[key] = digest
        supervision.note_output(text[-12000:])

    event_type = event.get("type")
    if event_type == "message":
        content = event.get("content")
        if event.get("role") == "assistant" and isinstance(content, str) and content.strip():
            supervision.note_output(content[-12000:])
        return
    item = event.get("item")
    if isinstance(item, dict) and event_type in {"item.started", "item.updated", "item.completed"}:
        item_type = item.get("type")
        if item_type == "command_execution":
            command = item.get("command")
            if isinstance(command, str):
                supervision.attach_process(proc, command=command[:4000])
            output = item.get("aggregated_output")
            if isinstance(output, str) and output.strip():
                note(f"command-output:{item.get('id', '')}", output)
            if event_type == "item.completed":
                note(f"command-complete:{item.get('id', '')}", "command completed")
        elif event_type == "item.completed" and item_type in {"agent_message", "file_change"}:
            note(f"item:{item.get('id', '')}", item.get("text") or item.get("changes"))
        return
    if event_type in {"assistant", "user"}:
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and content.strip():
            note("message-text", content)
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "tool_use":
                    tool_input = block.get("input")
                    command = tool_input.get("command") if isinstance(tool_input, dict) else None
                    tool = block.get("name", "tool")
                    supervision.attach_process(proc, command=str(command or tool)[:4000])
                elif block_type in {"text", "tool_result"}:
                    value = block.get("text") if block_type == "text" else block.get("content")
                    if value:
                        note(f"message:{block_type}:{block.get('tool_use_id', '')}", value)
        return
    if isinstance(event_type, str) and (event_type.startswith("tool.") or event_type in {
        "tool_use", "tool_call", "tool_start", "tool", "tool_result", "tool_output", "tool_end",
    }):
        payload = event.get("data")
        payload = payload if isinstance(payload, dict) else event
        command = payload.get("command") or payload.get("toolName") or payload.get("name") or payload.get("tool_name")
        arguments = payload.get("arguments") or payload.get("input") or payload.get("parameters")
        if isinstance(arguments, dict):
            command = arguments.get("command") or command
        if command:
            supervision.attach_process(proc, command=str(command)[:4000])
        output = payload.get("output") or payload.get("result")
        tool_id = payload.get("toolCallId") or payload.get("tool_id") or ""
        if output:
            note(f"tool-output:{tool_id}", output)
        if event_type in {"tool_result", "tool_end", "tool.execution_complete"}:
            note(f"tool-complete:{tool_id}", "tool completed")
    elif event_type in {"assistant.message", "result"}:
        payload = event.get("data")
        payload = payload if isinstance(payload, dict) else event
        text = payload.get("content") or payload.get("text") or payload.get("result")
        if isinstance(text, str) and text.strip():
            note("final-text", text)


def _native_turn_prompt(
    backend, request: str, session_data: dict, colony_items: list[str],
    *, budget: int, collaborative: bool, is_scheduled: bool,
    bootstrap_request: str | None = None, previous_draft: str | None = None,
    skills_catalog: str = "",
    budget_includes_request: bool = True,
    history_record: execution_state.ExecutionRecord | None = None,
    _prepare: bool = True,
) -> native_prompt.NativePrompt:
    """Budget named context before choosing bootstrap or exact-session delta."""
    memory = session_data.get("long_term", [])
    if not isinstance(memory, list):
        memory = []
    memory = [item for item in memory if isinstance(item, str) and item]
    colony_items = [item for item in colony_items if isinstance(item, str) and item]
    summary = session_data.get("summary", "")
    if not isinstance(summary, str):
        summary = ""
    messages = session_data.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    marks = native_prompt.history_snapshot(session_data)
    start = session.total_message_count(session_data) - len(messages)
    ledger = native_prompt.matching_ledger(history_record) or {}
    history = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        coordinate = str(start + index)
        digest = marks[coordinate]
        text = session.format_msg_line(message)
        complete = text == session.format_msg_line(message, cap=None)
        if not native_prompt.message_is_known_from_ledger(
            ledger, coordinate, digest, text,
        ):
            history.append((coordinate, digest, text, complete))
    descriptors = [
        ("colony", lambda limit: _bounded_context_list_block(
            "Colony", colony_items, lambda item: f"- {item}", limit)),
        ("memory", lambda limit: _bounded_context_list_block(
            "Long-term Memory", memory, lambda item: f"- {item}", limit)),
        ("summary", lambda limit: _bounded_context_block("Session Summary", summary, limit)),
    ]
    known_sections = ledger.get("sections", {})
    if not isinstance(known_sections, dict):
        known_sections = {}
    known_full_sections = {
        name for name, render in descriptors
        if known_sections.get(name) == json_digest([render(None), False])
    }
    lengths = [0 if name in known_full_sections else len(render(None)) for name, render in descriptors]
    lengths.append(sum(len(text) + 1 for _, _, text, _ in history))
    execution = execution_state.current_execution()
    request_identity = execution.execution_id if execution is not None else None
    request_view = native_prompt.NativePrompt(
        request, {}, bootstrap_request=bootstrap_request, request_identity=request_identity,
    ).request_text_for(history_record)
    request_cost = len(request_view) + 32 if budget_includes_request else 0
    quotas = _context_quotas(lengths, max(0, budget - request_cost))
    sections = {"policy and plugins": _build_backend_prompt(
        backend, "", collaborative=collaborative, allow_detached_requests=not is_scheduled,
    ).strip()}
    if skills_catalog:
        sections["skills catalog"] = skills_catalog
    partial_sections: set[str] = set()
    for index, (name, render) in enumerate(descriptors):
        natural = render(None)
        excerpt = natural if name in known_full_sections else render(quotas[index])
        if natural and excerpt != natural:
            partial_sections.add(name)
            if not excerpt:
                excerpt = (f"[{name} omitted by this turn's history budget; "
                           "retain previously supplied context.]")
        sections[name] = excerpt
    selected: list[tuple[str, str, str]] = []
    message_sources: dict[str, tuple[str, bool]] = {}
    remaining = quotas[-1]
    for coordinate, digest, text, complete in reversed(history):
        if len(text) + 1 > remaining:
            if not selected and remaining:
                excerpt = _truncate_context_text(text, remaining)
                if native_prompt.message_is_known_from_ledger(
                    ledger, coordinate, digest, excerpt,
                ):
                    continue
                selected.append((coordinate, native_prompt.delivery_fingerprint(digest, excerpt), excerpt))
                message_sources[coordinate] = (digest, False)
            break
        selected.append((coordinate, native_prompt.delivery_fingerprint(digest, text), text))
        message_sources[coordinate] = (digest, complete)
        remaining -= len(text) + 1
    selected.reverse()
    result = native_prompt.NativePrompt(
        request, sections, selected, bootstrap_request=bootstrap_request,
        bootstrap_sections={"previous draft": (
            "[Previous draft and evidence; retain completed work]\n"
            + flexible.bounded_draft_excerpt(previous_draft)
        )} if previous_draft else {},
        partial_sections=partial_sections,
        message_sources=message_sources,
        request_identity=request_identity,
        turn_instructions=skills.refresh_instructions(cli=True),
    )
    if _prepare:
        def prepare(record: execution_state.ExecutionRecord | None) -> native_prompt.NativePrompt:
            prepared = _native_turn_prompt(
                backend, request, session_data, colony_items,
                budget=budget, collaborative=collaborative, is_scheduled=is_scheduled,
                bootstrap_request=bootstrap_request, previous_draft=previous_draft,
                skills_catalog=skills_catalog, budget_includes_request=budget_includes_request,
                history_record=record, _prepare=False,
            )
            # Flexible appends role-specific sections after building shared context.
            prepared.sections.update({name: text for name, text in result.sections.items()
                                      if name not in {"colony", "memory", "summary"}})
            prepared.request_identity = result.request_identity
            prepared.turn_instructions = result.turn_instructions
            return prepared
        result.prepare = prepare
    return result


async def _drive_backend(
    backend,
    workspace_path: str,
    full_prompt: str,
    model: str | None,
    approval: str,
    *,
    effort: int,
    on_event: Callable[[ChatEvent], Awaitable[None]] | None = None,
    inject_queue: asyncio.Queue[str] | None = None,
    injected: list[str] | None = None,
    close_inject_on_completion: bool = False,
    on_process: Callable[[Any], None] | None = None,
    native_context: native_prompt.NativePrompt | None = None,
    native_worker: str | None = None,
) -> tuple[AgentResult, bool]:
    async with execution_state.backend_scope(
        backend.name, model or getattr(backend, "default_model", None), approval, effort,
        native=backend.name in _NATIVE_BACKENDS, native_worker=native_worker,
    ) as state:
        context_template = native_context if backend.name in _NATIVE_BACKENDS else None
        prompt_context = context_template.for_record(state) if context_template is not None else None

        async def invoke(
            prompt_text: str, delivery: native_prompt.NativePrompt | None,
            delivered_request: str | None,
        ) -> tuple[AgentResult, bool]:
            recoveries = 0
            retained = AgentResult()
            turn_instructions = delivery.turn_instructions if delivery is not None else ""
            while True:
                try:
                    result, restarting = await attempt(prompt_text, delivery, delivered_request)
                    carry_forward_turn_output(retained, result, workspace_path)
                    return result, restarting
                except _NativeHangRecovered as exc:
                    carry_forward_turn_output(exc.partial, retained, workspace_path)
                    recoveries += 1
                    if _take_pending_injections(inject_queue, injected):
                        return retained, True
                    diagnosis = exc.diagnosis[:8000]
                    guidance = (
                        "[Automatic hang recovery: the previous owned CLI and its child processes "
                        "were stopped after a confirmed lack of useful progress. Inspect the current "
                        "workspace and preserved native conversation, diagnose and repair the cause, "
                        "then finish the SAME job. Preserve completed work and valid evidence; "
                        "check commands with unknown outcomes before repeating them.]\n" + diagnosis
                    )
                    if recoveries >= 3:
                        retained.error = "Repeated native hangs require another recovery attempt: " + diagnosis
                        retained.continuation_reason = "prerequisite"
                        retained.continue_instruction = guidance
                        retained.blocked_reason = retained.error
                        return retained, False
                    if delivery is not None:
                        delivery = dataclasses.replace(
                            delivery, turn_instructions="\n\n".join(filter(None, (
                                turn_instructions, guidance,
                            ))),
                        )
                        prompt_text = delivery.render(state)
                        delivered_request = delivery.request_for(state)
                    else:
                        prompt_text = full_prompt + "\n\n" + guidance
                    logger.warning("%s continuing its saved native session after confirmed hang", backend.name)

        async def attempt(
            prompt_text: str, delivery: native_prompt.NativePrompt | None,
            delivered_request: str | None,
        ) -> tuple[AgentResult, bool]:
            acknowledged = False
            if state is not None and prompt_context is not None:
                await state.save(native_tools_started=False)

            async def acknowledge_activity() -> None:
                nonlocal acknowledged
                if (acknowledged or state is None or delivery is None
                        or not state.get("native_tools_started")
                        or execution_state.native_session_id(state.get("native_session_id")) is None):
                    return
                await delivery.acknowledge(state, delivered_request=delivered_request)
                acknowledged = True
                execution = execution_state.current_execution()
                if execution is not None and delivered_request is not None:
                    execution.remember_native_request(state, delivered_request)

            try:
                return await _drive_backend_impl(
                    backend, workspace_path, prompt_text, model, approval, effort=effort,
                    on_event=on_event, inject_queue=inject_queue, injected=injected,
                    close_inject_on_completion=False, on_process=on_process,
                    on_native_activity=acknowledge_activity if delivery is not None else None,
                )
            except NativeSessionUnavailableError as exc:
                if (state is None or exc.backend_name != backend.name
                        or state.get("native_unavailable_session_id") != exc.native_session_id
                        or not state.get("native_resume_unavailable")
                        or state.get("native_tools_started")):
                    raise
                return AgentResult(error="The exact native session is unavailable."), False
            finally:
                await acknowledge_activity()

        result, restarting = await invoke(
            prompt_context.render(state) if prompt_context is not None else full_prompt,
            prompt_context, prompt_context.request_for(state) if prompt_context is not None else None,
        )
        failure_text = (result.error or "").strip()
        if (not restarting and state is not None and state.get("native_resume_unavailable")
                and not state.get("native_tools_started")
                and not any(event.kind in {"tool", "file", "attachment"}
                            or event.kind == "text" and event.content.strip() not in {
                                failure_text, f"Error: {failure_text}",
                            } for event in result.events) and not result.detached_tasks):
            # Only a provider-confirmed missing exact session, before any
            # activity, permits one fresh launch. Never retry ambiguous work.
            await state.save(native_resume_unavailable=False, native_session_id=None)
            logger.info("%s exact native session unavailable before work; recovering from saved evidence", backend.name)
            bootstrap_context = context_template.for_record(None) if context_template is not None else None
            bootstrap = bootstrap_context.render(None) if bootstrap_context is not None else full_prompt
            result, restarting = await invoke(
                bootstrap + "\n\n[The exact provider session was unavailable before any new work began. "
                "Inspect existing workspace changes and saved evidence; continue unfinished work "
                "without repeating completed changes or commands with an unknown outcome.]",
                bootstrap_context,
                bootstrap_context.request_for(None) if bootstrap_context is not None else None,
            )
        if not restarting and _take_pending_injections(inject_queue, injected):
            restarting = True
        if not restarting and close_inject_on_completion:
            _close_inject_queue(inject_queue)
        return result, restarting


async def _drive_backend_impl(
    backend,
    workspace_path: str,
    full_prompt: str,
    model: str | None,
    approval: str,
    *,
    effort: int,
    on_event: Callable[[ChatEvent], Awaitable[None]] | None = None,
    inject_queue: asyncio.Queue[str] | None = None,
    injected: list[str] | None = None,
    close_inject_on_completion: bool = False,
    on_process: Callable[[Any], None] | None = None,
    on_native_activity: Callable[[], Awaitable[None]] | None = None,
) -> tuple[AgentResult, bool]:
    from . import supervision

    ticket = (supervision.new_operation(
        workspace_path, "native", details=f"backend={backend.name} model={model or 'default'}",
    ) if backend.name in _NATIVE_BACKENDS else None)
    partial = AgentResult()
    with supervision.operation_scope(ticket):
        call = _drive_backend_process(
            backend, workspace_path, full_prompt, model, approval, effort=effort,
            on_event=on_event, inject_queue=inject_queue, injected=injected,
            close_inject_on_completion=close_inject_on_completion, on_process=on_process,
            on_native_activity=on_native_activity, partial=partial,
        )
        if ticket is None:
            return await call
        task = asyncio.create_task(call, name=f"native-worker:{backend.name}")
        try:
            diagnosis = await supervision.wait_owned(task, ticket)
        finally:
            await cancel_and_settle(task)
        if diagnosis is not None:
            raise _NativeHangRecovered(diagnosis, partial)
        return task.result()


async def _drive_backend_process(
    backend,
    workspace_path: str,
    full_prompt: str,
    model: str | None,
    approval: str,
    *,
    effort: int,
    on_event: Callable[[ChatEvent], Awaitable[None]] | None,
    inject_queue: asyncio.Queue[str] | None,
    injected: list[str] | None,
    close_inject_on_completion: bool,
    on_process: Callable[[Any], None] | None,
    on_native_activity: Callable[[], Awaitable[None]] | None,
    partial: AgentResult,
) -> tuple[AgentResult, bool]:
    """Launch *backend*, stream its events, and collect the result.

    Returns ``(result, restarting)``. *restarting* is True when a message
    arrived on *inject_queue* mid-run: the subprocess was killed, the
    message appended to *injected*, and the caller should rebuild the
    prompt and drive the backend again.

    *on_process* is an optional non-sensitive progress hook: it is
    called synchronously with the launched process object right after
    launch (and, for flexible workers, once per worker launch) so the
    caller can record the backend PID for stuck-wait attribution. The
    hook receives only the process object — never prompts or arguments.

    Raises :exc:`BackendUnavailableError` when the CLI isn't installed.
    """
    from . import supervision

    try:
        proc = await backend.launch(
            workspace_path, full_prompt, model, approval, effort=effort,
        )
    except FileNotFoundError as e:
        raise BackendUnavailableError(backend) from e
    except (OSError, RuntimeError) as exc:
        # Surface launch failures instead of stranding queued work.
        result = AgentResult()
        set_error_result(result, f"{backend.name} could not start: {exc}")
        # A pre-failure /inject still counts.
        if _take_pending_injections(inject_queue, injected):
            return result, True
        if close_inject_on_completion:
            _close_inject_queue(inject_queue)
        return result, False

    if on_process is not None:
        try:
            on_process(proc)
        except Exception:
            logger.debug("on_process hook failed", exc_info=True)

    if backend.name in _NATIVE_BACKENDS:
        supervision.attach_process(proc)
    result = partial
    restarting = False
    completed_stream = False
    cancelled = False
    stderr_task = asyncio.create_task(drain_text_stream(proc.stderr))

    def _log_non_json_line(line: str) -> None:
        logger.debug("Non-JSON line: %s", line)

    # Watch inject_queue: kill the subprocess on arrival.
    async def _watch_inject(
        active_proc: asyncio.subprocess.Process = proc,
    ) -> None:
        nonlocal restarting
        assert inject_queue is not None  # only scheduled when set
        msg = await inject_queue.get()
        if injected is not None:
            injected.append(msg)
        restarting = True
        # Kill the process group with grandchildren (async: taskkill
        # runs off-loop so the watcher never blocks its own loop).
        await terminate_process_group_async(active_proc)

    inject_task: asyncio.Task | None = None
    if inject_queue is not None:
        inject_task = asyncio.create_task(_watch_inject())

    assert proc.stdout is not None  # spawned with stdout=PIPE
    try:
        async for event in iter_process_json_events(
            proc,
            on_invalid=_log_non_json_line,
            preserve_process_tree=lambda: bool(result.detached_tasks),
        ):
            prev_count = len(result.events)
            backend.parse_event(event, result)
            if backend.name in _NATIVE_BACKENDS:
                _note_native_progress(proc, event)

            if on_native_activity is not None:
                await on_native_activity()

            if on_event:
                for ev in result.events[prev_count:]:
                    await on_event(ev)

        await wait_for_process_exit(proc)
        completed_stream = True
    except asyncio.CancelledError:
        cancelled = True
        budget = _TURN_BUDGET.get()
        if budget is not None:
            budget.capture(result, workspace_path)
        logger.info(
            "%s run cancelled, killing subprocess %d",
            backend.name, proc.pid,
        )
        raise
    finally:
        # Never leak the drain task.
        try:
            # No durable provider task: tear down the whole tree.
            if (
                proc.returncode is None
                or cancelled
                or not completed_stream
                or restarting
            ):
                await kill_and_wait(proc)
            if inject_task and not inject_task.done():
                inject_task.cancel()
                await await_cancelled(inject_task)
            stderr = await finish_process_stderr(
                proc,
                stderr_task,
                preserve_process_tree=lambda: bool(result.detached_tasks),
            )
        except BaseException:
            # Late teardown failure is not a clean turn.
            if has_managed_process_group(proc):
                await terminate_process_group_async(proc)
            raise
        finally:
            # Teardown can be cancelled; never leak watchers.
            if inject_task and not inject_task.done():
                inject_task.cancel()
                await await_cancelled(inject_task)
            if not stderr_task.done():
                await abandon_subprocess_stream_task(proc, 2, stderr_task)
            await cleanup_backend_process(backend, proc, log=logger)
    if stderr:
        logger.debug("%s stderr: %s", backend.name, stderr)

    budget = _TURN_BUDGET.get()
    if budget is not None:
        budget.capture(result, workspace_path)
    if restarting:
        return result, True

    # Final drain: accepted messages survive teardown.
    if _take_pending_injections(inject_queue, injected):
        # Watcher stopped; terminate before rebuilding the prompt.
        if has_managed_process_group(proc):
            await terminate_process_group_async(proc)
        return result, True
    if close_inject_on_completion:
        _close_inject_queue(inject_queue)

    # Completed turn: stop same-group children.
    if has_managed_process_group(proc) and not result.detached_tasks:
        await terminate_process_group_async(proc)

    if completed_stream and proc.returncode not in (0, None):
        observe_exit = getattr(backend, "observe_process_exit", None)
        if callable(observe_exit):
            observe_exit(proc.returncode, stderr)

    # A partial reply never turns a failed run green.
    # Keep provider errors more specific than a generic exit status.
    if proc.returncode != 0 and not result.error:
        msg = f"{backend.name} exited with code {proc.returncode}"
        if stderr:
            msg += f"\n{stderr}"
        if result.text:
            record_backend_error(result, msg)
        else:
            set_error_result(result, msg, display_text=msg)

    return result, False


async def _run_with_inject_watch(
    coro: Awaitable[str | None],
    inject_queue: asyncio.Queue[str] | None,
    injected: list[str] | None,
    *,
    close_inject_on_completion: bool = False,
) -> tuple[str | None, bool]:
    """Run *coro* (an internal-backend call) but abandon it if a message
    arrives on *inject_queue* first.

    The flexible planner and merge steps call the summary backend directly
    rather than through :func:`_drive_backend`, so they don't watch the inject
    queue. Without this, an ``/inject`` sent during those windows (the merge in
    particular can run for minutes) would sit in the queue unread and then be
    discarded when the turn completes - even though ``cmd_inject`` already told
    the user "Injected." Racing the call against the queue lets the inject
    abandon the pipeline and replan, matching how the worker steps behave.

    Returns ``(text, restarting)``; *text* is None when *restarting* is True.
    When *inject_queue* is None the call runs unwatched (unchanged behavior).
    """
    call_task: asyncio.Task[str | None] = asyncio.ensure_future(coro)
    if inject_queue is None:
        return await call_task, False
    watch_task: asyncio.Task[str] = asyncio.ensure_future(inject_queue.get())
    try:
        await asyncio.wait(
            {call_task, watch_task}, return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError:
        call_task.cancel()
        watch_task.cancel()
        await await_cancelled(call_task)
        await await_cancelled(watch_task)
        raise
    if watch_task.done() and not watch_task.cancelled():
        # Inject arrived first: abandon the call.
        msg = watch_task.result()
        if injected is not None:
            injected.append(msg)
        call_task.cancel()
        try:
            await call_task
        except asyncio.CancelledError:
            pass
        except Exception:
            # Same-tick race: inject wins.
            logger.debug(
                "Discarding internal backend failure after an inject",
                exc_info=True,
            )
        return None, True
    # Call finished first: stop watching.
    watch_task.cancel()
    await await_cancelled(watch_task)
    # A message can arrive after cancel; collect and restart.
    if _take_pending_injections(inject_queue, injected):
        # Late injects win over finished failures.
        try:
            call_task.result()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug(
                "Discarding internal backend failure after an inject",
                exc_info=True,
            )
        return None, True
    if close_inject_on_completion:
        # Terminal phase: close before I/O.
        _close_inject_queue(inject_queue)
    return call_task.result(), False


# Flexible: plan, route, merge; usage summed.
_USAGE_TOTAL_FIELDS = ("input_tokens", "output_tokens", "total_cost_usd")


def _accumulate_usage(totals: dict, usage: dict | None) -> None:
    if not isinstance(usage, dict):
        return
    for field in _USAGE_TOTAL_FIELDS:
        value = usage.get(field)
        if _valid_usage_number(value):
            previous = totals.get(field, 0)
            if not _valid_usage_number(previous):
                previous = 0
            combined = previous + value
            if _valid_usage_number(combined):
                totals[field] = combined


def carry_forward_turn_output(
    previous: AgentResult, current: AgentResult, workspace_path: str,
) -> None:
    """Retain artifacts and accounting when a continuation replaces a draft."""
    _consume_detached_task_requests(previous)
    retained_events = [event for event in previous.events if event.kind != "text"]
    retained_sources = {
        attachment_source_path(event.content, workspace_path)
        for event in retained_events if event.kind == "attachment"
    }
    for source in sorted(_explicit_attachment_sources(previous.events, workspace_path)):
        if source not in retained_sources:
            retained_events.append(ChatEvent(kind="attachment", content=source))
    current.events = retained_events + current.events
    current.detached_tasks = previous.detached_tasks + current.detached_tasks
    current.detached_task_requests = (
        previous.detached_task_requests + current.detached_task_requests
    )
    usage_totals: dict = {}
    _accumulate_usage(usage_totals, previous.usage)
    _accumulate_usage(usage_totals, current.usage)
    current.usage = usage_totals or None


def _split_attach_markers(text: str) -> tuple[str, list[str]]:
    """Pull ``[[attach: ...]]`` markers out of a worker's report.

    Workers' text never reaches the user - only the merged answer does -
    so their attachment markers have to be carried over by hand or the
    files they meant to send would be dropped along with the text.
    """
    if not isinstance(text, str):
        return "", []
    ranges = list(_iter_attachment_markers(text))
    if not ranges:
        return text, []
    markers = [text[start:end] for start, end, _capture in ranges]
    cleaned = _remove_protocol_markers(text, ranges)
    return _collapse_extra_blank_lines(cleaned), markers


def _retain_blocked_verdict(
    result: AgentResult, verdict: flexible.JudgeVerdict,
    *, turn: "_InterruptedTurn | None" = None,
) -> None:
    """Retain unfinished work for automatic prerequisite recovery and retry."""
    result.completion_verified = False
    result.blocked_reason = verdict.missing
    result.continuation_reason = "prerequisite"
    result.continue_instruction = (
        "Continue the SAME unfinished job automatically. First recover or recheck "
        "the remaining prerequisite using the current authorized tools and host "
        "capabilities; routine recovery does not need permission to continue. "
        "Preserve completed changes and valid build/test evidence. Do not repeat "
        "whole audits or builds merely because this is a retry. If recovery is "
        "still unavailable, report the exact prerequisite for automatic retry; "
        "never fabricate credentials, required user information, or completion.\n"
        f"Remaining prerequisite: {verdict.missing}"
    )
    explanation = f"PARTIAL: recovering a required prerequisite automatically: {verdict.missing}"
    if verdict.next_instruction:
        result.continue_instruction += f"\nRecovery target: {verdict.next_instruction}"
        explanation += f"\nRecovery target: {verdict.next_instruction}"
    append_text_result(result, "\n\n".join(filter(None, (result.text, explanation))))
    budget = _TURN_BUDGET.get()
    if budget is not None and turn is not None:
        budget.pending = [result.continue_instruction]
        result.continuation_checkpoint = budget.checkpoint(
            "A required prerequisite needs automatic recovery or a later availability check.", turn,
        )
    performance_logger.info("stage=judge outcome=blocked automatic_continuation=true")


async def _judge_draft(
    request: str,
    draft: str,
    workspace_path: str,
    *,
    summary_backend_name: str,
    summary_model: str | None,
    inject_queue: asyncio.Queue[str] | None,
    injected: list[str] | None,
    on_event: Callable[[ChatEvent], Awaitable[None]] | None = None,
    round_no: int = 1,
) -> tuple[flexible.JudgeVerdict, bool]:
    """Ask the summary backend whether *draft* fully answers *request*.

    Universal continue-judge: every backend and every turn (direct and
    flexible) routes through here. Returns ``(verdict, restarting)`` like
    :func:`_drive_backend` - an inject mid-judge abandons the verdict and
    the caller restarts the turn. ``[[await]]`` drafts, empty drafts, and
    judge failures retain the draft without verifying completion. The
    caller can pause the queued request until verification is available.
    """
    cleaned, awaiting = extract_await(draft or "")
    if awaiting or not cleaned.strip():
        return flexible.JudgeVerdict(should_continue=False, verified=False), False
    try:
        summary_backend = backends_agent.get_backend(summary_backend_name)
    except (ValueError, AttributeError, TypeError) as exc:
        logger.warning("Continue judge unavailable (%s) - retaining unverified draft", exc)
        return flexible.JudgeVerdict(should_continue=False, verified=False), False
    if on_event is not None:
        await on_event(ChatEvent(
            kind="tool",
            content=f"judge: round {round_no} with"
            f" {summary_backend.name}/{summary_model}",
        ))
    from . import supervision

    try:
        with supervision.worker_scope(
            workspace_path, summary_backend_name=summary_backend_name, summary_model=summary_model,
        ):
            raw, restarting = await _run_with_stage_timing(_run_with_inject_watch(
                run_internal_backend(
                    summary_backend,
                    workspace_path,
                    flexible.build_judge_prompt(request, cleaned),
                    summary_model,
                    label="Continue judge",
                    log=logger,
                    missing_executable_message=(
                        "%s CLI not found - retaining the draft without verifying completion"
                    ),
                    missing_level=logging.WARNING,
                ),
                inject_queue,
                injected,
            ), "judge", summary_backend.name, summary_model)
    except (AttributeError, TypeError, OSError, RuntimeError) as exc:
        logger.warning("Continue judge unavailable (%s) - retaining unverified draft", exc)
        return flexible.JudgeVerdict(should_continue=False, verified=False), False
    if restarting:
        return flexible.JudgeVerdict(should_continue=False, verified=False), True
    if raw is None:
        return flexible.JudgeVerdict(should_continue=False, verified=False), False
    return flexible.parse_judge(raw or ""), False


async def _run_parallel_flexible_workers(
    calls: list[Awaitable[tuple[AgentResult, bool]]],
    inject_queue: asyncio.Queue[str] | None,
    injected: list[str],
) -> tuple[list[tuple[AgentResult, bool]], bool]:
    """Own a reader wave's tasks and its single injection channel.

    Stop, a deadline, a failed launch, or an injection cancels and awaits
    every sibling, including subprocess cleanup, before returning control.
    """
    tasks = [asyncio.ensure_future(call) for call in calls]
    group = asyncio.gather(*tasks)

    async def watch() -> None:
        assert inject_queue is not None
        injected.append(await inject_queue.get())

    watcher = asyncio.create_task(watch()) if inject_queue is not None else None
    try:
        if watcher is not None:
            waiters: set[asyncio.Future[Any]] = {group, watcher}
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            if watcher.done():
                watcher.result()
                return [], True
            watcher.cancel()
            await await_cancelled(watcher)
            if _take_pending_injections(inject_queue, injected):
                return [], True
        return list(await group), False
    finally:
        if watcher is not None:
            watcher.cancel()
        for task in tasks:
            if not task.done():
                task.cancel()
        # Watch gather too when it fails or cancels.
        await asyncio.gather(group, *tasks, *([watcher] if watcher else []),
                             return_exceptions=True)


def _new_owner_obligation(instruction: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scope = json_digest(flexible.instruction_identity(instruction))
    generation = 1 + max((row["generation"] for row in rows if row["scope"] == scope), default=-1)
    return {"id": json_digest([scope, generation]), "scope": scope, "generation": generation,
            "instruction": instruction, "repeats": 0, "resolved": False}


async def _continue_flexible_owner(instruction: str, draft: str) -> bool:
    """Reopen the same owner with remaining work and a fresh phase outcome."""
    saved = await execution_state.record("flexible_plan")
    if saved is None or saved.get("owner_mode") is not True:
        return False
    tasks = flexible.restore_tasks(saved.get("queue"))
    if tasks is None or len(tasks) != 1 or tasks[0].read_only:
        raise ValueError("Saved Flexible owner is invalid")
    instruction = instruction.strip()
    if not instruction:
        return False
    obligations = {key: [dict(row) for row in rows]
                   for key, rows in saved.get("owner_obligations", {}).items()}
    rows = obligations.setdefault("0", [])
    identity = json_digest(flexible.instruction_identity(instruction))
    budget = _TURN_BUDGET.get()
    if budget is not None and instruction not in budget.pending:
        budget.pending.append(instruction)
    if not any(row["scope"] == identity and not row["resolved"] for row in rows):
        if len(rows) >= flexible.MAX_SUBTASKS * 4:
            raise _TurnBudgetExhaustedError("the owner obligation context is full; remaining scope is checkpointed")
        rows.append(_new_owner_obligation(instruction, rows))
    output = saved.get("output", {})
    errors = saved.get("worker_errors", [])
    audit = saved.get("owner_error_audit", [])
    if (not isinstance(output, dict) or not isinstance(errors, list)
            or not all(isinstance(error, str) for error in errors)
            or not isinstance(audit, list) or not all(isinstance(error, str) for error in audit)):
        raise ValueError("Saved Flexible owner outcome is invalid")
    if isinstance(output.get("error"), str) and output["error"] and output["error"] not in errors:
        errors.append(output["error"])
    audit = [flexible.truncate_report(error) for error in (audit + errors)[-8:]]
    output["error"] = None
    await saved.save(
        active=True, cursor=0, reports=[], worker_outputs={}, owner_obligations=obligations,
        phase_previous_draft=flexible.bounded_draft_excerpt(draft),
        worker_errors=[], blocked=[], output=output, owner_error_audit=audit,
    )
    return True


async def _run_flexible(
    contextual_prompt: str,
    request: str,
    workspace_path: str,
    *,
    approval: str,
    effort: int,
    collaborative: bool,
    summary_backend_name: str,
    summary_model: str | None,
    on_event: Callable[[ChatEvent], Awaitable[None]] | None,
    inject_queue: asyncio.Queue[str] | None,
    injected: list[str],
    close_inject_on_completion: bool = True,
    previous_draft: str | None = None,
    on_process: Callable[[Any], None] | None = None,
    session_data: dict | None = None,
    colony_items: list[str] | None = None,
    history_budget: int | None = None,
) -> tuple[AgentResult, bool]:
    """Run one turn of the ``flexible`` meta-agent.

    Three phases: the summary agent understands the request and splits it
    into difficulty-graded sub-tasks; each sub-task runs as a full agent
    turn on the agent+model bound to its tier; the summary agent merges
    the workers' reports into the one reply the user sees.

    Returns ``(result, restarting)`` like :func:`_drive_backend` - an
    inject mid-pipeline abandons the run and replans with the new context.
    Continuations pass *previous_draft* to the workers as completion evidence.

    *on_process* is forwarded to each worker's :func:`_drive_backend`
    call so stuck-wait attribution follows the currently running worker
    process (planner/merge/judge phases run through
    :func:`run_internal_backend` without a foreground process to report).
    """
    from . import execution_state
    from .backends_agent.base import DetachedTaskRef

    summary_backend = backends_agent.get_backend(summary_backend_name)
    tiers = workspace_mod.get_flexible_run_config(workspace_path)
    tier_efforts = workspace_mod.get_flexible_reasoning_efforts(workspace_path, default=effort)
    effort_policy: int | dict[str, int] = (next(iter(tier_efforts.values()))
                                         if len(set(tier_efforts.values())) == 1 else tier_efforts)
    native_skills_catalog = skills.catalog_prompt(
        skills.current_catalog(workspace_path), cli=True,
    )
    shared_session = session_data if session_data is not None else {}
    shared_colony = colony_items if colony_items is not None else []
    shared_budget = (history_budget if history_budget is not None
                     else workspace_mod.get_history_budget(workspace_path))
    execution = execution_state.current_execution()
    original_request = execution.original_request if execution is not None else request
    signature_context = {
        "tiers": tiers, "approval": approval, "effort": effort_policy,
        "summary_backend": summary_backend_name, "summary_model": summary_model,
        "collaborative": collaborative,
    }

    def request_signature(original: str, additions: list[str]) -> str:
        return json_digest({
            "request": original, "injected": additions, **signature_context,
        })

    signature = request_signature(original_request, injected)
    folded_signatures = []
    if injected:
        # BotPlatform._run_turn moves accepted guidance into its canonical
        # request after an allowance ends. These are its two exact renderings:
        # normal checkpoint additions, or full session recovery when additions
        # were clipped. Record only proven equivalents; never parse or normalize
        # arbitrary user text, whitespace, message order, or requested settings.
        folded_signatures = sorted({
            request_signature(original_request + "".join(
                "\n\n[Added while working]:\n" + addition for addition in injected if addition
            ), []),
            request_signature(original_request + "\n\n[Added while you were thinking]:\n"
                              + "\n".join(injected), []),
        })
    saved_plan = await execution_state.record("flexible_plan")
    saved_folded_signatures = saved_plan.get("folded_signatures", []) if saved_plan is not None else []
    if (not isinstance(saved_folded_signatures, list) or len(saved_folded_signatures) > 2
            or not all(isinstance(value, str) for value in saved_folded_signatures)):
        saved_folded_signatures = []
    restored_tasks = (flexible.restore_tasks(saved_plan.get("queue"))
                      if saved_plan is not None and saved_plan.get("active")
                      and (saved_plan.get("signature") == signature
                           or signature in saved_folded_signatures) else None)
    cursor = saved_plan.get("cursor", 0) if restored_tasks and saved_plan is not None else 0
    saved_reports = saved_plan.get("reports", []) if restored_tasks and saved_plan is not None else []
    restored = bool(restored_tasks and isinstance(cursor, int) and not isinstance(cursor, bool)
                    and 0 <= cursor <= len(restored_tasks) and isinstance(saved_reports, list)
                    and len(saved_reports) == cursor and all(isinstance(text, str) for text in saved_reports))

    async def invalidate() -> None:
        if saved_plan is not None:
            await saved_plan.save(active=False)

    async def restart(value: AgentResult) -> tuple[AgentResult, bool]:
        await invalidate()
        return value, True

    async def status(text: str) -> None:
        if on_event:
            await on_event(ChatEvent(kind="tool", content=text))

    # A durable cursor is authoritative across allowance boundaries. Resume
    # the pending worker without another planner or replaying completed work.
    if restored:
        assert restored_tasks is not None and saved_plan is not None
        if not injected:
            # Further allowances still represent the same accepted requirements.
            # Retain the original bounded proof, without accumulating aliases.
            folded_signatures = saved_folded_signatures
        owner_mode = saved_plan.get("owner_mode", False)
        owner_reason = saved_plan.get("owner_reason", "")
        if (not isinstance(owner_mode, bool) or not isinstance(owner_reason, str)
                or owner_mode and (len(restored_tasks) != 1 or restored_tasks[0].read_only)):
            raise ValueError("Saved Flexible owner is invalid")
        plan = flexible.Plan(str(saved_plan.get("understanding", "")), tuple(restored_tasks),
                             owner_mode=owner_mode, owner_reason=owner_reason)
        if owner_mode:
            owner_request = saved_plan.get("owner_request")
            if not isinstance(owner_request, str) or not owner_request.strip():
                raise ValueError("Saved Flexible owner request is invalid")
            request = owner_request
        previous_draft = saved_plan.get("phase_previous_draft")
        await status(f"flexible: resuming saved plan at {cursor + 1}/{len(restored_tasks)}")
    else:
        await invalidate()
        await status(f"flexible: planning with {summary_backend.name}/{summary_model}")
        with agent_tools.skill_planning_scope():
            raw_plan, restarting = await _run_with_stage_timing(_run_with_inject_watch(
                run_internal_backend(
                    summary_backend,
                    workspace_path,
                    flexible.build_plan_prompt(
                        _planner_context(contextual_prompt, request),
                        collaborative=collaborative,
                        skills_catalog=(native_skills_catalog
                                        if summary_backend.name in _NATIVE_BACKENDS else ""),
                    ),
                    summary_model,
                    label="Flexible planner",
                    log=logger,
                    missing_executable_message=(
                        "%s CLI not found - flexible falling back to a single task"
                    ),
                    missing_level=logging.WARNING,
                ),
                inject_queue,
                injected,
            ), "flexible_planner", summary_backend.name, summary_model)
        if restarting:
            return await restart(AgentResult())
        plan = flexible.parse_plan(raw_plan or "", request)
    budget = _TURN_BUDGET.get()

    # Only the planner may ask.
    if plan.question:
        if collaborative:
            # A planner question ends the turn pre-merge.
            if _take_pending_injections(inject_queue, injected):
                return await restart(AgentResult())
            if close_inject_on_completion:
                _close_inject_queue(inject_queue)
            result = AgentResult()
            append_text_result(result, f"{plan.question}\n\n[[await]]")
            return result, False
        plan = flexible.fallback_plan(request)

    logger.info(
        "Flexible plan: %s",
        ", ".join(
            f"[{subtask.tier}] {subtask.instruction[:60 - len('… [clipped]')]}… [clipped]"
            if len(subtask.instruction) > 60 else f"[{subtask.tier}] {subtask.instruction}"
            for subtask in plan.subtasks
        ),
    )
    performance_logger.info(
        "stage=flexible_routing event=%s low=%d mid=%d high=%d",
        "resume" if restored else "plan",
        *(sum(task.tier == tier for task in plan.subtasks) for tier in flexible.TIERS),
    )
    owner_mode = plan.owner_mode

    # 2. Route sub-tasks to tiers.
    result = AgentResult()
    reports: list[str] = []
    attach_markers: list[str] = []
    usage_totals: dict = {}
    usage_audit: dict = {}
    blocked: list[int] = []
    worker_errors: list[str] = []
    unavailable_models: dict[tuple[str, str | None], str] = {}
    unrun_followups: list[flexible.Subtask] = []
    queue: list[flexible.Subtask] = list(plan.subtasks)
    done = 0
    total = len(queue)
    worker_outputs: dict[str, dict] = {}
    restored_outputs: set[str] = set()
    scope_members: dict[str, list[str]] = {}
    completed_read_scopes: set[str] = set()
    routing_evidence: dict[str, list[dict[str, str]]] = {}
    retained_worker_reports: set[str] = set()
    tier_leases: dict[str, list[dict[str, Any]]] = {}
    lease_rounds: dict[str, int] = {}
    lease_delivery: dict[str, int] = {}
    interim_followups: dict[str, list[dict[str, Any]]] = {}
    owner_obligations: dict[str, list[dict[str, Any]]] = {}
    owner_delivery: dict[str, int] = {}

    def encode_output(value: AgentResult, *, text: bool = False) -> dict:
        encoded: dict[str, Any] = {
            "events": [dataclasses.asdict(event) for event in value.events
                       if event.kind in {"attachment", "file"}],
            "detached_tasks": [dataclasses.asdict(task) for task in value.detached_tasks],
            "error": value.error, "usage": value.usage,
        }
        if text:
            escalation = flexible.parse_escalation(value.text)
            lease = flexible.parse_lease(value.text)
            encoded.update({
                "text": flexible.truncate_report(value.text),
                "followups": "\n".join(f"[followup:{task.tier}] {task.instruction}"
                                       for task in flexible.parse_followups(value.text)),
                "awaiting": extract_await(value.text)[1],
                "attach_markers": _split_attach_markers(value.text)[1],
                "complete": not (value.continuation_reason
                                 or value.error and "tool budget exhausted" in value.error.casefold()),
                "escalation": dataclasses.asdict(escalation) if escalation is not None else None,
                "lease": dataclasses.asdict(lease) if lease is not None else None,
            })
        return encoded

    def decode_output(value: dict) -> AgentResult:
        text = value.get("text", "")
        for marker in value.get("attach_markers", []):
            if marker not in text:
                text += "\n" + marker
        return AgentResult(
            text=text, error=value.get("error"), usage=value.get("usage"),
            events=[ChatEvent(**row) for row in value.get("events", [])],
            detached_tasks=[DetachedTaskRef(**row) for row in value.get("detached_tasks", [])],
        )

    if restored:
        assert saved_plan is not None
        done = cursor
        reports = list(saved_reports)
        result = decode_output(saved_plan.get("output", {}))
        result.usage = None  # Bot already carries earlier allowance usage.
        usage_audit = saved_plan.get("usage_processed_audit", {})
        attach_markers = saved_plan.get("attach_markers", [])
        blocked = saved_plan.get("blocked", [])
        worker_errors = saved_plan.get("worker_errors", [])
        unavailable_models = {(row["backend"], row["model"]): row["cause"]
                              for row in saved_plan.get("unavailable_models", [])}
        unrun_followups = flexible.restore_tasks(saved_plan.get("unrun_followups", []), limit=None) or []
        worker_outputs = saved_plan.get("worker_outputs", {})
        restored_outputs = set(worker_outputs)
        scope_members = saved_plan.get("scope_members", {})
        completed_read_scopes = set(saved_plan.get("completed_read_scopes", []))
        retained_routing = saved_plan.get("routing_evidence", {})
        if not isinstance(retained_routing, dict):
            raise ValueError("Saved Flexible routing evidence is invalid")
        for key, rows in retained_routing.items():
            if (not isinstance(key, str) or key not in {str(index) for index in range(len(queue))}
                    or not isinstance(rows, list) or len(rows) > len(flexible.TIERS) - 1):
                raise ValueError("Saved Flexible routing evidence is invalid")
            for row in rows:
                if (not isinstance(row, dict) or row.get("tier") not in flexible.TIERS
                        or not isinstance(row.get("report"), str)
                        or flexible.restore_escalation({
                            field: row.get(field) for field in ("reason", "evidence", "remaining")
                        }) is None):
                    raise ValueError("Saved Flexible routing evidence is invalid")
            routing_evidence[key] = rows
        tier_leases = saved_plan.get("tier_leases", {})
        lease_rounds = saved_plan.get("lease_rounds", {})
        lease_delivery = saved_plan.get("lease_delivery", {})
        interim_followups = saved_plan.get("interim_followups", {})
        owner_obligations = saved_plan.get("owner_obligations", {})
        owner_delivery = saved_plan.get("owner_delivery", {})
        if (not isinstance(owner_obligations, dict) or not isinstance(owner_delivery, dict)
                or any(not isinstance(key, str) or key not in {str(index) for index in range(len(queue))}
                       or not isinstance(rows, list) or len(rows) > flexible.MAX_SUBTASKS * 4
                       for key, rows in owner_obligations.items())):
            raise ValueError("Saved Flexible owner obligations are invalid")
        for rows in owner_obligations.values():
            seen_owner_scopes: set[str] = set()
            for row in rows:
                if (not isinstance(row, dict) or not isinstance(row.get("instruction"), str)
                        or not row["instruction"].strip()
                        or row.get("scope") != json_digest(flexible.instruction_identity(row["instruction"]))
                        or not isinstance(row.get("generation"), int) or isinstance(row["generation"], bool)
                        or not 0 <= row["generation"] < flexible.MAX_SUBTASKS * 4
                        or row.get("id") != json_digest([row["scope"], row["generation"]])
                        or row["id"] in seen_owner_scopes
                        or not isinstance(row.get("resolved"), bool)
                        or not isinstance(row.get("repeats"), int) or isinstance(row["repeats"], bool)
                        or not 0 <= row["repeats"] <= 3):
                    raise ValueError("Saved Flexible owner obligation is invalid")
                seen_owner_scopes.add(row["id"])
        if any(not isinstance(key, str) or not isinstance(value, int) or isinstance(value, bool)
               or not 0 <= value <= flexible.MAX_SUBTASKS * 4 for key, value in owner_delivery.items()):
            raise ValueError("Saved Flexible owner delivery is invalid")
        if (not isinstance(interim_followups, dict) or any(
            not isinstance(key, str) or flexible.restore_tasks(rows, limit=None) is None
            for key, rows in interim_followups.items()
        )):
            raise ValueError("Saved Flexible followups are invalid")
        if not all(isinstance(value, dict) for value in (tier_leases, lease_rounds, lease_delivery)):
            raise ValueError("Saved Flexible leases are invalid")
        for key, rows in tier_leases.items():
            if (key not in {str(index) for index in range(len(queue))}
                    or not isinstance(rows, list) or len(rows) > flexible.MAX_TIER_LEASES):
                raise ValueError("Saved Flexible leases are invalid")
            for row in rows:
                if (not isinstance(row, dict)
                        or flexible.restore_lease({field: row.get(field) for field in ("tier", "reason", "task")}) is None
                        or row.get("state") not in {"pending", "complete"}
                        or not isinstance(row.get("id"), str)
                        or row.get("parent_tier") not in flexible.TIERS
                        or not isinstance(row.get("round"), int)
                        or not isinstance(row.get("output", {}), dict)):
                    raise ValueError("Saved Flexible leases are invalid")
        for counters in (lease_rounds, lease_delivery):
            if any(not isinstance(key, str) or not isinstance(value, int) or isinstance(value, bool)
                   or not 0 <= value <= flexible.MAX_TIER_LEASES for key, value in counters.items()):
                raise ValueError("Saved Flexible lease counters are invalid")
        for value in worker_outputs.values():
            retained = decode_output(value)
            result.events.extend(event for event in retained.events if event not in result.events)
            result.detached_tasks.extend(task for task in retained.detached_tasks
                                         if task not in result.detached_tasks)
        saved_markers = attach_markers + [marker for value in worker_outputs.values()
                                         for marker in value.get("attach_markers", [])]
        carry_forward_turn_output(AgentResult(text="\n".join(saved_markers)), result, workspace_path)
        if budget is not None:
            budget.reports.extend((_checkpoint_scope(queue[index].instruction, request), report)
                                  for index, report in enumerate(reports))
            retained = AgentResult(text="\n".join(attach_markers), events=result.events,
                                   detached_tasks=result.detached_tasks)
            budget.capture(retained, workspace_path, account_usage=False)

    if budget is not None:
        budget.pending = [_checkpoint_scope(task.instruction, request) for task in queue[done:]]
        budget.pending.extend(task.instruction for task in unrun_followups)
        budget.pending.extend(item["instruction"] for rows in owner_obligations.values() for item in rows
                              if not item["resolved"] and item["instruction"] not in budget.pending)

    if owner_mode and not restored:
        initial_leases = []
        for number, specialist in enumerate(plan.specialists, 1):
            lease_id = json_digest({"origin": "initial", "number": number, **dataclasses.asdict(specialist)})
            initial_leases.append({
                **dataclasses.asdict(specialist), "id": lease_id, "state": "pending",
                "parent_tier": queue[0].tier, "round": number, "origin": "initial",
            })
        if initial_leases:
            tier_leases["0"] = initial_leases
            lease_rounds["0"] = len(initial_leases)
    if owner_mode:
        initial_items = [item for item in tier_leases.get("0", []) if item.get("origin") == "initial"]
        performance_logger.info(
            "stage=flexible_routing event=owner tier=%s specialists=%d pending_specialists=%d",
            queue[0].tier, len(initial_items), sum(item["state"] != "complete" for item in initial_items),
        )

    async def save_progress() -> None:
        if saved_plan is None:
            return
        await saved_plan.save(
            active=True, signature=signature, understanding=plan.understanding,
            folded_signatures=folded_signatures,
            queue=[flexible.task_record(task) for task in queue], cursor=done,
            reports=[flexible.truncate_report(report) for report in reports],
            output=encode_output(result), usage_processed_audit=usage_audit,
            attach_markers=attach_markers, blocked=blocked, worker_errors=worker_errors,
            unavailable_models=[{"backend": target[0], "model": target[1], "cause": cause}
                                for target, cause in unavailable_models.items()],
            unrun_followups=[flexible.task_record(task) for task in unrun_followups],
            worker_outputs=dict(worker_outputs),
            phase_previous_draft=(flexible.bounded_draft_excerpt(previous_draft)
                                  if previous_draft is not None else None),
            scope_members=scope_members,
            completed_read_scopes=sorted(completed_read_scopes),
            routing_evidence=routing_evidence,
            tier_leases=tier_leases, lease_rounds=lease_rounds, lease_delivery=lease_delivery,
            interim_followups=interim_followups,
            owner_mode=owner_mode, owner_reason=plan.owner_reason,
            owner_request=request if owner_mode else None,
            owner_obligations=owner_obligations, owner_delivery=owner_delivery,
        )

    await save_progress()

    def retain_report(index: int, sub_result: AgentResult) -> None:
        if budget is None:
            return
        report, _ = extract_await(sub_result.text)
        report, _ = _split_attach_markers(report)
        if sub_result.error:
            report += f"\n\n(the agent then failed: {sub_result.error})"
        scope = _checkpoint_scope(queue[index].instruction, request)
        budget.reports.append((scope, flexible.truncate_report(report)))
        if (scope in budget.pending and flexible.parse_lease(sub_result.text) is None
                and flexible.parse_escalation(sub_result.text) is None
                and not sub_result.continuation_reason
                and not (owner_mode and flexible.parse_followups(sub_result.text))):
            budget.pending.remove(scope)
        if sub_result.error and "tool budget exhausted" in sub_result.error.casefold():
            budget.pending.insert(0, scope)
            raise _TurnBudgetExhaustedError(sub_result.error)

    def can_inspect(index: int) -> bool:
        task = queue[index]
        backend_name, _ = tiers[task.tier]
        backend = backends_agent.get_backend(backend_name)
        return bool(task.read_only and task.independent
                    and getattr(backend, "supports_read_only_workers", False))

    def worker_scope(index: int) -> str:
        if owner_mode:
            return "flexible:owner"
        task = queue[index]
        return "flexible:" + json_digest({
            "tier": task.tier, "instruction": flexible.instruction_identity(task.instruction),
            "read_only": task.read_only, "independent": task.independent,
        })

    async def run_worker(
        index: int, earlier: list[str], *, parallel: bool,
    ) -> tuple[AgentResult, bool]:
        task = queue[index]
        backend_name, model = tiers[task.tier]
        backend = backends_agent.get_backend(backend_name)
        if not task.read_only:
            # A writer can invalidate previously inspected inputs even if it
            # is interrupted before producing a completed report.
            completed_read_scopes.clear()
        cached = worker_outputs.get(str(index))
        if cached is not None and cached.get("complete"):
            sub_result = decode_output(cached)
            if budget is not None:
                budget.capture(sub_result, workspace_path, account_usage=False)
            if str(index) not in retained_worker_reports:
                retain_report(index, sub_result)
                retained_worker_reports.add(str(index))
            return sub_result, False
        restored_outputs.discard(str(index))
        if budget is not None:
            budget.claim_worker()
        await status(
            f"flexible [{index + 1}/{total}] {task.tier} ·"
            f" {backend_name}/{model}: {task.instruction}"
        )
        # Worker role only; workspace approval stays.
        # The coordinator and all editing workers retain the chosen approval.
        worker_approval = (
            "confirm" if task.read_only and approval in {"auto", "full", "confirm"}
            else approval
        )
        # The plan changes when guidance changes, but an identical remaining
        # role can keep its native conversation and receive that new guidance.
        # Backend records separately isolate provider/model/approval/effort.
        scope = worker_scope(index)
        worker_sections = flexible.build_subtask_sections(
            request, plan, index, earlier, previous_draft=previous_draft,
        )
        attempts = routing_evidence.get(str(index), [])
        if attempts:
            worker_sections["routing_evidence"] = (
                "\nEarlier attempts on this same unfinished task:\n"
                + "\n\n".join(
                    f"[{attempt['tier']}] {flexible.truncate_report(attempt['report'])}\n"
                    f"Reason: {attempt['reason']}\nEvidence: {attempt['evidence']}"
                    for attempt in attempts
                )
                + f"\nRemaining work: {attempts[-1]['remaining']}\n"
                "Complete only that remaining work within the original task's scope and constraints."
                " Reuse completed work and valid evidence; do not restart the whole audit or rebuild."
            )
        completed_leases = [item for item in tier_leases.get(str(index), []) if item["state"] == "complete"]
        lease_sections = {
            "lease_result:" + item["id"]: (
                f"\nHelper question: {item['task']}\n"
                f"Helper findings: {item['output'].get('text', '')}\n"
                + (f"Helper limitation: {item['output']['error']}\n" if item["output"].get("error") else "")
                + "These are advisory findings. You retain responsibility for the original task and its verification."
            ) for item in completed_leases
        }
        if completed_leases:
            lease_sections["lease_continuation"] = (
                f"\nHelper request {lease_rounds.get(str(index), 0)} has been handled."
                " Continue your original task using the supplied findings. Reuse completed evidence."
                " Do not repeat a helper question already answered."
            )
        worker_sections.update(lease_sections)
        obligations = owner_obligations.get(str(index), [])
        obligation_sections = {
            "owner_obligation:" + item["id"]: (
                f"\nDiscovered obligation {number} for this same owner: {item['instruction']}\n"
                "Keep it within the original request and follow the current progress state."
            ) for number, item in enumerate(obligations, 1)
        }
        if obligations:
            resolved = ", ".join(str(number) for number, item in enumerate(obligations, 1) if item["resolved"])
            pending = ", ".join(str(number) for number, item in enumerate(obligations, 1) if not item["resolved"])
            obligation_sections["owner_progress"] = (
                f"\nPreviously reported completed obligations: {resolved or 'none'}."
                f" Remaining obligations: {pending or 'none'}."
                " Reuse completed work and valid verification; recheck only invalidated evidence or a new finding."
            )
            obligation_sections["owner_continuation"] = (
                "\nContinue your original job and finish the retained obligations."
                f" Outstanding-scope reports: {sum(item['repeats'] + 1 for item in obligations if not item['resolved'])}."
                " Reuse verified evidence. If reasoning prevents completion, request specialist help or escalation."
            )
        worker_sections.update(obligation_sections)
        worker_context = None
        if backend.name in _NATIVE_BACKENDS:
            worker_context = _native_turn_prompt(
                backend, request, shared_session, shared_colony,
                budget=shared_budget, collaborative=False, is_scheduled=True,
                skills_catalog=native_skills_catalog,
                # A long saved checkpoint must not consume the separately
                # bounded context needed to join split worker conversations.
                budget_includes_request=False,
                previous_draft=previous_draft if owner_mode else None,
            )
            worker_context.sections.update({
                key: text for key, text in worker_sections.items()
                if key != "request" and not (owner_mode and key == "previous_draft")
            })
        delivery_key = f"{index}:{task.tier}"
        http_resume = backend.name not in _NATIVE_BACKENDS and (
            delivery_key in lease_delivery and bool(completed_leases)
            or delivery_key in owner_delivery and bool(obligations)
        )
        prompt_sections = worker_sections
        if http_resume:
            delivered = lease_delivery.get(delivery_key, 0)
            prompt_sections = {
                key: value for key, value in lease_sections.items()
                if key == "lease_continuation" or any(
                    key == "lease_result:" + item["id"] and item["round"] > delivered
                    for item in completed_leases
                )
            }
            delivered_obligations = owner_delivery.get(delivery_key, 0)
            prompt_sections.update({
                key: value for key, value in obligation_sections.items()
                if key in {"owner_continuation", "owner_progress"} or any(
                    key == "owner_obligation:" + item["id"]
                    for item in obligations[delivered_obligations:]
                )
            })
            prompt_sections["parent_scope"] = f"\nContinue parent task: {task.instruction}"
        bootstrap_prompt = _build_backend_prompt(
            backend, "\n".join(worker_sections.values()), collaborative=False,
            allow_detached_requests=False, skills_catalog=native_skills_catalog,
        )
        worker_prompt = bootstrap_prompt
        if http_resume:
            worker_prompt = _build_backend_prompt(
                backend, "\n".join(prompt_sections.values()), collaborative=False,
                allow_detached_requests=False, skills_catalog=native_skills_catalog,
            )
        with (execution_state.worker_scope(scope),
              http_continuation_scope(bootstrap_prompt if http_resume else None)):
            sub_result, restarting = await _run_with_stage_timing(_drive_backend(
                backend, workspace_path, worker_prompt,
                model, worker_approval, effort=tier_efforts[task.tier], on_event=on_event,
                # A parallel wave has ONE inject watcher; it stops every sibling.
                inject_queue=None if parallel else inject_queue,
                injected=None if parallel else injected, on_process=on_process,
                native_context=worker_context,
                # Job scope pins retain their detailed task identity. Native
                # writers reuse the compatible conversation, while inspectors
                # have distinct slots so independent work stays concurrent.
                native_worker=((f"flexible:reader:{index}" if task.read_only else "main")
                               if worker_context is not None else None),
            ), "flexible_worker", backend_name, model,
                worker_index=index + 1, worker_count=total, worker_tier=task.tier)
        if sub_result.continuation_reason == "prerequisite":
            if budget is not None:
                budget.capture(sub_result, workspace_path)
            _accumulate_usage(usage_totals, sub_result.usage)
            _accumulate_usage(usage_audit, sub_result.usage)
            worker_outputs[str(index)] = encode_output(sub_result, text=True)
            await save_progress()
            raise _NativeRecoveryDeferred(sub_result)
        if budget is not None:
            budget.capture(sub_result, workspace_path)
        if not restarting:
            lease_delivery[delivery_key] = lease_rounds.get(str(index), 0)
            owner_delivery[delivery_key] = len(obligations)
            worker_outputs[str(index)] = encode_output(sub_result, text=True)
            try:
                retain_report(index, sub_result)
                retained_worker_reports.add(str(index))
            except _TurnBudgetExhaustedError:
                await save_progress()
                raise
            if parallel:
                # Siblings save independently, so one cancelled reader cannot
                # erase another reader's completed evidence. Serial workers
                # save once below, after advancing the durable cursor.
                await save_progress()
        return sub_result, restarting

    async def hold_current(message: str, details: str) -> tuple[AgentResult, bool]:
        result.error = message
        result.completion_verified = False
        result.usage = usage_totals or None
        append_text_result(result, "PARTIAL: " + details + "\nThe current task and its dependent work remain unfinished.")
        worker_outputs.setdefault(str(done), {})["terminal_error"] = message
        worker_outputs[str(done)]["terminal_text"] = flexible.truncate_report(result.text)
        if budget is not None:
            scope = _checkpoint_scope(queue[done].instruction, request)
            if scope not in budget.pending:
                budget.pending.append(scope)
        await save_progress()
        return result, False

    async def run_lease(
        index: int, item: dict[str, Any], *, parallel: bool = False, catalog: str | None = None,
    ) -> tuple[AgentResult, bool]:
        if item["state"] == "complete":
            return decode_output(item["output"]), False
        helper_name, helper_model = tiers[item["tier"]]
        helper_target = (helper_name, helper_model)
        unavailable_cause = unavailable_models.get(helper_target)
        if unavailable_cause is not None:
            value = AgentResult(
                text="Helper was not run because this account/model is already unavailable.",
                error=unavailable_cause,
            )
            item.update(state="complete", output=encode_output(value, text=True))
            await save_progress()
            return value, False
        helper = backends_agent.get_backend(helper_name)
        restricted_cli = isinstance(helper.permission_arg_sets, dict) and "restricted" in helper.permission_arg_sets
        if not (getattr(helper, "supports_read_only_workers", False) or restricted_cli):
            value = AgentResult(error="The configured helper backend cannot enforce read-only work.")
            item.update(state="complete", output=encode_output(value, text=True))
            await save_progress()
            return value, False
        if budget is not None:
            budget.claim_worker()
        helper_catalog = (catalog if catalog is not None
                          else skills.catalog_prompt(await skills.refresh(workspace_path), cli=True))
        helper_sections = {
            "parent_scope": f"\nParent task: {queue[index].instruction}",
            "helper_question": f"\nHelper question: {item['task']}\nReason: {item['reason']}",
            "lease_request": (
                f"\nHelper request {item['round']}: inspect current inputs for this request."
                " Recheck findings whose inputs changed since an earlier answer."
            ),
            "helper_role": (
                "\nYou are a read-only helper. Answer ONLY the helper question using analysis and existing files or logs."
                " Do not edit files, run builds/tests/processes, change state, or publish anything."
                " The parent owns implementation and final verification. Give concrete findings and limitations."
                " Do not request another helper or emit LEASE, ESCALATE, or followup control blocks."
            ),
        }
        if owner_mode:
            helper_sections["original_request"] = "\nOriginal user request and constraints:\n" + request
        if item.get("origin") == "initial":
            helper_sections["initial_consultation"] = (
                "\nThis is an independent initial consultation before the owner starts."
                " Use existing evidence and do not rely on another specialist's findings."
                " Resolve the question so the owner can choose and implement its approach."
            )
        helper_context = None
        if helper.name in _NATIVE_BACKENDS:
            helper_context = _native_turn_prompt(
                helper, item["task"], shared_session, shared_colony,
                budget=shared_budget, collaborative=False, is_scheduled=True,
                skills_catalog=helper_catalog, budget_includes_request=False,
            )
            helper_context.sections.update(helper_sections)
        helper_approval = "confirm" if approval in {"auto", "full", "confirm"} else approval
        scope = "flexible:lease:" + json_digest({"parent": index, "lease": item["id"]})
        await status(f"flexible: task {index + 1}/{total} borrowing {item['tier']} for read-only help")
        with execution_state.worker_scope(scope):
            value, restarting = await _run_with_stage_timing(_drive_backend(
                helper, workspace_path,
                _build_backend_prompt(helper, "\n".join(helper_sections.values()),
                                      collaborative=False, allow_detached_requests=False,
                                      skills_catalog=helper_catalog),
                helper_model, helper_approval, effort=tier_efforts[item["tier"]],
                on_event=on_event, inject_queue=None if parallel else inject_queue,
                injected=None if parallel else injected,
                on_process=on_process, native_context=helper_context,
                native_worker=f"flexible:lease:{item['tier']}" if helper_context is not None else None,
            ), "flexible_lease", helper_name, helper_model,
                worker_index=index + 1, worker_count=total, worker_tier=item["tier"])
        if budget is not None:
            budget.capture(value, workspace_path)
        _accumulate_usage(usage_totals, value.usage)
        _accumulate_usage(usage_audit, value.usage)
        if value.error and _account_provider_unavailable(value.error):
            unavailable_models[helper_target] = value.error
        result.events.extend(event for event in value.events if event.kind != "text")
        for marker in _split_attach_markers(value.text)[1]:
            if marker not in attach_markers:
                attach_markers.append(marker)
        if (flexible.parse_lease(value.text) is not None
                or flexible.parse_escalation(value.text) is not None):
            value.error = "Nested helper requests are disabled; the helper did not resolve its question."
        item["output"] = encode_output(value, text=True)
        item["state"] = "pending" if restarting or value.continuation_reason else "complete"
        await save_progress()
        return value, restarting

    max_parallel = config.get_flexible_max_parallel_workers()
    if owner_mode:
        while True:
            pending_initial = [item for item in tier_leases.get("0", [])
                               if item.get("origin") == "initial" and item["state"] != "complete"]
            if not pending_initial:
                break
            if _take_pending_injections(inject_queue, injected):
                return await restart(result)
            wave = pending_initial[:max_parallel]
            parallel = len(wave) > 1 and all(
                getattr(backends_agent.get_backend(tiers[item["tier"]][0]), "supports_read_only_workers", False)
                for item in wave
            ) and (budget is None or budget.max_workers - budget.workers_started >= len(wave))
            if parallel:
                catalog = skills.catalog_prompt(await skills.refresh(workspace_path), cli=True)
                values, restarting = await _run_parallel_flexible_workers(
                    [run_lease(0, item, parallel=True, catalog=catalog) for item in wave],
                    inject_queue, injected,
                )
                if restarting:
                    return await restart(result)
            else:
                values = [await run_lease(0, wave[0])]
            for helper_result, helper_restarting in values:
                if helper_restarting:
                    return await restart(result)
                if helper_result.continuation_reason:
                    carry_forward_turn_output(result, helper_result, workspace_path)
                    helper_result.usage = usage_totals or None
                    return helper_result, False

    while done < len(queue):
        if _take_pending_injections(inject_queue, injected):
            return await restart(result)
        task = queue[done]
        tier_backend_name, tier_model = tiers[task.tier]
        target = (tier_backend_name, tier_model)
        unavailable_cause = unavailable_models.get(target)
        if unavailable_cause is not None and not worker_outputs.get(str(done), {}).get("complete"):
            await status(
                f"flexible [{done + 1}/{total}] {task.tier} ·"
                f" {tier_backend_name}/{tier_model}: skipping unavailable account/model"
            )
            await asyncio.sleep(0)
            if _take_pending_injections(inject_queue, injected):
                return await restart(result)
            reports.append(
                f"PARTIAL: sub-task {done + 1} was not run: {task.instruction}\n"
                f"The earlier {tier_backend_name}/{tier_model} worker reported"
                f" an account/auth limit: {unavailable_cause}\n"
                "This plan item remains unfinished."
            )
            worker_errors.append(unavailable_cause)
            done += 1
            await save_progress()
            continue
        end = done + 1
        if can_inspect(done):
            wave_scopes = {worker_scope(done)}
            while (end < len(queue) and end - done < max_parallel
                   and can_inspect(end) and tiers[queue[end].tier] not in unavailable_models):
                scope = worker_scope(end)
                if scope in wave_scopes:
                    # Normalized duplicate wording can share an existing job
                    # pin. Keep those occurrences sequential across old plans.
                    break
                wave_scopes.add(scope)
                end += 1
        if budget is not None:
            remaining = budget.max_workers - budget.workers_started
            fresh = 0
            admitted = done
            for index in range(done, end):
                if not worker_outputs.get(str(index), {}).get("complete"):
                    if fresh >= remaining:
                        break
                    fresh += 1
                admitted = index + 1
            if admitted == done:
                budget.claim_worker()  # raises with the established checkpoint protocol
            end = admitted
        earlier = list(reports)
        if any(not worker_outputs.get(str(index), {}).get("complete") for index in range(done, end)):
            # Each actual worker wave selects skills from disk again. A long
            # preceding worker may have outlived edits to native/shared skills.
            # Parallel readers share one fresh snapshot, rather than scanning
            # and synchronizing the same roots once per sibling.
            worker_catalog = await skills.refresh(workspace_path)
            native_skills_catalog = skills.catalog_prompt(worker_catalog, cli=True)
        try:
            if end - done > 1:
                await status(f"flexible: inspecting {end - done} independent tasks in parallel")
                values, restarting = await _run_parallel_flexible_workers(
                    [run_worker(index, earlier, parallel=True) for index in range(done, end)],
                    inject_queue, injected,
                )
                if restarting:
                    return await restart(result)
            else:
                values = [await run_worker(done, earlier, parallel=False)]
        except _NativeRecoveryDeferred as exc:
            carry_forward_turn_output(result, exc.result, workspace_path)
            exc.result.usage = usage_totals or None
            return exc.result, False
        # Always merge in plan order, regardless of completion order.
        for sub_result, restarting in values:
            task = queue[done]
            tier_backend_name, tier_model = tiers[task.tier]
            target = (tier_backend_name, tier_model)
            if restarting:
                return await restart(result)
            cached_output = worker_outputs.get(str(done), {})
            if cached_output.get("terminal_error"):
                result.error = cached_output["terminal_error"]
                result.completion_verified = False
                if not result.text:
                    append_text_result(result, cached_output.get("terminal_text", "PARTIAL: the current task remains unfinished."))
                return result, False
            # Worker text is internal (feeds the merge).
            result.events.extend(
                ev for ev in sub_result.events if ev.kind != "text"
            )
            for task_ref in sub_result.detached_tasks:
                if task_ref not in result.detached_tasks:
                    result.detached_tasks.append(task_ref)
            if not cached_output.get("usage_processed"):
                if str(done) not in restored_outputs:
                    _accumulate_usage(usage_totals, sub_result.usage)
                _accumulate_usage(usage_audit, sub_result.usage)
                cached_output["usage_processed"] = True

            if sub_result.error:
                worker_errors.append(sub_result.error)
                if _account_provider_unavailable(sub_result.error):
                    unavailable_models[target] = sub_result.error
                logger.warning(
                    "Flexible sub-task %d/%d (%s/%s) failed: %s",
                    done + 1, total, tier_backend_name, tier_model, sub_result.error,
                )

            report, worker_awaiting = extract_await(sub_result.text)
            worker_awaiting = worker_awaiting or worker_outputs.get(str(done), {}).get("awaiting", False)
            report, markers = _split_attach_markers(report)
            markers.extend(worker_outputs.get(str(done), {}).get("attach_markers", []))
            attach_markers.extend(markers)
            report = report.strip()
            pending_followups = interim_followups.setdefault(str(done), [])
            reported_followups = flexible.parse_followups(cached_output.get("followups", ""))
            for extra in reported_followups:
                row = flexible.task_record(extra)
                if row not in pending_followups:
                    pending_followups.append(row)
            if owner_mode and reported_followups and not cached_output.get("owner_followups_processed"):
                obligations = owner_obligations.setdefault(str(done), [])
                delivery_key = f"{done}:{task.tier}"
                delivered = owner_delivery.get(delivery_key, 0)
                reported_ids: set[str] = set()
                for extra in reported_followups:
                    identity = json_digest(flexible.instruction_identity(extra.instruction))
                    if identity in reported_ids:
                        continue
                    reported_ids.add(identity)
                    if budget is not None and extra.instruction not in budget.pending:
                        budget.pending.append(extra.instruction)
                    existing = next((item for item in obligations
                                     if item["scope"] == identity and not item["resolved"]), None)
                    if existing is None:
                        if len(obligations) >= flexible.MAX_SUBTASKS * 4:
                            return await hold_current(
                                "Flexible owner obligation capacity reached; unfinished scope is retained.",
                                "The owner discovered more scope than can fit its bounded continuation context. "
                                "All reported obligations remain in the saved checkpoint.",
                            )
                        obligations.append(_new_owner_obligation(extra.instruction, obligations))
                    elif obligations.index(existing) < delivered:
                        existing["repeats"] = min(3, existing["repeats"] + 1)
                cached_output["owner_followups_processed"] = True

            if owner_mode and sub_result.continuation_reason:
                await save_progress()
                carry_forward_turn_output(result, sub_result, workspace_path)
                sub_result.usage = usage_totals or None
                return sub_result, False

            lease = flexible.restore_lease(cached_output.get("lease"))
            if sub_result.detached_tasks and (lease is not None or cached_output.get("escalation")
                                              or owner_mode and reported_followups):
                return await hold_current(
                    "Flexible helper handoff has unsettled background work.",
                    "The worker requested another tier while background work was still active. "
                    "Its task and background references are saved before any dependent work runs.",
                )
            if (lease is not None and not sub_result.error and not sub_result.continuation_reason
                    and not worker_awaiting and not sub_result.detached_tasks):
                key = str(done)
                if budget is not None:
                    scope = _checkpoint_scope(task.instruction, request)
                    if scope not in budget.pending:
                        budget.pending.append(scope)
                if "lease_round" not in cached_output:
                    rounds = lease_rounds.get(key, 0)
                    if rounds >= flexible.MAX_TIER_LEASES:
                        return await hold_current(
                            "Flexible helper lease limit reached; the current task remains unfinished.",
                            "The task requested more helper work after its bounded helper attempts. "
                            "Completed helper findings and parent work are saved.",
                        )
                    lease_rounds[key] = rounds + 1
                    cached_output["lease_round"] = rounds + 1
                lease_id = json_digest({"tier": lease.tier, "task": lease.task,
                                        "round": cached_output["lease_round"]})
                items = tier_leases.setdefault(key, [])
                item = next((entry for entry in items if entry["id"] == lease_id), None)
                if item is None:
                    item = {**dataclasses.asdict(lease), "id": lease_id, "state": "pending",
                            "parent_tier": task.tier, "round": cached_output["lease_round"]}
                    items.append(item)
                await save_progress()
                if lease.tier == task.tier:
                    helper_result = AgentResult(error="Use your own tier directly; helpers must use another tier.")
                    helper_restarting = False
                    item["output"] = encode_output(helper_result, text=True)
                    item["state"] = "complete"
                else:
                    helper_result, helper_restarting = await run_lease(done, item)
                if helper_restarting:
                    return await restart(result)
                if helper_result.continuation_reason:
                    carry_forward_turn_output(result, helper_result, workspace_path)
                    helper_result.usage = usage_totals or None
                    return helper_result, False
                await save_progress()
                worker_outputs.pop(key, None)
                restored_outputs.discard(key)
                retained_worker_reports.discard(key)
                if budget is not None:
                    scope = _checkpoint_scope(task.instruction, request)
                    if scope not in budget.pending:
                        budget.pending.append(scope)
                await save_progress()
                break

            escalation = flexible.restore_escalation(worker_outputs.get(str(done), {}).get("escalation"))
            if (escalation is not None and not sub_result.error
                    and not sub_result.continuation_reason and not worker_awaiting
                    and not sub_result.detached_tasks):
                tier_index = flexible.TIERS.index(task.tier)
                if budget is not None:
                    scope = _checkpoint_scope(task.instruction, request)
                    if scope not in budget.pending:
                        budget.pending.append(scope)
                if tier_index == len(flexible.TIERS) - 1:
                    return await hold_current(
                        "Flexible reasoning escalation exhausted at high; the current task remains unfinished.",
                        "The remaining work could not be resolved after the available reasoning tiers.\n"
                        f"Remaining work: {escalation.remaining}\n"
                        f"Evidence: {escalation.evidence}\n",
                    )
                next_tier = flexible.TIERS[tier_index + 1]
                routing_evidence.setdefault(str(done), []).append({
                    "tier": task.tier, "report": flexible.truncate_report(report),
                    **dataclasses.asdict(escalation),
                })
                queue[done] = dataclasses.replace(task, tier=next_tier)
                plan = dataclasses.replace(
                    plan, subtasks=tuple(queue),
                    owner_reason=(f"Escalated to {next_tier}: {escalation.reason}"
                                  if owner_mode else plan.owner_reason),
                )
                worker_outputs.pop(str(done), None)
                restored_outputs.discard(str(done))
                retained_worker_reports.discard(str(done))
                performance_logger.info(
                    "stage=flexible_routing event=escalate tier=%s next_tier=%s worker_index=%d",
                    task.tier, next_tier, done + 1,
                )
                await save_progress()
                await status(f"flexible: continuing task {done + 1}/{total} on {next_tier} with saved evidence")
                break

            if (owner_mode and reported_followups and not sub_result.error
                    and not sub_result.continuation_reason and not worker_awaiting
                    and not sub_result.detached_tasks):
                if any(not item["resolved"] and item["repeats"] >= 3
                       for item in owner_obligations.get(str(done), [])):
                    return await hold_current(
                        "Flexible owner repeatedly returned unresolved obligations.",
                        "The same owner returned the same unfinished scope after three supplied continuations "
                        "without resolving it or requesting specialist help. Its evidence and obligations are saved.",
                    )
                if budget is not None:
                    scope = _checkpoint_scope(task.instruction, request)
                    if scope not in budget.pending:
                        budget.pending.append(scope)
                worker_outputs.pop(str(done), None)
                restored_outputs.discard(str(done))
                retained_worker_reports.discard(str(done))
                interim_followups.pop(str(done), None)
                await save_progress()
                await status("flexible: the same owner is completing newly discovered obligations")
                break

            # Empty worker report: name the tier.
            if not report:
                report = (
                    f"(no output: {tier_backend_name}/{tier_model} ended the"
                    " sub-task without a reply"
                    + (f" - {sub_result.error}" if sub_result.error else "")
                    + ")"
                )
            elif sub_result.error:
                # Half answer + failure: tell the merge.
                report += f"\n\n(the agent then failed: {sub_result.error})"
            reports.append(report)

            # Stuck worker: merge ends on its question.
            if worker_awaiting:
                blocked.append(done)

            if owner_mode:
                if not sub_result.error and not worker_awaiting:
                    for item in owner_obligations.get(str(done), []):
                        item["resolved"] = True
                        if budget is not None:
                            while item["instruction"] in budget.pending:
                                budget.pending.remove(item["instruction"])
                done += 1
                interim_followups.pop(str(done - 1), None)
                worker_outputs.pop(str(done - 1), None)
                await save_progress()
                continue

            # Keep every distinct leftover, but consolidate compatible adjacent
            # scopes instead of multiplying a report into dozens of workers.
            followup_report = "\n".join(
                f"[followup:{extra['tier']}] {extra['instruction']}"
                for extra in interim_followups.get(str(done), [])
            )
            # Completed checks may need a legitimate rerun after a repair.
            # Only still-queued scope and duplicates inside this report can
            # be elided without assuming unchanged inputs.
            if can_inspect(done) and not sub_result.error and not worker_awaiting:
                completed_read_scopes.add(flexible.task_identity(task))
            known_scopes = set(completed_read_scopes)
            for pending in queue[done + 1:] + unrun_followups:
                identity = flexible.task_identity(pending)
                known_scopes.add(identity)
                known_scopes.update(scope_members.get(identity, []))
            followups, overflow, members = flexible.coalesce_followups(followup_report, known_scopes)
            scope_members.update(members)
            unrun_followups.extend(overflow)
            if budget is not None:
                budget.pending.extend(task.instruction for task in overflow)
            for extra in followups:
                if budget is not None:
                    scope = _checkpoint_scope(extra.instruction, request)
                    if scope not in budget.pending:
                        budget.pending.append(scope)
                if len(queue) < flexible.MAX_SUBTASKS * 2:
                    queue.append(extra)
                elif extra not in unrun_followups:
                    unrun_followups.append(extra)
            if len(queue) != total:
                total = len(queue)
                plan = flexible.Plan(
                    understanding=plan.understanding,
                    subtasks=tuple(queue),
                    question=None,
                )
                await status(f"flexible: plan grew to {total} sub-tasks")

            done += 1
            interim_followups.pop(str(done - 1), None)
            worker_outputs.pop(str(done - 1), None)
            await save_progress()

    remainder = ""
    if unrun_followups:
        remainder = (
            "PARTIAL: the follow-up task limit was reached; these tasks were not run:\n"
            + "\n".join(
                f"- [{task.tier}] {task.instruction}" for task in unrun_followups
            )
        )

    if worker_errors and (
        len(worker_errors) == len(queue)
        or any(_terminal_provider_error(error) for error in worker_errors)
    ):
        # Terminal worker failures must not become new judge/auto-chain work.
        result.error = "; ".join(dict.fromkeys(worker_errors))

    async def finish(final: str, *, awaiting: bool) -> tuple[AgentResult, bool]:
        """Apply the same output protocol to merged and fallback replies."""
        for marker in attach_markers:
            if marker not in final:
                final += f"\n\n{marker}"
        if awaiting:
            final += "\n\n[[await]]"
        if remainder:
            final += "\n\n" + remainder
        append_text_result(result, final)
        result.usage = usage_totals or None
        # The judge can reopen the owner with concrete missing work.
        await invalidate()
        return result, False

    if len(queue) == 1 or result.error:
        if _take_pending_injections(inject_queue, injected):
            return await restart(AgentResult())
        if close_inject_on_completion:
            _close_inject_queue(inject_queue)
        final = (
            reports[0] if len(queue) == 1
            else flexible.merge_fallback(plan, reports)
        ).strip()
        if result.error and len(worker_errors) < len(queue):
            final = "PARTIAL: task failures prevented full completion.\n\n" + final
        # Every iteration appends a non-empty report, so `final` is non-empty
        # (empty workers get a tier diagnostic; the all-empty merge path below
        # handles the rest).
        return await finish(final, awaiting=bool(collaborative and blocked))

    # 3. Merge reports into the user reply.
    await status(
        f"flexible: merging with {summary_backend.name}/{summary_model}"
    )
    merged, restarting = await _run_with_stage_timing(_run_with_inject_watch(
        run_internal_backend(
            summary_backend,
            workspace_path,
            flexible.build_merge_prompt(
                request, plan, reports,
                collaborative=collaborative, blocked=blocked,
            ),
            summary_model,
            label="Flexible merge",
            log=logger,
            missing_executable_message=(
                "%s CLI not found - flexible returning the raw worker reports"
            ),
            missing_level=logging.WARNING,
        ),
        inject_queue,
        injected,
        close_inject_on_completion=close_inject_on_completion,
    ), "flexible_merge", summary_backend.name, summary_model)
    if restarting:
        return await restart(AgentResult())
    final = (merged or "").strip() or flexible.merge_fallback(plan, reports)

    # Only the merge may end on a question.
    final, merge_awaiting = extract_await(final)
    final = final.strip()

    # All-empty workers + merge is a failure.
    if not final:
        logger.warning(
            "Flexible produced no text: %d worker report(s) all empty",
            len(reports),
        )
        final = (
            "The workers produced no output for this turn. Check the agent"
            " and model bound to each difficulty tier with"
            " /agent_flexible_low, _mid, and _high."
        )

    return await finish(final, awaiting=bool(collaborative and (merge_awaiting or blocked)))


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
    on_process: Callable[[Any], None] | None = None,
    on_session: Callable[[str], Awaitable[None]] | None = None,
    execution_id: str | None = None,
    original_request: str | None = None,
    is_scheduled: bool = False,
) -> AgentResult:
    """Run a turn, serialized per workspace.

    Concurrent turns in the same workspace - a user message racing a
    scheduled ``/reserve`` run, or two allow-listed users - would otherwise
    interleave the agent's file edits. A dedicated per-workspace run lock
    (separate from the file lock used for session logging/compaction, so no
    reentrancy deadlock) forces turns to run one at a time. The lock is
    released as soon as the turn finishes; background titling runs after.

    *on_process* is an optional non-sensitive progress hook forwarded to
    :func:`_drive_backend` (and flexible workers): it receives the
    launched backend process object so the caller can record the backend
    PID for stuck-wait attribution. It is never given prompts or args.
    *on_session* is awaited after routing and before agent work so callers
    can durably pin the conversation used by a recoverable queued request.
    *is_scheduled* controls unattended policy independently of a pinned session.
    """
    async with (
        workspace_mod.get_run_lock(workspace_path),
        skills.scope(workspace_path),
        execution_state.execution_scope(
            workspace_path, execution_id, original_request if original_request is not None else prompt,
        ),
    ):
        await _wait_for_pending_session_writes(workspace_path)
        return await _run_turn(
            prompt,
            workspace_path,
            user_id,
            model=model,
            summary_model=summary_model,
            approval=approval,
            on_event=on_event,
            inject_queue=inject_queue,
            backend_name=backend_name,
            summary_backend_name=summary_backend_name,
            session_id=session_id,
            on_process=on_process,
            on_session=on_session,
            is_scheduled=is_scheduled,
        )


async def _resolve_or_create_user_session(
    prompt: str,
    workspace_path: str,
    user_id: int | str,
    summary_model: str | None,
    summary_backend: str,
) -> tuple[str, dict]:
    """Resume a user's active session or route a new turn, then persist it.

    Foreground and detached turns must attach to the same conversation when
    they are started by the same user. Keeping the fallback-to-router policy
    in one place prevents either path from drifting when session persistence
    changes.
    """
    last_session_id = session.get_last_session(workspace_path, user_id)
    last_data = (
        session.load_session(workspace_path, last_session_id)
        if last_session_id else None
    )
    if last_data is not None:
        session_id, session_data = last_session_id, last_data
    else:
        session_id, session_data = await router.select_or_create_session(
            prompt,
            workspace_path,
            summary_model,
            backend_name=summary_backend,
        )
    assert isinstance(session_id, str) and session_id
    # Explicit routing-time session choice wins for the next message.
    if (
        session_id != last_session_id
        and session.get_last_session(workspace_path, user_id) == last_session_id
    ):
        session.set_last_session(workspace_path, user_id, session_id)
    return session_id, session_data


def _resolve_summary_model(
    summary_model: str | None, summary_backend: str,
) -> str | None:
    """Fill an omitted summary model from its selected backend.

    Foreground and detached turns both route and maintain the same session
    state.  Keeping this fallback here ensures they cannot silently diverge
    when a backend changes its summary-model default.
    """
    if summary_model:
        return summary_model
    default_summary_model = getattr(
        backends_agent.get_backend(summary_backend),
        "default_summary_model",
        None,
    )
    if isinstance(default_summary_model, str) and default_summary_model:
        return default_summary_model
    return summary_model


async def _settle_detached_operation(task: asyncio.Task[_StageResult]) -> _StageResult:
    """Keep ownership of a provider operation through repeated cancellation."""
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError:
            continue
    return task.result()


async def _stop_unclaimed_detached_task(backend, workspace_path: str, task_id: str) -> None:
    """Stop a known task that cannot be handed to its durable bot owner."""
    async def stop() -> None:
        try:
            stopped = await backend.stop_detached_task(workspace_path, task_id)
        except (Exception, asyncio.CancelledError):
            logger.warning(
                "Failed to stop unclaimed %s background task %s",
                backend.name, task_id, exc_info=True,
            )
        else:
            if not stopped:
                logger.warning(
                    "%s did not accept cancellation of unclaimed background task %s",
                    backend.name, task_id,
                )

    cleanup = asyncio.create_task(stop(), name=f"stop-unclaimed-detached:{task_id}")
    await _settle_detached_operation(cleanup)


async def launch_detached(
    prompt: str,
    workspace_path: str,
    user_id: int | str,
    model: str | None = None,
    summary_model: str | None = None,
    approval: str = "auto",
    *,
    backend_name: str | None = None,
    summary_backend_name: str | None = None,
) -> DetachedTaskLaunch:
    """Start a provider-owned task whose result can arrive after this turn.

    Unlike :func:`run`, this does not keep a foreground CLI process open.
    The backend must expose a durable task id; the bot persists that id before
    acknowledging the launch and later polls it through the same adapter.
    """
    async with workspace_mod.get_run_lock(workspace_path), skills.scope(workspace_path):
        await _wait_for_pending_session_writes(workspace_path)
        backend = backends_agent.get_backend(backend_name)
        if not getattr(backend, "supports_detached_tasks", False):
            raise RuntimeError(
                f"{backend.name} does not support restart-safe detached tasks.",
            )

        summary_backend = summary_backend_name or backend.name
        summary_model = _resolve_summary_model(summary_model, summary_backend)
        session_id, session_data = await _resolve_or_create_user_session(
            prompt,
            workspace_path,
            user_id,
            summary_model,
            summary_backend,
        )

        contextual_prompt = _build_contextual_prompt(
            prompt,
            session_data,
            colony.get_items(workspace_path),
            budget=workspace_mod.get_history_budget(workspace_path),
        )
        # Once the provider starts work, cancellation must not discard the late
        # identifier needed to stop it. Routing and workspace-lock waits above
        # remain cancellable; only this provider handshake gets an owner.
        handshake = asyncio.create_task(backend.launch_detached(
            workspace_path,
            _build_backend_prompt(
                backend,
                contextual_prompt,
                collaborative=False,
                allow_detached_requests=False,
                skills_catalog=skills.catalog_prompt(skills.current_catalog(workspace_path), cli=True),
            ),
            model,
            approval,
            effort=workspace_mod.get_reasoning_effort(workspace_path),
        ), name=f"launch-detached:{backend.name}")
        try:
            await asyncio.wait((handshake,))
            task_id = handshake.result()
        except asyncio.CancelledError:
            try:
                task_id = await _settle_detached_operation(handshake)
            except (Exception, asyncio.CancelledError):
                # Observe a failed handshake without replacing the caller's stop.
                logger.debug("Detached launch failed after cancellation", exc_info=True)
            else:
                await _stop_unclaimed_detached_task(backend, workspace_path, task_id)
            raise

        try:
            started = AgentResult(session_id=session_id)
            append_text_result(
                started,
                f"Started detached {backend.name} task {task_id}. "
                "Cozter will deliver its completion here.",
            )
            if not await _persist_detached_session_write(
                workspace_path,
                lambda: _log_to_session(workspace_path, session_id, prompt, started),
            ):
                raise RuntimeError("Could not log the detached task's session")
            # Deliver cancellation requested during session logging
            # before the task identifier leaves this cleanup owner's scope.
            await asyncio.sleep(0)
            return DetachedTaskLaunch(
                backend_name=backend.name,
                task_id=task_id,
                session_id=session_id,
            )
        except BaseException:
            await _stop_unclaimed_detached_task(backend, workspace_path, task_id)
            raise


async def log_detached_task_completion(
    workspace_path: str,
    session_id: str | None,
    text: str,
) -> None:
    """Append a detached task's delivered outcome to its original session."""
    if not session_id:
        return
    try:
        await _persist_detached_session_write(
            workspace_path,
            lambda: session.append_messages(workspace_path, session_id, [{
                "role": "assistant",
                "content": text,
            }]) > 0,
        )
    except Exception:
        logger.error("Failed to log detached task completion", exc_info=True)


async def _persist_detached_session_write(
    workspace_path: str, write: Callable[[], bool],
) -> bool:
    """Keep detached transcript I/O off-loop and own an admitted write to completion.

    Waiting for the session lock remains cancellable before a write starts.
    Once admitted, cancellation must settle the worker while retaining the lock,
    before launch cleanup can stop the provider or another writer reads state.
    """
    async with workspace_mod.get_lock(workspace_path):
        return await run_owned_worker(write)


def _track_session_write_task(
    workspace_path: str, task: asyncio.Task[Any],
) -> None:
    """Publish a write owner before another turn can read its session."""
    lock = workspace_mod.get_lock(workspace_path)
    owners = _PENDING_SESSION_WRITES.setdefault(lock, set())
    owners.add(task)

    def retire(done: asyncio.Task[Any]) -> None:
        owners.discard(done)
        if not owners:
            _PENDING_SESSION_WRITES.pop(lock, None)

    task.add_done_callback(retire)


async def _wait_for_pending_session_writes(workspace_path: str) -> None:
    """Read the next turn's context only after stopped-turn writes settle."""
    lock = workspace_mod.get_lock(workspace_path)
    while owners := {
        task for task in _PENDING_SESSION_WRITES.get(lock, ())
        if not task.done()
    }:
        # Caller cancellation must not cancel the detached write owners.
        await asyncio.wait(owners)


async def _persist_session_write(
    workspace_path: str, write: Callable[[], bool],
) -> bool:
    """Own the file lock until an off-loop durable write actually finishes.

    Callers run this in a detached task and await it with ``asyncio.wait`` so
    cancelling the turn returns immediately without cancelling its write.
    The owner also survives shutdown cancellation: an executor future stays
    alive until the thread stops, so the lock never exposes an in-flight
    read/modify/write to another session writer.
    """
    lock = workspace_mod.get_lock(workspace_path)
    while True:
        try:
            await lock.acquire()
            break
        except asyncio.CancelledError:
            continue
    try:
        context = contextvars.copy_context()
        future = asyncio.get_running_loop().run_in_executor(
            None, context.run, write,
        )
        while not future.done():
            try:
                await asyncio.wait((future,))
            except asyncio.CancelledError:
                continue
        return future.result()
    finally:
        lock.release()


def _append_session_messages(
    workspace_path: str, session_id: str, messages: list[dict],
    error_message: str,
) -> bool:
    """Return whether one synchronous session append completed successfully."""
    try:
        return session.append_messages(workspace_path, session_id, messages) > 0
    except Exception:
        logger.error(error_message, exc_info=True)
        return False


def _compaction_context_targets(
    workspace_path: str,
    backend,
    model: str | None,
    summary_backend: str,
    summary_model: str | None,
) -> tuple[tuple[str, str | None], ...]:
    """Return every model that can receive this turn's saved context.

    A direct turn has one recipient. Flexible feeds the same contextual
    prompt to its planner/merger and may route any subtask to any configured
    tier, so the compaction threshold must respect the smallest known window
    across all of them.
    """
    if backend.name != flexible.BACKEND_NAME:
        return ((backend.name, model),)
    tier_targets = tuple(workspace_mod.get_flexible_run_config(
        workspace_path,
    ).values())
    return ((summary_backend, summary_model), *tier_targets)


_POST_TURN_MAINTENANCE_LOCKS: weakref.WeakValueDictionary[
    tuple[str, str], asyncio.Lock,
] = weakref.WeakValueDictionary()


async def _run_post_turn_maintenance(
    workspace_path: str,
    session_id: str,
    summary_model: str | None,
    summary_backend: str,
    context_targets: tuple[tuple[str, str | None], ...],
) -> None:
    """Run non-critical session maintenance after the user reply is ready.

    Compaction can require another full model call. Keeping it out of the
    foreground turn means chat platforms can deliver the completed answer
    immediately. Titling follows compaction so a title emitted by the
    compactor wins over the lightweight auto-title fallback. Serialize the
    complete pipeline per session: another completed turn must not skip an
    in-flight compaction and then start a competing title request. Waiters
    reload current session state after acquiring the lock.
    """
    key = (
        workspace_mod.canonicalize_workspace_path(workspace_path), session_id,
    )
    lock = _POST_TURN_MAINTENANCE_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _POST_TURN_MAINTENANCE_LOCKS[key] = lock
    async with lock:
        await compaction.maybe_compact(
            workspace_path, session_id, summary_model,
            backend_name=summary_backend,
            context_targets=context_targets,
        )
        await titling.maybe_auto_title(
            workspace_path, session_id, summary_model,
            backend_name=summary_backend,
        )


INTERRUPTED_TURN_MAX_CHARS = 1200
INTERRUPTED_TURN_ACTIVITY_CHARS = 300
# Digest keeps the newest material.
INTERRUPTED_TURN_EVENT_KEEP = 100


@dataclass
class _InterruptedTurn:
    """Durable-trace accumulator for one agent run.

    :func:`_run_turn` fills this in as the turn progresses. If the turn is
    cancelled (/stop, shutdown, or a restart mid-turn), whatever was
    accumulated is appended to the session, so a later "resume" prompt sees
    the stopped attempt instead of a history that ends silently at the
    previous completed turn.
    """

    workspace_path: str
    prompt: str
    session_id: str | None = None
    events: list[ChatEvent] = dataclasses.field(default_factory=list)
    injected: list[str] = dataclasses.field(default_factory=list)
    logged_normally: bool = False
    normal_log_task: asyncio.Task[bool] | None = None


def _interrupted_turn_note(turn: _InterruptedTurn) -> str:
    """Build the assistant-side session entry for a cancelled turn."""
    lines = [
        "[Interrupted turn — stopped before completion at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}. Work it "
        "already did in this workspace may be incomplete or "
        "half-applied.]",
    ]
    digest = _format_session_response(
        AgentResult(events=list(turn.events)), turn.workspace_path,
    )
    if digest:
        if len(digest) > INTERRUPTED_TURN_MAX_CHARS:
            marker = (
                "… [earlier partial output dropped for budget — preview"
                " of newest only; say PARTIAL + remainder]\n"
            )
            digest = marker + digest[
                -(INTERRUPTED_TURN_MAX_CHARS - len(marker)):
            ]
        lines.append("Partial output before the stop:")
        lines.append(digest)
    else:
        activity = next(
            (
                ev.content for ev in reversed(turn.events)
                if ev.kind in ("tool", "file") and ev.content.strip()
            ),
            None,
        )
        if activity:
            lines.append("Last activity before the stop:")
            lines.append(_truncate_context_text(
                " ".join(activity.split()), INTERRUPTED_TURN_ACTIVITY_CHARS,
            ))
    return "\n\n".join(lines)


async def _acknowledge_native_user_messages(
    execution: execution_state.Execution | None, data: dict,
    *, exclude: execution_state.ExecutionRecord | None = None,
    conversation_only: bool = False,
) -> None:
    """Only workers that accepted this exact logged request know its user row."""
    if execution is None:
        return
    messages = data.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 2:
        return
    message = messages[-2]
    if not isinstance(message, dict) or message.get("role") != "user":
        return
    content = message.get("content")
    if not isinstance(content, str):
        return
    for record, native_id, requests in execution.native_request_deliveries():
        if record is exclude or content not in requests:
            continue
        # A detached interruption write can outlive deletion of its queue job.
        # Conversation records have their own lifetime and reject late writes.
        if conversation_only and record.execution.namespace != "native_sessions":
            continue
        if execution_state.native_session_id(record.get("native_session_id")) != native_id:
            continue
        await native_prompt.acknowledge_logged_turn(record, data, request_only=True)


def _log_interrupted_turn(turn: _InterruptedTurn) -> None:
    """Persist a cancelled turn's prompt and partial output to its session.

    Runs as a detached background task: the cancelling turn must re-raise
    immediately, and a second /stop must not be able to abort the write.
    Skipped when the turn already completed its normal session log, or was
    cancelled before it had a session to log into.
    """
    if turn.logged_normally or not turn.session_id:
        return
    note = _interrupted_turn_note(turn)
    session_id = turn.session_id
    execution = execution_state.current_execution()
    logged_native_data: dict | None = None

    def append_and_snapshot() -> bool:
        nonlocal logged_native_data
        logged = _append_session_messages(
            turn.workspace_path, session_id, [
                {"role": "user", "content": turn.prompt},
                {"role": "assistant", "content": note},
            ], "Failed to log interrupted turn",
        )
        if logged and execution is not None and execution.native_request_deliveries():
            logged_native_data = session.load_session(turn.workspace_path, session_id)
        return logged

    async def _append() -> None:
        # Wait for a live write so it stays single.
        if turn.normal_log_task is not None:
            while not turn.normal_log_task.done():
                try:
                    await asyncio.wait((turn.normal_log_task,))
                except asyncio.CancelledError:
                    continue
            if turn.logged_normally:
                return
        await _persist_session_write(
            turn.workspace_path,
            append_and_snapshot,
        )
        if logged_native_data is not None:
            await _acknowledge_native_user_messages(
                execution, logged_native_data, conversation_only=True,
            )

    try:
        task = create_background_task(
            _append(),
            name=f"log-interrupted-turn:{session_id}",
            log=logger,
        )
        _track_session_write_task(turn.workspace_path, task)
    except Exception:
        logger.error("Failed to schedule interrupted-turn log", exc_info=True)


async def _run_turn(
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
    on_process: Callable[[Any], None] | None = None,
    on_session: Callable[[str], Awaitable[None]] | None = None,
    is_scheduled: bool = False,
) -> AgentResult:
    """Run the selected agent backend with session history prepended.

    backend_name selects the registered agent adapter. When None, the
    default backend is used. The workspace's configured backend should be
    passed in by the caller.

    session_id pins the run to a specific session, including resumed interactive
    jobs and ephemeral scheduled sessions. is_scheduled selects unattended
    policy independently of that routing choice. When None, the user's last session is resumed if
    available; otherwise ``select_or_create_session`` routes the prompt to
    an existing session or creates a new one when none is a good fit.

    on_event  - called for each parsed event as it arrives (streaming).
    inject_queue - when a message is put, the running subprocess is killed
                   and restarted with the injected context appended.
    on_process - optional non-sensitive hook receiving the launched
                 backend process object (for backend-PID attribution);
                 never given prompts or arguments.

    Cancellation (/stop, shutdown, or a mid-turn restart) still persists the
    prompt and any partial output to the session before propagating, so a
    later resume has real memory of the stopped attempt.
    """
    turn = _InterruptedTurn(
        workspace_path=workspace_path, prompt=prompt, session_id=session_id,
    )
    with turn_budget() as budget:
        assert budget is not None
        budget.start()
        if not budget.pending:
            budget.pending = [_DEFAULT_PENDING_SCOPE]
        try:
            return await _run_turn_impl(
                turn, prompt, workspace_path, user_id,
                model=model,
                summary_model=summary_model,
                approval=approval,
                on_event=on_event,
                inject_queue=inject_queue,
                backend_name=backend_name,
                summary_backend_name=summary_backend_name,
                session_id=session_id,
                on_process=on_process,
                on_session=on_session,
                is_scheduled=is_scheduled,
            )
        except _TurnBudgetExhaustedError as exc:
            reason = str(exc)
            limit_kind = "tools" if "tool budget exhausted" in reason.casefold() else "workers"
        except asyncio.CancelledError:
            # /stop and daemon shutdown retain their original cancellation contract.
            _log_interrupted_turn(turn)
            raise
        continuation_reason: ContinuationReason = (
            "tool_budget" if limit_kind == "tools"
            else "worker_budget"
        )
        # An accepted message stays in this request.
        _close_inject_queue(inject_queue)
        _take_pending_injections(inject_queue, turn.injected)
        if turn.injected:
            turn.prompt = prompt + "\n\n[Added while you were thinking]:\n" + "\n".join(turn.injected)
        result = budget.finish(reason, turn, continuation_reason=continuation_reason)
        performance_logger.info(
            "stage=turn_budget outcome=exhausted limit=%s workers_started=%d",
            limit_kind, budget.workers_started,
        )
        if turn.session_id:
            budget_session_id = turn.session_id

            async def log_budget_result() -> bool:
                execution = execution_state.current_execution()
                logged_native_data: dict | None = None

                def log_and_snapshot() -> bool:
                    nonlocal logged_native_data
                    logged = _log_to_session(workspace_path, budget_session_id, turn.prompt, result)
                    if logged and execution is not None and execution.native_request_deliveries():
                        logged_native_data = session.load_session(workspace_path, budget_session_id)
                    return logged

                turn.logged_normally = await _persist_session_write(
                    workspace_path, log_and_snapshot,
                )
                if logged_native_data is not None:
                    await _acknowledge_native_user_messages(execution, logged_native_data)
                return turn.logged_normally

            turn.normal_log_task = create_background_task(
                log_budget_result(), name=f"log-budget-turn:{budget_session_id}", log=logger,
            )
            _track_session_write_task(workspace_path, turn.normal_log_task)
            try:
                await asyncio.wait((turn.normal_log_task,))
                turn.normal_log_task.result()
            except asyncio.CancelledError:
                _log_interrupted_turn(turn)
                raise
        return result


async def _run_turn_impl(
    turn: _InterruptedTurn,
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
    on_process: Callable[[Any], None] | None = None,
    on_session: Callable[[str], Awaitable[None]] | None = None,
    is_scheduled: bool = False,
) -> AgentResult:
    """Drive the backend phases of one turn, updating *turn* as it goes.

    Session routing, context prepending, streaming, the normal session log,
    and post-turn maintenance live here; interruption logging and the public
    contract live on :func:`_run_turn`. *on_process* is forwarded to
    :func:`_drive_backend` (and flexible workers) for backend-PID
    attribution; it is never given prompts or arguments.
    """
    backend = backends_agent.get_backend(backend_name)
    is_flexible = backend.name == flexible.BACKEND_NAME

    # Router/compaction/titling/planner/merge share the summary backend.
    summary_backend = summary_backend_name or (
        backends_agent.DEFAULT_DIRECT_BACKEND if is_flexible else backend.name
    )
    summary_model = _resolve_summary_model(summary_model, summary_backend)
    compaction_context_targets = _compaction_context_targets(
        workspace_path,
        backend,
        model,
        summary_backend,
        summary_model,
    )

    # Pinned sessions never clobber last_session.
    explicit_session = session_id is not None

    # session_data reused across inject restarts.
    if explicit_session:
        assert session_id is not None  # set by both branches above
        session_data = session.load_session(workspace_path, session_id)
        if session_data is None:
            # Pinned session deleted; bail.
            result = AgentResult()
            set_error_result(
                result,
                f"session {session_id} not found in {workspace_path}.",
            )
            return result
    else:
        session_id, session_data = await _resolve_or_create_user_session(
            prompt,
            workspace_path,
            user_id,
            summary_model,
            summary_backend,
        )

    # session_id set by both branches.
    assert session_id is not None
    execution_state.bind_conversation(session_id)
    turn.session_id = session_id
    if on_session is not None:
        await on_session(session_id)
    # Workspace memory loads once.
    colony_items = colony.get_items(workspace_path)

    # Scheduled/ephemeral turns run autonomous.
    collaborative = _is_collaborative_turn(
        workspace_path, is_scheduled=is_scheduled,
    )

    # Prepended context-block budget.
    history_budget = workspace_mod.get_history_budget(workspace_path)
    native_skills_catalog = skills.catalog_prompt(
        skills.current_catalog(workspace_path), cli=True,
    )
    phase_started = False

    injected = turn.injected
    effort = workspace_mod.get_reasoning_effort(workspace_path)

    async def _stream_event(event: ChatEvent) -> None:
        """Forward backend events, retaining recent ones for /stop recovery."""
        turn.events.append(event)
        if len(turn.events) > INTERRUPTED_TURN_EVENT_KEEP:
            del turn.events[:len(turn.events) - INTERRUPTED_TURN_EVENT_KEEP]
        if on_event is not None:
            await on_event(event)

    async def _run_phase(
        phase_prompt: str,
        phase_request: str,
        *,
        previous_draft: str | None = None,
        native_request: str | None = None,
    ) -> tuple[AgentResult, bool]:
        from . import supervision

        with supervision.worker_scope(
            workspace_path, summary_backend_name=summary_backend, summary_model=summary_model,
        ):
            return await _run_phase_impl(
                phase_prompt, phase_request, previous_draft=previous_draft, native_request=native_request,
            )

    async def _run_phase_impl(
        phase_prompt: str,
        phase_request: str,
        *,
        previous_draft: str | None = None,
        native_request: str | None = None,
    ) -> tuple[AgentResult, bool]:
        """Drive initial and continuation phases with the same turn policy."""
        nonlocal native_skills_catalog, phase_started
        if phase_started:
            # An accepted injection is a new user request inside the existing
            # run scope. Continuations also need current skill sources.
            catalog = await skills.refresh(workspace_path)
            native_skills_catalog = skills.catalog_prompt(catalog, cli=True)
        phase_started = True
        if is_flexible:
            return await _run_flexible(
                phase_prompt, phase_request, workspace_path,
                approval=approval,
                effort=effort,
                collaborative=collaborative,
                summary_backend_name=summary_backend,
                summary_model=summary_model,
                on_event=_stream_event,
                inject_queue=inject_queue,
                injected=injected,
                close_inject_on_completion=False,
                previous_draft=previous_draft,
                on_process=on_process,
                session_data=session_data,
                colony_items=colony_items,
                history_budget=history_budget,
            )
        return await _drive_backend(
            backend, workspace_path,
            _build_backend_prompt(
                backend, phase_prompt,
                collaborative=collaborative,
                allow_detached_requests=not is_scheduled,
                skills_catalog=native_skills_catalog,
            ),
            model, approval,
            effort=effort,
            on_event=_stream_event,
            inject_queue=inject_queue,
            injected=injected,
            close_inject_on_completion=False,
            on_process=on_process,
            native_context=_native_turn_prompt(
                backend, native_request if native_request is not None else phase_request,
                session_data, colony_items, budget=history_budget,
                collaborative=collaborative, is_scheduled=is_scheduled,
                bootstrap_request=phase_request, previous_draft=previous_draft,
                skills_catalog=native_skills_catalog,
            ) if backend.name in _NATIVE_BACKENDS else None,
        )

    # Tee phases so a cancelled turn records progress.
    while True:  # restart loop for inject
        effective_prompt = prompt
        if injected:
            effective_prompt += (
                "\n\n[Added while you were thinking]:\n"
                + "\n".join(injected)
            )
        turn.prompt = effective_prompt

        contextual_prompt = _build_contextual_prompt(
            effective_prompt, session_data, colony_items,
            budget=history_budget,
        )

        # Workspace scans off the event loop.
        attachment_images_before = await asyncio.to_thread(
            _snapshot_attachment_images, workspace_path,
        )

        try:
            if not is_flexible:
                logger.info(
                    "Running %s (prompt %d chars, context %d chars)",
                    backend.name, len(prompt), len(contextual_prompt),
                )
            result, restarting = await _run_phase(contextual_prompt, effective_prompt)
            result.completion_verified = False
        except BackendUnavailableError as e:
            return _error_result(session_id, str(e), inject_queue)

        if restarting:
            await _announce_restart(
                backend.name, injected, inject_queue, _stream_event,
            )
            continue

        await _append_new_attachment_images(result, attachment_images_before, workspace_path)

        # CONTINUE re-drives the backend (capped).
        budget = _TURN_BUDGET.get()
        if budget is not None:
            budget.latest_draft = flexible.bounded_draft_excerpt(result.text)
            budget.capture(result, workspace_path, account_usage=not is_flexible)
            if result.error and "tool budget exhausted" in result.error.casefold():
                raise _TurnBudgetExhaustedError(result.error)
        judged_rounds = 0
        while judged_rounds < flexible.JUDGE_MAX_CONTINUES:
            draft = result.text
            _draft_clean, draft_awaiting = extract_await(draft or "")
            if result.error or draft_awaiting or not _draft_clean.strip():
                break
            verdict, judge_restarting = await _judge_draft(
                effective_prompt, draft, workspace_path,
                summary_backend_name=summary_backend,
                summary_model=summary_model,
                inject_queue=inject_queue,
                injected=injected,
                on_event=_stream_event,
                round_no=judged_rounds + 1,
            )
            if judge_restarting:
                await _announce_restart(
                    backend.name, injected, inject_queue, _stream_event,
                )
                result = None  # type: ignore[assignment]
                restarting = True
                break
            if not verdict.should_continue:
                result.completion_verified = verdict.verified
                if verdict.blocked:
                    _retain_blocked_verdict(result, verdict, turn=turn)
                    if is_flexible and result.continue_instruction:
                        await _continue_flexible_owner(result.continue_instruction, draft)
                break
            judged_rounds += 1
            followup = (
                f"\n\n[Judge follow-up {judged_rounds}/"
                f"{flexible.JUDGE_MAX_CONTINUES}:"
                f" {verdict.missing} -- {verdict.next_instruction}]"
                if verdict.missing
                else f"\n\n[Judge follow-up {judged_rounds}/"
                f"{flexible.JUDGE_MAX_CONTINUES}:"
                f" {verdict.next_instruction}]"
            )
            await _stream_event(ChatEvent(
                kind="tool",
                content="Continuing with judge follow-up"
                f" {judged_rounds}/{flexible.JUDGE_MAX_CONTINUES}...",
            ))
            attachment_images_before = await asyncio.to_thread(
                _snapshot_attachment_images, workspace_path,
            )
            previous_result = result
            if is_flexible:
                await _continue_flexible_owner(followup.strip(), draft)
            continuation_prompt = _build_continuation_prompt(
                contextual_prompt, effective_prompt, draft, followup,
                draft_limit=history_budget,
            )
            try:
                result, restarting = await _run_phase(
                    continuation_prompt, effective_prompt + followup,
                    previous_draft=draft,
                    native_request=followup.strip(),
                )
                result.completion_verified = False
            except BackendUnavailableError as e:
                return _error_result(session_id, str(e), inject_queue)
            if restarting:
                await _announce_restart(
                    backend.name, injected, inject_queue, _stream_event,
                )
                break
            # The new draft replaces prior text, but artifacts and accounting
            # still belong to the same user turn.
            if budget is not None:
                budget.latest_draft = flexible.bounded_draft_excerpt(result.text)
                if not is_flexible:
                    budget.capture(result, workspace_path)
            carry_forward_turn_output(previous_result, result, workspace_path)
            await _append_new_attachment_images(result, attachment_images_before, workspace_path)
            if budget is not None:
                budget.capture(result, workspace_path, account_usage=False)
                if result.error and "tool budget exhausted" in result.error.casefold():
                    raise _TurnBudgetExhaustedError(result.error)
            # Cap exhausted: keep the verdict for chaining.
            if judged_rounds >= flexible.JUDGE_MAX_CONTINUES:
                _draft_tail, _tail_awaiting = extract_await(result.text or "")
                if result.error or _tail_awaiting or not _draft_tail.strip():
                    break
                _tail_verdict, _tail_restarting = await _judge_draft(
                    effective_prompt, result.text or "", workspace_path,
                    summary_backend_name=summary_backend,
                    summary_model=summary_model,
                    inject_queue=inject_queue,
                    injected=injected,
                    on_event=_stream_event,
                    round_no=judged_rounds + 1,
                )
                if _tail_restarting:
                    await _announce_restart(
                        backend.name, injected, inject_queue, _stream_event,
                    )
                    restarting = True
                    break
                result.completion_verified = _tail_verdict.verified and not _tail_verdict.should_continue
                if _tail_verdict.blocked:
                    _retain_blocked_verdict(result, _tail_verdict, turn=turn)
                    if is_flexible and result.continue_instruction:
                        await _continue_flexible_owner(result.continue_instruction, result.text)
                if _tail_verdict.should_continue:
                    _tail_next = (_tail_verdict.next_instruction or "").strip()
                    if _tail_next:
                        result.continue_instruction = (
                            f"[Auto-continued remainder"
                            f" ({judged_rounds + 1}):"
                            f" {_tail_verdict.missing} -- {_tail_next}]"
                            if _tail_verdict.missing
                            else f"[Auto-continued remainder"
                            f" ({judged_rounds + 1}): {_tail_next}]"
                        )
                        result.continuation_reason = "judge_limit"
                        if is_flexible:
                            await _continue_flexible_owner(result.continue_instruction, result.text)
                        if budget is not None:
                            budget.pending = [result.continue_instruction]
                            result.continuation_checkpoint = budget.checkpoint(
                                "The completion judge found unfinished work after its continuation limit.",
                                turn,
                            )
                break
        if restarting:
            continue

        if _take_pending_injections(inject_queue, injected):
            await _announce_restart(
                backend.name, injected, inject_queue, _stream_event,
            )
            continue
        break

    _close_inject_queue(inject_queue)
    # Discard Queue messages after the final answer.
    _drain_queue(inject_queue)

    # Consume the marker before session/chat delivery.
    _consume_detached_task_requests(result)

    # The detached owner keeps the lock on /stop.
    execution = execution_state.current_execution()
    native_record = execution.native_main_record if execution is not None else None
    logged_native_data: dict | None = None

    def _log_and_snapshot() -> bool:
        nonlocal logged_native_data
        logged = _log_to_session(workspace_path, session_id, effective_prompt, result)
        if logged and execution is not None and (
            native_record is not None and not is_flexible and not result.error
            or execution.native_request_deliveries()
        ):
            # Capture our exact pair while holding the session write lock.
            # A detached completion can append as soon as that lock releases.
            logged_native_data = session.load_session(workspace_path, session_id)
        return logged

    async def _log_normal_turn() -> bool:
        turn.logged_normally = await _persist_session_write(
            workspace_path, _log_and_snapshot,
        )
        return turn.logged_normally

    turn.normal_log_task = create_background_task(
        _log_normal_turn(),
        name=f"log-session-turn:{session_id}",
        log=logger,
    )
    _track_session_write_task(workspace_path, turn.normal_log_task)
    await asyncio.wait((turn.normal_log_task,))
    turn.normal_log_task.result()

    if logged_native_data is not None:
        own_pair = native_record if not is_flexible and not result.error else None
        if own_pair is not None:
            await native_prompt.acknowledge_logged_turn(own_pair, logged_native_data)
        await _acknowledge_native_user_messages(execution, logged_native_data, exclude=own_pair)

    # Maintenance receives every input explicitly. Give its task a clean
    # context so a slow compactor/title or pipeline-lock wait cannot retain
    # completed backend transcripts, turn budgets, or expired timeout owners.
    contextvars.Context().run(
        create_background_task,
        _run_post_turn_maintenance(
            workspace_path,
            session_id,
            summary_model,
            summary_backend,
            compaction_context_targets,
        ),
        name=f"post-turn-maintenance:{session_id}",
        log=logger,
    )

    if result.text and not any(event.kind == "text" for event in result.events):
        append_text_result(result, result.text)
    has_output = False
    for event in result.events:
        if event.kind == "text" or event.kind == "attachment":
            has_output = True
            break
    if not has_output and not result.detached_task_requests:
        append_text_result(result, result.text or NO_RESPONSE_TEXT)

    result.session_id = session_id
    return result


def _log_to_session(
    workspace_path: str, session_id: str, prompt: str, result: AgentResult,
) -> bool:
    """Append the user prompt and AI response in a single read+write."""
    try:
        return _append_session_messages(workspace_path, session_id, [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": _format_session_response(result, workspace_path),
            },
        ], "Failed to log session")
    except Exception:
        logger.error("Failed to log session", exc_info=True)
        return False


def _format_session_response(
    result: AgentResult, workspace_path: str,
) -> str:
    """Return the assistant's final text reply for session logging.

    Tool and file events are intermediate 'thinking' — the text reply
    already summarizes what was done, and skipping them keeps the saved
    history (and the context fed to future turns) compact.

    ``[[await]]`` is stripped: it is a control marker the bot consumes,
    not something the assistant said. Logging it would replay it as
    conversation on every later turn — and into compaction summaries and
    auto-titles — teaching the model to emit it when nothing is blocked.
    """
    text_parts: list[str] = []
    for ev in result.events:
        if ev.kind != "text":
            continue
        cleaned, _ = extract_await(ev.content)
        cleaned = cleaned.strip()
        if cleaned:
            text_parts.append(cleaned)
    attachment_parts: list[str] = []
    ws_real = os.path.realpath(workspace_path)
    for ev in result.events:
        if ev.kind != "attachment":
            continue
        try:
            path = os.path.realpath(
                _workspace_candidate_path(ev.content, workspace_path),
            )
            if is_path_within(path, ws_real):
                path = os.path.relpath(path, workspace_path)
        except (OSError, ValueError):
            path = ev.content
        attachment_parts.append(f"[Attachment: {path}]")
    if text_parts or attachment_parts:
        return "\n\n".join([*text_parts, *attachment_parts])
    cleaned, _ = extract_await(result.text)
    return cleaned.strip()
