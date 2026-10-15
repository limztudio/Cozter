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
import weakref
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import dataclasses
from datetime import datetime
from typing import Any, TypeVar

from . import (
    agent_tools, backends_agent, colony, compaction, config, execution_state, flexible, router,
    session, titling,
)
from . import workspace as workspace_mod
from .agent_tools.base import copy_file_atomically
from .backends_agent.base import (
    DOC_FOLLOWING_RULE, NO_RESPONSE_TEXT, WHOLE_SCOPE_RULE, AgentResult,
    ChatEvent, ContinuationCheckpoint, ContinuationReason, ContinuationReport,
    append_detached_task_request,
    append_text_result, record_backend_error, set_error_result,
)
from .utils import (
    IMAGE_EXTENSIONS,
    abandon_subprocess_stream_task,
    await_cancelled,
    cleanup_backend_process,
    create_background_task,
    drain_text_stream,
    finish_process_stderr,
    has_managed_process_group,
    is_path_within,
    iter_process_json_events,
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
    """One execution attempt's allowance, shared by its immediate continuations."""

    timeout_seconds: int
    max_workers: int = flexible.MAX_SUBTASKS * 2
    deadline: float | None = None
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
        # Start after the workspace run lock; time spent queuing is excluded.
        if not self.started:
            self.started = True
            if self.timeout_seconds:
                self.deadline = asyncio.get_running_loop().time() + self.timeout_seconds
            performance_logger.info(
                "stage=turn_budget event=start timeout_seconds=%d max_workers=%d",
                self.timeout_seconds, self.max_workers,
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
            flexible._bounded_draft_excerpt(streamed, limit=1200),
        ]
        quotas = _context_quotas([len(value) for value in values], CONTINUATION_CHECKPOINT_CHARS)
        bounded = [
            flexible._bounded_draft_excerpt(value, limit=quota)
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
                or flexible._DRAFT_OMISSION_MARKER in self.latest_draft
                or any(flexible._REPORT_TRUNCATION_MARKER in report for _, report in reports)
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
                f"Task: {instruction}\n{flexible._truncate_report(report)}"
                for instruction, report in self.reports
            )
            sections.append("Completed worker reports:\n" +
                            flexible._bounded_draft_excerpt(reports))
        elif self.latest_draft:
            sections.append("Latest completed draft:\n" +
                            flexible._bounded_draft_excerpt(self.latest_draft))
        streamed = next((
            event.content for event in reversed(turn.events)
            if event.kind == "text" and event.content.strip()
        ), "")
        if streamed and streamed != self.latest_draft:
            sections.append("Latest streamed output (unverified):\n" +
                            flexible._bounded_draft_excerpt(streamed, limit=1200))
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
                            flexible._bounded_draft_excerpt(pending, limit=6000))
        else:
            sections.append("Final completion verification remains unfinished.")
        append_text_result(result, "\n\n".join(sections))
        return result


_TURN_BUDGET: ContextVar[TurnBudget | None] = ContextVar("cozter_turn_budget", default=None)
_TURN_TIMEOUT: ContextVar[asyncio.Timeout | None] = ContextVar("cozter_turn_timeout", default=None)


@contextmanager
def turn_budget(*, fresh: bool = False):
    """Reuse an attempt's budget, or explicitly start the next allowance."""
    current = _TURN_BUDGET.get()
    if not fresh and current is not None and current.owner is _current_task_or_none():
        yield current
        return
    token = _TURN_BUDGET.set(TurnBudget(config.get_agent_turn_timeout()))
    try:
        yield _TURN_BUDGET.get()
    finally:
        _TURN_BUDGET.reset(token)


_PENDING_SESSION_WRITES: dict[asyncio.Lock, set[asyncio.Task[Any]]] = {}


async def _run_with_stage_timing(
    call: Awaitable[_StageResult], stage: str, backend_name: str,
    model: str | None,
    *, worker_index: int | None = None, worker_count: int | None = None,
) -> _StageResult:
    """Record stage starts and durations without private call content."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    worker_metadata = ""
    if worker_index is not None:
        worker_metadata += f" worker_index={worker_index}"
    if worker_count is not None:
        worker_metadata += f" worker_count={worker_count}"
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
    if re.search(r"\bHTTP\s*(?:status\s*)?(?:401|403|429)\b", error, re.IGNORECASE):
        return True
    normalized = error.casefold().replace("_", " ").replace("-", " ")
    return any(marker in normalized for marker in (
        "usage limit", "quota exceeded", "quota exhausted", "quota limit",
        "insufficient quota", "insufficient credits", "out of credits",
        "credit balance exhausted", "invalid api key", "authentication failed",
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
    " canonical commands with 3600s cap (cancel still stops instantly; capture"
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
    workspace_path: str, *, explicit_session: bool,
) -> bool:
    """Return whether this turn may pause for a live user answer."""
    return (
        not explicit_session
        and workspace_mod.get_interaction_style(workspace_path)
        == "collaborative"
    )


_ATTACH_RE = re.compile(
    r"\[\[attach:\s*([^\]\n]+?)\s*\]\]", re.IGNORECASE,
)
_AWAIT_RE = re.compile(r"\[\[await\]\]", re.IGNORECASE)
_DETACHED_TASK_REQUEST_RE = re.compile(
    r"(?ims)^[ \t]*\[\[background:\s*(?P<prompt>.*?)\s*\]\]"
    r"[ \t]*(?:\r?\n|$)",
)
_EXTRA_BLANK_LINES_RE = re.compile(r"\n{3,}")
_SAFE_IMAGE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_DETACHED_TASK_REQUESTS = 3
_MAX_DETACHED_TASK_PROMPT_CHARS = 12_000

_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
)
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

    def _sub(match: re.Match) -> str:
        rel = match.group(1).strip()
        if not rel:
            return ""
        source = _resolve_attachment_source(rel, workspace_path)
        if source is None:
            return ""
        source_path, _ = source
        if source_path in seen_sources:
            return ""
        seen_sources.add(source_path)
        paths.append(source_path)
        return ""

    cleaned = _ATTACH_RE.sub(_sub, text)
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

    def _take(match: re.Match) -> str:
        prompt = match.group("prompt").strip()
        if len(prompt) > _MAX_DETACHED_TASK_PROMPT_CHARS:
            logger.warning("Ignoring oversized detached task request")
        elif prompt and prompt not in prompts:
            if len(prompts) < _MAX_DETACHED_TASK_REQUESTS:
                prompts.append(prompt)
            else:
                logger.warning("Ignoring excess detached task request")
        return ""

    cleaned = _DETACHED_TASK_REQUEST_RE.sub(_take, text)
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
            head = file_handle.read(16)
    except OSError:
        return None
    for magic, ext in _IMAGE_MAGIC:
        if head.startswith(magic):
            return ext
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return ".webp"
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


def _iter_image_files(root: str) -> list[str]:
    paths: list[str] = []
    try:
        root_real = os.path.realpath(root)
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
                if not is_path_within(path, root_real) or not os.path.isfile(path):
                    continue
                relative_parts = os.path.relpath(path, root_real).split(os.sep)[:-1]
                # A file link can point into a skipped tree inside the workspace.
                if not any(_skip_automatic_image_directory(part) for part in relative_parts):
                    paths.append(path)
    except OSError:
        logger.warning("Failed to scan image artifacts under %s", root,
                       exc_info=True)
    return paths


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
    for path in _iter_image_files(ws_real):
        try:
            st = os.stat(path)
        except OSError:
            continue
        snapshot[path] = (st.st_mtime_ns, st.st_size)
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

    # Bound durable blocks too.
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
        # Fit context: keep the request intact.
        fixed_length = len(continuation) + 1 + len(prompt)
        if fixed_length > budget:
            return prompt
        # Count separators too.
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
    parts.append(contextual_prompt)
    return "\n\n".join(parts)


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
) -> tuple[AgentResult, bool]:
    async with execution_state.backend_scope(
        backend.name, model or getattr(backend, "default_model", None), approval, effort,
    ) as state:
        result, restarting = await _drive_backend_impl(
            backend, workspace_path, full_prompt, model, approval, effort=effort,
            on_event=on_event, inject_queue=inject_queue, injected=injected,
            close_inject_on_completion=False, on_process=on_process,
        )
        if (not restarting and state is not None and state.get("native_resume_unavailable")
                and not state.get("native_tools_started")
                and not any(event.kind in {"tool", "file", "attachment"}
                            or event.kind == "text" and event.content.strip() not in {
                                result.error, f"Error: {result.error}",
                            } for event in result.events) and not result.detached_tasks):
            # Only a provider-confirmed missing exact session, before any
            # activity, permits one fresh launch. Never retry ambiguous work.
            await state.save(native_resume_unavailable=False, native_session_id=None)
            logger.info("%s exact native session unavailable before work; recovering from saved evidence", backend.name)
            result, restarting = await _drive_backend_impl(
                backend, workspace_path,
                full_prompt + "\n\n[The exact provider session was unavailable before any new work began. "
                "Inspect existing workspace changes and saved evidence; continue unfinished work "
                "without repeating completed changes or commands with an unknown outcome.]",
                model, approval, effort=effort, on_event=on_event,
                inject_queue=inject_queue, injected=injected,
                close_inject_on_completion=False, on_process=on_process,
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

    result = AgentResult()
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

    # A partial reply does not make a failed process a successful turn.
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
    markers = [match.group(0) for match in _ATTACH_RE.finditer(text)]
    if not markers:
        return text, []
    cleaned = _ATTACH_RE.sub("", text)
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
    try:
        raw, restarting = await _run_with_stage_timing(_run_with_inject_watch(
            run_internal_backend(
                summary_backend,
                workspace_path,
                flexible.build_judge_prompt(request, cleaned),
                summary_model,
                timeout=flexible.JUDGE_TIMEOUT,
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
        # gather itself must also be observed if it failed or was cancelled.
        await asyncio.gather(group, *tasks, *([watcher] if watcher else []),
                             return_exceptions=True)


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
    execution = execution_state.current_execution()
    signature = flexible.state_signature({
        "request": execution.original_request if execution is not None else request,
        "injected": injected, "tiers": tiers, "approval": approval, "effort": effort,
        "summary_backend": summary_backend_name, "summary_model": summary_model,
        "collaborative": collaborative,
    })
    saved_plan = await execution_state.record("flexible_plan")
    restored_tasks = (flexible.restore_tasks(saved_plan.get("queue"))
                      if saved_plan is not None and saved_plan.get("active")
                      and saved_plan.get("signature") == signature else None)
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
        plan = flexible.Plan(str(saved_plan.get("understanding", "")), tuple(restored_tasks))
        previous_draft = saved_plan.get("phase_previous_draft")
        await status(f"flexible: resuming saved plan at {cursor + 1}/{len(restored_tasks)}")
    else:
        await invalidate()
        await status(f"flexible: planning with {summary_backend.name}/{summary_model}")
        raw_plan, restarting = await _run_with_stage_timing(_run_with_inject_watch(
            run_internal_backend(
                summary_backend,
                workspace_path,
                flexible.build_plan_prompt(
                    _planner_context(contextual_prompt, request),
                    collaborative=collaborative,
                ),
                summary_model,
                timeout=flexible.PLAN_TIMEOUT,
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

    def encode_output(value: AgentResult, *, text: bool = False) -> dict:
        encoded: dict[str, Any] = {
            "events": [dataclasses.asdict(event) for event in value.events
                       if event.kind in {"attachment", "file"}],
            "detached_tasks": [dataclasses.asdict(task) for task in value.detached_tasks],
            "error": value.error, "usage": value.usage,
        }
        if text:
            encoded.update({
                "text": flexible._truncate_report(value.text),
                "followups": "\n".join(f"[followup:{task.tier}] {task.instruction}"
                                       for task in flexible.parse_followups(value.text)),
                "awaiting": extract_await(value.text)[1],
                "attach_markers": _split_attach_markers(value.text)[1],
                "complete": not (value.error and "tool budget exhausted" in value.error.casefold()),
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

    async def save_progress() -> None:
        if saved_plan is None:
            return
        await saved_plan.save(
            active=True, signature=signature, understanding=plan.understanding,
            queue=[flexible.task_record(task) for task in queue], cursor=done,
            reports=[flexible._truncate_report(report) for report in reports],
            output=encode_output(result), usage_processed_audit=usage_audit,
            attach_markers=attach_markers, blocked=blocked, worker_errors=worker_errors,
            unavailable_models=[{"backend": target[0], "model": target[1], "cause": cause}
                                for target, cause in unavailable_models.items()],
            unrun_followups=[flexible.task_record(task) for task in unrun_followups],
            worker_outputs=dict(worker_outputs),
            phase_previous_draft=(flexible._bounded_draft_excerpt(previous_draft)
                                  if previous_draft is not None else None),
            scope_members=scope_members,
            completed_read_scopes=sorted(completed_read_scopes),
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
        budget.reports.append((scope, flexible._truncate_report(report)))
        if scope in budget.pending:
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

    async def run_worker(
        index: int, earlier: list[str], *, parallel: bool,
    ) -> tuple[AgentResult, bool]:
        task = queue[index]
        backend_name, model = tiers[task.tier]
        backend = backends_agent.get_backend(backend_name)
        if not can_inspect(index):
            # A writer can invalidate previously inspected inputs even if it
            # is interrupted before producing a completed report.
            completed_read_scopes.clear()
        cached = worker_outputs.get(str(index))
        if cached is not None and cached.get("complete"):
            sub_result = decode_output(cached)
            if budget is not None:
                budget.capture(sub_result, workspace_path, account_usage=False)
            retain_report(index, sub_result)
            return sub_result, False
        restored_outputs.discard(str(index))
        if budget is not None:
            budget.claim_worker()
        await status(
            f"flexible [{index + 1}/{total}] {task.tier} ·"
            f" {backend_name}/{model}: {task.instruction}"
        )
        # This is a worker role, not a change to the workspace permission.
        # The coordinator and all editing workers retain the chosen approval.
        worker_approval = (
            "confirm" if can_inspect(index) and approval in {"auto", "full", "confirm"}
            else approval
        )
        # The plan changes when guidance changes, but an identical remaining
        # role can keep its native conversation and receive that new guidance.
        # Backend records separately isolate provider/model/approval/effort.
        scope = "flexible:" + flexible.state_signature({
            "tier": task.tier, "instruction": flexible.instruction_identity(task.instruction),
            "read_only": task.read_only, "independent": task.independent,
        })
        with execution_state.worker_scope(scope):
            sub_result, restarting = await _run_with_stage_timing(_drive_backend(
                backend, workspace_path,
                _build_backend_prompt(
                    backend,
                    flexible.build_subtask_prompt(
                        request, plan, index, earlier, previous_draft=previous_draft,
                    ),
                    collaborative=False, allow_detached_requests=False,
                ),
                model, worker_approval, effort=effort, on_event=on_event,
                # A parallel wave has ONE inject watcher; it stops every sibling.
                inject_queue=None if parallel else inject_queue,
                injected=None if parallel else injected, on_process=on_process,
            ), "flexible_worker", backend_name, model,
                worker_index=index + 1, worker_count=total)
        if budget is not None:
            budget.capture(sub_result, workspace_path)
        if not restarting:
            worker_outputs[str(index)] = encode_output(sub_result, text=True)
            try:
                retain_report(index, sub_result)
            except _TurnBudgetExhaustedError:
                await save_progress()
                raise
            if parallel:
                # Siblings save independently, so one cancelled reader cannot
                # erase another reader's completed evidence. Serial workers
                # save once below, after advancing the durable cursor.
                await save_progress()
        return sub_result, restarting

    max_parallel = config.get_flexible_max_parallel_workers()
    while done < len(queue):
        if _take_pending_injections(inject_queue, injected):
            return await restart(result)
        task = queue[done]
        tier_backend_name, tier_model = tiers[task.tier]
        target = (tier_backend_name, tier_model)
        unavailable_cause = unavailable_models.get(target)
        if unavailable_cause is not None:
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
            done += 1
            await save_progress()
            continue
        end = done + 1
        if can_inspect(done):
            while (end < len(queue) and end - done < max_parallel
                   and can_inspect(end) and tiers[queue[end].tier] not in unavailable_models):
                end += 1
        if budget is not None:
            remaining = budget.max_workers - budget.workers_started
            if remaining <= 0:
                budget.claim_worker()  # raises with the established checkpoint protocol
            end = min(end, done + remaining)
        earlier = list(reports)
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
        # Always merge in plan order, regardless of completion order.
        for sub_result, restarting in values:
            task = queue[done]
            tier_backend_name, tier_model = tiers[task.tier]
            target = (tier_backend_name, tier_model)
            if restarting:
                return await restart(result)
            # Worker text is internal (feeds the merge).
            result.events.extend(
                ev for ev in sub_result.events if ev.kind != "text"
            )
            for task_ref in sub_result.detached_tasks:
                if task_ref not in result.detached_tasks:
                    result.detached_tasks.append(task_ref)
            if str(done) not in restored_outputs:
                _accumulate_usage(usage_totals, sub_result.usage)
            _accumulate_usage(usage_audit, sub_result.usage)

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

            # Keep every distinct leftover, but consolidate compatible adjacent
            # scopes instead of multiplying a report into dozens of workers.
            followup_report = worker_outputs.get(str(done), {}).get("followups", report)
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
        # A completed phase must never be returned again to a judge CONTINUE.
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
            timeout=flexible.MERGE_TIMEOUT,
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
    """
    async with (
        workspace_mod.get_run_lock(workspace_path),
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
    async with workspace_mod.get_run_lock(workspace_path):
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
            async with workspace_mod.get_lock(workspace_path):
                if not _log_to_session(workspace_path, session_id, prompt, started):
                    raise RuntimeError("Could not log the detached task's session")
            # Deliver cancellation requested during synchronous session logging
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
        async with workspace_mod.get_lock(workspace_path):
            session.append_messages(workspace_path, session_id, [{
                "role": "assistant",
                "content": text,
            }])
    except Exception:
        logger.error("Failed to log detached task completion", exc_info=True)


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

    async def _append() -> None:
        # Wait for an in-flight normal write so it never gains a duplicate.
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
            lambda: _append_session_messages(
                turn.workspace_path, session_id, [
                    {"role": "user", "content": turn.prompt},
                    {"role": "assistant", "content": note},
                ],
                "Failed to log interrupted turn",
            ),
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
) -> AgentResult:
    """Run the selected agent backend with session history prepended.

    backend_name selects the registered agent adapter. When None, the
    default backend is used. The workspace's configured backend should be
    passed in by the caller.

    session_id pins the run to a specific session (used for ephemeral
    schedule sessions). When None, the user's last session is resumed if
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
        timer = asyncio.timeout_at(budget.deadline)
        token = _TURN_TIMEOUT.set(timer)
        try:
            async with timer:
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
                )
        except TimeoutError:
            if not timer.expired():
                raise
            reason = f"the {budget.timeout_seconds}s whole-turn time limit was reached"
            limit_kind = "time"
        except _TurnBudgetExhaustedError as exc:
            reason = str(exc)
            limit_kind = "tools" if "tool budget exhausted" in reason.casefold() else "workers"
        except asyncio.CancelledError:
            # /stop and daemon shutdown retain their original cancellation contract.
            _log_interrupted_turn(turn)
            raise
        finally:
            _TURN_TIMEOUT.reset(token)

        continuation_reason: ContinuationReason = (
            "time_budget" if limit_kind == "time"
            else "tool_budget" if limit_kind == "tools"
            else "worker_budget"
        )
        # An accepted message is part of this request even if the allowance
        # expires before its injected backend restart can run.
        _close_inject_queue(inject_queue)
        _take_pending_injections(inject_queue, turn.injected)
        if turn.injected:
            turn.prompt = prompt + "\n\n[Added while you were thinking]:\n" + "\n".join(turn.injected)
        result = budget.finish(reason, turn, continuation_reason=continuation_reason)
        performance_logger.info(
            "stage=turn_budget outcome=exhausted limit=%s workers_started=%d timeout_seconds=%d",
            limit_kind, budget.workers_started, budget.timeout_seconds,
        )
        if turn.session_id:
            budget_session_id = turn.session_id

            async def log_budget_result() -> bool:
                turn.logged_normally = await _persist_session_write(
                    workspace_path,
                    lambda: _log_to_session(workspace_path, budget_session_id, turn.prompt, result),
                )
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
        assert session_id is not None  # explicit_session == (session_id set)
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
    turn.session_id = session_id
    if on_session is not None:
        await on_session(session_id)
    # Workspace memory loads once.
    colony_items = colony.get_items(workspace_path)

    # Scheduled/ephemeral turns run autonomous.
    collaborative = _is_collaborative_turn(
        workspace_path, explicit_session=explicit_session,
    )

    # Prepended context-block budget.
    history_budget = workspace_mod.get_history_budget(workspace_path)

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
    ) -> tuple[AgentResult, bool]:
        """Drive initial and continuation phases with the same turn policy."""
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
            )
        return await _drive_backend(
            backend, workspace_path,
            _build_backend_prompt(
                backend, phase_prompt,
                collaborative=collaborative,
                allow_detached_requests=not explicit_session,
            ),
            model, approval,
            effort=effort,
            on_event=_stream_event,
            inject_queue=inject_queue,
            injected=injected,
            close_inject_on_completion=False,
            on_process=on_process,
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
            budget.latest_draft = flexible._bounded_draft_excerpt(result.text)
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
            continuation_prompt = (
                contextual_prompt
                + "\n\n[Previous draft — continue its work and return the completed answer]:\n"
                + flexible._bounded_draft_excerpt(draft, limit=history_budget)
                + followup
            )
            try:
                result, restarting = await _run_phase(
                    continuation_prompt, effective_prompt + followup,
                    previous_draft=draft,
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
                budget.latest_draft = flexible._bounded_draft_excerpt(result.text)
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

    # The completion deadline covers agent work, not durable reply/session writes.
    timer = _TURN_TIMEOUT.get()
    if timer is not None:
        timer.reschedule(None)
    # Close before session/reply yields.
    _close_inject_queue(inject_queue)
    # Discard Queue messages after the final answer.
    _drain_queue(inject_queue)

    # Consume the marker before session/chat delivery.
    _consume_detached_task_requests(result)

    # The detached owner retains the file lock if /stop interrupts the caller.
    async def _log_normal_turn() -> bool:
        turn.logged_normally = await _persist_session_write(
            workspace_path,
            lambda: _log_to_session(
                workspace_path, session_id, effective_prompt, result,
            ),
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

    # Compaction runs post-turn.
    create_background_task(
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
