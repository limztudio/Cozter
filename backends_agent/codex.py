"""Codex CLI backend."""

import asyncio
import json
import logging
import shutil
import subprocess
import threading
import time

from .base import (
    AgentResult, Backend, ChatEvent, append_text_result,
    create_prompt_subprocess, executable_command, set_error_result,
    truncate_status_text,
)

logger = logging.getLogger(__name__)


def _permission_args(approval: str) -> list[str]:
    """Translate Cozter's permission level to Codex CLI arguments.

    Codex's non-interactive CLI has no separate "no tools" switch.  Its
    read-only sandbox is therefore the strongest restriction available for
    ``confirm`` and ``deny``: it prevents writes and keeps the normal
    sandbox, although the model may still perform read-only inspection.
    """
    if approval == "full":
        return ["--dangerously-bypass-approvals-and-sandbox"]
    if approval == "auto":
        return ["--full-auto"]
    return ["--sandbox", "read-only"]


# Safety net for hosts where the CLI is unavailable, unauthenticated, or an
# older/company-managed build does not support ``codex debug models``.  The
# live catalog is preferred whenever the installed CLI can provide one.
_FALLBACK_MODELS = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex-spark",
)
_COMMON_EFFORT_LEVELS = ("low", "medium", "high", "xhigh")
_FALLBACK_MODEL_EFFORT_LEVELS = {
    "gpt-5.6-sol": (*_COMMON_EFFORT_LEVELS, "max", "ultra"),
    "gpt-5.6-terra": (*_COMMON_EFFORT_LEVELS, "max", "ultra"),
    "gpt-5.6-luna": (*_COMMON_EFFORT_LEVELS, "max"),
    "gpt-5.5": _COMMON_EFFORT_LEVELS,
    "gpt-5.4": _COMMON_EFFORT_LEVELS,
    "gpt-5.4-mini": _COMMON_EFFORT_LEVELS,
    "gpt-5.3-codex-spark": _COMMON_EFFORT_LEVELS,
}
_MODEL_DISCOVERY_TIMEOUT_SEC = 15
# Model catalogs can change when the locally installed CLI or its account
# policy changes. Keep picker results fresh without re-running the probe for
# every /model request in a long-lived bot process.
_MODEL_CATALOG_TTL_SEC = 60


def _parse_debug_models_catalog(
    output: str | bytes,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Extract picker-visible models and effort levels from Codex JSON.

    ``codex debug models`` is deliberately queried from the locally installed
    CLI: companies can expose a different, policy-controlled catalog than the
    public/default Codex picker.  Only ``visibility == 'list'`` entries are
    suitable for Cozter's model picker; hidden/internal entries should not be
    offered to users.
    """
    try:
        payload = json.loads(output)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        return (), {}
    if not isinstance(payload, dict):
        return (), {}
    catalog = payload.get("models")
    if not isinstance(catalog, list):
        return (), {}

    models: list[str] = []
    efforts_by_model: dict[str, tuple[str, ...]] = {}
    seen_models: set[str] = set()
    for entry in catalog:
        if not isinstance(entry, dict) or entry.get("visibility") != "list":
            continue
        slug = entry.get("slug")
        if not isinstance(slug, str):
            continue
        slug = slug.strip()
        if not slug or slug in seen_models:
            continue
        seen_models.add(slug)
        models.append(slug)

        efforts: list[str] = []
        seen_efforts: set[str] = set()
        levels = entry.get("supported_reasoning_levels")
        if isinstance(levels, list):
            for level in levels:
                if not isinstance(level, dict):
                    continue
                effort = level.get("effort")
                if not isinstance(effort, str):
                    continue
                effort = effort.strip()
                if effort and effort not in seen_efforts:
                    seen_efforts.add(effort)
                    efforts.append(effort)
        # An explicitly empty level list means no reasoning override should
        # be passed for this discovered model.
        efforts_by_model[slug] = tuple(efforts)

    return tuple(models), efforts_by_model


def _stderr_preview(value: str | bytes | None) -> str:
    """Return a safe short stderr preview without platform decoding errors."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return (value or "").strip()[:200]


class CodexBackend(Backend):
    name = "codex"
    executable = "codex"
    default_model = "gpt-5.6-sol"
    default_summary_model = "gpt-5.6-luna"
    tier_models = {
        "low": "gpt-5.4-mini",
        "mid": "gpt-5.6-luna",
        "high": "gpt-5.6-sol",
    }
    common_effort_levels = _COMMON_EFFORT_LEVELS
    effort_levels = (*common_effort_levels, "max", "ultra")

    def __init__(self) -> None:
        # Backends are process-wide singletons. Refresh the catalog on a
        # short interval: model selection is user-facing, so probing every
        # picker would be unnecessarily slow.
        self._cached_model_catalog: (
            tuple[tuple[str, ...], dict[str, tuple[str, ...]]] | None
        ) = None
        self._catalog_expires_at = 0.0
        self._model_catalog_lock = threading.Lock()

    # ---- model discovery -----------------------------------------------

    @property
    def available_models(self) -> tuple[str, ...]:  # type: ignore[override]
        """Models accepted by the installed Codex CLI.

        Company-managed Codex installations often expose a catalog that is
        different from Cozter's public fallback.  ``codex debug models``
        reports the active CLI/account catalog, so use it when available and
        retain the fallback when the command cannot run or parse.
        """
        return self._model_catalog()[0]

    @property
    def model_effort_levels(self) -> dict[str, tuple[str, ...]]:
        """Reasoning efforts advertised by the discovered model catalog.

        Do not start a blocking discovery just to launch a turn.  The picker
        normally warms this cache; until then, use the conservative fallback
        vocabulary for compatibility with existing direct model settings.
        """
        if self._cached_model_catalog is None:
            return _FALLBACK_MODEL_EFFORT_LEVELS
        return self._cached_model_catalog[1]

    def _model_catalog(self) -> tuple[
        tuple[str, ...], dict[str, tuple[str, ...]],
    ]:
        now = time.monotonic()
        if (
            self._cached_model_catalog is not None
            and now < self._catalog_expires_at
        ):
            return self._cached_model_catalog

        with self._model_catalog_lock:
            now = time.monotonic()
            if (
                self._cached_model_catalog is None
                or now >= self._catalog_expires_at
            ):
                self._cached_model_catalog = self._discover_models()
                self._catalog_expires_at = (
                    time.monotonic() + _MODEL_CATALOG_TTL_SEC
                )
        return self._cached_model_catalog

    def _discover_models(self) -> tuple[
        tuple[str, ...], dict[str, tuple[str, ...]],
    ]:
        binary = shutil.which(self.executable)
        if binary is None:
            logger.debug("codex not on PATH; using fallback model list")
            return _FALLBACK_MODELS, _FALLBACK_MODEL_EFFORT_LEVELS

        prefix = executable_command(self.executable)
        try:
            proc = subprocess.run(
                [*prefix, "debug", "models"],
                capture_output=True,
                timeout=_MODEL_DISCOVERY_TIMEOUT_SEC,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.debug(
                "codex debug models probe failed (%s); using fallback", exc,
            )
            return _FALLBACK_MODELS, _FALLBACK_MODEL_EFFORT_LEVELS
        if proc.returncode != 0:
            # A stale local reasoning setting can prevent even the
            # read-only catalog command from starting.  Retry with a valid,
            # temporary override; it does not write or otherwise change the
            # user's Codex configuration.  A genuine failure still uses the
            # built-in catalog below.
            try:
                recovered = subprocess.run(
                    [
                        *prefix,
                        "-c", 'model_reasoning_effort="high"',
                        "debug", "models",
                    ],
                    capture_output=True,
                    timeout=_MODEL_DISCOVERY_TIMEOUT_SEC,
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                logger.debug(
                    "codex debug models recovery probe failed (%s); "
                    "using fallback",
                    exc,
                )
                return _FALLBACK_MODELS, _FALLBACK_MODEL_EFFORT_LEVELS
            if recovered.returncode == 0:
                logger.debug(
                    "codex debug models recovered with a temporary "
                    "reasoning-effort override",
                )
                proc = recovered
            else:
                logger.debug(
                    "codex debug models exited %d (%s); using fallback",
                    recovered.returncode, _stderr_preview(recovered.stderr),
                )
                return _FALLBACK_MODELS, _FALLBACK_MODEL_EFFORT_LEVELS

        models, efforts = _parse_debug_models_catalog(proc.stdout)
        if not models:
            logger.debug(
                "codex debug models yielded no visible model catalog; "
                "using fallback",
            )
            return _FALLBACK_MODELS, _FALLBACK_MODEL_EFFORT_LEVELS
        return models, efforts

    def effort_levels_for_model(self, model: str | None) -> tuple[str, ...]:
        """Return the effort vocabulary accepted by the selected model."""
        selected_model = model or self.default_model
        return self.model_effort_levels.get(
            selected_model,
            self.common_effort_levels,
        )

    async def launch(
        self,
        workspace_path: str,
        prompt: str,
        model: str | None,
        approval: str,
        *,
        compaction: bool = False,
        effort: int = 0,
    ) -> asyncio.subprocess.Process:
        prefix = executable_command(self.executable)
        cmd = [*prefix, "exec", "--ephemeral", "--json", "-C", workspace_path]
        self.append_model_effort_args(
            cmd,
            model,
            effort,
            model_flag="-m",
            effort_flag="-c",
            # Codex CLI exposes reasoning effort via the generic config
            # override flag. Unknown levels are rejected by the CLI.
            effort_template="model_reasoning_effort={effort}",
            effort_levels=self.effort_levels_for_model(model),
        )

        # ``compaction`` identifies an internal text task; it must never
        # widen permissions.  Internal callers pass approval="deny".
        cmd += _permission_args(approval)
        cmd.append("-")  # read prompt from stdin

        return await create_prompt_subprocess(cmd, prompt)

    def parse_event(self, event: dict, result: AgentResult) -> None:
        etype = event.get("type", "")
        # ``or {}`` guards a malformed ``"item": null`` the way the
        # default alone can't: ``.get("item", {})`` returns {} only when
        # the key is absent, so a present-but-null value would otherwise
        # make ``item.get(...)`` raise AttributeError and crash the turn.
        item = event.get("item") or {}
        if not isinstance(item, dict):
            item = {}
        item_type = item.get("type", "")

        if etype == "item.completed":
            if item_type == "agent_message":
                text = item.get("text", "")
                if isinstance(text, str) and text:
                    append_text_result(result, text)

            elif item_type == "command_execution":
                cmd = item.get("command", "?")
                exit_code = item.get("exit_code", "?")
                output = item.get("aggregated_output", "")
                summary = f"$ {cmd} (exit {exit_code})"
                if output:
                    summary += f"\n{truncate_status_text(output)}"
                result.events.append(ChatEvent(kind="tool", content=summary))

            elif item_type == "file_change":
                changes = item.get("changes")
                if not isinstance(changes, list):
                    return
                for ch in changes:
                    if not isinstance(ch, dict):
                        continue
                    path = ch.get("path", "?")
                    kind = ch.get("kind", "?")
                    result.events.append(ChatEvent(
                        kind="file",
                        content=f"📄 {kind}: {path}",
                    ))

        elif etype == "turn.completed":
            usage = event.get("usage")
            if isinstance(usage, dict):
                result.usage = dict(usage)

        elif etype == "turn.failed":
            err_obj = event.get("error")
            if isinstance(err_obj, dict):
                err = err_obj.get("message") or "Unknown error"
            elif isinstance(err_obj, str):
                err = err_obj
            else:
                err = "Unknown error"
            set_error_result(result, err)

        elif etype == "error":
            # A stream-level failure (expired auth, usage limit, dropped
            # connection) does not always come with a turn.failed, and codex
            # can still exit 0 after one. Recording it is the only thing
            # standing between that and a turn that silently says nothing -
            # which the flexible merge step would read as an empty worker
            # report.
            msg = event.get("message", "Unknown error")
            logger.warning("Codex stream error: %s", msg)
            if result.text:
                # The model already answered. Keep the error, but never let
                # a late one overwrite the reply the user is owed.
                result.error = msg
            else:
                set_error_result(result, msg)

    def extract_agent_text(self, event: dict) -> str | None:
        if event.get("type") != "item.completed":
            return None
        item = event.get("item") or {}
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            return None
        return item.get("text") or None
