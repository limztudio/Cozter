"""Codex CLI backend."""

import asyncio
import json
import logging
import shutil
import subprocess
import threading
import time

from .base import (
    CLI_MODEL_DISCOVERY_TIMEOUT_SEC,
    MODEL_CATALOG_TTL_SEC, AgentResult, Backend, ChatEvent, append_text_result,
    attachment_image_paths,
    create_prompt_subprocess, executable_command, fallback_model_tables,
    record_backend_error,
    truncate_status_text,
)

logger = logging.getLogger(__name__)

_COMMON_EFFORT_LEVELS = ("low", "medium", "high", "xhigh")
# Discovery fallback: curated IDs + capabilities (0.160.1).
_FALLBACK_MODEL_SPECS = (
    ("gpt-6.1-sol", (*_COMMON_EFFORT_LEVELS, "max", "ultra"), 272_000),
    ("gpt-6-astra", (*_COMMON_EFFORT_LEVELS, "max", "ultra"), 272_000),
    ("gpt-6-sol", (*_COMMON_EFFORT_LEVELS, "max", "ultra"), 272_000),
    ("gpt-6-luna", (*_COMMON_EFFORT_LEVELS, "max"), 272_000),
    ("gpt-5.6-sol", (*_COMMON_EFFORT_LEVELS, "max", "ultra"), 272_000),
    ("gpt-5.6-terra", (*_COMMON_EFFORT_LEVELS, "max", "ultra"), 272_000),
    ("gpt-5.6-luna", (*_COMMON_EFFORT_LEVELS, "max"), 272_000),
    # GPT-5.5 remains selectable until its 2026-10-14 Codex retirement.
    ("gpt-5.5", _COMMON_EFFORT_LEVELS, 272_000),
)
(
    _FALLBACK_MODELS,
    _FALLBACK_MODEL_EFFORT_LEVELS,
    _FALLBACK_MODEL_CONTEXT_WINDOWS,
) = fallback_model_tables(_FALLBACK_MODEL_SPECS)


def _parse_debug_models_metadata(
    output: str | bytes,
) -> tuple[
    tuple[str, ...], dict[str, tuple[str, ...]], dict[str, int],
]:
    """Extract visible models, effort levels, and active context windows.

    ``max_context_window`` can be higher than what the CLI enables for the
    current account or service tier.  Compaction must follow the live
    ``context_window`` value instead, so a model with an optional 1M mode
    does not delay its safety trigger while operating in a 272K session.
    """
    try:
        payload = json.loads(output)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        return (), {}, {}
    if not isinstance(payload, dict):
        return (), {}, {}
    catalog = payload.get("models")
    if not isinstance(catalog, list):
        return (), {}, {}

    models: list[str] = []
    efforts_by_model: dict[str, tuple[str, ...]] = {}
    context_windows: dict[str, int] = {}
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
        # Empty level list = no reasoning override.
        efforts_by_model[slug] = tuple(efforts)

        # Only explicit context_window is the session input limit.
        context_window = entry.get("context_window")
        if (
            isinstance(context_window, int)
            and not isinstance(context_window, bool)
            and context_window > 0
        ):
            context_windows[slug] = context_window

    return tuple(models), efforts_by_model, context_windows


def _stderr_preview(value: str | bytes | None) -> str:
    """Return a safe short stderr preview without platform decoding errors."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    if len(cleaned) > 200:
        return cleaned[:200 - len("… [clipped]")] + "… [clipped]"
    return cleaned


class CodexBackend(Backend):
    name = "codex"
    executable = "codex"
    supports_vision = True
    vision_mode = "cli_file_flag"
    # No no-tools mode: read-only sandbox is the confirm/deny fallback.
    permission_arg_sets = {
        "full": ("--dangerously-bypass-approvals-and-sandbox",),
        "auto": ("--sandbox", "workspace-write"),
        "restricted": ("--sandbox", "read-only"),
    }
    default_model = "gpt-6.1-sol"
    default_summary_model = "gpt-6-luna"
    # Keep focused work on Luna and progressively stronger work on Sol.
    tier_models = {
        "low": "gpt-6-luna",
        "mid": "gpt-6-sol",
        "high": "gpt-6.1-sol",
    }
    common_effort_levels = _COMMON_EFFORT_LEVELS
    effort_levels = (*common_effort_levels, "max", "ultra")

    def __init__(self) -> None:
        # Singletons: short-interval refresh.
        self._cached_model_catalog: (
            tuple[tuple[str, ...], dict[str, tuple[str, ...]]] | None
        ) = None
        self._model_context_windows = dict(_FALLBACK_MODEL_CONTEXT_WINDOWS)
        self._catalog_expires_at = 0.0
        self._model_catalog_lock = threading.Lock()

    # discovery

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
        vocabulary for compatibility with existing direct model settings. An
        expired catalog is likewise not safe to use: the next picker refresh
        may reflect a changed account policy or CLI model set.
        """
        if (
            self._cached_model_catalog is None
            or time.monotonic() >= self._catalog_expires_at
        ):
            return _FALLBACK_MODEL_EFFORT_LEVELS
        return self._cached_model_catalog[1]

    def context_window_tokens(self, model: str | None) -> int | None:
        """Return fresh cached capacity without probing the CLI.

        Keep the conservative fallback after a live catalog expires. This
        prevents a removed private model from delaying compaction until the
        next model picker has had a chance to refresh the catalog.
        """
        selected_model = model or self.default_model
        if (
            self._cached_model_catalog is None
            or time.monotonic() >= self._catalog_expires_at
        ):
            return _FALLBACK_MODEL_CONTEXT_WINDOWS.get(selected_model)
        return self._model_context_windows.get(selected_model)

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
                models, efforts, context_windows = self._discover_models()
                self._cached_model_catalog = models, efforts
                self._model_context_windows = {
                    **_FALLBACK_MODEL_CONTEXT_WINDOWS,
                    **context_windows,
                }
                self._catalog_expires_at = (
                    time.monotonic() + MODEL_CATALOG_TTL_SEC
                )
        return self._cached_model_catalog

    def _discover_models(self) -> tuple[
        tuple[str, ...], dict[str, tuple[str, ...]], dict[str, int],
    ]:
        binary = shutil.which(self.executable)
        if binary is None:
            logger.debug("codex not on PATH; using fallback model list")
            return (
                _FALLBACK_MODELS,
                _FALLBACK_MODEL_EFFORT_LEVELS,
                _FALLBACK_MODEL_CONTEXT_WINDOWS,
            )

        prefix = executable_command(self.executable)
        try:
            proc = subprocess.run(
                [*prefix, "debug", "models"],
                capture_output=True,
                timeout=CLI_MODEL_DISCOVERY_TIMEOUT_SEC,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.debug(
                "codex debug models probe failed (%s); using fallback", exc,
            )
            return (
                _FALLBACK_MODELS,
                _FALLBACK_MODEL_EFFORT_LEVELS,
                _FALLBACK_MODEL_CONTEXT_WINDOWS,
            )
        if proc.returncode != 0:
            # Stale config can block the probe: retry with override.
            try:
                recovered = subprocess.run(
                    [
                        *prefix,
                        "-c", 'model_reasoning_effort="high"',
                        "debug", "models",
                    ],
                    capture_output=True,
                    timeout=CLI_MODEL_DISCOVERY_TIMEOUT_SEC,
                    check=False,
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                logger.debug(
                    "codex debug models recovery probe failed (%s); "
                    "using fallback",
                    exc,
                )
                return (
                    _FALLBACK_MODELS,
                    _FALLBACK_MODEL_EFFORT_LEVELS,
                    _FALLBACK_MODEL_CONTEXT_WINDOWS,
                )
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
                return (
                    _FALLBACK_MODELS,
                    _FALLBACK_MODEL_EFFORT_LEVELS,
                    _FALLBACK_MODEL_CONTEXT_WINDOWS,
                )

        models, efforts, context_windows = _parse_debug_models_metadata(
            proc.stdout,
        )
        if not models:
            logger.debug(
                "codex debug models yielded no visible model catalog; "
                "using fallback",
            )
            return (
                _FALLBACK_MODELS,
                _FALLBACK_MODEL_EFFORT_LEVELS,
                _FALLBACK_MODEL_CONTEXT_WINDOWS,
            )
        return models, efforts, context_windows

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
        self.append_launch_options(
            cmd,
            model,
            effort,
            approval,
            model_flag="-m",
            effort_flag="-c",
            # Effort rides the config-override flag.
            effort_template="model_reasoning_effort={effort}",
        )
        cmd.append("-")  # read prompt from stdin

        # Vision: repeatable -i/--image flags ride as real pixels.
        if self.supports_vision and not compaction:
            for _image_path in attachment_image_paths(prompt, workspace_path):
                cmd += ["--image", _image_path]

        return await create_prompt_subprocess(cmd, prompt)

    def parse_event(self, event: dict, result: AgentResult) -> None:
        if not isinstance(event, dict):
            return
        etype = event.get("type", "")
        # `or {}` guards present-but-null `"item"`.
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
            # Record, but never replace a streamed reply.
            record_backend_error(result, err)

        elif etype == "error":
            # Stream failure without turn.failed: record it.
            msg = event.get("message", "Unknown error")
            logger.warning("Codex stream error: %s", msg)
            # Never let a late error overwrite the reply.
            record_backend_error(result, msg)

    def extract_agent_text(self, event: dict) -> str | None:
        if not isinstance(event, dict):
            return None
        if event.get("type") != "item.completed":
            return None
        item = event.get("item") or {}
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            return None
        text = item.get("text")
        return text if isinstance(text, str) and text else None
