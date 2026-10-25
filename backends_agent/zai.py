"""Z.ai (Zhipu GLM) backend: OpenAI-compatible cloud API.

Z.ai serves the GLM models (glm-5.3, glm-5.3-flash/flashx, glm-5.2, glm-5.1,
glm-5, ...) through OpenAI-compatible endpoints with Bearer auth. It reuses
the shared :class:`OpenAIChatBackend` loop; this module supplies only Z.ai's
specifics - the endpoint, the Authorization header built from the configured
API key, the model, and the GLM model list.

Config: ``config.json``'s ``zai_api_key`` (required to use it),
``zai_base_url`` (default ``https://api.z.ai/api/paas/v4``, already
includes the version so only ``/chat/completions`` is appended),
and ``zai_max_retries``. Pick the model with
``/model`` (or set the workspace default); add private or regional GLM ids via
``extra_models`` in config without editing source.
"""

from __future__ import annotations

from typing import NamedTuple

from .. import config as cfg
from ._openai_agent import (
    MODEL_DISCOVERY_TIMEOUT_SEC as _MODEL_DISCOVERY_TIMEOUT_SEC,
    CachedOpenAIChatBackend,
    discover_bearer_models,
    fetch_model_ids,
)
from .base import effort_band


class _FallbackModelSpec(NamedTuple):
    """Curated Z.ai fallback metadata for one chat-completion model."""

    name: str
    context_window: int
    preserves_reasoning: bool
    streams_tools: bool


# Discovery fallback: curated chat IDs + capabilities.
_FALLBACK_MODEL_SPECS = (
    # General endpoint flagship.
    _FallbackModelSpec("glm-5.3", 1_000_000, True, True),
    # Flash/FlashX share the endpoint.
    _FallbackModelSpec("glm-5.3-flash", 1_000_000, True, True),
    _FallbackModelSpec("glm-5.3-flashx", 1_000_000, True, True),
    _FallbackModelSpec("glm-5.2", 1_000_000, True, True),
    # Vision: no text-only ``tool_stream`` extension.
    _FallbackModelSpec("glm-5v-turbo", 200_000, True, False),
    _FallbackModelSpec("glm-5.1", 200_000, True, True),
    _FallbackModelSpec("glm-5-turbo", 200_000, True, True),
    _FallbackModelSpec("glm-5", 200_000, True, True),
    _FallbackModelSpec("glm-4.7", 200_000, True, True),
    _FallbackModelSpec("glm-4.7-flash", 200_000, True, True),
    _FallbackModelSpec("glm-4.7-flashx", 200_000, True, True),
    _FallbackModelSpec("glm-4.6", 200_000, True, True),
    # Vision variants: native functions, no ``tool_stream`` extension.
    _FallbackModelSpec("glm-4.6v", 128_000, True, False),
    _FallbackModelSpec("glm-4.6v-flashx", 128_000, True, False),
    _FallbackModelSpec("glm-4.6v-flash", 128_000, True, False),
    _FallbackModelSpec("glm-4.5v", 64_000, True, False),
    _FallbackModelSpec("glm-4.5", 128_000, True, False),
    _FallbackModelSpec("glm-4.5-air", 128_000, True, False),
    _FallbackModelSpec("glm-4.5-x", 128_000, True, False),
    _FallbackModelSpec("glm-4.5-airx", 128_000, True, False),
    _FallbackModelSpec("glm-4.5-flash", 200_000, True, False),
    _FallbackModelSpec("glm-4-32b-0414-128k", 128_000, False, False),
)
# Coding-Plan-only IDs: keep the ``[1m]`` pin out of the fallback.
_CODING_PLAN_FALLBACK_MODEL_SPECS = (
    _FallbackModelSpec("glm-5.3[1m]", 1_000_000, True, True),
    _FallbackModelSpec("glm-5.3-flash[1m]", 1_000_000, True, True),
)
_FALLBACK_MODELS = tuple(spec.name for spec in _FALLBACK_MODEL_SPECS)
_CODING_PLAN_FALLBACK_MODELS = tuple(
    spec.name for spec in (
        *_CODING_PLAN_FALLBACK_MODEL_SPECS,
        *_FALLBACK_MODEL_SPECS,
    )
    # https://docs.z.ai/guides/vlm/glm-5.3-flash (2026-10-21).
    if spec.name != "glm-5.3-flashx"
)
_ALL_FALLBACK_MODEL_SPECS = (
    *_CODING_PLAN_FALLBACK_MODEL_SPECS,
    *_FALLBACK_MODEL_SPECS,
)
_MODEL_CONTEXT_WINDOWS = {
    spec.name: spec.context_window for spec in _ALL_FALLBACK_MODEL_SPECS
}
_PRESERVED_THINKING_MODELS = frozenset(
    spec.name
    for spec in _ALL_FALLBACK_MODEL_SPECS
    if spec.preserves_reasoning
)
_TOOL_STREAM_MODELS = frozenset(
    spec.name for spec in _ALL_FALLBACK_MODEL_SPECS if spec.streams_tools
)
# Non-chat API paths (image/OCR/audio/video): unknown IDs stay selectable.
_NON_CHAT_COMPLETION_MODEL_IDS = frozenset({
    "glm-ocr",
    "glm-image",
    "cogview-4-250304",
    "glm-asr-2512",
    # https://docs.z.ai/api-reference/video/generate-video
    "cogvideox-3",
    # Phone-use agent, not a workspace chat model.
    "autoglm-phone-multilingual",
})
# 4.6V has native functions (not 4.5V): keep 4.5V text-only.
_NO_FUNCTION_TOOL_MODELS = frozenset({"glm-4.5v"})
# These two always reason.
_COMPULSORY_THINKING_MODELS = frozenset({"glm-4.7", "glm-4.5v"})
_GLM_5_3_REASONING_MODELS = frozenset({"glm-5.3", "glm-5.3-flash", "glm-5.3-flashx"})
_GLM_5_3_EFFORT_LEVELS = ("low", "high", "max")


def _capability_model_id(model: str | None) -> str:
    """Normalize Z.ai's model suffixes for local capability lookups.

    The request must keep the exact selected ID -- notably the Coding Plan's
    ``glm-5.3[1m]`` / ``glm-5.3-flash[1m]`` long-context spelling -- while
    effort, tool-streaming, context, and preserved-thinking support are
    shared with its base model.
    """
    if not isinstance(model, str):
        return ""
    return model.strip().casefold().removesuffix("[1m]")


def _coding_plan_fallback_models(base_url: str) -> tuple[str, ...]:
    """Return the compatible curated catalog for a configured Z.ai URL."""
    normalized = base_url.rstrip("/").casefold()
    if normalized.endswith("/api/coding/paas/v4"):
        return _CODING_PLAN_FALLBACK_MODELS
    return _FALLBACK_MODELS


class ZaiBackend(CachedOpenAIChatBackend):
    name = "zai"
    executable = "z.ai"  # HTTP backend; never spawns a subprocess

    default_model = "glm-5.3"
    default_summary_model = "glm-4.5-air"
    # Cheap now; mid stays on 4.7 so tiers differ.
    tier_models = {
        "low": "glm-5.3-flash",
        "mid": "glm-4.7",
        "high": "glm-5.3",
    }
    # GLM-5.2 aliases xhigh to max; keep only its distinct levels.
    # https://docs.z.ai/guides/capabilities/thinking
    effort_levels = ("none", "high", "max")

    def effort_levels_for_model(self, model: str | None) -> tuple[str, ...]:
        """Return the documented reasoning vocabulary for one GLM model."""
        selected = _capability_model_id(model or self.default_model)
        if selected in _GLM_5_3_REASONING_MODELS:
            return _GLM_5_3_EFFORT_LEVELS
        if selected == "glm-5.2":
            return self.effort_levels
        return ()

    def context_window_tokens(self, model: str | None) -> int | None:
        """Return a published capacity for a curated Z.ai model ID."""
        selected = _capability_model_id(model or self.default_model)
        return _MODEL_CONTEXT_WINDOWS.get(selected)

    def _fetch_models(self) -> tuple[str, ...]:
        base_url = self._base_url()
        return discover_bearer_models(
            label="Z.ai",
            api_key=self._api_key(),
            base_url=base_url,
            timeout=_MODEL_DISCOVERY_TIMEOUT_SEC,
            fallback_models=_coding_plan_fallback_models(base_url),
            keep_model_id=lambda model_id: (
                isinstance(model_id, str)
                and model_id.casefold()
                not in _NON_CHAT_COMPLETION_MODEL_IDS
            ),
            fetch=fetch_model_ids,
        )

    def _base_url(self) -> str:
        return cfg.get_zai_base_url()

    def _effort_fields(
        self,
        percent: int,
        model: str | None = None,
    ) -> dict:
        if percent <= 0:
            return {}
        selected = _capability_model_id(model or self.default_model)
        if selected in _GLM_5_3_REASONING_MODELS or selected == "glm-5.2":
            return {
                "thinking": {"type": "enabled"},
                "reasoning_effort": effort_band(
                    percent, self.effort_levels_for_model(model),
                ),
            }
        if selected in _COMPULSORY_THINKING_MODELS:
            return {"thinking": {"type": "enabled"}}
        return {
            "thinking": {
                "type": "enabled" if percent >= 50 else "disabled",
            },
        }

    def _supports_tools_for_model(self, model: str | None) -> bool:
        """Avoid sending a function schema to documented chat-only models."""
        return _capability_model_id(
            model or self.default_model,
        ) not in _NO_FUNCTION_TOOL_MODELS

    def _preserve_reasoning_content(self, model: str | None) -> bool:
        """Whether this documented GLM model accepts retained reasoning."""
        return _capability_model_id(
            model or self.default_model,
        ) in _PRESERVED_THINKING_MODELS

    def _preserved_reasoning_request_fields(
        self,
        model: str | None,
        effort_fields: dict,
    ) -> dict:
        """Enable Z.ai's preserved-thinking contract for an agent turn.

        The Coding Plan endpoint enables this by default, while the standard
        endpoint requires ``clear_thinking: false``. Preserve the caller's
        explicit thinking mode and reasoning effort; when no effort override
        was selected, request the provider's normal enabled-thinking mode so
        an upcoming tool result can carry the required opaque block.
        """
        del model  # Capability was checked by _preserve_reasoning_content.
        thinking = effort_fields.get("thinking")
        if isinstance(thinking, dict):
            return {
                "thinking": {**thinking, "clear_thinking": False},
            }
        return {"thinking": {"type": "enabled", "clear_thinking": False}}

    def _tool_request_fields(self, model: str | None) -> dict:
        """Enable incremental tool-call deltas on documented agent models.

        ``tool_stream`` is supported by documented text chat-completion
        models from GLM-4.6 onward, plus the GLM-5.3-Flash/FlashX multimodal models.
        Older vision schemas deliberately omit the field, so those models
        keep standard streamed tool-call deltas. The shared SSE parser handles
        either shape. Unrecognized account-specific IDs intentionally omit the
        optional field until provider documentation confirms compatibility.
        """
        selected = _capability_model_id(model or self.default_model)
        return {"tool_stream": True} if selected in _TOOL_STREAM_MODELS else {}

    def _auto_continue_after_tool_limit(self) -> bool:
        # Long runs resume in a fresh slice.
        return True

    def _max_retries(self) -> int:
        return cfg.get_zai_max_retries()

    def _api_key(self) -> str:
        return cfg.get_zai_api_key()

    def _api_key_setting(self) -> str:
        return "zai_api_key"
