"""Z.ai (Zhipu GLM) backend: OpenAI-compatible cloud API.

Z.ai serves the GLM models (glm-5.3, glm-5.3-flash/flashx, glm-5.2, glm-5.1,
glm-5, ...) through OpenAI-compatible endpoints with Bearer auth. It reuses
the shared :class:`OpenAIChatBackend` loop; this module supplies only Z.ai's
specifics - the endpoint, the Authorization header built from the configured
API key, the model, and the GLM model list.

Config: ``config.json``'s ``zai_api_key`` (required to use it),
``zai_base_url`` (default ``https://api.z.ai/api/paas/v4``, already
includes the version so only ``/chat/completions`` is appended),
``zai_socket_timeout``, and ``zai_max_retries``. Pick the model with
``/model`` (or set the workspace default); add private or regional GLM ids via
``extra_models`` in config without editing source.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from .. import config as cfg
from ._openai_agent import (
    CachedOpenAIChatBackend,
    discover_bearer_models,
    fetch_model_ids,
)

logger = logging.getLogger(__name__)


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
_MODEL_DISCOVERY_TIMEOUT_SEC = 10


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

    # Documented multimodal: photo uploads ride as real image parts.
    supports_vision = True
    vision_mode = "openai_parts"

    default_model = "glm-5.3"
    default_summary_model = "glm-4.5-air"
    # Cheap current general model; mid stays on 4.7 so tiers stay distinct.
    tier_models = {
        "low": "glm-5.3-flash",
        "mid": "glm-4.7",
        "high": "glm-5.3",
    }
    # GLM-5.2: seven levels; 5.3 family: three; others: switch only.
    effort_levels = (
        "none", "minimal", "low", "medium", "high", "xhigh", "max",
    )

    def effort_levels_for_model(self, model: str | None) -> tuple[str, ...]:
        """Return the documented reasoning vocabulary for one GLM model."""
        if (
            _capability_model_id(model or self.default_model)
            in _GLM_5_3_REASONING_MODELS
        ):
            return _GLM_5_3_EFFORT_LEVELS
        return self.effort_levels

    def context_window_tokens(self, model: str | None) -> int | None:
        """Return a published capacity for a curated Z.ai model ID."""
        selected = _capability_model_id(model or self.default_model)
        return _MODEL_CONTEXT_WINDOWS.get(selected)

    # model discovery

    def _fetch_models(self) -> tuple[str, ...]:
        return discover_bearer_models(
            label="Z.ai",
            api_key=cfg.get_zai_api_key(),
            base_url=cfg.get_zai_base_url(),
            timeout=_MODEL_DISCOVERY_TIMEOUT_SEC,
            fallback_models=_coding_plan_fallback_models(
                cfg.get_zai_base_url(),
            ),
            keep_model_id=lambda model_id: (
                isinstance(model_id, str)
                and model_id.casefold()
                not in _NON_CHAT_COMPLETION_MODEL_IDS
            ),
            fetch=fetch_model_ids,
        )

    # OpenAIChatBackend hooks

    def _chat_endpoint(self) -> str:
        # base_url already carries the version segment: append directly.
        return cfg.get_zai_base_url().rstrip("/") + "/chat/completions"

    def _auth_headers(self) -> dict[str, str]:
        key = cfg.get_zai_api_key()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _request_model(self, model: str | None) -> str:
        # Model field is required; default when unset.
        return model or self.default_model

    def _effort_fields(
        self,
        percent: int,
        model: str | None = None,
    ) -> dict:
        if percent <= 0:
            return {}
        selected = _capability_model_id(model or self.default_model)
        if selected in _GLM_5_3_REASONING_MODELS:
            # Reasoning-only family: map percentage onto three levels.
            levels = self.effort_levels_for_model(model)
            index = min(
                percent * len(levels) // 100,
                len(levels) - 1,
            )
            return {
                "thinking": {"type": "enabled"},
                "reasoning_effort": levels[index],
            }
        if selected == "glm-5.2":
            return {
                "thinking": {"type": "enabled"},
                "reasoning_effort": self.convert_effort(percent),
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
        # Long runs may need more tool turns: continue in a fresh segment.
        return True

    def _socket_timeout(self) -> int | None:
        # Real-work cap; cancel stops instantly.
        try:
            return cfg.get_zai_socket_timeout()
        except Exception:
            return 3600

    def _socket_timeout_setting(self) -> str:
        return "zai_socket_timeout"

    def _max_retries(self) -> int:
        return cfg.get_zai_max_retries()

    def _api_key(self) -> str:
        return cfg.get_zai_api_key()

    def _api_key_setting(self) -> str:
        return "zai_api_key"
