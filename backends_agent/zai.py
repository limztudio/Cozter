"""Z.ai (Zhipu GLM) backend: OpenAI-compatible cloud API.

Z.ai serves the GLM models (glm-5.2, glm-5.1, glm-5, ...) through an
OpenAI-compatible endpoint at ``https://api.z.ai/api/paas/v4`` with Bearer
auth. It reuses the shared :class:`OpenAIChatBackend` loop; this module
supplies only Z.ai's specifics - the endpoint, the Authorization header
built from the configured API key, the model, and the GLM model list.

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
from ._openai_agent import CachedOpenAIChatBackend, fetch_model_ids

logger = logging.getLogger(__name__)


class _FallbackModelSpec(NamedTuple):
    """Curated Z.ai fallback metadata for one chat-completion model."""

    name: str
    context_window: int
    preserves_reasoning: bool
    streams_tools: bool


# Safety net for unavailable/unauthorized model discovery. The installed
# account's ``/models`` catalog is preferred whenever it can be queried. Keep
# every fallback capability alongside its model ID so picker order, compaction,
# preserved reasoning, and tool streaming cannot drift apart. Provider-published
# capacities apply only to these curated public IDs; private/discovered models
# stay unknown until an operator configures model_context_windows.
#
# GLM-4.5V accepts text-only turns but not function tools in the documented
# vision request schema. The older 4-32B fallback has no documented
# preserved-thinking contract.
_FALLBACK_MODEL_SPECS = (
    _FallbackModelSpec("glm-5.2", 1_000_000, True, True),
    # This vision model supports text and native functions, but its request
    # schema does not accept the text-only ``tool_stream`` extension.
    _FallbackModelSpec("glm-5v-turbo", 200_000, True, False),
    _FallbackModelSpec("glm-5.1", 200_000, True, True),
    _FallbackModelSpec("glm-5-turbo", 200_000, True, True),
    _FallbackModelSpec("glm-5", 200_000, True, True),
    _FallbackModelSpec("glm-4.7", 200_000, True, True),
    _FallbackModelSpec("glm-4.7-flash", 200_000, True, True),
    _FallbackModelSpec("glm-4.7-flashx", 200_000, True, True),
    _FallbackModelSpec("glm-4.6", 200_000, True, True),
    # Vision variants also accept text-only turns and native function calls,
    # but not the text-only ``tool_stream`` extension.
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
_FALLBACK_MODELS = tuple(spec.name for spec in _FALLBACK_MODEL_SPECS)
_MODEL_CONTEXT_WINDOWS = {
    spec.name: spec.context_window for spec in _FALLBACK_MODEL_SPECS
}
_PRESERVED_THINKING_MODELS = frozenset(
    spec.name for spec in _FALLBACK_MODEL_SPECS if spec.preserves_reasoning
)
_TOOL_STREAM_MODELS = frozenset(
    spec.name for spec in _FALLBACK_MODEL_SPECS if spec.streams_tools
)
# Z.ai's catalog can include models that are invoked through a different API
# path (for example image generation, OCR, or audio transcription). Cozter
# drives ``/chat/completions`` and should not display those IDs as agent
# choices. Keep this deliberately small and exact: unknown/private IDs stay
# selectable because they may be valid chat models on an operator's account.
_NON_CHAT_COMPLETION_MODEL_IDS = frozenset({
    "glm-ocr",
    "glm-image",
    "cogview-4-250304",
    "glm-asr-2512",
    # This is a specialized phone-use agent, not a general chat-completions
    # model for Cozter's workspace tool surface.
    "autoglm-phone-multilingual",
})
# Z.ai's vision request schema documents native function tools for the
# GLM-4.6V family (and its specialized phone agent), but not GLM-4.5V. Keep
# GLM-4.5V available for text-only use while never sending it an unsupported
# tool schema. Unknown/private IDs retain the normal OpenAI-compatible path.
_NO_FUNCTION_TOOL_MODELS = frozenset({"glm-4.5v"})
# The provider documents these two exact models as always reasoning even when
# the thinking toggle is supplied. Do not send the contradictory disabled
# setting for a low Cozter effort percentage.
_COMPULSORY_THINKING_MODELS = frozenset({"glm-4.7", "glm-4.5v"})
_MODEL_DISCOVERY_TIMEOUT_SEC = 10


def _chat_completion_model_ids(model_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Drop known Z.ai IDs that require an endpoint other than chat."""
    return tuple(
        model_id for model_id in model_ids
        if model_id.casefold() not in _NON_CHAT_COMPLETION_MODEL_IDS
    )


def _capability_model_id(model: str | None) -> str:
    """Normalize Z.ai's model suffixes for local capability lookups.

    The request must keep the exact selected ID -- notably the Coding Plan's
    ``glm-5.2[1m]`` long-context spelling -- while effort, tool-streaming,
    context, and preserved-thinking support are shared with its base model.
    """
    return (model or "").strip().casefold().removesuffix("[1m]")


class ZaiBackend(CachedOpenAIChatBackend):
    name = "zai"
    executable = "z.ai"  # HTTP backend; never spawns a subprocess

    default_model = "glm-5.2"
    default_summary_model = "glm-4.5-air"
    tier_models = {"low": "glm-4.5-air", "mid": "glm-4.7", "high": "glm-5.2"}
    # GLM-5.2 accepts seven reasoning-effort values. Other current text models
    # expose only the thinking switch, handled separately in _effort_fields.
    effort_levels = (
        "none", "minimal", "low", "medium", "high", "xhigh", "max",
    )

    def context_window_tokens(self, model: str | None) -> int | None:
        """Return a published capacity for a curated Z.ai model ID."""
        selected = _capability_model_id(model or self.default_model)
        return _MODEL_CONTEXT_WINDOWS.get(selected)

    # ---- model discovery -----------------------------------------------

    def _models_endpoint(self) -> str:
        return cfg.get_zai_base_url().rstrip("/") + "/models"

    def _fetch_models(self) -> tuple[str, ...]:
        key = cfg.get_zai_api_key()
        if not key:
            logger.debug("Z.ai API key is unset; using fallback model list")
            return _FALLBACK_MODELS

        url = self._models_endpoint()
        try:
            model_ids = fetch_model_ids(
                url,
                timeout=_MODEL_DISCOVERY_TIMEOUT_SEC,
                headers={"Authorization": f"Bearer {key}"},
            )
        except Exception as exc:
            logger.debug(
                "Could not query Z.ai models at %s (%s); using fallback",
                url, exc,
            )
            return _FALLBACK_MODELS

        chat_models = _chat_completion_model_ids(model_ids)
        if not chat_models and model_ids:
            logger.debug(
                "Z.ai model catalog contained no chat-completion models; "
                "using fallback",
            )
        return chat_models or _FALLBACK_MODELS

    # ---- OpenAIChatBackend hooks ---------------------------------------

    def _chat_endpoint(self) -> str:
        # base_url already carries the /api/paas/v4 version segment, so we
        # append /chat/completions directly (NOT /v1/chat/completions).
        return cfg.get_zai_base_url().rstrip("/") + "/chat/completions"

    def _auth_headers(self) -> dict[str, str]:
        key = cfg.get_zai_api_key()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _request_model(self, model: str | None) -> str:
        # Z.ai requires a model field; fall back to the configured default.
        return model or self.default_model

    def _effort_fields(
        self,
        percent: int,
        model: str | None = None,
    ) -> dict:
        if percent <= 0:
            return {}
        if _capability_model_id(model or self.default_model) == "glm-5.2":
            return {
                "thinking": {"type": "enabled"},
                "reasoning_effort": self.convert_effort(percent),
            }
        if _capability_model_id(
            model or self.default_model,
        ) in _COMPULSORY_THINKING_MODELS:
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

        ``tool_stream`` is supported by Z.ai's text chat-completion models
        from GLM-4.6 onward. Its vision request schema deliberately omits the
        field, so multimodal models keep standard streamed tool-call deltas.
        The shared SSE parser handles either shape. Unrecognized
        account-specific IDs intentionally omit the optional field until
        provider documentation confirms their compatibility.
        """
        selected = _capability_model_id(model or self.default_model)
        return {"tool_stream": True} if selected in _TOOL_STREAM_MODELS else {}

    def _auto_continue_after_tool_limit(self) -> bool:
        # Long z.ai coding runs can legitimately need more tool turns than
        # Cozter's per-segment guard. Keep going in a fresh segment instead
        # of forcing a no-tools final answer.
        return True

    def _socket_timeout(self) -> int:
        return cfg.get_zai_socket_timeout()

    def _socket_timeout_setting(self) -> str:
        return "zai_socket_timeout"

    def _max_retries(self) -> int:
        return cfg.get_zai_max_retries()

    def health_check(self) -> tuple[bool, str]:
        # HTTP backend: readiness is "is an API key configured?". We don't
        # spend a real request here (that would bill the account).
        if not cfg.get_zai_api_key():
            return False, "no API key set (set zai_api_key in config.json)"
        return True, (
            f"configured (endpoint {self._chat_endpoint()},"
            f" default model {self.default_model})"
        )
