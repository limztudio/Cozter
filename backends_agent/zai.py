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

from .. import config as cfg
from ._openai_agent import CachedOpenAIChatBackend, fetch_model_ids

logger = logging.getLogger(__name__)


# Safety net for unavailable/unauthorized model discovery.  The installed
# account's ``/models`` catalog is preferred whenever it can be queried.
_FALLBACK_MODELS = (
    "glm-5.2",
    # Current multimodal coding model. It accepts text-only agent turns and
    # function calls, so it stays usable even though Cozter does not yet send
    # image inputs through this backend.
    "glm-5v-turbo",
    "glm-5.1",
    "glm-5-turbo",
    "glm-5",
    "glm-4.7",
    "glm-4.7-flash",
    "glm-4.7-flashx",
    "glm-4.6",
    # These vision-capable variants also accept text-only turns and native
    # function calls, so they can run Cozter's in-process agent loop even
    # before the frontend grows image-attachment support for this backend.
    "glm-4.6v",
    "glm-4.6v-flashx",
    "glm-4.6v-flash",
    # GLM-4.5V accepts text alongside its image/video/file inputs. It can
    # therefore serve ordinary text agent turns, although it does not support
    # the optional incremental ``tool_stream`` protocol below.
    "glm-4.5v",
    "glm-4.5",
    "glm-4.5-air",
    "glm-4.5-x",
    "glm-4.5-airx",
    "glm-4.5-flash",
    "glm-4-32b-0414-128k",
)
# Provider-published capacities for the curated, public IDs above. The live
# OpenAI-compatible /models response only standardizes IDs, so private or
# newly discovered models intentionally remain unknown until an operator adds
# model_context_windows to config.json.
_MODEL_CONTEXT_WINDOWS = {
    "glm-5.2": 1_000_000,
    "glm-5v-turbo": 200_000,
    "glm-5.1": 200_000,
    "glm-5-turbo": 200_000,
    "glm-5": 200_000,
    "glm-4.7": 200_000,
    "glm-4.7-flash": 200_000,
    "glm-4.7-flashx": 200_000,
    "glm-4.6": 200_000,
    "glm-4.6v": 128_000,
    "glm-4.6v-flashx": 128_000,
    "glm-4.6v-flash": 128_000,
    "glm-4.5v": 64_000,
    "glm-4.5": 128_000,
    "glm-4.5-air": 128_000,
    "glm-4.5-x": 128_000,
    "glm-4.5-airx": 128_000,
    "glm-4.5-flash": 200_000,
    "glm-4-32b-0414-128k": 128_000,
}
# Z.ai documents ``tool_stream`` for GLM-4.6 and newer. Its current vision
# documentation explicitly confirms native function calling for the GLM-4.6V
# family and function calling plus streaming for GLM-5V-Turbo, so those
# curated agent models can use the same incremental tool-call path as the
# text models. Account-private IDs remain opt-in until their capability is
# known rather than receiving an unsupported optional field.
_TOOL_STREAM_MODELS = frozenset({
    "glm-5.2",
    "glm-5v-turbo",
    "glm-5.1",
    "glm-5-turbo",
    "glm-5",
    "glm-4.7",
    "glm-4.7-flash",
    "glm-4.7-flashx",
    "glm-4.6",
    "glm-4.6v",
    "glm-4.6v-flashx",
    "glm-4.6v-flash",
})
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
})
_MODEL_DISCOVERY_TIMEOUT_SEC = 10


def _chat_completion_model_ids(model_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Drop known Z.ai IDs that require an endpoint other than chat."""
    return tuple(
        model_id for model_id in model_ids
        if model_id.casefold() not in _NON_CHAT_COMPLETION_MODEL_IDS
    )


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
        selected = model or self.default_model
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
        if model == "glm-5.2":
            return {
                "thinking": {"type": "enabled"},
                "reasoning_effort": self.convert_effort(percent),
            }
        return {
            "thinking": {
                "type": "enabled" if percent >= 50 else "disabled",
            },
        }

    def _tool_request_fields(self, model: str | None) -> dict:
        """Enable incremental tool-call deltas on documented agent models.

        ``tool_stream`` is supported by Z.ai's GLM-4.6-and-newer tool-capable
        model families. The shared SSE parser already joins incremental
        OpenAI-style tool-call arguments. Unrecognized account-specific IDs
        intentionally omit the optional field until provider documentation
        confirms their compatibility.
        """
        selected = model or self.default_model
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
