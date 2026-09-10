"""Meta Model API backend: OpenAI-compatible cloud API for the Muse models.

Meta's Model API serves the Muse families (Muse Spark text/reasoning chat
models) through an OpenAI-compatible endpoint with Bearer auth. It reuses
the shared :class:`OpenAIChatBackend` loop; this module supplies only Meta's
specifics - the compat endpoint, the Authorization header built from the
configured API key, the model, and the Muse chat-model fallback list.

Config: ``config.json``'s ``meta_api_key`` (required to use it),
``meta_base_url`` (default ``https://api.llama.com/compat/v1``, already
includes the version so only ``/chat/completions`` is appended),
``meta_socket_timeout``, and ``meta_max_retries``. Pick the model with
``/model`` (or set the workspace default); add private or preview model ids
via ``extra_models`` in config without editing source.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from .. import config as cfg
from ._openai_agent import CachedOpenAIChatBackend, fetch_model_ids

logger = logging.getLogger(__name__)


class _FallbackModelSpec(NamedTuple):
    """Curated Meta Model API fallback metadata for one chat model."""

    name: str
    context_window: int


# Safety net for unavailable/unauthorized model discovery. The account's
# ``/models`` catalog is preferred whenever it can be queried. Keep every
# fallback capability alongside its model ID so picker order and compaction
# cannot drift apart. Provider-published capacities apply only to these
# curated public IDs; private/discovered models stay unknown until an
# operator configures model_context_windows.
#
# Muse Spark is Meta Superintelligence Labs' multimodal reasoning model for
# agentic tasks (tool calling, coding, computer use) with a documented
# 1-million-token context window that the model actively manages. Only
# chat-completion model IDs belong here: Muse Image (image generation) and
# Muse Voice Transcribe (speech-to-text) are invoked through other endpoints.
_FALLBACK_MODEL_SPECS = (
    # Muse Spark 1.3 is the current flagship on the Model API.
    _FallbackModelSpec("muse-spark-1.3", 1_000_000),
    _FallbackModelSpec("muse-spark-1.2", 1_000_000),
    _FallbackModelSpec("muse-spark-1.1", 1_000_000),
)
_FALLBACK_MODELS = tuple(spec.name for spec in _FALLBACK_MODEL_SPECS)
_MODEL_CONTEXT_WINDOWS = {
    spec.name: spec.context_window for spec in _FALLBACK_MODEL_SPECS
}
# Meta's catalog also includes models invoked through non-chat endpoints.
# Keep this deliberately small and exact: unknown/private IDs stay selectable
# because they may be valid chat models on an operator's account.
_NON_CHAT_COMPLETION_MODEL_PREFIXES = (
    "muse-image",
    "muse-voice",
)
_MODEL_DISCOVERY_TIMEOUT_SEC = 10


def _chat_completion_model_ids(model_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Drop known Meta IDs that require an endpoint other than chat."""
    return tuple(
        model_id for model_id in model_ids
        if not _is_non_chat_completion_model_id(model_id)
    )


def _is_non_chat_completion_model_id(model_id: str) -> bool:
    normalized = model_id.strip().casefold()
    return any(
        normalized.startswith(prefix)
        for prefix in _NON_CHAT_COMPLETION_MODEL_PREFIXES
    )


def _capability_model_id(model: str | None) -> str:
    """Normalize Meta's pricing-tier suffixes for capability lookups.

    The request must keep the exact selected ID, but a tier variant such as
    ``muse-spark-1.3-contributor`` shares the base model's published context
    window and streaming behavior, so capability lookups use the base name.
    """
    return (model or "").strip().casefold().removesuffix("-contributor")


class MetaModelApiBackend(CachedOpenAIChatBackend):
    name = "meta"
    executable = "meta"  # HTTP backend; never spawns a subprocess

    default_model = "muse-spark-1.3"
    default_summary_model = "muse-spark-1.2"
    # The contributor pricing tier's exact API ID is not published yet, so
    # the verified previous generation (1.2) anchors the cheap tier; once
    # discovery works the account's real catalog supersedes this table.
    tier_models = {
        "low": "muse-spark-1.2",
        "mid": "muse-spark-1.3",
        "high": "muse-spark-1.3",
    }

    def context_window_tokens(self, model: str | None) -> int | None:
        """Return the published capacity for a curated Muse chat model."""
        selected = _capability_model_id(model or self.default_model)
        return _MODEL_CONTEXT_WINDOWS.get(selected)

    # ---- model discovery -----------------------------------------------

    def _fetch_models(self) -> tuple[str, ...]:
        key = cfg.get_meta_api_key()
        base_url = cfg.get_meta_base_url()
        if not key:
            logger.debug(
                "Meta Model API key is unset; using fallback model list",
            )
            return _FALLBACK_MODELS

        url = base_url.rstrip("/") + "/models"
        try:
            model_ids = fetch_model_ids(
                url,
                timeout=_MODEL_DISCOVERY_TIMEOUT_SEC,
                headers={"Authorization": f"Bearer {key}"},
            )
        except Exception as exc:
            logger.debug(
                "Could not query Meta Model API models at %s (%s); "
                "using fallback",
                url, exc,
            )
            return _FALLBACK_MODELS

        chat_models = _chat_completion_model_ids(model_ids)
        if not chat_models and model_ids:
            logger.debug(
                "Meta Model API catalog contained no chat-completion "
                "models; using fallback",
            )
        return chat_models or _FALLBACK_MODELS

    # ---- OpenAIChatBackend hooks ---------------------------------------

    def _chat_endpoint(self) -> str:
        # base_url already carries the /compat/v1 version segment, so we
        # append /chat/completions directly (NOT /v1/chat/completions).
        return cfg.get_meta_base_url().rstrip("/") + "/chat/completions"

    def _auth_headers(self) -> dict[str, str]:
        key = cfg.get_meta_api_key()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _request_model(self, model: str | None) -> str:
        # The Model API requires a model field; fall back to the default.
        return model or self.default_model

    def _auto_continue_after_tool_limit(self) -> bool:
        # Muse Spark is trained for long agentic coding runs; keep going in
        # a fresh segment instead of forcing a no-tools final answer.
        return True

    def _socket_timeout(self) -> int:
        return cfg.get_meta_socket_timeout()

    def _socket_timeout_setting(self) -> str:
        return "meta_socket_timeout"

    def _max_retries(self) -> int:
        return cfg.get_meta_max_retries()

    def health_check(self) -> tuple[bool, str]:
        # HTTP backend: readiness is "is an API key configured?". We don't
        # spend a real request here (that would bill the account).
        if not cfg.get_meta_api_key():
            return False, "no API key set (set meta_api_key in config.json)"
        return True, (
            f"configured (endpoint {self._chat_endpoint()},"
            f" default model {self.default_model})"
        )
