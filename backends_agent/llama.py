"""llama-server backend: OpenAI-compatible in-process agent loop.

llama-server is a local chat-completions endpoint with no auth. The whole
tool-calling loop lives in :class:`OpenAIChatBackend`; this module only
supplies llama's specifics - the local endpoint URL, dynamic model
discovery from ``/v1/models``, and the per-loop limits from the
``llama_*`` config knobs.

Config: ``config.json``'s ``llama_server_url`` (default
``http://127.0.0.1:8080``). The server must speak OpenAI-compatible
``/v1/chat/completions`` with streaming and tool calls.
"""

from __future__ import annotations

import logging

from .. import config as cfg
from ._openai_agent import CachedOpenAIChatBackend, fetch_model_ids

logger = logging.getLogger(__name__)


class LlamaBackend(CachedOpenAIChatBackend):
    name = "llama"
    executable = "llama-server"  # "not found" error text only
    # From /v1/models, cached briefly.
    default_model = "auto"
    default_summary_model = "auto"

    def _models_url(self) -> str:
        return cfg.get_llama_server_url().rstrip("/") + "/v1/models"

    def _model_catalog_key(self) -> object:
        return self._models_url()

    def _fetch_models(self) -> tuple[str, ...]:
        url = self._models_url()
        try:
            return fetch_model_ids(url, timeout=2.0) or ("auto",)
        except Exception as exc:
            logger.debug(
                "Could not query %s for models (%s); using 'auto'",
                url, exc,
            )
            return ("auto",)

    def health_check(self) -> tuple[bool, str]:
        # HTTP endpoint: probe /v1/models, not PATH.
        url = self._models_url()
        try:
            ids = fetch_model_ids(url, timeout=2.0)
        except Exception as exc:
            return False, f"server unreachable at {url}: {exc}"
        if ids:
            return True, f"server up at {url} ({len(ids)} model(s))"
        return True, f"server up at {url} (no models listed)"

    def _chat_endpoint(self) -> str:
        return cfg.get_llama_server_url().rstrip("/") + "/v1/chat/completions"

    def _request_model(self, model: str | None) -> str | None:
        # "auto" = server decides: drop the field.
        return None if not model or model == "auto" else model

    def _max_agent_turns(self) -> int:
        return cfg.get_llama_max_agent_turns()

    def _tool_repeat_limit(self) -> int:
        return cfg.get_llama_tool_repeat_limit()

    def _max_retries(self) -> int:
        return cfg.get_llama_max_retries()
