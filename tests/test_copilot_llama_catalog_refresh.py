import time
import unittest
from unittest import mock

from Cozter.backends_agent.copilot import CopilotBackend
from Cozter.backends_agent import llama


class CopilotCatalogRefreshTests(unittest.TestCase):
    def test_refresh_clears_failure_throttle_without_evicting_other_workspace(self):
        backend = CopilotBackend()
        first = "/catalog/first"
        other = "/catalog/other"
        other_key = backend._workspace_catalog_key(other)
        backend._workspace_model_catalogs[other_key] = (
            ("auto", "other-policy-model"), time.monotonic() + 60,
        )
        with mock.patch.object(
            backend, "_discover_models", side_effect=[None, ("auto", "new-model")],
        ) as discover:
            self.assertEqual(backend.available_models_for_workspace(first), ("auto",))
            backend.invalidate_model_catalog(first)
            self.assertEqual(
                backend.available_models_for_workspace(first), ("auto", "new-model"),
            )
            self.assertEqual(
                backend.available_models_for_workspace(other), ("auto", "other-policy-model"),
            )
        self.assertEqual(discover.call_count, 2)

    def test_all_workspace_invalidation_drops_policy_and_failure_caches(self):
        backend = CopilotBackend()
        now = time.monotonic()
        backend._workspace_model_catalogs["first"] = (("auto", "old-model"), now + 60)
        backend._workspace_fallback_expires_at["second"] = now + 60

        backend.invalidate_model_catalog()

        self.assertEqual(backend._workspace_model_catalogs, {})
        self.assertEqual(backend._workspace_fallback_expires_at, {})


class LlamaCatalogRefreshTests(unittest.TestCase):
    def test_endpoint_edit_cannot_reuse_previous_server_catalog(self):
        backend = llama.LlamaBackend()
        with (
            mock.patch.object(llama.cfg, "get_llama_server_url", return_value="http://first:8080") as url,
            mock.patch.object(llama, "fetch_model_ids", side_effect=[("first-model",), ("second-model",)]) as fetch,
        ):
            self.assertEqual(backend.available_models, ("first-model",))
            url.return_value = "http://first:8080/"
            self.assertEqual(backend.available_models, ("first-model",))
            url.return_value = "http://second:8080"
            self.assertEqual(backend.available_models, ("second-model",))
        self.assertEqual(
            fetch.call_args_list,
            [
                mock.call("http://first:8080/v1/models", timeout=2.0),
                mock.call("http://second:8080/v1/models", timeout=2.0),
            ],
        )
