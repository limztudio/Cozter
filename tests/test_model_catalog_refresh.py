import concurrent.futures
import threading
import time
import unittest
from unittest import mock

from Cozter import workspace
from Cozter.backends_agent.base import CachedModelCatalog
from Cozter.backends_agent.codex import CodexBackend


class ModelCatalogRefreshTests(unittest.TestCase):
    def test_explicit_refresh_discards_private_codex_capabilities_without_probing_turn(self):
        backend = CodexBackend()
        backend._cached_model_catalog = (
            ("private-model",), {"private-model": ("ultra",)},
        )
        backend._model_context_windows = {"private-model": 1_000_000}
        backend._catalog_expires_at = time.monotonic() + 60

        with mock.patch.object(
            backend, "_discover_models", return_value=(
                ("new-model",), {"new-model": ("low",)}, {"new-model": 100_000},
            ),
        ) as discover:
            backend.invalidate_model_catalog("/workspace")
            self.assertNotIn("private-model", backend.model_effort_levels)
            self.assertIsNone(backend.context_window_tokens("private-model"))
            self.assertEqual(backend.context_window_tokens("gpt-6.1-sol"), 272_000)
            discover.assert_not_called()
            self.assertEqual(backend.available_models, ("new-model",))
            self.assertEqual(backend.context_window_tokens("new-model"), 100_000)
            discover.assert_called_once_with()

    def test_workspace_refresh_bypasses_unexpired_http_cache_and_keeps_extras(self):
        backend = mock.Mock()
        cached = CachedModelCatalog()
        with mock.patch.object(cached, "_fetch_models", side_effect=[("old",), ("new",)]):
            self.assertEqual(cached._live_model_catalog(), ("old",))
            backend.invalidate_model_catalog.side_effect = cached.invalidate_model_catalog
            backend.available_models_for_workspace.side_effect = lambda _ws: cached._live_model_catalog()
            backend.allow_unverified_extra_models = True
            with (
                mock.patch.object(workspace.backends_agent, "get_backend", return_value=backend),
                mock.patch.object(workspace.config, "get_extra_models", return_value=["private-extra"]),
            ):
                self.assertEqual(
                    workspace.refresh_available_models_for_backend("meta", "/workspace"),
                    ["new", "private-extra"],
                )
            backend.invalidate_model_catalog.assert_called_once_with("/workspace")

    def test_account_changed_during_fetch_cannot_publish_previous_account_models(self):
        cached = CachedModelCatalog()
        with (
            mock.patch.object(cached, "_model_catalog_key", side_effect=["old", "new", "new", "new"]),
            mock.patch.object(cached, "_fetch_models", side_effect=[("old-private",), ("new-model",)]) as fetch,
        ):
            self.assertEqual(cached._live_model_catalog(), ("new-model",))
            self.assertEqual(cached._cached_models, ("new-model",))
            self.assertEqual(fetch.call_count, 2)

    def test_codex_metadata_never_waits_for_a_slow_catalog_probe(self):
        backend = CodexBackend()
        with backend._model_catalog_lock:
            self.assertEqual(backend.context_window_tokens("gpt-6.1-sol"), 272_000)
            self.assertIsNone(backend.context_window_tokens("private-model"))
            self.assertEqual(backend.effort_levels_for_model("gpt-6-luna")[-1], "max")

    def test_warm_codex_picker_cannot_bypass_an_in_progress_refresh(self):
        backend = CodexBackend()
        backend._cached_model_catalog = (("old-private",), {})
        backend._catalog_expires_at = time.monotonic() + 60
        started = threading.Event()

        def picker():
            started.set()
            return backend.available_models

        with (
            mock.patch.object(backend, "_discover_models", return_value=(("new-model",), {}, {})),
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool,
        ):
            with backend._model_catalog_lock:
                future = pool.submit(picker)
                self.assertTrue(started.wait(2))
                with self.assertRaises(concurrent.futures.TimeoutError):
                    future.result(timeout=0.1)
                backend._cached_model_catalog = None
                backend._catalog_expires_at = 0.0
            self.assertEqual(future.result(timeout=2), ("new-model",))
