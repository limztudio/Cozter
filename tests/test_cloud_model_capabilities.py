"""Regressions for provider-specific model effort and endpoint capabilities."""

from __future__ import annotations

import unittest
from unittest import mock

from Cozter.backends_agent import meta as meta_mod
from Cozter.backends_agent import zai as zai_mod
from Cozter.backends_agent.meta import MetaModelApiBackend
from Cozter.backends_agent.zai import ZaiBackend


class MetaReasoningCapabilityTests(unittest.TestCase):
    def test_standard_spark_1_3_exposes_extended_reasoning(self) -> None:
        backend = MetaModelApiBackend()
        expected = ("minimal", "low", "medium", "high", "xhigh", "max")
        for model in (None, "", "muse-spark-1.3", " MUSE-SPARK-1.3 "):
            with self.subTest(model=model):
                self.assertEqual(backend.effort_levels_for_model(model), expected)
                self.assertEqual(
                    backend._effort_fields(100, model),
                    {"reasoning_effort": "max"},
                )
                self.assertEqual(
                    backend._effort_fields(70, model),
                    {"reasoning_effort": "xhigh"},
                )

    def test_older_and_contributor_models_never_request_max(self) -> None:
        backend = MetaModelApiBackend()
        expected = ("minimal", "low", "medium", "high", "xhigh")
        for model in (
            "muse-spark-1.2", "muse-spark-1.1",
            "muse-spark-1.3-contributor", " MUSE-SPARK-1.2-CONTRIBUTOR ",
        ):
            with self.subTest(model=model):
                self.assertEqual(backend.effort_levels_for_model(model), expected)
                self.assertEqual(
                    backend._effort_fields(100, model),
                    {"reasoning_effort": "xhigh"},
                )
                self.assertEqual(backend.context_window_tokens(model), 1_048_576)
                self.assertEqual(backend._request_model(model), model)

    def test_reasoning_override_keeps_provider_default_at_zero(self) -> None:
        backend = MetaModelApiBackend()
        for model in (*meta_mod._FALLBACK_MODELS, "muse-spark-1.3-contributor"):
            with self.subTest(model=model):
                self.assertEqual(backend._effort_fields(0, model), {})
                self.assertEqual(
                    backend._effort_fields(1, model),
                    {"reasoning_effort": "minimal"},
                )

    def test_private_models_keep_conservative_effort_vocabulary(self) -> None:
        backend = MetaModelApiBackend()
        self.assertEqual(
            backend.effort_levels_for_model("private-muse"),
            ("minimal", "low", "medium", "high"),
        )
        self.assertEqual(
            backend._effort_fields(100, "private-muse"),
            {"reasoning_effort": "high"},
        )


class ZaiModelDiscoveryCapabilityTests(unittest.TestCase):
    def test_video_generation_model_is_not_selectable_for_chat(self) -> None:
        with (
            mock.patch.object(zai_mod.cfg, "get_zai_api_key", return_value="key"),
            mock.patch.object(
                zai_mod.cfg, "get_zai_base_url",
                return_value="https://api.z.ai/api/paas/v4",
            ),
            mock.patch.object(
                zai_mod, "fetch_model_ids",
                return_value=(
                    "cogvideox-3", "glm-5.3", "COGVIDEOX-3",
                    "glm-5.3-flashx", "private-video-chat",
                ),
            ),
        ):
            self.assertEqual(
                ZaiBackend().available_models,
                ("glm-5.3", "glm-5.3-flashx", "private-video-chat"),
            )

    def test_video_only_discovery_falls_back_to_chat_models(self) -> None:
        with (
            mock.patch.object(zai_mod.cfg, "get_zai_api_key", return_value="key"),
            mock.patch.object(
                zai_mod.cfg, "get_zai_base_url",
                return_value="https://api.z.ai/api/paas/v4",
            ),
            mock.patch.object(
                zai_mod, "fetch_model_ids", return_value=("cogvideox-3",),
            ),
        ):
            self.assertEqual(ZaiBackend().available_models, zai_mod._FALLBACK_MODELS)


if __name__ == "__main__":
    unittest.main()
