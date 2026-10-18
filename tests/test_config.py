"""Configuration reads preserve validation, cache safety, and endpoint rules."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from Cozter import config
from Cozter.tests.helpers import temporary_config


class ConfigCacheTests(unittest.TestCase):
    def test_stamped_reads_reuse_cache_and_changed_stamp_refreshes(self) -> None:
        with temporary_config({"value": "initial"}):
            initial = config._load_config_object()
            with patch("builtins.open", side_effect=AssertionError("cache miss")):
                self.assertIs(config._load_config_object(), initial)
            Path(config.CONFIG_PATH).write_text(
                json.dumps({"value": "updated with a different size"}),
                encoding="utf-8",
            )
            self.assertEqual(
                config._load_config_object(),
                {"value": "updated with a different size"},
            )

    def test_stat_failure_reads_current_file_without_replacing_cache(self) -> None:
        with temporary_config({"value": "initial"}):
            stamped = config._load_config_object()
            Path(config.CONFIG_PATH).write_text(
                json.dumps({"value": "updated"}), encoding="utf-8",
            )
            with patch.object(config.os, "stat", side_effect=PermissionError("stat denied")):
                self.assertEqual(config._load_config_object(), {"value": "updated"})
                self.assertIs(config._config_cache_data, stamped)

    def test_stat_failure_still_rejects_invalid_file_content(self) -> None:
        for content, message in (
            ("[]", "config.json must contain a JSON object"),
            ("not json", "Expecting value"),
            ("[" * 102 + "0" + "]" * 102, "JSON nesting exceeds"),
        ):
            with self.subTest(content=content), temporary_config({"value": "initial"}):
                stamped = config._load_config_object()
                Path(config.CONFIG_PATH).write_text(content, encoding="utf-8")
                with patch.object(config.os, "stat", side_effect=PermissionError("stat denied")):
                    with self.assertRaisesRegex(ValueError, message):
                        config._load_config_object()
                    self.assertIs(config._config_cache_data, stamped)

    def test_stat_failure_does_not_hide_unreadable_file(self) -> None:
        with temporary_config({"value": "initial"}):
            with patch.object(config.os, "stat", side_effect=PermissionError("stat denied")):
                with patch("builtins.open", side_effect=PermissionError("read denied")):
                    with self.assertRaisesRegex(PermissionError, "read denied"):
                        config._load_config_object()


class HttpsEndpointTests(unittest.TestCase):
    def test_invalid_ports_fall_back_to_the_declared_endpoint(self) -> None:
        for key, getter in (
            ("zai_base_url", config.get_zai_base_url),
            ("meta_base_url", config.get_meta_base_url),
        ):
            for endpoint in ("https://example.com:bad/v1", "https://example.com:65536/v1"):
                with self.subTest(key=key, endpoint=endpoint):
                    with temporary_config({key: endpoint}):
                        self.assertEqual(getter(), config._DEFAULT_CONFIG[key])

    def test_valid_https_port_is_retained(self) -> None:
        for key, getter in (
            ("zai_base_url", config.get_zai_base_url),
            ("meta_base_url", config.get_meta_base_url),
        ):
            endpoint = "https://example.com:8443/v1"
            with self.subTest(key=key), temporary_config({key: endpoint}):
                self.assertEqual(getter(), endpoint)
