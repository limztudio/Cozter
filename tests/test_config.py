"""Endpoint getters reject malformed overrides while preserving valid ports."""

import unittest

from Cozter import config
from Cozter.tests.helpers import temporary_config


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
