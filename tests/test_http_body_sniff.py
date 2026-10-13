"""Optional JSON detection must accept deeply nested arbitrary text bodies."""

import asyncio
from contextlib import asynccontextmanager
import unittest
from unittest import mock

from Cozter.agent_tools.plugins import http_request
from Cozter.tests.test_agent_plugins import _FakeResponse, _FakeSession


class HttpBodySniffTests(unittest.TestCase):
    def test_nested_json_body_uses_plain_text_if_the_optional_parser_refuses_it(self) -> None:
        body = '{"nested":' + '[' * 2_000 + '0' + ']' * 2_000 + '}'
        session = _FakeSession([
            _FakeResponse(status=200, url="https://api.example.com/",
                          headers={"Content-Type": "text/plain"}, body=b"accepted"),
        ])

        @asynccontextmanager
        async def fake_session():
            yield session

        with (
            mock.patch.object(http_request, "open_public_http_session", fake_session),
            # JSON decoder stack limits differ among supported Python versions.
            mock.patch.object(http_request.json, "loads", side_effect=RecursionError),
        ):
            result = asyncio.run(http_request.HttpRequestTool().run(".", {
                "url": "https://api.example.com/", "method": "POST", "body": body,
            }))
        self.assertIn("accepted", result)
        self.assertEqual(session.calls[0]["data"], body)
        self.assertEqual(session.calls[0]["headers"]["Content-Type"], "text/plain")


if __name__ == "__main__":
    unittest.main()
