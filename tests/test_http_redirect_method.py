"""Report the method of the response after applying redirect semantics."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import unittest
from unittest import mock

from Cozter.agent_tools.plugins import http_request
from Cozter.tests.test_agent_plugins import FakeResponse, FakeSession


class HttpRedirectMethodTests(unittest.TestCase):
    def test_result_reports_the_final_request_method(self) -> None:
        for status, method, final_method in (
            (301, "POST", "GET"), (302, "PATCH", "GET"),
            (303, "DELETE", "GET"), (307, "POST", "POST"),
            (308, "PUT", "PUT"),
        ):
            with self.subTest(status=status, method=method):
                session = FakeSession([
                    FakeResponse(
                        status=status, url="https://api.example.com/start",
                        headers={"Location": "https://api.example.com/final"},
                    ),
                    FakeResponse(
                        status=200, url="https://api.example.com/final",
                        headers={"Content-Type": "text/plain"}, body=b"final response",
                    ),
                ])

                @asynccontextmanager
                async def fake_session():
                    yield session

                with mock.patch.object(http_request, "open_public_http_session", fake_session):
                    result = asyncio.run(http_request.HttpRequestTool().run(".", {
                        "url": "https://api.example.com/start", "method": method,
                        "body": "request payload",
                    }))
                self.assertEqual(session.calls[-1]["method"], final_method)
                self.assertIn(f"\n{final_method} https://api.example.com/final\n", result)
                self.assertIn("final response", result)

    def test_binary_redirect_result_also_reports_the_final_method(self) -> None:
        session = FakeSession([
            FakeResponse(status=303, url="https://api.example.com/start",
                          headers={"Location": "/image"}),
            FakeResponse(status=200, url="https://api.example.com/image",
                          headers={"Content-Type": "image/png"}, body=b"binary"),
        ])

        @asynccontextmanager
        async def fake_session():
            yield session

        with mock.patch.object(http_request, "open_public_http_session", fake_session):
            result = asyncio.run(http_request.HttpRequestTool().run(".", {
                "url": "https://api.example.com/start", "method": "POST", "body": "payload",
            }))
        self.assertIn("\nGET https://api.example.com/image\n", result)
        self.assertIn("binary response body", result)


if __name__ == "__main__":
    unittest.main()
