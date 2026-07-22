from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import unittest
from unittest import mock

from Cozter.agent_tools.builtin.web_fetch import (
    WebFetchTool,
    _NonPublicAddressError,
    _PublicResolver,
    _validate_public_url,
)


class _StaticResolver:
    def __init__(self, answers: list[dict[str, object]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, int, object]] = []
        self.closed = False

    async def resolve(
        self, host: str, port: int = 0, family: object = None,
    ) -> list[dict[str, object]]:
        self.calls.append((host, port, family))
        return self.answers

    async def close(self) -> None:
        self.closed = True


class _Content:
    def __init__(self, body: bytes) -> None:
        self.body = body

    async def read(self, limit: int) -> bytes:
        chunk, self.body = self.body[:limit], self.body[limit:]
        return chunk


class _Response:
    charset = "utf-8"

    def __init__(
        self,
        *,
        status: int,
        url: str,
        headers: dict[str, str],
        body: bytes = b"",
    ) -> None:
        self.status = status
        self.url = url
        self.headers = headers
        self.content = _Content(body)

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        return None


class _ResponseSession:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        self.requests.append((url, kwargs))
        return self.responses.pop(0)


class WebFetchSecurityTests(unittest.TestCase):
    def test_rejects_credentials_and_nonpublic_numeric_hosts(self) -> None:
        blocked_urls = (
            "http://127.0.0.1/",  # loopback
            "http://10.0.0.1/",  # private
            "http://169.254.1.1/",  # link-local
            "http://240.0.0.1/",  # reserved
            "http://224.0.0.1/",  # multicast
            "http://[::1]/",  # IPv6 loopback
            "http://2130706433/",  # legacy numeric loopback spelling
            "http://127。0。0。1/",  # IDNA-normalized loopback spelling
        )
        for url in blocked_urls:
            with self.subTest(url=url):
                self.assertEqual(
                    _validate_public_url(url),
                    "Error: URL host must be a publicly routable address",
                )

        self.assertEqual(
            _validate_public_url("https://user:secret@example.com/"),
            "Error: URL credentials are not allowed",
        )
        self.assertEqual(
            _validate_public_url("https:///missing-host"),
            "Error: invalid URL host",
        )
        self.assertIsNone(_validate_public_url("https://example.com/"))

    def test_dns_results_must_all_be_public(self) -> None:
        async def run() -> None:
            delegate = _StaticResolver([
                {"host": "8.8.8.8"},
                {"host": "10.0.0.1"},
            ])
            with mock.patch(
                "Cozter.agent_tools.builtin.web_fetch.DefaultResolver",
                return_value=delegate,
            ):
                resolver = _PublicResolver()

            with self.assertRaises(_NonPublicAddressError):
                await resolver.resolve("mixed.example", 443)
            self.assertEqual(delegate.calls[0][:2], ("mixed.example", 443))

            await resolver.close()
            self.assertTrue(delegate.closed)

        asyncio.run(run())

    def test_public_dns_answer_is_allowed(self) -> None:
        async def run() -> None:
            delegate = _StaticResolver([{"host": "8.8.8.8"}])
            with mock.patch(
                "Cozter.agent_tools.builtin.web_fetch.DefaultResolver",
                return_value=delegate,
            ):
                resolver = _PublicResolver()

            self.assertEqual(
                await resolver.resolve("public.example", 443),
                delegate.answers,
            )
            await resolver.close()
            self.assertTrue(delegate.closed)

        asyncio.run(run())

    def test_redirect_to_private_target_is_not_requested(self) -> None:
        async def run() -> str:
            session = _ResponseSession([
                _Response(
                    status=302,
                    url="https://public.example/start",
                    headers={"location": "http://127.0.0.1/admin"},
                ),
            ])

            @asynccontextmanager
            async def fake_session():
                yield session

            with mock.patch(
                "Cozter.agent_tools.builtin.web_fetch."
                "_open_public_http_session",
                fake_session,
            ):
                result = await WebFetchTool().run(
                    "",
                    {"url": "https://public.example/start"},
                )

            self.assertEqual(
                session.requests,
                [
                    (
                        "https://public.example/start",
                        mock.ANY,
                    ),
                ],
            )
            self.assertFalse(session.requests[0][1]["allow_redirects"])
            return result

        self.assertEqual(
            asyncio.run(run()),
            "Error: URL host must be a publicly routable address",
        )
