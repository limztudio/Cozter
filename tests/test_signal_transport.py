"""Windows Signal transport selection without any live daemon or provider."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter.backends_bot import signal as signal_module
from Cozter.backends_bot.signal import SignalBot, SignalCliError


SIGNAL_GROUP_URL = "https://signal.group/#transport-fixture"


def _without_unix_api(*, connect, sleep=None):
    """Model Windows asyncio, which has no open_unix_connection attribute.

    Replace only the adapter's module reference so event-loop internals and
    unrelated tests continue using the actual asyncio module.
    """
    return mock.patch.object(signal_module, "asyncio", SimpleNamespace(
        get_running_loop=asyncio.get_running_loop,
        wait_for=asyncio.wait_for,
        open_connection=connect,
        sleep=sleep or mock.AsyncMock(),
    ))


class WindowsSignalTransportTests(unittest.IsolatedAsyncioTestCase):
    def test_ipv6_authority_rejects_characters_after_closing_bracket(self) -> None:
        for endpoint in (
            "tcp://[::1]junk:7584", "tcp://[::1]]:7584",
            "tcp://[::1]::7584",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(ValueError, "signal_jsonrpc_socket"):
                    SignalBot([SIGNAL_GROUP_URL], jsonrpc_socket=endpoint)

    async def test_tcp_works_when_unix_connection_api_is_absent(self) -> None:
        reader, writer = object(), object()
        connect = mock.AsyncMock(return_value=(reader, writer))
        bot = SignalBot(
            [SIGNAL_GROUP_URL], jsonrpc_socket="tcp://127.0.0.1:7583",
        )
        with _without_unix_api(connect=connect):
            self.assertFalse(hasattr(signal_module.asyncio, "open_unix_connection"))
            self.assertEqual(await bot._open_jsonrpc_socket(), (reader, writer))
        connect.assert_awaited_once_with(
            "127.0.0.1", 7583, limit=signal_module._JSONRPC_STREAM_LIMIT,
        )

    async def test_tcp_ipv6_uri_passes_unbracketed_host_to_tcp_connector(self) -> None:
        streams = (object(), object())
        connect = mock.AsyncMock(return_value=streams)
        bot = SignalBot([SIGNAL_GROUP_URL], jsonrpc_socket="tcp://[::1]:7583")
        with _without_unix_api(connect=connect):
            self.assertEqual(await bot._open_jsonrpc_socket(), streams)
        connect.assert_awaited_once_with(
            "::1", 7583, limit=signal_module._JSONRPC_STREAM_LIMIT,
        )

    async def test_missing_unix_api_gives_tcp_guidance_without_retrying(self) -> None:
        connect, sleep = mock.AsyncMock(), mock.AsyncMock()
        bot = SignalBot([SIGNAL_GROUP_URL], jsonrpc_socket="/tmp/unsupported")
        with _without_unix_api(connect=connect, sleep=sleep):
            with self.assertRaisesRegex(SignalCliError, "tcp://HOST:PORT") as raised:
                await bot._open_jsonrpc_socket()
        self.assertIsInstance(raised.exception.__cause__, AttributeError)
        connect.assert_not_awaited()
        sleep.assert_not_awaited()

    async def test_tcp_attribute_error_is_not_misreported_as_missing_unix_support(self) -> None:
        failure = AttributeError("fixture TCP connector failure")
        connect = mock.AsyncMock(side_effect=failure)
        sleep = mock.AsyncMock()
        bot = SignalBot([SIGNAL_GROUP_URL], jsonrpc_socket="tcp://127.0.0.1:7583")
        with _without_unix_api(connect=connect, sleep=sleep):
            with self.assertRaises(AttributeError) as raised:
                await bot._open_jsonrpc_socket()
        self.assertIs(raised.exception, failure)
        connect.assert_awaited_once()
        sleep.assert_not_awaited()

    async def test_unix_connector_attribute_error_is_not_misreported_as_missing_api(self) -> None:
        failure = AttributeError("fixture Unix connector failure")
        connect = mock.AsyncMock(side_effect=failure)
        bot = SignalBot([SIGNAL_GROUP_URL], jsonrpc_socket="/tmp/socket")
        with mock.patch.object(
            signal_module.asyncio, "open_unix_connection", connect, create=True,
        ):
            with self.assertRaises(AttributeError) as raised:
                await bot._open_jsonrpc_socket()
        self.assertIs(raised.exception, failure)
        connect.assert_awaited_once()

    async def test_tcp_not_implemented_is_not_misreported_as_missing_unix_support(self) -> None:
        failure = NotImplementedError("fixture TCP connector failure")
        connect = mock.AsyncMock(side_effect=failure)
        sleep = mock.AsyncMock()
        bot = SignalBot([SIGNAL_GROUP_URL], jsonrpc_socket="tcp://127.0.0.1:7584")
        with _without_unix_api(connect=connect, sleep=sleep):
            with self.assertRaises(NotImplementedError) as raised:
                await bot._open_jsonrpc_socket()
        self.assertIs(raised.exception, failure)
        connect.assert_awaited_once()
        sleep.assert_not_awaited()

    async def test_tcp_connection_retries_transient_refusal(self) -> None:
        streams = (object(), object())
        connect = mock.AsyncMock(side_effect=[ConnectionRefusedError(), streams])
        sleep = mock.AsyncMock()
        bot = SignalBot([SIGNAL_GROUP_URL], jsonrpc_socket="tcp://127.0.0.1:7584")
        with _without_unix_api(connect=connect, sleep=sleep):
            self.assertEqual(await bot._open_jsonrpc_socket(), streams)
        self.assertEqual(connect.await_count, 2)
        sleep.assert_awaited_once_with(0.25)

    async def test_tcp_connection_deadline_surfaces_transport_failure(self) -> None:
        failure = ConnectionRefusedError("fixture listener unavailable")
        connect = mock.AsyncMock(side_effect=failure)
        sleep = mock.AsyncMock()
        bot = SignalBot([SIGNAL_GROUP_URL], jsonrpc_socket="tcp://127.0.0.1:7584")
        with _without_unix_api(connect=connect, sleep=sleep):
            with self.assertRaisesRegex(SignalCliError, "not ready") as raised:
                await bot._open_jsonrpc_socket(timeout=0)
        self.assertIs(raised.exception.__cause__, failure)
        connect.assert_awaited_once()
        sleep.assert_not_awaited()

    async def test_tcp_connection_cancellation_propagates(self) -> None:
        connect = mock.AsyncMock(side_effect=asyncio.CancelledError)
        sleep = mock.AsyncMock()
        bot = SignalBot([SIGNAL_GROUP_URL], jsonrpc_socket="tcp://127.0.0.1:7584")
        with _without_unix_api(connect=connect, sleep=sleep):
            with self.assertRaises(asyncio.CancelledError):
                await bot._open_jsonrpc_socket()
        connect.assert_awaited_once()
        sleep.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
