import asyncio
import contextlib
import json
import unittest
from unittest import mock
from typing import Any

from Cozter.backends_bot.base import MessageHandle
from Cozter.backends_bot.signal import (
    SignalCliError,
    SignalBot,
    _attachment_id,
    _extract_account_id,
    _extract_sender_id,
    _extract_timestamp_from_value,
    _md_to_signal_body_and_spans,
    _normalize_group_id,
    _signal_rich_text_chunks,
)
from Cozter.backends_bot.telegram import _telegram_message_id


SIGNAL_GROUP_URL = "https://signal.group/#test"


class SignalFormattingTests(unittest.TestCase):
    def test_markdown_becomes_signal_body_and_styles(self) -> None:
        body, spans = _md_to_signal_body_and_spans(
            "# Title\nA **bold** _it_ ~~gone~~ `code` ||secret||"
        )

        self.assertEqual(body, "Title\nA bold it gone code secret")
        self.assertEqual(
            spans,
            [
                (0, 5, "BOLD"),
                (8, 4, "BOLD"),
                (13, 2, "ITALIC"),
                (16, 4, "STRIKETHROUGH"),
                (21, 4, "MONOSPACE"),
                (26, 6, "SPOILER"),
            ],
        )

    def test_rich_chunks_use_utf16_offsets(self) -> None:
        chunks = _signal_rich_text_chunks("😀 **ok**")

        self.assertEqual(chunks, [("😀 ok", ["3:2:BOLD"])])

    def test_code_block_is_monospace_without_fences(self) -> None:
        chunks = _signal_rich_text_chunks("a\n```\nx*y\n```\nb")

        self.assertEqual(chunks, [("a\nx*y\nb", ["2:3:MONOSPACE"])])

    def test_longer_fence_preserves_short_backtick_content(self):
        self.assertEqual(
            _signal_rich_text_chunks("````python\n```literal\n**plain**\n````"),
            [("```literal\n**plain**", ["0:20:MONOSPACE"])],
        )

    def test_tilde_fenced_code_keeps_markdown_literal(self):
        self.assertEqual(
            _signal_rich_text_chunks("~~~python\n**plain**\n~~~"),
            [("**plain**", ["0:9:MONOSPACE"])],
        )

    def test_styles_are_clipped_across_chunks(self) -> None:
        chunks = _signal_rich_text_chunks("**abcdef**", limit=3)

        self.assertEqual(
            chunks,
            [
                ("abc", ["0:3:BOLD"]),
                ("def", ["0:3:BOLD"]),
            ],
        )

    def test_rich_chunks_preserve_newline_at_size_boundary(self) -> None:
        text = "a" * 3_999 + "\n" + "b"

        chunks = _signal_rich_text_chunks(text, limit=4_000)

        self.assertEqual("".join(body for body, _styles in chunks), text)
        self.assertEqual([body for body, _styles in chunks], [
            "a" * 3_999 + "\n", "b",
        ])


class _CapturingSignalBot(SignalBot):
    def __init__(self) -> None:
        super().__init__([SIGNAL_GROUP_URL], jsonrpc_socket="/tmp/s")
        self._group_ids = {"group"}
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def _rpc_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: int | float | None = 60,
    ) -> Any:
        self.calls.append((method, params))
        return {"timestamp": "123"}


class _ResolvingSignalBot(SignalBot):
    def __init__(self, responses: list[Any]) -> None:
        super().__init__([SIGNAL_GROUP_URL], jsonrpc_socket="/tmp/s")
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def _rpc_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: int | float | None = 60,
    ) -> Any:
        self.calls.append((method, params))
        if not self.responses:
            raise AssertionError(f"unexpected Signal RPC call: {method}")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class SignalSendFormattingTests(unittest.TestCase):
    def test_rich_send_uses_signal_text_styles_shape(self) -> None:
        async def run() -> _CapturingSignalBot:
            bot = _CapturingSignalBot()
            await bot.send_text(
                "group", "A **bold** and `code`", rich=True,
            )
            return bot

        bot = asyncio.run(run())

        self.assertEqual(len(bot.calls), 1)
        self.assertEqual(
            bot.calls[0],
            (
                "send",
                {
                    "groupId": "group",
                    "message": "A bold and code",
                    "textStyle": ["2:4:BOLD", "11:4:MONOSPACE"],
                },
            ),
        )

    def test_single_rich_span_uses_singular_param(self) -> None:
        async def run() -> _CapturingSignalBot:
            bot = _CapturingSignalBot()
            await bot.send_text("group", "A **bold**", rich=True)
            return bot

        bot = asyncio.run(run())

        self.assertEqual(
            bot.calls[0][1],
            {
                "groupId": "group",
                "message": "A bold",
                "textStyle": "2:4:BOLD",
            },
        )

    def test_rich_edit_uses_signal_text_style_shape(self) -> None:
        async def run() -> _CapturingSignalBot:
            bot = _CapturingSignalBot()
            await bot.edit_text(
                MessageHandle(chat_id="group", message_id="42"),
                "Thinking...\n\n**partial** `reply`",
                rich=True,
            )
            return bot

        bot = asyncio.run(run())

        self.assertEqual(
            bot.calls[0][1],
            {
                "groupId": "group",
                "editTimestamp": 42,
                "message": "Thinking...\n\npartial reply",
                "textStyle": ["13:7:BOLD", "21:5:MONOSPACE"],
            },
        )


class SignalGroupResolutionTests(unittest.TestCase):
    def test_known_non_member_group_is_joined_then_rechecked(self) -> None:
        async def run() -> _ResolvingSignalBot:
            bot = _ResolvingSignalBot([
                [
                    {
                        "id": "group",
                        "isMember": False,
                        "groupInviteLink": SIGNAL_GROUP_URL,
                    },
                ],
                {"groupId": "group"},
                [
                    {
                        "id": "group",
                        "isMember": True,
                        "groupInviteLink": SIGNAL_GROUP_URL,
                    },
                ],
            ])
            self.assertEqual(
                await bot._resolve_group_id(SIGNAL_GROUP_URL), "group"
            )
            return bot

        bot = asyncio.run(run())

        self.assertEqual(
            bot.calls,
            [
                ("listGroups", None),
                ("joinGroup", {"uri": SIGNAL_GROUP_URL}),
                ("listGroups", None),
            ],
        )

    def test_known_member_group_does_not_join_again(self) -> None:
        async def run() -> _ResolvingSignalBot:
            bot = _ResolvingSignalBot([
                [
                    {
                        "id": "group",
                        "isMember": True,
                        "groupInviteLink": SIGNAL_GROUP_URL,
                    },
                ],
            ])
            self.assertEqual(
                await bot._resolve_group_id(SIGNAL_GROUP_URL), "group"
            )
            return bot

        bot = asyncio.run(run())

        self.assertEqual(bot.calls, [("listGroups", None)])

    def test_group_still_pending_after_join_is_rejected(self) -> None:
        async def run() -> None:
            bot = _ResolvingSignalBot([
                [
                    {
                        "id": "group",
                        "isMember": False,
                        "groupInviteLink": SIGNAL_GROUP_URL,
                    },
                ],
                {"groupId": "group"},
                [
                    {
                        "id": "group",
                        "isMember": False,
                        "groupInviteLink": SIGNAL_GROUP_URL,
                    },
                ],
            ])
            with self.assertRaisesRegex(RuntimeError, "not a member"):
                await bot._resolve_group_id(SIGNAL_GROUP_URL)

        asyncio.run(run())

    def test_already_joined_response_is_rechecked(self) -> None:
        async def run() -> _ResolvingSignalBot:
            bot = _ResolvingSignalBot([
                [],
                SignalCliError("already a member"),
                [
                    {
                        "id": "group",
                        "isMember": True,
                        "groupInviteLink": SIGNAL_GROUP_URL,
                    },
                ],
            ])
            self.assertEqual(
                await bot._resolve_group_id(SIGNAL_GROUP_URL), "group"
            )
            return bot

        bot = asyncio.run(run())

        self.assertEqual(
            bot.calls,
            [
                ("listGroups", None),
                ("joinGroup", {"uri": SIGNAL_GROUP_URL}),
                ("listGroups", None),
            ],
        )


class SignalReceiveStartupTests(unittest.TestCase):
    def test_receive_start_recovers_when_socket_eof_precedes_state_mark(self) -> None:
        """An EOF just after subscribe must not leave startup disconnected."""
        async def run() -> tuple[SignalBot, mock.AsyncMock]:
            bot = SignalBot([SIGNAL_GROUP_URL], jsonrpc_socket="/tmp/s")

            async def reconnect(*, resubscribe: bool) -> None:
                self.assertTrue(resubscribe)
                bot._receive_subscription = 42
                bot._receive_subscribed = True

            reconnect_mock = mock.AsyncMock(side_effect=reconnect)

            # Simulate reader clearing transport mid-subscribe, before receive goes active.
            bot._subscribe_receive = mock.AsyncMock(return_value=41)  # type: ignore[method-assign]
            bot._jsonrpc_connected = mock.Mock(return_value=False)  # type: ignore[method-assign]
            bot._connect_jsonrpc = reconnect_mock  # type: ignore[method-assign]

            await bot._start_receive_subscription()
            return bot, reconnect_mock

        bot, reconnect_mock = asyncio.run(run())

        reconnect_mock.assert_awaited_once_with(resubscribe=True)
        self.assertTrue(bot._receive_started)
        self.assertTrue(bot._receive_subscribed)
        self.assertEqual(bot._receive_subscription, 42)


class SignalJsonRpcRequestTests(unittest.TestCase):
    def test_cancelled_request_is_removed_from_pending_map(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.written = asyncio.Event()

            def write(self, _data: bytes) -> None:
                self.written.set()

            async def drain(self) -> None:
                return None

        async def run() -> None:
            bot = SignalBot([SIGNAL_GROUP_URL], jsonrpc_socket="/tmp/s")
            writer = Writer()
            bot._jsonrpc_writer = writer  # type: ignore[assignment]
            bot._jsonrpc_connected = lambda: True  # type: ignore[method-assign]

            request = asyncio.create_task(bot._rpc_request_once("test"))
            await writer.written.wait()
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
            self.assertEqual(bot._jsonrpc_pending, {})

        asyncio.run(run())


class SignalJsonRpcTransportTests(unittest.TestCase):
    def test_tcp_requests_share_a_real_line_delimited_connection(self) -> None:
        """Exercise framing and responses larger than asyncio's default limit."""
        async def run() -> None:
            requests: list[dict[str, Any]] = []
            clients: set[asyncio.Task] = set()
            large_result = "x" * 70_000

            async def serve(
                reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
            ) -> None:
                task = asyncio.current_task()
                assert task is not None
                clients.add(task)
                try:
                    while line := await reader.readline():
                        request = json.loads(line)
                        requests.append(request)
                        response = {
                            "jsonrpc": "2.0",
                            "id": request["id"],
                            "result": large_result,
                        }
                        writer.write((json.dumps(response) + "\n").encode())
                        await writer.drain()
                finally:
                    writer.close()
                    with contextlib.suppress(ConnectionError):
                        await writer.wait_closed()
                    clients.discard(task)

            server = await asyncio.start_server(serve, "127.0.0.1", 0)
            assert server.sockets is not None
            port = server.sockets[0].getsockname()[1]
            bot = SignalBot(
                [SIGNAL_GROUP_URL], jsonrpc_socket=f"tcp://127.0.0.1:{port}",
            )
            try:
                self.assertEqual(
                    await bot._rpc_request("listGroups", timeout=5),
                    large_result,
                )
                self.assertEqual(
                    await bot._rpc_request(
                        "getGroup", {"groupId": "example"}, timeout=5,
                    ),
                    large_result,
                )
                self.assertEqual(len(clients), 1)
            finally:
                bot._stop_requested.set()
                await bot._stop_jsonrpc()
                server.close()
                await server.wait_closed()
                if clients:
                    await asyncio.gather(*clients)
            self.assertEqual(
                requests,
                [
                    {"jsonrpc": "2.0", "method": "listGroups", "id": "1"},
                    {
                        "jsonrpc": "2.0", "method": "getGroup", "id": "2",
                        "params": {"groupId": "example"},
                    },
                ],
            )

        asyncio.run(run())

    def test_invalid_tcp_urls_fail_before_attempting_a_connection(self) -> None:
        for endpoint in (
            "http://127.0.0.1:7583", "tcp:127.0.0.1:7584",
            "tcp://127.0.0.1", "tcp://:7584", "tcp://host:0",
            "tcp://host:65536", "tcp://host:bad", "tcp://host:-1",
            "tcp://user:secret@host:7584", "tcp://host:7584/",
            "tcp://host:7584?account=1", "tcp://host:7584#fragment",
            "tcp://ho\nst:7584", "tcp://[::1:7584",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(ValueError, "signal_jsonrpc_socket"):
                    SignalBot([SIGNAL_GROUP_URL], jsonrpc_socket=endpoint)

    def test_unix_socket_path_still_uses_unix_connector(self) -> None:
        async def run() -> None:
            bot = SignalBot(
                [SIGNAL_GROUP_URL], jsonrpc_socket="/run/signal-cli/socket",
            )
            reader, writer = mock.Mock(), mock.Mock()
            with mock.patch(
                "Cozter.backends_bot.signal.asyncio.open_unix_connection",
                new=mock.AsyncMock(return_value=(reader, writer)),
                create=True,
            ) as connect:
                self.assertEqual(
                    await bot._open_jsonrpc_socket(), (reader, writer),
                )
                self.assertEqual(connect.await_args.args, ("/run/signal-cli/socket",))

        asyncio.run(run())

    def test_unavailable_unix_connector_explains_tcp_configuration(self) -> None:
        async def run() -> None:
            bot = SignalBot([SIGNAL_GROUP_URL], jsonrpc_socket="/tmp/s")
            with mock.patch(
                "Cozter.backends_bot.signal.asyncio.open_unix_connection",
                new=mock.AsyncMock(side_effect=NotImplementedError),
                create=True,
            ):
                with self.assertRaisesRegex(SignalCliError, "tcp://HOST:PORT"):
                    await bot._open_jsonrpc_socket()

        asyncio.run(run())


class SignalIdCoercionTests(unittest.TestCase):
    def test_timestamp_rejects_bool_and_junk(self) -> None:
        self.assertIsNone(_extract_timestamp_from_value({"timestamp": True}))
        self.assertIsNone(_extract_timestamp_from_value({"timestamp": False}))
        self.assertIsNone(_extract_timestamp_from_value({"timestamp": None}))
        self.assertIsNone(_extract_timestamp_from_value({"timestamp": {"x": 1}}))
        self.assertIsNone(_extract_timestamp_from_value({"timestamp": "  "}))
        self.assertEqual(_extract_timestamp_from_value({"timestamp": 0}), "0")
        self.assertEqual(_extract_timestamp_from_value({"timestamp": 123}), "123")
        self.assertEqual(
            _extract_timestamp_from_value({"timestamp": " 42 "}), "42",
        )
        self.assertIsNone(_extract_timestamp_from_value({"timestamp": -1}))
        self.assertIsNone(_extract_timestamp_from_value({"timestamp": "nan"}))
        self.assertIsNone(_extract_timestamp_from_value({"timestamp": "inf"}))
        self.assertIsNone(_extract_timestamp_from_value({"timestamp": "abc"}))
        self.assertEqual(
            _extract_timestamp_from_value({"timestamp": 42.0}), "42",
        )

    def test_sender_and_account_reject_bool_and_junk(self) -> None:
        self.assertEqual(_extract_sender_id({"source": True}), "")
        self.assertEqual(_extract_sender_id({"source": {"n": 1}}), "")
        self.assertEqual(_extract_sender_id({"source": 7}), "7")
        self.assertEqual(_extract_sender_id({"source": " +123 "}), "+123")
        self.assertEqual(
            _extract_sender_id({"sourceAddress": {"number": True}}), "",
        )
        self.assertEqual(_extract_account_id({"account": False}), "")
        self.assertEqual(_extract_account_id({"account": 9}), "9")
        self.assertEqual(_extract_account_id({"account": {"a": 1}}), "")

    def test_group_id_and_attachment_id_reject_bool_lists(self) -> None:
        self.assertEqual(_normalize_group_id([True, False]), "")
        self.assertEqual(_normalize_group_id([]), "")
        self.assertEqual(_attachment_id({"id": True}), "")
        self.assertEqual(_attachment_id({"id": 5}), "5")

    def test_telegram_message_id_rejects_junk(self) -> None:
        self.assertEqual(
            _telegram_message_id(
                MessageHandle(chat_id="c", message_id="42"),
            ),
            42,
        )
        for bad in ("abc", "", "  ", True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                    _telegram_message_id(
                        MessageHandle(chat_id="c", message_id=bad),  # type: ignore[arg-type]
                    )


if __name__ == "__main__":
    unittest.main()
