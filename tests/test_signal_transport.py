"""Windows Signal transport selection without any live daemon or provider."""

import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter.backends_bot import signal as signal_module
from Cozter.backends_bot.base import PlatformNotReadyError
from Cozter.backends_bot.signal import (
    SignalAccountError, SignalBot, SignalCliError,
)


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
            with self.assertRaisesRegex(PlatformNotReadyError, "not ready") as raised:
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


class SignalAccountStartupTests(unittest.IsolatedAsyncioTestCase):
    def _bot(self, *, account=""):
        return SignalBot(
            [SIGNAL_GROUP_URL], jsonrpc_socket="tcp://127.0.0.1:7584", account=account,
        )

    async def test_account_is_applied_to_scoped_rpcs_without_mutating_params(self):
        bot = self._bot(account="+12025550123")
        requests = []

        class Writer:
            def write(self, data):
                request = json.loads(data)
                requests.append(request)
                bot._jsonrpc_pending.pop(request["id"]).set_result({"result": []})

            async def drain(self):
                pass

        bot._jsonrpc_writer = Writer()
        bot._jsonrpc_connected = lambda: True
        for method in (
            "listGroups", "joinGroup", "send", "sendReaction", "remoteDelete",
            "getAttachment", "subscribeReceive",
        ):
            with self.subTest(method=method):
                params = {"groupId": "group", "account": "wrong-account"}
                await bot._rpc_request_once(method, params)
                self.assertEqual(requests[-1]["params"], {
                    "groupId": "group", "account": "+12025550123",
                })
                self.assertEqual(params, {"groupId": "group", "account": "wrong-account"})
        await bot._rpc_request_once("listGroups")
        self.assertEqual(requests[-1]["params"], {"account": "+12025550123"})
        await bot._rpc_request_once("listAccounts")
        self.assertNotIn("params", requests[-1])
        await bot._rpc_request_once("unsubscribeReceive", {"subscription": 7})
        self.assertEqual(requests[-1]["params"], {"subscription": 7})
        self.assertEqual(bot._jsonrpc_pending, {})

    async def test_account_required_response_has_a_typed_error(self):
        bot = self._bot()

        class Writer:
            def write(self, data):
                request = json.loads(data)
                bot._jsonrpc_pending.pop(request["id"]).set_result({
                    "error": {"message": "Method requires valid account parameter"},
                })

            async def drain(self):
                pass

        bot._jsonrpc_writer = Writer()
        bot._jsonrpc_connected = lambda: True
        with self.assertRaises(SignalAccountError):
            await bot._rpc_request_once("listGroups")

    async def test_healthy_single_account_daemon_does_not_discover_accounts(self):
        bot = self._bot(account="+12025550123")
        bot._rpc_request = mock.AsyncMock(return_value=[])
        self.assertEqual(await bot._list_groups(), [])
        bot._rpc_request.assert_awaited_once_with("listGroups")

    async def test_multi_account_mode_selects_the_only_number_or_numberless_aci(self):
        for row, expected in (
            ({"number": "+12025550123", "aci": "unneeded"}, "+12025550123"),
            ({"number": None, "aci": "11111111-1111-4111-8111-111111111111"},
             "11111111-1111-4111-8111-111111111111"),
        ):
            with self.subTest(row=row):
                bot = self._bot()
                bot._rpc_request = mock.AsyncMock(side_effect=[
                    SignalAccountError("Method requires valid account parameter"),
                    [row], [{"id": "group"}],
                ])
                self.assertEqual(await bot._list_groups(), [{"id": "group"}])
                self.assertEqual(bot.account, expected)
                self.assertEqual(bot._rpc_request.await_args_list, [
                    mock.call("listGroups"), mock.call("listAccounts"), mock.call("listGroups"),
                ])

    async def test_empty_loaded_accounts_is_retryable_for_default_and_explicit_selection(self):
        for account in ("", "+12025550123"):
            with self.subTest(account=account):
                bot = self._bot(account=account)
                cause = SignalAccountError("Method requires valid account parameter")
                bot._rpc_request = mock.AsyncMock(side_effect=[cause, []])
                with self.assertRaisesRegex(PlatformNotReadyError, "no loaded accounts") as raised:
                    await bot._resolve_group_ids()
                self.assertIs(raised.exception.__cause__, cause)
                self.assertEqual(bot._rpc_request.await_count, 2)

    async def test_ambiguous_or_wrong_explicit_account_does_not_choose_another(self):
        for account, rows, expected in (
            ("", [{"number": "+12025550123"}, {"number": "+12025550456"}], "multiple"),
            ("+12025550789", [{"number": "+12025550123"}], "configured signal_account"),
        ):
            with self.subTest(account=account):
                bot = self._bot(account=account)
                bot._rpc_request = mock.AsyncMock(side_effect=[
                    SignalAccountError("Method requires valid account parameter"), rows,
                ])
                with self.assertRaisesRegex(SignalAccountError, expected):
                    await bot._list_groups()
                self.assertEqual(bot.account, account)
                self.assertEqual(bot._rpc_request.await_count, 2)

    async def test_group_startup_lists_known_groups_only_once(self):
        other = SIGNAL_GROUP_URL + "-second"
        bot = SignalBot([SIGNAL_GROUP_URL, other], jsonrpc_socket="tcp://127.0.0.1:7584")
        bot._rpc_request = mock.AsyncMock(return_value=[
            {"id": "one", "groupInviteLink": SIGNAL_GROUP_URL},
            {"id": "two", "groupInviteLink": other},
        ])
        self.assertEqual(await bot._resolve_group_ids(), {SIGNAL_GROUP_URL: "one", other: "two"})
        bot._rpc_request.assert_awaited_once_with("listGroups")

    async def test_account_and_transport_failures_are_not_skipped_as_bad_group_urls(self):
        for failure in (
            SignalAccountError("fixture account failure"),
            PlatformNotReadyError("fixture account not loaded"),
            ConnectionResetError("fixture transport closed"),
        ):
            with self.subTest(failure=failure):
                bot = self._bot()
                bot._list_groups = mock.AsyncMock(return_value=[])
                bot._resolve_group_id = mock.AsyncMock(side_effect=failure)
                with self.assertRaises(type(failure)) as raised:
                    await bot._resolve_group_ids()
                self.assertIs(raised.exception, failure)

    async def test_genuine_group_failure_retains_original_cause(self):
        bot = self._bot()
        failure = RuntimeError("fixture group is awaiting approval")
        bot._list_groups = mock.AsyncMock(return_value=[])
        bot._resolve_group_id = mock.AsyncMock(side_effect=failure)
        with self.assertLogs(signal_module.logger, level="WARNING"):
            with self.assertRaisesRegex(RuntimeError, "No configured Signal group") as raised:
                await bot._resolve_group_ids()
        self.assertIs(raised.exception.__cause__, failure)

    async def test_subscription_readiness_precedes_queue_restore_and_failed_start_cleans_up(self):
        for failure in (PlatformNotReadyError("fixture transport"), asyncio.CancelledError()):
            with self.subTest(failure=failure):
                bot = self._bot(account="+12025550123")
                bot._outbound_journal.load = mock.AsyncMock()
                bot._connect_jsonrpc = mock.AsyncMock()
                bot._resolve_group_ids = mock.AsyncMock(return_value={SIGNAL_GROUP_URL: "group"})
                bot._start_receive_subscription = mock.AsyncMock(side_effect=failure)
                bot._start_daemon_services = mock.AsyncMock()
                bot.stop = mock.AsyncMock()
                with self.assertRaises(type(failure)):
                    await bot.start()
                bot._start_daemon_services.assert_not_awaited()
                bot.stop.assert_awaited_once()
                self.assertTrue(bot._stop_requested.is_set())

    async def test_transient_connection_failure_cleans_up_and_retries_with_fresh_account_selection(self):
        bot = self._bot()
        bot.account = "+12025550456"
        bot._outbound_journal.load = mock.AsyncMock()
        bot._connect_jsonrpc = mock.AsyncMock(side_effect=ConnectionRefusedError("fixture boot"))
        bot._resolve_group_ids = mock.AsyncMock()
        bot.stop = mock.AsyncMock()
        with self.assertRaises(PlatformNotReadyError) as raised:
            await bot.start()
        self.assertIsInstance(raised.exception.__cause__, ConnectionRefusedError)
        self.assertEqual(bot.account, "")
        bot._resolve_group_ids.assert_not_awaited()
        bot.stop.assert_awaited_once()

    async def test_early_scoped_notification_waits_until_durable_queue_restore_finishes(self):
        bot = self._bot(account="11111111-1111-4111-8111-111111111111")
        order = []
        received = []
        bot._outbound_journal.load = mock.AsyncMock()
        bot._connect_jsonrpc = mock.AsyncMock()
        bot._resolve_group_ids = mock.AsyncMock(return_value={SIGNAL_GROUP_URL: "group"})
        bot._jsonrpc_connected = lambda: True

        async def handle(item):
            order.append("incoming")
            received.append(item)

        bot._handle_received_item = mock.AsyncMock(side_effect=handle)
        payload = {"params": {"subscription": 7, "result": {"account": "+12025550123"}}}
        notification = None

        async def subscribe():
            nonlocal notification
            notification = asyncio.create_task(bot._handle_jsonrpc_receive(payload))
            await asyncio.sleep(0)
            self.assertFalse(notification.done())
            bot._receive_subscription = 7
            bot._receive_subscribed = True

        async def restore():
            await asyncio.sleep(0)
            self.assertEqual(received, [])
            order.append("restored")

        bot._start_receive_subscription = subscribe
        bot._start_daemon_services = restore
        await bot.start()
        await asyncio.wait_for(notification, 1)
        self.assertEqual(order, ["restored", "incoming"])
        self.assertEqual(received, [{"account": "+12025550123"}])

    async def test_state_file_failures_are_fatal_instead_of_retrying_partial_queue_restoration(self):
        for phase in ("journal", "daemon"):
            with self.subTest(phase=phase):
                bot = self._bot()
                failure = PermissionError("fixture durable state cannot be written")
                bot._outbound_journal.load = mock.AsyncMock(
                    side_effect=failure if phase == "journal" else None,
                )
                bot._connect_jsonrpc = mock.AsyncMock()
                bot._resolve_group_ids = mock.AsyncMock(return_value={SIGNAL_GROUP_URL: "group"})
                bot._start_receive_subscription = mock.AsyncMock()
                bot._start_daemon_services = mock.AsyncMock(
                    side_effect=failure if phase == "daemon" else None,
                )
                bot.stop = mock.AsyncMock()
                with self.assertRaises(PermissionError) as raised:
                    await bot.start()
                self.assertIs(raised.exception, failure)
                bot.stop.assert_awaited_once()

    async def test_selected_account_ignores_automatic_and_other_subscription_notifications(self):
        bot = self._bot(account="11111111-1111-4111-8111-111111111111")
        bot._receive_subscription = 7
        bot._receive_ready.set()
        bot._handle_received_item = mock.AsyncMock()
        for params in (
            {"result": {"account": "+12025550456"}},
            {"subscription": 8, "result": {"account": "+12025550456"}},
            {"subscription": True, "result": {"account": "+12025550456"}},
        ):
            await bot._handle_jsonrpc_receive({"params": params})
        bot._handle_received_item.assert_not_awaited()
        await bot._handle_jsonrpc_receive({"params": {
            "subscription": 7, "result": {"account": "+12025550123"},
        }})
        bot._handle_received_item.assert_awaited_once_with({"account": "+12025550123"})

    async def test_unselected_single_account_mode_keeps_automatic_receive_notifications(self):
        bot = self._bot()
        bot._receive_ready.set()
        bot._handle_received_item = mock.AsyncMock()
        await bot._handle_jsonrpc_receive({"params": {"account": "+12025550123"}})
        bot._handle_received_item.assert_awaited_once_with({"account": "+12025550123"})

    async def test_reused_subscription_id_cannot_release_old_transport_notification(self):
        bot = self._bot(account="+12025550123")
        old_writer, new_writer = mock.Mock(), mock.Mock()
        bot._jsonrpc_writer = old_writer
        bot._handle_received_item = mock.AsyncMock()
        pending = asyncio.create_task(bot._handle_jsonrpc_receive({"params": {
            "subscription": 7, "result": {"account": "+12025550123"},
        }}, transport=old_writer))
        await asyncio.sleep(0)
        self.assertFalse(pending.done())
        bot._jsonrpc_writer = new_writer
        bot._receive_subscription = 7
        bot._receive_ready.set()
        await asyncio.wait_for(pending, 1)
        bot._handle_received_item.assert_not_awaited()

    async def test_failed_start_cancels_and_joins_notification_tasks_waiting_for_readiness(self):
        for failure in (PlatformNotReadyError("fixture unavailable"), asyncio.CancelledError()):
            with self.subTest(failure=failure):
                bot = self._bot(account="+12025550123")
                bot._outbound_journal.load = mock.AsyncMock()
                bot._connect_jsonrpc = mock.AsyncMock()
                bot._resolve_group_ids = mock.AsyncMock(return_value={SIGNAL_GROUP_URL: "group"})
                bot._stop_daemon_services = mock.AsyncMock()
                bot._flush_outbound_journal = mock.AsyncMock()
                bot._stop_jsonrpc = mock.AsyncMock()
                notification = None

                async def subscribe():
                    nonlocal notification
                    notification = asyncio.create_task(bot._handle_jsonrpc_receive({"params": {
                        "subscription": 7, "result": {"account": "+12025550123"},
                    }}))
                    bot._notification_tasks.add(notification)
                    notification.add_done_callback(bot._notification_tasks.discard)
                    await asyncio.sleep(0)
                    raise failure

                bot._start_receive_subscription = subscribe
                with self.assertRaises(type(failure)):
                    await bot.start()
                self.assertTrue(notification.cancelled())
                self.assertEqual(bot._notification_tasks, set())
                self.assertFalse(bot._receive_ready.is_set())
                bot._stop_daemon_services.assert_awaited_once()
                bot._stop_jsonrpc.assert_awaited_once()

    async def test_startup_cleanup_failure_is_fatal_and_preserves_both_causes(self):
        for phase in ("daemon", "journal"):
            with self.subTest(phase=phase):
                bot = self._bot()
                original = PlatformNotReadyError("fixture boot connection unavailable")
                cleanup_failure = OSError("fixture cleanup state write failed")
                bot._outbound_journal.load = mock.AsyncMock()
                bot._connect_jsonrpc = mock.AsyncMock(side_effect=original)
                bot._stop_daemon_services = mock.AsyncMock(
                    side_effect=cleanup_failure if phase == "daemon" else None,
                )
                bot._flush_outbound_journal = mock.AsyncMock(
                    side_effect=cleanup_failure if phase == "journal" else None,
                )
                bot._stop_jsonrpc = mock.AsyncMock()
                with self.assertRaises(ExceptionGroup) as raised:
                    await bot.start()
                self.assertEqual(raised.exception.exceptions, (original, cleanup_failure))
                bot._connect_jsonrpc.assert_awaited_once()
                bot._stop_daemon_services.assert_awaited_once()
                bot._flush_outbound_journal.assert_awaited_once()
                bot._stop_jsonrpc.assert_awaited_once()

    async def test_stop_attempts_every_cleanup_and_reports_all_failures(self):
        bot = self._bot()
        daemon_failure = RuntimeError("fixture watcher stop failed")
        journal_failure = PermissionError("fixture journal write failed")
        bot._receive_subscription = 7
        bot._stop_daemon_services = mock.AsyncMock(side_effect=daemon_failure)
        bot._rpc_request = mock.AsyncMock()
        bot._flush_outbound_journal = mock.AsyncMock(side_effect=journal_failure)
        bot._stop_jsonrpc = mock.AsyncMock()
        notification = asyncio.create_task(asyncio.sleep(60))
        bot._notification_tasks.add(notification)
        notification.add_done_callback(bot._notification_tasks.discard)
        with self.assertRaises(ExceptionGroup) as raised:
            await bot.stop()
        self.assertEqual(raised.exception.exceptions, (daemon_failure, journal_failure))
        bot._rpc_request.assert_awaited_once_with(
            "unsubscribeReceive", {"subscription": 7}, timeout=5,
        )
        self.assertTrue(notification.cancelled())
        self.assertFalse(bot._notification_tasks)
        bot._stop_jsonrpc.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
