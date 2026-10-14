"""Signal send diagnostics correlate acknowledgements without storing message text."""

import asyncio
import unittest
from unittest import mock

from Cozter.backends_bot import signal


class SignalSendObservabilityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = signal.SignalBot([], jsonrpc_socket="tcp://127.0.0.1:7584")
        self.bot._group_ids = {"room", "private-room-identity"}
        self.bot._outbound_journal.remember = mock.AsyncMock()
        self.bot._paced_rpc_request = mock.AsyncMock(return_value={"timestamp": 123456789})

    async def asyncTearDown(self):
        await self.bot._flush_outbound_journal()

    async def test_acknowledgement_correlates_message_and_room_without_plaintext(self):
        room = "private-room-identity"
        text = "Private task progress with a sensitive filename"
        with self.assertLogs("Cozter.performance", "INFO") as captured:
            handle = await self.bot.send_text(room, text)
        self.assertEqual(handle.message_id, "123456789")
        entry = captured.output[0]
        self.assertIn("outcome=acknowledged", entry)
        self.assertIn("kind=text", entry)
        self.assertIn("group_hash=" + signal._echo_hash(room), entry)
        self.assertIn("text_hash=" + signal._echo_hash(text), entry)
        self.assertIn("timestamp=123456789", entry)
        self.assertNotIn(room, entry)
        self.assertNotIn(text, entry)

    async def test_rpc_cancellation_is_distinguished_from_acknowledgement(self):
        self.bot._paced_rpc_request.side_effect = asyncio.CancelledError
        with (self.assertLogs("Cozter.performance", "INFO") as captured,
              self.assertRaises(asyncio.CancelledError)):
            await self.bot.send_text("room", "progress")
        self.assertIn("outcome=cancelled phase=rpc", captured.output[0])
        self.assertNotIn("acknowledged", captured.output[0])
        self.assertEqual(self.bot._outbound_journal_tasks, set())

    async def test_pre_send_journal_failure_does_not_claim_an_attempt_or_send(self):
        self.bot._outbound_journal.remember.side_effect = OSError("private account path")
        with (self.assertLogs("Cozter.performance", "INFO") as captured,
              self.assertRaises(OSError)):
            await self.bot.send_text("room", "progress")
        self.assertIn("outcome=error phase=journal", captured.output[0])
        self.assertIn("error_type=OSError", captured.output[0])
        self.assertNotIn("private account path", captured.output[0])
        self.bot._paced_rpc_request.assert_not_awaited()

    async def test_rpc_error_is_logged_without_exception_content(self):
        self.bot._paced_rpc_request.side_effect = signal.SignalCliError("secret recipient address")
        with (self.assertLogs("Cozter.performance", "INFO") as captured,
              self.assertRaises(signal.SignalCliError)):
            await self.bot.send_text("room", "progress")
        self.assertIn("outcome=error phase=rpc", captured.output[0])
        self.assertNotIn("secret recipient address", captured.output[0])

    async def test_result_without_timestamp_is_distinguished_from_known_ack(self):
        self.bot._paced_rpc_request.return_value = {}
        with self.assertLogs("Cozter.performance", "INFO") as captured:
            self.assertIsNone(await self.bot.send_text("room", "progress"))
        self.assertIn("outcome=no_timestamp", captured.output[0])
        self.assertNotIn("outcome=acknowledged", captured.output[0])
        self.assertEqual(self.bot._outbound_journal_tasks, set())

    async def test_mixed_success_failure_ack_is_reported_without_resending_or_exposing_addresses(self):
        self.bot._paced_rpc_request.return_value = {"timestamp": 123456789, "results": [
            {"recipientAddress": "private-success-address", "type": "SUCCESS"},
            {"recipientAddress": "private-failure-address", "type": "NETWORK_FAILURE"},
            {"recipientAddress": "private-identity-address", "type": "IDENTITY_FAILURE"},
            {"recipientAddress": "private-newtype-address", "type": "PRIVATE_NEW_TYPE"},
        ]}
        with self.assertLogs("Cozter.performance", "INFO") as captured:
            handle = await self.bot.send_text("room", "progress")
        self.assertEqual(handle.message_id, "123456789")
        self.bot._paced_rpc_request.assert_awaited_once()
        self.assertIn("outcome=acknowledged_with_failures", captured.output[0])
        self.assertIn("result_successes=1 result_failures=2 result_unclassified=1", captured.output[0])
        self.assertNotIn("private-", captured.output[0])
        self.assertNotIn("PRIVATE_NEW_TYPE", captured.output[0])
