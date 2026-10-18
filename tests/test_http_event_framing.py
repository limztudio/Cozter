"""Allowed HTTP replies survive local JSON framing without widening CLI caps."""

import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent, utils
from Cozter.backends_agent import _http_process as hp
from Cozter.backends_agent import _openai_agent as oa
from Cozter.tests.helpers import OfflineHttpBackend


class HttpEventFramingTests(unittest.IsolatedAsyncioTestCase):
    async def _roundtrip(self, event):
        process = hp.HttpAgentProcess("offline framing fixture")
        process.emit(event)
        process.stdout.feed_eof()
        events = [value async for value in utils.iter_json_events(process.stdout)]
        self.assertEqual(events, [event])

    async def test_three_mib_unicode_reply_is_encoded_as_utf8_and_preserved(self):
        text = "한" * (1024 * 1024)
        self.assertEqual(len(text.encode("utf-8")), 3 * 1024 * 1024)
        process = hp.HttpAgentProcess("offline Unicode fixture")
        event = {"type": "assistant_text", "text": text}
        process.emit(event)
        encoded = await process.stdout.read(len(process.stdout._buffer))
        self.assertLess(len(encoded), utils._MAX_STREAM_LINE_BYTES)
        self.assertNotIn(b"\\ud55c", encoded)
        self.assertEqual(json.loads(encoded), event)

    async def test_exact_allowed_ascii_completion_survives_envelope_overhead(self):
        await self._roundtrip({"type": "assistant_text", "text": "x" * oa._MAX_COMPLETION_TEXT_BYTES})

    async def test_exact_allowed_control_completion_survives_sixfold_escaping(self):
        text = "\x00" * oa._MAX_COMPLETION_TEXT_BYTES
        self.assertLess(len(text) * 6 + 128, utils.HTTP_EVENT_LINE_MAX_BYTES)
        await self._roundtrip({"type": "assistant_text", "text": text})

    async def test_lone_surrogates_use_json_escape_fallback_without_replacement(self):
        text = "prefix " + "\ud800" * (1024 * 1024) + " suffix"
        await self._roundtrip({"type": "assistant_text", "text": text})

    async def test_worst_case_tool_argument_event_is_preserved(self):
        await self._roundtrip({
            "type": "tool_use", "name": "write_file",
            "input": {"path": "output.txt", "content": "\x01" * oa._MAX_TOOL_ARGUMENT_BYTES},
            "file_action": "write",
        })

    async def test_completed_driver_drains_large_final_event_after_prior_events(self):
        process = hp.HttpAgentProcess("offline completed driver")
        text = "\x00" * oa._MAX_COMPLETION_TEXT_BYTES

        async def produce():
            process.emit({"type": "assistant_text", "text": "first"})
            process.emit({"type": "assistant_text", "text": text})

        process.start(produce())
        await process.wait()
        self.assertEqual(process.returncode, 0)
        with mock.patch.object(utils, "terminate_process_group_async", new_callable=mock.AsyncMock) as terminate:
            events = [event async for event in utils.iter_process_json_events(process)]
        self.assertEqual([event["text"] for event in events], ["first", text])
        terminate.assert_not_awaited()

    async def test_model_driver_retains_full_reply_instead_of_latest_chunk(self):
        text = "🙂" * (oa._MAX_COMPLETION_TEXT_BYTES // 4)
        with (
            mock.patch.object(oa, "current_run", return_value=None),
            mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock,
                              return_value=(text, "", [])),
        ):
            result, restarting = await agent._drive_backend(
                OfflineHttpBackend(), "/workspace", "offline fixture", "offline-model", "full", effort=0,
            )
        self.assertFalse(restarting)
        self.assertIsNone(result.error)
        self.assertEqual(result.text, text)
        self.assertEqual([event.content for event in result.events if event.kind == "text"], [text])

    async def test_oversized_emitter_event_fails_loudly_before_feeding_payload(self):
        process = hp.HttpAgentProcess("offline oversized event")

        async def produce():
            process.emit({"type": "assistant_text", "text": "x" * 200})

        with (
            mock.patch.object(hp, "HTTP_EVENT_LINE_MAX_BYTES", 128),
            self.assertLogs(hp.logger, level="WARNING"),
        ):
            process.start(produce())
            await process.wait()
        events = [event async for event in utils.iter_process_json_events(process)]
        self.assertEqual(process.returncode, 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")
        self.assertIn("transport limit", events[0]["message"])

    async def test_untrusted_native_stream_keeps_original_line_limit(self):
        reader = asyncio.StreamReader()
        reader.feed_data(json.dumps({"text": "x" * 100}).encode() + b"\n" + b'{"text":"kept"}\n')
        reader.feed_eof()
        with mock.patch.object(utils, "_MAX_STREAM_LINE_BYTES", 64), self.assertLogs(utils.logger, level="WARNING"):
            events = [event async for event in utils.iter_json_events(reader)]
        self.assertEqual(events, [{"text": "kept"}])

    async def test_nonidentity_marker_cannot_widen_native_streams(self):
        for marker in (True, utils.HTTP_EVENT_LINE_MAX_BYTES, "http", object()):
            with self.subTest(marker=marker):
                reader = asyncio.StreamReader()
                setattr(reader, "_cozter_http_event_stream_token", marker)
                reader.feed_data(json.dumps({"text": "x" * 100}).encode() + b"\n")
                reader.feed_eof()
                with mock.patch.object(utils, "_MAX_STREAM_LINE_BYTES", 64), self.assertLogs(utils.logger, level="WARNING"):
                    events = [event async for event in utils.iter_json_events(reader)]
                self.assertEqual(events, [])

    async def test_native_post_exit_budget_remains_enforced(self):
        reader = asyncio.StreamReader()
        reader.feed_data(b'{"text":"native"}\n' * 30)
        reader.feed_eof()
        process = SimpleNamespace(stdout=reader, returncode=0, pid=0)
        with (
            mock.patch.object(utils, "_POST_EXIT_STREAM_DRAIN_BYTES", 64),
            mock.patch.object(utils, "terminate_process_group_async", new_callable=mock.AsyncMock) as terminate,
            self.assertLogs(utils.logger, level="WARNING"),
        ):
            events = [event async for event in utils.iter_process_json_events(process)]
        self.assertTrue(events)
        self.assertLess(len(events), 30)
        terminate.assert_awaited_once_with(process)
