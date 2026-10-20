"""A parent delta needs matching HTTP history or a complete bootstrap."""

import copy
import hashlib
import tempfile
import unittest
from unittest import mock

from Cozter.backends_agent import _openai_agent as oa
from Cozter.tests.helpers import OfflineHttpBackend, OfflineRecord


class _MutableEndpointBackend(OfflineHttpBackend):
    supports_vision = True
    endpoint = "https://first.invalid/chat/completions"

    def _chat_endpoint(self):
        return self.endpoint


class HttpLeaseBootstrapTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.record = OfflineRecord()
        self.requests = []
        self.backend = _MutableEndpointBackend()
        self.enterContext(mock.patch.object(oa, "current_run", return_value=self.record))
        self.enterContext(mock.patch.object(oa, "_system_prompt", return_value="Current policy"))
        self.enterContext(mock.patch.object(oa, "_tools_for_approval", return_value=None))
        self.enterContext(mock.patch.object(oa, "_stream_completion", side_effect=self.stream))

    async def stream(self, _endpoint, payload, *_args, **_kwargs):
        self.assertIsNone(oa._HTTP_CONTINUATION_BOOTSTRAP.get())
        self.requests.append(copy.deepcopy(payload))
        return "Segment finished.", "", []

    async def run_segment(self, prompt, *, bootstrap=None, model="model-a", approval="full"):
        with oa.http_continuation_scope(bootstrap):
            proc = await self.backend.launch(self.ws, prompt, model, approval)
            self.assertEqual(await proc.wait(), 0)
        self.assertIsNone(oa._HTTP_CONTINUATION_BOOTSTRAP.get())
        return self.requests[-1]["messages"]

    async def test_matching_parent_history_appends_only_new_helper_result(self):
        original = "ORIGINAL-PARENT-REQUEST"
        await self.run_segment(original)
        first = await self.run_segment("HELPER-A-DELTA", bootstrap="BOOTSTRAP-WITH-ORIGINAL-AND-A")
        self.assertEqual([row["content"] for row in first if row["role"] == "user"],
                         [original, "HELPER-A-DELTA"])
        second = await self.run_segment("HELPER-B-DELTA", bootstrap="BOOTSTRAP-WITH-ORIGINAL-A-AND-B")
        self.assertEqual([row["content"] for row in second if row["role"] == "user"],
                         [original, "HELPER-A-DELTA", "HELPER-B-DELTA"])
        self.assertEqual(self.record.values["http_prompt_digest"], hashlib.sha256(b"HELPER-B-DELTA").hexdigest())

    async def test_identity_changes_bootstrap_original_scope_and_vision(self):
        for changed in ("endpoint", "model", "approval"):
            with self.subTest(changed=changed):
                self.record.values.clear()
                self.backend.endpoint = "https://first.invalid/chat/completions"
                await self.run_segment("OLD-PARENT-REQUEST")
                options = {}
                if changed == "endpoint":
                    self.backend.endpoint = "https://second.invalid/chat/completions"
                elif changed == "model":
                    options["model"] = "model-b"
                else:
                    options["approval"] = "confirm"
                bootstrap = "ORIGINAL-SCOPE + HELPER-RESULT; attachment saved to: bootstrap.png"
                content = [{"type": "text", "text": bootstrap},
                           {"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture"}}]
                with mock.patch.object(oa, "_vision_parts_for_prompt", return_value=content) as vision:
                    messages = await self.run_segment("HELPER-ONLY-DELTA", bootstrap=bootstrap, **options)
                vision.assert_called_once_with(bootstrap, self.ws)
                self.assertEqual([row["content"] for row in messages if row["role"] == "user"], [content])
                self.assertEqual(self.record.values["http_prompt_digest"],
                                 hashlib.sha256(bootstrap.encode()).hexdigest())

    async def test_empty_matching_history_still_requires_bootstrap(self):
        await self.run_segment("ORIGINAL-PARENT-REQUEST")
        self.record.values["http_messages"] = []
        messages = await self.run_segment("HELPER-DELTA", bootstrap="COMPLETE-PARENT-AND-HELPER-SCOPE")
        self.assertEqual([row["content"] for row in messages if row["role"] == "user"],
                         ["COMPLETE-PARENT-AND-HELPER-SCOPE"])


if __name__ == "__main__":
    unittest.main()
