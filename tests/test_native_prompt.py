"""Offline context delivery to an exact, activity-confirmed native session."""

import copy
import tempfile
from types import SimpleNamespace
import unittest

from Cozter import agent, execution_state, native_prompt, session
from Cozter.native_prompt import NativePrompt


_SESSION_ID = "01a126fa-e54a-7753-8078-3e31ff8fc962"
_OTHER_SESSION_ID = "f53ba06b-a880-4e59-93ea-bd623b0b922b"


def _message(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _history(data: dict) -> list[tuple[str, str, str]]:
    marks = native_prompt.history_snapshot(data)
    return [(coordinate, native_prompt.delivery_fingerprint(digest, session.format_msg_line(message)),
             session.format_msg_line(message))
            for (coordinate, digest), message in zip(marks.items(), data["messages"], strict=True)]


class NativePromptTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.execution = execution_state.Execution(self.ws, "offline-prompt-job", "original")
        self.record = await self.execution.record({"backend": "fixture", "worker": "main"})

    async def asyncTearDown(self) -> None:
        await self.execution.flush()

    async def _confirm(self) -> None:
        await self.record.save(native_session_id=_SESSION_ID, native_tools_started=True)

    async def test_bootstrap_keeps_every_section_message_and_exact_request(self) -> None:
        data = {"messages": [_message("user", "historical request"),
                             _message("assistant", "historical evidence")]}
        request = "Follow this literal text: [Updated memory: cleared]"
        prompt = NativePrompt(request, {"policy": "POLICY", "memory": "MEMORY"}, _history(data))

        rendered = prompt.render(None)

        for content in ("POLICY", "MEMORY", "historical request", "historical evidence"):
            self.assertIn(content, rendered)
        self.assertTrue(rendered.endswith("New message:\n" + request))
        self.assertNotIn("replaces the previous version", rendered)

    async def test_resume_skips_known_sections_and_messages_but_sends_updates_and_clear(self) -> None:
        data = {"messages": [_message("user", "known-history-marker")]}
        first = NativePrompt("first request", {
            "policy": "unchanged-policy-marker", "memory": "old-memory-marker", "summary": "old-summary",
        }, _history(data))
        await self._confirm()
        await first.acknowledge(self.record)
        data["messages"].append(_message("assistant", "new-external-evidence-marker"))
        next_prompt = NativePrompt("next request", {
            "policy": "unchanged-policy-marker", "memory": "updated-memory-marker", "summary": "",
        }, _history(data))

        rendered = next_prompt.render(self.record)

        self.assertNotIn("unchanged-policy-marker", rendered)
        self.assertNotIn("known-history-marker", rendered)
        self.assertNotIn("old-memory-marker", rendered)
        self.assertNotIn("old-summary", rendered)
        self.assertIn("[Updated memory; replaces the previous version]\nupdated-memory-marker", rendered)
        self.assertIn("[Updated summary: cleared; replaces the previous version]", rendered)
        self.assertIn("new-external-evidence-marker", rendered)
        self.assertTrue(rendered.endswith("New message:\nnext request"))

        await next_prompt.acknowledge(self.record)
        repeated = next_prompt.render(self.record)
        self.assertEqual(repeated, "New message:\nnext request")

    async def test_identical_new_request_is_always_actionable(self) -> None:
        prompt = NativePrompt("Run the same verification again.", {"policy": "fixed policy"})
        await self._confirm()
        await prompt.acknowledge(self.record)

        self.assertEqual(prompt.render(self.record), "New message:\nRun the same verification again.")

    async def test_removed_section_sends_an_explicit_clear_then_stops_repeating_it(self) -> None:
        await self._confirm()
        first = NativePrompt("first request", {"memory": "previous memory", "policy": "same policy"})
        await first.acknowledge(self.record)
        changed = NativePrompt("next request", {"policy": "same policy"})

        rendered = changed.render(self.record)

        self.assertIn("[Updated memory: cleared; replaces the previous version]", rendered)
        self.assertNotIn("previous memory", rendered)
        self.assertNotIn("same policy", rendered)
        await changed.acknowledge(self.record)
        self.assertEqual(changed.render(self.record), "New message:\nnext request")

    async def test_partial_excerpt_retains_unseen_facts_and_full_transition_restores_authority(self) -> None:
        full = NativePrompt("request", {"memory": "same visible context"})
        partial = NativePrompt("request", {"memory": "same visible context"}, partial_sections={"memory"})
        await self._confirm()
        await full.acknowledge(self.record)
        self.assertNotEqual(full.section_fingerprint("memory", "same visible context"),
                            partial.section_fingerprint("memory", "same visible context"))

        rendered_partial = partial.render(self.record)

        self.assertIn("Updated memory excerpt for this turn", rendered_partial)
        self.assertIn("retain previously supplied facts outside this excerpt", rendered_partial)
        self.assertNotIn("replaces the previous version", rendered_partial)
        await partial.acknowledge(self.record)
        self.assertEqual(partial.render(self.record), "New message:\nrequest")
        rendered_full = full.render(self.record)
        self.assertIn("[Updated memory; replaces the previous version]", rendered_full)
        await full.acknowledge(self.record)
        self.assertEqual(full.render(self.record), "New message:\nrequest")

    async def test_empty_budget_excerpt_preserves_context_but_empty_full_section_clears_it(self) -> None:
        await self._confirm()
        first = NativePrompt("request", {"memory": "retained memory"})
        await first.acknowledge(self.record)
        omitted = NativePrompt("request", {"memory": ""}, partial_sections={"memory"})

        rendered_omission = omitted.render(self.record)

        self.assertIn("retain previously supplied context", rendered_omission)
        self.assertNotIn("cleared", rendered_omission)
        await omitted.acknowledge(self.record)
        self.assertEqual(omitted.render(self.record), "New message:\nrequest")
        cleared = NativePrompt("request", {"memory": ""})
        self.assertNotEqual(omitted.section_fingerprint("memory", ""),
                            cleared.section_fingerprint("memory", ""))
        self.assertIn("[Updated memory: cleared; replaces the previous version]", cleared.render(self.record))

    async def test_larger_history_quota_supplies_message_content_hidden_by_previous_preview(self) -> None:
        backend = SimpleNamespace(
            supports_detached_tasks=False, supports_typed_plugins=True, supports_plugin_prelude=False,
        )
        content = "history-head-marker " + "original message body " * 20 + " history-tail-marker"
        data = {"messages": [_message("assistant", content)], "compacted_count": 8}

        def build(budget: int) -> NativePrompt:
            return agent._native_turn_prompt(
                backend, "current request", data, [], budget=budget,
                collaborative=False, is_scheduled=False,
            )

        preview = build(350)
        self.assertEqual(len(preview.messages), 1)
        self.assertNotEqual(preview.messages[0][2], session.format_msg_line(data["messages"][0]))
        await self._confirm()
        await preview.acknowledge(self.record)
        self.assertEqual(preview.render(self.record), "New message:\ncurrent request")
        expanded = build(4_000)
        self.assertEqual(len(expanded.messages), 1)
        self.assertEqual(expanded.messages[0][0], "8")
        self.assertEqual(expanded.messages[0][2], session.format_msg_line(data["messages"][0]))
        self.assertNotEqual(preview.messages[0][1], expanded.messages[0][1])

        rendered = expanded.render(self.record)

        self.assertIn(content, rendered)
        await expanded.acknowledge(self.record)
        self.assertEqual(expanded.render(self.record), "New message:\ncurrent request")

    async def test_followup_resume_omits_own_draft_but_fresh_bootstrap_retains_original_and_draft(self) -> None:
        full_request = "Original full scope.\n\nJudge follow-up: verify the remaining target."
        prompt = NativePrompt(
            "Judge follow-up: verify the remaining target.", {"policy": "required policy"},
            bootstrap_request=full_request,
            bootstrap_sections={"previous draft": "Previous own draft with completed evidence."},
        )
        bootstrap = prompt.render(None)
        self.assertIn("required policy", bootstrap)
        self.assertIn("Original full scope.", bootstrap)
        self.assertIn("Previous own draft with completed evidence.", bootstrap)
        self.assertTrue(bootstrap.endswith("New message:\n" + full_request))
        await self._confirm()
        await prompt.acknowledge(self.record)

        self.assertEqual(
            prompt.render(self.record), "New message:\nJudge follow-up: verify the remaining target.",
        )
        # A provider-confirmed missing session or a changed native identity
        # must restore the complete request and evidence for its fresh start.
        for native_id in (None, _OTHER_SESSION_ID):
            with self.subTest(native_id=native_id):
                await self.record.save(native_session_id=native_id)
                self.assertEqual(prompt.render(self.record), bootstrap)

    async def test_changed_native_uuid_requires_complete_bootstrap(self) -> None:
        data = {"messages": [_message("assistant", "old session history")]}
        prompt = NativePrompt("request", {"memory": "retained context"}, _history(data))
        await self._confirm()
        await prompt.acknowledge(self.record)
        await self.record.save(native_session_id=_OTHER_SESSION_ID)

        self.assertEqual(prompt.render(self.record), prompt.render(None))
        await prompt.acknowledge(self.record)
        self.assertEqual(self.record.get("native_prompt_ledger")["session_id"], _OTHER_SESSION_ID)
        self.assertEqual(prompt.render(self.record), "New message:\nrequest")

    async def test_invalid_native_ids_never_suppress_or_acknowledge_bootstrap(self) -> None:
        prompt = NativePrompt("request", {"memory": "required bootstrap"})
        for invalid in (None, True, 1, "latest", "--last", _SESSION_ID[:8], _SESSION_ID + "\n"):
            with self.subTest(native_session_id=invalid):
                ledger = {"session_id": invalid, "sections": {
                    "memory": native_prompt.fingerprint("required bootstrap"),
                }, "messages": {}}
                await self.record.save(native_session_id=invalid, native_tools_started=True,
                                       native_prompt_ledger=ledger)
                self.assertEqual(prompt.render(self.record), prompt.render(None))
                await prompt.acknowledge(self.record)
                self.assertEqual(self.record.get("native_prompt_ledger"), ledger)

    async def test_acknowledgement_requires_activity_beyond_native_initialization(self) -> None:
        prompt = NativePrompt("request", {"memory": "required context"})
        previous = {"sentinel": "do not replace"}
        for native_id, activity in ((None, True), (_SESSION_ID, False), (_SESSION_ID, None),
                                    (_SESSION_ID, "false")):
            with self.subTest(native_id=native_id, activity=activity):
                await self.record.save(native_session_id=native_id, native_tools_started=activity,
                                       native_prompt_ledger=previous)
                await prompt.acknowledge(self.record)
                self.assertEqual(self.record.get("native_prompt_ledger"), previous)
        await self._confirm()
        await prompt.acknowledge(self.record)
        ledger = self.record.get("native_prompt_ledger")
        self.assertEqual(ledger["session_id"], _SESSION_ID)
        self.assertNotIn("required context", str(ledger))

    async def test_message_coordinates_survive_compaction_without_replaying_retained_messages(self) -> None:
        messages = [_message("user", "old first message"), _message("assistant", "old first reply"),
                    _message("user", "retained-message-marker")]
        before = {"messages": messages, "compacted_count": 0}
        prompt = NativePrompt("first request", {"summary": "previous summary"}, _history(before))
        await self._confirm()
        await prompt.acknowledge(self.record)
        after = {"compacted_count": 2, "messages": [messages[2],
                 _message("assistant", "new-message-after-compaction-marker")]}
        retained_hash = native_prompt.history_snapshot(before)["2"]
        self.assertEqual(native_prompt.history_snapshot(after)["2"], retained_hash)
        next_prompt = NativePrompt("next request", {"summary": "updated compacted summary"}, _history(after))

        rendered = next_prompt.render(self.record)

        self.assertNotIn("retained-message-marker", rendered)
        self.assertIn("new-message-after-compaction-marker", rendered)
        self.assertIn("updated compacted summary", rendered)

    async def test_rewritten_message_at_same_coordinate_is_supplied_again(self) -> None:
        before = {"compacted_count": 10, "messages": [
            _message("user", "original oversized prefix and unseen suffix"),
        ]}
        prompt = NativePrompt("first request", {}, _history(before))
        await self._confirm()
        await prompt.acknowledge(self.record)
        after = {"compacted_count": 10, "messages": [_message("user", "remaining unseen suffix")]}
        self.assertEqual(set(native_prompt.history_snapshot(before)), {"10"})
        self.assertEqual(set(native_prompt.history_snapshot(after)), {"10"})
        self.assertNotEqual(native_prompt.history_snapshot(before)["10"],
                            native_prompt.history_snapshot(after)["10"])

        rendered = NativePrompt("next request", {}, _history(after)).render(self.record)

        self.assertIn("remaining unseen suffix", rendered)
        self.assertNotIn("original oversized prefix", rendered)

    async def test_logged_turn_acknowledges_only_own_pair_leaving_other_backends_unseen(self) -> None:
        known = [_message("user", "known historical request"),
                 _message("assistant", "known historical answer")]
        before = {"messages": known, "compacted_count": 20}
        prompt = NativePrompt("current own request", {"policy": "same policy"}, _history(before))
        await self._confirm()
        await prompt.acknowledge(self.record)
        current = {"compacted_count": 20, "messages": known + [
            _message("user", "other-backend-request-marker"),
            _message("assistant", "other-backend-answer-marker"),
            _message("user", "current own request"),
            _message("assistant", "current-own-answer-marker"),
        ]}

        await native_prompt.acknowledge_logged_turn(self.record, current)
        ledger = self.record.get("native_prompt_ledger")
        self.assertEqual(set(ledger["messages"]), {"20", "21", "24", "25"})
        next_prompt = NativePrompt("next request", {"policy": "same policy"}, _history(current))
        rendered = next_prompt.render(self.record)
        self.assertIn("other-backend-request-marker", rendered)
        self.assertIn("other-backend-answer-marker", rendered)
        self.assertNotIn("known historical request", rendered)
        self.assertNotIn("known historical answer", rendered)
        self.assertNotIn("current own request", rendered)
        self.assertNotIn("current-own-answer-marker", rendered)

    async def test_logged_turn_cannot_acknowledge_without_same_active_session_ledger(self) -> None:
        data = {"messages": [_message("user", "request"), _message("assistant", "answer")]}
        for native_id, activity, ledger in (
            (None, True, {}),
            (_SESSION_ID, False, {"session_id": _SESSION_ID, "messages": {}}),
            (_SESSION_ID, True, {"session_id": _OTHER_SESSION_ID, "messages": {}}),
            ("--last", True, {"session_id": "--last", "messages": {}}),
            (_SESSION_ID, True, {}),
        ):
            with self.subTest(native_id=native_id, activity=activity, ledger=ledger):
                original = copy.deepcopy(ledger)
                await self.record.save(native_session_id=native_id, native_tools_started=activity,
                                       native_prompt_ledger=ledger)
                await native_prompt.acknowledge_logged_turn(self.record, data)
                self.assertEqual(self.record.get("native_prompt_ledger"), original)
