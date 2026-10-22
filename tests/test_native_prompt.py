"""Offline context delivery to an exact, activity-confirmed native session."""

import copy
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent, execution_state, native_prompt, session
from Cozter.native_prompt import NativePrompt
from Cozter.utils import json_digest


_SESSION_ID = "01a126fa-e54a-7753-8078-3e31ff8fc962"
_OTHER_SESSION_ID = "f53ba06b-a880-4e59-93ea-bd623b0b922b"


def _message(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _history(data: dict) -> list[tuple[str, str, str]]:
    marks = native_prompt.history_snapshot(data)
    return [(coordinate, native_prompt.delivery_fingerprint(digest, session.format_msg_line(message)),
             session.format_msg_line(message))
            for (coordinate, digest), message in zip(marks.items(), data["messages"], strict=True)]


def _request_only(prompt: NativePrompt, request: str) -> str:
    return "\n\n".join(part for part in (prompt.turn_instructions, "New message:\n" + request) if part)


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

    async def test_turn_instructions_remain_on_new_request_without_replaying_known_catalog_or_history(self) -> None:
        data = {"messages": [_message("user", "already-delivered-history")]}
        sections = {"skills catalog": "unchanged-catalog", "policy": "unchanged-policy"}
        reminder = "Re-read the relevant authoritative skill instructions for this request."
        first = NativePrompt(
            "first request", sections, _history(data), request_identity="first-job",
            turn_instructions=reminder,
        )
        bootstrap = first.render(None)
        self.assertIn("already-delivered-history", bootstrap)
        self.assertIn("unchanged-catalog", bootstrap)
        self.assertTrue(bootstrap.endswith(reminder + "\n\nNew message:\nfirst request"))
        await self._confirm()
        await first.acknowledge(self.record)

        next_prompt = NativePrompt(
            "next request", sections, _history(data), request_identity="next-job",
            turn_instructions=reminder,
        )
        rendered = next_prompt.render(self.record)

        self.assertEqual(rendered, reminder + "\n\nNew message:\nnext request")
        self.assertNotIn("already-delivered-history", rendered)
        self.assertNotIn("unchanged-catalog", rendered)
        self.assertNotIn("unchanged-policy", rendered)
        self.assertEqual(self.record.get("native_session_id"), _SESSION_ID)
        ledger = self.record.get("native_prompt_ledger")
        self.assertEqual(set(ledger["sections"]), set(sections))
        self.assertNotIn("turn_instructions", ledger)
        self.assertNotIn(reminder, str(ledger))

    async def test_turn_instructions_remain_on_same_job_continuation_without_changing_receipts(self) -> None:
        data = {"messages": [_message("assistant", "known native evidence")]}
        request = "Complete the original durable request."
        prompt = NativePrompt(
            request, {"skills catalog": "known-catalog"}, _history(data),
            request_identity="same-job", turn_instructions="Initial invocation reminder.",
        )
        await self._confirm()
        await prompt.acknowledge(self.record)
        before = copy.deepcopy(self.record.get("native_prompt_ledger"))
        prompt.turn_instructions = "Read the relevant skill files again before this continuation."

        rendered = prompt.render(self.record)

        self.assertTrue(rendered.startswith(prompt.turn_instructions + "\n\nNew message:\n"))
        self.assertIn("Continue the current job from this native conversation.", rendered)
        self.assertNotIn(request, rendered)
        self.assertNotIn("known native evidence", rendered)
        self.assertNotIn("known-catalog", rendered)
        await prompt.acknowledge(self.record)
        self.assertEqual(self.record.get("native_prompt_ledger"), before)
        self.assertEqual(self.record.get("native_session_id"), _SESSION_ID)

    async def test_removing_turn_instructions_does_not_clear_any_native_context(self) -> None:
        prompt = NativePrompt("request", {"skills catalog": "retained-catalog"},
                              turn_instructions="Per-invocation reminder.")
        await self._confirm()
        await prompt.acknowledge(self.record)
        prompt.turn_instructions = ""

        self.assertEqual(prompt.render(self.record), "New message:\nrequest")

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
        self.assertEqual(preview.render(self.record), _request_only(preview, "current request"))
        expanded = build(4_000)
        self.assertEqual(len(expanded.messages), 1)
        self.assertEqual(expanded.messages[0][0], "8")
        self.assertEqual(expanded.messages[0][2], session.format_msg_line(data["messages"][0]))
        self.assertNotEqual(preview.messages[0][1], expanded.messages[0][1])

        rendered = expanded.render(self.record)

        self.assertIn(content, rendered)
        await expanded.acknowledge(self.record)
        self.assertEqual(expanded.render(self.record), _request_only(expanded, "current request"))

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
                    "memory": json_digest("required bootstrap"),
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

    async def test_budget_shrink_retains_omitted_known_messages_when_budget_expands(self) -> None:
        backend = SimpleNamespace(
            supports_detached_tasks=False, supports_typed_plugins=True, supports_plugin_prelude=False,
        )
        data = {"messages": [
            _message("user", "older-known-marker " + "older content " * 35),
            _message("assistant", "recent-known-marker"),
        ]}

        def build(budget: int) -> NativePrompt:
            return agent._native_turn_prompt(
                backend, "request", data, [], budget=budget, collaborative=False, is_scheduled=False,
            )

        full = build(2_000)
        self.assertEqual([row[0] for row in full.messages], ["0", "1"])
        await self._confirm()
        await full.acknowledge(self.record)
        smaller = build(200)
        self.assertEqual([row[0] for row in smaller.messages], ["1"])
        await smaller.acknowledge(self.record)

        repeated = build(2_000)
        self.assertEqual(repeated.render(self.record), _request_only(repeated, "request"))
        self.assertEqual(set(self.record.get("native_prompt_ledger")["messages"]), {"0", "1"})
        data["messages"][0] = _message("user", "rewritten-older-message-marker")
        rewritten = build(2_000)
        self.assertIn("rewritten-older-message-marker", rewritten.render(self.record))
        self.assertNotIn("recent-known-marker", rewritten.render(self.record))
        await rewritten.acknowledge(self.record)
        self.assertEqual(rewritten.render(self.record), _request_only(rewritten, "request"))

    async def test_budgeted_acknowledgment_does_not_carry_known_messages_to_changed_uuid(self) -> None:
        messages = _history({"messages": [
            _message("user", "old-session-only-marker"),
            _message("assistant", "new-session-provided-marker"),
        ]})
        full = NativePrompt("request", {}, messages)
        await self._confirm()
        await full.acknowledge(self.record)
        await self.record.save(native_session_id=_OTHER_SESSION_ID)
        smaller = NativePrompt("request", {}, messages[1:])
        await smaller.acknowledge(self.record)

        ledger = self.record.get("native_prompt_ledger")
        self.assertEqual(ledger["session_id"], _OTHER_SESSION_ID)
        self.assertEqual(set(ledger["messages"]), {"1"})
        rendered = full.render(self.record)
        self.assertIn("old-session-only-marker", rendered)
        self.assertNotIn("new-session-provided-marker", rendered)

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

    async def test_full_source_receipt_suppresses_every_preview_without_coverage_downgrade(self) -> None:
        message = _message("assistant", "full-known-head " + "evidence " * 200 + " full-known-tail")
        source = json_digest(message)
        full_text = session.format_msg_line(message, cap=None)
        full = NativePrompt(
            "request", {}, [("7", native_prompt.delivery_fingerprint(source, full_text), full_text)],
            message_sources={"7": (source, True)},
        )
        await self._confirm()
        await full.acknowledge(self.record)

        for text in (session.format_msg_line(message), full_text[:100], full_text[:2]):
            with self.subTest(preview_length=len(text)):
                self.assertTrue(native_prompt.message_is_known_from_ledger(
                    native_prompt.matching_ledger(self.record), "7", source, text,
                ))
                smaller = NativePrompt(
                    "request", {}, [("7", native_prompt.delivery_fingerprint(source, text), text)],
                    message_sources={"7": (source, False)},
                )
                self.assertEqual(smaller.render(self.record), "New message:\nrequest")
                await smaller.acknowledge(self.record)
                self.assertTrue(self.record.get("native_prompt_ledger")["message_sources"]["7"]["complete"])
        self.assertTrue(native_prompt.message_is_known_from_ledger(
            native_prompt.matching_ledger(self.record), "7", source, full_text,
        ))

        # A rewrite can preserve its visible prefix while changing its source.
        rewritten = dict(message, content=message["content"] + " changed-unseen-tail")
        changed_source = json_digest(rewritten)
        same_preview = full_text[:100]
        self.assertFalse(native_prompt.message_is_known_from_ledger(
            native_prompt.matching_ledger(self.record), "7", changed_source, same_preview,
        ))
        changed = NativePrompt(
            "request", {}, [("7", native_prompt.delivery_fingerprint(changed_source, same_preview), same_preview)],
            message_sources={"7": (changed_source, False)},
        )
        await changed.acknowledge(self.record)
        receipt = self.record.get("native_prompt_ledger")["message_sources"]["7"]
        self.assertEqual(receipt["source"], changed_source)
        self.assertFalse(receipt["complete"])
        self.assertFalse(native_prompt.message_is_known_from_ledger(
            native_prompt.matching_ledger(self.record), "7", source, full_text,
        ))

    async def test_partial_receipts_remember_exact_views_without_claiming_full_source(self) -> None:
        source = json_digest(_message("user", "source with content beyond both previews"))

        async def supply(text: str, *, complete: bool = False) -> NativePrompt:
            prompt = NativePrompt(
                "request", {}, [("12", native_prompt.delivery_fingerprint(source, text), text)],
                message_sources={"12": (source, complete)},
            )
            await prompt.acknowledge(self.record)
            return prompt

        await self._confirm()
        first = await supply("first partial preview")
        await supply("different smaller preview")
        self.assertEqual(first.render(self.record), "New message:\nrequest")
        self.assertTrue(native_prompt.message_is_known_from_ledger(
            native_prompt.matching_ledger(self.record), "12", source, "first partial preview",
        ))
        self.assertFalse(native_prompt.message_is_known_from_ledger(
            native_prompt.matching_ledger(self.record), "12", source, "first partial",
        ))
        # A partial receipt cannot hide a different preview until the full
        # source is proven.
        self.assertFalse(self.record.get("native_prompt_ledger")["message_sources"]["12"]["complete"])
        await supply("full canonical message", complete=True)
        self.assertTrue(native_prompt.message_is_known_from_ledger(
            native_prompt.matching_ledger(self.record), "12", source, "any shorter preview",
        ))

    async def test_existing_exact_delivery_hash_proves_only_its_view_without_source_upgrade(self) -> None:
        data = {"messages": [_message("assistant", "previously supplied full short message")]}
        await self._confirm()
        await NativePrompt("request", {}, _history(data)).acknowledge(self.record)
        source = native_prompt.history_snapshot(data)["0"]
        text = session.format_msg_line(data["messages"][0], cap=None)
        self.assertTrue(native_prompt.message_is_known_from_ledger(
            native_prompt.matching_ledger(self.record), "0", source, text,
        ))
        self.assertFalse(native_prompt.message_is_known_from_ledger(
            native_prompt.matching_ledger(self.record), "0", source, text[:10],
        ))
        self.assertEqual(self.record.get("native_prompt_ledger")["message_sources"], {})

    async def test_source_receipts_survive_reload_but_never_cross_native_uuid(self) -> None:
        source = json_digest(_message("assistant", "durable full evidence"))
        prompt = NativePrompt(
            "request", {}, [("3", native_prompt.delivery_fingerprint(source, "full evidence"), "full evidence")],
            message_sources={"3": (source, True)},
        )
        await self._confirm()
        await prompt.acknowledge(self.record)
        await self.execution.flush()
        await self.record.reload()
        self.assertTrue(native_prompt.message_is_known_from_ledger(
            native_prompt.matching_ledger(self.record), "3", source, "excerpt",
        ))
        for changed_id in (_OTHER_SESSION_ID, None, "--last"):
            with self.subTest(changed_id=changed_id):
                await self.record.save(native_session_id=changed_id)
                self.assertFalse(native_prompt.message_is_known_from_ledger(
                    native_prompt.matching_ledger(self.record), "3", source, "excerpt",
                ))

    async def test_budget_omission_preserves_full_source_receipts_but_uuid_change_drops_them(self) -> None:
        source = json_digest(_message("assistant", "known full content"))
        prompt = NativePrompt(
            "request", {}, [("9", native_prompt.delivery_fingerprint(source, "known full content"),
                              "known full content")],
            message_sources={"9": (source, True)},
        )
        await self._confirm()
        await prompt.acknowledge(self.record)
        omitted = NativePrompt("request", {})
        await omitted.acknowledge(self.record)
        self.assertTrue(native_prompt.message_is_known_from_ledger(
            native_prompt.matching_ledger(self.record), "9", source, "shorter preview",
        ))

        await self.record.save(native_session_id=_OTHER_SESSION_ID)
        await omitted.acknowledge(self.record)
        self.assertEqual(self.record.get("native_prompt_ledger")["message_sources"], {})
        self.assertFalse(native_prompt.message_is_known_from_ledger(
            native_prompt.matching_ledger(self.record), "9", source, "shorter preview",
        ))

    async def test_logged_own_long_messages_are_full_coverage_but_flexible_merge_is_unseen(self) -> None:
        await self._confirm()
        await NativePrompt("request", {}).acknowledge(self.record)
        data = {"compacted_count": 40, "messages": [
            _message("user", "own long request " + "requirements " * 200),
            _message("assistant", "merged answer from other workers " + "evidence " * 200),
        ]}
        marks = native_prompt.history_snapshot(data)
        await native_prompt.acknowledge_logged_turn(self.record, data, request_only=True)
        self.assertTrue(native_prompt.message_is_known_from_ledger(
            native_prompt.matching_ledger(self.record), "40", marks["40"], "tiny request preview",
        ))
        self.assertFalse(native_prompt.message_is_known_from_ledger(
            native_prompt.matching_ledger(self.record), "41", marks["41"], session.format_msg_line(data["messages"][1]),
        ))
        self.assertEqual(set(self.record.get("native_prompt_ledger")["messages"]), {"40"})

        await native_prompt.acknowledge_logged_turn(self.record, data)
        self.assertTrue(native_prompt.message_is_known_from_ledger(
            native_prompt.matching_ledger(self.record), "41", marks["41"], session.format_msg_line(data["messages"][1], cap=None),
        ))

    async def test_same_durable_request_uses_continuation_but_identical_new_job_remains_actionable(self) -> None:
        request = "large-current-request-marker " + "exact requirements " * 2_000
        prompt = NativePrompt(request, {}, request_identity="durable-job")
        self.assertTrue(prompt.render(None).endswith(request))
        await self._confirm()
        delivered = prompt.request_for(self.record)
        await prompt.acknowledge(self.record, delivered_request=delivered)

        resumed = prompt.render(self.record)
        self.assertLess(len(resumed), 250)
        self.assertIn("Continue the current job", resumed)
        self.assertNotIn("large-current-request-marker", resumed)
        self.assertEqual(prompt.request_for(self.record), request)
        self.assertEqual(
            NativePrompt(request, {}, request_identity="different-job").render(self.record),
            "New message:\n" + request,
        )
        self.assertEqual(
            NativePrompt("updated scope", {}, request_identity="durable-job").render(self.record),
            "New message:\nupdated scope",
        )

    async def test_bootstrap_request_receipt_tracks_body_delivered_before_native_initialization(self) -> None:
        prompt = NativePrompt(
            "short follow-up", {}, bootstrap_request="Original full task\nshort follow-up",
            request_identity="same-durable-job",
        )
        bootstrap_body = prompt.request_for(None)
        await self._confirm()
        await prompt.acknowledge(self.record, delivered_request=bootstrap_body)
        self.assertEqual(self.record.get("native_prompt_ledger")["request"]["body"],
                         json_digest(bootstrap_body))
        self.assertEqual(prompt.request_for(self.record), "short follow-up")
        self.assertEqual(prompt.render(self.record), "New message:\nshort follow-up")
        await prompt.acknowledge(self.record, delivered_request="short follow-up")
        self.assertIn("Continue the current job", prompt.render(self.record))

        for missing_id in (None, _OTHER_SESSION_ID):
            with self.subTest(missing_id=missing_id):
                await self.record.save(native_session_id=missing_id)
                self.assertEqual(prompt.request_for(self.record), bootstrap_body)
                self.assertEqual(prompt.render(self.record), "New message:\n" + bootstrap_body)

    async def test_preparation_spends_history_quota_on_unseen_older_message_before_known_recent(self) -> None:
        backend = SimpleNamespace(
            supports_detached_tasks=False, supports_typed_plugins=True, supports_plugin_prelude=False,
        )
        data = {"messages": [
            _message("user", "older-unseen-marker " + "older evidence " * 35),
            _message("assistant", "recent-already-known-marker " + "recent evidence " * 35),
        ]}
        recent = data["messages"][1]
        source = native_prompt.history_snapshot(data)["1"]
        recent_text = session.format_msg_line(recent, cap=None)
        known = NativePrompt(
            "previous request", {}, [("1", native_prompt.delivery_fingerprint(source, recent_text), recent_text)],
            message_sources={"1": (source, True)},
        )
        await self._confirm()
        await known.acknowledge(self.record)
        template = agent._native_turn_prompt(
            backend, "current request", data, [], budget=350,
            collaborative=False, is_scheduled=False,
        )
        cold_coordinates = [coordinate for coordinate, _, _ in template.messages]
        self.assertEqual(cold_coordinates, ["1"])

        prepared = template.for_record(self.record)

        self.assertEqual([coordinate for coordinate, _, _ in prepared.messages], ["0"])
        self.assertIn("older-unseen-marker", prepared.render(self.record))
        self.assertNotIn("recent-already-known-marker", prepared.render(self.record))
        self.assertEqual([coordinate for coordinate, _, _ in template.messages], cold_coordinates)
        self.assertEqual(template.for_record(None).render(None), template.render(None))

    async def test_known_recent_partial_view_releases_its_quota_for_older_unseen_history(self) -> None:
        backend = SimpleNamespace(
            supports_detached_tasks=False, supports_typed_plugins=True, supports_plugin_prelude=False,
        )
        data = {"messages": [
            _message("user", "older-partial-marker " + "older evidence " * 150),
            _message("assistant", "recent-partial-marker " + "recent evidence " * 150),
        ]}

        def build(budget: int) -> NativePrompt:
            return agent._native_turn_prompt(
                backend, "current request", data, [], budget=budget,
                collaborative=False, is_scheduled=False,
            )

        await self._confirm()
        recent = build(350).for_record(self.record)
        self.assertEqual([row[0] for row in recent.messages], ["1"])
        self.assertFalse(recent.message_sources["1"][1])
        await recent.acknowledge(self.record)

        older = build(350).for_record(self.record)

        self.assertEqual([row[0] for row in older.messages], ["0"])
        self.assertIn("older-partial-marker", older.render(self.record))
        self.assertNotIn("recent-partial-marker", older.render(self.record))
        await older.acknowledge(self.record)
        self.assertEqual(build(350).for_record(self.record).messages, [])

        expanded = build(4_000).for_record(self.record)
        self.assertEqual([row[0] for row in expanded.messages], ["0", "1"])
        # The ordinary 800-character display cap is still partial, even when
        # the surrounding turn budget can fit both entire displayed views.
        self.assertTrue(all(not complete for _, complete in expanded.message_sources.values()))
        await expanded.acknowledge(self.record)
        self.assertEqual(build(350).for_record(self.record).messages, [])

    async def test_known_large_current_request_leaves_quota_for_newly_missing_history(self) -> None:
        backend = SimpleNamespace(
            supports_detached_tasks=False, supports_typed_plugins=True, supports_plugin_prelude=False,
        )
        request = "huge-existing-current-request " + "requirements " * 3_000
        data = {"messages": [_message("assistant", "newly-missing-history-marker")]}
        with mock.patch.object(execution_state, "current_execution", return_value=self.execution):
            template = agent._native_turn_prompt(
                backend, request, data, [], budget=1_000,
                collaborative=False, is_scheduled=False,
            )
            self.assertEqual(template.messages, [])
            await self._confirm()
            await template.acknowledge(self.record, delivered_request=request)

            prepared = template.for_record(self.record)

        self.assertEqual([row[0] for row in prepared.messages], ["0"])
        self.assertIn("newly-missing-history-marker", prepared.render(self.record))
        self.assertNotIn("huge-existing-current-request", prepared.render(self.record))
        self.assertEqual(prepared.request_for(self.record), request)

    async def test_render_and_preparation_copy_delivery_ledger_once_per_operation_not_per_message(self) -> None:
        backend = SimpleNamespace(
            supports_detached_tasks=False, supports_typed_plugins=True, supports_plugin_prelude=False,
        )
        data = {"messages": [_message("assistant", f"known history message {index}") for index in range(150)]}
        marks = native_prompt.history_snapshot(data)
        template = agent._native_turn_prompt(
            backend, "current request", data, [], budget=30_000,
            collaborative=False, is_scheduled=False,
        )
        await self._confirm()
        await template.acknowledge(self.record)
        self.assertEqual(set(self.record.get("native_prompt_ledger")["messages"]), set(marks))

        with mock.patch.object(self.record, "get", wraps=self.record.get) as get:
            self.assertEqual(template.render(self.record), _request_only(template, "current request"))
        ledger_reads = sum(call.args[0] == "native_prompt_ledger" for call in get.call_args_list)
        self.assertEqual(ledger_reads, 1)

        with mock.patch.object(self.record, "get", wraps=self.record.get) as get:
            prepared = template.for_record(self.record)
        self.assertEqual(prepared.messages, [])
        ledger_reads = sum(call.args[0] == "native_prompt_ledger" for call in get.call_args_list)
        # History and request budgeting may each take one validated snapshot.
        self.assertLessEqual(ledger_reads, 2)
