"""Structured, UUID-bound context delivery for native CLI conversations."""

from dataclasses import dataclass, field
import hashlib
import json
from typing import Callable

from . import execution_state, session


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def delivery_fingerprint(source_hash: str, rendered_text: str) -> str:
    """Bind known message content to the exact preview supplied to the CLI."""
    return fingerprint([source_hash, rendered_text])


def _matching_ledger(record: execution_state.ExecutionRecord | None) -> dict | None:
    if record is None:
        return None
    native_id = execution_state.native_session_id(record.get("native_session_id"))
    ledger = record.get("native_prompt_ledger", {})
    if native_id and isinstance(ledger, dict) and ledger.get("session_id") == native_id:
        return ledger
    return None


def message_is_known(
    record: execution_state.ExecutionRecord | None, coordinate: str, source_hash: str,
    rendered_text: str, *, complete: bool = False,
) -> bool:
    """Only proven full coverage can suppress a different view of a message.

    Call this before allocating history space, then again on a budgeted excerpt
    before spending its allowance. A known partial preview must not prevent
    older unseen messages from using the same space.
    """
    return message_is_known_from_ledger(
        _matching_ledger(record), coordinate, source_hash, rendered_text, complete=complete,
    )


def message_is_known_from_ledger(
    ledger: dict | None, coordinate: str, source_hash: str, rendered_text: str,
    *, complete: bool = False,
) -> bool:
    """Check one message against an already UUID-validated immutable snapshot."""
    if ledger is None:
        return False
    sources = ledger.get("message_sources", {})
    receipt = sources.get(coordinate) if isinstance(sources, dict) else None
    digest = delivery_fingerprint(source_hash, rendered_text)
    if isinstance(receipt, dict) and receipt.get("source") == source_hash:
        if receipt.get("complete") is True:
            return True
        deliveries = receipt.get("deliveries", [])
        if isinstance(deliveries, list) and digest in deliveries:
            return True
    # An exact projected hash proves that view only, never full coverage.
    messages = ledger.get("messages", {})
    return isinstance(messages, dict) and messages.get(coordinate) == digest


def _merge_source_receipt(
    previous: object, source_hash: str, digest: str, *, complete: bool,
) -> dict:
    previous = previous if isinstance(previous, dict) else {}
    same_source = previous.get("source") == source_hash
    prior_deliveries = previous.get("deliveries", []) if same_source else []
    deliveries = [item for item in prior_deliveries if isinstance(item, str)] if (
        isinstance(prior_deliveries, list)
    ) else []
    if digest not in deliveries:
        deliveries.append(digest)
    return {
        "source": source_hash,
        "complete": complete or bool(same_source and previous.get("complete") is True),
        "deliveries": deliveries,
    }


def history_snapshot(data: dict) -> dict[str, str]:
    """Message coordinates survive compaction; hashes detect in-place rewrites."""
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        return {}
    start = session.total_message_count(data) - len(messages)
    return {str(start + index): fingerprint(message)
            for index, message in enumerate(messages) if isinstance(message, dict)}


@dataclass
class NativePrompt:
    """Keep bootstrap and incremental inputs explicit, without parsing prose."""

    request: str
    sections: dict[str, str]
    messages: list[tuple[str, str, str]] = field(default_factory=list)
    bootstrap_request: str | None = None
    bootstrap_sections: dict[str, str] = field(default_factory=dict)
    partial_sections: set[str] = field(default_factory=set)
    message_sources: dict[str, tuple[str, bool]] = field(default_factory=dict)
    request_identity: str | None = None
    prepare: Callable[[execution_state.ExecutionRecord | None], "NativePrompt"] | None = field(
        default=None, repr=False,
    )
    turn_instructions: str = ""

    def for_record(self, record: execution_state.ExecutionRecord | None) -> "NativePrompt":
        """Choose missing context only after selecting this worker's native lane."""
        return self.prepare(record) if self.prepare is not None else self

    def section_fingerprint(self, name: str, content: str) -> str:
        """The same excerpt can be partial or an authoritative full section."""
        return fingerprint([content, name in self.partial_sections])

    def request_for(self, record: execution_state.ExecutionRecord | None) -> str:
        """Return the actual request body, even when rendering a continuation."""
        return self._request_for_ledger(_matching_ledger(record))

    def _request_for_ledger(self, ledger: dict | None) -> str:
        if ledger is None and self.bootstrap_request is not None:
            return self.bootstrap_request
        return self.request

    def request_text_for(self, record: execution_state.ExecutionRecord | None) -> str:
        """Use exactly the same full request or continuation for quota and output."""
        return self._request_text_for_ledger(_matching_ledger(record))

    def _request_text_for_ledger(self, ledger: dict | None) -> str:
        request = self._request_for_ledger(ledger)
        receipt = (ledger or {}).get("request", {})
        if (self.request_identity is not None and isinstance(receipt, dict)
                and receipt.get("identity") == self.request_identity
                and receipt.get("body") == fingerprint(request)):
            return ("Continue the current job from this native conversation. "
                    "Its request is already supplied; complete the remaining work.")
        return request

    def render(self, record: execution_state.ExecutionRecord | None) -> str:
        ledger = _matching_ledger(record)
        resumed = ledger is not None
        ledger = ledger or {}
        previous_sections = ledger.get("sections", {}) if resumed else {}
        previous_messages = ledger.get("messages", {}) if resumed else {}
        if not isinstance(previous_sections, dict):
            previous_sections = {}
        if not isinstance(previous_messages, dict):
            previous_messages = {}
        parts = []
        for name in dict.fromkeys([*self.sections, *previous_sections]):
            content = self.sections.get(name, "")
            if resumed and previous_sections.get(name) == self.section_fingerprint(name, content):
                continue
            if content:
                if not resumed:
                    parts.append(content)
                elif name in self.partial_sections:
                    parts.append(
                        f"[Updated {name} excerpt for this turn; retain previously supplied facts"
                        f" outside this excerpt]\n{content}"
                    )
                else:
                    parts.append(f"[Updated {name}; replaces the previous version]\n{content}")
            elif resumed and name in previous_sections:
                if name in self.partial_sections:
                    parts.append(f"[{name} omitted by this turn's budget; retain previously supplied context]")
                else:
                    parts.append(f"[Updated {name}: cleared; replaces the previous version]")
        if not resumed:
            parts.extend(content for content in self.bootstrap_sections.values() if content)
        unseen = []
        for coordinate, digest, text in self.messages:
            source = self.message_sources.get(coordinate)
            known = message_is_known_from_ledger(ledger, coordinate, source[0], text, complete=source[1]) if (
                source is not None
            ) else resumed and previous_messages.get(coordinate) == digest
            if not known:
                unseen.append(text)
        if unseen:
            parts.append("[Conversation messages not yet supplied to this native session]\n"
                         + "\n".join(unseen))
        # Invocation requirements remain fresh even when the native session
        # already knows every section, message, and durable request below.
        # They are deliberately excluded from context delivery receipts.
        if self.turn_instructions:
            parts.append(self.turn_instructions)
        # Identical new jobs remain actionable; an interrupted durable job can
        # continue from the full request already in this exact native session.
        parts.append("New message:\n" + self._request_text_for_ledger(ledger if resumed else None))
        return "\n\n".join(parts)

    async def acknowledge(
        self, record: execution_state.ExecutionRecord, *, delivered_request: str | None = None,
    ) -> None:
        native_id = execution_state.native_session_id(record.get("native_session_id"))
        if native_id is None or record.get("native_tools_started") is not True:
            return
        previous = _matching_ledger(record) or {}
        known = previous.get("messages", {})
        if not isinstance(known, dict):
            known = {}
        known_sources = previous.get("message_sources", {})
        if not isinstance(known_sources, dict):
            known_sources = {}
        sources = dict(known_sources)
        for coordinate, digest, _ in self.messages:
            if coordinate not in self.message_sources:
                continue
            source_hash, complete = self.message_sources[coordinate]
            sources[coordinate] = _merge_source_receipt(
                sources.get(coordinate), source_hash, digest, complete=complete,
            )
        ledger = {
            "session_id": native_id,
            "sections": {name: self.section_fingerprint(name, text) for name, text in self.sections.items()},
            "messages": {**known, **{coordinate: digest for coordinate, digest, _ in self.messages}},
            "message_sources": sources,
        }
        if self.request_identity is not None:
            ledger["request"] = {
                "identity": self.request_identity,
                "body": fingerprint(self.request_for(record) if delivered_request is None else delivered_request),
            }
        await record.save(native_prompt_ledger=ledger)


async def acknowledge_logged_turn(
    record: execution_state.ExecutionRecord, data: dict, *, request_only: bool = False,
) -> None:
    """The direct worker already knows its own request and final response."""
    ledger = record.get("native_prompt_ledger", {})
    native_id = execution_state.native_session_id(record.get("native_session_id"))
    if (native_id is None or (not request_only and record.get("native_tools_started") is not True)
            or not isinstance(ledger, dict) or ledger.get("session_id") != native_id):
        return
    # Include only the just-logged pair. Other backends' messages remain unseen.
    marks = history_snapshot(data)
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        return
    start = session.total_message_count(data) - len(messages)
    logged = {}
    stop = max(0, len(messages) - 1) if request_only else len(messages)
    sources = ledger.get("message_sources", {})
    if not isinstance(sources, dict):
        sources = {}
    sources = dict(sources)
    for index in range(max(0, len(messages) - 2), stop):
        coordinate = str(start + index)
        message = messages[index]
        if coordinate in marks and isinstance(message, dict):
            logged[coordinate] = delivery_fingerprint(marks[coordinate], session.format_msg_line(message))
            sources[coordinate] = _merge_source_receipt(
                sources.get(coordinate), marks[coordinate], logged[coordinate], complete=True,
            )
    known = ledger.get("messages", {})
    if not isinstance(known, dict):
        known = {}
    ledger["messages"] = {**known, **logged}
    ledger["message_sources"] = sources
    await record.save(native_prompt_ledger=ledger)
