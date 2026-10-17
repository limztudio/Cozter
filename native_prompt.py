"""Structured, UUID-bound context delivery for native CLI conversations."""

from dataclasses import dataclass, field
import hashlib
import json
import uuid

from . import execution_state, session


def _native_session_id(value: object) -> str | None:
    """Only a full UUID can identify a resumable native conversation."""
    if not isinstance(value, str) or len(value) != 36:
        return None
    try:
        canonical = str(uuid.UUID(value))
    except ValueError:
        return None
    return canonical if canonical == value.lower() else None


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def delivery_fingerprint(source_hash: str, rendered_text: str) -> str:
    """Bind known message content to the exact preview supplied to the CLI."""
    return fingerprint([source_hash, rendered_text])


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

    def section_fingerprint(self, name: str, content: str) -> str:
        """The same excerpt can be partial or an authoritative full section."""
        return fingerprint([content, name in self.partial_sections])

    def render(self, record: execution_state.ExecutionRecord | None) -> str:
        native_id = _native_session_id(record.get("native_session_id")) if record is not None else None
        ledger = record.get("native_prompt_ledger", {}) if record is not None else {}
        resumed = bool(native_id and isinstance(ledger, dict)
                       and ledger.get("session_id") == native_id)
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
        unseen = [text for coordinate, digest, text in self.messages
                  if not resumed or previous_messages.get(coordinate) != digest]
        if unseen:
            parts.append("[Conversation messages not yet supplied to this native session]\n"
                         + "\n".join(unseen))
        # Always send the actionable request, even when it equals a prior turn.
        request = self.bootstrap_request if not resumed and self.bootstrap_request is not None else self.request
        parts.append("New message:\n" + request)
        return "\n\n".join(parts)

    async def acknowledge(self, record: execution_state.ExecutionRecord) -> None:
        native_id = _native_session_id(record.get("native_session_id"))
        if native_id is None or record.get("native_tools_started") is not True:
            return
        previous = record.get("native_prompt_ledger", {})
        known = previous.get("messages", {}) if (
            isinstance(previous, dict) and previous.get("session_id") == native_id
        ) else {}
        if not isinstance(known, dict):
            known = {}
        await record.save(native_prompt_ledger={
            "session_id": native_id,
            "sections": {name: self.section_fingerprint(name, text) for name, text in self.sections.items()},
            "messages": {**known, **{coordinate: digest for coordinate, digest, _ in self.messages}},
        })


async def acknowledge_logged_turn(record: execution_state.ExecutionRecord, data: dict) -> None:
    """The direct worker already knows its own request and final response."""
    ledger = record.get("native_prompt_ledger", {})
    native_id = _native_session_id(record.get("native_session_id"))
    if (native_id is None or record.get("native_tools_started") is not True
            or not isinstance(ledger, dict) or ledger.get("session_id") != native_id):
        return
    # Include only the just-logged pair. Other backends' messages remain unseen.
    marks = history_snapshot(data)
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        return
    start = session.total_message_count(data) - len(messages)
    logged = {}
    for index in range(max(0, len(messages) - 2), len(messages)):
        coordinate = str(start + index)
        message = messages[index]
        if coordinate in marks and isinstance(message, dict):
            logged[coordinate] = delivery_fingerprint(marks[coordinate], session.format_msg_line(message))
    known = ledger.get("messages", {})
    if not isinstance(known, dict):
        known = {}
    ledger["messages"] = {**known, **logged}
    await record.save(native_prompt_ledger=ledger)
