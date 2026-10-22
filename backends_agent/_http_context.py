"""Bound provider input without changing durable execution history."""

from __future__ import annotations

import json
import re
from typing import Any


class ContextLengthError(RuntimeError):
    """The provider rejected input before generating a completion."""

    def __init__(self, message: str, window: int | None = None) -> None:
        super().__init__(message)
        self.window = window


def context_error(error: object, message: str) -> ContextLengthError | None:
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    if not isinstance(code, str) or code.casefold() not in {
        "context_length_exceeded", "max_context_length_exceeded",
        "context_window_exceeded", "model_context_window_exceeded", "context_length_error",
    }:
        return None
    window = None
    for key in ("context_window", "max_context_tokens", "context_limit"):
        value = error.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            window = value
            break
    if window is None:
        detail = error.get("message")
        match = re.search(
            r"(?:context window(?: of)?|maximum context length(?: is)?|context limit(?: is)?)"
            r"\s*([0-9][0-9,_]*)", detail, re.IGNORECASE,
        ) if isinstance(detail, str) else None
        if match:
            try:
                window = int(match.group(1).replace(",", "").replace("_", "")) or None
            except ValueError:
                pass
    return ContextLengthError(message, window)


def payload_bytes(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def context_budget(window: int) -> int:
    # Bytes are a conservative token bound; leave room for framing and output.
    return max(1, window * 4 // 5 - min(16_384, max(256, window // 16)))


def _reference(path: str, first: int, last: int | None = None) -> str:
    selected = str(first) if last is None or first == last else f"{first} through {last}"
    return f"{path}; JSON values.http_messages indexes {selected}"


def project_payload(
    payload: dict[str, Any], budget: int, archive_path: str | None,
) -> dict[str, Any]:
    """Return a wire-only preview; every omitted byte remains in its record."""
    total_bytes = payload_bytes(payload)
    if total_bytes <= budget:
        return payload
    if not archive_path:
        raise RuntimeError(
            "HTTP input exceeds the model context budget and has no durable tool history "
            "to preview. Reduce the current request, attachments, or tool schema; "
            "user instructions have not been discarded."
        )
    original = payload.get("messages")
    if not isinstance(original, list):
        raise RuntimeError("HTTP context projection requires a message list.")
    messages = list(original)
    result = dict(payload, messages=messages)
    message_sizes = [payload_bytes(message) for message in messages]
    groups: list[tuple[int, int]] = []
    position = 0
    while position < len(messages):
        message = messages[position]
        if not isinstance(message, dict):
            raise RuntimeError("HTTP context projection found an invalid message.")
        role = message.get("role")
        if role in {"system", "user"}:
            position += 1
            continue
        if role != "assistant":
            raise RuntimeError("HTTP context projection found an unmatched tool result.")
        start = position
        position += 1
        calls = message.get("tool_calls")
        if calls:
            if not isinstance(calls, list):
                raise RuntimeError("HTTP context projection found invalid tool calls.")
            expected = [call.get("id") for call in calls if isinstance(call, dict)]
            actual = []
            while position < len(messages) and messages[position].get("role") == "tool":
                actual.append(messages[position].get("tool_call_id"))
                position += 1
            if len(expected) != len(calls) or actual != expected:
                raise RuntimeError("HTTP context projection requires complete ordered tool results.")
        groups.append((start, position))

    recent = {index for start, end in groups[-2:] for index in range(start, end)}
    for index, message in enumerate(original):
        body = message.get("content")
        if index in recent or message.get("role") != "tool" or not isinstance(body, str) or len(body) <= 1024:
            continue
        reference = _reference(archive_path, index - 1)
        marker = (
            f"\n[Tool result preview: middle omitted; original {len(body)} characters. "
            f"Exact result: {reference}; tool_call_id={message.get('tool_call_id', '')}. "
            "Use read_file with line/char_offset paging to inspect the saved JSON when needed. "
            "This preview is not full evidence. Do not replay completed side effects.]\n"
        )
        preview = body[:384] + marker + body[-384:]
        if len(preview) >= len(body):
            continue
        messages[index] = dict(message, content=preview)
        preview_bytes = payload_bytes(messages[index])
        total_bytes += preview_bytes - message_sizes[index]
        message_sizes[index] = preview_bytes
        if total_bytes <= budget:
            return result

    removed: set[int] = set()
    removed_bytes = 0
    ranges = []
    # Keep the newest two assistant/tool waves, including opaque reasoning.
    for start, end in groups[:-2]:
        if messages[start].get("reasoning_content"):
            continue
        removed.update(range(start, end))
        removed_bytes += sum(message_sizes[start:end])
        ranges.append(f"{start - 1}:{end - 2}")
        notice = {
            "role": "assistant",
            "content": (
                "[Earlier HTTP evidence omitted from this wire preview. Full exact messages "
                f"remain in {archive_path}; JSON values.http_messages inclusive index ranges "
                + ", ".join(ranges)
                + ". Read the saved JSON with read_file line/char_offset paging when needed. "
                "All user instructions are retained. Completed tools must not be replayed; "
                "unknown outcomes require verification. This is not full evidence.]"
            ),
        }
        candidate_bytes = total_bytes - removed_bytes - len(removed) + payload_bytes(notice) + 1
        if candidate_bytes <= budget:
            visible = []
            inserted = False
            for index, message in enumerate(messages):
                if index in removed:
                    if not inserted:
                        visible.append(notice)
                        inserted = True
                else:
                    visible.append(message)
            return dict(payload, messages=visible)
    raise RuntimeError(
        "HTTP input cannot fit the model context budget while preserving all user instructions, "
        "current system/skills, tool schemas, attachments, and recent tool/reasoning pairs. "
        "The exact execution history is preserved. Reduce the request or schema, or use a "
        "model with a larger context window; completed tools have not been replayed."
    )
