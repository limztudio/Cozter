"""Codex CLI wrapper — runs codex exec and parses JSON event output."""

import asyncio
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ChatEvent:
    """An event produced during a codex exec turn."""
    kind: str  # "tool", "file", "text"
    content: str


@dataclass
class CodexResult:
    """Collected result from a single codex exec run."""
    events: list[ChatEvent] = field(default_factory=list)
    text: str = "(no response)"


async def run(
    prompt: str,
    workspace_path: str,
    model: str | None = None,
    approval: str = "auto",
) -> CodexResult:
    """Run ``codex exec --json`` and return parsed events.

    approval maps to sandbox/approval flags:
      - "auto"    → --full-auto
      - "confirm" → default (codex asks on its own, but non-interactive
                     mode auto-approves, so we use on-request sandbox)
      - "deny"    → --sandbox read-only
    """
    cmd = ["codex", "exec", "--json", "-C", workspace_path]

    if model:
        cmd += ["-m", model]

    if approval == "deny":
        cmd += ["--sandbox", "read-only"]
    elif approval == "confirm":
        cmd += ["--sandbox", "workspace-write"]
    else:
        cmd += ["--full-auto"]

    cmd.append(prompt)

    logger.info("Running: %s", " ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    result = CodexResult()

    # Parse JSON Lines from stdout
    while True:
        line = await proc.stdout.readline()
        if not line:
            break

        line = line.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Non-JSON line: %s", line)
            continue

        _process_event(event, result)

    await proc.wait()

    # Capture any stderr for debugging
    stderr = (await proc.stderr.read()).decode("utf-8", errors="replace").strip()
    if stderr:
        logger.debug("codex stderr: %s", stderr)

    if proc.returncode != 0 and not result.events:
        result.text = f"Codex exited with code {proc.returncode}"
        if stderr:
            result.text += f"\n{stderr}"
        result.events.append(ChatEvent(kind="text", content=result.text))

    # Ensure there's at least one text event
    if not any(e.kind == "text" for e in result.events):
        result.events.append(ChatEvent(kind="text", content=result.text))

    return result


def _process_event(event: dict, result: CodexResult) -> None:
    """Parse a single JSON event from codex exec output."""
    etype = event.get("type", "")
    item = event.get("item", {})
    item_type = item.get("type", "")

    if etype == "item.completed":
        if item_type == "agent_message":
            text = item.get("text", "")
            if text:
                result.text = text
                result.events.append(ChatEvent(kind="text", content=text))

        elif item_type == "command_execution":
            cmd = item.get("command", "?")
            exit_code = item.get("exit_code", "?")
            output = item.get("aggregated_output", "")
            summary = f"$ {cmd} (exit {exit_code})"
            if output:
                # Truncate long output
                if len(output) > 200:
                    output = output[:200] + "..."
                summary += f"\n{output}"
            result.events.append(ChatEvent(kind="tool", content=summary))

        elif item_type == "file_change":
            changes = item.get("changes", [])
            for ch in changes:
                path = ch.get("path", "?")
                kind = ch.get("kind", "?")
                result.events.append(ChatEvent(
                    kind="file",
                    content=f"📄 {kind}: {path}",
                ))

    elif etype == "turn.failed":
        err = event.get("error", {}).get("message", "Unknown error")
        result.text = f"Error: {err}"
        result.events.append(ChatEvent(kind="text", content=result.text))

    elif etype == "error":
        msg = event.get("message", "Unknown error")
        logger.warning("Codex stream error: %s", msg)
