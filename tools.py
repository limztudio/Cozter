"""Workspace tools that the AI can invoke."""

import difflib
import json
import logging
import os
import shutil
import subprocess

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from workspace root."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from workspace root."},
                    "content": {"type": "string", "description": "File content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact string in a file with a new string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from workspace root."},
                    "old_string": {"type": "string", "description": "The exact text to find."},
                    "new_string": {"type": "string", "description": "The replacement text."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from workspace root."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from workspace root. Use '.' for root."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a directory (and parents) in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from workspace root."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_directory",
            "description": "Delete a directory and all its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from workspace root."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git",
            "description": "Run a git command in the workspace. Examples: 'status', 'add .', 'commit -m \"msg\"', 'push', 'pull', 'diff', 'log --oneline -10'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {"type": "string", "description": "Git subcommand and arguments (without 'git' prefix)."},
                },
                "required": ["args"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch content from a URL (GET request).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command in the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."},
                },
                "required": ["command"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Diff helper
# ---------------------------------------------------------------------------

def make_diff(path: str, old_content: str | None, new_content: str | None) -> tuple[str, int, int]:
    """Generate a unified diff string for display.
    Returns (diff_text, lines_added, lines_removed)."""
    old_lines = (old_content or "").splitlines(keepends=True)
    new_lines = (new_content or "").splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{path}", tofile=f"b/{path}",
        lineterm="",
    )
    diff_lines = list(diff)
    if not diff_lines:
        return "", 0, 0

    result = []
    added = 0
    removed = 0
    for line in diff_lines:
        if line.startswith("---") or line.startswith("+++"):
            result.append(line.rstrip())
        elif line.startswith("@@"):
            result.append(line.rstrip())
        elif line.startswith("+"):
            result.append(line.rstrip())
            added += 1
        elif line.startswith("-"):
            result.append(line.rstrip())
            removed += 1
    return "\n".join(result), added, removed


# ---------------------------------------------------------------------------
# Tool execution — returns (result_for_ai, diff_for_display_or_None)
# ---------------------------------------------------------------------------

def _resolve(workspace: str, path: str) -> str:
    """Resolve a relative path within the workspace, preventing escape."""
    full = os.path.normpath(os.path.join(workspace, path))
    ws_norm = os.path.normpath(workspace)
    if not full.startswith(ws_norm):
        raise ValueError(f"Path escapes workspace: {path}")
    return full


def _diff_stats(added: int, removed: int) -> str:
    parts = []
    if removed:
        parts.append(f"-{removed}")
    if added:
        parts.append(f"+{added}")
    return ", ".join(parts) + " lines" if parts else ""


def execute(workspace: str, name: str, arguments: dict) -> tuple[str, str | None]:
    """
    Execute a tool.
    Returns (result_text, diff_text_or_None).
    diff_text is set for file-modifying operations.
    """
    try:
        match name:
            case "read_file":
                fpath = _resolve(workspace, arguments["path"])
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    content = f.read()
                if len(content) > 50000:
                    content = content[:50000] + "\n... (truncated)"
                return content, None

            case "write_file":
                fpath = _resolve(workspace, arguments["path"])
                rel = arguments["path"]
                old = None
                if os.path.exists(fpath):
                    with open(fpath, encoding="utf-8", errors="replace") as f:
                        old = f.read()
                new = arguments["content"]
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(new)
                diff, added, removed = make_diff(rel, old, new)
                stats = _diff_stats(added, removed)
                result = f"Written: {rel} ({stats})" if stats else f"Written: {rel}"
                return result, diff or None

            case "edit_file":
                fpath = _resolve(workspace, arguments["path"])
                rel = arguments["path"]
                with open(fpath, encoding="utf-8") as f:
                    old = f.read()
                old_str = arguments["old_string"]
                if old_str not in old:
                    return f"Error: old_string not found in {rel}", None
                new = old.replace(old_str, arguments["new_string"], 1)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(new)
                diff, added, removed = make_diff(rel, old, new)
                stats = _diff_stats(added, removed)
                result = f"Edited: {rel} ({stats})" if stats else f"Edited: {rel}"
                return result, diff or None

            case "delete_file":
                fpath = _resolve(workspace, arguments["path"])
                rel = arguments["path"]
                old = None
                if os.path.exists(fpath):
                    with open(fpath, encoding="utf-8", errors="replace") as f:
                        old = f.read()
                os.remove(fpath)
                diff, added, removed = make_diff(rel, old, None)
                stats = _diff_stats(added, removed)
                result = f"Deleted: {rel} ({stats})" if stats else f"Deleted: {rel}"
                return result, diff or None

            case "list_directory":
                dpath = _resolve(workspace, arguments["path"])
                entries = sorted(os.listdir(dpath))
                result = []
                for e in entries:
                    full = os.path.join(dpath, e)
                    prefix = "[dir] " if os.path.isdir(full) else "      "
                    result.append(f"{prefix}{e}")
                return ("\n".join(result) if result else "(empty directory)"), None

            case "create_directory":
                dpath = _resolve(workspace, arguments["path"])
                os.makedirs(dpath, exist_ok=True)
                return f"Created: {arguments['path']}", None

            case "delete_directory":
                dpath = _resolve(workspace, arguments["path"])
                shutil.rmtree(dpath)
                return f"Deleted: {arguments['path']}", None

            case "git":
                args = arguments["args"]
                result = subprocess.run(
                    f"git {args}",
                    shell=True, cwd=workspace,
                    capture_output=True, text=True, timeout=60,
                )
                output = result.stdout + result.stderr
                if len(output) > 10000:
                    output = output[:10000] + "\n... (truncated)"
                return (output.strip() or "(no output)"), None

            case "web_fetch":
                resp = httpx.get(arguments["url"], timeout=30, follow_redirects=True)
                content = resp.text
                if len(content) > 30000:
                    content = content[:30000] + "\n... (truncated)"
                return f"HTTP {resp.status_code}\n\n{content}", None

            case "shell":
                result = subprocess.run(
                    arguments["command"],
                    shell=True, cwd=workspace,
                    capture_output=True, text=True, timeout=120,
                )
                output = result.stdout + result.stderr
                if len(output) > 10000:
                    output = output[:10000] + "\n... (truncated)"
                return (output.strip() or "(no output)"), None

            case _:
                return f"Unknown tool: {name}", None

    except Exception as e:
        return f"Error: {type(e).__name__}: {e}", None
