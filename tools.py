"""Workspace tools that the AI can invoke."""

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
            "name": "rename_file",
            "description": "Rename or move a file within the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_path": {"type": "string", "description": "Current relative path."},
                    "new_path": {"type": "string", "description": "New relative path."},
                },
                "required": ["old_path", "new_path"],
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
# Tool execution
# ---------------------------------------------------------------------------

def _resolve(workspace: str, path: str) -> str:
    """Resolve a relative path within the workspace, preventing escape."""
    full = os.path.normpath(os.path.join(workspace, path))
    ws_norm = os.path.normpath(workspace)
    if not full.startswith(ws_norm):
        raise ValueError(f"Path escapes workspace: {path}")
    return full


def _line_stats(old: str | None, new: str | None) -> str:
    old_n = len((old or "").splitlines())
    new_n = len((new or "").splitlines())
    added = max(0, new_n - old_n)
    removed = max(0, old_n - new_n)
    parts = []
    if removed:
        parts.append(f"-{removed}")
    if added:
        parts.append(f"+{added}")
    return ", ".join(parts) + " lines" if parts else ""


def execute(workspace: str, name: str, arguments: dict) -> str:
    """Execute a tool and return the result text."""
    try:
        match name:
            case "read_file":
                fpath = _resolve(workspace, arguments["path"])
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    content = f.read()
                if len(content) > 50000:
                    content = content[:50000] + "\n... (truncated)"
                return content

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
                stats = _line_stats(old, new)
                return f"Written: {rel} ({stats})" if stats else f"Written: {rel}"

            case "edit_file":
                fpath = _resolve(workspace, arguments["path"])
                rel = arguments["path"]
                with open(fpath, encoding="utf-8") as f:
                    old = f.read()
                old_str = arguments["old_string"]
                if old_str not in old:
                    return f"Error: old_string not found in {rel}"
                new = old.replace(old_str, arguments["new_string"], 1)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(new)
                stats = _line_stats(old, new)
                return f"Edited: {rel} ({stats})" if stats else f"Edited: {rel}"

            case "delete_file":
                fpath = _resolve(workspace, arguments["path"])
                rel = arguments["path"]
                old = None
                if os.path.exists(fpath):
                    with open(fpath, encoding="utf-8", errors="replace") as f:
                        old = f.read()
                os.remove(fpath)
                stats = _line_stats(old, None)
                return f"Deleted: {rel} ({stats})" if stats else f"Deleted: {rel}"

            case "rename_file":
                old_fpath = _resolve(workspace, arguments["old_path"])
                new_fpath = _resolve(workspace, arguments["new_path"])
                os.makedirs(os.path.dirname(new_fpath), exist_ok=True)
                os.rename(old_fpath, new_fpath)
                return f"Renamed: {arguments['old_path']} -> {arguments['new_path']}"

            case "list_directory":
                dpath = _resolve(workspace, arguments["path"])
                entries = sorted(os.listdir(dpath))
                result = []
                for e in entries:
                    full = os.path.join(dpath, e)
                    prefix = "[dir] " if os.path.isdir(full) else "      "
                    result.append(f"{prefix}{e}")
                return "\n".join(result) if result else "(empty directory)"

            case "create_directory":
                dpath = _resolve(workspace, arguments["path"])
                os.makedirs(dpath, exist_ok=True)
                return f"Created: {arguments['path']}"

            case "delete_directory":
                dpath = _resolve(workspace, arguments["path"])
                shutil.rmtree(dpath)
                return f"Deleted: {arguments['path']}"

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
                return output.strip() or "(no output)"

            case "web_fetch":
                resp = httpx.get(arguments["url"], timeout=30, follow_redirects=True)
                content = resp.text
                if len(content) > 30000:
                    content = content[:30000] + "\n... (truncated)"
                return f"HTTP {resp.status_code}\n\n{content}"

            case "shell":
                result = subprocess.run(
                    arguments["command"],
                    shell=True, cwd=workspace,
                    capture_output=True, text=True, timeout=120,
                )
                output = result.stdout + result.stderr
                if len(output) > 10000:
                    output = output[:10000] + "\n... (truncated)"
                return output.strip() or "(no output)"

            case _:
                return f"Unknown tool: {name}"

    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
