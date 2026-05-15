# Cozter

A chat-surface that wraps coding-agent CLIs (codex, claude_code, copilot)
and a local llama-server HTTP backend, exposing them through Telegram,
Slack, or a plain terminal. One bot process, multiple workspaces,
per-workspace settings, durable sessions with automatic compaction, and
a drop-in plugin system that works across every backend.

## What it gives you

- **Four interchangeable agent backends**, picked per workspace:
  - `codex` — OpenAI's CLI (`codex exec`)
  - `claude_code` — Anthropic's CLI (`claude --print`)
  - `copilot` — GitHub's CLI
  - `llama` — any OpenAI-compatible HTTP server (llama-server, LM Studio,
    Mistral API, etc.); the agent loop runs in-process and uses the
    typed tools in `agent_tools/`
- **Three chat surfaces**, selected at launch:
  - Telegram (`python -m Cozter`)
  - Slack (Socket Mode; same launcher, set `slack_bot_token` in config)
  - CLI (`python -m Cozter -cli`) — the terminal becomes the chat
- **Per-workspace state**, scoped to `<workspace>/.cozter/`:
  sessions, compaction history, agent choice, model, permission level,
  reasoning effort, summary backend, colony (long-term memory)
- **Durable sessions with two-layer memory**: each conversation
  auto-compacts every N turns into a scratch summary plus a persistent
  long-term-memory list, both injected into every subsequent turn
- **Auto-update**: the bot polls origin, fast-forward-pulls, and exits
  for the supervisor (`systemd` or its equivalent) to restart it

## Quick start

```bash
git clone https://github.com/limztudio/Cozter.git
python -m Cozter
# first run writes Cozter/.config/config.json and exits;
# fill in tokens and run again
```

`requirements.txt` is auto-installed on startup. Run `python -m Cozter -cli`
if you don't have a Telegram or Slack token and want to try it in a
terminal.

## Configuration

`Cozter/.config/config.json` (created on first run; the example layout
lives in `.config/config.example.json`):

```json
{
  "telegram_bot_tokens": ["..."],
  "user_ids": [123456789],

  "slack_bot_token": "",
  "slack_app_token": "",
  "slack_channel_ids": [],

  "llama_server_url": "http://127.0.0.1:8080",
  "llama_max_agent_turns": 60,
  "llama_tool_repeat_limit": 3,
  "llama_socket_timeout": 1800,

  "update_check_interval": 10,
  "recent_workspace_limit": 10
}
```

Either `telegram_bot_tokens` + `user_ids` *or* `slack_bot_token` +
`slack_app_token` + `slack_channel_ids` must be populated — not both.
The CLI surface needs neither.

## Workspace concept

A *workspace* is just a directory on disk. The bot edits files in it,
runs commands in it, and stores per-workspace state under
`<workspace>/.cozter/`:

- `sessions/` — conversation history files (one per session)
- `last_session.json` — per-user pointer to the session each user was
  last writing into; consulted on every turn (and across bot restarts)
  so conversations resume in place instead of being re-routed
- `settings.json` — chosen agent, model, permission, reasoning effort,
  summary backend, summary model, compact interval

Workspaces are recorded globally in `Cozter/.config/workspaces.json`
(per-user current pick + the recent-workspaces list).

## Commands

All chat surfaces speak the same command set:

| Command | What it does |
|---|---|
| `/new <path>` | Create a new workspace directory and select it |
| `/open <path-or-number>` | Switch to an existing workspace |
| `/agent` | Pick the agent backend (codex / claude_code / copilot / llama) |
| `/model` | Pick the chat model for the current backend |
| `/summaryagent` | Pick the backend used for compaction / titling / routing |
| `/summarymodel` | Pick the model for the summary backend |
| `/permission` | full / auto / confirm / deny — how the agent treats tool calls |
| `/effort` | 0–100 reasoning effort; each backend maps to its native scale |
| `/compact` | Show compaction state / set interval / `/compact now` to force a pass |
| `/newsession` | Start a fresh session (next message will go into a new conversation) |
| `/colony` | Set the per-workspace colony-consolidation interval |
| `/refresh` | Drop the workspace's `.codex/` cache (use after an upgrade) |
| `/stop` | Cancel the running agent turn |
| `/inject <text>` | Add context to the running turn (the agent restarts with it) |
| `/reserve`, `/schedules` | Cron-style scheduled prompts |
| `/version`, `/cancel`, `/start` | Self-explanatory |

## Plugins

Drop a `.py` file into `agent_tools/plugins/` and every agent picks
it up on next restart. One file, two invocation paths:

- **HTTP backends** (`llama` and any future API backend) see plugins
  as typed tools in the chat-completions `tools` schema, alongside
  the 14 built-in tools in `agent_tools/builtin/`
- **CLI backends** (`codex`, `claude_code`, `copilot`) can't have
  external tools injected into their fixed toolkit. The bot
  instead lists each plugin in their prompt and tells the model to
  invoke it through the backend's own `bash` / `shell` tool as
  `python -m Cozter.agent_tools.plugins.<filename> '<JSON args>'`.

Plugin template:

```python
"""Plugin: <one-line description>."""
from __future__ import annotations
from ..base import AgentTool


class MyTool(AgentTool):
    name = "my_tool"
    description = "What this does, from the model's perspective."
    parameters = {
        "type": "object",
        "properties": {"thing": {"type": "string"}},
        "required": ["thing"],
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        return f"got: {args.get('thing')}"


if __name__ == "__main__":
    MyTool.run_as_script()
```

The `__main__` block at the bottom is what makes the same file work
as both a Python module (loaded by the bot for typed-tool use) and a
standalone script (invoked by CLI backends via `bash`). See
`agent_tools/plugins/README.md` and the shipped `current_time.py`
plugin.

## Reasoning effort

`/effort` accepts `0`–`100` and is stored per workspace. `0` means "no
override — server defaults apply"; `1`–`100` are explicit. Each backend
implements `convert_effort(percent) -> str | None` to map the percentage
to its own vocabulary:

| Backend | Bands | What gets sent at 100% |
|---|---|---|
| `codex` | 5 levels @ 20% each | `-c model_reasoning_effort=xhigh` |
| `llama` | 4 levels @ 25% each | `payload["reasoning_effort"] = "high"` |
| `claude_code` | binary @ 50% threshold | logged; no CLI flag exists |
| `copilot` | always `None` | nothing; no CLI flag exists |

The setting applies only to user-facing chat turns. Internal calls
(compaction, routing, titling, colony consolidation) skip the effort
parameter, so utility work stays cheap regardless of the workspace
setting.

## Architecture

```
Cozter/
├── __main__.py           entry point; sets PYTHONPATH; runs the bot
├── backends_bot/         chat surfaces (Telegram / Slack / CLI)
├── agent.py              orchestrator: builds prompt, runs backend, streams events
├── session.py            per-workspace conversation persistence
├── compaction.py         scratch-summary + long-term-memory rewriter
├── colony.py             cross-session long-term memory consolidation
├── router.py             session picker for first-turn-in-workspace (subsequent turns reuse last_session.json)
├── titling.py            auto-titles new sessions from their first turn
├── schedules.py          /reserve cron-style scheduled prompts
├── workspace.py          per-workspace settings (model, permission, effort, ...)
├── config.py             global config.json reader
├── updater.py            git fetch + restart loop
├── utils.py              shared helpers (atomic_write, drain_queue, ...)
│
├── backends_agent/       agent backends (one file per agent)
│   ├── base.py             abstract Backend; convert_effort, supports_typed_plugins
│   ├── codex.py            wraps `codex exec`
│   ├── claude_code.py      wraps `claude --print`
│   ├── copilot.py          wraps `copilot`
│   └── llama.py            in-process loop against OpenAI-compatible /v1/chat/completions
│
└── agent_tools/          tool surface for HTTP backends + plugin registry
    ├── base.py             AgentTool ABC; run_as_script; resolve_inside_workspace; html_to_text
    ├── builtin/            14 built-in tools (read_file, edit_file, glob, grep, bash, web_search, ...)
    └── plugins/            user drop-in zone (current_time.py shipped as a live plugin)
```

The agent loop in `agent.py:run()` is shared across backends. Each
`Backend.launch()` spawns the right subprocess (or the in-process llama
session); the orchestrator reads JSONL events from stdout, translates
them via `Backend.parse_event()` into `ChatEvent`s, and streams a
"Thinking..." status line to the user with the latest few tool actions.

## Auto-update

`updater.fetch_and_pull()` runs every `update_check_interval` seconds.
If `HEAD` changed (remote update *or* manual pull while the bot was
running), the bot installs any new `requirements.txt`, broadcasts a
"restarting" message, and exits. The init system (`systemd Restart=always`,
or any equivalent) brings it back.

## Reading order

If you want to understand the codebase, the high-leverage entry points
are:

1. `__main__.py` → `backends_bot/base.py` to see how a turn enters the system
2. `agent.py:run()` to see the orchestrator
3. `backends_agent/llama.py` for the full HTTP agent loop including
   the tool dispatcher
4. `agent_tools/__init__.py` for the auto-discovery and plugin
   bridging

The CLI-backend files (`codex.py`, `claude_code.py`, `copilot.py`) are
thin: each defines `launch()` (build argv, spawn subprocess) and
`parse_event()` (translate the CLI's JSONL events to `ChatEvent`s).
