# Cozter

A chat-surface that wraps coding-agent CLIs (codex, claude_code, copilot)
and a local llama-server HTTP backend, exposing them through Telegram,
Slack, Signal, or a plain terminal. One bot process, multiple workspaces,
per-workspace settings, durable sessions with automatic compaction,
persistent turn queues, file attachments, and a drop-in plugin system
that works across every backend.

## What it gives you

- **Four interchangeable agent backends**, picked per workspace:
  - `codex` — OpenAI's CLI (`codex exec`)
  - `claude_code` — Anthropic's CLI (`claude --print`)
  - `copilot` — GitHub's CLI
  - `llama` — any OpenAI-compatible HTTP server (llama-server, LM Studio,
    Mistral API, etc.); the agent loop runs in-process and uses the
    typed tools in `agent_tools/`
- **Four chat surfaces**, selected at launch:
  - Telegram (`python -m Cozter`)
  - Slack (Socket Mode; same launcher, set `slack_bot_token` in config)
  - Signal (same launcher, set `signal_group_urls` and the daemon socket)
  - CLI (`python -m Cozter -cli`) — the terminal becomes the chat
- **Per-workspace state**, scoped to `<workspace>/.cozter/`:
  sessions, compaction history, agent choice, model, permission level,
  reasoning effort, summary backend, colony (long-term memory), uploads,
  generated image attachments, and schedules
- **Durable sessions with two-layer memory**: each conversation
  auto-compacts every N turns into a scratch summary plus a persistent
  long-term-memory list, both injected into every subsequent turn
- **Persistent turn queues**: if a user sends more work while an agent
  turn is running, or while an update restart is pending, the messages
  are queued on disk and restored after restart
- **File flow in both directions**: chat uploads are saved into the
  workspace and text-like files are inlined into the next prompt; agent
  replies can upload workspace files or generated images back to chat
- **Auto-update**: the bot polls origin, fast-forward-pulls, and exits
  for the supervisor (`systemd` or its equivalent) to restart it

## Quick start

```bash
git clone https://github.com/limztudio/Cozter.git
cd Cozter
python __main__.py -cli
```

That starts the local terminal chat surface without requiring bot tokens.
From the parent directory you can run the package form instead:

```bash
python -m Cozter -cli
```

For Telegram, Slack, or Signal daemon mode, run without `-cli`:

```bash
python -m Cozter
# first run writes Cozter/.config/config.json and exits;
# fill in tokens and run again
```

On startup, Cozter re-execs through a project-local `.venv` when needed
and installs `requirements.txt` before importing the bot runtime.

CLI mode intentionally skips daemon configuration: it does not read or
create `.config/config.json`, and it uses the stable local platform key
`cli:local` for workspace/session state. Daemon mode (`python -m Cozter`
without `-cli`) validates `.config/config.json` before any platform
starts.

## Requirements

- Python 3.10+ (the codebase uses modern type syntax)
- One agent backend CLI or server:
  `codex`, `claude`, `copilot`, or an OpenAI-compatible HTTP server for
  the `llama` backend
- Optional chat-surface dependencies:
  Telegram uses `python-telegram-bot`, Slack uses `slack-bolt`, and
  Signal requires a separately installed `signal-cli` daemon

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

  "signal_group_urls": [],
  "signal_jsonrpc_socket": "",

  "llama_server_url": "http://127.0.0.1:8080",
  "llama_max_agent_turns": 60,
  "llama_tool_repeat_limit": 3,
  "llama_socket_timeout": 1800,

  "update_check_interval": 10,
  "recent_workspace_limit": 10,
  "message_queue_size": 50
}
```

Exactly one daemon chat surface must be populated: `telegram_bot_tokens`
+ `user_ids`, `slack_bot_token` + `slack_app_token` +
`slack_channel_ids`, or `signal_group_urls` + `signal_jsonrpc_socket`.
The CLI surface needs neither.

`message_queue_size` caps each user's pending chat turns. The queue is
persisted in `Cozter/.config/queue_<platform>.json`, so clean restarts,
auto-updates, and crash recovery do not drop already accepted messages.
CLI mode uses its own stable platform key (`cli:local`) and does not read
or create daemon config.

For Signal, `signal-cli` must already be installed, registered, and
running as a JSON-RPC daemon. Each invite URL in `signal_group_urls` is
resolved from the daemon's known groups or joined at startup. Set
`signal_jsonrpc_socket` to the Unix socket exposed by that daemon, for
example `/run/signal-cli/socket`. Cozter only connects to the socket; run
and restart the `signal-cli daemon` from a service manager such as
systemd.

The Signal phone number, socket path, and `signal-cli` binary location are
owned by that daemon/service setup rather than by Cozter's
`.config/config.json`. A local daemon config or service environment might
carry fields like:

```json
{
  "phone_number": "+10000000000",
  "socket_path": "/run/signal-cli/socket",
  "signal_cli_path": "signal-cli"
}
```

## Workspace concept

A *workspace* is just a directory on disk. The bot edits files in it,
runs commands in it, and stores per-workspace state under
`<workspace>/.cozter/`:

- `sessions/` — conversation history files (one per session)
- `last_session.json` — per-user pointer to the session each user was
  last writing into; consulted on every turn (and across bot restarts)
  so conversations resume in place instead of being re-routed
- `settings.json` — chosen agent, model, permission, reasoning effort,
  summary backend, summary model, compact interval, and colony interval
- `colony.json` — workspace-wide long-term memory consolidated across
  sessions
- `schedules.json` — recurring `/reserve` prompts and their last-fired
  timestamps
- `uploads/` — files received from Telegram, Slack, or Signal
- `generated_images/` — external generated images copied into the
  workspace before upload back to chat

Workspaces are recorded globally in `Cozter/.config/workspaces.json`
(per-user current pick + the recent-workspaces list). Platform turn
queues live beside it as `queue_<platform>.json`.

The global runtime files are deliberately small JSON documents:

- `.config/config.json` — daemon chat-surface and backend settings
- `.config/workspaces.json` — current/recent workspace selections per
  user and platform
- `.config/queue_<platform>.json` — persisted pending turns so accepted
  work survives restarts, crashes, and auto-updates

The session router is only used when there is no valid
`last_session.json` entry, such as a new workspace, a deleted session, or
after `/newsession`. Otherwise each user continues the same session across
bot restarts and platform reconnects.

## Commands

All chat surfaces speak the same command set:

| Command | What it does |
|---|---|
| `/new` | Prompt for a new workspace directory, create it, and select it |
| `/open [path-or-number]` | Switch to an existing workspace |
| `/agent` | Pick the agent backend (codex / claude_code / copilot / llama) |
| `/model` | Pick the chat model for the current backend |
| `/summaryagent` | Pick the backend used for compaction / titling / routing |
| `/summarymodel` | Pick the model for the summary backend |
| `/permission` | full / auto / confirm / deny — how the agent treats tool calls |
| `/effort` | 0–100 reasoning effort; each backend maps to its native scale |
| `/compact [number]` | Show compaction state, or set messages between compactions |
| `/newsession` | Start a fresh session (next message will go into a new conversation) |
| `/colony [number\|now]` | Show memory state, set the consolidation interval, or run it now |
| `/refresh` | Drop the workspace's `.codex/` cache (use after an upgrade) |
| `/stop` | Cancel the running agent turn and clear queued work |
| `/inject <text>` | Add context to the running turn (the agent restarts with it) |
| `/reserve` | Create a recurring scheduled prompt |
| `/schedules` | List schedules and delete one by number |
| `/version`, `/cancel`, `/start` | Self-explanatory |

Most picker commands accept either the displayed number or the literal
name. `/open` also accepts a recent-workspace number directly as
`/open 2`.

Schedules are stored per workspace in `.cozter/schedules.json`. The
scheduler checks every 30 seconds and records `last_fired`, so a missed
slot fires once after restart instead of being lost. Scheduled prompts run
through the same persistent queue as user messages, but use a fresh
ephemeral session that is deleted after the turn; they do not append to
the user's current conversation.

## Files and attachments

Telegram, Slack, and Signal uploads are copied into
`<workspace>/.cozter/uploads/` before the agent sees them. The generated
prompt includes the saved relative path, and text-like files up to
50,000 characters are inlined directly into the prompt. Larger text files
and binary files are referenced by path so the selected backend can inspect
them with its normal tools.

Agents can attach files back to chat by emitting a line like:

```text
[[attach: path/inside/workspace.png]]
```

The path may be relative to the workspace or an absolute path inside it.
Generated images under `$CODEX_HOME/generated_images` and any directories
listed in `COZTER_ATTACHMENT_ROOTS` are also accepted; Cozter copies those
images into `.cozter/generated_images/` before uploading so chat platforms
never receive arbitrary external paths. Replies can end with `[[await]]`
when the agent needs a user decision; the marker is stripped and that
user's queued work pauses until the next message arrives.

## Plugins

The built-in HTTP-toolkit includes filesystem, shell, search, and fetch
tools: `bash`, `read_file`, `write_file`, `edit_file`, `multi_edit`,
`delete_file`, `copy_file`, `move_file`, `make_dir`, `list_dir`, `glob`,
`grep`, `web_search`, and `web_fetch`.

Drop a `.py` file into `agent_tools/plugins/` and every agent picks it up
on next restart. Files whose names start with `_` are skipped, which is
useful for disabled examples or local scratch tools. One file, two
invocation paths:

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

The current plugin can also be run directly from the parent directory:

```bash
Cozter/.venv/bin/python -m Cozter.agent_tools.plugins.current_time '{"timezone":"Asia/Seoul"}'
```

HTTP-backend tool results are capped before they are fed back into the
model, keeping accidental huge outputs from consuming the whole context.
CLI backends rely on their own bundled shell tool for plugin execution, so
the plugin prelude only exposes how to call the extra tools; it does not
change the CLI's native tool sandbox.

## Backend behavior

Each backend defines its own model list and permission mapping in
`backends_agent/`:

| Backend | Launch path | Default chat model | Default summary model |
|---|---|---|---|
| `codex` | `codex exec --json` | `gpt-5.5` | `gpt-5.4-mini` |
| `claude_code` | `claude --print --output-format stream-json` | `default` | `haiku` |
| `copilot` | `copilot --output-format json` | `claude-sonnet-4.6` | `claude-haiku-4.5` |
| `llama` | OpenAI-compatible `/v1/chat/completions` | `auto` | `auto` |

Permission modes are best-effort across third-party CLIs. `codex` maps
all four modes to native sandbox/approval flags. `llama` disables typed
tools in `deny` mode. `claude_code` uses plan mode for `deny`, but cannot
prompt interactively for `confirm` in chat mode. `copilot` has no usable
interactive approval flow here, so non-`full` modes run with its
non-blocking tool flag and stricter intents are logged.

## Reasoning effort

`/effort` accepts `0`–`100` and is stored per workspace. `0` means "no
override — server defaults apply"; `1`–`100` are explicit. Each backend
implements `convert_effort(percent) -> str | None` to map the percentage
to its own vocabulary:

| Backend | Bands | What gets sent at 100% |
|---|---|---|
| `codex` | 5 levels @ 20% each | `-c model_reasoning_effort=xhigh` |
| `llama` | 4 levels @ 25% each | `payload["reasoning_effort"] = "high"` |
| `claude_code` | 5 levels @ 20% each | `--effort max` |
| `copilot` | 5 levels @ 20% each | `--effort max` |

The setting applies only to user-facing chat turns. Internal calls
(compaction, routing, titling, colony consolidation) skip the effort
parameter, so utility work stays cheap regardless of the workspace
setting.

## Architecture

```
Cozter/
├── __main__.py           entry point; sets PYTHONPATH; runs the bot
├── .config/              config, workspace index, and persistent queues
├── backends_bot/         chat surfaces (Telegram / Slack / Signal / CLI)
├── agent.py              orchestrator: builds prompt, runs backend, streams events and attachments
├── session.py            per-workspace conversation persistence
├── compaction.py         scratch-summary + long-term-memory rewriter
├── colony.py             cross-session long-term memory consolidation
├── router.py             session picker for first-turn-in-workspace (subsequent turns reuse last_session.json)
├── titling.py            auto-titles new sessions from their first turn
├── schedules.py          /reserve cron-style scheduled prompts
├── workspace.py          per-workspace settings (model, permission, effort, ...)
├── config.py             global .config/config.json reader
├── updater.py            git fetch + restart loop
├── utils.py              shared helpers (atomic_write, drain_queue, ...)
├── tests/                unittest coverage for state/config fallbacks and agent tools
├── .config/config.example.json
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
Update checks wait for active turns to finish, pause new intake briefly,
and queue any messages that arrive during a pending restart. If `HEAD`
changed (remote update *or* manual pull while the bot was running), the
bot installs any new `requirements.txt`, broadcasts a "restarting"
message, and exits. The init system (`systemd Restart=always`, or any
equivalent) brings it back, and persisted queues resume after startup.

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

## Development checks

Run the current unit tests from the parent directory, or set
`PYTHONPATH` to the parent when running inside the repository. Discovery
covers the state/config fallback tests plus the agent-tool helper and
built-in edit-tool tests.

```bash
cd ..
Cozter/.venv/bin/python -m unittest discover -s Cozter/tests
```

From inside `Cozter/`:

```bash
PYTHONPATH=.. .venv/bin/python -m unittest discover -s tests
```
