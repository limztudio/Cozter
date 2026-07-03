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
  sessions, last-session pointers, compaction history, agent choice,
  model, permission level, reasoning effort, summary backend, colony
  (long-term memory), uploads, generated image attachments, and schedules
- **Durable sessions with two-layer memory**: each conversation
  auto-compacts every N turns into a scratch summary plus a persistent
  long-term-memory list, both injected into every subsequent turn
- **Persistent turn queues**: if a user sends more work while an agent
  turn is running, or while an update restart is pending, the messages
  are queued on disk and restored after restart
- **File flow in both directions**: chat uploads are saved into the
  workspace and text-like files are inlined into the next prompt; agent
  replies can upload workspace files or generated images back to chat
- **Recurring scheduled prompts**: `/reserve` queues prompts on selected
  weekdays and runs them in throwaway sessions so routine jobs do not
  pollute the user's active conversation
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
- Python package dependencies from `requirements.txt`:
  `python-telegram-bot`, `httpx`, `slack-bolt`, and `aiohttp`. The
  launcher installs them into the project-local `.venv` before importing
  the runtime.
- Optional external services:
  Telegram and Slack need their platform tokens; Signal also requires a
  separately installed and running `signal-cli` JSON-RPC daemon.

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
  "message_queue_size": 50,

  "extra_models": {}
}
```

Exactly one daemon chat surface must be populated: `telegram_bot_tokens`
+ `user_ids`, `slack_bot_token` + `slack_app_token` +
`slack_channel_ids`, or `signal_group_urls` + `signal_jsonrpc_socket`.
The CLI surface needs neither.

`recent_workspace_limit` controls how many paths `/open` shows.
`message_queue_size` caps each user's pending chat turns.

`extra_models` adds model IDs to a backend's `/model` and `/summarymodel`
pickers on top of its built-in list, keyed by backend name — for example
`{"codex": ["gpt-5.6"], "copilot": ["claude-opus-5.0"]}`. The built-in
lists in `backends_agent/` are a curated snapshot that goes stale as
providers ship new models; this lets you use a newer or private model
without editing source. Malformed entries are ignored. The queue is
persisted in `Cozter/.config/queue_<platform>.json`, so clean restarts,
auto-updates, and crash recovery do not drop already accepted messages.
Platform IDs are sanitized for those filenames; for example the CLI's
stable platform key `cli:local` is stored as `queue_cli_local.json`. CLI
mode does not read or create daemon config.

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
- `settings.json` — chosen agent, model, permission, interaction style,
  reasoning effort, summary backend, summary model, compact interval,
  colony interval, and context budget
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
| `/style` | collaborative / autonomous — whether the agent asks before big/ambiguous actions or runs full-auto |
| `/effort` | 0–100 reasoning effort; each backend maps to its native scale |
| `/compact [number]` | Show compaction state, or set messages between compactions |
| `/context [number]` | Show or set the per-turn context budget (characters of prepended history) |
| `/newsession` | Start a fresh session (next message will go into a new conversation) |
| `/colony [number\|now]` | Show memory state, set the consolidation interval, or run it now |
| `/refresh` | Drop the workspace's `.codex/` cache (use after an upgrade) |
| `/stop` | Cancel the running agent turn and clear queued work |
| `/inject <text>` | Add context to the running turn (the agent restarts with it) |
| `/reserve` | Create a recurring scheduled prompt |
| `/schedules` | List schedules and delete one by number |
| `/version` | Show the current git version and last commit date |
| `/cancel` | Cancel a picker/wizard, pending answer, running turn, or queued work |
| `/start` | Confirm the bot is running |

Most picker commands accept either the displayed number or the literal
name. `/open` also accepts a recent-workspace number directly as
`/open 2`.

Schedules are stored per workspace in `.cozter/schedules.json` and use
the host's local time. The scheduler checks every 30 seconds and records
`last_fired`, so a missed slot fires once after restart instead of being
lost. Scheduled prompts run through the same persistent queue as user
messages, but use a fresh ephemeral session that is deleted after the
turn; they do not append to the user's current conversation.

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
Generated images under `$CODEX_HOME/generated_images` (or
`~/.codex/generated_images` when `CODEX_HOME` is unset) and any
directories listed in `COZTER_ATTACHMENT_ROOTS` are also accepted. Cozter
copies those images into `.cozter/generated_images/` before upload so
chat platforms never receive arbitrary external paths. At the end of a
run, Cozter also snapshots newly created or modified image files in the
workspace and trusted generated-image roots and attaches them unless the
agent already referenced them explicitly. Replies can end with
`[[await]]` when the agent needs a user decision; the marker is stripped
and that user's queued work pauses until the next message arrives.

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
| `copilot` | `copilot --output-format json` | `auto` | `claude-haiku-4.5` |
| `llama` | OpenAI-compatible `/v1/chat/completions` | `auto` | `auto` |

The CLI-backend model lists are a hand-maintained snapshot; add newer or
private models through `extra_models` in config (see Configuration) rather
than editing source. The `llama` backend instead discovers models live
from its server.

Permission modes are best-effort across third-party CLIs, because a chat
bot can't surface a per-tool-call approval dialog. `codex` maps all four
modes to native sandbox/approval flags. `llama` runs the loop in-process,
so `deny` exposes no tools and `confirm` exposes read-only tools only —
writes and shell are withheld, and blocked as a backstop in
`execute_tool`. `claude_code` uses plan mode for `deny` but falls back to
non-interactive for `confirm`; `copilot` has no usable interactive
approval flow, so non-`full` modes run with its non-blocking tool flag.
Stricter intents a backend can't enforce are logged. For ask-before-acting
behavior on any backend, use `/style collaborative` — it pauses the turn
(via `[[await]]`) for your reply instead of relying on the CLI's own
approval flow.

The `llama` model picker queries `llama_server_url/v1/models` and falls
back to `auto` if the server is down or returns no model IDs. The
`copilot` backend keeps prompts under the Windows command-line limit by
dropping the oldest composed context when a prompt exceeds its cap; the
current user message is kept at the tail.

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

## Interaction style

`/style` chooses how collaborative the agent is on chat turns, stored per
workspace:

- `collaborative` (default) — when a request is ambiguous or before a
  large, destructive, or hard-to-reverse action, the agent asks a short
  question and ends with `[[await]]`, pausing the queue until you reply.
  Small, reversible choices are made without asking. This is a
  backend-agnostic prompt policy, so it steers every backend (codex,
  copilot, claude_code, llama) the same way — not just the CLIs that ask
  on their own.
- `autonomous` — the agent decides and proceeds without asking, closer to
  a full-auto run.

Scheduled `/reserve` turns run in throwaway sessions that cannot pause for
a reply, so they always use the autonomous policy regardless of this
setting.

## Architecture

```
Cozter/
├── __init__.py           package marker and version
├── __main__.py           entry point; sets PYTHONPATH; runs the bot
├── requirements.txt      Python runtime dependencies installed into .venv
├── .config/              runtime config dir; only config.example.json is tracked
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
├── tests/                unittest coverage for state/config, backend defaults, queues, and tools
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

## Source inventory

The tracked workspace is intentionally flat and small. A complete source
audit should use `git ls-files` so hidden tracked files, especially
`.config/config.example.json`, are included even though `.config/*` is
ignored for local secrets and runtime queues.

- Package entry and runtime setup: `__init__.py`, `__main__.py`,
  `config.py`, `updater.py`, and `utils.py`
- Conversation, memory, and workspace state: `agent.py`, `workspace.py`,
  `session.py`, `router.py`, `titling.py`, `compaction.py`,
  `colony.py`, and `schedules.py`
- Chat-platform adapters: `backends_bot/base.py`, `cli.py`,
  `telegram.py`, `slack.py`, and `signal.py`
- Agent adapters: `backends_agent/base.py`, `_http_proc.py`,
  `codex.py`, `claude_code.py`, `copilot.py`, and `llama.py`
- Agent tool surface: `agent_tools/__init__.py`, `agent_tools/base.py`,
  the 14 files under `agent_tools/builtin/`, and user plugins plus their
  README under `agent_tools/plugins/`
- Project metadata and docs: `requirements.txt`,
  `.config/config.example.json`, `.gitignore`, and this README
- Tests: `tests/test_agent_tools.py`, `tests/test_backends_agent.py`,
  `tests/test_state_fallbacks.py`, and `tests/test_utils.py`

The normal working checkout may also contain ignored runtime state such as
`.venv/`, `.cozter/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`,
`.log/`, and local assistant/editor directories. Treat those as local
machine state unless a file is deliberately being promoted into tracked
source.

The agent loop in `agent.py:run()` is shared across backends. Each
`Backend.launch()` spawns the right subprocess (or the in-process llama
session); the orchestrator reads JSONL events from stdout, translates
them via `Backend.parse_event()` into `ChatEvent`s, and streams a
"Thinking..." status line to the user with the latest few tool actions.

## Repository state

Tracked source is intentionally small: the top-level runtime modules,
`backends_bot/`, `backends_agent/`, `agent_tools/`, `tests/`,
`requirements.txt`, this README, and `.config/config.example.json`.
Everything else created by a running bot is local state.

Do not commit these runtime artifacts:

- `.config/config.json`, `.config/workspaces.json`, and
  `.config/queue_<platform>.json` - local tokens, workspace selections,
  and persisted pending messages. Platform IDs are sanitized for queue
  filenames, so a runtime key like `cli:local` becomes a filesystem-safe
  `queue_cli_local.json`.
- `.cozter/` — sessions, workspace settings, colony memory, schedules,
  uploads, and generated images; this directory can appear at the repo
  root when Cozter is used on its own checkout
- `.log/` — rotating runtime logs and crash reports
- `.venv/`, `__pycache__/`, `.ruff_cache/`, coverage output, and build
  artifacts
- Local assistant/editor directories, such as `.claude/`, unless you
  intentionally add shared project settings

The shipped `.gitignore` keeps Cozter runtime files and common Python
artifacts out of normal commits while still tracking
`.config/config.example.json` so new installs have a template. Local
assistant/editor directories may rely on your global excludes; review
them before staging. If you add a new user-facing plugin, place it under
`agent_tools/plugins/` and commit it intentionally; files whose names
start with `_` are ignored by the plugin loader but are not ignored by
git.

Useful audit commands before documentation or release commits:

```bash
git pull --ff-only
git ls-files
find . -maxdepth 2 -type d -not -path './.git*' -print | sort
PYTHONPATH=.. .venv/bin/python -m unittest discover -s tests
git status --short
```

## Auto-update

`updater.fetch_and_pull()` runs every `update_check_interval` seconds.
Update checks wait for active turns to finish, pause new intake briefly,
and queue any messages that arrive during a pending restart. If `HEAD`
changed (remote update *or* manual pull while the bot was running), the
bot installs any new `requirements.txt`, broadcasts a "restarting"
message, and exits. The init system (`systemd Restart=always`, or any
equivalent) brings daemon mode back. CLI mode uses an outer respawner
process and relaunches itself in the same terminal. Persisted queues
resume after either path starts again.

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
covers malformed state/config fallbacks, persistent queue restoration,
schedule parsing, backend model defaults, subprocess-drain behavior,
agent-tool helpers, and built-in discovery/edit tools.

```bash
cd ..
Cozter/.venv/bin/python -m unittest discover -s Cozter/tests
```

From inside `Cozter/`:

```bash
PYTHONPATH=.. .venv/bin/python -m unittest discover -s tests
```

Before committing, run the tests and check `git status --short` for only
intentional source or documentation edits. Runtime JSON, logs, sessions,
virtualenv files, and caches should stay local.
