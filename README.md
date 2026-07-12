# Cozter

A chat-surface that wraps coding-agent CLIs (codex, claude_code, copilot)
and OpenAI-compatible HTTP backends (local llama-server and Z.ai), exposing
them through Telegram, Slack, Signal, or a plain terminal. One bot process,
multiple workspaces, per-workspace settings, durable sessions with
automatic compaction, persistent turn queues, file attachments, and a
drop-in plugin system that works across every backend.

## What it gives you

- **Five interchangeable agent backends**, picked per workspace:
  - `codex` — OpenAI's CLI (`codex exec`)
  - `claude_code` — Anthropic's CLI (`claude --print`)
  - `copilot` — GitHub's CLI
  - `llama` — any OpenAI-compatible HTTP server (llama-server, LM Studio,
    Mistral API, etc.); the agent loop runs in-process and uses the
    typed tools in `agent_tools/`
  - `zai` — Z.ai's cloud API (Zhipu GLM models: `glm-5.2`, `glm-5.1`, …);
    OpenAI-compatible, so it shares the in-process loop — set `zai_api_key`
    in config
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
- **Auto-update**: the bot polls origin, fast-forward-pulls only when the
  checkout is clean and not locally ahead, then exits for the supervisor
  (`systemd` or its equivalent) to restart it

## Quick start

```bash
git clone https://gitlab.com/mgneh/cozter.git
cd Cozter
python __main__.py -cli
```

The primary remote is **GitLab** (`git@gitlab.com:mgneh/cozter.git`). GitHub
(`github.com/limztudio/Cozter`) is kept as a read-only mirror; pull from
GitLab to stay current.

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
- One agent backend CLI, server, or API key:
  `codex`, `claude`, `copilot`, an OpenAI-compatible HTTP server for the
  `llama` backend, or Z.ai credentials for the `zai` backend
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
  "llama_max_retries": 2,

  "zai_api_key": "",
  "zai_base_url": "https://api.z.ai/api/paas/v4",
  "zai_socket_timeout": 300,
  "zai_max_retries": 2,

  "tool_timeout": 120,
  "update_idle_timeout": 1200,
  "dump_traceback_interval": 0,
  "update_check_interval": 10,
  "recent_workspace_limit": 10,
  "message_queue_size": 50,

  "extra_models": {},
  "max_permission": "full",
  "show_usage": true
}
```

Exactly one daemon chat surface must be populated: `telegram_bot_tokens`
+ `user_ids`, `slack_bot_token` + `slack_app_token` +
`slack_channel_ids`, or `signal_group_urls` + `signal_jsonrpc_socket`.
The CLI surface needs neither.

`recent_workspace_limit` controls how many paths `/open` shows.
`message_queue_size` caps each user's pending chat turns.

Agent turns do not have a wall-clock timeout; long-running work is
allowed to finish. `tool_timeout` (default 120s) still caps each
individual tool call for HTTP backends, so one wedged plugin call cannot
block the agent loop indefinitely. `update_idle_timeout` (default 1200s)
controls how often the auto-update loop dumps diagnostics while waiting
for active turns; it keeps waiting instead of restarting through active
work. `dump_traceback_interval` (default 0) enables optional periodic
thread dumps when set above zero; on-demand `SIGUSR1` diagnostics remain
available either way.

`extra_models` adds model IDs to a backend's `/model` and `/summarymodel`
pickers on top of its built-in list, keyed by backend name — for example
`{"codex": ["my-private-codex-model"], "copilot": ["claude-opus-5.0"]}`. The built-in
lists in `backends_agent/` are a curated snapshot that goes stale as
providers ship new models; this lets you use a newer or private model
without editing source. Malformed entries are ignored.

`llama_max_retries` (default 2) is how many times a transient llama HTTP
failure — a dropped connection, a read timeout, or an HTTP 429/5xx — is
retried with exponential backoff before the turn fails. Set it to `0` to
disable retries. Only the `llama` backend uses it; the CLI backends have
their own process-level behavior.

`zai_api_key` enables the `zai` backend (Z.ai / Zhipu GLM) — get one from
your Z.ai account and paste it here. `zai_base_url` defaults to
`https://api.z.ai/api/paas/v4` (already includes the version, so only
`/chat/completions` is appended); override it for a regional endpoint.
`zai_socket_timeout` (default 300s) and `zai_max_retries` (default 2)
mirror the llama knobs for the cloud call. Select `zai` with `/agent`, pick
a model with `/model` (default `glm-5.2`), and add private or regional GLM
ids via `extra_models` (`{"zai": ["glm-…"]}`). Long z.ai coding turns
automatically continue into another tool-enabled segment when Cozter's
internal tool-call segment limit is reached, instead of stopping for a
manual "continue".

`max_permission` (default `full`) caps the highest `/permission` mode any
workspace may use, bot-wide, in privilege order `deny < confirm < auto <
full`. Since `full` bypasses the sandbox (arbitrary code execution) for
anyone on the `user_ids` allowlist, set this to `auto` to keep every
workspace sandboxed, or `deny` for a read-only bot. `/permission` rejects
a higher mode, and an already-stored higher value is clamped down.

`show_usage` (default `true`) appends a compact per-turn token/cost footer
(e.g. `📊 12.5k in · 28 out · $0.01`) after each reply, for backends that
report usage — `codex` (`turn.completed`) and `claude_code` (`result`).
Other backends stay silent. Set it to `false` to suppress the footer.

Pending chat turns are persisted in
`Cozter/.config/queue_<platform>.json`, so clean restarts, auto-updates,
and crash recovery do not drop already accepted messages. Platform IDs
are sanitized for those filenames; for example the CLI's stable platform
key `cli:local` is stored as `queue_cli_local.json`. CLI mode does not
read or create daemon config.

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
| `/agent` | Pick the agent backend (codex / claude_code / copilot / llama / zai) |
| `/model` | Pick the chat model for the current backend |
| `/summaryagent` | Pick the backend used for compaction / titling / routing |
| `/summarymodel` | Pick the model for the summary backend |
| `/permission` | full / auto / confirm / deny — how the agent treats tool calls |
| `/style` | collaborative / autonomous — whether the agent asks before big/ambiguous actions or runs full-auto |
| `/effort` | 0–100 reasoning effort; each backend maps to its native scale |
| `/compact [number]` | Show compaction state, or set messages between compactions |
| `/context [number]` | Show or set the per-turn context budget (characters of prepended history) |
| `/newsession` | Start a fresh session (next message will go into a new conversation) |
| `/sessions [number\|name]` | List this workspace's sessions, or switch to one |
| `/colony [number\|now]` | Show memory state, set the consolidation interval, or run it now |
| `/refresh` | Drop the workspace's `.codex/` cache (use after an upgrade) |
| `/stop` | Cancel the running agent turn and clear queued work |
| `/inject <text>` | Add context to the running turn (the agent restarts with it) |
| `/reserve` | Create a recurring scheduled prompt |
| `/schedules` | List schedules and delete one by number |
| `/version` | Show the current git version and last commit date |
| `/doctor` | Check each backend's readiness (CLI on PATH / HTTP backend configured or reachable) |
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
turn; they do not append to the user's current conversation. They run
autonomously and can continue even while normal chat work is paused
waiting for a collaborative `[[await]]` answer.

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
`apply_patch`, `delete_file`, `copy_file`, `move_file`, `make_dir`,
`list_dir`, `tree`, `glob`, `grep`, `web_search`, and `web_fetch`.

Drop a `.py` file into `agent_tools/plugins/` and every agent picks it up
on next restart. Files whose names start with `_` are skipped, which is
useful for disabled examples or local scratch tools. One file, two
invocation paths:

- **HTTP backends** (`llama`, `zai`, and any future API backend) see plugins
  as typed tools in the chat-completions `tools` schema, alongside
  the 16 built-in tools in `agent_tools/builtin/`
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
The `web_search` and `web_fetch` tools also cap downloaded response bodies
at 5 MiB. Both tools use the shared `open_http_response()` request setup and
`read_bounded_text()` reader in `agent_tools/base.py`, so their user agent,
timeouts, redirect policy, decoding, and response ceiling stay consistent.
The reader consumes chunked or slow responses until EOF or that ceiling
instead of treating a short network read as the complete body; `web_fetch`
then applies its separate `max_chars` output limit.
CLI backends rely on their own bundled shell tool for plugin execution, so
the plugin prelude only exposes how to call the extra tools; it does not
change the CLI's native tool sandbox.

## Backend behavior

Each backend defines its own model list and permission mapping in
`backends_agent/`:

| Backend | Launch path | Default chat model | Default summary model |
|---|---|---|---|
| `codex` | `codex exec --json` | `gpt-5.6-sol` | `gpt-5.6-luna` |
| `claude_code` | `claude --print --output-format stream-json` | `default` | `haiku` |
| `copilot` | `copilot --output-format json` | `auto` | `claude-haiku-4.5` |
| `llama` | OpenAI-compatible `/v1/chat/completions` | `auto` | `auto` |
| `zai` | Z.ai `…/api/paas/v4/chat/completions` (Bearer) | `glm-5.2` | `glm-4.5-air` |

The CLI-backend model lists are a hand-maintained snapshot; add newer or
private models through `extra_models` in config (see Configuration) rather
than editing source. The `llama` backend instead discovers models live
from its server. `llama` and `zai` share one in-process OpenAI-compatible
agent loop (`backends_agent/_openai_agent.py`); `zai` just adds the Bearer
auth header and points at Z.ai's endpoint.

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
maps the percentage to its own vocabulary and request shape:

| Backend | Bands | What gets sent at 100% |
|---|---|---|
| `codex` | 4 common levels @ 25% each | `-c model_reasoning_effort=xhigh` |
| `llama` | 4 levels @ 25% each | `payload["reasoning_effort"] = "high"` |
| `zai` | GLM-5.2: 7 levels; older GLM: thinking toggle | `payload["reasoning_effort"] = "max"` |
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
  copilot, claude_code, llama, zai) the same way — not just the CLIs that
  ask on their own.
- `autonomous` — the agent decides and proceeds without asking, closer to
  a full-auto run.

Scheduled `/reserve` turns run in throwaway sessions that cannot pause for
a reply, so they always use the autonomous policy regardless of this
setting. They can also drain past a paused collaborative chat queue;
ordinary queued chat still waits for the user's answer.

## Architecture

```
Cozter/
├── __init__.py           package marker and version
├── __main__.py           entry point; sets PYTHONPATH; runs the bot
├── requirements.txt      Python runtime dependencies installed into .venv
├── py.typed              marks the package as typed for downstream checkers
├── .config/              runtime config dir; only config.example.json is tracked
├── backends_bot/         chat surfaces and shared fenced-Markdown formatting
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
├── utils.py              shared state, queue, and backend-process helpers
├── tests/                unittest coverage for commands, state, queues, backends, prompts, tools, and updates
├── .config/config.example.json
│
├── backends_agent/       agent backends (one file per agent)
│   ├── base.py             abstract Backend; convert_effort, supports_typed_plugins
│   ├── codex.py            wraps `codex exec`
│   ├── claude_code.py      wraps `claude --print`
│   ├── copilot.py          wraps `copilot`
│   ├── _http_proc.py       process-like adapter and error handling for HTTP backends
│   ├── _openai_agent.py    shared in-process OpenAI-compatible agent loop
│   ├── llama.py            local /v1/chat/completions backend hooks
│   └── zai.py              Z.ai /api/paas/v4/chat/completions backend hooks
│
└── agent_tools/          tool surface for HTTP backends + plugin registry
    ├── base.py             AgentTool ABC; path/argument validation and shared HTTP helpers
    ├── builtin/            16 built-in tools (read_file, edit_file, glob, grep, bash, web_search, ...)
    └── plugins/            user drop-in zone (current_time.py shipped as a live plugin)
```

## Process and tool safety

Cozter owns the lifetime of every backend process it starts. User-facing
turns drain stderr concurrently with streamed JSON events, and every exit
path — normal completion, cancellation, an injected restart, event-parse
failure, or chat-delivery failure — reaps the child process and its drain
tasks. This prevents a failed callback or `/stop` from leaving an agent CLI
running in the background.

Internal LLM jobs (routing, session titling, compaction, and colony
consolidation) all go through `utils.run_internal_backend()`. The shared
runner applies each job's timeout, consumes stdout and stderr without pipe
deadlocks, kills timed-out or cancelled children, and logs stderr when a
backend exits without an assistant response. HTTP backends expose the same
process-shaped contract through `backends_agent/_http_proc.py`, so the
orchestrator uses one cleanup model for CLI and API agents.

The built-in file tools also fail closed at workspace boundaries. In
particular, `apply_patch` will not use a create patch to overwrite an existing
file, and a delete patch must match the current file and remove all of its
content before the file is unlinked. Failed hunks leave the target in place.
Regression coverage for these paths lives in
`tests/test_agent_process_cleanup.py`, `tests/test_utils.py`, and
`tests/test_agent_tools.py`.

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
- Chat-platform adapters: `backends_bot/base.py`, `formatting.py`,
  `cli.py`, `telegram.py`, `slack.py`, and `signal.py`
- Agent adapters: `backends_agent/base.py`, `_http_proc.py`,
  `_openai_agent.py`, `codex.py`, `claude_code.py`, `copilot.py`,
  `llama.py`, and `zai.py`
- Agent tool surface: `agent_tools/__init__.py`, `agent_tools/base.py`,
  the 16 files under `agent_tools/builtin/`, and user plugins plus their
  README under `agent_tools/plugins/`
- Project metadata, CI, and docs: `requirements.txt`, `py.typed`, `mypy.ini`,
  `.gitlab-ci.yml`, `.github/workflows/ci.yml`, `.config/config.example.json`,
  `.gitignore`, and this README
- Tests: `tests/conftest.py` plus focused `unittest` modules for
  agent attachments, prompts, and process cleanup; backend event parsing
  and retry behavior; bot commands; import binding; run locks and session
  picking; platform and Signal formatting; runtime diagnostics; state
  fallbacks; thinking-status display; updater behavior; utilities; and the
  built-in/plugin tool surface

The normal working checkout may also contain ignored runtime state such as
`.venv/`, `.cozter/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`,
`.log/`, and local assistant/editor directories. Treat those as local
machine state unless a file is deliberately being promoted into tracked
source.

When updating this README, cross-check user-facing facts against the source
that owns them:

- Config keys and defaults: `config.py`'s `_DEFAULT_CONFIG` and
  `.config/config.example.json`
- Commands and command behavior: `backends_bot/base.py`, with platform
  registration in `telegram.py`, `slack.py`, `signal.py`, and `cli.py`;
  shared fenced-Markdown segmentation and rendering live in
  `backends_bot/formatting.py`, including Signal's styled-span input
- Backend names, model defaults, effort bands, and health checks:
  `backends_agent/__init__.py` plus the concrete backend modules
- Tool/plugin behavior: `agent_tools/__init__.py`, `agent_tools/base.py`,
  `agent_tools/builtin/`, and `agent_tools/plugins/README.md`; shared
  validation, workspace-boundary checks, HTTP request setup, and bounded
  response reading live in `agent_tools/base.py`
- Workspace, session, queue, schedule, compaction, and colony state:
  `workspace.py`, `session.py`, `schedules.py`, `compaction.py`, and
  `colony.py`
- CI and local quality gates: `.gitlab-ci.yml`, `.github/workflows/ci.yml`,
  `mypy.ini`, and `tests/`

The agent loop in `agent.py:run()` is shared across backends. Each
`Backend.launch()` spawns the right subprocess or starts an in-process
OpenAI-compatible HTTP session; the orchestrator reads JSONL events from
stdout, translates them via `Backend.parse_event()` into `ChatEvent`s,
and streams a "Thinking..." status message that updates in place with the
latest few tool actions and a live preview of the answer text as it
arrives. On chat surfaces without editable messages (the CLI), tool
progress is emitted as separate status lines and the full answer arrives
at the end.

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
- `.log/` - rotating runtime logs, diagnostics dumps, and crash reports
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
git status -sb
git ls-files
find . -maxdepth 2 -type d -not -path './.git*' -print | sort
PYTHONPATH=.. .venv/bin/python -m unittest discover -s tests
git status --short
```

## Auto-update

`updater.fetch_and_pull()` runs every `update_check_interval` seconds.
It fetches `origin`, then fast-forward-pulls only when the working tree is
clean and the local branch is not ahead of its upstream. Dirty checkouts
and branches with local commits are treated as development state and are
left alone, so an auto-update pass does not fight an in-progress edit or
an unpushed commit.

Update checks wait for active turns to finish, pause new intake briefly,
and queue any messages that arrive during a pending restart. If `HEAD`
changed (remote update, manual pull, or a local commit while the bot was
running), the bot installs any new `requirements.txt`, broadcasts a
"restarting" message, and exits. The init system, such as `systemd` with
`Restart=always`, brings daemon mode back. CLI mode uses an outer
respawner process and relaunches itself in the same terminal. Persisted
queues resume after either path starts again.

## Runtime diagnostics

`__main__.py` writes rotating warning/error logs to `.log/cozter.log`.
Unhandled exceptions also get timestamped crash files in `.log/`, and
asyncio/thread dumps go to `.log/diagnostics.log`. On Unix-like hosts,
send `SIGUSR1` to the running daemon process to dump tasks, thread
stacks, and per-platform active-turn state without restarting it.

If `dump_traceback_interval` is set above zero, faulthandler emits
periodic stack dumps to the same diagnostics file. The auto-update path
uses the same machinery when it has waited longer than
`update_idle_timeout` for active turns to finish: it records diagnostics
and keeps waiting instead of killing in-flight work.

## Reading order

If you want to understand the codebase, the high-leverage entry points
are:

1. `__main__.py` → `backends_bot/base.py` to see how a turn enters the system
2. `agent.py:run()` to see the orchestrator
3. `backends_agent/_openai_agent.py` for the full HTTP agent loop and
   tool dispatcher
4. `backends_agent/llama.py` and `backends_agent/zai.py` for concrete
   HTTP backend hooks
5. `agent_tools/__init__.py` for the auto-discovery and plugin
   bridging

The CLI-backend files (`codex.py`, `claude_code.py`, `copilot.py`) are
thin: each defines `launch()` (build argv, spawn subprocess) and
`parse_event()` (translate the CLI's JSONL events to `ChatEvent`s).

## Development checks

Run the current unit tests from the parent directory, or set
`PYTHONPATH` to the parent when running inside the repository. Discovery
covers malformed state/config fallbacks, persistent queue restoration,
schedule parsing, backend model defaults and event parsing,
subprocess draining and exceptional-path cleanup, prompt construction,
attachment handling, run-lock cancellation, session picking,
platform/Signal rich-text formatting, runtime diagnostics, updater behavior,
agent-tool helpers, and built-in discovery/edit/patch safety.

```bash
cd ..
Cozter/.venv/bin/python -m unittest discover -s Cozter/tests
```

From inside `Cozter/`:

```bash
PYTHONPATH=.. .venv/bin/python -m unittest discover -s tests
```

CI runs on Python 3.11 and 3.12 and runs `ruff check` and `mypy` on every
push and merge request / PR. The canonical pipeline is `.gitlab-ci.yml`
(GitLab CI, the primary remote); `.github/workflows/ci.yml` mirrors it on
GitHub. mypy is adopted
gradually — enforced on clean modules, with pre-existing type debt
grandfathered per-module in `mypy.ini` (burn the list down over time).
`requirements.txt` contains runtime dependencies only, so install the CI
tooling explicitly before running the lint and type gates locally:

```bash
cd ..
Cozter/.venv/bin/python -m pip install -r Cozter/requirements.txt ruff mypy
Cozter/.venv/bin/ruff check Cozter
Cozter/.venv/bin/mypy --config-file Cozter/mypy.ini -p Cozter
```

Before committing, run the tests and check `git status --short` for only
intentional source or documentation edits. Runtime JSON, logs, sessions,
virtualenv files, and caches should stay local.
