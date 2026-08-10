# Cozter

A chat-surface that wraps coding-agent CLIs (codex, claude_code, copilot)
and OpenAI-compatible HTTP backends (local llama-server and Z.ai), exposing
them through Telegram, Slack, Signal, or a plain terminal. One bot process,
multiple workspaces, per-workspace settings, durable sessions with
automatic compaction, persistent turn queues, file attachments, and a
drop-in plugin system that works across every backend.

## What it gives you

- **A default meta-agent plus five direct agent backends**, picked per
  workspace:
  - `flexible` (default) — a meta-agent that sizes the work with a cheap
    summary-model call, splits it into up to 12 sub-tasks, routes each to
    the agent+model bound to its difficulty tier (`low` / `mid` / `high`),
    then merges the reports into one reply. Tiers can straddle backends
  - `codex` — OpenAI's CLI (`codex exec`)
  - `claude_code` — Anthropic's CLI (`claude --print`)
  - `copilot` — GitHub's CLI
  - `llama` — an unauthenticated OpenAI-compatible HTTP server (such as
    llama-server or LM Studio); the agent loop runs in-process and uses the
    typed tools in `agent_tools/`
  - `zai` — Z.ai's cloud API (Zhipu GLM models: `glm-5.2`,
    `glm-5v-turbo`, `glm-4.6v`, `glm-5.1`, …); OpenAI-compatible, so it
    shares the in-process loop — set `zai_api_key` in config
- **Four chat surfaces**, selected at launch:
  - Telegram (`python -m Cozter`)
  - Slack (Socket Mode; native Markdown rendering for AI replies; same
    launcher, set `slack_bot_token` in config)
  - Signal (same launcher, set `signal_group_urls` and the daemon socket)
  - CLI (`python -m Cozter -cli`) — the terminal becomes the chat
- **Per-workspace state**, scoped to `<workspace>/.cozter/`:
  sessions, last-session pointers, compaction history, agent choice,
  model, permission level, reasoning effort, summary backend, colony
  (long-term memory), uploads, generated image attachments, and schedules
- **Durable sessions with layered memory**: Cozter compacts older history
  from a conservative estimate of the active model's context capacity when
  that capacity is known, otherwise from the configured stored-message
  fallback. It writes a scratch summary and rewrites its long-term-memory
  list while retaining the latest five raw messages. Colony, long-term
  memory, summaries, and recent messages are prepended subject to the
  configured character budget; the new user message stays intact
- **Persistent turn queues on Telegram, Slack, and Signal**: if a user sends
  more work while an agent turn is running, or while an update restart is
  pending, the messages are queued on disk and restored after restart
- **Durable final-text delivery**: before a queued agent reply is sent,
  Cozter stages its text (and any usage footer) in a separate delivery
  ledger. A failed chat send or restart retries that finished text before
  later work, rather than rerunning agent tools; attachment uploads remain
  best-effort so a retry cannot duplicate a file
- **Platform-safe text delivery**: long replies are split at chat-surface API
  boundaries. Telegram applies its 4,096-character limit to both rich agent
  replies and plain command/status output; Signal preserves rich-text styling
  as it splits messages at 4,000 characters; and Slack keeps plain chunks
  below 39,000 characters and rich Markdown blocks below 12,000 while
  balancing fenced code blocks
- **File flow in both directions**: chat uploads are saved into the
  workspace and text-like files are inlined into the next prompt; agent
  replies can upload workspace files or generated images back to chat
- **Recurring scheduled prompts on Telegram, Slack, and Signal**: `/reserve`
  queues prompts on selected weekdays and runs them in throwaway sessions so
  routine jobs do not pollute the user's active conversation
- **Auto-update**: the bot polls origin, fast-forward-pulls only when the
  checkout is clean and not locally ahead, then restarts safely: it re-execs
  in place on POSIX and hands off to the Windows supervisor on Windows

## Quick start

Run Cozter directly from its source checkout; it does not ship an installable
package, console script, or build artifact. Deploy the checkout itself rather
than running `pip install .`. Run the following from the checkout's parent
directory. For the CI-covered path, first confirm that `python` is Python
3.11 or 3.12 (`python --version`):

```bash
git clone https://gitlab.com/mgneh/cozter.git Cozter
python -m Cozter --cli
```

GitLab (`git@gitlab.com:mgneh/cozter.git`) is the canonical upstream. GitHub
(`github.com/limztudio/Cozter`) is a mirror; clone and pull from GitLab to
stay current. Maintainers who publish to a mirror should check
`git remote -v` before pushing, since a checkout may use a separate push URL.
Keep the checkout directory named `Cozter` (with a capital `C`): the launcher,
updater, and CI import the source tree as that package rather than as an
installed distribution.

That starts the local terminal chat surface without requiring bot tokens.
Before sending work, create or select a workspace:

```text
/new
/absolute/path/to/new-workspace
# or: /open /absolute/path/to/existing-workspace
```

Then use `/doctor` to check backend readiness. A fresh workspace selects the
`flexible` meta-agent, whose summary agent and three tiers default to Codex.
If Codex is unavailable, select a configured direct backend with `/agent` and
`/summaryagent`, or rebind flexible's summary agent and tiers before sending a
task.

If you are already inside `Cozter/`, this compatibility launcher re-execs
through the same package entry point:

```bash
python __main__.py --cli
```

`-cli` and `--cli` are equivalent. Use `-h` or `--help` to print the
launcher usage without creating configuration or bootstrapping the virtual
environment; unrecognized options exit with an error rather than starting a
daemon.

For Telegram, Slack, or Signal daemon mode, run without `-cli`:

```bash
python -m Cozter
# first run writes Cozter/.config/config.json and exits;
# fill in tokens and run again
```

On startup, Cozter creates or re-execs through a project-local `.venv` when
needed. It installs `requirements.txt` only when a required runtime module is
missing or its installed version falls outside the declared requirement; an
update that changes the requirements installs them before restart.

### Windows Task Scheduler

For a long-running Windows deployment, use `run_cozter.ps1` as the task
action rather than a global Python executable. For a checkout at
`D:\Cozter`, set:

- **Program/script:** `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`
- **Arguments:** `-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "D:\Cozter\run_cozter.ps1"`
- **Start in:** `D:\`

Start Cozter normally once before registering the task so the project
`.venv` exists; the script expects `.venv\Scripts\python.exe` to be present.
It runs that venv directly and restarts Cozter after every exit, including
updates and failures, so Task Scheduler keeps a single supervised process.

### POSIX systemd

No POSIX service unit is shipped. For a checkout at `/srv/Cozter` owned by a
dedicated non-privileged `cozter` user, first create the venv and config
file with an initial daemon-mode launch, then fill in `.config/config.json`:

```bash
sudo -u cozter -H sh -c 'cd /srv/Cozter && python __main__.py'
```

Create `/etc/systemd/system/cozter.service` with the checkout's *parent* as
the working directory, because the source tree is imported as `Cozter`:

```ini
[Unit]
Description=Cozter chat-agent service
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=cozter
Group=cozter
WorkingDirectory=/srv
ExecStart=/srv/Cozter/.venv/bin/python -m Cozter
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then enable it with `sudo systemctl daemon-reload` followed by
`sudo systemctl enable --now cozter`. Give the `cozter` account access only
to the workspace directories it is intended to manage.

CLI mode intentionally skips daemon configuration at startup: it neither
requires nor creates `.config/config.json`, and it uses the stable local
platform key `cli:local` for workspace/session state. If that file already
exists, an agent turn can still read shared backend, tool, and permission
limits from it. Daemon mode (`python -m Cozter` without `-cli`) validates
`.config/config.json` before any platform starts.

## Requirements

- Python 3.11+ (CI targets 3.11 and 3.12; the codebase uses modern type syntax)
- One agent backend CLI, server, or API key:
  `codex`, `claude`, `copilot`, an unauthenticated OpenAI-compatible HTTP
  server for the `llama` backend, or Z.ai credentials for the `zai` backend
- Python package dependencies from `requirements.txt`:
  `python-telegram-bot`, `slack-bolt`, and `aiohttp`. The
  launcher bootstraps them into the project-local `.venv` when required
  runtime modules are missing or an installed version falls outside the
  declared requirement; normal starts do not invoke pip.
- Optional external services:
  Telegram and Slack need their platform tokens; Signal also requires a
  separately installed and running `signal-cli` JSON-RPC daemon.

## Deployment boundary

Cozter is intended for a trusted individual or small trusted group. An
authorized chat participant can select an existing workspace with `/open` or
request a new one with `/new` wherever the service account's filesystem
permissions allow. In write-capable modes, an agent can also run commands in
that selected workspace. Run Cozter as a dedicated, non-privileged OS user and
authorize only people, channels, and groups you trust.

Telegram authorizes the configured user IDs; Slack and Signal authorize the
configured channels or groups, so anyone able to send a message in an allowed
Slack channel or Signal group can use the bot. `/permission confirm` and
`deny` reduce tool access, but they do not replace OS-level isolation.

## Configuration

`Cozter/.config/config.json` (created on the first daemon-mode run; the
example layout lives in `.config/config.example.json`):

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
  "update_check_interval": 300,
  "recent_workspace_limit": 10,
  "message_queue_size": 50,
  "max_upload_bytes": 52428800,

  "extra_models": {},
  "model_context_windows": {},
  "max_permission": "auto",
  "show_usage": true
}
```

Exactly one daemon chat surface must be populated: `telegram_bot_tokens`
+ `user_ids`, `slack_bot_token` + `slack_app_token` +
`slack_channel_ids`, or `signal_group_urls` + `signal_jsonrpc_socket`.
The CLI surface needs neither.

The shipped `config.example.json` has a non-empty placeholder Telegram token.
When copying it, replace that value with a real token or change
`telegram_bot_tokens` to `[]`; otherwise the shape is valid and startup will
attempt to connect with the placeholder.

On POSIX hosts, daemon startup tightens the token-bearing `.config/`
directory to owner-only access (`0700`) and `config.json` to `0600`. If that
cannot be done, Cozter stops instead of leaving credentials exposed. Windows
keeps the existing directory ACLs because those permissions are not managed by
`chmod`.

`recent_workspace_limit` controls how many paths `/open` shows.
`message_queue_size` caps each user's pending chat turns.
`max_upload_bytes` (default 52,428,800 / 50 MiB) caps each attachment
accepted from or sent to Telegram, Slack, and Signal.

The llama safety settings are read at the start of every llama turn:
`llama_max_agent_turns` (default 60) limits tool-call turns before Cozter
forces a final answer, `llama_tool_repeat_limit` (default 3) skips an
identical call after that many executions, and `llama_socket_timeout`
(default 1800 seconds) is the per-read timeout for a slow local server.

Agent turns do not have a wall-clock timeout; long-running work is
allowed to finish. `tool_timeout` (default 120s) still caps each
cooperative/asynchronous individual tool call for HTTP backends. It is not a
plugin sandbox: trusted plugin code that blocks the event loop synchronously
can still stall the process and must isolate blocking work itself. CLI-backend
plugin scripts instead use that CLI's shell/tool policy and are not governed
by `tool_timeout`. `update_idle_timeout` (default 1200s) controls how often
the auto-update loop dumps diagnostics while waiting for active turns; it
keeps waiting instead of restarting through active work.
`dump_traceback_interval` (default 0) enables optional periodic thread dumps
when set above zero; on-demand `SIGUSR1` diagnostics remain available either
way.

`extra_models` adds model IDs to a backend's `/model`, `/summarymodel`, and
flexible-tier model pickers on top of its built-in list, keyed by backend name
— for example `{"codex": ["my-private-codex-model"]}`. It is useful for
static or self-hosted catalogs. Copilot deliberately ignores unverified
extras: its picker uses the authenticated account's policy-controlled catalog,
so an arbitrary configured ID cannot be shown if the account cannot use it.
Malformed entries are ignored.

`model_context_windows` supplies an explicit input capacity, in tokens, for
private, self-hosted, auto-routed, or otherwise unreported models. It is an
operator override keyed first by backend and then by model ID; an optional
`"*"` supplies a backend-wide default:

```json
{
  "model_context_windows": {
    "llama": {"qwen3-coder": 32768, "*": 16384},
    "zai": {"private-glm": 128000}
  }
}
```

Only positive integers are accepted. This setting affects automatic
compaction, not the provider request itself. Cozter uses built-in/live model
metadata for known public models; use this override when a private model's
capacity is known. If any model that can receive the saved conversation has
no capacity, Cozter safely uses the workspace's `/compact` message interval
instead.

`llama_max_retries` (default 2) is how many times a transient llama HTTP
failure — a dropped connection, a read timeout, an HTTP 429/5xx, or a streamed
completion that exceeds Cozter's retained-state limits — is retried with
exponential backoff before the turn fails. A capped completion is discarded
before any of its buffered tool calls execute. Set it to `0` to disable
retries. Only the `llama` backend uses it; the CLI backends have their own
process-level behavior.

`zai_api_key` enables the `zai` backend (Z.ai / Zhipu GLM) — get one from
your Z.ai account and paste it here. `zai_base_url` defaults to
`https://api.z.ai/api/paas/v4` (already includes the version, so only
`/chat/completions` is appended); override it for a regional endpoint with a
valid HTTPS URL. A blank, malformed, or non-HTTPS override falls back to the
default so the API key is never sent over cleartext HTTP.
`zai_socket_timeout` (default 300s) and `zai_max_retries` (default 2)
mirror the llama knobs and retry behavior for the cloud call. Select `zai`
with `/agent`, pick a model with `/model` (default `glm-5.2`), and add private
or regional GLM ids via `extra_models` (`{"zai": ["glm-…"]}`). Long z.ai
coding turns automatically continue into another tool-enabled segment when
Cozter's internal tool-call segment limit is reached, instead of stopping for
a manual "continue". For current documented GLM agent models, Cozter also
enables Z.ai's preserved-thinking protocol and returns the exact opaque
`reasoning_content` with the next tool result; that provider state is never
shown in the chat reply.

`max_permission` (default `auto`) caps the highest `/permission` mode any
workspace may use, bot-wide, in privilege order `deny < confirm < auto <
full`. `full` is the only mode that requests each CLI's explicit bypass flag
(arbitrary code execution) and exposes the in-process HTTP agents' direct host
shell for any authorized chat participant. Keep the default `auto` to prevent
those bypasses, or set it to `full` only when an operator explicitly accepts
that risk; provider-native sandbox and approval behavior still differs by CLI.
Trusted HTTP plugins are not sandboxed: an author must set
`requires_full_permission = True` for a plugin that can escape Cozter's
workspace-bounded tool model. See [Plugins](#plugins) for that contract.
`deny` exposes no tools to the HTTP and Copilot
backends; Codex and Claude Code use their strongest non-interactive
read-only/plan modes, which may still inspect the workspace. `/permission`
rejects a higher mode, and an already-stored higher value is clamped down.
Whitespace around a valid value is accepted; any other malformed value blocks
daemon startup and fails closed to `deny` if introduced while the daemon runs.

`show_usage` (default `true`) appends a compact per-turn token/cost footer
(e.g. `📊 12.5k in · 28 out · $0.01`) after each reply, for backends that
report usage — `codex` (`turn.completed`) and `claude_code` (`result`).
Other backends stay silent. Set it to `false` to suppress the footer.

Pending chat turns on daemon platforms are persisted in
`Cozter/.config/queue_<platform>.json`, so clean restarts, auto-updates,
and crash recovery do not drop already accepted messages. Platform IDs
are sanitized for those filenames; for example the CLI's stable platform
key `cli:local` maps to `queue_cli_local.json`. CLI mode does not restore
saved prompt-queue entries after a restart.

Before Cozter sends a completed queued reply, it writes the reply text and
any usage footer to `Cozter/.config/reply_deliveries_<platform>.json`. If the
outbound send fails or a restart interrupts delivery, it retries that staged
text before it starts another prompt for the same user, so it does not rerun
the completed agent turn and its tools. Delivery is intentionally
at-least-once: if a platform accepts text but Cozter cannot record that fact,
the text can be delivered again after recovery. Attachments stay on the
normal best-effort path because chat-file uploads do not have a safe
idempotency key. CLI mode likewise restores staged final text after restart.

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
  reasoning effort, summary backend, summary model, fallback compact interval,
  colony interval, and context budget
- `colony.json` — workspace-wide long-term memory consolidated across
  sessions
- `schedules.json` — recurring `/reserve` prompts and their last-fired
  timestamps
- `uploads/` — files received from Telegram, Slack, or Signal
- `generated_images/` — external generated images copied into the
  workspace before upload back to chat

Workspaces are recorded globally in `Cozter/.config/workspaces.json`
(per-user current pick + the recent-workspaces list). Daemon-platform turn
queues live beside it as `queue_<platform>.json`.

Signal intentionally scopes runtime state to the configured group rather than
the individual sender. Everyone in the same Signal group therefore shares its
selected workspace, active session pointer, and pending-turn queue; use a
separate configured group when that shared context is not appropriate.

When a workspace is selected, Cozter canonicalizes its path. Opening the
same directory through `.`, `..`, a trailing slash, or a symlink therefore
uses the same sessions, settings, recent-workspace entry, and per-workspace
turn locks rather than creating parallel state for path aliases.

Cozter also keeps its runtime state physically inside that canonical
workspace. When opening or using a workspace, it refuses a `.cozter`
directory—or an existing state component such as `sessions/`, `uploads/`, or
`generated_images/`—that resolves through a symlink outside the workspace.
Symlinks whose resolved targets remain inside the workspace continue to work.

The global runtime files are deliberately small JSON documents:

- `.config/config.json` — daemon chat-surface and backend settings
- `.config/workspaces.json` — current/recent workspace selections per
  user and platform
- `.config/queue_<platform>.json` — persisted pending daemon-platform turns
  so accepted work survives restarts, crashes, and auto-updates
- `.config/reply_deliveries_<platform>.json` — staged final text replies
  awaiting delivery, so a send failure does not repeat completed agent work
- `.config/detached_tasks_<platform>.json` — tracked external background
  tasks and any completion message awaiting delivery

Later updates to workspace selections and settings, sessions, colony memory,
schedules, queues, reply-delivery records, and detached-task records are
written through a temporary file and atomically replaced. The new file is
synced before replacement; on POSIX, Cozter also syncs the parent directory
so a completed rename is durable across a power loss. An interrupted write can
leave a harmless temporary file, but it cannot publish a half-written JSON
state document.

The session router is only used when there is no valid
`last_session.json` entry, such as a new workspace or a deleted session.
`/newsession` explicitly creates and pins a fresh session; otherwise each user
continues the same session across bot restarts and platform reconnects.
If an initial turn is still being routed while that user explicitly chooses a
different session, the newer choice remains pinned for the next message; the
in-flight turn keeps the session that was selected for it.

Colony consolidation includes sessions whose long-term list is empty, so their
names still help it retire stale workspace memory. If a workspace has no
sessions left, a colony pass clears its shared memory rather than carrying
deleted-session facts into later conversations.

For a direct agent turn, automatic compaction follows that selected model's
known input window. For a flexible turn, it follows the smallest known window
among the summary model and all three tier models. The stored context is
estimated conservatively and compacted at 60% of that window, leaving room for
system instructions, tools, the next request, and the reply. `/context` stays
a separate character cap; Cozter also compacts at 75% of that budget so it
summarizes raw history before the prompt builder needs to trim it. A model
with an unknown capacity falls back to `/compact`'s stored-message interval.

New sessions begin with a timestamp-based placeholder name. After their first
assistant reply, Cozter asks the selected summary backend for a short topical
title for the session picker. The result is written only while that placeholder
is still current, so a newer compaction title or custom persisted name is not
overwritten by a late background title request.

## Commands

All chat surfaces speak the same command set:

Every command also accepts a leading backslash in a regular message, such
as `\open 2`. This is useful in Slack workspaces where `/` commands are
reserved or unavailable; direct Slack mentions work too, for example
`@Cozter \open 2`.

| Command | What it does |
|---|---|
| `/new` | Prompt for a new workspace directory, create it, and select it |
| `/open [path-or-number]` | Switch to an existing workspace |
| `/agent` | Pick the agent backend (flexible / codex / claude_code / copilot / llama / zai) |
| `/model` | Pick the chat model for the current backend |
| `/agent_flexible_{low,mid,high}` | Pick the agent the flexible tier routes to |
| `/model_flexible_{low,mid,high}` | Pick the model the flexible tier routes to |
| `/summaryagent` | Pick the backend used for compaction / titling / routing, and for flexible's plan + merge |
| `/summarymodel` | Pick the model for the summary backend |
| `/permission` | full / auto / confirm / deny — how the agent treats tool calls |
| `/style` | collaborative / autonomous — whether the agent asks before big/ambiguous actions or runs full-auto |
| `/effort` | 0–100 reasoning effort; each backend maps to its native scale |
| `/compact [number]` | Show compaction state, or set the fallback message interval for unknown-capacity models |
| `/context [number]` | Show or set the per-turn context budget (characters of prepended history) |
| `/newsession` | Start a fresh session (next message will go into a new conversation) |
| `/sessions [number\|name]` | List this workspace's sessions, or switch to one |
| `/colony [number\|now]` | Show memory state, set the consolidation interval, or run it now |
| `/refresh` | Drop the workspace's `.codex/` cache (use after an upgrade) |
| `/stop` | Cancel the running agent turn and clear queued work |
| `/bg <task>` or `/background <task>` | Start a restart-safe detached task with the selected compatible backend |
| `/inject <text>` | Add context to the running turn; Cozter abandons its active phase and restarts with it |
| `/reserve` | Create a recurring scheduled prompt |
| `/schedules` | List schedules and delete one by number |
| `/version` | Show the current git version and last commit date |
| `/doctor` | Check each backend's readiness (CLI on PATH / HTTP backend configured or reachable) |
| `/cancel` | Cancel a picker/wizard, pending answer, running turn, or queued work |
| `/start` | Confirm the bot is running |

Most picker commands start at 1 and accept either the displayed number or
the literal name. The `/agent` picker deliberately reserves option `0` for
the default `flexible` meta-agent; its direct backends start at 1. `/open`
also accepts a recent-workspace number directly as `/open 2`.
If a picker entry is not recognized, Cozter keeps the picker open and asks
again; use `/cancel` to leave it.

`/context` applies a character budget to each composed turn. Cozter never
truncates the current user message: it trims saved context first and, if
needed, omits the continuation cue and sends the message alone.

An accepted `/inject` is either folded into a restarted turn or rejected once
the final reply has closed its injection window. This applies to every
`flexible` phase—including planning and merge calls as well as workers—so
context sent while the meta-agent is working cannot be silently lost between
phases.

`/bg` (or `/background`) currently uses Claude Code, so choose `claude_code`
with `/agent` first. Cozter persists the external task ID, polls independently, and sends
a new chat message when the task completes—even across a bot restart. A
compatible foreground agent can make the same handoff itself with its hidden
`[[background: ...]]` protocol marker; Cozter launches the durable session
and posts its completion rather than relying on a shell background process.
For Claude Code foreground and detached sessions, Cozter also installs a
session-only Bash hook that denies ordinary `&`, `nohup`, `disown`,
`run_in_background`, and nested `claude --bg` launches, so those jobs cannot
silently escape the callback ledger.
`/stop` (or `/cancel` when no picker is active) also stops a tracked detached
task. If Claude reports that a task is blocked waiting for input, Cozter
notifies the chat; use `claude agents` and `claude attach` in the workspace
to continue that interactive session.

Schedules are stored per workspace in `.cozter/schedules.json` and use
the host's local time. On Telegram, Slack, and Signal, the scheduler checks
every 30 seconds and records `last_fired`, so a missed slot fires once after
restart instead of being lost. Offset-bearing persisted timestamps are
normalized to that local basis, and malformed schedule fields are ignored.
Scheduled prompts run through the same persistent queue as user messages,
but use a fresh ephemeral session that is deleted after the turn; they do
not append to the user's current conversation. They run autonomously and can
continue even while normal chat work is paused waiting for a collaborative
`[[await]]` answer. CLI mode accepts `/reserve` but does not currently run a
scheduler or restore queued turns after a restart.

## Files and attachments

Telegram, Slack, and Signal uploads are copied into
`<workspace>/.cozter/uploads/` before the agent sees them. The generated
prompt includes the saved relative path, and text-like files up to
50,000 characters are inlined directly into the prompt. Larger text files
and binary files are referenced by path so the selected backend can inspect
them with its normal tools. Cozter reads at most 50,001 characters when
making that decision, so a large accepted text upload is not decoded in full
just to determine that it should be referenced by path.

`max_upload_bytes` is enforced before an outbound transfer and throughout an
inbound copy or download. Incoming files are staged beside their final path
and atomically renamed only after a complete, in-limit transfer succeeds, so
failed or oversized uploads do not leave a partial file for an agent to use.
Concurrent uploads reserve distinct destination names (for example,
`report (2).pdf`) before downloading, so matching filenames cannot overwrite
one another.

Telegram and Slack URL downloads share one bounded HTTP transport. Their
initial download URL must use HTTP(S); a declared `Content-Length` over the
configured cap is rejected before its body is read, and an absent or incorrect
length is still enforced while streaming. Slack supplies its bot authorization
header through that same path. Signal attachments arrive through its daemon as
local files or bounded base64 payloads and receive the same size enforcement
before they are published into the workspace.

Agents can attach files back to chat by emitting a line like:

```text
[[attach: path/inside/workspace.png]]
```

The path may be relative to the workspace or an absolute path inside it.
Generated images under `$CODEX_HOME/generated_images` (or
`~/.codex/generated_images` when `CODEX_HOME` is unset) and any
directories listed in `COZTER_ATTACHMENT_ROOTS` are also accepted. Cozter
copies explicitly referenced external images into `.cozter/generated_images/`
before upload so chat platforms never receive arbitrary external paths.
`COZTER_ATTACHMENT_ROOTS` is an OS-path-separator-delimited list (`:` on
POSIX, `;` on Windows); blank entries are ignored and `~` is expanded.
At the end of a run, Cozter also snapshots newly created or modified image
files in the workspace and attaches them unless the agent already referenced
them explicitly; shared external output directories always require an
explicit `[[attach: ...]]` marker. The workspace-only scan resolves image
links and ignores a link whose final target is outside the workspace, so a
trusted external artifact cannot become an automatic attachment merely by
being linked into a workspace. Replies can end with
`[[await]]` when the agent needs a user decision; the marker is stripped
and that user's queued work pauses until the next message arrives.

## Plugins

The built-in HTTP-toolkit includes filesystem, shell, search, and fetch
tools: `bash`, `read_file`, `write_file`, `edit_file`, `multi_edit`,
`apply_patch`, `delete_file`, `copy_file`, `move_file`, `make_dir`,
`list_dir`, `tree`, `glob`, `grep`, `web_search`, and `web_fetch`.

Drop a `.py` file into `agent_tools/plugins/` and every agent discovers it
on next restart. Whether a backend can invoke it still follows its selected
permission mode. Files whose names start with `_` are skipped, which is useful
for disabled examples or local scratch tools. One file, two invocation paths:

Treat plugins as trusted bot code: discovery imports their modules in the
Cozter process, so module-level code runs at startup and any dependencies must
be installed in the project environment. Restart after adding, removing, or
changing a plugin. For HTTP backends, `tool_timeout` bounds a
cooperative/asynchronous plugin call but cannot preempt synchronous
event-loop blocking; it does not sandbox plugin code. CLI-backend plugins run
through that CLI's own shell/tool policy and are not governed by Cozter's
`tool_timeout`.

- **HTTP backends** (`llama`, `zai`, and any future API backend) see plugins
  as typed tools in the chat-completions `tools` schema, alongside
  the 16 built-in tools in `agent_tools/builtin/`
- **CLI backends** (`codex`, `claude_code`, `copilot`) can't have
  external tools injected into their fixed toolkit. The bot
  instead lists each plugin in their prompt and tells the model to
  invoke it through the backend's own `bash` / `shell` tool as
  `python -m Cozter.agent_tools.plugins.<filename> '<JSON args>'`.

For HTTP backends, `deny` exposes no tools. Under `/permission confirm`,
plugins are never exposed or executed—even if a plugin reuses a built-in
read-only tool's name. That mode uses only Cozter's fixed read-only built-ins;
select `auto` or `full` when an HTTP plugin needs to run.

Under HTTP `/permission auto`, plugins are exposed by default. A plugin that
can access host resources outside Cozter's workspace-bounded tool model—for
example arbitrary paths, commands, credentials, or sockets—**must** declare
`requires_full_permission = True` on its `AgentTool` class; otherwise it
remains available in `auto`. This flag controls HTTP-tool availability only;
it does not sandbox trusted plugin code or alter CLI-backend shell behavior.
Tool names are global, so choose a unique `name`: a later plugin registration
replaces any earlier tool with the same name, including a built-in.

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
`read_file` reads at most 128 KiB per call; its line-offset scan is also
bounded at 16 MiB so a pathological offset cannot leave a worker thread
walking an enormous file.
`grep` only opens regular files up to 1 MB and runs its regex scan in a
killable worker process. It stops and reaps that worker after the smaller of
`tool_timeout` and 30 seconds, so an expensive pattern cannot keep consuming
CPU after a timeout; narrow the pattern or search path if that happens.
The `web_search` and `web_fetch` tools also cap downloaded response bodies
at 5 MiB and share the bounded `read_bounded_text()` reader in
`agent_tools/base.py`. `web_search` uses the common request setup;
`web_fetch` instead uses a public-network-only client with redirect targets
validated individually, preventing redirects into private addresses. The
reader consumes chunked or slow responses until EOF or that ceiling instead
of treating a short network read as the complete body; `web_fetch` then
applies its separate `max_chars` output limit.
CLI backends rely on their own bundled shell tool for plugin execution, so
the plugin prelude only exposes how to call the extra tools; it does not
change the CLI's native tool sandbox.

## The flexible agent

`flexible` is the default agent. It is not a CLI of its own — it is a
meta-agent that spends a cheap summary-model call to size the work up
front, then pays for the expensive model only where the work is actually
hard. One turn is three phases:

1. **Understand and split.** The summary agent (`/summaryagent`,
   `/summarymodel`) restates what you asked for and splits it into
   up to 12 sub-tasks, grading each one `low`, `mid`, or `high`.
2. **Route.** Each sub-task runs as a full agent turn — real tools, real
   file edits — on the agent and model bound to its difficulty tier. They
   run one at a time, in order, and each worker sees the reports of the
   ones before it.
3. **Merge.** The summary agent folds the workers' reports into the single
   reply you see. The workers' own text never reaches the chat; their tool
   and file events still stream into the live status display.

The grading rubric the planner is held to:

| Tier | When | Example |
|---|---|---|
| `low` | Straightforward, well-scoped work with clear intent | Add a small validation check, or extend an existing function with clearly defined behavior |
| `mid` | Some reasoning required, but the problem stays bounded | Write unit tests for an existing method with known inputs and outputs |
| `high` | Only when the task involves ambiguity, complex logic, or deeper system understanding | Refactor a system with unclear dependencies, or debug a non-obvious issue |

Each tier carries its own agent and model, so tiers can straddle
backends — a local `llama` for the easy parts, `claude_code` on `opus` for
the hard ones:

```
/agent_flexible_low     -> claude_code        /model_flexible_low   -> haiku
/agent_flexible_mid     -> codex              /model_flexible_mid   -> gpt-5.6-luna
/agent_flexible_high    -> claude_code        /model_flexible_high  -> opus
```

Defaults put all three tiers on `codex` (`gpt-5.4-mini` / `gpt-5.6-luna` /
`gpt-5.6-sol`); pointing a tier at another agent picks that agent's
cheap/mid/strong models automatically (its `tier_models` table). `/model`
and `/doctor` print the current wiring. A tier can only point at a *direct*
backend — never at `flexible` itself, which would plan forever.

Two behaviors are worth knowing. Under `/style collaborative`, the turn can
stop and wait for you (`[[await]]`) at either end of the pipeline: the
planner may ask **one** clarifying question instead of guessing, and the
merge step may end its reply on a question it genuinely needs answered
before the work can continue. The workers in between never stop to ask,
since nobody is reading them mid-pipeline. Under `/style autonomous` — and
on scheduled `/reserve` runs, which are always autonomous — nothing pauses:
a question the merge model emits anyway is stripped rather than left to
strand a run nobody is watching. And when planning fails outright (summary
CLI missing, unparseable output), the turn degrades to a single `high`-tier
sub-task carrying the original request, so a botched split never quietly
downgrades hard work to a weak model.

## Backend behavior

Each backend defines its own model list and permission mapping in
`backends_agent/`. `flexible` is omitted — it owns no CLI and no model of
its own, only the three tiers above.

| Backend | Launch path | Default chat model | Default summary model |
|---|---|---|---|
| `codex` | `codex exec --ephemeral --json` | `gpt-5.6-sol` | `gpt-5.6-luna` |
| `claude_code` | `claude --print --output-format stream-json --verbose` | `default` | `haiku` |
| `copilot` | `copilot --output-format json --no-color` | `auto` | `auto` |
| `llama` | Unauthenticated OpenAI-compatible `/v1/chat/completions` | `auto` | `auto` |
| `zai` | Z.ai `…/api/paas/v4/chat/completions` (Bearer) | `glm-5.2` | `glm-4.5-air` |

Codex discovers its visible local CLI catalog, while Copilot queries its
authenticated ACP model selector from the selected workspace and fails closed
to `auto` if that catalog cannot be read. This keeps enterprise-disabled and
workspace-policy-disabled Copilot models out of the picker; a stored Copilot
choice also uses `auto` until it appears in that workspace's fresh catalog.
Claude Code has no safe non-interactive account catalog, so it keeps
a curated list that `extra_models` can extend. Its picker offers standard
aliases, the supported `fable[1m]`, `sonnet[1m]`, `opus[1m]`, and
`opusplan[1m]` long-context aliases, and verified version pins (including
Fable 5, Sonnet 5, Opus 5, and explicit `[1m]` variants of other documented
long-context models). Those selected 1M variants have documented 1M-token
windows; mutable aliases and bare 4.x pins remain capacity-unknown because
their active window can vary by account and provider. Claude's `/fast` is a
session toggle rather than a selectable `*-fast` model ID. Llama and Z.ai
discover models live from their configured HTTP endpoints.
`llama` and `zai` share one in-process OpenAI-compatible agent loop
(`backends_agent/_openai_agent.py`); `zai` just adds the Bearer auth header
and points at Z.ai's endpoint. GLM-4.6-and-newer tool-capable models,
including the curated GLM-4.6V family and GLM-5V-Turbo, opt into Z.ai's
incremental tool-call argument stream, which the shared SSE parser merges
before executing the requested tool.

Permission modes are backend-specific because a chat bot cannot answer a
per-tool-call approval dialog. `codex` uses bypass only for `full`, its
workspace-write sandbox for `auto`, and a read-only sandbox for `confirm`
and `deny`. `llama` and `zai` run in-process: `deny` exposes no tools and
`confirm` exposes only read-only tools. `auto` permits Cozter's
workspace-bounded built-ins plus plugins that do not declare themselves
full-only with `requires_full_permission`; both it and `confirm` block the
built-in direct host shell again at execution time. `claude_code` uses bypass
only for `full`, `acceptEdits` for `auto`, and plan mode for `confirm`/`deny`.
`copilot` uses `--yolo` only
for `full`, `--allow-all-tools` (while retaining path and URL checks) for
`auto`, and an explicit empty tool list for `confirm`/`deny`. Internal
router, titling, and compaction calls always use `deny`, so conversation
content cannot elevate their permissions. For ask-before-acting behavior on
any backend, use `/style collaborative` — it pauses the turn (via
`[[await]]`) for your reply instead of relying on a CLI approval flow.

The `llama` model picker queries `llama_server_url/v1/models` and falls
back to `auto` if the server is down or returns no model IDs. The `zai`
picker queries the configured Z.ai `/models` endpoint and retains its curated
agent-capable fallback, including text-compatible multimodal models such as
`glm-5v-turbo`, `glm-4.6v`, and `glm-4.5v`, if the account cannot be queried.
It filters Z.ai's known image, OCR, and audio-only IDs because those require
different endpoints, while preserving unknown/private chat-model IDs. Codex,
llama, and Z.ai refresh their live catalogs periodically, so long-running
services see CLI, server, and account model changes. HTTP catalog responses
over 1 MiB use the backend's normal fallback; otherwise Cozter de-duplicates
the IDs, keeps at most 4,096, and ignores IDs longer than 512 characters.
Z.ai's documented GLM-4.5-and-newer agent models also preserve their opaque
reasoning blocks across tool calls, while the older GLM-4-32B fallback and
unknown/private IDs retain the ordinary OpenAI-compatible transcript shape.
Copilot
uses a short ACP handshake without sending
a prompt, and refreshes a successful workspace-specific catalog periodically.
Its successful-catalog and failed-probe caches are each pruned to 64
workspaces, evicting expired and least-fresh entries; an evicted workspace is
simply rediscovered when it is opened again. The ACP probe runs from the
selected workspace so project policy, including `.github/allowed_models.txt`,
applies to the picker and stored-model check.
Its picker also accepts ACP's provider-grouped model selectors, so
account-approved models stay visible without a hard-coded catalog. The
`copilot` backend keeps
prompts under the Windows
command-line limit by dropping the oldest composed context when a prompt
exceeds its cap; the current user message is kept at the tail. Each Copilot
run also uses a short-lived private CLI home, so its planner, worker, and
merge calls do not appear in Copilot's session history or get exported to
GitHub web and mobile; Cozter's workspace session remains the durable
conversation record. The private home copies `config.json` and `settings.json`
from `$COPILOT_HOME` when it is set, otherwise from `~/.copilot`; set
`COPILOT_HOME` before launch when the source profile lives elsewhere.

Codex uses discovered effort and context-window metadata only while its
60-second catalog cache is fresh. Until `/model` refreshes an expired cache,
known public models use Cozter's built-in metadata and a previously discovered
private model has no inferred context window, so the `/compact` message-
interval safeguard applies. An explicit `model_context_windows` entry remains
authoritative.

Provider event envelopes are treated as untrusted input. A missing, blank, or
non-text backend error message is normalized to `Unknown error` before it is
stored or shown, rather than exposing a provider object or breaking the turn
parser. If Codex has
already streamed an assistant reply, a late stream error is retained on the
turn without replacing that reply.

## Reasoning effort

`/effort` accepts `0`–`100` and is stored per workspace. `0` means "no
override — server defaults apply"; `1`–`100` are explicit. Each backend
maps the percentage to its own vocabulary and request shape:

| Backend | Bands | What gets sent at 100% |
|---|---|---|
| `codex` | Model-aware: 4–6 levels | `ultra` (Sol/Terra), `max` (Luna), or `xhigh` (others) |
| `llama` | 4 levels @ 25% each | `payload["reasoning_effort"] = "high"` |
| `zai` | GLM-5.2: 7 levels; older GLM: thinking toggle | `payload["reasoning_effort"] = "max"` |
| `claude_code` | Model-aware: current Fable / Sonnet 5 / Opus 4.7+ use 5 levels; 4.6 uses 4; Haiku and legacy pins use their defaults | `--effort max` for supported current models |
| `copilot` | 6 levels (`minimal` through `max`) for an explicit model; `auto` delegates to Copilot | `--effort max` for an explicit model; omitted for `auto` |

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
├── __init__.py           package marker
├── __main__.py           entry point; sets PYTHONPATH; runs the bot
├── requirements.txt      Python runtime dependencies installed into .venv
├── py.typed              marks the package as typed for downstream checkers
├── pyproject.toml        Ruff E4/E7/E9/F lint contract
├── .config/              runtime config dir; only config.example.json is tracked
├── backends_bot/         chat surfaces and shared fenced-Markdown formatting
├── agent.py              orchestrator: builds prompt, runs backend, streams events and attachments
├── session.py            per-workspace conversation persistence
├── compaction.py         scratch-summary + long-term-memory rewriter
├── colony.py             cross-session long-term memory consolidation
├── router.py             session picker for first-turn-in-workspace (subsequent turns reuse last_session.json)
├── titling.py            auto-titles new sessions from their first turn
├── schedules.py          /reserve cron-style scheduled prompts
├── flexible.py           flexible meta-agent prompt construction + plan/merge parsing
├── workspace.py          per-workspace settings (model, permission, effort, ...)
├── config.py             global .config/config.json reader
├── updater.py            git fetch + restart loop
├── utils.py              shared state, queue, and backend-process helpers
├── tests/                unittest coverage for commands, state, queues, schedules, compaction, backends, flexible, prompts, tools, attachments, and updates
├── .config/config.example.json
│
├── backends_agent/       agent backends (one file per agent)
│   ├── base.py             abstract Backend; convert_effort, supports_typed_plugins
│   ├── codex.py            wraps `codex exec`
│   ├── claude_code.py      wraps `claude --print`
│   ├── claude_background_guard.py
│   │                       session-only Claude Bash hook that blocks
│   │                       untracked background launches
│   ├── copilot.py          wraps `copilot`
│   ├── flexible.py         flexible meta-agent backend (no CLI of its own)
│   ├── _http_proc.py       process-like adapter and error handling for HTTP backends
│   ├── _openai_agent.py    shared in-process OpenAI-compatible agent loop
│   │                       and cached live-model discovery
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
Normal unified-diff hunks must also match the line counts declared in their
headers before any target is written, so a malformed or truncated patch is
rejected instead of being treated as a smaller valid edit.
`write_file`, `edit_file`, `multi_edit`, and updates to existing files through
`apply_patch` replace content through a temporary file. An unsuccessful
replacement leaves the prior contents intact, and existing file mode bits are
preserved.
`copy_file` only copies files, while `move_file` can move a file or directory
but refuses to move a directory into its own subtree. Both operations refuse
to replace an existing destination and create a missing destination parent
directory only after their source and destination checks pass.
The string-edit tools honor a broad `replace_all` operation only when its
tool argument is the literal JSON boolean `true`; malformed truthy values
retain the safer unique-match behavior.
Regression coverage for these paths lives in
`tests/test_agent_process_cleanup.py`, `tests/test_utils.py`, and
`tests/test_agent_tools.py`.

CLI JSONL and OpenAI-compatible HTTP SSE use the same bounded line reader in
`utils.iter_bounded_lines()`. Each transport retains at most 4 MiB for one
physical line; if a malformed peer never terminates a line, Cozter discards
that line and resumes at the next newline rather than allowing the bot's
memory use to grow without bound. A later valid event can still be processed.

OpenAI-compatible backends also bound the retained state for one streamed
completion: 4 MiB of assistant text, 4 MiB of arguments for any one tool
call, 8 MiB across retained text and tool arguments, and 128 tool calls. A
per-completion limit breach is handled as a retryable failure before any
buffered tool call executes. Tool-argument fragments are joined only after a
completion finishes, which avoids repeated copying as a long streamed
argument arrives. Across a tool-using HTTP agent run, the retained system,
user, assistant, tool, and continuation messages are also capped at 32 MiB.
Cozter refuses a message that would exceed that total and asks the user to
narrow the task or reduce tool output; if the assistant's requested tool-call
message itself cannot be retained, its tools are not run.

Persisted session state is treated as recovery data rather than trusted input:
malformed last-session pointers, unsafe session IDs, and session files whose
embedded ID does not match their filename are ignored. A damaged or
hand-edited state file therefore falls back to normal session selection rather
than reaching outside `.cozter/sessions/`. Workspace discovery retains normal
`**` glob behavior while memoizing match states, so an agent-supplied pattern
with many repeated globstars cannot turn matching into exponential work.

## Source inventory

The tracked workspace is intentionally flat and small. A complete source
audit should use `git ls-files` so hidden tracked files, especially
`.config/config.example.json`, are included even though `.config/*` is
ignored for local secrets and runtime queues.

- Package entry and runtime setup: `__init__.py`, `__main__.py`,
  `config.py`, `updater.py`, and `utils.py`
- Conversation, memory, and workspace state: `agent.py`, `workspace.py`,
  `session.py`, `router.py`, `titling.py`, `compaction.py`,
  `colony.py`, `flexible.py`, and `schedules.py`
- Chat-platform adapters: `backends_bot/base.py`, `formatting.py`,
  `cli.py`, `telegram.py`, `slack.py`, and `signal.py`
- Agent adapters: `backends_agent/base.py`, `_http_proc.py`,
  `_openai_agent.py`, `codex.py`, `claude_code.py`,
  `claude_background_guard.py`, `copilot.py`, `flexible.py`, `llama.py`,
  and `zai.py`
- Agent tool surface: `agent_tools/__init__.py`, `agent_tools/base.py`,
  the 16 files under `agent_tools/builtin/`, and user plugins plus their
  README under `agent_tools/plugins/`
- Project metadata, CI, and docs: `requirements.txt`, `py.typed`, `mypy.ini`,
  `pyproject.toml` (the Ruff E4/E7/E9/F lint contract), `.gitlab-ci.yml`,
  `.github/workflows/ci.yml`, `.config/config.example.json`,
  `run_cozter.ps1` (the Windows Task Supervisor launcher used by the update
  restart path), `.gitignore`, and this README
- Tests: `tests/conftest.py`, shared `tests/helpers.py`, plus focused
  `unittest` modules covering agent attachments, prompts, process cleanup,
  and post-turn behavior;
  backend model defaults, event parsing, and llama retry; bot and Slack
  commands; compaction; the flexible meta-agent; inject; import binding;
  run locks, session picking, and auto-titling; platform, Slack, and Signal rich-text
  formatting; durable reply delivery; runtime diagnostics; state fallbacks;
  status latency and thinking-status display; updater behavior; utilities; and the
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
- Flexible's tiers, grading rubric, planner/merge prompts, and plan
  parsing: `flexible.py`; its orchestration loop lives in
  `agent.py:_run_flexible()` and its per-tier settings in `workspace.py`
- Tool/plugin behavior: `agent_tools/__init__.py`, `agent_tools/base.py`,
  `agent_tools/builtin/`, and `agent_tools/plugins/README.md`; shared
  validation, workspace-boundary checks, HTTP request setup, and bounded
  response reading live in `agent_tools/base.py`
- Workspace, session, queue, schedule, compaction, and colony state:
  `workspace.py`, `session.py`, `schedules.py`, `compaction.py`, and
  `colony.py`
- CI and local quality gates: `.gitlab-ci.yml`, `.github/workflows/ci.yml`,
  `mypy.ini`, `pyproject.toml`, and `tests/`

The agent loop in `agent.py:run()` is shared across backends. Each
`Backend.launch()` spawns the right subprocess or starts an in-process
OpenAI-compatible HTTP session; `agent.py:_drive_backend()` reads JSONL
events from stdout, translates them via `Backend.parse_event()` into
`ChatEvent`s, and streams a "Thinking..." status message that updates in
place with the latest few tool actions and a live preview of the answer
text as it arrives. On chat surfaces without editable messages (the CLI),
tool progress is emitted as separate status lines and the full answer
arrives at the end. `flexible` is the one agent that never reaches
`Backend.launch()` itself: `agent.py:_run_flexible()` intercepts it and
drives one `_drive_backend()` pass per sub-task instead, on whichever
backend that sub-task's tier points at.

## Repository state

Tracked project files are intentionally small: the top-level runtime modules
and metadata, `backends_bot/`, `backends_agent/`, `agent_tools/`, `tests/`,
the CI definitions, this README, and `.config/config.example.json`.
Everything else created by a running bot is local state.

Do not commit these runtime artifacts:

- `.config/config.json`, `.config/workspaces.json`,
  `.config/queue_<platform>.json`, `.config/reply_deliveries_<platform>.json`,
  and `.config/detached_tasks_<platform>.json`
  - local tokens, workspace selections, persisted pending messages, staged
  replies, and detached-task state. Platform IDs are sanitized for queue
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

Useful read-only audit commands before documentation or release commits:

```bash
git status -sb
git ls-files
find . -maxdepth 2 -type d -not -path './.git*' -print | sort
PYTHONPATH=.. .venv/bin/python -m unittest discover -s tests
git status --short
```

To synchronize first, run `git pull --ff-only` separately from a clean
worktree; it can fast-forward the checkout and is therefore not an audit-only
operation.

## Auto-update

In daemon mode, `updater.check_for_update()` runs every
`update_check_interval` seconds (five minutes by default). CLI mode uses the
same five-minute interval but does not load daemon configuration at startup.
The updater fetches `origin` without blocking message intake, then checks
whether the clean local branch is behind its upstream. Dirty checkouts and
branches with local commits are treated as development state and are left
alone, so an auto-update pass does not fight an in-progress edit or an
unpushed commit. It requires `git`, an `origin` remote, and a tracking
upstream; if any of those are unavailable, it safely skips the auto-pull.

Only when an update is available does Cozter pause new AI turns, wait for
active turns to finish, fast-forward-pull, install any changed
`requirements.txt`, and broadcast a "restarting" message. On POSIX, the
daemon then re-execs itself in place. On Windows, it exits for the bootstrap
or `run_cozter.ps1` supervisor to relaunch it. Manual pulls and local commits
while the bot is running also trigger this safe restart path. A service
manager such as `systemd` with `Restart=always` remains useful for boot-time
startup and unexpected exits. CLI mode uses an outer respawner process and
relaunches itself in the same terminal. Daemon platforms restore persisted
queues after either path starts again; CLI mode currently does not.
If a message is accepted just as an update becomes pending, Cozter keeps that
already-persisted message queued and does not start its agent turn against a
checkout that is about to change.

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
schedule parsing, backend model defaults and event parsing, llama retry
behavior, the flexible meta-agent's planning/merge, post-turn and inject
flow, subprocess draining and exceptional-path cleanup, prompt
construction, attachment handling, run-lock cancellation, session
picking, auto-titling, compaction, platform/Slack/Signal rich-text formatting,
status-latency and thinking-status display, runtime diagnostics, updater
behavior, agent-tool helpers, and built-in discovery/edit/patch safety.

If `codex` is on `PATH`, one catalog-consistency test also invokes
`codex debug models` with a 15-second timeout; it skips when that command
fails, times out, or returns no visible models. To avoid that optional CLI
probe, run the suite with Codex absent from `PATH`.

For a local developer check with a CI-supported Python version, create the
project venv with Python 3.11 or 3.12 and install the runtime dependencies
plus the two CI tools:

```bash
cd ..
python3.12 -m venv Cozter/.venv  # or python3.11
Cozter/.venv/bin/python -m pip install -r Cozter/requirements.txt ruff mypy
```

```bash
cd ..
Cozter/.venv/bin/python -m unittest discover -s Cozter/tests
```

From inside `Cozter/`:

```bash
PYTHONPATH=.. .venv/bin/python -m unittest discover -s tests
```

CI runs the same three gates — `ruff check`, `mypy`, and the `unittest`
suite — on Python 3.11 and 3.12 for pushes to `main` and merge requests /
PRs. The canonical pipeline is
`.gitlab-ci.yml` (GitLab CI, the primary remote); `.github/workflows/ci.yml`
mirrors it on GitHub. mypy is enforced on the core runtime, agent backends,
and tool surface; `tests/` are excluded and the chat-platform adapters are
temporarily error-suppressed for their untyped SDK interactions, as defined
in `mypy.ini`.
`requirements.txt` contains runtime dependencies only, so install the CI
tooling explicitly before running the lint and type gates locally:

```bash
cd ..
Cozter/.venv/bin/python -m pip install -r Cozter/requirements.txt ruff mypy
Cozter/.venv/bin/ruff check Cozter
Cozter/.venv/bin/mypy --config-file Cozter/mypy.ini --python-version 3.11 -p Cozter
Cozter/.venv/bin/mypy --config-file Cozter/mypy.ini --python-version 3.12 -p Cozter
Cozter/.venv/bin/python -m unittest discover -s Cozter/tests
git -C Cozter diff --check
```

The two `mypy` invocations mirror CI's Python 3.11 and 3.12 target matrix. To
exercise the full CI matrix locally, run the `unittest` command from separate
3.11 and 3.12 virtual environments as well. Before committing, check `git
status --short` as well, to ensure only intentional source or documentation
edits are staged. Runtime JSON, logs, sessions, virtualenv files, and caches
should stay local.
