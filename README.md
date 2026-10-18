# Cozter

A chat-surface that wraps coding-agent CLIs (codex, claude_code, copilot, grok)
and OpenAI-compatible HTTP backends (local llama-server, Z.ai, and Meta Model API), exposing
them through Telegram, Slack, Signal, or a plain terminal. One Cozter
process hosts either the local terminal or every configured daemon surface
together—with one Telegram bot instance per configured token—across
multiple workspaces, so the same workspaces stay reachable on all platforms
at once,
with per-workspace settings, durable sessions with automatic compaction,
persistent turn queues, file attachments, and a drop-in plugin system that
works across every backend.

## What it gives you

- **A default meta-agent plus seven direct agent backends**, picked per
  workspace:
  - `flexible` (default) — a meta-agent that sizes the work with a cheap
    summary-model call, splits it into up to 12 sub-tasks, routes each to
    the agent+model bound to its difficulty tier (`low` / `mid` / `high`),
    then merges the reports into one reply. Tiers can straddle backends.
    Whole-scope (`all`/`entire`/`every`/`whole`) and doc-following requests
    are split so every item/rule is covered with verifiable done-criteria;
    build/verify requests keep target enumeration, canonical build/test,
    full-log inspection, repairs, rechecks, and runtime verification together
    in one cohesive task per target or related target group, with full-log
    evidence (commands + exit codes + error/warning counts);
    a plan that hits the 12-task cap carries an explicit PARTIAL + remainder
    note, and truncated worker reports are marked as previews so the merge
    requires retained scope and verification evidence before claiming done
  - `codex` — OpenAI's CLI (`codex exec`)
  - `claude_code` — Anthropic's CLI (`claude --print`)
  - `copilot` — GitHub's CLI
  - `grok` — xAI's Grok Build CLI (`grok --prompt-file`)
  - `llama` — an unauthenticated OpenAI-compatible HTTP server (such as
    llama-server or LM Studio); the agent loop runs in-process and uses the
    typed tools in `agent_tools/`
  - `zai` — Z.ai's cloud API (Zhipu GLM models: `glm-5.3`, `glm-5.3-flash`,
    `glm-5.3-flashx`, `glm-5.2`, `glm-5v-turbo`, `glm-5.1`, `glm-5`, `glm-4.7` family,
    `glm-4.6`/`glm-4.6v`/`glm-4.5` variants including `glm-4.5-air`
    (the default summary model), plus documented Coding Plan `[1m]` pins);
    OpenAI-compatible,
    so it shares the in-process loop — set `zai_api_key` in config
  - `meta` — Meta's Model API (Muse Spark models: `muse-spark-1.3`,
    `muse-spark-1.2`, `muse-spark-1.1`, …); OpenAI-compatible, so it shares
    the in-process loop — set `meta_api_key` in config
- **Four chat surfaces**, with Telegram, Slack, and Signal runnable together:
  - Telegram (`python -m Cozter`)
  - Slack (Socket Mode; native Markdown rendering for AI replies; same
    launcher, set `slack_bot_token` in config)
  - Signal (same launcher, set `signal_group_urls` and the daemon socket)
  - CLI (`python -m Cozter --cli`) — the terminal becomes the chat
- **Per-workspace state**, scoped to `<workspace>/.cozter/`:
  sessions, last-session pointers, compaction history, agent choice,
  model, permission, interaction style, reasoning effort, summary backend,
  colony (long-term memory), uploads, generated image attachments, and
  schedules
- **Durable sessions with layered memory**: Cozter compacts older history
  from a conservative estimate of the active model's context capacity when
  that capacity is known, otherwise from the configured stored-message
  fallback. It writes a scratch summary and rewrites its long-term-memory
  list while retaining the latest five raw messages. If a pass covers
  fewer than five ordinary messages (a handful of large ones), it keeps
  one fewer than it covered so history still shrinks instead of retrying
  the same prefix. Colony, long-term memory, summaries, and recent
  messages are prepended subject to the configured character budget
  (20,000 characters by default, per-workspace via `/context`); when that
  budget overflows, colony and long-term items most relevant to the
  current request are kept instead of newest-only. The
  new user message stays intact
- **Interrupted turns stay in memory**: `/stop`, shutdown, or a mid-turn
  restart appends the stopped prompt plus its partial output (or last tool
  activity) to the session before cancelling, so "resume where you left
  off" works against a real record of the stopped attempt instead of a
  history that ends at the previous completed turn
- **Persistent turn queues on Telegram, Slack, and Signal**: if a user sends
  more work while an agent turn is running, or while an update restart is
  pending, the messages are queued on disk and restored after restart
- **Durable final-text delivery**: before a queued agent reply is sent,
  Cozter stages its text (and any usage footer) in a separate delivery
  ledger. A failed chat send or restart retries that finished text before
  later work, rather than rerunning agent tools; attachment uploads remain
  best-effort so a retry cannot duplicate a file. Text arrives first;
  background picture uploads overlap the next agent turn. Later text waits at
  most five seconds for older uploads, while attachment batches retain their
  per-chat order with at most three file uploads active in each batch.
  `/cancel` and daemon stop cancel in-flight uploads
- **Platform-safe text delivery**: long replies are split at chat-surface API
  boundaries. Telegram applies its 4,096-character limit to both rich agent
  replies and plain command/status output, rendering rich Markdown to HTML
  before splitting with balanced tags and intact entities; Signal
  preserves rich-text styling as it splits messages at 4,000 characters; and
  Slack keeps plain chunks below 39,000 characters and rich Markdown blocks
  below 12,000 while balancing fenced code blocks
- **File flow in both directions**: chat uploads are saved into the
  workspace and text-like files are inlined into the next prompt; agent
  replies can upload workspace files or generated images back to chat, while
  CLI mode prints their local paths
- **Recurring scheduled prompts on Telegram, Slack, and Signal**: `/reserve`
  queues prompts on selected weekdays, dates every month, or yearly dates and
  runs them in throwaway sessions so routine jobs do not pollute the user's active conversation
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
task. Authenticate the selected CLI separately: `/doctor` checks executable,
endpoint, or API-key availability, not account authorization or a successful
agent turn.

Launch from the checkout's parent with `python -m Cozter`; add `--cli` for
the terminal surface. Use `-h` or `--help` to print the launcher usage without
creating configuration or bootstrapping the virtual environment. Unrecognized
options exit with an error rather than starting a daemon.

For Telegram, Slack, or Signal daemon mode, run without `--cli`:

```bash
python -m Cozter
# first run writes Cozter/.config/config.json and exits;
# fill in tokens and run again
```

On startup, Cozter creates or re-execs through a project-local `.venv` when
needed. It installs `requirements.txt` only when a required runtime module is
missing or its installed version falls outside the declared requirement; an
update that changes the requirements installs them before restart.
An inherited `COZTER_VENV_REEXEC` marker cannot bypass this check when a shell
tool launches system Python; the interpreter must actually be inside a venv.

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
The PowerShell supervisor owns a Windows launch job before starting Python.
Stopping that scheduled entry also stops its Python and tool descendants;
each restarted child shares the supervisor's job without retaining its handle.
Point the task at the deployment's own checkout (for example
`C:\Startup\Cozter\run_cozter.ps1` with **Start in** `C:\Startup`), not at an
unrelated source tree, and keep that deployment on a clean tracking branch so
`updater.check_for_update()` (every `update_check_interval` seconds, five
minutes by default) can fast-forward-pull and hand exit code 75 back to this
same supervisor loop. A direct `python -m Cozter` launch uses the launcher's
built-in Windows supervision loop; `run_cozter.ps1` starts the venv directly
for a single supervisor layer.

Use absolute paths for the task action and its **Start in** directory, both
from the same deployment. The supervisor derives its project root from its
own script location and changes to that checkout's parent before launching
`.venv\Scripts\python.exe -m Cozter`. Agent CLI subprocesses later run in the
selected workspace; that workspace is separate from the deployment working
directory. Keep developer shells pointed at the development checkout's parent
as described in [Development checks](#development-checks), even when a service
runs from a different deployment path.

### POSIX systemd

No POSIX service unit is shipped. For a checkout at `/srv/Cozter` owned by a
dedicated non-privileged `cozter` user, first create the venv and config
file with an initial daemon-mode launch, then fill in `.config/config.json`:

```bash
sudo -u cozter -H sh -c 'cd /srv && python -m Cozter'
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
limits from it. Daemon mode (`python -m Cozter` without `--cli`) validates
`.config/config.json` before any platform starts.

## Requirements

- Python 3.11+ (CI targets 3.11 and 3.12; the codebase uses modern type syntax)
- One agent backend CLI, server, or API key:
  `codex`, `claude`, `copilot`, `grok`, an unauthenticated OpenAI-compatible HTTP
  server for the `llama` backend that supports streaming
  `/v1/chat/completions` and function tools, Z.ai credentials for the
  `zai` backend, or Meta Model API credentials for the `meta` backend
- Python package dependencies from `requirements.txt`:
  `python-telegram-bot`, `slack-bolt`, and `aiohttp`. The
  launcher bootstraps them into the project-local `.venv` when required
  runtime modules are missing or an installed version falls outside the
  declared requirement; normal starts do not invoke pip. The declared ranges
  keep each integration on a reviewed major-release line
  (`python-telegram-bot >=21,<23`, `slack-bolt >=1.18,<2`, and
  `aiohttp >=3.14.3,<4`), so bootstrap can take compatible fixes without
  silently adopting an unreviewed major API change.
  A fresh managed launch also needs Python's `venv`/`ensurepip` support and
  access to the configured package index. For offline or restricted
  deployments, pre-create `.venv` and install `requirements.txt` from an
  approved package source before launching Cozter.
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
It is not a multi-tenant service.

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
  "signal_account": "",
  "signal_own_device_id": 0,

  "llama_server_url": "http://127.0.0.1:8080",
  "llama_max_agent_turns": 60,
  "llama_tool_repeat_limit": 3,
  "llama_max_retries": 2,

  "zai_api_key": "",
  "zai_base_url": "https://api.z.ai/api/paas/v4",
  "zai_max_retries": 2,

  "meta_api_key": "",
  "meta_base_url": "https://api.meta.ai/v1",
  "meta_max_retries": 2,

  "flexible_max_parallel_workers": 3,
  "update_idle_timeout": 1200,
  "dump_traceback_interval": 0,
  "update_check_interval": 300,
  "recent_workspace_limit": 10,
  "message_queue_size": 5000,
  "max_upload_bytes": 52428800,

  "extra_models": {},
  "model_context_windows": {},
  "max_permission": "auto",
  "show_usage": true
}
```

Treat `config.json` as a local secret: it can contain bot and API tokens, so
do not commit it or share it.

Populate any combination of daemon chat surfaces: `telegram_bot_tokens`
+ `user_ids`, `slack_bot_token` + `slack_app_token` +
`slack_channel_ids`, and/or `signal_group_urls` + `signal_jsonrpc_socket`.
Every configured surface starts side by side and shares the same workspaces
(each platform keeps its own current-workspace pointer, queues, and delivery
ledger; workspace files, sessions, and recent lists are shared).
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
`message_queue_size` caps each user's pending chat turns (default 5,000).
If a restored backlog exceeds the current cap, every accepted entry stays
in FIFO order while the queue drains; shrinking the cap does not discard
accepted work or replace the queue held by an active producer or drain loop.
`max_upload_bytes` (default 52,428,800 / 50 MiB) caps each attachment
accepted from or sent to Telegram, Slack, and Signal.

The llama safety settings are read at the start of every llama turn:
`llama_max_agent_turns` (default 60) limits tool-call turns before Cozter
forces a final answer, and `llama_tool_repeat_limit` (default 3) skips an
identical call after that many executions.

Cozter imposes no fixed execution deadline on foreground turns, planning,
flexible workers, merging, completion judging, routing, compaction, colony
consolidation, or titling. Built-in and plugin tools, HTTP provider generation,
updater Git commands, and dependency installation likewise run until completion,
explicit cancellation, or an actual failure. `/stop` cancels active work, and
`/inject` interrupts its active phase to apply new context; ordinary messages
queue for the next turn. Cancellation retains ownership of an already-started
file mutation until that write finishes, preventing a later operation from
racing it. Connection establishment, Signal RPC, model discovery, post-exit
pipe drains, and retry counts retain their separate transport safeguards.
Low-level callers may explicitly request an operation timeout; normal agent
execution supplies no such deadline.

Flexible attempts retain their maximum of 24 workers. Worker-count and backend
tool-round limits checkpoint the current job instead of completing it: Cozter
saves the original request, prior worker evidence, remaining scope, workspace,
session, artifacts, and accounting, then resumes the same job with a fresh
worker/tool allowance. Later queued jobs remain held until the whole request
completes. A pending update may restart between those attempts; the durable
checkpoint restores the same job first.
Codex (`codex`), Claude Code (`claude_code`), Copilot (`copilot`), and Grok
(`grok`) reuse their confirmed native CLI conversation across turns in the
same Cozter session, including after a queue job completes or Cozter restarts.
Native state is isolated by canonical workspace, Cozter session, backend,
resolved model, permission, reasoning effort, and execution lane. Direct turns
and Flexible's serial editing workers share a compatible serial lane, so
changing a subtask's wording or difficulty does not discard their native
conversation. Concurrent inspection workers use separate reader lanes and
native IDs; they do not attach to the editing worker's conversation or each
other's active session. Job checkpoints still retain their exact subtask scope
and native-generation pin, independently of these reusable lanes.
`/newsession` starts a fresh conversation; `/sessions` returns to the selected
conversation's own state. Changing the workspace, model, permission, or effort
selects a separate native identity.
An unfinished job that already owns a native checkpoint keeps that checkpoint
through completion; conversation-owned reuse begins with a subsequent launch.

On a confirmed resume, Cozter sends the current request and changed context
instead of appending the complete Cozter transcript again. New conversations
receive the full bootstrap context. Cozter resumes only an exact provider ID
that it recorded from the CLI: it never adopts a CLI's latest session or an
unrelated conversation. If the provider confirms that this exact session is
missing before any new work begins, Cozter can start a replacement with full
saved context. Ambiguous failures or failures after work has begun do not
trigger a fresh replay. Invalid saved state is preserved and reported rather
than silently starting over.

Cozter records context delivery as soon as the provider confirms that work has
started, before a long-running turn finishes. That receipt is written to disk,
so a recovered job can skip history and prompt sections already supplied to
its exact native session. Session initialization alone or an error before work
starts does not acknowledge the prompt. A full message receipt also covers
shorter budget previews of that same message; changed source content remains
eligible for synchronization. An unchanged request on the same recovered job
uses a continuation cue. Sending identical wording as a new user request still
asks the agent to perform it again.

Flexible's native workers receive bounded shared Cozter context as well as
their own plan, assigned instructions, earlier completed reports, and any
retained draft. Their shared history allowance is separate from the full
current request and checkpoint, so a long continuation still receives bounded
history backfill. Each native lane tracks the shared messages it has received.
Switching between direct and Flexible mode, or between backends, supplies
intervening conversation messages to the returning lane without replaying
unchanged context. A merged Flexible reply is supplied later to workers that
have not seen it, including when a queue job has already completed. A worker's
receipt for the exact user request prevents that request from being repeated
as historical context; it does not acknowledge the merger's unseen answer.
Parallel readers receive the same shared context independently; their current findings
reach later workers through completed reports after the inspection wave ends.
The planner uses bounded shared Cozter history and stays stateless.

These are provider-managed sessions and can appear in the native CLI's normal
local history where supported. Copilot keeps a persistent, conversation-owned
CLI home, including its local session history, until that Cozter conversation
is deleted. Cozter still stores its own transcript for session selection,
memory, and recovery. Native resumption avoids repeatedly sending the whole
Cozter transcript; the provider can still process and bill retained history,
so it does not make historical context token-free.

Job checkpoints remain under `.cozter/executions/`, separate from schedules
and the conversation-owned native state under
`.cozter/native_sessions/<session-hash>/<generation-record-hash>/`.
Meta, Z.ai, and llama restore their
job's assistant/tool transcript and completed tool results. An interrupted
command with no recorded result is reported to the resumed model as having an
unknown outcome, requiring inspection before repeating it. No saved tool call
is automatically replayed. These checkpoints remain isolated by workspace,
queue entry, worker, backend, model, permission, and effort. Completed queue
entries remove their job checkpoints while retaining native conversation
state. Deleting a temporary scheduled session also deletes its native state.
Internal routing, titling, planning, merging, judging, summary, compaction, and
colony calls stay stateless and do not attach to a worker's native session.
Large snapshots are privately copied and encoded on worker threads, with one
serialization per atomic checkpoint write. Independent HTTP reads share a
checkpoint per bounded wave, retaining completed results if that wave is
interrupted. Each state-changing tool still records its result durably before
the next tool begins. Concurrent saves use per-key ordering so a slow older
snapshot cannot replace a newer native-session event or tool outcome.

Durable prerequisite retries, staged replies and detached completions run on
independent polling cycles. Reply and detached passes use at most four owner
workers; one owner's records remain ordered, and an owed final reply takes
priority over its detached notifications. Each bounded pass waits for its
slowest owner before its next cycle; chat-delivery and control-RPC operations
retain their own safeguards. A slow operation in one class leaves the other classes running.
The continuation instruction asks workers to reuse per-case validation
checkpoints when inputs and the build are unchanged, such as CTest `-F`,
while checking retained full logs for prior failures. A successful remaining
subset or a skipped case alone cannot establish that the whole suite passed.
Provider failures, unavailable completion verification, or a required user answer
pause that job and hold later work until guidance arrives or `/stop` cancels
it. These count limits do not change permissions, models, reasoning effort,
context/output bounds, or provider quotas. `update_idle_timeout` (default 1200s)
controls how often the auto-update loop dumps diagnostics while waiting for
active turns; it
keeps waiting instead of restarting through active work.
An already-applied local commit is detected before contacting the remote, so
network delays do not postpone recognition or activation of that update.
Background updater Git processes retain cached credentials and disable
interactive authentication prompts through process-local
[`GIT_TERMINAL_PROMPT`](https://git-scm.com/docs/git#Documentation/git.adoc-GITTERMINALPROMPT)
and [`GCM_INTERACTIVE`](https://github.com/git-ecosystem/git-credential-manager/blob/main/docs/environment.md#gcm_interactive).
The operator's interactive Git environment and configuration stay unchanged.
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
    "zai": {"private-glm": 128000},
    "meta": {"*": 1000000}
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
failure — a dropped connection, an HTTP 429/5xx, or a streamed
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
GLM Coding Plan users can instead set it to
`https://api.z.ai/api/coding/paas/v4`. The normal endpoint's fallback already
includes `glm-5.3` and multimodal `glm-5.3-flash`/`glm-5.3-flashx`; the Coding Plan fallback
also offers the documented `glm-5.3[1m]` and `glm-5.3-flash[1m]` pins.
`meta_api_key` enables the `meta` backend (Meta Model API — Muse Spark).
Get one from the Meta Model API developer console and paste it here (the
docs also expose it as a `MODEL_API_KEY` environment variable). The backend
talks to Meta's OpenAI-compatible endpoint at `https://api.meta.ai/v1`;
`meta_base_url` overrides it if Meta publishes a new path — it must include
the version segment, because only `/chat/completions` is appended, and it
must be HTTPS so the API key is never sent over cleartext HTTP.
Muse Image, Muse Voice Transcribe, and SAM vision segmentation need other endpoints
and are never
offered as agent models. Model discovery queries the account's `/models`
catalog live; the offline fallback lists the published chat models
(`muse-spark-1.3`, default, with `muse-spark-1.2` anchoring the cheap
summary/low tier). Private or preview IDs can be added via `extra_models`
(`{"meta": ["muse-spark-…"]}`); published Muse Spark context is 1M tokens,
overridable per model via `model_context_windows`.
`meta_max_retries` (default 2) mirrors the other cloud retry knob.

`zai_max_retries` (default 2) mirrors the llama retry behavior for the cloud
call. Z.ai generation has no imposed elapsed-time deadline. Select `zai`
with `/agent`, pick a model with `/model` (default `glm-5.3`), and add private
or regional GLM ids via `extra_models` (`{"zai": ["glm-…"]}`). Long z.ai
coding turns automatically continue into another tool-enabled segment when
Cozter's internal tool-call segment limit is reached, instead of stopping for
a manual "continue". For documented GLM models with Z.ai's
preserved-thinking request contract, Cozter returns the exact opaque
`reasoning_content` with the next tool result; that provider state is never
shown in the chat reply. `glm-4.5v` remains selectable for text-only chat,
but Cozter does not send it the generic function-tool schema that Z.ai's
vision API does not support.

`max_permission` (default `auto`) caps the highest `/permission` mode any
workspace may use, bot-wide, in privilege order `deny < confirm < auto <
full`. `full` is the only mode that requests each CLI's explicit bypass flag
(arbitrary code execution) and exposes the in-process HTTP agents' direct host
shell for any authorized chat participant. For HTTP agents, `full` also permits
built-in filesystem tools to access paths outside the workspace, subject to the
host account's operating system permissions; relative paths still start at the
workspace. `auto` and `confirm` keep these tools confined to the workspace.
Keep the default `auto` to prevent those bypasses, or set it to `full` only when
an operator explicitly accepts that risk; provider-native sandbox and approval
behavior still differs by CLI.
Trusted HTTP plugins are not sandboxed: an author must set
`requires_full_permission = True` for a plugin that can escape Cozter's
workspace-bounded tool model. See [Plugins](#plugins) for that contract.
For ordinary execution, `deny` exposes no tools to the HTTP and Copilot
backends; Codex and Claude Code use their strongest non-interactive
read-only/plan modes, which may still inspect the workspace. `/permission`
rejects a higher mode, and an already-stored higher value is clamped down.
Whitespace around a valid value is accepted; any other malformed value blocks
daemon startup and fails closed to `deny` if introduced while the daemon runs.

`show_usage` (default `true`) appends a compact per-turn token/cost footer
(e.g. `📊 12.5k in · 28 out · $0.01`) after each reply, for backends that
report usage — `codex` (`turn.completed`) and `claude_code` / `grok`
(`result`). Other backends stay silent. Set it to `false` to suppress
the footer.

Pending chat turns on daemon platforms are persisted in
`Cozter/.config/queue_<platform>.sqlite3`, so clean restarts, auto-updates,
and crash recovery do not drop already accepted messages. Platform IDs
are sanitized for those filenames; for example the CLI's stable platform
key `cli:local` maps to `queue_cli_local.sqlite3`. CLI mode does not restore
saved prompt-queue entries after a restart.

Queue admission has one commit point for the job and its scheduled occurrence.
All message and schedule producers share a serialized admission barrier; the
configured limit applies to new waiting work, while restored accepted work
remains runnable even if that limit was lowered. Queue completion and `/stop`
retain schedule watermarks, so a failed `last_fired` projection cannot repeat
an occurrence. The scheduler retries the human-readable `schedules.json`
status projection independently; projection errors do not block execution or
other schedules.

On the first launch after the queue database upgrade, Cozter migrates the old
`queue_<platform>.json` exactly once, preserving owner order, job IDs, native
sessions and checkpoints. It saves an exact `.json.migrated.<sha256>` backup
before committing, then removes the imported source. The versioned database
is authoritative after migration; a subsequently recreated old JSON file is
never imported. Include the database and its migration backup in operational
backups. Corrupt input or an unsupported database version stops startup rather
than silently replacing accepted work.

Before Cozter sends a completed queued reply, it writes the reply text and
any usage footer to `Cozter/.config/reply_deliveries_<platform>.json`. If the
outbound send fails or a restart interrupts delivery, it retries that staged
text before it starts another prompt for the same user, so it does not rerun
the completed agent turn and its tools. Delivery is intentionally
at-least-once: if a platform accepts text but Cozter cannot record that fact,
the text can be delivered again after recovery. Attachments stay on the
normal best-effort path because chat-file uploads do not have a safe
idempotency key: texts (and the usage footer) send synchronously first,
then pictures upload in a per-chat background chain, so the turn returns
as soon as texts are out and the next queued turn's agent work overlaps
the uploads; the next reply's texts wait for the earlier tail first,
preserving order without blocking agent work. CLI mode likewise restores staged final text after restart.

For Signal, `signal-cli` must already be installed, registered, and
running as a JSON-RPC daemon. Each invite URL in `signal_group_urls` is
resolved from the daemon's known groups or joined at startup. Set
`signal_jsonrpc_socket` to the Unix socket exposed by that daemon, for
example `/run/signal-cli/socket`. Cozter only connects to the socket; run
and restart the `signal-cli daemon` from a service manager such as
systemd.

`signal_account` selects the daemon account by its phone number or ACI. Leave
it empty when using a single-account daemon. With a multi-account daemon,
Cozter can select the only loaded account automatically; if several accounts
are loaded, configure the intended account explicitly. Account selection
applies to group lookup, message delivery, deletion, and attachments. A daemon
with no loaded accounts is unavailable even if its TCP port is listening.
Startup waits for transient Signal readiness failures with a retry delay
that grows from one second to at most thirty seconds; shutdown cancels that
wait. Invalid configuration remains an explicit error. Queues and schedules
start after the group lookup and receive subscription are ready.

Platform startup attempts run independently. A platform waiting for its
transport does not delay a healthy platform's queues, startup announcement,
or the updater. Retry requires successful cleanup of the failed attempt;
configuration, durable-state and cleanup errors remain visible failures.
Shutdown owns and joins startup attempts and queue consumers before closing
transports. Cancelling an incoming handler leaves its accepted job with the
owned consumer; `/stop` explicitly cancels that consumer and clears the work.

For a Windows daemon serving one account, start `signal-cli` with
`--data-dir <directory> -a <account> daemon ...` and enable Windows service
recovery. A failed account load then exits and can be retried by the service
manager. Delayed automatic startup also gives networking time to initialize
after a reboot.

On Windows the `signal-cli` daemon usually runs as a restricted service
account such as `NT AUTHORITY\LocalService`, so it can only attach files
that account can open. Cozter's `SignalBot.send_file` hands the daemon a
local path and the daemon itself opens that path. A per-user `Temp` file
such as `C:\Users\<you>\AppData\Local\Temp\...` can be inaccessible to the
service; an unreadable attachment makes the `send` RPC fail with `Access is
denied` wrapped as `AttachmentInvalidException` (surfaced as
`SignalCliError`), with no delivery. Stage live Signal attachments in a
directory whose actual Windows ACLs allow the daemon account to read the file
and access its parent directories, then remove the staged copy after the send.
Grant access to the daemon's configured identity, such as
`NT AUTHORITY\LocalService`, according to the deployment's needs.
`C:\Users\Public\...` can be suitable after checking its directory and file
ACLs; the path name alone does not guarantee access or `Everyone` full control.

Active Signal tasks show one `Thinking...` message, including scheduled
tasks once they start. Progress updates edit that message at most once per
minute, with at most ten edit attempts per message, respecting [Signal's message edit
limit](https://support.signal.org/hc/en-us/articles/6255134251546-Edit-Message).
Long tasks can replace an exhausted status message and remove the old one.
Creating or recovering a Signal status has a 30-second background deadline;
the initial foreground wait remains at most four seconds. A late message
handle is retained so progress can edit it. If creation fails or returns no
editable timestamp, Cozter can retry instead of silencing the entire task.
New status attempts are separated by at least five minutes, and activity is
coalesced into the latest preview. Status work is owned by the current turn
and stopped before its final reply, preserving task cancellation and queue
processing.

Signal formatting caches failed inline-marker searches and projects each style
only onto message chunks it overlaps. Long identifier inventories and heavily
formatted replies therefore avoid repeated whole-reply scans. Telegram and Slack
choose collision-free inline-code/bold placeholders in one pass, preserving
literal user text even when it contains the private marker characters.

Set `signal_own_device_id` to the numeric device ID used by this installation's
signal-cli daemon after checking its account's `listDevices` response. The default
`0` (or `null`) leaves device identification unset; Cozter never guesses from a
device name. Messages synced from the configured daemon device are ignored before
command, workspace-input, or queue handling. Commands from the primary phone and
other linked devices remain usable, including text that repeats an earlier reply.

When a source device is unknown, Signal echo protection uses a bounded durable
`.config/signal_outbound.json` journal. Cozter records a hash of the group and
rendered text before sending, then retains acknowledged timestamps. This catches
duplicate or delayed own-message syncs after a restart or a lost send response.
The journal contains no message plaintext and retains at most 8,192 records for
seven days. A failed pre-send journal write prevents that send; a failure after
an acknowledged send is logged without resending the confirmed message. Exact
`Queued (number/number).` protocol acknowledgements, including the directory-input
error template wrapping one, are never accepted as new tasks from any device.

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
- `native_sessions/` — confirmed native CLI conversations, context delivery
  records, and persistent Copilot homes, isolated per Cozter session and worker
- `executions/` — per-job checkpoints, HTTP tool transcripts, and flexible
  planning/worker evidence; completed queue jobs remove their own checkpoints
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
- `skills/` — reusable workspace workflows, one Markdown file per skill
- `uploads/` — files received from Telegram, Slack, or Signal
- `generated_images/` — external generated images copied into the
  workspace before upload back to chat

Workspaces are recorded globally in `Cozter/.config/workspaces.json`
(per-user current pick + the recent-workspaces list). Daemon-platform turn
queues live beside it as `queue_<platform>.sqlite3`.

Signal intentionally scopes runtime state to the configured group rather than
the individual sender. Everyone in the same Signal group therefore shares its
selected workspace, active session pointer, and pending-turn queue; use a
separate configured group when that shared context is not appropriate. Workspace
selections, session pointers, and queues use the exact current platform and
owner keys. Startup and incoming messages do not migrate or adopt state from
another key.

When a workspace is selected, Cozter canonicalizes its path. Opening the
same directory through `.`, `..`, a trailing slash, or a symlink therefore
uses the same sessions, settings, recent-workspace entry, and per-workspace
turn locks rather than creating parallel state for path aliases.

Cozter also keeps its runtime state physically inside that canonical
workspace. When opening or using a workspace, it refuses a `.cozter`
directory—or an existing state component such as `sessions/`, `uploads/`, or
`generated_images/`—that resolves through a symlink outside the workspace.
Creating a nested state directory such as `uploads/` or `generated_images/`
goes through `ensure_workspace_state_dir()`, which re-resolves the path after
`makedirs` so a symlink planted during creation cannot escape.
Symlinks whose resolved targets remain inside the workspace continue to work.

Foreground agent turns for one workspace are serialized, including turns from
different authorized users and scheduled `/reserve` jobs. This prevents their
file edits from interleaving; work in separate workspaces can still run
independently.

Global runtime settings and ledgers use JSON; the work queue uses SQLite:

- `.config/config.json` — daemon chat-surface and backend settings
- `.config/workspaces.json` — current/recent workspace selections per
  user and platform
- `.config/queue_<platform>.sqlite3` — persisted pending daemon-platform turns
  so accepted work survives restarts, crashes, and auto-updates
- `.config/reply_deliveries_<platform>.json` — staged final text replies
  awaiting delivery, so a send failure does not repeat completed agent work
- `.config/detached_tasks_<platform>.json` — tracked external background
  tasks and any completion message awaiting delivery

Later updates to workspace selections and settings, sessions, colony memory,
schedules, reply-delivery records, and detached-task records are
written through a temporary file and atomically replaced. The new file is
synced before replacement; on POSIX, Cozter also syncs the parent directory
so a completed rename is durable across a power loss. An interrupted write can
leave a harmless temporary file, but it cannot publish a half-written JSON
state document. Queue entries and schedule admission watermarks commit together
in a SQLite transaction with full synchronization.

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
The consolidation snapshot retains session identity, name, date, and memory
only, parsing one transcript at a time. Snapshot reads run outside the event
loop; stale checks and actual writes run as one owned worker operation while
the workspace writer lock remains held through cancellation. Unchanged memory
lists skip their full-transcript reread and rewrite. Session-block parsing
advances forward through malformed or incomplete model output.

Maintenance prompts treat persisted model output as recovery data, not as an
unbounded source of context, and every budget reserves its truncation marker
inside the cap so a clipped maintenance prompt always reads as a PARTIAL
preview rather than full coverage. The session router parses and sends at most
12 sessions — ordered newest-first by mtime so older sessions are counted but
skipped, never parsed — and
caps each description at 400 characters, marking truncated session blocks as
previews to route on without inventing unseen content. Compaction clips an oversized prior
summary and sends a contiguous oldest prefix of raw history; even when its
first raw message alone is too large, it sends a marked prefix so a later pass
can advance the history without skipping to a newer message. After a successful
partial compaction, Cozter atomically retains that first raw message's unseen
suffix, so text that was not sent to the summarizer is never discarded. When a
successful pass covers five or fewer ordinary messages, it keeps one fewer
than that cover count instead of the usual five, so a prefix of large messages
cannot stall compaction forever. A failed or stale compaction leaves the
original message unchanged. Colony consolidation caps its aggregate input at
100,000 characters, reserves at most 25,000 for the previous colony list, and
keeps every included session block complete when it trims a long title or
memory item. Truncated colony items, clipped session names, and capped input
carry explicit PARTIAL/remainder markers, and the consolidation prompt
consolidates only the sessions/items shown. Over-cap colony loads keep the
newest entries with a warning rather than silently keeping the oldest. Session
auto-titles likewise reserve their clip markers inside the 4,000-character
request and mark tight-budget cuts with a visible ellipsis. These limits affect only the internal maintenance prompt;
compaction and colony state are rewritten only after a successful response.
Maintenance calls discard responses from timed-out or failed backends.
Compaction preserves long-term items omitted from its input; a partial colony
pass preserves existing shared and session facts whose evidence was clipped,
so a preview cannot silently erase memory the model never received.
Partial memory rewrites reserve the existing facts' capacity before accepting
new facts, so additions cannot push unseen memory out of the 100-item colony
or 50-item session cap. When an oversized previous summary is clipped for the
compaction prompt, its omitted middle remains verbatim beside the new summary;
later passes can shrink that retained text without discarding unseen evidence.

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
Likewise, a compaction title derived from an older snapshot is discarded if
the session was renamed while compaction ran, so it cannot overwrite the
manual name.
The complete auto-title request is capped at 4,000 characters, with half of
its remaining context budget reserved for recent messages. An oversized stored
summary therefore cannot crowd the current exchange out of the title prompt.

## Workspace skills

Put reusable instructions in flat Markdown files under
`<workspace>/.cozter/skills/`, for example `release-check.md` or
`log-inspection.md`. Each file is one skill; no subdirectory or `SKILL.md`
wrapper is required in this source folder. Cozter creates the folder when it
prepares a workspace for execution. Every flat `.md` file, including a
`README.md`, is treated as a skill; keep general usage notes and examples
outside that folder. Skill files can be plain Markdown, or use
optional frontmatter for a name and short description:

```markdown
---
name: release-check
description: Verify a release using the project's existing build and test workflow.
---

# Release check

Read the project's release instructions and identify the affected targets.
Reuse still-valid verification evidence. Run the remaining required checks,
inspect their complete logs, and report commands, exit codes, and any limitations.
```

The example above is documentation; Cozter does not install it as an active
skill. Add only workflows you want available in that workspace. Ask for a
skill by name, or describe a task that matches its description. Skills supply
workflow instructions; the current request, worker role, and selected tool
permissions still govern execution. An inspection worker stays read-only even
when a skill describes a later repair or build step.
Without frontmatter, Cozter derives display metadata from the heading, first
paragraph, and filename. Skill documents use UTF-8 and are limited to 256 KiB.

Cozter initially supplies a lightweight catalog with each skill's identity,
description, source path, and revision. Workers load the relevant full file
when needed instead of adding every skill body to every prompt. Native CLI
workers can use their file-reading tools and native skill interface; API
workers use `list_skills` and `read_skill`, including the returned paging
cursors when a catalog or document spans multiple pages. Before each new user
request and each worker starting or resuming work, the agent selects relevant
skills again and re-reads their current files and the supporting references
needed for the task. This includes native CLI skills, even when the CLI loaded
them earlier or the shared catalog has not changed. Earlier skill reads remain
historical evidence; they do not replace a fresh read for the current request.
An unchanged native catalog and known conversation history are still omitted
on an exact session resume, while a short skill-refresh instruction is always
delivered. Native conversations and API tool transcripts remain intact.
`list_skills` returns stable IDs, revisions, and `next_offset`; pass a returned
ID as the `name` to `read_skill`. Continue a long document using its
`next_offset` and `revision`, so a file changed between pages is detected.
The optional `resource` argument reads a supporting UTF-8 reference relative
to that skill's source directory. Listing and reading skills do not execute
scripts or grant additional shell or filesystem permissions.
Native CLI workers can also page the catalog with
`python -m Cozter.agent_tools.builtin.list_skills '<JSON arguments>'` and load
instructions with `python -m Cozter.agent_tools.builtin.read_skill '<JSON arguments>'`
from the workspace.

The Flexible planner sees the same catalog and can read selected instructions
before choosing a workflow when its backend provides a reading interface.
A planner without read tools selects relevant skills from metadata, then
explicitly assigns its worker to load the full skill before choosing or
executing the detailed workflow. Its API planning tools are limited to skill
listing and reading; implementation runs in the assigned worker. Selected
skills are named in that worker's instructions. Discovery and native bridges
refresh when injected user guidance restarts a phase and before each actual
Flexible worker wave; parallel readers share one fresh catalog snapshot.
Skill updates refresh the catalog for remaining work while retaining the saved plan cursor and
completed evidence. Routers, titles, judges, merges, summaries, compaction,
and colony consolidation do not receive a Cozter-added skill catalog.

Edit or remove a source file to update its instructions. Revisions identify
changed skills, and a stale selection must be refreshed before using changed
instructions. Cozter also prepares native skill wrappers for its supported
CLI backends without replacing unrelated user-created native skills. The
flat `.cozter/skills/*.md` files remain the source of truth for those wrappers.
Generated `cozter-*` wrappers live under `.agents/skills/` and `.claude/skills/`;
their ownership is tracked in `.cozter/skill_bridges.json`. Cozter updates or
removes only its unchanged owned wrappers and preserves foreign or manually
modified native skills. Existing supported project and personal native skill
folders also appear in the shared catalog. Native skill names remain
provider-specific; the catalog's stable ID identifies the exact source when
display names overlap. Copilot keeps personal skill search paths available
while its conversation history stays in the private native session home.

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
| `/agent` | Pick the agent backend (flexible / codex / claude_code / copilot / grok / llama / meta / zai) |
| `/model` | Pick the chat model for the current backend |
| `/agent_flexible_{low,mid,high}` | Pick the agent the flexible tier routes to |
| `/model_flexible_{low,mid,high}` | Pick the model the flexible tier routes to |
| `/effort_flexible_{low,mid,high} [0–100\|default]` | Set each Flexible tier's effort; `default` inherits `/effort` |
| `/summaryagent` | Pick the backend used for compaction / titling / routing, and for flexible's plan + merge |
| `/summarymodel` | Pick the model for the summary backend |
| `/permission` | full / auto / confirm / deny — how the agent treats tool calls |
| `/style` | collaborative / autonomous — whether the agent asks before big/ambiguous actions or runs full-auto |
| `/effort` | 0–100 reasoning effort; each backend maps to its native scale |
| `/compact [number]` | Show compaction state, or set the fallback message interval for unknown-capacity models |
| `/context [number]` | Show or set the per-turn context budget (characters of prepended history, default 20000) |
| `/newsession` | Start a fresh session (next message will go into a new conversation) |
| `/sessions [number\|name]` | List this workspace's sessions, or switch to one |
| `/colony [number\|now]` | Show memory state, set the consolidation interval, or run it now |
| `/refresh` | Report disposable CLI-cache availability for the selected backend; preserve project configuration and saved native conversations |
| `/stop` | Cancel the running agent turn and clear queued work |
| `/bg <task>` or `/background <task>` | Start a restart-safe detached task with the selected compatible backend |
| `/inject <text>` | Add context to the running turn; Cozter abandons its active phase and restarts with it |
| `/reserve` | Create a recurring scheduled prompt |
| `/schedules` | List schedules and delete one by number |
| `/version` | Show the current git version and last commit date |
| `/doctor` | Check each backend's readiness (CLI executable, server endpoint, or API key) |
| `/cancel` | Cancel a picker/wizard, pending answer, running turn, or queued work |
| `/start` | Confirm the bot is running |

Most picker commands start at 1 and accept either the displayed number or
the literal name. The `/agent` picker deliberately reserves option `0` for
the default `flexible` meta-agent; its direct backends start at 1. `/open`
also accepts a recent-workspace number directly as `/open 2`.
If a picker entry is not recognized, Cozter keeps the picker open and asks
again; use `/cancel` to leave it. An unrecognized slash command (for
example `/nope`) replies `Unknown command: /nope` followed by the sorted
list of available commands (`/agent /bg /cancel …`), so the valid names
are visible without guessing.

`/context` applies a character budget to each composed turn. Cozter never
truncates the current user message: it trims saved context first and, if
needed, omits the continuation cue and sends the message alone.

An accepted `/inject` is either folded into a restarted turn or, once the
final reply has closed its injection window, queued as a follow-up turn
("Queued ... Running next.") instead of being dropped. This applies to every
`flexible` phase—including planning and merge calls as well as workers—so
context sent while the meta-agent is working cannot be silently lost between
phases. A failed planner still keeps a pending `/inject` for the fallback
high-tier turn rather than dropping it. The confirmation reply itself is
cosmetic — the message is already queued — so its send is bounded at
8 seconds: a slow platform send during a scheduled turn cannot stall the
handler, and the queued context still reaches the agent.

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
task, including its user-owned picture uploads. A detached completion's
`[[await]]` marker is stripped from the delivered text and never pauses the
normal queue; an existing interactive answer pause remains in place.
If Claude reports that a task is blocked waiting for input, Cozter
notifies the chat; use `claude agents` and `claude attach` in the workspace
to continue that interactive session.

Schedules are stored per workspace in `.cozter/schedules.json` and use
the host's local time. In `/reserve`, weekday names such as `mon,wed,fri`
repeat weekly; numbers such as `1,15,31` mean those dates every month.
Dates in `MM/dd` format, such as `12/25`, repeat every year. Short input such
as `2/9` is stored as `02/09`. `all` means daily, and `mon,15,12/25` runs
every Monday, on the 15th of each month, and on December 25 every year.
A date that does not exist in a month is skipped: a rule for `31`
does not create an April 30 slot, and a rule for `29` creates no February slot
in non-leap years. Yearly `02/29` runs only in leap years; impossible dates
such as `02/30` are rejected. An earlier missed valid occurrence can still
catch up once, including a leap-day occurrence after a restart in a later year.
When weekly, monthly, or yearly rules coincide, the schedule runs once at
its configured time. The wizard, schedule list, and deletion confirmation
label each rule as `Daily`, `Weekly`, `Monthly`, `Yearly`, or a combination.

The `days` array stores weekday abbreviations and yearly `MM/dd` dates as
strings, and monthly dates as JSON integers. For example:

| `days` value | Recurrence |
|---|---|
| `["mon", "wed", "fri"]` | Every Monday, Wednesday, and Friday |
| `[1, 15]` | The 1st and 15th of every month |
| `[31]` | The 31st in months that have one |
| `["mon", 15]` | Every Monday and the 15th of every month |
| `["12/25"]` | December 25 every year |
| `["02/29"]` | February 29 in leap years |
| `["mon", 15, "12/25"]` | Every Monday, the 15th of every month, and December 25 every year |
| `["mon", "tue", "wed", "thu", "fri", "sat", "sun"]` | Daily (`all` in the wizard) |

Numeric values are monthly dates, including `1` through `7`; use weekday
names for a weekly rule. Existing weekday-string arrays keep their weekly
meaning and need no conversion. Update every Cozter instance that reads a
workspace before adding monthly or yearly dates to its schedules.

Each schedule explicitly stores its creating
`platform_id`. Only records tagged for the current platform can run, appear in
the schedule picker, or be deleted through that picker. Missing or empty
platform identities are not adopted. Cozter does not rewrite schedule
definitions automatically.
A shared-lock check prevents concurrent ticks from firing the same slot twice. On Telegram, Slack, and Signal, the scheduler checks
every 30 seconds and records `last_fired`, so a missed slot fires once after
restart instead of being lost. Offset-bearing persisted timestamps are
normalized to that local basis, and malformed schedule fields are ignored.
Each schedule carries a unique id (minted at creation when missing or
already taken), and firing stamps every record sharing that id, so a
duplicate id cannot leave a stale twin due forever and refiring
every tick.
Scheduled prompts run through the same persistent queue as user messages,
but use a job-owned session that is deleted after full completion or explicit cancellation; they do
not append to the user's current conversation. Each owner runs one turn at a
time, so scheduled jobs wait behind active work and older queued messages.
Admission requires the schedule's originating workspace; a missing or empty
origin cannot create a durable scheduled job. `last_fired` advances after the
queue accepts the job. A full queue or failed admission leaves the slot due
for a later tick. The queued job records its schedule and slot identity so a
restart between queue admission and the schedule stamp cannot enqueue the
same occurrence twice.
`Queued scheduled task` confirms the prompt has been saved in the queue;
`Starting scheduled task` appears when that job is selected to run. These
notices time out after four seconds so a slow chat connection cannot hold
the scheduler or agent indefinitely. They run autonomously and can
not overtake an unfinished job that is waiting for a collaborative
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

Photo uploads carry the saved path plus verified dimensions, format, and
size, and every direct backend can see the pixels natively: `llama`,
`meta`, and `zai` receive them as OpenAI-style `image_url` parts,
`codex` via repeatable `--image` flags, `copilot` via repeatable
`--attachment` flags, `grok` via a private `--prompt-file` containing
JSON text/image blocks (plain text when no image is referenced), and
`claude_code` via a prompt hint telling the model to open the saved path
with its own Read tool. The shared helper keeps only in-workspace images
referenced by `[… attachment saved to: …]` markers (up to 4 per turn,
4 MiB each) and never raises, so an ordinary text turn is unchanged. Its
marker pattern is compiled once and plain text turns short-circuit on a
cheap substring check before the regex runs. The same module-level
precompile hygiene covers the other hot paths: the `apply_patch`
hunk-header parser, the `web_fetch` page-title scan, and the platform
state filename sanitizer in `backends_bot/base.py`. The
`read_file` tool likewise returns verified dimensions/format/size for
image files (`png`/`jpg`/`gif`/`webp`/`bmp`) instead of binary noise, so
the model answers from what is actually in the picture rather than
guessing.

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
On the local CLI, an attachment is not transferred anywhere; Cozter prints
`[Attached file: /absolute/path]` so it can be opened locally.
Generated images under `$CODEX_HOME/generated_images` (or
`~/.codex/generated_images` when `CODEX_HOME` is unset) and any
directories listed in `COZTER_ATTACHMENT_ROOTS` are also accepted. Cozter
copies explicitly referenced external images into `.cozter/generated_images/`
before upload so chat platforms never receive arbitrary external paths. That
copy is published without clobbering an existing file: if a name collides,
including with a concurrent writer, Cozter tries readable `-2`, `-3`, …
suffixes (up to 1,000 candidates) and leaves the competing file untouched.
`COZTER_ATTACHMENT_ROOTS` is an OS-path-separator-delimited list (`:` on
POSIX, `;` on Windows); blank entries are ignored and `~` is expanded.
At the end of a run, Cozter also snapshots newly created or modified image
files in the workspace and attaches them unless the agent already referenced
them explicitly. Automatic discovery skips generated build, test, and cache
trees, including `__cmake`, `build`, `CMakeFiles`, `Testing`, and `cmake-build-*`.
It selects at most eight images totaling 20 MiB at selection time per execution
attempt across all judge phases and reports omitted previews for that attempt;
the files remain
in the workspace. Explicitly requested attachments remain eligible, including
images inside build trees, and do not consume the automatic preview allowance.
Shared external output directories always require an
explicit `[[attach: ...]]` marker. The workspace-only scan resolves image
links and ignores a link whose final target is outside the workspace, so a
trusted external artifact cannot become an automatic attachment merely by
being linked into a workspace. Replies can end with
`[[await]]` when the agent needs a user decision; the marker is stripped
and that user's queued work pauses until the next message arrives.
Texts (and the usage footer) send synchronously first; pictures then
upload in a per-chat background chain, so the turn returns as soon as
texts are out and the next queued turn's agent work overlaps the uploads.
The next reply waits at most five seconds for earlier uploads before sending
text. Attachment batches keep their per-chat order. `/cancel` and daemon
stop cancel in-flight uploads, and in-flight uploads count as active
turns in status and diagnostics.

## Plugins

The built-in HTTP-toolkit includes filesystem, shell, search, and fetch
tools: `bash`, `read_file`, `read_files`, `write_file`, `edit_file`, `multi_edit`,
`apply_patch`, `delete_file`, `copy_file`, `move_file`, `make_dir`,
`list_dir`, `tree`, `glob`, `grep`, `web_search`, and `web_fetch`.
For build/test/verify work the `bash` tool description directs the model
to run until completion or cancellation, capture output to a file,
grep the full log for error/warning/exception/traceback, and never claim
clean from a truncated tail-only preview.

Drop a `.py` file into `agent_tools/plugins/` and every agent discovers it
on next restart. Whether a backend can invoke it still follows its selected
permission mode. Files whose names start with `_` are skipped, which is useful
for disabled examples or local scratch tools. One file, two invocation paths:

The launcher exports its package parent through `PYTHONPATH` while preserving
existing entries, so a CLI working in the selected workspace can still import
`Cozter.agent_tools.plugins`. A direct plugin invocation from another directory
needs the same import path. On a development checkout, select that checkout's
parent instead of an inherited deployment path; on GitLab CI, retain the
temporary parent containing the `Cozter` symlink. See
[Development checks](#development-checks) for both layouts.

Treat plugins as trusted bot code: discovery imports their modules in the
Cozter process, so module-level code runs at startup and any dependencies must
be installed in the project environment. Restart after adding, removing, or
changing a plugin. Plugin calls have no global execution deadline. Cancellation
stops cancellable work promptly; an already-started mutation settles before its
owner returns. Trusted plugin code that blocks the event loop synchronously
can still stall the process and must isolate blocking work itself. Cozter does
not sandbox plugin code. CLI-backend plugins run through that CLI's own
shell/tool policy.

- **HTTP backends** (`llama`, `meta`, `zai`, and any future API backend) see plugins
  as typed tools in the chat-completions `tools` schema, alongside
  the 19 built-in tools under `agent_tools/builtin/` (`bash`, `read_file`, `read_files`,
  `write_file`, `edit_file`, `multi_edit`, `apply_patch`, `delete_file`,
  `copy_file`, `move_file`, `make_dir`, `list_dir`, `tree`, `glob`, `grep`,
  `web_search`, `web_fetch`, `list_skills`, and `read_skill`)
- **CLI backends** (`codex`, `claude_code`, `copilot`, `grok`) can't have
  external tools injected into their fixed toolkit. The bot
  instead lists each plugin in their prompt and tells the model to
  invoke it through the backend's own `bash` / `shell` tool as
  `python -m Cozter.agent_tools.plugins.<filename> '<JSON args>'`.

For ordinary HTTP execution, `deny` exposes no tools. The Flexible planner's
separate planning scope permits only `list_skills` and `read_skill` while
keeping all execution tools disabled. Under `/permission confirm`,
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

Plugins default to serial execution. A trusted plugin may declare
`parallel_safe = True` only when every supported action is an independent
read-only operation. The shipped `current_time`, `calculator`, `memory`, and
`git_info` plugins opt in; HTTP backends can run consecutive independent reads
in waves of at most four, preserving result order and durable barriers before
writes. This flag does not change permissions. CLI plugin guidance identifies
the same independent reads so native agents can group them with their own tools.
Run any plugin module with `--help` to inspect its full argument schema without
executing the plugin. Standalone plugins write UTF-8 to stdout and stderr,
including when Windows redirects their output through a non-UTF-8 locale.

The default HTTP result budget is 4,000 characters. Plugin code can declare
`max_result_chars` as a positive integer; the dispatcher limits it to 65,536
characters. Call arguments cannot raise this budget. Shipped Git plugins retain
their own 12,000-character results, and `http_request` allows 16,384 characters
for its advertised body preview plus response metadata. Plugins still bound
their own output and explicitly mark any omitted content.

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
`agent_tools/plugins/README.md` and the shipped plugins.

Shipped plugins:

- `current_time` - current date/time, optional IANA timezone.
- `calculator` - exact arithmetic through a whitelisted AST evaluator
  (`+ - * / // % **`, sqrt/log/trig, factorial, pi/e/tau/phi), with
  magnitude guards so `9**9**9` is refused instead of wedging the host.
  The 50,000-digit magnitude policy also applies to integer intermediates;
  multiplication refuses oversized products before allocating them.
- `notes` - persistent workspace notes at `.cozter/notes.md`
  (`append`/`read`/`clear`). Notes survive compaction, `/stop`, and
  restarts, so an agent can record progress and resume cleanly; the
  newest entries win when the 64 KiB ceiling forces a trim, and both
  over-long entries and tail-only reads are explicitly marked as
  clipped/preview output with PARTIAL + remainder pointers.
- `git_info` - read-only `status`/`log`/`diff`/`branches`/`tags`/
  `stashes`/`show`/`blame` inspection of the workspace repository, so
  HTTP backends without a shell can still see repo state; the argv is
  fixed and read-only, and `path` arguments must stay inside the
  workspace. Long results have a character `next_offset`: repeat the same
  request with that `offset` to fetch the remainder, restarting at zero after
  repository changes. `blame` can inspect a range with `line` and `line_count`
  (up to 200 lines) instead of requiring one call per line. Output capture
  retains only the requested bounded window while draining the child process.
- `git_ops` - local-only Git writes (`add`/`unstage`/`commit`/
  `checkout`/`branch-*`/`stash-*`/`reset`/`merge`/`rebase`/`tag-*`/
  `discard`/`clean`) with a fixed argv: branch/tag/ref names are
  validated, paths stay inside the workspace, `clean` needs explicit
  confirmation, and `push` never appears here. Available in `auto`.
- `git_sync` - network Git sync (`fetch`/`pull`/`push`/`remotes`) with a
  fixed argv, `--ff-only` pulls, and `--force-with-lease` instead of
  `--force`. Full permission only; terminal and credential-manager prompts stay
  disabled through process-local settings, while cached credentials remain
  available. Read-only Git operations use the same prompt policy.
- `memory` - search/read the workspace's durable chat memory (capped colony
  lists, excerpts, result counts, and session lists carry explicit
  PARTIAL/remainder pointers instead of silent cuts): past
  session transcripts, per-session summaries and long-term notes, and
  the shared colony notes (all under `.cozter/`). `search` is
  case-insensitive full-text with bounded excerpts, `list` shows
  sessions newest-first, `read` prints a session's recent messages by
  name, full id, id prefix, or `last`. Full IDs read only their own transcript;
  names, prefixes, and newest aliases resolve through lightweight ordering
  metadata and reload only the selected transcript. Search counts all matching
  sections while retaining one transcript at a time, then reloads only sessions
  intersecting the requested page. Created-time order, filename ties, section
  order, colony placement, and exact total counts are preserved. A selected
  transcript changing between passes requires a fresh page rather than mixing
  counts and excerpts from different file versions.
  Search and list also return continuation offsets, counting matches and
  sessions respectively. Limits remain search 20, list 40, and read 100.
  Read pages return an `offset` for older messages, counting only rows actually
  shown after fitting the output budget; repeat with that offset to reach older
  history, restarting from zero if messages change. Individual message excerpts
  remain bounded.
  Read-only and self-contained.
- `http_request` - call HTTP APIs on public hosts with any
  GET/POST/PUT/PATCH/DELETE, optional headers and body (Content-Type
  auto-sniffed for JSON). Reuses `web_fetch`'s public-network-only
  client and re-validates every redirect hop, and unlike `web_fetch`
  it shows 4xx/5xx bodies, where APIs explain their errors.

A plugin can also be run directly from the parent directory:

```bash
Cozter/.venv/bin/python -m Cozter.agent_tools.plugins.current_time '{"timezone":"Asia/Seoul"}'
```

HTTP error/status previews are clipped with the same visible marker
(`… [clipped]` reserved inside the budget), and transcript-retention-limit
notices state the retained transcript is a truncated preview. HTTP-backend tool results are capped before they are fed back into the
model, keeping accidental huge outputs from consuming the whole context.
Every cap reserves its truncation marker inside the advertised budget, so a
clipped result always arrives visibly marked as a PARTIAL preview with a
remainder pointer (for example `read_file` offset/limit/char_offset or `grep` to page
further) instead of a silent bare cut — and budgets too tight for the full
marker keep a visible ellipsis. `read_file` returns at most 32,768 characters
per call, including its continuation marker. `offset` still counts lines;
follow the returned `char_offset` with unchanged `offset`/`limit` to recover
the remainder of a long line or selected range. Its line-offset and
character-offset scans are bounded at 16 MiB so a pathological offset cannot leave a worker thread
walking an enormous file.
Session transcript lines (`session.format_msg_line`) carry the same honesty
contract: capped content keeps a `… [truncated N chars total … PARTIAL +
remainder]` marker with the suffix reserved inside the cap, and tight caps
keep a visible ellipsis. `list_dir` and `tree` skip an entry whose `is_dir()` raises (a dangling
symlink, ELOOP, or vanished file) instead of aborting the rest of the listing.
`grep` only opens regular files up to 1 MB and runs its regex scan in a
killable worker process that cancel always reaps; a slow scan simply keeps
running until it finishes or the turn is cancelled, so narrow the pattern
or search path if a query is too broad.
The `web_search` and `web_fetch` tools also cap downloaded response bodies
at 5 MiB and share the bounded `read_bounded_text()` reader in
`agent_tools/base.py`, which reports whether the cap was hit so the model
sees an explicit PARTIAL preview marker instead of a silent cut
(`web_fetch` appends a fetch-capped note; `web_search` marks the capped page
and notes when only the first N results are shown). The fetch-capped
footer itself is one shared `with_fetch_cap_marker()` helper in
`agent_tools/base.py`, used by both `web_fetch` and the `http_request`
plugin so their wording stays in sync. `web_search` starts DuckDuckGo's `html`
and `lite` frontends concurrently, with two attempts per frontend and a short
retry delay. It returns the first usable result page and cancels the remaining
request, so a stalled frontend cannot delay healthy results. Sponsored (`ad_*`)
and DuckDuckGo-internal links never become results.
Title and search-anchor extraction use forward-only, quote-aware scanners in
`agent_tools/html_scan.py`. Malformed opening tags and missing closing tags do
not trigger repeated whole-suffix regex searches. Pure formatting and result
parsing run on helper threads over the existing 5 MiB response cap; network
contexts close on cancellation while that bounded CPU work can finish without
side effects.
`web_fetch` instead uses a public-network-only client with redirect targets
validated individually, preventing redirects into private addresses.
One HTTP session serves both fetch attempts so the retry reuses the
connection pool instead of paying for a fresh connector + DNS validation pass.
Inside an HTTP agent worker, `web_fetch` and `http_request` also borrow a lazy
worker-owned connector across calls, bounded to eight connections total and
four per host. Every call has its own session, cookies, and custom headers;
workers never share pools. Connector DNS caching stays disabled, and every new
connection still passes through the public-address resolver. The exact owned
connector and resolver close on success, error, or cancellation, including
repeated cancellation. Standalone CLI plugin calls retain their one-call
session lifetime. `web_search` keeps its separate frontend request helper.
Transient transport failures (connection reset, resolver hiccup)
are retried once with a short delay; HTTP error statuses, refused content
types, and redirect-loop or non-public redirect targets are final and are
never retried. It
classifies `text/*`, HTML, JSON, and XML media types case-insensitively, so
servers that capitalize labels such as `Text/Plain` or `APPLICATION/JSON`
still return readable content. The reader consumes chunked or slow responses
until EOF or that ceiling instead of treating a short network read as the
complete body; `web_fetch` then applies its separate `max_chars` output
limit.
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
   file edits — on the agent and model bound to its difficulty tier.
   Tasks run in order unless the planner explicitly marks consecutive tasks
   `[parallel-read]`: these inspect independent scopes together, with the
   reports from completed earlier tasks. Changes and follow-up tasks run
   serially; one later writer can repair and verify related findings together.
3. **Merge.** The summary agent folds the workers' reports into the single
   reply you see. The workers' own text never reaches the chat; their tool
   and file events still stream into the live status display.

`[parallel-read]` means both read-only and independent of the other consecutive
marked tasks. For example, inspecting existing transport logs and reviewing an
unrelated parser can run together before one repair worker consumes both reports.
Inspection workers do not edit files, build, run tests, launch processes, commit,
push, or expand their assignment into the full original task. Their reports give
the later writer concrete findings and evidence. The runtime uses concurrency
only for backends that can enforce the inspection restrictions; other backends
run the marked work serially. An unmarked task or a dependency ends the group.

`flexible_max_parallel_workers` in `.config/config.json` limits this inspection
concurrency to **1-4**, with **3** as the default. Set it to **1** for serial
execution. Missing or invalid values use the default. Ordinary plans without
markers retain their ordered behavior. The planner prefers the fewest workers
that cover the request and uses saved checkpoints to avoid repeating completed
audits, revalidating only the evidence needed for remaining work.
The active plan, completed reports and worker cursor are saved before another
worker starts. A later execution allowance resumes pending work without
another planning call or replaying completed workers. Consecutive follow-up
obligations assigned to the same tier are combined into cohesive workers;
all individual obligations and overflow remain explicit. Duplicate pending
instructions are removed, while verification after changes and stronger-tier
escalations remain distinct work.
Fallback workers retain a full checkpoint request once, with an explicit
reference instead of a second verbatim copy. Exactly identical long reports
share their retained evidence while preserving every plan-item header and
blocked marker. Judge continuations retain the prior draft before the exact
current request suffix, preventing the planner from appending that request
again. Distinct requirements and evidence remain present.

The grading rubric the planner is held to:

| Tier | When | Example |
|---|---|---|
| `low` | Straightforward, well-scoped work with clear intent | Add a small validation check, or extend an existing function with clearly defined behavior |
| `mid` | Some reasoning required, but the problem stays bounded | Write unit tests for an existing method with known inputs and outputs |
| `high` | Only when the task involves ambiguity, complex logic, or deeper system understanding | Refactor a system with unclear dependencies, or debug a non-obvious issue |

Each tier carries its own agent, model and reasoning effort, so tiers can straddle
backends — a local `llama` for the easy parts, `claude_code` on `opus` for
the hard ones:

```
/agent_flexible_low     -> claude_code        /model_flexible_low   -> haiku
/agent_flexible_mid     -> codex              /model_flexible_mid   -> gpt-6-sol
/agent_flexible_high    -> claude_code        /model_flexible_high  -> opus
```

Set each tier's effort independently:

```text
/effort_flexible_low 25
/effort_flexible_mid 60
/effort_flexible_high 100
```

Without an argument, each command opens a picker. A tier inherits the workspace
`/effort` setting until you override it; `default` restores that inheritance.
An explicit `0` asks that tier's backend to use its own default. Only `100`
selects the chosen model's highest supported effort. Overrides are stored in
`.cozter/settings.json` as `flexible_low_reasoning_effort`,
`flexible_mid_reasoning_effort` and `flexible_high_reasoning_effort`.
These settings apply to task workers; planner, merge and other internal calls
retain their backend defaults. Each plan snapshots its tier efforts so parallel
workers use one consistent policy. Effective effort changes invalidate worker
checkpoints for that policy; adding the feature preserves checkpoints when all
tiers still inherit their existing workspace effort.

Defaults put all three tiers on `codex` (`gpt-6-luna` / `gpt-6-sol` /
`gpt-6.1-sol`), with `gpt-6.1-sol` for chat and `gpt-6-luna` for summaries.
These defaults follow the [current Codex model guidance](https://developers.openai.com/codex/models)
and the installed CLI catalog (verified 2026-10-18 against codex-cli 0.163.0
and grok 1.0.52; codex's `gpt-5.5` retired from the live catalog on 2026-10-14 and is no longer in the fallback; Claude picker adds Haiku 5.5,
released 2026-10-07).
Model access depends on the account and client; select an available model on
older or company-managed installations. Existing saved model choices are retained. Pointing a
tier at another agent picks that agent's cheap/mid/strong models
automatically (its `tier_models` table) — for `zai` that is
`glm-5.3-flash` / `glm-4.7` / `glm-5.3`, and for `meta`
`muse-spark-1.2` / `muse-spark-1.3` / `muse-spark-1.3`. `/model` and
`/doctor` print the current wiring. A tier can only point at a *direct* backend — never at
`flexible` itself, which would plan forever.

A flexible turn can make one planner call (history and request normally fit
within 12,000 characters, including a middle-context omission marker).
A request or durable checkpoint that already exceeds that allowance is sent
intact once, without an additional duplicate history excerpt. It can then make
up to 12 planned worker calls plus worker-discovered follow-ups
(each worker may report genuine leftovers as `[followup:<tier>]` lines,
all of which are parsed and de-duplicated). Distinct follow-ups join the queue
in reported order up to an execution cap of 24 worker tasks for that plan,
including the initial tasks. Progress reads `done/total` with a growing total,
e.g. `3/12` then `3/13`. Any further distinct follow-ups are retained in an explicit
PARTIAL remainder listing the tasks that were not run. The turn then makes one
merge call over the bare request. Each worker report is capped at 6,000
characters with its truncation marker reserved inside the cap, so a clipped
report always arrives marked `[report truncated: remainder omitted]` as a
preview rather than a silent cut. A single-subtask plan skips the merge
and returns the worker report directly. When the planner hits the 12-task
cap, the parsed plan carries an explicit PARTIAL + remainder note unless the
planner already wrote one. Because tiers may use different
backends, a single request can
also be sent to multiple configured providers. Select a direct backend when a
single-provider path or more predictable request cost is important.
The merge step checks every plan item against the reports before claiming
done: partial, missing, or failed reports force a PARTIAL + remainder answer.
A truncated report remains a preview; an omitted middle alone does not prove
unfinished work. Completion still requires retained scope and verification
evidence, and missing evidence forces PARTIAL + remainder. Doc/standard
requests are never accepted from first/last-few coverage alone.
Build/verify requests are claimed clean only when every report lists its
commands + exit codes + error/warning counts from full (not preview) logs
with no unexpected failures remaining; expected negative-path diagnostics,
test names, platform skips, and historical entries are classified separately.
Missing evidence means PARTIAL + remainder.
Workers reuse verified build and test evidence when the relevant inputs and
environment remain unchanged, and rerun checks invalidated by later changes.
They use the project's supported parallel jobs for independent compilation or
tests, but serialize configure/build commands sharing generated dependency state.
Each attempt also runs a continue-judge loop (up to 3 immediate follow-ups).
When the judge still says CONTINUE after that limit, the job saves its
progress and resumes in the same session with a fresh allowance. There is no
overall continuation-count limit that discards unfinished work. An unavailable
or malformed completion verdict does not count as DONE. `/stop` explicitly
cancels the job and its queued work; `/inject` interrupts its active phase to
apply new context. A question ending with `[[await]]` keeps the current job
pending; replying supplies guidance and resumes it ahead of later jobs.
A judge can also return `BLOCKED` when required work needs a missing external
prerequisite, such as a visible desktop for explicitly requested graphics checks.
After other actionable work is finished, Cozter retains the unfinished job's
evidence, missing prerequisite, and next recovery action. It automatically tries
authorized remediation or rechecks availability with a durable delay increasing
from 30 seconds to at most 5 minutes. The first unfinished job keeps its place;
later queued jobs remain held. A restart retains the checkpoint and retry time.
Completed audits and builds are reused when still valid, rather than repeated
because the prerequisite remains unavailable. Required work is never marked
complete merely because it is blocked.

Routine recovery uses the request's existing authorization and configured
permissions, including full permission when set; it does not ask for approval
to continue. Missing credentials or facts only the user can provide remain an
honestly stated information requirement. Cozter completes everything else it can
and checks whether the prerequisite has become available; it never invents
credentials or claims unavailable access. A genuine explicit `[[await]]` question
still requests necessary information, instead of approval for routine work.

Two behaviors are worth knowing. Under `/style collaborative`, the turn can
stop and wait for you (`[[await]]`) at either end of the pipeline: the
planner may ask **one** clarifying question instead of guessing, and the
merge step may end its reply on a question it genuinely needs answered
before the work can continue. The workers in between never stop to ask,
since nobody is reading them mid-pipeline. Under `/style autonomous` — and
on scheduled `/reserve` runs, which are always autonomous — routine work continues
without waiting for approval:
a question the merge model emits anyway is stripped rather than left to
strand a run nobody is watching. Every agent turn preamble now also carries the whole-scope /
doc-following / verify-with-evidence completeness rule (list every target first, do each,
re-check leftovers; enumerate every rule/section when told to follow a
doc; build/test/verify means enumerate targets, run canonical commands
until completion or cancellation, captured to a file, grep the
full logs for error/warning/exception, classify each hit, repair unexpected
failures, and re-run the affected checks,
exercise runtime paths and check logs, and report commands + exit codes +
counts; never claim done with work left or clean from a truncated preview
or unrun steps — say PARTIAL + remainder), and
composed context trims carry matching omission markers within budget so
the model cannot mistake a trimmed preview for full coverage. And when planning fails outright (summary
CLI missing, unparseable output), the turn degrades to a single `high`-tier
sub-task carrying the original request, so a botched split never quietly
downgrades hard work to a weak model.
The continue judge requests another turn only for concrete missing work that
can run in the current environment. It does not expand the task to provision
unconfigured providers or verify unavailable platforms unless requested.

## Backend behavior

Each backend defines its own model list and permission mapping in
`backends_agent/`. `flexible` is omitted — it owns no CLI and no model of
its own, only the three tiers above.

| Backend | Launch path | Default chat model | Default summary model |
|---|---|---|---|
| `codex` | `codex exec --ephemeral --json` | `gpt-6.1-sol` | `gpt-6-luna` |
| `claude_code` | `claude --print --output-format stream-json --verbose` | `default` | `haiku` |
| `copilot` | `copilot --output-format json --no-color` | `auto` | `auto` |
| `grok` | `grok --prompt-file … --output-format streaming-messages-json` | `grok-4.7` | `grok-4.7` |
| `llama` | Unauthenticated OpenAI-compatible `/v1/chat/completions` | `auto` | `auto` |
| `meta` | Meta Model API `…/v1/chat/completions` (Bearer) | `muse-spark-1.3` | `muse-spark-1.2` |
| `zai` | Z.ai `…/api/paas/v4/chat/completions` (Bearer) | `glm-5.3` | `glm-4.5-air` |

When a caller does not explicitly select a summary model, Cozter uses the
selected summary backend's `default_summary_model` for routing, planning,
compaction, and session titling. This gives direct library callers the same
summary-model defaults that workspace settings use.

Codex and Grok discover their visible local CLI catalogs, while Copilot queries
its authenticated ACP model selector from the selected workspace and fails closed
to `auto` if that catalog cannot be read. This keeps enterprise-disabled and
workspace-policy-disabled Copilot models out of the picker; a stored Copilot
choice also uses `auto` until it appears in that workspace's fresh catalog.
Claude Code has no safe non-interactive account catalog, so it keeps
a curated list that `extra_models` can extend. Its picker offers standard
aliases, the supported `sonnet[1m]`, `opus[1m]`, `fable[1m]`, and
`opusplan[1m]` long-context aliases, and verified version pins (including
Fable 5.1, Fable 5, Haiku 5.5, Sonnet 5.5, Sonnet 5, Opus 5.5, Opus 5, `claude-sonnet-5-5[1m]`, `claude-sonnet-5[1m]`, and explicit
`[1m]` variants of other documented long-context models).
Only the explicit `[1m]` selections receive Cozter's 1M-token context metadata —
plus the 1M-native `claude-haiku-5-5` bare pin, which has no `[1m]` suffix;
aliases and bare version pins remain capacity-unknown because their active
window can vary by account and provider. Claude Code's own `/fast` is a
session toggle, not a Cozter command or a selectable `*-fast` model ID. Llama,
Z.ai, and Meta Model API discover models live from their configured HTTP
endpoints.
`llama`, `meta`, and `zai` share one in-process OpenAI-compatible agent loop
(`backends_agent/_openai_agent.py`); `zai` just adds the Bearer auth header
and points at Z.ai's endpoint; `meta` adds Meta's Bearer header and compat
endpoint. `meta` and `zai` model discovery also share that module's Bearer
`/models` probe (`discover_bearer_models`: non-HTTP(S) catalog-URL refusal,
no-key fallback, Bearer fetch,
provider non-chat filter, empty-catalog fallback) and its shared key health
check.
That loop reuses one HTTP session for the
turn's provider requests and retries. Consecutive trusted builtin reads and
explicitly opted-in read-only plugins run in groups of at most four, with
results retained in request order. Writes, shell/build commands, mixed-action
plugins, and undeclared plugins form serial barriers;
the loop advertises `parallel_tool_calls` so the model may batch independent
calls in a single turn. Z.ai's text chat-completion models from GLM-4.6
onward, plus multimodal `glm-5.3-flash`/`glm-5.3-flashx`, opt into Z.ai's incremental
tool-call argument stream; older vision models use their standard streamed
function-call deltas because that vision request schema does not accept
`tool_stream`. The shared SSE parser merges
either shape before executing a requested tool.
The builtin `read_files` tool reads up to 16 independent file ranges in one
model round trip, with at most four simultaneous reads, up to 32,768 characters
per file preview and a 65,536-character aggregate cap shared fairly across the
requested files. A free read slot starts the next file immediately while
results retain request order. Each truncated file retains its `char_offset`
cursor within its assigned budget; use it with `read_file` and the same
`offset`/`limit` to retrieve the remainder. Complete files and batches have no truncation marker. The
dispatcher retains up to 32,768 characters for builtin `read_file`, `glob`, and
`grep`, 32,768 for `web_fetch` (including its URL/title), and 65,536 for
`read_files`. Discovery pages preserve complete matches
and return the exact next `offset` when more results remain; offsets count
matches in deterministic directory traversal order. Repeat the same search
with that offset, or restart at zero if files change. Exact result limits do
not imply a remainder unless a lookahead finds another match. Other tools and
plugins retain the 4,000-character default unless their code explicitly declares
another bounded budget. All backends receive guidance to batch independent
inspections and
reuse still-valid verification evidence without weakening completion checks.

Filesystem mutations (`write_file`, `edit_file`, `multi_edit`, `apply_patch`,
`copy_file`, `move_file`, `delete_file`, and `make_dir`) run on worker threads,
so disk writes and patch matching do not block status updates or unrelated
reads. Each call retains ownership until its worker finishes. Cancellation waits for an already-started mutation to settle before returning;
later operations in the same tool sequence cannot overtake it. This preserves
ordered writes and existing atomic replacement behavior. The notes plugin uses
the same ownership rule for append and clear, and serializes their complete
transactions across CLI processes using a lock keyed by the canonical notes
path. A cancelled owner retains the lock until its mutation finishes.

Permission modes are backend-specific because a chat bot cannot answer a
per-tool-call approval dialog. `codex` uses bypass only for `full`, its
workspace-write sandbox for `auto`, and a read-only sandbox for `confirm`
and `deny`. `llama`, `meta`, and `zai` run in-process: ordinary `deny` execution exposes no tools and
`confirm` exposes only read-only tools. `auto` permits Cozter's
workspace-bounded built-ins plus plugins that do not declare themselves
full-only with `requires_full_permission`; both it and `confirm` block the
built-in direct host shell again at execution time. `full` enables the direct
host shell and removes the workspace boundary for built-in filesystem tools,
including absolute paths and paths that resolve outside through `..` or
symlinks. Operating system permissions and tool-specific plugin restrictions
still apply. HTTP agent prompts state the effective permission on every turn,
so the model can follow changes without assuming a workspace sandbox in
`full`. `claude_code` uses bypass only for `full`, `acceptEdits` for `auto`, and
plan mode for `confirm`/`deny`.
`copilot` uses `--yolo` only
for `full`, `--allow-all-tools` (while retaining path and URL checks) for
`auto`, and an explicit empty tool list for `confirm`/`deny`.
`grok` uses `--always-approve` only for `full`. `auto` still auto-approves
tools so a headless turn cannot hang on a permission prompt, but confines
them to Grok's workspace sandbox. Grok's native `auto` classifier is a TUI
feature that this CLI currently ignores in headless runs, so Cozter does
not send `--permission-mode auto`. `confirm`/`deny` use `dontAsk` with the
read-only sandbox and only `read_file`, `grep`, and `list_dir` exposed.
Grok still shows its MCP discovery/execution pair under an allowlist, so
those calls are also denied explicitly.
Internal router, titling, and compaction calls always use `deny`, so conversation
content cannot elevate their permissions. For ask-before-acting behavior on
any backend, use `/style collaborative` — it pauses the turn (via
`[[await]]`) for your reply instead of relying on a CLI approval flow.

The `llama` model picker queries `llama_server_url/v1/models` and falls
back to `auto` if the server is down or returns no model IDs. The `zai`
picker queries the configured Z.ai `/models` endpoint and retains its curated
agent-capable fallback, including text-compatible multimodal models such as
`glm-5v-turbo`, `glm-4.6v`, and `glm-4.5v`, if the account cannot be queried.
The normal Z.ai fallback includes the general endpoint's default
`glm-5.3` model and multimodal `glm-5.3-flash`/`glm-5.3-flashx`. A Coding Plan base URL also
surfaces `glm-5.3[1m]` and `glm-5.3-flash[1m]`. Cozter maps `/effort` for
both families to Z.ai's supported `low` / `high` / `max` reasoning levels, preserves
their opaque reasoning state between tool calls, and enables their documented
tool-call streaming.
It filters Z.ai's known image, OCR, and audio-only IDs because those require
different endpoints, while preserving unknown/private chat-model IDs. Codex,
llama, Z.ai, and Grok refresh their live catalogs lazily when a model picker is
opened after its 60-second cache expires, so long-running services see CLI,
server, and account model changes without a restart. Grok and the
OpenAI-compatible HTTP backends share that fail-closed TTL cache
(`CachedModelCatalog` in `backends_agent/base.py`). Codex and Copilot keep
their own caches because they store extra per-model metadata or
workspace-scoped policy catalogs. HTTP catalog responses
over 1 MiB use the backend's normal fallback; otherwise Cozter de-duplicates
the IDs, keeps at most 4,096, and ignores IDs longer than 512 characters.
Copilot applies the same 512-character-ID and 4,096-model bounds to its
authenticated ACP catalog, and stops parsing a malformed group tree after
16,384 option nodes or 16 nested groups before it reaches a model picker or
workspace cache.
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
Its prompt keeps the leading system preamble (including the whole-scope /
doc-following / verify-with-evidence coverage rule) and the complete composed
context intact through redirected stdin.
Its picker accepts ACP's provider-grouped model selectors and
`models.availableModels` session metadata, so account-approved models stay
visible without a hard-coded catalog. The `copilot` backend supplies its prompt through the
CLI's [piped-prompt support](https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-programmatic-reference),
omitting `-p`. File-backed stdin sends the exact UTF-8 bytes and EOF without
trimming for platform argv limits or passing newlines, quotes, and shell
metacharacters through Windows `.cmd` forwarding. The shared launcher returns
before the child reads stdin, so captured startup output cannot deadlock
prompt transfer. Foreground Copilot workers use a persistent private CLI home
owned by their Cozter conversation and isolated native identity. Its session
history survives completed jobs and later turns, and can be inspected through
Copilot with that home selected. Cozter passes `--no-remote-export`, so these
local sessions are not exported to GitHub web and mobile. Internal planner,
merge, and other maintenance calls still use short-lived homes and remain
stateless. Each private home copies `config.json` and `settings.json` from
`$COPILOT_HOME` when it is set, otherwise from `~/.copilot`; set `COPILOT_HOME`
before launch when the source profile lives elsewhere. Short-lived homes are
remembered by process-object identity (`ProcessResourceMap` in
`backends_agent/base.py`), not PID, so concurrent processes cannot clobber
each other's cleanup after PID reuse. A persistent home is removed when its
Cozter conversation is deleted, rather than when one process exits.

On Windows, recognized standard npm Node shims launch their bundled or
PATH-resolved Node interpreter and existing `node_modules` script directly.
This preserves arguments such as the empty restricted tool list when the
installation path contains spaces, and avoids an extra command-processor
hop. Unreadable, missing, or custom batch launchers retain their existing
`cmd.exe /c` fallback.

Codex uses discovered effort and context-window metadata only while its
60-second catalog cache is fresh. Until `/model` refreshes an expired cache,
known public models use Cozter's built-in metadata and a previously discovered
private model has no inferred context window, so the `/compact` message-
interval safeguard applies. An explicit `model_context_windows` entry remains
authoritative. That built-in Codex fallback (verified 2026-10-18 against
codex-cli 0.163.0, matching the live `codex debug models` catalog) lists the `gpt-6.1-sol` / `gpt-6-astra` / `gpt-6-sol` / `gpt-6-luna` quartet first in live-listed order — Astra shipped
on the OpenAI API on 2026-09-04 with the same 272K active Codex window as
the GPT-5.6 family (`gpt-6.1-sol` runs a 500K active window; Luna the fast
affordable tier, also 272K active) — followed by
the GPT-5.6 Sol/Terra/Luna family (`gpt-5.5` retired from
Codex with ChatGPT sign-in on October 14, 2026 and is `hide` in the live catalog); `gpt-5.3-codex-spark` is
gone from the live catalog, and `gpt-5.4` and `gpt-5.4-mini` were retired
from Codex ChatGPT sign-in. Grok's published `grok-4.7`, `grok-4.7-build-fast`,
`grok-4.6`, and `grok-4.5` IDs use a 500K-token
window for that same trigger; custom or private Grok models stay unknown
until an operator sets `model_context_windows`. Grok delivers its prompt
through `--prompt-file` rather than `-p`, so Cozter's history budget is not
truncated by the platform argv limit. The prompt file is a private temp path
keyed the same way as Copilot's CLI home, so concurrent Grok turns cannot
reuse or delete each other's files.

Provider event envelopes are treated as untrusted input. A missing, blank, or
non-text backend error message is normalized to `Unknown error` before it is
stored or shown, rather than exposing a provider object or breaking the turn
parser. Claude Code and Grok apply the same Messages-style terminal `result`
helper (`apply_terminal_result_event`), so usage, cost, fallback text, and
late errors stay consistent across those two CLIs. They also share
`messages_content_texts()` to flatten a Messages-style assistant `content`
value — a bare string or a list of typed text blocks — into reply text.
`apply_messages_assistant_content()` walks that envelope for live turns, and
`extract_messages_style_agent_text()` prefers the terminal result for
internal summary calls. Grok and Copilot share `summarize_cli_tool()` for
compact tool-status lines (command, then path). If a backend has
already streamed an assistant reply, a late stream or terminal error is
retained on the turn without replacing that reply.

## Reasoning effort

`/effort` accepts `0`–`100` and is stored per workspace. `0` means "no
override — server defaults apply"; `1`–`100` are explicit. Each backend
maps the percentage to its own vocabulary and request shape:

Only `100` explicitly selects a model's highest supported effort level.
Values `1`–`99` keep the existing lower-level cutoffs and stop at the next
level down. This applies to CLI and HTTP backends, including Flexible task
workers. For Opus 5.5, `60`–`99` selects `xhigh` and `100` selects `max`.
The highest level may have another name: for example `ultra` on supported
Codex models, or `high` on a four-level HTTP model. A model with only one
explicit level receives it only at `100`; lower values leave its default.
Models without effort controls and GLMs with only a thinking on/off switch
retain their native behavior.

| Backend | Bands | What gets sent at 100% |
|---|---|---|
| `codex` | Model-aware: 4–6 levels | `ultra` (Astra/Sol/Terra), `max` (Luna), or `xhigh` (others) |
| `llama` | 4 levels; highest reserved for `100` | `payload["reasoning_effort"] = "high"` |
| `meta` | Standard Muse Spark 1.3: 6 levels (`minimal` through `max`); older and Contributor models: 5 levels (`minimal` through `xhigh`) | `payload["reasoning_effort"] = "max"` for standard 1.3; `xhigh` for older and Contributor models |
| `zai` | GLM-5.3/Flash/FlashX: 3 levels; GLM-5.2: 7 levels; other GLMs use documented thinking behavior | `payload["reasoning_effort"] = "max"` |
| `claude_code` | Model-aware: current Fable / Haiku 5.5 / Sonnet 5.5 / Sonnet 5 / Opus 4.7+ use 5 levels; Opus 4.6 and Sonnet 4.6 use 4; Opus 4.5, Haiku and older Sonnet pins use their defaults | `--effort max` for supported current models |
| `copilot` | 6 levels (`minimal` through `max`) for an explicit model; `auto` delegates to Copilot | `--effort max` for an explicit model; omitted for `auto` |
| `grok` | Model-aware: grok-4.7 / grok-4.7-build-fast / grok-4.6 use 4 levels; grok-4.5 and unknown models use 3 | `--effort xhigh` on 4.7/4.6; `--effort high` otherwise |

The setting applies only to user-facing chat turns. Internal calls
(compaction, routing, titling, colony consolidation) skip the effort
parameter, so utility work stays cheap regardless of the workspace
setting. Flexible task workers use their tier's `/effort_flexible_low`,
`/effort_flexible_mid` or `/effort_flexible_high` override when present,
and inherit `/effort` otherwise. Flexible planner, merge and judge calls
also keep their backend defaults.

## Interaction style

`/style` chooses how collaborative the agent is on chat turns, stored per
workspace:

- `collaborative` (default) — when a request is ambiguous or before a
  large, destructive, or hard-to-reverse action, the agent asks a short
  question and ends with `[[await]]`, pausing the queue until you reply.
  Small, reversible choices are made without asking. This is a
  backend-agnostic prompt policy, so it steers every backend (codex,
  copilot, claude_code, grok, llama, meta, zai) the same way — not just the CLIs that
  ask on their own.
- `autonomous` — the agent decides and proceeds without asking, closer to
  a full-auto run.

Scheduled `/reserve` turns run in throwaway sessions that cannot pause for
a reply, so they always use the autonomous policy regardless of this
setting. They remain queued behind an unfinished job waiting for a
collaborative answer; replying resumes that job before later scheduled
or ordinary chat work.

## Architecture

```
Cozter/
├── __init__.py           package marker
├── __main__.py           entry point; sets PYTHONPATH; runs the bot
├── launch_job.py         Windows launcher job owns descendant lifetimes
├── process_launch.py     shared subprocess launch and cancellation ownership
├── requirements.txt      Python runtime dependencies installed into .venv
├── py.typed              marks the package as typed for downstream checkers
├── pyproject.toml        Ruff E4/E7/E9/F lint contract
├── .config/              runtime config dir; only config.example.json is tracked
├── backends_bot/         chat surfaces and shared fenced-Markdown formatting
├── agent.py              orchestrator: builds prompt, runs backend, streams events and attachments
├── session.py            per-workspace conversation persistence
├── execution_state.py    isolated job checkpoints and durable native conversation state
├── native_prompt.py      structured bootstrap and incremental native CLI context
├── skills.py             flat workspace skills, shared native catalogs, and owned bridges
├── compaction.py         scratch-summary + long-term-memory rewriter
├── colony.py             cross-session long-term memory consolidation
├── router.py             session picker for first-turn-in-workspace (subsequent turns reuse last_session.json)
├── titling.py            auto-titles new sessions from their first turn
├── schedules.py          /reserve cron-style scheduled prompts
├── flexible.py           flexible meta-agent prompt construction + plan/merge parsing
├── queue_store.py        transactional platform queue body plus schedule receipts (SQLite)
├── workspace.py          per-workspace settings; containment-checked nested .cozter dirs
├── config.py             global .config/config.json reader
├── updater.py            git fetch + restart loop
├── utils.py              shared state, queue, lock, marker-block, and backend-process helpers
├── tests/                unittest coverage for commands, state, queues, schedules, compaction, backends, catalogs, flexible, prompts, tools, attachments, and updates
│
├── backends_agent/       agent backends (one file per agent)
│   ├── base.py             abstract Backend; convert_effort, supports_typed_plugins;
│   │                       shared catalog TTL cache, fallback tables,
│   │                       ProcessResourceMap, CLI tool summaries, and
│   │                       Messages-style result and content helpers
│   ├── codex.py            wraps `codex exec`
│   ├── claude_code.py      wraps `claude --print`
│   ├── claude_background_guard.py
│   │                       session-only Claude Bash hook that blocks
│   │                       untracked background launches
│   ├── copilot.py          wraps `copilot`; persistent native homes, ProcessResourceMap for stateless homes
│   ├── grok.py             wraps `grok --prompt-file`; CachedModelCatalog and
│   │                       ProcessResourceMap for prompt files
│   ├── flexible.py         flexible meta-agent backend (no CLI of its own)
│   ├── _http_process.py       process-like adapter and error handling for HTTP backends
│   ├── _openai_agent.py    shared in-process OpenAI-compatible agent loop;
│   │                       one HTTP session per turn; shared catalog cache,
│   │                       Bearer discovery probe, and key health check
│   ├── llama.py            local /v1/chat/completions backend hooks
│   ├── zai.py              Z.ai /api/paas/v4/chat/completions backend hooks
│   └── meta.py             Meta Model API /v1/chat/completions backend hooks
│
└── agent_tools/          tool surface for HTTP backends + plugin registry
    ├── base.py             AgentTool ABC; path/argument validation, shared HTTP
    │                       helpers, and the shared fetch-cap PARTIAL footer
    ├── discovery.py        bounded whole-match pages with continuation offsets
    ├── html_scan.py        forward-only title and search-anchor extraction
    ├── owned_worker.py     blocking mutations with cancellation settlement
    ├── builtin/            20 files: 19 built-in tools (bash, read_file, read_files, write_file, edit_file, multi_edit, apply_patch, delete_file, copy_file, move_file, make_dir, list_dir, tree, glob, grep, web_search, web_fetch, list_skills, read_skill) plus __init__.py
    └── plugins/            user drop-in zone (current_time, calculator, notes, git_info, git_ops, git_sync, memory, http_request shipped live; _git_common.py holds the git tools' shared runner/validation/truncation and is skipped by the loader)
```

## Process and tool safety

Cozter owns the lifetime of every backend process it starts. User-facing
turns drain stderr concurrently with streamed JSON events, and every exit
path — normal completion, cancellation, an injected restart, event-parse
failure, or chat-delivery failure — reaps the child process and its drain
tasks. This prevents a failed callback or `/stop` from leaving an agent CLI
running in the background. CLI prompts supplied through stdin use an OS-managed
temporary file, so launch returns without waiting for the CLI to consume the
prompt before its output readers can start. This avoids a startup deadlock when
a CLI writes more output than its pipes can hold before reading its input.
The child receives the same UTF-8 bytes and EOF; the temporary backing disappears
when its handles close. If launch is cancelled while the OS is creating the
process, Cozter keeps the temporary input alive until creation settles and
reaps any child before returning cancellation, including repeated cancellation
requests. Noninteractive Git, update, package-install, and CLI model-catalog
commands use closed stdin so they cannot contend with the terminal's input
reader or consume the next chat command. Copilot's ACP catalog handshake instead
owns a dedicated protocol pipe. Shell commands also watch process exit independently
of output EOF: after the shell exits, they allow one second to drain buffered
output before closing a pipe still held by a background child and marking the
captured output as potentially incomplete. Git tools use the same one-second
post-exit allowance for stdout and stderr together. They drain both streams
concurrently, retaining bounded previews even for verbose hooks or large
diffs; excess output does not stop an ongoing Git operation. Running commands continue until completion or cancellation.
Live chat status streams mark their own trims too: only the newest few
status lines are kept (older updates flagged omitted), over-long reply
previews carry a `… [preview]` marker, and long tool-content lines and
labels use the shared clipped marker rather than a bare cut.
Tool status lines share `utils.clip_status_value()`, so compact
summaries (paths, args, stderr previews, calculator/git/notes/http plugin
statuses) clip with a visible marker instead of a silent cut. Backends that
stage temporary launch resources (Grok's prompt file or a stateless Copilot
home) remember that path by process-object identity and delete it in
`cleanup_process` after the child is reaped. Conversation-owned Copilot homes
survive process cleanup and are removed only with their Cozter conversation.

Attachment, background-request, and saved-upload markers use forward scans
that preserve their existing captures and whitespace rules. Unfinished markers
and long whitespace runs cannot trigger repeated whole-suffix searches.
External-image attachment preparation retains its ordered validation and copy
on an owned worker, settling an in-flight copy before cancellation unwinds.

Internal LLM jobs (routing, session titling, compaction, and colony
consolidation) all go through `utils.run_internal_backend()`. The shared
runner imposes no deadline on these jobs. A low-level caller that explicitly
requests a positive operation timeout applies it to startup and response
together. The runner consumes stdout and stderr without pipe deadlocks, kills
cancelled children, and logs stderr when a backend exits without an assistant
response. Streamed
backend events retain an already-buffered final backlog, including when chat
delivery is slow. Once the parent exits, the remaining backlog has a byte
limit and a stalled reader has a short drain allowance, so continuously ready
stdout cannot bypass cleanup.
HTTP backends expose the same
process-shaped contract through `backends_agent/_http_process.py`, so the
orchestrator uses one cleanup model for CLI and API agents.
Its local event transport uses compact UTF-8 JSON and a separately marked
32 MiB frame budget, enough for the JSON escaping of existing 4 MiB text or
tool-argument limits. Lone JSON surrogates retain their escaped representation.
Locally owned HTTP streams drain to their guaranteed EOF, so already completed
responses cannot be dropped by native descendant-pipe guards. Native CLI line
and post-exit limits retain their existing bounds. An oversized local frame
fails explicitly before any bytes are added to the stream.
HTTP task cancellation stays on its owning event loop. Native Windows
process-tree cleanup runs blocking `taskkill` in a worker thread so the loop
remains responsive; POSIX group signals run directly on the loop.

Foreground session logging and interrupted-turn logging also run durable
file writes outside the event loop. A background owner keeps the workspace
file lock until the write finishes, including after cancellation of the turn.
An interrupted-turn fallback waits for any normal write's outcome, so a
successful completed reply is recorded once and a failed write can still
leave the interrupted attempt in session history.
Subsequent turns and detached launches wait for those write owners before
reading session context, so a resumed request sees the committed memory.

Durable queue, reply-ledger, and detached-task ledger operations, and temporary scheduled-session
creation and deletion, also perform file I/O outside the event loop. Once an
atomic operation starts, it finishes under its file lock before cancellation
unwinds the caller's admission or delivery locks. Reply bookkeeping follows
the completed write before cancellation is propagated, preserving FIFO and
restart recovery. A cancelled scheduled-session creation still removes the
temporary session.
Cancellation retains the user's admission lock through durable cleanup and
blocks new detached notifications until that cleanup ends. Incoming work can
then enter the queue in order without being removed by the earlier cancellation.

The built-in file tools also fail closed at workspace boundaries. In
particular, an `apply_patch` create patch uses an atomic no-clobber link where
the filesystem supports it, with an exclusive-create fallback: it cannot
overwrite a file another writer creates concurrently, and a failed preparation
leaves no partial target. New patch-created files use normal file-creation
permissions (subject to the process umask); a delete patch must match the
current file and remove all of its content before the file is unlinked. Failed
hunks leave the target in place. Files in a multi-file patch are applied
independently: an I/O failure is reported against its target while earlier
outcomes remain visible and the remaining files are still attempted.
Each built-in file path is checked both lexically and after resolution, so an
existing symlink cannot escape the workspace. Content tools follow a safe
in-workspace final symlink; `delete_file` and `move_file` instead operate on
that final link itself, while still rejecting a link that resolves outside the
workspace.
Every patch hunk requires a complete unified-diff header of the form
`@@ -start[,count] +start[,count] @@`; an omitted count means one line.
The body must match both declared counts, and line zero requires a zero
count. Bare, malformed, truncated, or overfull headers/hunks reject the entire
patch before any target is written. Optional section context follows the
closing `@@` after a space or tab.
Likewise, a lone file header, a file header without a hunk, or extra hunk-body
content beyond its declared counts rejects the entire patch before any target
is touched.
`apply_patch` caps one incoming diff at 1 MiB and 20,000 lines, and caps each
existing or resulting target at 1 MiB and 50,000 lines. The relevant target is
left unchanged when one of those bounds is exceeded. `edit_file` and
`multi_edit` similarly refuse source text above 5 MiB, including if the file
grows after its initial size check.
Patch parsing also preserves the requested final-newline state from a unified
diff's `\ No newline at end of file` marker. A misplaced marker or a creation
hunk that claims old-side/context lines is rejected rather than silently
changing the patch's meaning.
`write_file`, `edit_file`, `multi_edit`, and updates to existing files through
`apply_patch` replace content through a temporary file. An unsuccessful
replacement leaves the prior contents intact, and existing file mode bits are
preserved.
`copy_file` only copies files, while `move_file` can move a file or directory
but refuses to move a directory into its own subtree. `copy_file` stages
through a same-directory temporary copy (creating a missing destination
parent first), fsyncs through a read-write handle so Windows Python builds
that reject read-only syncs still persist the bytes, then atomically
publishes a new destination
(with an exclusive-create fallback where hard links are unavailable), so a
concurrent file or symlink can never be overwritten. `move_file` publishes
regular files and symlinks without clobbering a late destination, then removes
the source only after that publication is safe; if that final removal cannot
be completed, it rolls back only the target it created. For directories and
other non-file paths, it uses the OS's no-replace rename primitive and fails
where that guarantee is unavailable rather than falling back to a replacing
rename. Both operations refuse to replace an existing destination and create
a missing destination parent directory only after their source and destination
checks pass.
The string-edit tools honor a broad `replace_all` operation only when its
tool argument is the literal JSON boolean `true`; malformed truthy values
retain the safer unique-match behavior.
For existing UTF-8 text files, `edit_file`, `multi_edit`, and modification
patches normalize CRLF while matching their line-oriented input, then restore
the file's original CRLF convention on write. Replacement text that already
contains CRLF is normalized before restoration, so an edit cannot introduce
duplicate carriage returns. Non-UTF-8 files are refused rather than rewritten
with replacement characters.
Regression coverage for these paths lives in
`tests/test_agent_process_cleanup.py`, `tests/test_prompt_subprocess.py`,
`tests/test_cli_commands.py`, `tests/test_launch_job.py`, `tests/test_utils.py`,
`tests/test_agent_tools.py`, and `tests/test_discovery_tools_async.py`.

CLI JSONL and OpenAI-compatible HTTP SSE use the same bounded line reader in
`utils.iter_bounded_lines()`. Each transport retains at most 4 MiB for one
physical line. CLI JSONL discards an oversized line and resumes at the next
newline, so a later valid event can still be processed. An oversized HTTP SSE
line discards the entire completion and uses the configured retry budget,
before any buffered tool calls execute, so losing an argument fragment cannot
change a tool call's meaning.

OpenAI-compatible backends also bound the retained state for one streamed
completion: 4 MiB of assistant text, 4 MiB of retained reasoning content,
4 MiB of arguments for any one tool call, 8 MiB across retained text,
reasoning content, and tool arguments, 512 characters each for a tool-call
ID and name, and 128 tool calls. A per-completion limit breach is handled as
a retryable failure before any buffered tool call executes. Tool-argument
fragments are joined only after a completion finishes, which avoids repeated
copying as a long streamed argument arrives.
If a provider ends with `finish_reason: "length"` while a tool call is
pending, Cozter uses its configured retry budget rather than executing
potentially partial arguments; a text-only length-limited reply is delivered
as truncated text. Across a tool-using HTTP agent run, the retained system,
user, assistant, tool, and continuation messages are also capped at 32 MiB.
Cozter refuses a message that would exceed that total and asks the user to
narrow the task or reduce tool output; if the assistant's requested tool-call
message itself cannot be retained, its tools are not run.

Persisted session state is treated as recovery data rather than trusted input:
malformed last-session pointers, unsafe session IDs, and session files whose
embedded ID does not match their filename are ignored. A damaged or
hand-edited state file therefore falls back to normal session selection rather
than reaching outside `.cozter/sessions/`. Workspace discovery retains normal
`**` glob behavior with an iterative matcher, so repeated globstars avoid
recursive stack limits and a cache of every equivalent match state.

## Source inventory

The tracked workspace is intentionally flat and small. A complete source
audit should use `git ls-files` so hidden tracked files, especially
`.config/config.example.json`, are included even though `.config/*` is
ignored for local secrets and runtime queues.
The runtime contains 77 Python source files, separate from the tests below.

- Package entry and runtime setup: `__init__.py`, `__main__.py`,
  `launch_job.py`, `process_launch.py`, `config.py`, `updater.py`, and `utils.py`
- Conversation, memory, and workspace state: `agent.py`, `workspace.py`,
  `session.py`, `router.py`, `titling.py`, `compaction.py`,
  `colony.py`, `flexible.py`, `execution_state.py`, `native_prompt.py`, `skills.py`, `schedules.py`, and `queue_store.py`
- Chat-platform adapters: `backends_bot/__init__.py`, `base.py`, `formatting.py`,
  `cli.py`, `telegram.py`, `slack.py`, and `signal.py`
- Agent adapters: `backends_agent/__init__.py`, `base.py`, `_http_process.py`,
  `_openai_agent.py`, `codex.py`, `claude_code.py`,
  `claude_background_guard.py`, `copilot.py`, `grok.py`, `flexible.py`,
  `llama.py`, `zai.py`, and `meta.py`
- Agent tool surface: `agent_tools/__init__.py`, `agent_tools/base.py`,
  `agent_tools/owned_worker.py`, `agent_tools/_http_types.py`,
  `agent_tools/_public_resolver.py`,
  `agent_tools/discovery.py`, `agent_tools/html_scan.py`,
  the 20 files under `agent_tools/builtin/` (`__init__.py`, `apply_patch.py`,
  `bash.py`, `copy_file.py`, `delete_file.py`, `edit_file.py`, `glob.py`,
  `grep.py`, `list_dir.py`, `list_skills.py`, `make_dir.py`, `move_file.py`, `multi_edit.py`,
  `read_file.py`, `read_files.py`, `read_skill.py`, `tree.py`, `web_fetch.py`, `web_search.py`, `write_file.py`),
  and the 11 files under `agent_tools/plugins/` (`__init__.py`,
  `_git_common.py`, `calculator.py`, `current_time.py`, `git_info.py`,
  `git_ops.py`, `git_sync.py`, `http_request.py`, `memory.py`, `notes.py`,
  plus its `README.md`)
- Project metadata, CI, and docs: `requirements.txt`, `py.typed`, `mypy.ini`,
  `pyproject.toml` (the Ruff E4/E7/E9/F lint contract), `.gitlab-ci.yml`,
  `.github/workflows/ci.yml`, `.config/config.example.json`,
  `run_cozter.ps1` (the Windows Task Supervisor launcher used by the update
  restart path), `.gitignore`, and this README
- Tests (164 files: `tests/__init__.py`, `tests/conftest.py`, shared `tests/helpers.py`, plus 161 `test_*.py` modules:
  `test_agent_attachments.py`, `test_agent_image_flood.py`, `test_agent_plugins.py`,
  `test_agent_post_turn.py`, `test_agent_process_cleanup.py`, `test_agent_prompt.py`,
  `test_agent_tools.py`, `test_apply_patch_header_contract.py`, `test_apply_patch_limits.py`,
  `test_attachment_delivery_latency.py`, `test_attachment_preparation_ownership.py`,
  `test_backend_audit_regressions.py`, `test_backend_event_parsing.py`, `test_backend_repair.py`,
  `test_backends_agent.py`, `test_bash_process_output.py`, `test_bash_progress.py`,
  `test_blocked_judge_runtime.py`, `test_bot_commands.py`, `test_bot_recovery_and_lifecycle.py`,
  `test_calculator_responsiveness.py`, `test_canonical_platform_state.py`, `test_checkpoint_scope.py`,
  `test_claude_background_detection.py`, `test_claude_background_guard.py`,
  `test_claude_model_capabilities.py`, `test_cli_commands.py`, `test_cli_native_continuation.py`,
  `test_cloud_model_capabilities.py`, `test_colony_streaming.py`, `test_compaction.py`,
  `test_config.py`, `test_copilot_catalog_cleanup.py`, `test_detached_tasks.py`,
  `test_discovery_canonical_root.py`, `test_discovery_optimization.py`, `test_discovery_paging.py`,
  `test_discovery_pruning.py`, `test_discovery_tools_async.py`, `test_durable_poll_parallelism.py`,
  `test_edit_file_read_limits.py`, `test_effort_maximum.py`, `test_execution_state.py`, `test_flexible.py`,
  `test_flexible_continuation.py`, `test_flexible_effort.py`, `test_flexible_injection_continuation.py`,
  `test_flexible_instruction_identity.py`, `test_flexible_native_history.py`,
  `test_flexible_native_injection_continuation.py`, `test_flexible_native_sessions.py`,
  `test_flexible_parallel_plan.py`, `test_flexible_parallel_runtime.py`,
  `test_flexible_prompt_sections.py`,
  `test_format_marker_latency.py`, `test_git_checkout_integrity.py`, `test_git_cold_launch.py`,
  `test_git_plugin_paging.py`, `test_git_process_output.py`, `test_grok_model_line_latency.py`,
  `test_http_body_sniff.py`, `test_http_event_framing.py`, `test_http_execution_continuation.py`,
  `test_http_latency.py`, `test_http_owned_mutation_cancellation.py`,
  `test_http_pending_checkpoints.py`, `test_http_redirect_method.py`, `test_http_restore_accounting.py`,
  `test_http_skill_workflow.py`, `test_image_dimensions.py`,
  `test_image_snapshot_metadata.py`, `test_inject.py`, `test_interactive_schedule_refresh.py`,
  `test_internal_backend_deadline.py`, `test_interrupted_turns.py`, `test_job_continuation.py`,
  `test_job_prerequisite_retry.py`, `test_judge_blocked.py`,
  `test_launch_job.py`, `test_launcher_and_followups.py`, `test_lazy_tool_http_imports.py`,
  `test_maintenance_context.py`, `test_memory_corrupt_state.py`, `test_memory_preservation.py`,
  `test_memory_repair_regressions.py`, `test_memory_streamed_search.py`,
  `test_monthly_schedule_execution.py`, `test_monthly_schedules.py`,
  `test_native_conversation_state.py`, `test_native_current_protocol.py`, `test_native_delta_sync.py`,
  `test_native_launch_contract.py`, `test_native_prompt.py`, `test_native_session_turns.py`,
  `test_native_skill_launch.py`,
  `test_notes_owned_worker.py`,
  `test_notes_transactions.py`, `test_openai_agent.py`, `test_owned_file_mutations.py`,
  `test_patch_limit_optimization.py`, `test_planner_context_latency.py`, `test_platform_formatting.py`,
  `test_plugin_cli_unicode.py`, `test_plugin_memory_calculator_performance.py`,
  `test_plugin_parallelism.py`, `test_plugin_result_budgets.py`, `test_prerequisite_checkpoint.py`,
  `test_prompt_payload_dedup.py`, `test_prompt_subprocess.py`, `test_protocol_marker_latency.py`,
  `test_public_http_pool.py`, `test_queue_persistence_latency.py`, `test_queue_store.py`, `test_read_file_character_cursor.py`,
  `test_read_file_offset_chunks.py`, `test_read_files.py`, `test_reply_delivery.py`, `test_run_lock.py`,
  `test_runtime_diagnostics.py`, `test_schedule_execution.py`, `test_session_picker.py`,
  `test_session_write_latency.py`, `test_signal_account_config.py`,
  `test_signal_attachment_decoding.py`, `test_signal_feedback.py`,
  `test_signal_formatting.py`, `test_signal_inline_latency.py`, `test_signal_literal_runs.py`,
  `test_signal_send_observability.py`, `test_signal_span_projection.py`, `test_signal_status.py`,
  `test_signal_status_recovery.py`, `test_signal_transport.py`, `test_slack_commands.py`,
  `test_skill_prompt_integration.py`, `test_skill_tools.py`,
  `test_slack_formatting.py`, `test_state_fallbacks.py`, `test_state_integrity.py`,
  `test_status_and_shell_latency.py`, `test_status_latency.py`, `test_thinking_display.py`,
  `test_titling.py`, `test_tool_permissions.py`, `test_tool_runtime.py`,
  `test_tool_safety_and_output_limits.py`, `test_tool_state_integrity.py`,
  `test_tools_audit_regressions.py`, `test_tree_junction_permissions.py`, `test_turn_budget.py`,
  `test_unbounded_tool_http_execution.py`, `test_update_background_admission.py`,
  `test_updater.py`, `test_upload_limits.py`, `test_utils.py`,
  `test_vision_marker_scanning.py`, `test_web_fetch.py`, `test_web_html_latency.py`,
  `test_web_search.py`, `test_workspace_skills.py`, `test_yearly_schedules.py`)
  covering config validation, agent attachments, prompts,
  process cleanup, interrupted-turn persistence, and post-turn behavior;
  backend model defaults (Claude and cloud provider capabilities), shared
  catalog/result/content helpers, CLI tool summaries, process-resource maps,
  event parsing, and the shared OpenAI-compatible retry/backoff loop; bot,
  CLI, and Slack commands, including canonical native APIs/protocols, strict
  unified-patch headers, exact platform ownership, explicit compaction snapshots,
  unbounded tool/provider execution with cancellation, native stdin ownership,
  native conversations retained across completed jobs, shared direct/Flexible
  editing lanes, isolated parallel reader histories, exact job scope and
  model/policy isolation, missing-context selection before history budgeting,
  durable first-activity receipts, exact user-only Flexible acknowledgments,
  changed-context delivery, and full bootstrap only on
  confirmed pre-work session loss; progressive workspace/native skill catalogs,
  bounded skill loading, native bridge ownership, planner-only skill tools, and skill-aware saved-plan
  continuation without replaying completed workers; restored-queue
  FIFO/reference preservation, weekly/monthly/yearly/daily schedule parsing and presentation,
  short-month and leap-year dates, yearly catch-up, mixed-rule deduplication, and schedule edits
  while waiting for a lock;
  scheduled-job admission versus execution; Windows launcher descendant
  lifetimes; compaction and durable memory preservation across partially
  covered maintenance input; the flexible meta-agent including all reported
  follow-ups and the 24-worker (`MAX_SUBTASKS * 2`) execution cap plus the
  per-attempt worker/tool-count limits with durable continuation of unfinished jobs before
  later queue entries, including restart recovery and blocked-work handling;
  inject (including the bounded confirmation send); run locks, session picking, and
  auto-titling; platform, Slack,
  and Signal rich-text formatting; durable reply delivery; detached tasks
  and Claude's background-launch guard; runtime diagnostics; state
  integrity and fallbacks including containment-checked nested workspace dirs;
  status latency (including chat status I/O, shell/Signal latency, and
  thinking-status display) and Signal transport/status/attachment-decoding
  regressions; native shell/Git subprocess output draining with inherited
  pipes, prompt-subprocess delivery, unbounded internal/backend execution with
  explicit operation-timeout coverage and cancellation; updater behavior; utilities
  including the shared lock helper and image-dimension probing; upload limits; tool permissions; and
  the built-in/plugin tool surface, including patch/edit bounds, output-limit
  and state-integrity regressions, off-event-loop discovery scans that continue after
  a failed `is_dir()`, and discovery pruning/optimization that preserves
  file selection, containment, and permission checks

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
  `backends_agent/__init__.py` plus the concrete backend modules.
  Shared catalog TTL, fallback tables, Messages-style terminal result
  handling, assistant content walking and summary-text extraction,
  `ProcessResourceMap`, and CLI tool-status summaries live in
  `backends_agent/base.py`. Shared lock and `[TAG]` marker-block helpers
  live in `utils.py`
- Flexible's tiers, grading rubric, planner/merge prompts, and plan
  parsing: `flexible.py`; its orchestration loop lives in
  `agent.py:_run_flexible()` and its per-tier settings in `workspace.py`
- Tool/plugin behavior: `agent_tools/__init__.py`, `agent_tools/base.py`,
  `agent_tools/builtin/`, and `agent_tools/plugins/README.md`; shared
  validation, workspace-boundary checks, no-clobber file publication, HTTP
  request setup, and bounded response reading live in `agent_tools/base.py`;
  the git tools' shared subprocess runner, ref validation, truncation marker,
  and stderr/output finishing live in `agent_tools/plugins/_git_common.py`
- Workspace, session, queue, schedule, compaction, and colony state:
  `workspace.py` (including `ensure_workspace_state_dir()`), `session.py`,
  `schedules.py`, `compaction.py`, and `colony.py`. Hot JSON reads
  (`config.json`, workspace state/settings, colony, last-session pointers,
  and per-workspace schedules) are cached by file mtime+size via the shared
  `utils.stat_mtime_size()` helper, so repeated per-turn/per-tool reads skip
  re-parsing until an edit changes mtime/size
- CI and local quality gates: `.gitlab-ci.yml`, `.github/workflows/ci.yml`,
  `mypy.ini`, `pyproject.toml`, and `tests/`

The agent loop in `agent.py:run()` is shared across backends. Each
`Backend.launch()` spawns the right subprocess or starts an in-process
OpenAI-compatible HTTP session; `agent.py:_drive_backend()` reads JSONL
events from stdout, translates them via `Backend.parse_event()` into
`ChatEvent`s, and streams a "Thinking..." status message that updates in
place with the latest few tool actions and a live preview of the answer
text as it arrives. Interim progress is Thinking... status only, never a
chat reply: flexible worker progress reads `done/total` with a growing
total as `[followup:<tier>]` leftovers join live (e.g. `3/12` then
`3/13`), PARTIAL + remainder appears only in the final reply when work
truly remains, and the user is notified DONE only when everything is
done. On chat surfaces without editable messages (the CLI),
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
  `.config/queue_<platform>.sqlite3`, `.config/reply_deliveries_<platform>.json`,
  `.config/queue_<platform>.json.migrated.*`, SQLite journal files,
  and `.config/detached_tasks_<platform>.json`
  - local tokens, workspace selections, persisted pending messages, staged
  replies, and detached-task state. Platform IDs are sanitized for queue
  filenames, so a runtime key like `cli:local` becomes a filesystem-safe
  `queue_cli_local.sqlite3`.
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

When an update is available, Cozter waits for a natural idle point while
schedulers, queued jobs, and incoming work continue normally. Busy work can
defer an update; it does not freeze other queues or schedule admission.
At the idle point, Cozter closes execution gates and checks activity again.
If new work arrived during that handoff, it reopens the gates before waiting
again. It fast-forward-pulls and installs changed `requirements.txt` only
after confirming that no active turn won the race.

The final handoff keeps newly accepted messages durable until the restart.
Explicit `/bg` launches ask for a retry during that short handoff. An aborted
handoff reopens every prepared platform, including one whose preparation was
interrupted. Git and dependency operations finish before cancellation can
release their gates. Once the checkout changes, installation failure still
requires a restart so the startup dependency check can repair it; the old
runtime must not resume against the changed source. On POSIX, the
daemon then re-execs itself in place. On Windows, it exits for the bootstrap
or `run_cozter.ps1` supervisor to relaunch it. Manual pulls and local commits
while the bot is running also trigger this safe restart path. A service
manager such as `systemd` with `Restart=always` remains useful for boot-time
startup and unexpected exits. CLI mode uses an outer respawner process and
relaunches itself in the same terminal. Daemon platforms restore persisted
queues after either path starts again; CLI mode currently does not.
If a message is accepted during the final update handoff, Cozter keeps that
already-persisted message queued and does not start its agent turn against a
checkout that is about to change.

## Runtime diagnostics

`__main__.py` writes rotating warning/error logs to `.log/cozter.log`.
It also writes metadata-only stage timings to `.log/performance.log`:
planner, worker, merge and judge starts and durations, including worker position;
HTTP completion durations and
retry counts; time to the first content, reasoning or tool delta; and tool
starts, individual durations/outcomes, and batch durations and counts.
Signal send outcomes correlate the hashed destination and text with the
acknowledged message timestamp, or identify cancellation, failure, and missing
timestamps. An RPC acknowledgement confirms submission to signal-cli; it does
not establish that a recipient displayed or read the message. Plaintext
messages, destination IDs, and exception contents are excluded from these entries.
Mixed successful and failed recipient results are counted separately and
reported as `acknowledged_with_failures`; logging them does not resend a
message that some recipients may already have received.
Quiet shell commands emit a `stage=tool_wait` entry every 60 seconds with the
process ID and elapsed time. This heartbeat leaves the command and its output
capture running. Waits for earlier attachment uploads record whether they
settled or reached the five-second limit.
Tool failures record their outcome without imposing an execution deadline.
Prompts, responses, tool arguments and API keys are excluded. The timing log
rotates at 5 MiB with three backups.
HTTP provider connection attempts are bounded at 30 seconds. Generation and
retry backoff have no global elapsed-time deadline; retries still respect their
configured count. A low-level caller's explicit positive operation timeout,
when supplied, covers the complete operation including retries and backoff.
Recognized successful completion events retain their final content and tool
arguments before finishing; a stalled trailer cannot turn completed work into
an unbounded trailer wait or retry. Requests use equivalent compact JSON, and plain
text prompts avoid the vision-preparation thread hop.
Normal completion trailers get a short drain (at most 50 ms and 64 KiB) to
preserve connection reuse; the drain is skipped near the request deadline.
Flexible worker prompts retain their full actionable instruction once instead
of repeating it in the plan overview. The merger receives each complete plan
instruction once; numbered report headers refer back to those plan items.
Bounded report and continuation draft excerpts keep their opening and final
evidence; an omitted middle alone does not imply unfinished work, and completion
still requires retained evidence for the requested scope.
Background compaction and titling run as one pipeline per session. A later
turn's maintenance waits for the earlier pipeline, then checks current state,
so it can use the compactor's title without launching a competing title request.
These background waits do not delay the foreground answer.
Maintenance starts in a fresh task context with its inputs passed explicitly,
so it cannot retain a finished execution's worker transcripts or worker allowance
while waiting for the session pipeline or a summary response.
When a worker reports an explicit exhausted quota or authentication failure,
later plan items for that exact backend and model are marked unrun for the
current turn. Other models and providers remain eligible. Temporary throttling
and task-specific tool refusals do not suppress later workers.
File discovery matches the requested glob before resolving targets. Bounded
permission modes still check matching targets for symlink escapes, while full
mode avoids unused containment work. Discovery also prunes directories that
cannot match the pattern's fixed prefix or nonrecursive depth; recursive
globstars retain every eligible branch.
Repeated-only tool rounds and three consecutive entirely blocked rounds force
a final report from the available evidence, while useful tool work can continue.
On Windows, the Python entry process owns a process-lifetime job before it
spawns module or virtual-environment replacements. Stopping the registered
Python launcher also terminates those descendants, so a restart replaces the
old instance instead of leaving it active alongside another one.
Unhandled exceptions also get timestamped crash files in `.log/`
(newest 50 retained; older ones are pruned automatically), and
asyncio/thread dumps go to `.log/diagnostics.log`. On Unix-like hosts,
send `SIGUSR1` to the running daemon process to dump tasks, thread
stacks, and per-platform active-turn state without restarting it.

Asyncio debug mode uses the normal `slow_callback_duration` of **0.10 seconds**
on Python 3.11, 3.12, and 3.14; Cozter does not raise that
threshold. Debug mode can be enabled with `PYTHONASYNCIODEBUG=1`, and
`unittest.IsolatedAsyncioTestCase` enables it for its test loops. An
`Executing ... took ... seconds` warning measures a callback or task step
holding the event loop at least that long. Inspect the reported location for
synchronous blocking work and correlate it with `.log/performance.log` stage
timings. A long asynchronous wait for a backend can leave the loop responsive;
the slow-callback warning and a whole-stage duration measure different work.

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
4. `backends_agent/llama.py`, `backends_agent/zai.py`, and
   `backends_agent/meta.py` for concrete HTTP backend hooks
5. `agent_tools/__init__.py` for the auto-discovery and plugin
   bridging

The CLI-backend files (`codex.py`, `claude_code.py`, `copilot.py`,
`grok.py`) are thin: each defines `launch()` (build argv, spawn
subprocess) and `parse_event()` (translate the CLI's JSONL events to
`ChatEvent`s).

## Development checks

Run the current unit tests from the parent directory, or set
`PYTHONPATH` to the parent when running inside the repository. The
[source inventory](#source-inventory) describes the complete test coverage.

If `codex` is on `PATH`, one catalog-consistency test also invokes
`codex debug models` with a 15-second timeout; it skips when that command
fails, times out, or returns no visible models. To avoid that optional CLI
probe, run the suite with Codex absent from `PATH`.

When Git is available, a native CLI fixture checks that `/version` responds
while its input pipe remains open, then exits on EOF. Prompt-transfer,
cancellation, and Windows launcher-lifetime fixtures also launch isolated
test children; they do not start configured chat services.

CI runs the same three gates — `ruff check`, `mypy`, and the `unittest`
suite — on Python 3.11 and 3.12 for pushes to `main` and merge requests /
PRs. The canonical pipeline is
`.gitlab-ci.yml` (GitLab CI, the primary remote); `.github/workflows/ci.yml`
mirrors it on GitHub. mypy is enforced on the core runtime, agent backends,
and tool surface; `tests/` are excluded and the chat-platform adapters are
temporarily error-suppressed for their untyped SDK interactions, as defined
in `mypy.ini`.

GitLab runners may check out this repository under a name other than `Cozter`.
The canonical pipeline creates a temporary directory containing
`Cozter -> $CI_PROJECT_DIR` and sets `PYTHONPATH` to that temporary directory.
Keep that import alias in child subprocess environments as well as the test
runner's environment. Resolving a test file with `Path(__file__).resolve()`
follows the symlink to the physical checkout, whose parent might not contain
an importable `Cozter` package. When a child needs an additional package
parent, prepend it without discarding the intentionally configured inherited
`PYTHONPATH` entries. An import check in both the runner and its child should
resolve to this checkout. Local shells should first remove stale deployment
entries as described below.

For a local developer check, create the project venv with Python 3.11 or
3.12. `requirements.txt` contains runtime dependencies only, so install the
CI tooling explicitly before running the lint, type, and test gates:

```bash
cd ..
python3.12 -m venv Cozter/.venv  # or python3.11; skip if it already exists
Cozter/.venv/bin/python -m pip install -r Cozter/requirements.txt ruff mypy
Cozter/.venv/bin/ruff check Cozter
Cozter/.venv/bin/mypy --config-file Cozter/mypy.ini --python-version 3.11 -p Cozter
Cozter/.venv/bin/mypy --config-file Cozter/mypy.ini --python-version 3.12 -p Cozter
Cozter/.venv/bin/python -m unittest discover -s Cozter/tests -v
git -C Cozter diff --check
```

From inside `Cozter/`, run the suite with
`PYTHONPATH=.. .venv/bin/python -m unittest discover -s tests -v`.

The two `mypy` invocations mirror CI's Python 3.11 and 3.12 target matrix. To
exercise the full CI matrix locally, run the `unittest` command from separate
3.11 and 3.12 virtual environments as well. Before committing, check `git
status --short` as well, to ensure only intentional source or documentation
edits are staged. Runtime JSON, logs, sessions, virtualenv files, and caches
should stay local.

### Stale `PYTHONPATH` on Windows checkouts

On Windows, a shell that inherits `PYTHONPATH=C:\Startup` can import
`C:\Startup\Cozter` instead of the development checkout, even when the
working directory is `C:\workstation\cozter`. Running from inside the package
does not place its parent on the import path. An import mismatch can make
current tests exercise old production code and report unrelated failures.

Replace the inherited value with the development checkout's parent, with that
checkout as the working directory. Use the Python 3.11 or 3.12 virtual
environment selected for the gate, rather than assuming the global `python`
has the intended version. For example, from a checkout with that venv:

```powershell
$env:PYTHONPATH = "C:\workstation"
Set-Location -LiteralPath "C:\workstation\cozter"
$cozterCheckPython = ".\.venv\Scripts\python.exe"
& $cozterCheckPython --version
& $cozterCheckPython -c "import Cozter; print(Cozter.__file__)"
& $cozterCheckPython -m ruff check .
& $cozterCheckPython -m mypy --config-file mypy.ini --python-version 3.11 -p Cozter
& $cozterCheckPython -m unittest discover -s tests -v
```

Repeat the gates with a separate Python 3.12 environment and
`--python-version 3.12` for its mypy invocation. The import check must resolve
inside `C:\workstation\cozter` for this example; Windows may preserve different
letter casing in the printed path.
Python 3.14 can be checked with a third environment and
`--python-version 3.14`; the repository CI matrix remains Python 3.11/3.12.

A shell launched by a service can inherit that service's deployment
`PYTHONPATH`. Check the effective value and `Cozter.__file__` before trusting
results, and use a development shell with a clean import path. On POSIX the
same check applies to inherited deployment paths; use `PYTHONPATH=..` inside
a normally named checkout, or explicitly set it to the intentional import
alias parent when following the GitLab symlink layout.
