# Cozter user plugins

Drop a `.py` file here to add a new tool to every agent. One file
serves both modes:

- **HTTP backends** (`llama`, `zai`, future Mistral/Gemini/etc.) see plugins
  as typed tools — same schema as the built-in toolkit in `../builtin/`.
- **CLI backends** (`codex`, `claude_code`, `copilot`, `grok`) can't have
  external tools injected, so the bot tells the model about each
  plugin in the system prompt and the model invokes it via the
  backend's built-in `bash` tool:

  ```bash
  python -m Cozter.agent_tools.plugins.<name> '<JSON args>'
  ```

The bot exports `PYTHONPATH` on startup so the `-m` import resolves;
the CLI subprocess already runs with `cwd` set to the workspace, so
the plugin's `os.getcwd()` returns the right path.

## Conventions

- File name `foo.py` → tool name typically `foo` (match `cls.name`).
- Tool names are global. Choose a unique `name`: a later plugin registration
  replaces an earlier tool with the same name, including a built-in.
- File names starting with `_` are skipped by the loader, which is useful
  for disabled examples or local scratch plugins.
- Shipped live plugins: `current_time.py`, `calculator.py`,
  `notes.py` (persistent workspace notes at `.cozter/notes.md`),
  `git_info.py` (read-only `status`/`log`/`diff`/`branches`/`tags`/
  `stashes`/`show`/`blame` inspection with a fixed, non-mutating argv),
  `git_ops.py` (local-only `add`/`commit`/`checkout`/`branch`/`stash`/
  `reset`/`merge`/`rebase`/`tag`/`discard`/`clean` with validated names
  and workspace-bounded paths; `auto`-visible), `git_sync.py`
  (`fetch`/`pull`/`push`/`remotes` with `requires_full_permission`),
  `memory.py` (search/read past session
  transcripts, summaries, long-term notes, and colony items under
  `.cozter/`), and `http_request.py` (GET/POST/PUT/PATCH/DELETE against
  public hosts with web_fetch's public-address safety model; shows
  4xx/5xx bodies).
- The shipped `current_time.py` plugin is live and can be invoked through
  the same module path pattern as any other plugin.

## Permissions and safety

Plugins are trusted code. An HTTP backend in `/permission auto` exposes a
plugin unless its class sets `requires_full_permission = True`. Set that flag
for any capability that can access host resources outside Cozter's
workspace-bounded tool model, such as arbitrary paths, commands, credentials,
or sockets. The flag only controls HTTP-tool availability; it does not
sandbox plugin code. Plugin calls carry the 3600s real-work cap. Cancellation
stops cancellable work promptly; notes append and clear first wait for an
already-started disk mutation to finish, so a subsequent operation cannot race
an abandoned write or deletion. Their complete transactions also share a
canonical-path lock across CLI processes, preventing simultaneous appends
from overwriting one another. CLI-backend plugin scripts follow that
CLI's own shell/tool policy instead.

## Throughput and result bounds

`parallel_safe = True` opts a plugin into bounded HTTP execution waves of at
most four calls. Declare it only if every supported action is an independent
read-only operation. The default is false; mixed-action tools and mutations
remain ordered barriers. The shipped `calculator`, `current_time`, `memory`,
and `git_info` tools opt in. Permission checks remain unchanged. CLI plugin
guidance identifies the same independent reads for native agents.

Plugin code can declare a positive integer `max_result_chars` to retain more
than the default 4,000-character HTTP result. Overrides are capped at 65,536
characters and cannot be chosen through model arguments. Git plugins declare
12,000; `http_request` declares 16,384 for its body preview and response
metadata. Each plugin must still bound its own work and show any remainder.

`git_info` returns `next_offset` for long output. Repeat the same request with
that character `offset`, and restart at zero if the repository changes. Blame
also supports `line_count` up to 200, starting at `line` or 1. Git output
capture retains a bounded character window while fully draining the subprocess.
Both inspection and sync disable terminal/credential-manager prompts locally
to the child process and continue to use cached credentials.

`memory` reads an exact session ID directly instead of loading unrelated
transcripts. Names, prefixes, and newest aliases retain lightweight ordering
metadata and reload only the selected transcript. Search counts all matching
sections one transcript at a time and renders excerpts only from sessions
intersecting the requested page. Ordering and exact counts remain unchanged;
changed selected transcripts require a fresh page. Search/list support
`offset` paging by matches/sessions, with limits up to 20/40 respectively.
Read pages return the next `offset` for older messages, counting
only the rows actually emitted after the output budget is applied. Repeat with
that offset and the same session, restarting from zero if messages change.
Individual message excerpts retain their existing bound.
HTTP-worker `http_request` calls borrow that worker's checked connection pool,
with a separate session and cookies/headers per call. New connections still
resolve through the public-address checker with DNS caching disabled. Pools
close with their owner, including cancellation; standalone CLI plugin calls
retain their original one-call session lifetime.
`calculator` checks its existing 50,000-digit magnitude policy
at every integer intermediate and before allocating oversized products.
Its bounded validation, parsing, evaluation, and formatting run on a worker
thread so large valid arithmetic does not block the event loop. Cancelling
the caller may leave pure computation finishing briefly, with no filesystem
or network side effects; arithmetic limits and results are unchanged.

For any plugin module, `--help` (or `-h`) prints its argument schema and
metadata without executing it. This lets CLI agents discover enum choices and
paging arguments without reading the implementation or retrying guessed calls.

## Template

```python
"""Plugin: <one-line description>."""

from __future__ import annotations

from ..base import AgentTool


class MyTool(AgentTool):
    name = "my_tool"
    description = "What this does, from the model's perspective."
    # Set True for a host-escaping capability; otherwise auto may expose it.
    # requires_full_permission = True
    parameters = {
        "type": "object",
        "properties": {
            "thing": {"type": "string", "description": "..."},
        },
        "required": ["thing"],
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        # workspace_path is the user's current workspace (cwd in script mode).
        return f"got: {args.get('thing')}"


if __name__ == "__main__":
    MyTool.run_as_script()
```

That's the entire plugin. Restart Cozter and `my_tool` becomes visible
to every agent backend.
