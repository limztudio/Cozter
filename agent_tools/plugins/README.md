# Cozter user plugins

Drop a `.py` file here to add a new tool to every agent. One file
serves both modes:

- **HTTP backends** (`llama`, future Mistral/Gemini/etc.) see plugins
  as typed tools — same schema as the built-in toolkit in `../builtin/`.
- **CLI backends** (`codex`, `claude_code`, `copilot`) can't have
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
- File names starting with `_` are skipped by the loader. The shipped
  example file uses this so it doesn't activate by accident.

## Template

```python
"""Plugin: <one-line description>."""

from __future__ import annotations

from ..base import AgentTool


class MyTool(AgentTool):
    name = "my_tool"
    description = "What this does, from the model's perspective."
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
