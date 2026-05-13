"""User plugin drop-in zone.

Drop a ``.py`` file here that subclasses ``AgentTool`` and it gets
auto-loaded on bot startup. The same file works two ways:

  - HTTP backends (llama, future Mistral/Gemini/etc.) see plugins as
    typed tools alongside the built-in toolkit - no special handling
    needed.
  - CLI backends (codex, claude_code, copilot) cannot have external
    tools injected, so the bot enumerates each plugin in their prompt
    and tells the model to invoke them via ``bash`` using
    ``python -m Cozter.agent_tools.plugins.<name>``.

See the example file in this directory and ``README.md`` for the
template.

Files starting with ``_`` are skipped by the loader, so an example can
ship in-tree without being live until renamed.
"""
