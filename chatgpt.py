"""OpenAI API client with tool-calling and session support."""

import json
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from . import auth
from . import session
from . import tools
from . import workspace

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are Cozter, a coding assistant operating inside a workspace directory.
You have full permission to create, read, edit, and delete files and directories,
run git commands, execute shell commands, and fetch content from the internet.

Prefer using dedicated tools (write_file, edit_file, rename_file, delete_file) over shell commands.
Always work within the workspace. Use the tools provided to accomplish tasks.
When you make changes, briefly explain what you did and how many lines changed.\
"""

COMPACT_PROMPT = (
    "Summarize the conversation so far into a concise context block. "
    "Preserve all important decisions, file changes, tool results, and "
    "the current state of work. This summary will replace the full history "
    "to save space, so include everything needed to continue seamlessly."
)

REREAD_PROMPT = (
    "You are resuming a conversation. Below is the session history so far. "
    "Read it carefully and continue from where you left off.\n\n"
)

# Responses API uses a flat tool format (name/description/parameters at top level)
RESPONSES_TOOL_DEFS = [
    {
        "type": "function",
        "name": td["function"]["name"],
        "description": td["function"]["description"],
        "parameters": td["function"]["parameters"],
    }
    for td in tools.TOOL_DEFS
]

# Models that use the Responses API (required for tools + reasoning)
RESPONSES_API_MODELS = {"gpt-5.4", "gpt-5.4-pro"}

# Models that support the reasoning effort parameter
REASONING_EFFORT_MODELS = {"o3", "o4-mini", "gpt-5.4", "gpt-5.4-pro"}


@dataclass
class ChatEvent:
    """An event produced during a chat turn."""
    kind: str  # "tool", "diff", "text"
    content: str
    tool_name: str | None = None
    file_path: str | None = None


class ChatGPTClient:
    def __init__(self):
        self._client: AsyncOpenAI | None = None
        self._histories: dict[int, list[dict]] = {}
        self._msg_counts: dict[int, int] = {}

    def _ensure_client(self) -> AsyncOpenAI:
        if self._client is None:
            api_key = auth.get_api_key()
            if not api_key:
                raise RuntimeError("Not logged in or API key exchange failed.")
            self._client = AsyncOpenAI(api_key=api_key)
        return self._client

    def reset_client(self) -> None:
        self._client = None

    def clear_history(self, user_id: int) -> None:
        self._histories.pop(user_id, None)
        self._msg_counts.pop(user_id, None)

    async def load_session(self, user_id: int, workspace_path: str, session_id: str) -> None:
        self._histories.pop(user_id, None)
        self._msg_counts[user_id] = 0

        messages = session.get_messages(workspace_path, session_id)
        if messages:
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            history.append({
                "role": "user",
                "content": REREAD_PROMPT + json.dumps(messages, indent=2, default=str),
            })
            self._histories[user_id] = history
        else:
            self._histories[user_id] = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]

    async def chat(
        self,
        user_id: int,
        message: str,
        workspace_path: str,
        model: str = "gpt-5.4",
        effort: str = "medium",
    ) -> list[ChatEvent]:
        """Send a user message, run tool calls, return list of ChatEvents."""
        client = self._ensure_client()
        events: list[ChatEvent] = []

        session_id = session.ensure_session(workspace_path, user_id)

        if user_id not in self._histories:
            await self.load_session(user_id, workspace_path, session_id)

        history = self._histories[user_id]

        user_msg = {"role": "user", "content": message}
        history.append(user_msg)
        session.append_message(workspace_path, session_id, user_msg)

        count = self._msg_counts.get(user_id, 0) + 1
        self._msg_counts[user_id] = count

        compact_interval = workspace.get_compact_interval(workspace_path)
        reread_interval = workspace.get_reread_interval(workspace_path)

        if count > 0 and count % reread_interval == 0:
            logger.info("Auto-reread triggered for user %s (count=%d)", user_id, count)
            await self.load_session(user_id, workspace_path, session_id)
            history = self._histories[user_id]
            history.append(user_msg)

        # Filter tools by workspace permissions
        allowed = workspace.get_allowed_tools(workspace_path)
        completions_tools = [
            td for td in tools.TOOL_DEFS
            if td["function"]["name"] in allowed
        ]
        responses_tools = [
            td for td in RESPONSES_TOOL_DEFS
            if td["name"] in allowed
        ]

        if model in RESPONSES_API_MODELS:
            events, assistant_text = await self._chat_responses(
                client, history, model, effort, workspace_path, session_id,
                responses_tools,
            )
        else:
            events, assistant_text = await self._chat_completions(
                client, history, model, effort, workspace_path, session_id,
                completions_tools,
            )

        # Auto compact
        if count > 0 and count % compact_interval == 0:
            logger.info("Auto-compact triggered for user %s (count=%d)", user_id, count)
            await self._compact_session(user_id, workspace_path, session_id, model)

        # Ensure there's at least one text event
        if not any(e.kind == "text" for e in events):
            events.append(ChatEvent(kind="text", content=assistant_text))

        return events

    # ------------------------------------------------------------------
    # Responses API path (gpt-5.4+)
    # ------------------------------------------------------------------

    async def _chat_responses(
        self,
        client: AsyncOpenAI,
        history: list[dict],
        model: str,
        effort: str,
        workspace_path: str,
        session_id: str,
        tool_defs: list[dict],
    ) -> tuple[list[ChatEvent], str]:
        events: list[ChatEvent] = []
        assistant_text = "(no response)"

        # Build input: skip the system message (passed as instructions)
        input_items = [msg for msg in history if msg.get("role") != "system"]

        while True:
            kwargs: dict = {
                "model": model,
                "instructions": SYSTEM_PROMPT,
                "input": input_items,
                "tools": tool_defs,
            }
            if model in REASONING_EFFORT_MODELS:
                kwargs["reasoning"] = {"effort": effort}

            response = await client.responses.create(**kwargs)

            # Check for function_call items
            function_calls = [
                item for item in response.output
                if item.type == "function_call"
            ]

            if function_calls:
                # Append all model output to input (preserves reasoning items)
                input_items += [item.to_dict() for item in response.output]

                for fc in function_calls:
                    fn_name = fc.name
                    fn_args = json.loads(fc.arguments)
                    logger.info("Tool call: %s(%s)", fn_name, fn_args)

                    result, diff = tools.execute(workspace_path, fn_name, fn_args)

                    events.append(ChatEvent(
                        kind="tool",
                        content=f"🔧 {fn_name}: {result}",
                        tool_name=fn_name,
                        file_path=fn_args.get("path"),
                    ))

                    if diff:
                        events.append(ChatEvent(
                            kind="diff",
                            content=diff,
                            file_path=fn_args.get("path"),
                        ))

                    input_items.append({
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": result,
                    })

                # Save tool interactions to session
                session.append_message(workspace_path, session_id, {
                    "_responses_api": True,
                    "tool_calls": [
                        {"name": fc.name, "arguments": fc.arguments}
                        for fc in function_calls
                    ],
                })
                continue

            # Final text response
            assistant_text = response.output_text or "(no response)"

            # Update history with a summary for future turns
            history.append({"role": "assistant", "content": assistant_text})
            session.append_message(
                workspace_path, session_id,
                {"role": "assistant", "content": assistant_text},
            )

            if assistant_text != "(no response)":
                events.append(ChatEvent(kind="text", content=assistant_text))
            break

        return events, assistant_text

    # ------------------------------------------------------------------
    # Chat Completions API path (gpt-4o, o3, etc.)
    # ------------------------------------------------------------------

    async def _chat_completions(
        self,
        client: AsyncOpenAI,
        history: list[dict],
        model: str,
        effort: str,
        workspace_path: str,
        session_id: str,
        tool_defs: list[dict],
    ) -> tuple[list[ChatEvent], str]:
        events: list[ChatEvent] = []
        assistant_text = "(no response)"

        while True:
            kwargs: dict = {
                "model": model,
                "messages": history,
                "tools": tool_defs,
            }
            if model in REASONING_EFFORT_MODELS:
                kwargs["reasoning_effort"] = effort

            response = await client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message
            msg_dict = msg.to_dict()

            history.append(msg_dict)

            if msg.tool_calls:
                session.append_message(workspace_path, session_id, msg_dict)

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)
                    logger.info("Tool call: %s(%s)", fn_name, fn_args)

                    result, diff = tools.execute(workspace_path, fn_name, fn_args)

                    events.append(ChatEvent(
                        kind="tool",
                        content=f"🔧 {fn_name}: {result}",
                        tool_name=fn_name,
                        file_path=fn_args.get("path"),
                    ))

                    if diff:
                        events.append(ChatEvent(
                            kind="diff",
                            content=diff,
                            file_path=fn_args.get("path"),
                        ))

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                    history.append(tool_msg)
                    session.append_message(workspace_path, session_id, tool_msg)
                continue

            # Final text
            assistant_text = msg.content or "(no response)"
            assistant_msg = {"role": "assistant", "content": assistant_text}
            session.append_message(workspace_path, session_id, assistant_msg)

            if msg.content:
                events.append(ChatEvent(kind="text", content=assistant_text))
            break

        return events, assistant_text

    # ------------------------------------------------------------------
    # Session compaction
    # ------------------------------------------------------------------

    async def _compact_session(
        self, user_id: int, workspace_path: str, session_id: str, model: str,
    ) -> None:
        client = self._ensure_client()
        history = self._histories.get(user_id, [])
        if len(history) < 4:
            return

        compact_messages = list(history) + [
            {"role": "user", "content": COMPACT_PROMPT},
        ]

        try:
            if model in RESPONSES_API_MODELS:
                input_items = [m for m in compact_messages if m.get("role") != "system"]
                resp = await client.responses.create(
                    model=model,
                    instructions=SYSTEM_PROMPT,
                    input=input_items,
                )
                summary = resp.output_text or ""
            else:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=compact_messages,
                )
                summary = resp.choices[0].message.content or ""
        except Exception:
            logger.exception("Compaction failed")
            return

        compacted = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": f"[Session summary]\n{summary}"},
        ]
        self._histories[user_id] = compacted
        session.replace_messages(workspace_path, session_id, compacted)
        logger.info("Session %s compacted for user %s", session_id, user_id)
