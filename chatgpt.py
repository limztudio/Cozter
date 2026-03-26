"""OpenAI API client with tool-calling and session support."""

import json
import logging

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

Always work within the workspace. Use the tools provided to accomplish tasks.
When you make changes, briefly explain what you did.\
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


class ChatGPTClient:
    def __init__(self):
        self._client: AsyncOpenAI | None = None
        # In-memory conversation: {user_id: [messages]}
        self._histories: dict[int, list[dict]] = {}
        # Track message counts for compact/reread triggers: {user_id: int}
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
        """Load a session's messages into memory, replacing any existing history."""
        self._histories.pop(user_id, None)
        self._msg_counts[user_id] = 0

        messages = session.get_messages(workspace_path, session_id)
        if messages:
            # Start with system prompt + a reread instruction + the saved messages
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            history.append({
                "role": "user",
                "content": REREAD_PROMPT + json.dumps(messages, indent=2, default=str),
            })
            # We'll let the AI acknowledge on the first real user message
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
        model: str = "gpt-4o",
        effort: str = "medium",
    ) -> str:
        """Send a user message, run tool calls in a loop, return final text."""
        client = self._ensure_client()

        # Ensure session exists
        session_id = session.ensure_session(workspace_path, user_id)

        # Initialize history if not loaded
        if user_id not in self._histories:
            await self.load_session(user_id, workspace_path, session_id)

        history = self._histories[user_id]

        # Append user message to history and persist
        user_msg = {"role": "user", "content": message}
        history.append(user_msg)
        session.append_message(workspace_path, session_id, user_msg)

        # Increment message counter
        count = self._msg_counts.get(user_id, 0) + 1
        self._msg_counts[user_id] = count

        compact_interval = workspace.get_compact_interval(workspace_path)
        reread_interval = workspace.get_reread_interval(workspace_path)

        # Auto reread: reload session from disk periodically for better memory
        if count > 0 and count % reread_interval == 0:
            logger.info("Auto-reread triggered for user %s (count=%d)", user_id, count)
            await self.load_session(user_id, workspace_path, session_id)
            history = self._histories[user_id]
            # Re-add the current user message since load_session doesn't include it
            history.append(user_msg)

        reasoning_effort_models = {"o3", "o4-mini"}

        # Run the completion loop
        assistant_text = ""
        while True:
            kwargs = {
                "model": model,
                "messages": history,
                "tools": tools.TOOL_DEFS,
            }
            if model in reasoning_effort_models:
                kwargs["reasoning"] = {"effort": effort}

            response = await client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message
            msg_dict = msg.to_dict()

            history.append(msg_dict)

            if msg.tool_calls:
                # Persist assistant tool-call message
                session.append_message(workspace_path, session_id, msg_dict)

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)
                    logger.info("Tool call: %s(%s)", fn_name, fn_args)

                    result = tools.execute(workspace_path, fn_name, fn_args)
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                    history.append(tool_msg)
                    session.append_message(workspace_path, session_id, tool_msg)
                continue

            # Final text response
            assistant_text = msg.content or "(no response)"
            assistant_msg = {"role": "assistant", "content": assistant_text}
            session.append_message(workspace_path, session_id, assistant_msg)
            break

        # Auto compact: summarize conversation to save storage
        if count > 0 and count % compact_interval == 0:
            logger.info("Auto-compact triggered for user %s (count=%d)", user_id, count)
            await self._compact_session(user_id, workspace_path, session_id, model)

        return assistant_text

    async def _compact_session(
        self, user_id: int, workspace_path: str, session_id: str, model: str,
    ) -> None:
        """Ask the AI to summarize the conversation, then replace session history."""
        client = self._ensure_client()
        history = self._histories.get(user_id, [])
        if len(history) < 4:
            return

        compact_messages = list(history) + [
            {"role": "user", "content": COMPACT_PROMPT},
        ]

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=compact_messages,
            )
            summary = response.choices[0].message.content or ""
        except Exception:
            logger.exception("Compaction failed")
            return

        # Replace history with system prompt + summary
        compacted = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": f"[Session summary]\n{summary}"},
        ]
        self._histories[user_id] = compacted
        session.replace_messages(workspace_path, session_id, compacted)
        logger.info("Session %s compacted for user %s", session_id, user_id)
