"""OpenAI API client with tool-calling support."""

import json
import logging

from openai import AsyncOpenAI

from . import auth
from . import tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are Cozter, a coding assistant operating inside a workspace directory.
You have full permission to create, read, edit, and delete files and directories,
run git commands, execute shell commands, and fetch content from the internet.

Always work within the workspace. Use the tools provided to accomplish tasks.
When you make changes, briefly explain what you did.\
"""


class ChatGPTClient:
    def __init__(self):
        self._client: AsyncOpenAI | None = None
        # Per-user conversation histories: {user_id: [messages]}
        self._histories: dict[int, list[dict]] = {}

    def _ensure_client(self) -> AsyncOpenAI:
        if self._client is None:
            api_key = auth.get_api_key()
            if not api_key:
                raise RuntimeError("Not logged in or API key exchange failed.")
            self._client = AsyncOpenAI(api_key=api_key)
        return self._client

    def reset_client(self) -> None:
        """Force re-creation of client (e.g. after token refresh)."""
        self._client = None

    def clear_history(self, user_id: int) -> None:
        self._histories.pop(user_id, None)

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

        history = self._histories.setdefault(user_id, [
            {"role": "system", "content": SYSTEM_PROMPT},
        ])
        history.append({"role": "user", "content": message})

        reasoning_effort_models = {"o3", "o4-mini"}

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

            # Append assistant message to history
            history.append(msg.to_dict())

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)
                    logger.info("Tool call: %s(%s)", fn_name, fn_args)

                    result = tools.execute(workspace_path, fn_name, fn_args)
                    history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue  # loop back to get next response

            # No tool calls — return the final text
            return msg.content or "(no response)"
