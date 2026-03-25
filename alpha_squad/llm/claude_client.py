"""Custom AutoGen model client that calls `claude` CLI (uses Max subscription)."""

import json
import subprocess
from types import SimpleNamespace


class ClaudeCLIClient:
    """AutoGen-compatible model client that delegates to `claude` CLI.

    This avoids needing an ANTHROPIC_API_KEY — uses the user's
    existing Claude Code / Max authentication.
    """

    def __init__(self, config: dict, **kwargs):
        self.model = config.get("model", "claude-sonnet-4-20250514")
        self.max_tokens = config.get("max_tokens", 4096)

    def create(self, params: dict) -> SimpleNamespace:
        """Called by AutoGen to generate a response."""
        messages = params.get("messages", [])

        # Build the prompt: system message + conversation history
        system_msg = ""
        conversation = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            name = msg.get("name", "")

            if role == "system":
                system_msg = content
            else:
                speaker = name or role
                conversation.append(f"[{speaker}]: {content}")

        prompt = ""
        if system_msg:
            prompt += system_msg + "\n\n"
        prompt += "---\n지금까지의 회의 내용:\n\n"
        prompt += "\n\n".join(conversation[-10:])  # last 10 messages for context
        prompt += "\n\n---\n위 회의 내용을 바탕으로 당신의 역할에 맞게 발언하세요."

        # Call claude CLI
        try:
            result = subprocess.run(
                [
                    "claude",
                    "-p", prompt,
                    "--output-format", "text",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            response_text = result.stdout.strip()
            if not response_text:
                response_text = "(응답 없음)"
        except subprocess.TimeoutExpired:
            response_text = "(타임아웃: 120초 초과)"
        except Exception as e:
            response_text = f"(에러: {e})"

        # Return in AutoGen-expected format
        choice = SimpleNamespace(
            message=SimpleNamespace(
                content=response_text,
                function_call=None,
                tool_calls=None,
            )
        )
        response = SimpleNamespace(
            choices=[choice],
            model=self.model,
            usage=SimpleNamespace(
                prompt_tokens=len(prompt) // 4,
                completion_tokens=len(response_text) // 4,
            ),
        )
        return response

    def message_retrieval(self, response: SimpleNamespace) -> list[str]:
        """Extract messages from response."""
        return [response.choices[0].message.content]

    @staticmethod
    def cost(response: SimpleNamespace) -> float:
        """Return 0 — cost is handled by Max subscription."""
        return 0.0

    @staticmethod
    def get_usage(response: SimpleNamespace) -> dict:
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.prompt_tokens + response.usage.completion_tokens,
            "cost": 0.0,
            "model": response.model,
        }
