"""
Anthropic agent provider for Claude models.

Supports Claude 3 Opus, Sonnet, Haiku, and future models.
"""

import os
import requests
from typing import List, Dict, Any, Optional

from .base import BaseAgent, Tool, Message


class AnthropicAgent(BaseAgent):
    """
    Agent that uses Anthropic API for function calling with Claude models.

    Supports:
    - claude-3-opus-20240229
    - claude-3-sonnet-20240229
    - claude-3-haiku-20240307
    - claude-3-5-sonnet-20241022
    - And future Claude models

    Example:
        >>> import os
        >>> agent = AnthropicAgent(
        ...     model="claude-3-5-sonnet-20241022",
        ...     api_key=os.getenv("ANTHROPIC_API_KEY")
        ... )
        >>> response = agent.chat("Hello!", tools=[calculator_tool])
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com/v1",
        **kwargs
    ):
        """
        Initialize Anthropic agent.

        Args:
            model: Model name (e.g., "claude-3-opus-20240229")
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            base_url: API base URL
            **kwargs: Additional parameters
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Anthropic API key must be provided via api_key parameter "
                    "or ANTHROPIC_API_KEY environment variable"
                )

        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    def chat(
        self,
        message: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: str = "auto",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat message to Anthropic API.

        Args:
            message: User message to send
            tools: List of Tool objects
            tool_choice: "auto", "any", or {"type": "tool", "name": "..."}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            system: System message (optional)
            **kwargs: Additional Anthropic parameters

        Returns:
            Response from the model (normalized to OpenAI-like format)
        """
        if message:
            self.add_message("user", message)

        # Extract system messages from history if any
        messages = []
        system_message = system

        for msg in self.conversation_history:
            msg_dict = msg.to_dict()
            if msg.role == "system":
                # Claude uses separate system parameter
                system_message = msg.content
            else:
                # Convert tool messages to Claude format
                if msg.role == "tool":
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }]
                    })
                else:
                    # Convert regular messages
                    content = msg.content if msg.content else ""

                    # Add tool calls if present
                    if msg.tool_calls:
                        # Claude uses "tool_use" blocks in content
                        content_blocks = []
                        if content:
                            content_blocks.append({"type": "text", "text": content})

                        for tool_call in msg.tool_calls:
                            import json
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tool_call.get("id", ""),
                                "name": tool_call["function"]["name"],
                                "input": json.loads(tool_call["function"]["arguments"]),
                            })
                        messages.append({"role": msg.role, "content": content_blocks})
                    else:
                        messages.append({"role": msg.role, "content": content})

        # Prepare request
        request_data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_message:
            request_data["system"] = system_message

        # Add any additional kwargs
        request_data.update(kwargs)

        # Add tools if provided
        if tools:
            request_data["tools"] = self._convert_tools_to_provider_format(tools)

            # Convert tool_choice to Claude format
            if tool_choice == "required" or tool_choice == "any":
                request_data["tool_choice"] = {"type": "any"}
            elif tool_choice != "auto":
                if isinstance(tool_choice, dict):
                    request_data["tool_choice"] = tool_choice
                else:
                    # Assume it's a tool name
                    request_data["tool_choice"] = {"type": "tool", "name": tool_choice}

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        # Send request to Anthropic
        response = requests.post(
            f"{self.base_url}/messages",
            json=request_data,
            headers=headers,
        )

        if response.status_code != 200:
            error_detail = response.json().get("error", {}).get("message", response.text)
            raise RuntimeError(
                f"Anthropic API returned {response.status_code}: {error_detail}"
            )

        result = response.json()

        # Convert Anthropic response to OpenAI-like format for consistency
        normalized_result = self._normalize_response(result)

        # Add assistant response to history
        assistant_message = normalized_result["choices"][0]["message"]
        msg = Message(
            role="assistant",
            content=assistant_message.get("content"),
            tool_calls=assistant_message.get("tool_calls"),
        )
        self.conversation_history.append(msg)

        return normalized_result

    def _convert_tools_to_provider_format(
        self,
        tools: List[Tool]
    ) -> List[Dict[str, Any]]:
        """
        Convert Tool objects to Anthropic format.

        Anthropic uses a slightly different format than OpenAI.
        """
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        return anthropic_tools

    def _normalize_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Anthropic response format to OpenAI-like format for consistency.

        This allows the same code to work with different providers.
        """
        import json

        content_text = ""
        tool_calls = []

        # Parse content blocks
        for block in result.get("content", []):
            if block["type"] == "text":
                content_text += block["text"]
            elif block["type"] == "tool_use":
                # Convert to OpenAI tool call format
                tool_calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"]),
                    }
                })

        # Build OpenAI-like response
        normalized = {
            "id": result.get("id"),
            "model": result.get("model"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content_text if content_text else None,
                },
                "finish_reason": result.get("stop_reason"),
            }],
            "usage": result.get("usage"),
        }

        if tool_calls:
            normalized["choices"][0]["message"]["tool_calls"] = tool_calls

        return normalized
