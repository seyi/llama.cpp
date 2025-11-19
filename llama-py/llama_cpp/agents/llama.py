"""
Llama agent provider for local llama-server.

This provider connects to a locally running llama-server instance,
following the pattern from test_tool_call.py.
"""

import os
import requests
from typing import List, Dict, Any, Optional

from .base import BaseAgent, Tool, Message


class LlamaAgent(BaseAgent):
    """
    Agent that uses local llama-server for function calling.

    This follows the pattern from tools/server/tests/unit/test_tool_call.py,
    using the OpenAI-compatible function calling API provided by llama-server.

    Example:
        >>> agent = LlamaAgent(
        ...     model="llama-3.1-8b",
        ...     base_url="http://localhost:8080"
        ... )
        >>> response = agent.chat("Hello!", tools=[weather_tool])
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",  # Default for OpenAI compatibility
        api_key: Optional[str] = None,  # Not used for local server
        base_url: str = "http://localhost:8080",
        **kwargs
    ):
        """
        Initialize Llama agent.

        Args:
            model: Model name (for compatibility, can be any string)
            api_key: Not used for local server
            base_url: URL of the llama-server instance
            **kwargs: Additional parameters
        """
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    def chat(
        self,
        message: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: str = "auto",
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat message to llama-server.

        This follows the pattern from test_tool_call.py:84-94.

        Args:
            message: User message to send
            tools: List of Tool objects
            tool_choice: "auto", "required", or "none"
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Response from the model
        """
        if message:
            self.add_message("user", message)

        # Prepare request following test_tool_call.py pattern
        request_data = {
            "model": self.model,
            "messages": self.get_history(),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add any additional kwargs
        request_data.update(kwargs)

        # Add tools if provided
        if tools:
            request_data["tools"] = self._convert_tools_to_provider_format(tools)
            request_data["tool_choice"] = tool_choice

        # Send request to llama-server
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Llama-server returned {response.status_code}: {response.text}"
            )

        result = response.json()
        assistant_message = result["choices"][0]["message"]

        # Add assistant response to history
        msg = Message(
            role="assistant",
            content=assistant_message.get("content"),
            tool_calls=assistant_message.get("tool_calls"),
        )
        self.conversation_history.append(msg)

        return result

    def _convert_tools_to_provider_format(
        self,
        tools: List[Tool]
    ) -> List[Dict[str, Any]]:
        """
        Convert Tool objects to llama-server format.

        Llama-server uses OpenAI-compatible format, so we just convert
        Tool objects to their dict representation.
        """
        return [tool.to_dict() for tool in tools]

    def health_check(self) -> bool:
        """Check if llama-server is running and healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
