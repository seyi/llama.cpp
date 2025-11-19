"""
OpenAI agent provider for GPT models.

Supports GPT-3.5, GPT-4, GPT-4 Turbo, GPT-4.5, GPT-5, and future models.
"""

import os
import requests
from typing import List, Dict, Any, Optional

from .base import BaseAgent, Tool, Message


class OpenAIAgent(BaseAgent):
    """
    Agent that uses OpenAI API for function calling.

    Supports all GPT models that have function calling capabilities:
    - gpt-3.5-turbo
    - gpt-4
    - gpt-4-turbo
    - gpt-4.5-preview
    - gpt-5 (when available)
    - And future models

    Example:
        >>> import os
        >>> agent = OpenAIAgent(
        ...     model="gpt-4-turbo",
        ...     api_key=os.getenv("OPENAI_API_KEY")
        ... )
        >>> response = agent.chat("What's 2+2?", tools=[calculator_tool])
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        organization: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI agent.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-4-turbo", "gpt-5")
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            base_url: API base URL (for custom endpoints)
            organization: OpenAI organization ID (optional)
            **kwargs: Additional parameters
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenAI API key must be provided via api_key parameter "
                    "or OPENAI_API_KEY environment variable"
                )

        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        self.organization = organization

    def chat(
        self,
        message: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: str = "auto",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat message to OpenAI API.

        Args:
            message: User message to send
            tools: List of Tool objects
            tool_choice: "auto", "required", "none", or {"type": "function", "function": {"name": "..."}}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            **kwargs: Additional OpenAI parameters (top_p, presence_penalty, etc.)

        Returns:
            Response from the model
        """
        if message:
            self.add_message("user", message)

        # Prepare request
        request_data = {
            "model": self.model,
            "messages": self.get_history(),
            "temperature": temperature,
        }

        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens

        # Add any additional kwargs
        request_data.update(kwargs)

        # Add tools if provided
        if tools:
            request_data["tools"] = self._convert_tools_to_provider_format(tools)
            request_data["tool_choice"] = tool_choice

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        # Send request to OpenAI
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=request_data,
            headers=headers,
        )

        if response.status_code != 200:
            error_detail = response.json().get("error", {}).get("message", response.text)
            raise RuntimeError(
                f"OpenAI API returned {response.status_code}: {error_detail}"
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
        Convert Tool objects to OpenAI format.

        OpenAI uses the same format as our Tool.to_dict().
        """
        return [tool.to_dict() for tool in tools]

    @staticmethod
    def list_models(api_key: Optional[str] = None) -> List[str]:
        """
        List available OpenAI models.

        Args:
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)

        Returns:
            List of model IDs
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("API key required")

        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        if response.status_code != 200:
            raise RuntimeError(f"Failed to list models: {response.text}")

        models = response.json()
        return [model["id"] for model in models["data"]]
