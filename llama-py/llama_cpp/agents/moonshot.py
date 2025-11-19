"""
Moonshot agent provider for Kimi models.

Supports Moonshot AI's Kimi (月之暗面) language models.
"""

import os
import requests
from typing import List, Dict, Any, Optional

from .base import BaseAgent, Tool, Message


class MoonshotAgent(BaseAgent):
    """
    Agent that uses Moonshot AI API for function calling with Kimi models.

    Moonshot AI (月之暗面) provides Kimi models with OpenAI-compatible API.

    Supports:
    - moonshot-v1-8k
    - moonshot-v1-32k
    - moonshot-v1-128k
    - And future Kimi models

    Example:
        >>> import os
        >>> agent = MoonshotAgent(
        ...     model="moonshot-v1-32k",
        ...     api_key=os.getenv("MOONSHOT_API_KEY")
        ... )
        >>> response = agent.chat("你好！", tools=[calculator_tool])
    """

    def __init__(
        self,
        model: str = "moonshot-v1-32k",
        api_key: Optional[str] = None,
        base_url: str = "https://api.moonshot.cn/v1",
        **kwargs
    ):
        """
        Initialize Moonshot agent.

        Args:
            model: Model name (e.g., "moonshot-v1-32k")
            api_key: Moonshot API key (or set MOONSHOT_API_KEY env var)
            base_url: API base URL
            **kwargs: Additional parameters
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("MOONSHOT_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Moonshot API key must be provided via api_key parameter "
                    "or MOONSHOT_API_KEY environment variable"
                )

        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    def chat(
        self,
        message: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: str = "auto",
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,  # Moonshot recommends 0.3 default
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat message to Moonshot API.

        Moonshot uses OpenAI-compatible API, so the interface is similar to OpenAI.

        Args:
            message: User message to send
            tools: List of Tool objects
            tool_choice: "auto", "none", or specific tool
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (Moonshot recommends 0.3)
            **kwargs: Additional Moonshot parameters

        Returns:
            Response from the model
        """
        if message:
            self.add_message("user", message)

        # Prepare request (OpenAI-compatible format)
        request_data = {
            "model": self.model,
            "messages": self.get_history(),
            "temperature": temperature,
        }

        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens

        # Add any additional kwargs
        request_data.update(kwargs)

        # Add tools if provided (OpenAI-compatible format)
        if tools:
            request_data["tools"] = self._convert_tools_to_provider_format(tools)
            request_data["tool_choice"] = tool_choice

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Send request to Moonshot
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=request_data,
            headers=headers,
        )

        if response.status_code != 200:
            error_detail = response.json().get("error", {}).get("message", response.text)
            raise RuntimeError(
                f"Moonshot API returned {response.status_code}: {error_detail}"
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
        Convert Tool objects to Moonshot format.

        Moonshot uses OpenAI-compatible format, so we just convert
        Tool objects to their dict representation.
        """
        return [tool.to_dict() for tool in tools]

    @staticmethod
    def list_models(api_key: Optional[str] = None) -> List[str]:
        """
        List available Moonshot models.

        Args:
            api_key: Moonshot API key (or uses MOONSHOT_API_KEY env var)

        Returns:
            List of model IDs
        """
        if api_key is None:
            api_key = os.getenv("MOONSHOT_API_KEY")
            if api_key is None:
                raise ValueError("API key required")

        response = requests.get(
            "https://api.moonshot.cn/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        if response.status_code != 200:
            raise RuntimeError(f"Failed to list models: {response.text}")

        models = response.json()
        return [model["id"] for model in models["data"]]
