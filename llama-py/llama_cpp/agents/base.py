"""
Base agent classes for multi-provider function calling support.

This module defines the abstract interface that all agent providers must implement,
following the pattern established in test_tool_call.py.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Tool:
    """
    Tool/function definition following OpenAI format.

    This matches the pattern from test_tool_call.py and is compatible
    with the OpenAI function calling API.
    """
    name: str
    description: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible tool definition"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


@dataclass
class Message:
    """Message in a conversation"""
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # For tool response messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format"""
        msg = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


class BaseAgent(ABC):
    """
    Abstract base class for AI agents with function calling support.

    All provider implementations (OpenAI, Anthropic, Google, etc.) must
    inherit from this class and implement the abstract methods.

    This design follows the pattern from test_tool_call.py while providing
    a unified interface across different AI providers.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the agent.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-opus-20240229")
            api_key: API key for the provider (can also use environment variables)
            base_url: Custom base URL for the API (optional)
            **kwargs: Additional provider-specific parameters
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.conversation_history: List[Message] = []
        self.kwargs = kwargs

    @abstractmethod
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
        Send a chat message and optionally provide tools.

        Args:
            message: User message to send
            tools: List of Tool objects available to the agent
            tool_choice: "auto", "required", "none", or specific tool name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific parameters

        Returns:
            Response from the model in a standardized format
        """
        pass

    @abstractmethod
    def _convert_tools_to_provider_format(
        self,
        tools: List[Tool]
    ) -> List[Dict[str, Any]]:
        """
        Convert Tool objects to provider-specific format.

        Args:
            tools: List of Tool objects

        Returns:
            List of tools in provider's expected format
        """
        pass

    def add_message(self, role: str, content: Optional[str] = None, **kwargs):
        """Add a message to conversation history"""
        msg = Message(role=role, content=content, **kwargs)
        self.conversation_history.append(msg)

    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str
    ):
        """
        Add tool execution result to conversation.

        This follows the pattern from test_tool_call.py.
        """
        msg = Message(
            role="tool",
            content=result,
            tool_call_id=tool_call_id,
            name=tool_name,
        )
        self.conversation_history.append(msg)

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history in dict format"""
        return [msg.to_dict() for msg in self.conversation_history]

    @staticmethod
    def create_tool(
        name: str,
        description: str,
        parameters: Dict[str, Any]
    ) -> Tool:
        """
        Helper method to create a Tool object.

        Example:
            >>> tool = BaseAgent.create_tool(
            ...     name="get_weather",
            ...     description="Get current weather",
            ...     parameters={
            ...         "type": "object",
            ...         "properties": {
            ...             "location": {"type": "string"}
            ...         },
            ...         "required": ["location"]
            ...     }
            ... )
        """
        return Tool(name=name, description=description, parameters=parameters)
