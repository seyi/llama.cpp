"""
Multi-provider agent support for function calling.

This module provides a unified interface for working with different AI providers,
all supporting function calling / tool use capabilities.

Supported providers:
- LlamaAgent: Local llama-server (following test_tool_call.py pattern)
- OpenAIAgent: OpenAI GPT models (GPT-4, GPT-4.5, GPT-5, etc.)
- AnthropicAgent: Anthropic Claude models (Opus, Sonnet, Haiku)
- GoogleAgent: Google Gemini models (1.5 Pro, 1.5 Flash, 2.0)
- MoonshotAgent: Moonshot Kimi models

Example:
    >>> from llama_cpp.agents import OpenAIAgent, Tool
    >>>
    >>> # Create a tool
    >>> weather_tool = Tool(
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
    >>>
    >>> # Use with OpenAI
    >>> agent = OpenAIAgent(model="gpt-4-turbo", api_key="sk-...")
    >>> response = agent.chat("What's the weather in Tokyo?", tools=[weather_tool])
    >>>
    >>> # Or use with Claude
    >>> agent = AnthropicAgent(model="claude-3-opus-20240229", api_key="sk-ant-...")
    >>> response = agent.chat("What's the weather in Tokyo?", tools=[weather_tool])
"""

from .base import BaseAgent, Tool, Message
from .llama import LlamaAgent
from .openai import OpenAIAgent
from .anthropic import AnthropicAgent
from .google import GoogleAgent
from .moonshot import MoonshotAgent

__all__ = [
    "BaseAgent",
    "Tool",
    "Message",
    "LlamaAgent",
    "OpenAIAgent",
    "AnthropicAgent",
    "GoogleAgent",
    "MoonshotAgent",
]
