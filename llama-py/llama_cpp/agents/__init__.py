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

Hierarchical agent support:
- HierarchicalAgent: Organize agents in a tree structure
- SharedContext: Shared context and message passing
- ContextScope: Define context visibility scope

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
    >>> # Or use hierarchical agents
    >>> from llama_cpp.agents import HierarchicalAgent, SharedContext
    >>> context = SharedContext()
    >>> root = HierarchicalAgent("root", OpenAIAgent(...), context)
    >>> child = root.add_child("child", AnthropicAgent(...))
"""

from .base import BaseAgent, Tool, Message
from .llama import LlamaAgent
from .openai import OpenAIAgent
from .anthropic import AnthropicAgent
from .google import GoogleAgent
from .moonshot import MoonshotAgent
from .context import SharedContext, ContextScope, AgentMessage, ContextEntry
from .hierarchical import HierarchicalAgent, AgentTask

# A2A Protocol support (optional)
try:
    from .a2a import (
        A2AMessage, Task, TaskState, TaskStatus, Artifact,
        AgentCard, AgentSkill, AgentCapabilities,
        TextPart, FilePart, DataPart, PartKind,
        create_task, parse_message
    )
    _a2a_exports = [
        "A2AMessage", "Task", "TaskState", "TaskStatus", "Artifact",
        "AgentCard", "AgentSkill", "AgentCapabilities",
        "TextPart", "FilePart", "DataPart", "PartKind",
        "create_task", "parse_message",
    ]
except ImportError:
    _a2a_exports = []

__all__ = [
    # Base classes
    "BaseAgent",
    "Tool",
    "Message",
    # Provider agents
    "LlamaAgent",
    "OpenAIAgent",
    "AnthropicAgent",
    "GoogleAgent",
    "MoonshotAgent",
    # Hierarchical agents
    "HierarchicalAgent",
    "SharedContext",
    "ContextScope",
    "AgentMessage",
    "ContextEntry",
    "AgentTask",
] + _a2a_exports
