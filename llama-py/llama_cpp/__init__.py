"""
llama_cpp - Python bindings for llama.cpp

This package provides ctypes-based Python bindings for llama.cpp, following the
pattern established in gguf-py for consistency with the llama.cpp repository.

The bindings expose the C API defined in include/llama.h, allowing you to:
- Load and run GGUF models
- Perform text generation
- Use function calling / tool use (agent functionality)
- Access tokenization and sampling capabilities

Examples:
    Basic inference:
    >>> from llama_cpp import LlamaModel, LlamaContext
    >>> model = LlamaModel.from_file("model.gguf")
    >>> ctx = LlamaContext(model)

    Agent functionality with multiple providers:
    >>> from llama_cpp.agents import OpenAIAgent, AnthropicAgent, GoogleAgent, Tool
    >>> tool = Tool(name="calculator", description="Calculate", parameters={...})
    >>> agent = OpenAIAgent(model="gpt-4-turbo", api_key="sk-...")
    >>> response = agent.chat("What is 2+2?", tools=[tool])
"""

from .llama import (
    LlamaModel,
    LlamaContext,
    LlamaSampler,
    LlamaBatch,
    llama_token,
    llama_pos,
    llama_seq_id,
)

# Agent functionality is imported via llama_cpp.agents
# Example: from llama_cpp.agents import OpenAIAgent, AnthropicAgent, etc.

__all__ = [
    "LlamaModel",
    "LlamaContext",
    "LlamaSampler",
    "LlamaBatch",
    "llama_token",
    "llama_pos",
    "llama_seq_id",
]

__version__ = "0.1.0"
