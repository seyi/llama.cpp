"""
llama_cpp - Python bindings for llama.cpp

This package provides ctypes-based Python bindings for llama.cpp, following the
pattern established in gguf-py for consistency with the llama.cpp repository.

The bindings expose the C API defined in include/llama.h, allowing you to:
- Load and run GGUF models
- Perform text generation
- Use function calling / tool use (agent functionality)
- Access tokenization and sampling capabilities

Example:
    >>> from llama_cpp import LlamaModel, LlamaContext
    >>> model = LlamaModel.from_file("model.gguf")
    >>> ctx = LlamaContext(model)
    >>> # Use for generation or function calling
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
