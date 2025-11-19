#!/usr/bin/env python3
"""
Simple text generation example using the llama_cpp Python bindings.

This example demonstrates the basic usage pattern following the simple.cpp
example from llama.cpp/examples/simple/simple.cpp
"""

import sys
from pathlib import Path

# Add parent directory to path to import llama_cpp
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import LlamaModel, LlamaContext, LlamaSampler, LlamaBatch


def main():
    # Configuration
    model_path = "models/your-model.gguf"  # Update this path
    prompt = "Hello my name is"
    n_predict = 32
    n_gpu_layers = 0  # Set to > 0 for GPU offloading

    print(f"Loading model from {model_path}...")

    # Load the model (following simple.cpp pattern)
    model = LlamaModel.from_file(
        model_path,
        n_gpu_layers=n_gpu_layers,
        use_mmap=True,
    )

    print(f"Model loaded. Vocabulary size: {model.n_vocab()}")

    # Tokenize the prompt
    print(f"\nTokenizing prompt: '{prompt}'")
    prompt_tokens = model.tokenize(prompt, add_special=True, parse_special=True)
    print(f"Tokens: {prompt_tokens}")

    # Create context
    n_ctx = len(prompt_tokens) + n_predict
    print(f"\nCreating context with n_ctx={n_ctx}")
    ctx = LlamaContext(
        model,
        n_ctx=n_ctx,
        n_batch=len(prompt_tokens),
        n_threads=-1,  # Use all available threads
    )

    # Create a greedy sampler
    sampler = LlamaSampler.greedy(model._lib)

    # Print the prompt
    print(f"\nPrompt: ", end="", flush=True)
    for token in prompt_tokens:
        piece = model.token_to_piece(token)
        print(piece, end="", flush=True)

    # Create batch from prompt tokens
    batch = LlamaBatch.from_tokens(prompt_tokens, model._lib)

    # Encode or decode based on model type
    if model.has_encoder():
        ctx.encode(batch)
    else:
        ctx.decode(batch)

    # Generate tokens
    print("\n\nGeneration: ", end="", flush=True)

    for i in range(n_predict):
        # Sample next token
        new_token = sampler.sample(ctx, -1)

        # Check for end of sequence
        if new_token == 0:  # Assuming 0 is EOS, might need adjustment
            break

        # Print the token
        piece = model.token_to_piece(new_token)
        print(piece, end="", flush=True)

        # Prepare next batch with single token
        batch = LlamaBatch.from_tokens([new_token], model._lib)
        ctx.decode(batch)

    print("\n\nDone!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("\nUsage: Update the model_path in the script to point to your GGUF model", file=sys.stderr)
        sys.exit(1)
