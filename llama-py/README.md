# llama-py: Python Bindings for llama.cpp

Official Python bindings for llama.cpp, following the established pattern from `gguf-py`.

## Overview

This package provides ctypes-based Python bindings for the llama.cpp inference library. It follows the same pattern as `gguf-py`, using ctypes to directly call C functions from the llama.cpp shared library.

## Features

- **Low-level C API access**: Direct ctypes bindings to `libllama.so`
- **Pythonic interface**: High-level classes wrapping the C API
- **Agent functionality**: Support for function calling / tool use
- **Pattern consistency**: Follows the established `gguf-py` pattern
- **Zero dependencies**: Only requires Python standard library

## Installation

### Prerequisites

1. **Build llama.cpp with shared libraries**:
   ```bash
   cd /path/to/llama.cpp
   cmake -B build -DBUILD_SHARED_LIBS=ON
   cmake --build build --config Release
   ```

2. **Verify library exists**:
   ```bash
   ls build/bin/libllama.so  # Linux
   ls build/bin/libllama.dylib  # macOS
   ls build/bin/llama.dll  # Windows
   ```

### Install the package

```bash
cd llama-py
pip install -e .
```

## Usage

### Basic Text Generation

```python
from llama_cpp import LlamaModel, LlamaContext, LlamaSampler, LlamaBatch

# Load model
model = LlamaModel.from_file("model.gguf", n_gpu_layers=32)

# Create context
ctx = LlamaContext(model, n_ctx=2048)

# Tokenize prompt
tokens = model.tokenize("Hello, my name is", add_special=True)

# Create batch and decode
batch = LlamaBatch.from_tokens(tokens, model._lib)
ctx.decode(batch)

# Sample next token
sampler = LlamaSampler.greedy(model._lib)
next_token = sampler.sample(ctx, -1)

# Convert to text
text = model.token_to_piece(next_token)
print(text)
```

### Agent / Function Calling

For agent functionality with function calling (tool use), use the llama-server's OpenAI-compatible API:

```python
from llama_cpp.agent import LlamaAgent

# Create agent (requires llama-server running)
agent = LlamaAgent(base_url="http://localhost:8080")

# Define a tool
weather_tool = agent.define_tool(
    name="get_current_weather",
    description="Get the current weather in a given location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    }
)

# Chat with tool calling
response = agent.chat(
    message="What's the weather in Tokyo?",
    tools=[weather_tool],
)

# Check for tool calls
tool_calls = response["choices"][0]["message"].get("tool_calls")
if tool_calls:
    for tool_call in tool_calls:
        print(f"Tool: {tool_call['function']['name']}")
        print(f"Args: {tool_call['function']['arguments']}")
```

See `examples/agent_function_calling.py` for a complete example.

## Architecture

### Design Pattern

This package follows the **same pattern as `gguf-py/tests/test_quants.py`**:

1. **ctypes for C bindings**: Direct loading of shared library
2. **Structure definitions**: Python classes matching C structs
3. **Function signatures**: Explicit argtypes/restype definitions
4. **Pythonic wrappers**: High-level classes with `__init__`, `__del__`, etc.

### File Structure

```
llama-py/
├── llama_cpp/
│   ├── __init__.py          # Package exports
│   └── llama.py             # Core ctypes bindings
├── examples/
│   ├── simple_generation.py        # Basic example
│   └── agent_function_calling.py   # Agent/tool use example
├── README.md
└── pyproject.toml
```

### How It Works

**From `gguf-py/tests/test_quants.py` pattern**:

```python
import ctypes

class GGMLQuants:
    def __init__(self, libggml: Path):
        self.libggml = ctypes.CDLL(str(libggml))
        # Define function signatures
        self.libggml.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.libggml.ggml_quantize_chunk.argtypes = (...)
```

**Applied to llama.cpp**:

```python
class LlamaLibrary:
    def __init__(self, lib_path: Path):
        self.lib = ctypes.CDLL(str(lib_path))
        # Define llama.h functions
        self.lib.llama_model_load_from_file.restype = ctypes.c_void_p
        self.lib.llama_model_load_from_file.argtypes = (...)
```

## Comparison with Existing Solutions

### This Package (llama-py)

- ✅ **Official**: Lives in llama.cpp repository
- ✅ **Consistent**: Follows established `gguf-py` pattern
- ✅ **Simple**: Pure ctypes, no compilation needed
- ✅ **Minimal**: No external dependencies
- ⚠️ **Basic**: Low-level API, requires manual memory management

### Third-Party (llama-cpp-python)

- ✅ **Feature-rich**: High-level API, automatic memory management
- ✅ **Mature**: Well-tested, widely used
- ✅ **Batteries included**: Built-in server, OpenAI compatibility
- ⚠️ **External**: Not part of llama.cpp repo
- ⚠️ **Complex**: Cython/pybind11 compilation required

## Function Calling / Agent Support

llama.cpp implements function calling through:

1. **Chat templates** (`common/chat.h`): Parse tool definitions
2. **Grammar constraining**: Force valid JSON tool call output
3. **OpenAI API**: Compatible function calling endpoint

See:
- `docs/function-calling.md` - Documentation
- `common/chat.h` - Tool call structures
- `tools/server/tests/unit/test_tool_call.py` - Test examples

### Supported Models

From `docs/function-calling.md`:

**Native support**:
- Llama 3.1, 3.2, 3.3
- Functionary v3.1 / v3.2
- Hermes 2 Pro / 3
- Qwen 2.5
- Mistral Nemo
- FireFunction v2
- DeepSeek R1

**Generic support**: All other models (may be less efficient)

## Examples

### Simple Generation

```bash
python examples/simple_generation.py
```

See `examples/simple_generation.py` for the complete example following `examples/simple/simple.cpp`.

### Agent Function Calling

**Start llama-server**:
```bash
llama-server --jinja -m model.gguf --port 8080
```

**Run agent example**:
```bash
python examples/agent_function_calling.py
```

This demonstrates:
- Tool/function definition
- Function calling requests
- Tool result integration
- Multi-turn conversations

## API Reference

### LlamaModel

```python
model = LlamaModel.from_file(
    path: str | Path,
    n_gpu_layers: int = 0,
    use_mmap: bool = True,
    use_mlock: bool = False,
    vocab_only: bool = False,
) -> LlamaModel

model.tokenize(text: str) -> List[int]
model.token_to_piece(token: int) -> str
model.n_vocab() -> int
model.has_encoder() -> bool
```

### LlamaContext

```python
ctx = LlamaContext(
    model: LlamaModel,
    n_ctx: int = 512,
    n_batch: int = 512,
    n_threads: int = -1,
    flash_attn: bool = False,
)

ctx.decode(batch: LlamaBatch) -> None
ctx.encode(batch: LlamaBatch) -> None  # For encoder-decoder models
ctx.get_logits(idx: int = -1) -> List[float]
```

### LlamaBatch

```python
batch = LlamaBatch.from_tokens(tokens: List[int], lib: LlamaLibrary)
```

### LlamaSampler

```python
sampler = LlamaSampler.greedy(lib: LlamaLibrary)
token = sampler.sample(ctx: LlamaContext, idx: int = -1) -> int
```

## Development

### Running Tests

```bash
# Ensure llama.cpp is built with shared libraries
cd /path/to/llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release

# Run basic test
cd llama-py
python -c "from llama_cpp import LlamaModel; print('Import successful')"
```

### Contributing

When adding new functionality:

1. **Follow the ctypes pattern** from `gguf-py/tests/test_quants.py`
2. **Match C API** from `include/llama.h`
3. **Reference examples** from `examples/simple/simple.cpp`
4. **Document thoroughly** with code comments

## License

Same as llama.cpp (MIT License)

## See Also

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Main repository
- [gguf-py](../gguf-py) - Pattern reference
- [Function Calling Docs](../docs/function-calling.md) - Agent functionality
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Third-party alternative
