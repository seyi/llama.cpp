# Design Document: llama-py Python Bindings

## Critical Examination of Existing Patterns in llama.cpp

This document provides a critical analysis of how Python bindings are implemented in the llama.cpp repository and how llama-py follows those established patterns.

## Existing Python Code in llama.cpp

### 1. gguf-py: The Reference Pattern

**Location**: `gguf-py/`

**Purpose**: Python library for GGUF file format manipulation (reading/writing model files)

**Key Characteristics**:
- **Pure Python** implementation for GGUF operations
- **ctypes** used ONLY in tests to verify against C implementation
- **Pattern**: `gguf-py/tests/test_quants.py` demonstrates ctypes usage

**Critical Pattern from test_quants.py (lines 40-85)**:

```python
class GGMLQuants:
    def __init__(self, libggml: Path):
        # Load shared library
        self.libggml = ctypes.CDLL(str(libggml))

        # Define function signatures explicitly
        self.libggml.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.libggml.ggml_quantize_chunk.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        )
```

**Why This Pattern?**
1. ✅ **No compilation required** - Works immediately after building C library
2. ✅ **Direct C API access** - Minimal abstraction overhead
3. ✅ **Python standard library** - No external dependencies
4. ✅ **Cross-platform** - ctypes handles platform differences

### 2. No Official Inference Bindings

**Critical Finding**: llama.cpp does NOT have official Python bindings for inference

**Evidence**:
- No `llama-py/` or similar directory in the repo
- README.md points to third-party `llama-cpp-python` by abetlen
- No pybind11, Cython, or other binding frameworks in use
- All inference examples are in C++ (see `examples/simple/simple.cpp`)

**Implications**:
- Our bindings fill an official gap
- We must follow the established gguf-py pattern for consistency
- ctypes is the only precedent in the repository

### 3. Function Calling / Agent Support

**Location**:
- `common/chat.h` - Core structures
- `tools/server/tests/unit/test_tool_call.py` - Test examples
- `docs/function-calling.md` - Documentation

**Critical Structures** (from common/chat.h:14-92):

```cpp
struct common_chat_tool_call {
    std::string name;
    std::string arguments;
    std::string id;
};

struct common_chat_msg {
    std::string role;
    std::string content;
    std::vector<common_chat_tool_call> tool_calls;
    // ... more fields
};

struct common_chat_tool {
    std::string name;
    std::string description;
    std::string parameters;  // JSON schema
};
```

**How "Agents" Work in llama.cpp**:

1. **NOT a C API** - Function calling is implemented in:
   - C++ layer (`common/chat.h`)
   - Server layer (llama-server with `--jinja` flag)
   - OpenAI-compatible HTTP API

2. **Access Pattern**:
   - ❌ Cannot directly call from Python via ctypes (C++ objects)
   - ✅ Must use HTTP API to llama-server
   - ✅ Server provides OpenAI-compatible `/v1/chat/completions`

3. **Test Pattern** (from test_tool_call.py):

```python
def do_test_weather(server: ServerProcess, **kwargs):
    body = server.make_any_request("POST", "/v1/chat/completions", data={
        "messages": [
            {"role": "system", "content": "You are a tool-calling agent."},
            {"role": "user", "content": "What is the weather in Istanbul?"},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": { ... }
            }
        }]
    })
```

## llama-py Design Decisions

### Decision 1: Follow gguf-py Pattern

**Rationale**:
- Only existing Python code pattern in the repository
- Proven to work with llama.cpp build system
- Maintains consistency across the project

**Implementation**:
```python
class LlamaLibrary:
    def __init__(self, lib_path: Path):
        self.lib = ctypes.CDLL(str(lib_path))
        self._setup_functions()  # Define all function signatures
```

This directly mirrors `GGMLQuants.__init__` from test_quants.py.

### Decision 2: Expose Core C API Only

**What We Expose**:
- ✅ Model loading (`llama_model_load_from_file`)
- ✅ Context creation (`llama_init_from_model`)
- ✅ Tokenization (`llama_tokenize`, `llama_token_to_piece`)
- ✅ Inference (`llama_decode`, `llama_encode`)
- ✅ Sampling (`llama_sampler_*`)

**What We Don't Expose**:
- ❌ C++ chat template system (not in C API)
- ❌ Direct common_chat_tool structures (C++ only)

**Rationale**:
- C API in `include/llama.h` is stable and exported
- C++ API in `common/` is internal and changes frequently
- ctypes can only call C functions, not C++ classes

### Decision 3: Agent API via HTTP

**Implementation**: Separate `LlamaAgent` class

```python
class LlamaAgent:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

    def chat(self, message: str, tools: List[Dict]) -> Dict:
        # Use requests to call /v1/chat/completions
        return requests.post(f"{self.base_url}/v1/chat/completions", ...)
```

**Rationale**:
- Matches test_tool_call.py pattern exactly
- Avoids C++/Python boundary issues
- Uses documented, stable API
- Aligns with how llama.cpp team tests function calling

## Comparison with Third-Party Solutions

### llama-cpp-python (abetlen)

**Approach**: pybind11 bindings

**Pros**:
- ✅ Full C++ API access
- ✅ Automatic memory management
- ✅ Type safety
- ✅ Comprehensive features

**Cons**:
- ❌ Requires compilation step
- ❌ Not part of official repo
- ❌ Different patterns than gguf-py
- ❌ External dependencies

### llama-py (this implementation)

**Approach**: ctypes bindings

**Pros**:
- ✅ Official (lives in llama.cpp repo)
- ✅ Follows gguf-py pattern
- ✅ No compilation needed
- ✅ Zero external dependencies
- ✅ Consistent with project style

**Cons**:
- ❌ Manual memory management
- ❌ Lower-level API
- ❌ Limited to C API only
- ❌ More verbose than pybind11

## Critical Analysis of C API

### What's Available (from include/llama.h)

**Stable Exports** (with `LLAMA_API` macro):

```c
// Model lifecycle
LLAMA_API struct llama_model * llama_model_load_from_file(
    const char * path_model,
    struct llama_model_params params);
LLAMA_API void llama_model_free(struct llama_model * model);

// Context lifecycle
LLAMA_API struct llama_context * llama_init_from_model(
    struct llama_model * model,
    struct llama_context_params params);
LLAMA_API void llama_free(struct llama_context * ctx);

// Tokenization
LLAMA_API int32_t llama_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special);

// Inference
LLAMA_API int32_t llama_decode(
    struct llama_context * ctx,
    struct llama_batch batch);
```

### What's Missing

**Not in C API**:
- ❌ Chat template application
- ❌ Tool call parsing
- ❌ Grammar constraining
- ❌ High-level conversation management

**Workaround**: Use llama-server HTTP API

## File Structure Rationale

```
llama-py/
├── llama_cpp/              # Mirrors gguf-py structure
│   ├── __init__.py
│   └── llama.py           # Core bindings (like gguf/gguf_reader.py)
├── examples/
│   ├── simple_generation.py        # Mirrors examples/simple/simple.cpp
│   └── agent_function_calling.py   # Mirrors test_tool_call.py
├── README.md
├── DESIGN.md              # This file
└── pyproject.toml         # Mirrors gguf-py/pyproject.toml
```

**Design Consistency**:
- Directory name: `llama-py` (matches `gguf-py`)
- Package name: `llama_cpp` (standard Python convention)
- Structure mirrors gguf-py exactly
- Examples mirror existing C++ examples

## Memory Management Pattern

Following simple.cpp (lines 86-121):

```cpp
// C++ RAII pattern
llama_model * model = llama_model_load_from_file(...);
llama_context * ctx = llama_init_from_model(model, ...);
// ... use model and context ...
llama_free(ctx);
llama_model_free(model);
```

Python equivalent:

```python
class LlamaModel:
    def __del__(self):
        if hasattr(self, "_model") and self._model:
            self._lib.lib.llama_model_free(self._model)
```

**Rationale**:
- Matches C++ RAII semantics
- Automatic cleanup via Python garbage collection
- Prevents memory leaks

## API Surface Design

### Minimal but Complete

We expose exactly what's needed to replicate simple.cpp:

1. **Model loading** → `LlamaModel.from_file()`
2. **Tokenization** → `model.tokenize()`
3. **Context creation** → `LlamaContext(model)`
4. **Inference** → `ctx.decode(batch)`
5. **Sampling** → `sampler.sample(ctx)`

### Not Exposing Advanced Features (Yet)

**Rationale for minimal initial API**:
- Start with stable, well-tested C API
- Add features incrementally as needed
- Maintain compatibility with C API changes
- Keep bindings simple and maintainable

## Testing Strategy

### Following test_quants.py Pattern

```python
def test_basic_inference():
    """Test following simple.cpp pattern"""
    model = LlamaModel.from_file("test.gguf")
    tokens = model.tokenize("Hello")
    assert len(tokens) > 0
```

### Agent Testing

```python
def test_agent_weather():
    """Test following test_tool_call.py pattern"""
    agent = LlamaAgent()
    response = agent.chat(
        message="What's the weather?",
        tools=[weather_tool],
    )
    assert "tool_calls" in response["choices"][0]["message"]
```

## Future Considerations

### Potential Additions

1. **More samplers**:
   - Top-K, Top-P, Temperature
   - Following C API additions

2. **Batch operations**:
   - More flexible batch creation
   - Multi-sequence support

3. **KV cache management**:
   - Save/load state
   - Sequence management

### Non-Goals

1. ❌ **Don't reimplement C++ features** in Python
   - Chat templates → Use server API
   - Grammar constraining → Use server API
   - Tool parsing → Use server API

2. ❌ **Don't compete with llama-cpp-python**
   - They provide high-level API
   - We provide official low-level bindings

3. ❌ **Don't break from established patterns**
   - Always follow gguf-py conventions
   - Match C API exactly
   - Keep it simple

## Conclusion

**llama-py provides**:
- ✅ Official Python bindings following established patterns
- ✅ Low-level C API access via ctypes
- ✅ Agent functionality via llama-server HTTP API
- ✅ Consistency with gguf-py design
- ✅ Zero-dependency core library

**This fills the gap** between:
- **gguf-py**: File format only
- **llama-cpp-python**: Third-party, feature-rich but not official

**By critically following** the patterns already in llama.cpp, we ensure:
- Maintainability alongside C++ codebase
- Familiar structure for contributors
- Stable API aligned with C interface
- Official status within the repository
