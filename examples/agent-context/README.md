# Agent Context Management System

This directory contains an implementation of a context management system for AI agents, following the patterns used in llama.cpp.

## Overview

The agent context system provides a structured way to manage:
- **State**: Conversation history and agent memory
- **Resources**: Tools and execution capabilities
- **Configuration**: Agent parameters and behavior
- **Lifecycle**: Initialization and cleanup (RAII pattern)

## Architecture

The system follows llama.cpp's context pattern with these core components:

### Core Files

Located in `include/`:

1. **agent_types.h** - Core type definitions
   - Message types and structures
   - Action types
   - Task definitions
   - Callback types

2. **agent_params.h** - Configuration parameters
   - Model settings
   - Memory configuration
   - Tool settings
   - Feature flags

3. **agent_memory.h** - Memory interface
   - Abstract memory interface
   - Buffer memory implementation
   - State persistence

4. **agent_tools.h** - Tool executor interface
   - Abstract tool executor
   - Function-based implementation

5. **agent_context.h** - Main context object
   - RAII lifecycle management
   - Resource ownership
   - Metrics tracking

6. **agent_executor.h** - Execution logic
   - Task execution
   - Action processing
   - State save/load

## Key Features

### 1. RAII Pattern
Resources are automatically managed through constructor/destructor:
```cpp
agent_context * ctx = agent_init(params);
// Use context...
agent_free(ctx);  // Automatic cleanup
```

### 2. Polymorphic Memory
Easy to swap memory implementations:
```cpp
params.memory_type = AGENT_MEMORY_BUFFER;  // FIFO buffer
params.memory_type = AGENT_MEMORY_VECTOR;  // Vector DB (future)
params.memory_type = AGENT_MEMORY_GRAPH;   // Graph memory (future)
```

### 3. Tool Registry
Dynamic tool registration:
```cpp
auto * tool_exec = static_cast<function_tool_executor*>(ctx->tools.get());
tool_exec->register_tool("calculator", calculator_tool);
tool_exec->register_tool("web_search", web_search_tool);
```

### 4. State Persistence
Save and restore agent state:
```cpp
agent_save_state(ctx, "session.state");
agent_load_state(ctx, "session.state");
```

### 5. Callbacks
Monitor execution without modifying core code:
```cpp
params.on_progress = my_progress_callback;
params.on_tool_call = my_tool_callback;
params.on_error = my_error_callback;
```

## Usage Example

```cpp
#include "agent_context.h"
#include "agent_executor.h"

int main() {
    // 1. Configure agent
    agent_params params = agent_default_params();
    params.model_name = "gpt-4";
    params.max_iterations = 10;

    // 2. Initialize context
    agent_context * ctx = agent_init(params);

    // 3. Register tools
    auto * tools = static_cast<function_tool_executor*>(ctx->tools.get());
    tools->register_tool("calculator", calculator_tool);

    // 4. Execute task
    agent_task task;
    task.instruction = "Calculate 2+2";
    task.max_steps = 5;

    agent_result result = agent_execute(ctx, task);

    // 5. Cleanup
    agent_free(ctx);

    return 0;
}
```

## Building

### Using Make (Recommended)

```bash
# Build everything (example + tests)
make

# Build only the example
make example

# Build only the tests
make test

# Clean build artifacts
make clean
```

### Manual Compilation

```bash
# Build the example
g++ -std=c++17 -I../../include agent-example.cpp -o agent-example

# Build the tests
g++ -std=c++17 -I../../include test-agent-context.cpp -o test-agent-context

# Using CMake (if integrated)
cmake --build . --target agent-example
```

## Testing

The project includes a comprehensive test suite that validates all components:

### Running Tests

```bash
# Using Make
make test

# Or run directly
./test-agent-context
```

### Test Coverage

The test suite includes 18 test cases covering:

- **Configuration**: Default parameters, custom settings
- **Context Management**: Initialization, cleanup, timing
- **Memory Operations**: Store, retrieve, overflow handling, persistence
- **Tool System**: Registration, execution, error handling
- **Task Execution**: Basic execution, iteration limits, metrics
- **State Persistence**: Save/load functionality
- **Callbacks**: Progress, tool call, and error callbacks
- **Edge Cases**: Null handling, buffer limits, tool failures

All tests use assertions and will exit with error codes on failure.

### Example Test Output

```
====================================
Agent Context Management Test Suite
====================================

Running default_params... PASSED
Running context_initialization... PASSED
Running memory_store_and_retrieve... PASSED
...
Running custom_memory_window_size... PASSED

====================================
All tests PASSED!
====================================
```

## Design Principles

Following llama.cpp patterns:

1. **Separation of Concerns**: Model, memory, execution are independent
2. **Interface Segregation**: Small, focused interfaces
3. **Dependency Inversion**: Depend on abstractions
4. **RAII**: Resource cleanup through destructors
5. **Factory Pattern**: Default parameter functions

## Extending the System

### Adding New Memory Types

Implement the `agent_memory_i` interface:

```cpp
class vector_memory : public agent_memory_i {
public:
    void store(const message & msg) override {
        // Implement vector DB storage
    }
    // ... implement other methods
};
```

### Adding New Tools

Register any function matching the signature:

```cpp
tool_result my_tool(const std::string & args) {
    tool_result result;
    // Implement tool logic
    return result;
}

tools->register_tool("my_tool", my_tool);
```

### Custom Backends

Implement the `tool_executor_i` interface for different execution backends (local, remote, sandboxed).

## Future Enhancements

Potential additions:

- [ ] Vector database memory implementation
- [ ] Graph-based memory for entity tracking
- [ ] Multi-backend tool execution (local/remote)
- [ ] Streaming execution support
- [ ] Distributed agent coordination
- [ ] Advanced metrics and profiling

## References

- Implementation guide: Based on llama.cpp context patterns
- Context analysis: See `CONTEXT_ANALYSIS_FOR_AI_AGENTS.md`
- llama.cpp source: `src/llama-context.h`, `src/llama-memory.h`
