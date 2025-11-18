# AI Agent Context Implementation Guide

## Quick Start Implementation

This guide provides a minimal working implementation of an agent context system based on llama.cpp patterns.

---

## Minimal Implementation (C++)

### 1. Core Types and Structures

```cpp
// agent_types.h
#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <any>

// Forward declarations
struct agent_context;
class agent_memory_i;
class tool_executor_i;

// Message types
enum message_role {
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL
};

struct message {
    message_role role;
    std::string content;
    std::map<std::string, std::any> metadata;
    int64_t timestamp_us;
};

// Action types
enum action_type {
    ACTION_REASONING,
    ACTION_TOOL_CALL,
    ACTION_FINAL_ANSWER,
    ACTION_WAIT
};

struct agent_action {
    action_type type;
    std::string description;
    std::string tool_name;  // For ACTION_TOOL_CALL
    std::string arguments;  // JSON string
    bool is_final;
};

struct agent_result {
    std::string output;
    std::vector<agent_action> steps;
    size_t tool_calls_count;
    int64_t execution_time_us;
    bool success;
};

// Task definition
struct agent_task {
    std::string instruction;
    std::map<std::string, std::any> context;
    size_t max_steps;
};

// Callbacks
using progress_callback_fn = std::function<void(size_t, const agent_action&, const std::string&)>;
using tool_callback_fn = std::function<void(const std::string&, const std::string&)>;
using error_callback_fn = std::function<void(const std::string&)>;
```

### 2. Agent Parameters

```cpp
// agent_params.h
#pragma once
#include "agent_types.h"

enum agent_memory_type {
    AGENT_MEMORY_BUFFER,
    AGENT_MEMORY_VECTOR,
    AGENT_MEMORY_GRAPH
};

struct agent_params {
    // Model configuration
    std::string model_name;
    std::string api_key;
    std::string api_base_url;

    // Context sizing
    size_t max_context_tokens;
    size_t max_completion_tokens;
    size_t max_iterations;

    // Memory configuration
    agent_memory_type memory_type;
    size_t memory_window_size;
    bool enable_memory_persistence;

    // Tool configuration
    std::vector<std::string> enabled_tools;
    bool allow_parallel_tool_calls;
    size_t max_tool_calls_per_step;

    // Generation parameters
    float temperature;
    float top_p;
    int seed;

    // Callbacks (optional)
    progress_callback_fn on_progress;
    tool_callback_fn on_tool_call;
    error_callback_fn on_error;

    // Feature flags
    bool enable_streaming;
    bool enable_logging;
    bool enable_metrics;
    bool enable_caching;

    // Threading
    int32_t n_threads;
};

// Default parameters
inline agent_params agent_default_params() {
    agent_params params;

    // Model defaults
    params.model_name = "gpt-4";
    params.api_key = "";
    params.api_base_url = "https://api.openai.com/v1";

    // Context defaults
    params.max_context_tokens = 8192;
    params.max_completion_tokens = 2048;
    params.max_iterations = 20;

    // Memory defaults
    params.memory_type = AGENT_MEMORY_BUFFER;
    params.memory_window_size = 10;
    params.enable_memory_persistence = false;

    // Tool defaults
    params.enabled_tools = {};
    params.allow_parallel_tool_calls = false;
    params.max_tool_calls_per_step = 1;

    // Generation defaults
    params.temperature = 0.7f;
    params.top_p = 1.0f;
    params.seed = -1;

    // Callbacks (null by default)
    params.on_progress = nullptr;
    params.on_tool_call = nullptr;
    params.on_error = nullptr;

    // Feature flags
    params.enable_streaming = false;
    params.enable_logging = true;
    params.enable_metrics = true;
    params.enable_caching = false;

    // Threading
    params.n_threads = 1;

    return params;
}
```

### 3. Memory Interface

```cpp
// agent_memory.h
#pragma once
#include "agent_types.h"
#include <vector>
#include <deque>

// Abstract memory interface (similar to llama_memory_i)
class agent_memory_i {
public:
    virtual ~agent_memory_i() = default;

    // Store a message
    virtual void store(const message & msg) = 0;

    // Retrieve relevant context
    virtual std::vector<message> retrieve_all() const = 0;
    virtual std::vector<message> retrieve_recent(size_t n) const = 0;

    // Clear memory
    virtual void clear() = 0;

    // State persistence
    virtual size_t save_state(std::ostream & out) const = 0;
    virtual size_t load_state(std::istream & in) = 0;

    // Statistics
    virtual size_t size() const = 0;
    virtual bool is_full() const = 0;
};

// Simple buffer memory implementation
class buffer_memory : public agent_memory_i {
public:
    explicit buffer_memory(size_t max_size) : max_size(max_size) {}

    void store(const message & msg) override {
        messages.push_back(msg);
        if (messages.size() > max_size) {
            messages.pop_front();
        }
    }

    std::vector<message> retrieve_all() const override {
        return std::vector<message>(messages.begin(), messages.end());
    }

    std::vector<message> retrieve_recent(size_t n) const override {
        std::vector<message> result;
        size_t start = messages.size() > n ? messages.size() - n : 0;
        for (size_t i = start; i < messages.size(); ++i) {
            result.push_back(messages[i]);
        }
        return result;
    }

    void clear() override {
        messages.clear();
    }

    size_t save_state(std::ostream & out) const override {
        // Simple serialization
        size_t written = 0;
        size_t count = messages.size();
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        written += sizeof(count);

        for (const auto & msg : messages) {
            // Serialize each message
            // (simplified - production code would use proper serialization)
            size_t role = static_cast<size_t>(msg.role);
            out.write(reinterpret_cast<const char*>(&role), sizeof(role));

            size_t content_size = msg.content.size();
            out.write(reinterpret_cast<const char*>(&content_size), sizeof(content_size));
            out.write(msg.content.data(), content_size);

            written += sizeof(role) + sizeof(content_size) + content_size;
        }

        return written;
    }

    size_t load_state(std::istream & in) override {
        size_t read = 0;
        size_t count;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        read += sizeof(count);

        messages.clear();
        for (size_t i = 0; i < count; ++i) {
            message msg;

            size_t role;
            in.read(reinterpret_cast<char*>(&role), sizeof(role));
            msg.role = static_cast<message_role>(role);

            size_t content_size;
            in.read(reinterpret_cast<char*>(&content_size), sizeof(content_size));

            msg.content.resize(content_size);
            in.read(&msg.content[0], content_size);

            messages.push_back(msg);
            read += sizeof(role) + sizeof(content_size) + content_size;
        }

        return read;
    }

    size_t size() const override {
        return messages.size();
    }

    bool is_full() const override {
        return messages.size() >= max_size;
    }

private:
    std::deque<message> messages;
    size_t max_size;
};

using agent_memory_ptr = std::unique_ptr<agent_memory_i>;
```

### 4. Tool Executor Interface

```cpp
// agent_tools.h
#pragma once
#include "agent_types.h"
#include <functional>

struct tool_result {
    bool success;
    std::string output;
    std::string error;
    int64_t execution_time_us;
};

// Abstract tool executor
class tool_executor_i {
public:
    virtual ~tool_executor_i() = default;

    virtual tool_result execute(
        const std::string & tool_name,
        const std::string & arguments) = 0;

    virtual bool has_tool(const std::string & tool_name) const = 0;
    virtual std::vector<std::string> list_tools() const = 0;
};

// Simple function-based tool executor
class function_tool_executor : public tool_executor_i {
public:
    using tool_fn = std::function<tool_result(const std::string&)>;

    void register_tool(const std::string & name, tool_fn fn) {
        tools[name] = fn;
    }

    tool_result execute(const std::string & tool_name, const std::string & arguments) override {
        auto it = tools.find(tool_name);
        if (it == tools.end()) {
            return tool_result{false, "", "Tool not found: " + tool_name, 0};
        }

        auto start = std::chrono::steady_clock::now();
        tool_result result = it->second(arguments);
        auto end = std::chrono::steady_clock::now();

        result.execution_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count();

        return result;
    }

    bool has_tool(const std::string & tool_name) const override {
        return tools.find(tool_name) != tools.end();
    }

    std::vector<std::string> list_tools() const override {
        std::vector<std::string> result;
        for (const auto & [name, _] : tools) {
            result.push_back(name);
        }
        return result;
    }

private:
    std::map<std::string, tool_fn> tools;
};

using tool_executor_ptr = std::unique_ptr<tool_executor_i>;
```

### 5. Agent Context

```cpp
// agent_context.h
#pragma once
#include "agent_types.h"
#include "agent_params.h"
#include "agent_memory.h"
#include "agent_tools.h"
#include <chrono>

struct agent_metrics {
    size_t total_iterations;
    size_t total_tool_calls;
    size_t total_tokens_used;
    int64_t total_time_us;

    void reset() {
        total_iterations = 0;
        total_tool_calls = 0;
        total_tokens_used = 0;
        total_time_us = 0;
    }
};

struct agent_context {
    // Constructor (RAII pattern)
    agent_context(const agent_params & params_in)
        : params(params_in)
        , t_start_us(get_time_us()) {

        // Initialize memory based on type
        switch (params.memory_type) {
            case AGENT_MEMORY_BUFFER:
                memory = std::make_unique<buffer_memory>(params.memory_window_size);
                break;
            // Add other memory types here
            default:
                memory = std::make_unique<buffer_memory>(params.memory_window_size);
        }

        // Initialize tool executor
        tools = std::make_unique<function_tool_executor>();

        // Reset metrics
        metrics.reset();
    }

    // Destructor (automatic cleanup)
    ~agent_context() {
        if (params.enable_logging) {
            // Log final metrics
            // fprintf(stderr, "Agent context destroyed. Total time: %lld us\n", get_time_us() - t_start_us);
        }
    }

    // Prevent copying (follow llama.cpp pattern)
    agent_context(const agent_context &) = delete;
    agent_context & operator=(const agent_context &) = delete;

    // Get current time in microseconds
    static int64_t get_time_us() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    }

    // Public members (following llama.cpp style)
    agent_params params;
    agent_memory_ptr memory;
    tool_executor_ptr tools;
    agent_metrics metrics;

    int64_t t_start_us;
    int64_t t_last_exec_us;
};

// Factory function (similar to llama_init_from_model)
inline agent_context * agent_init(const agent_params & params) {
    try {
        return new agent_context(params);
    } catch (const std::exception & err) {
        if (params.on_error) {
            params.on_error(std::string("Failed to initialize agent: ") + err.what());
        }
        return nullptr;
    }
}

// Cleanup function (similar to llama_free)
inline void agent_free(agent_context * ctx) {
    delete ctx;
}
```

### 6. Agent Execution

```cpp
// agent_executor.h
#pragma once
#include "agent_context.h"
#include <sstream>

// Execute a task with the agent
inline agent_result agent_execute(agent_context * ctx, const agent_task & task) {
    if (!ctx) {
        return agent_result{"", {}, 0, 0, false};
    }

    auto start_time = agent_context::get_time_us();
    agent_result result;
    result.success = false;

    // Add task to memory as a user message
    message user_msg;
    user_msg.role = ROLE_USER;
    user_msg.content = task.instruction;
    user_msg.timestamp_us = start_time;
    ctx->memory->store(user_msg);

    // Execute task in steps
    size_t max_steps = task.max_steps > 0 ? task.max_steps : ctx->params.max_iterations;

    for (size_t step = 0; step < max_steps; ++step) {
        ctx->metrics.total_iterations++;

        // Get next action from agent
        // This would call your LLM here
        agent_action action = get_next_action(ctx);

        if (ctx->params.on_progress) {
            ctx->params.on_progress(step, action, "Processing...");
        }

        if (action.is_final) {
            result.output = action.description;
            result.success = true;
            break;
        }

        // Execute action
        if (action.type == ACTION_TOOL_CALL) {
            ctx->metrics.total_tool_calls++;

            if (ctx->params.on_tool_call) {
                ctx->params.on_tool_call(action.tool_name, action.arguments);
            }

            tool_result tool_res = ctx->tools->execute(action.tool_name, action.arguments);

            // Store tool result in memory
            message tool_msg;
            tool_msg.role = ROLE_TOOL;
            tool_msg.content = tool_res.output;
            tool_msg.metadata["tool_name"] = action.tool_name;
            tool_msg.timestamp_us = agent_context::get_time_us();
            ctx->memory->store(tool_msg);

            if (!tool_res.success && ctx->params.on_error) {
                ctx->params.on_error("Tool execution failed: " + tool_res.error);
            }
        }

        result.steps.push_back(action);
    }

    result.tool_calls_count = ctx->metrics.total_tool_calls;
    result.execution_time_us = agent_context::get_time_us() - start_time;
    ctx->metrics.total_time_us += result.execution_time_us;

    return result;
}

// Placeholder for LLM call (implement with your LLM provider)
inline agent_action get_next_action(agent_context * ctx) {
    // This is where you would:
    // 1. Get conversation history from memory
    // 2. Format as prompt for LLM
    // 3. Call LLM API
    // 4. Parse response to extract action

    // Simplified example:
    auto history = ctx->memory->retrieve_all();

    // TODO: Call LLM with history
    // std::string llm_response = call_llm(ctx->params.model_name, history);

    // TODO: Parse LLM response to action
    // return parse_action(llm_response);

    // Dummy action for compilation
    agent_action action;
    action.type = ACTION_FINAL_ANSWER;
    action.description = "Task complete";
    action.is_final = true;
    return action;
}

// Save context state
inline bool agent_save_state(agent_context * ctx, const std::string & filepath) {
    if (!ctx) return false;

    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;

    // Save memory state
    ctx->memory->save_state(file);

    // Save metrics
    file.write(reinterpret_cast<const char*>(&ctx->metrics), sizeof(ctx->metrics));

    return true;
}

// Load context state
inline bool agent_load_state(agent_context * ctx, const std::string & filepath) {
    if (!ctx) return false;

    std::ifstream file(filepath, std::ios::binary);
    if (!file) return false;

    // Load memory state
    ctx->memory->load_state(file);

    // Load metrics
    file.read(reinterpret_cast<char*>(&ctx->metrics), sizeof(ctx->metrics));

    return true;
}
```

---

## Complete Usage Example

```cpp
#include "agent_context.h"
#include "agent_executor.h"
#include <iostream>

// Example tool: calculator
tool_result calculator_tool(const std::string & args) {
    // Parse args (simplified)
    // Expected format: "2 + 2"
    tool_result result;
    result.success = true;
    result.output = "4";  // Simplified
    return result;
}

// Example tool: web search
tool_result web_search_tool(const std::string & args) {
    tool_result result;
    result.success = true;
    result.output = "Search results for: " + args;
    return result;
}

// Progress callback
void my_progress_callback(size_t step, const agent_action & action, const std::string & status) {
    std::cout << "[Step " << step << "] " << action.description << " - " << status << std::endl;
}

int main() {
    // 1. Configure agent
    agent_params params = agent_default_params();
    params.model_name = "gpt-4";
    params.max_iterations = 10;
    params.memory_type = AGENT_MEMORY_BUFFER;
    params.memory_window_size = 20;
    params.enable_logging = true;
    params.on_progress = my_progress_callback;

    // 2. Initialize context (RAII - automatic cleanup on scope exit)
    agent_context * ctx = agent_init(params);
    if (!ctx) {
        std::cerr << "Failed to initialize agent context" << std::endl;
        return 1;
    }

    // 3. Register tools
    auto * tool_exec = static_cast<function_tool_executor*>(ctx->tools.get());
    tool_exec->register_tool("calculator", calculator_tool);
    tool_exec->register_tool("web_search", web_search_tool);

    // 4. Create and execute task
    agent_task task;
    task.instruction = "Search for the latest AI news and calculate 2+2";
    task.max_steps = 5;

    agent_result result = agent_execute(ctx, task);

    // 5. Display results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Success: " << (result.success ? "Yes" : "No") << std::endl;
    std::cout << "Output: " << result.output << std::endl;
    std::cout << "Steps taken: " << result.steps.size() << std::endl;
    std::cout << "Tool calls: " << result.tool_calls_count << std::endl;
    std::cout << "Execution time: " << result.execution_time_us / 1000.0 << " ms" << std::endl;

    // 6. Save state for later
    if (agent_save_state(ctx, "session.state")) {
        std::cout << "State saved successfully" << std::endl;
    }

    // 7. Cleanup (automatic via RAII, but explicit call shown here)
    agent_free(ctx);

    return 0;
}
```

---

## Python Implementation (Simplified)

For those preferring Python:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from abc import ABC, abstractmethod
import time
import pickle

@dataclass
class Message:
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp_us: int = 0

@dataclass
class AgentAction:
    type: str  # 'reasoning', 'tool_call', 'final_answer'
    description: str
    tool_name: Optional[str] = None
    arguments: Optional[str] = None
    is_final: bool = False

@dataclass
class AgentResult:
    output: str
    steps: List[AgentAction] = field(default_factory=list)
    tool_calls_count: int = 0
    execution_time_us: int = 0
    success: bool = False

class AgentMemory(ABC):
    """Abstract memory interface"""

    @abstractmethod
    def store(self, msg: Message) -> None:
        pass

    @abstractmethod
    def retrieve_all(self) -> List[Message]:
        pass

    @abstractmethod
    def retrieve_recent(self, n: int) -> List[Message]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

class BufferMemory(AgentMemory):
    """Simple FIFO buffer memory"""

    def __init__(self, max_size: int = 10):
        self.messages: List[Message] = []
        self.max_size = max_size

    def store(self, msg: Message) -> None:
        self.messages.append(msg)
        if len(self.messages) > self.max_size:
            self.messages.pop(0)

    def retrieve_all(self) -> List[Message]:
        return self.messages.copy()

    def retrieve_recent(self, n: int) -> List[Message]:
        return self.messages[-n:] if len(self.messages) >= n else self.messages.copy()

    def clear(self) -> None:
        self.messages.clear()

@dataclass
class AgentParams:
    model_name: str = "gpt-4"
    max_iterations: int = 20
    memory_window_size: int = 10
    temperature: float = 0.7
    enable_logging: bool = True
    on_progress: Optional[Callable] = None
    on_tool_call: Optional[Callable] = None
    on_error: Optional[Callable] = None

class AgentContext:
    """Main context object (RAII pattern via context manager)"""

    def __init__(self, params: AgentParams):
        self.params = params
        self.memory = BufferMemory(params.memory_window_size)
        self.tools: Dict[str, Callable] = {}
        self.metrics = {
            'total_iterations': 0,
            'total_tool_calls': 0,
            'total_time_us': 0
        }
        self.t_start_us = time.time_ns() // 1000

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (automatic cleanup)"""
        if self.params.enable_logging:
            total_time = (time.time_ns() // 1000) - self.t_start_us
            print(f"Agent context closed. Total time: {total_time / 1000:.2f} ms")
        return False

    def register_tool(self, name: str, func: Callable) -> None:
        """Register a tool function"""
        self.tools[name] = func

    def execute_tool(self, name: str, args: str) -> Dict[str, Any]:
        """Execute a registered tool"""
        if name not in self.tools:
            return {'success': False, 'output': '', 'error': f'Tool not found: {name}'}

        try:
            start = time.time_ns() // 1000
            output = self.tools[name](args)
            end = time.time_ns() // 1000
            return {
                'success': True,
                'output': str(output),
                'error': '',
                'execution_time_us': end - start
            }
        except Exception as e:
            return {'success': False, 'output': '', 'error': str(e)}

    def save_state(self, filepath: str) -> bool:
        """Save context state to file"""
        try:
            with open(filepath, 'wb') as f:
                state = {
                    'messages': self.memory.messages,
                    'metrics': self.metrics
                }
                pickle.dump(state, f)
            return True
        except Exception as e:
            if self.params.on_error:
                self.params.on_error(f"Failed to save state: {e}")
            return False

    def load_state(self, filepath: str) -> bool:
        """Load context state from file"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                self.memory.messages = state['messages']
                self.metrics = state['metrics']
            return True
        except Exception as e:
            if self.params.on_error:
                self.params.on_error(f"Failed to load state: {e}")
            return False

def agent_execute(ctx: AgentContext, task: Dict[str, Any]) -> AgentResult:
    """Execute a task with the agent"""
    start_time = time.time_ns() // 1000

    # Add task to memory
    user_msg = Message(
        role='user',
        content=task['instruction'],
        timestamp_us=start_time
    )
    ctx.memory.store(user_msg)

    result = AgentResult(output='', success=False)
    max_steps = task.get('max_steps', ctx.params.max_iterations)

    for step in range(max_steps):
        ctx.metrics['total_iterations'] += 1

        # Get next action (would call LLM here)
        action = get_next_action(ctx)

        if ctx.params.on_progress:
            ctx.params.on_progress(step, action, "Processing...")

        if action.is_final:
            result.output = action.description
            result.success = True
            break

        # Execute tool if needed
        if action.type == 'tool_call':
            ctx.metrics['total_tool_calls'] += 1

            if ctx.params.on_tool_call:
                ctx.params.on_tool_call(action.tool_name, action.arguments)

            tool_result = ctx.execute_tool(action.tool_name, action.arguments or '')

            # Store tool result
            tool_msg = Message(
                role='tool',
                content=tool_result['output'],
                metadata={'tool_name': action.tool_name},
                timestamp_us=time.time_ns() // 1000
            )
            ctx.memory.store(tool_msg)

        result.steps.append(action)

    result.tool_calls_count = ctx.metrics['total_tool_calls']
    result.execution_time_us = (time.time_ns() // 1000) - start_time

    return result

def get_next_action(ctx: AgentContext) -> AgentAction:
    """Get next action from agent (placeholder for LLM call)"""
    # TODO: Call LLM with conversation history
    # history = ctx.memory.retrieve_all()
    # llm_response = call_llm(ctx.params.model_name, history)
    # return parse_action(llm_response)

    # Dummy action for now
    return AgentAction(
        type='final_answer',
        description='Task complete',
        is_final=True
    )

# Usage example
if __name__ == '__main__':
    # Configure agent
    params = AgentParams(
        model_name='gpt-4',
        max_iterations=10,
        memory_window_size=20,
        enable_logging=True,
        on_progress=lambda step, action, status: print(f"[Step {step}] {action.description}")
    )

    # Use context manager for automatic cleanup
    with AgentContext(params) as ctx:
        # Register tools
        ctx.register_tool('calculator', lambda args: eval(args))
        ctx.register_tool('web_search', lambda args: f"Search results for: {args}")

        # Execute task
        task = {
            'instruction': 'Calculate 2+2 and search for AI news',
            'max_steps': 5
        }

        result = agent_execute(ctx, task)

        # Display results
        print(f"\n=== Results ===")
        print(f"Success: {result.success}")
        print(f"Output: {result.output}")
        print(f"Steps: {len(result.steps)}")
        print(f"Tool calls: {result.tool_calls_count}")
        print(f"Time: {result.execution_time_us / 1000:.2f} ms")

        # Save state
        ctx.save_state('session.pkl')
```

---

## Key Implementation Notes

1. **RAII Pattern**: Context automatically manages resources via constructor/destructor (C++) or context manager (Python)

2. **Polymorphic Memory**: Easy to swap memory implementations by changing the memory_type parameter

3. **Tool Registry**: Dynamic tool registration allows flexible agent capabilities

4. **Callbacks**: Progress, tool, and error callbacks enable monitoring without modifying core code

5. **State Persistence**: Save/load enables session resumption and debugging

6. **Metrics Tracking**: Built-in performance measurement for optimization

7. **Type Safety**: Strong typing prevents common errors and improves maintainability

This minimal implementation captures the essence of llama.cpp's context pattern while remaining simple enough to understand and extend.
