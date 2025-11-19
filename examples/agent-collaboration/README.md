# Agent Collaboration Examples

This directory contains examples demonstrating the agent collaboration framework for llama.cpp.

## Overview

The agent collaboration framework enables multiple agents to work together, share context, and coordinate tasks with robust failure handling. The implementation is inspired by the zen-mcp-server architecture.

## Examples

### simple-chat.cpp

A comprehensive example demonstrating:
- Agent creation and registration
- Multi-turn conversations with memory
- Agent discovery by capabilities
- Multi-agent consensus gathering
- Failure handling and retry policies
- Statistics and monitoring

### Building

```bash
mkdir build && cd build
cmake ..
make simple-chat
```

### Running

```bash
./simple-chat
```

## Key Features Demonstrated

### 1. Agent Creation

```cpp
auto agent = agent_factory::create_local_agent(
    "Agent Name",
    "Description",
    {"capability1", "capability2"},
    &memory
);
```

### 2. Setting Inference Callback

```cpp
auto local_agent = static_cast<local_agent*>(agent.get());
local_agent->set_inference_callback([](const std::string& prompt, const auto& params) {
    // Your inference logic here
    return "response";
});
```

### 3. Registering Agents

```cpp
auto& registry = agent_registry::instance();
registry.register_agent(std::move(agent));
```

### 4. Sending Requests

```cpp
agent_request request;
request.prompt = "Your prompt here";
request.max_tokens = 500;

auto response = registry.send_request(agent_id, request);
```

### 5. Multi-Turn Conversations

```cpp
// First request
agent_request req1;
req1.prompt = "Initial question";
auto resp1 = registry.send_request(agent_id, req1);

// Continue conversation
agent_request req2;
req2.prompt = "Follow-up question";
req2.thread_id = resp1.thread_id;  // Link to previous conversation
auto resp2 = registry.send_request(agent_id, req2);
```

### 6. Multi-Agent Consensus

```cpp
auto consensus = registry.consensus_request(
    {agent_id1, agent_id2, agent_id3},
    request,
    true  // synthesize responses
);
```

### 7. Agent Discovery

```cpp
agent_query query;
query.capabilities = {"testing", "code_analysis"};
query.min_status = AGENT_STATUS_IDLE;

auto agents = registry.find_agents(query);
```

### 8. Failure Handling

```cpp
failure_policy policy = failure_policy::default_policy();
policy.max_retries = 3;
policy.enable_failover = true;
policy.fallback_agents = {backup_agent_id};

auto response = registry.send_request_with_policy(agent_id, request, policy);
```

## Architecture

The framework consists of:

- **Agents**: Individual processing units (local or remote)
- **Registry**: Central coordination and discovery
- **Conversation Memory**: Thread-based conversation tracking
- **Message Protocol**: Standardized communication format
- **Failure Handler**: Retry, failover, and circuit breaker patterns

## Integration with llama.cpp

To integrate with actual llama.cpp inference:

```cpp
// In your inference callback
local_agent->set_inference_callback([ctx](const std::string& prompt, const auto& params) {
    // Use llama_context to generate response
    // Parse params for temperature, max_tokens, etc.
    std::string response = your_llama_inference(ctx, prompt, params);
    return response;
});
```

## Best Practices

1. **Always set conversation memory** before registering agents that need multi-turn support
2. **Use failure policies** for production scenarios to handle transient errors
3. **Monitor agent statistics** to track performance and identify issues
4. **Clean up expired threads** periodically to manage memory
5. **Use agent discovery** instead of hardcoding agent IDs for flexibility

## Advanced Patterns

### Workflow Orchestration

```cpp
// Step 1: Code analysis
auto analysis = registry.send_request(code_agent_id, analyze_req);

// Step 2: Generate tests based on analysis
test_req.prompt = "Generate tests for: " + analysis.content;
test_req.thread_id = analysis.thread_id;
auto tests = registry.send_request(test_agent_id, test_req);

// Step 3: Document the code
doc_req.prompt = "Document this code and tests";
doc_req.thread_id = tests.thread_id;
auto docs = registry.send_request(doc_agent_id, doc_req);
```

### Context Branching

```cpp
// Branch conversation for alternative approaches
std::string new_thread_id = memory.branch_thread(original_thread_id, agent_id);

agent_request alt_req;
alt_req.prompt = "Try a different approach";
alt_req.thread_id = new_thread_id;
auto alt_response = registry.send_request(agent_id, alt_req);
```

## Troubleshooting

### Agents not responding

- Check agent status with `agent->get_info().status`
- Verify inference callback is set for local agents
- Check registry statistics for failures

### Thread not found errors

- Threads expire based on TTL (default 3 hours)
- Use `memory.touch_thread(thread_id)` to keep threads alive
- Check thread existence with `memory.has_thread(thread_id)`

### High memory usage

- Reduce conversation TTL
- Call `memory.cleanup_expired()` regularly
- Limit max_threads in conversation_memory constructor

## See Also

- [Agent Collaboration Documentation](../../docs/agent-collaboration.md)
- [zen-mcp-server](https://github.com/seyi/zen-mcp-server)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
