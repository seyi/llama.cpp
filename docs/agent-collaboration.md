# Agent Collaboration Framework for llama.cpp

## Overview

This document describes the agent collaboration framework for llama.cpp, inspired by the zen-mcp-server architecture. The framework enables multiple agents to collaborate, share context, and coordinate tasks while maintaining robust failure handling.

## Architecture

### Core Components

#### 1. Agent Registry
- **Purpose**: Manage agent lifecycle, discovery, and routing
- **Key Features**:
  - Agent registration and deregistration
  - Capability-based agent discovery
  - Health monitoring and status tracking
  - Load balancing across agents

#### 2. Conversation Memory
- **Purpose**: Maintain conversation state across agent interactions
- **Key Features**:
  - Thread-based conversation tracking with UUIDs
  - Turn-by-turn conversation history
  - Context reconstruction for continuation
  - TTL-based expiration (configurable, default 3 hours)
  - Token-aware context budgeting

#### 3. Message Protocol
- **Purpose**: Standard communication format between agents
- **Key Features**:
  - JSON-based message format
  - Request/response patterns
  - Streaming support for long-running tasks
  - Message prioritization
  - Broadcast and unicast modes

#### 4. Failure Handling
- **Purpose**: Graceful degradation and recovery
- **Key Features**:
  - Agent health checks
  - Automatic failover
  - Retry policies with exponential backoff
  - Dead letter queue for failed messages
  - Timeout handling

#### 5. Context Sharing
- **Purpose**: Enable agents to share state and artifacts
- **Key Features**:
  - File reference tracking
  - Shared memory for artifacts
  - Context handoff between agents
  - Scoped visibility (private, thread, global)

## Design Patterns (from zen-mcp-server)

### 1. Stateless-to-Stateful Bridge
- Agents are stateless but conversation memory provides stateful continuity
- Thread IDs enable context reconstruction
- Continuation offers allow multi-turn interactions

### 2. Dual Prioritization Strategy
- **Files**: Newest-first throughout (recent files take precedence)
- **Turns**: Newest-first collection, chronological presentation
- Token-aware budgeting prevents context overflow

### 3. Cross-Agent Continuation
- Any agent can continue a conversation started by another
- Full conversation history accessible via thread ID
- Tool attribution tracking

### 4. Provider Priority Cascade
- Native providers first (local llama.cpp)
- Remote providers as fallback
- Graceful degradation on failure

### 5. Blinded Consensus
- Agents can provide independent opinions
- Initial responses don't see other agents' outputs
- Final synthesis combines perspectives

## Implementation Structure

```
common/
├── agent/
│   ├── agent.h              # Base agent interface
│   ├── agent.cpp            # Agent implementation
│   ├── registry.h           # Agent registry
│   ├── registry.cpp         # Registry implementation
│   ├── conversation.h       # Conversation memory
│   ├── conversation.cpp     # Memory implementation
│   ├── message.h            # Message protocol
│   ├── message.cpp          # Protocol implementation
│   └── failure.h            # Failure handling utilities
│   └── failure.cpp          # Failure implementation
examples/
├── agent-collaboration/
│   ├── simple-chat.cpp      # Basic agent chat
│   ├── consensus.cpp        # Multi-agent consensus
│   ├── workflow.cpp         # Multi-step workflow
│   └── README.md            # Examples documentation
tools/
├── agent-server/
│   ├── agent-server.cpp     # Agent collaboration server
│   └── README.md            # Server documentation
```

## Core Data Structures

### Agent Definition
```cpp
struct agent_info {
    std::string id;              // Unique agent identifier (UUID)
    std::string name;            // Human-readable name
    std::string description;     // Agent purpose/capabilities
    std::vector<std::string> capabilities;  // Agent capabilities
    std::string endpoint;        // Connection endpoint
    agent_status status;         // ACTIVE, IDLE, BUSY, ERROR, OFFLINE
    int64_t last_heartbeat;      // Last heartbeat timestamp
    std::map<std::string, std::string> metadata;  // Custom metadata
};
```

### Conversation Thread
```cpp
struct conversation_turn {
    std::string role;            // "user", "assistant", "system"
    std::string content;         // Turn content
    int64_t timestamp;           // Unix timestamp
    std::vector<std::string> files;  // Referenced files
    std::string agent_id;        // Agent that created turn
    std::string model;           // Model used (if applicable)
    std::map<std::string, std::string> metadata;  // Custom metadata
};

struct conversation_thread {
    std::string thread_id;       // UUID
    std::string parent_id;       // Parent thread (for branching)
    int64_t created_at;          // Creation timestamp
    int64_t updated_at;          // Last update timestamp
    std::string initiating_agent;  // Agent that created thread
    std::vector<conversation_turn> turns;  // Conversation history
    std::map<std::string, std::string> context;  // Initial context
    int64_t expires_at;          // Expiration timestamp
};
```

### Message Protocol
```cpp
struct agent_message {
    std::string message_id;      // UUID
    std::string from_agent;      // Source agent ID
    std::string to_agent;        // Destination agent ID (empty for broadcast)
    message_type type;           // REQUEST, RESPONSE, NOTIFICATION, ERROR
    std::string payload;         // JSON payload
    std::string thread_id;       // Associated conversation thread
    int64_t timestamp;           // Message timestamp
    int priority;                // Message priority (0-10)
    std::map<std::string, std::string> metadata;  // Custom metadata
};

struct agent_request {
    std::string prompt;          // User prompt
    std::string thread_id;       // Continuation thread ID (optional)
    std::vector<std::string> files;  // File references
    std::map<std::string, std::string> params;  // Request parameters
    int max_tokens;              // Token limit
    float temperature;           // Sampling temperature
};

struct agent_response {
    response_status status;      // SUCCESS, ERROR, CONTINUATION_REQUIRED
    std::string content;         // Response content
    std::string thread_id;       // Thread ID for continuation
    int tokens_used;             // Tokens consumed
    std::string error_message;   // Error details (if failed)
    std::map<std::string, std::string> metadata;  // Custom metadata
};
```

### Failure Handling
```cpp
struct failure_policy {
    int max_retries;             // Maximum retry attempts
    int64_t retry_delay_ms;      // Initial retry delay
    float backoff_multiplier;    // Exponential backoff factor
    int64_t timeout_ms;          // Request timeout
    bool enable_failover;        // Auto-failover on failure
    std::vector<std::string> fallback_agents;  // Failover agents
};

struct failure_record {
    std::string agent_id;        // Failed agent
    std::string error_type;      // Error classification
    std::string error_message;   // Error details
    int64_t timestamp;           // Failure timestamp
    std::string thread_id;       // Associated thread
    std::string message_id;      // Failed message
    int retry_count;             // Retry attempts
};
```

## API Examples

### Registering an Agent
```cpp
agent_registry registry;
agent_info agent = {
    .id = "agent-001",
    .name = "Code Analyzer",
    .description = "Analyzes code quality and suggests improvements",
    .capabilities = {"code_analysis", "refactoring", "documentation"},
    .endpoint = "http://localhost:8080",
    .status = AGENT_STATUS_ACTIVE,
    .metadata = {{"version", "1.0"}, {"language", "cpp"}}
};
registry.register_agent(agent);
```

### Starting a Conversation
```cpp
conversation_memory memory;
agent_request request = {
    .prompt = "Analyze this code for performance issues",
    .files = {"/path/to/code.cpp"},
    .params = {{"analysis_depth", "thorough"}},
    .max_tokens = 2048,
    .temperature = 0.7
};

// Create new thread
std::string thread_id = memory.create_thread("agent-001", request);

// Add user turn
memory.add_turn(thread_id, "user", request.prompt, request.files, "agent-001");
```

### Continuing a Conversation
```cpp
agent_request continuation = {
    .prompt = "Can you provide code examples for the improvements?",
    .thread_id = thread_id,  // Continue existing thread
    .max_tokens = 2048
};

// Reconstruct context
conversation_thread thread = memory.get_thread(thread_id);
std::string full_context = memory.build_conversation_history(thread);
```

### Agent-to-Agent Messaging
```cpp
agent_message msg = {
    .message_id = generate_uuid(),
    .from_agent = "agent-001",
    .to_agent = "agent-002",
    .type = MESSAGE_TYPE_REQUEST,
    .payload = request_to_json(request),
    .thread_id = thread_id,
    .priority = 5
};

registry.send_message(msg);
```

### Multi-Agent Consensus
```cpp
// Consult multiple agents on same prompt
std::vector<std::string> agent_ids = {"agent-001", "agent-002", "agent-003"};
std::vector<agent_response> responses;

for (const auto& agent_id : agent_ids) {
    auto response = registry.send_request(agent_id, request);
    responses.push_back(response);
}

// Synthesize consensus
std::string consensus = synthesize_responses(responses);
```

### Failure Handling
```cpp
failure_policy policy = {
    .max_retries = 3,
    .retry_delay_ms = 1000,
    .backoff_multiplier = 2.0,
    .timeout_ms = 30000,
    .enable_failover = true,
    .fallback_agents = {"agent-002", "agent-003"}
};

auto response = registry.send_request_with_policy("agent-001", request, policy);
if (response.status == RESPONSE_STATUS_ERROR) {
    // Handle failure
    auto failure = registry.get_last_failure("agent-001");
    LOG_ERROR("Agent failed: %s", failure.error_message.c_str());
}
```

## Integration with llama.cpp Server

The agent collaboration framework integrates with the existing llama-server:

1. **New Endpoints**:
   - `POST /v1/agents/register` - Register new agent
   - `POST /v1/agents/message` - Send message to agent
   - `POST /v1/agents/consensus` - Multi-agent consensus
   - `GET /v1/agents/threads/{id}` - Get conversation thread
   - `POST /v1/agents/threads` - Create new thread

2. **Server Extension**:
   - Agent registry integrated into server context
   - Conversation memory managed per-server instance
   - Message queue for async processing
   - Health monitoring endpoint

## Configuration

```json
{
  "agent_collaboration": {
    "enabled": true,
    "conversation_ttl_hours": 3,
    "max_threads": 10000,
    "max_turns_per_thread": 1000,
    "message_queue_size": 10000,
    "health_check_interval_ms": 30000,
    "failure_policy": {
      "max_retries": 3,
      "retry_delay_ms": 1000,
      "backoff_multiplier": 2.0,
      "timeout_ms": 30000,
      "enable_failover": true
    }
  }
}
```

## Future Enhancements

1. **Distributed Agent Registry**: Support for agent discovery across multiple nodes
2. **Event Bus**: Pub/sub messaging for complex workflows
3. **Agent Orchestration**: DAG-based task orchestration
4. **Persistent Storage**: Database backend for conversation history
5. **Metrics & Monitoring**: Prometheus/OpenTelemetry integration
6. **Security**: Authentication, authorization, encryption
7. **Consensus Algorithms**: Voting, weighted consensus, confidence scoring

## References

- zen-mcp-server: https://github.com/seyi/zen-mcp-server
- MCP Protocol: https://modelcontextprotocol.io/
- llama.cpp: https://github.com/ggerganov/llama.cpp
