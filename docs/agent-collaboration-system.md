# Agent Collaboration System for llama.cpp

## Overview

The Agent Collaboration System is a novel multi-agent architecture inspired by Apache Spark's distributed computing principles, specifically designed for coordinating multiple LLM-based agents in llama.cpp.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Orchestrator                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │Task Scheduler│  │Agent Registry│  │Event Bus     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────┬────────────────────────────────────────────────────┘
             │
    ┌────────┼────────┬────────────┬──────────────┐
    │        │        │            │              │
┌───▼───┐ ┌─▼────┐ ┌─▼─────┐  ┌──▼──────┐  ┌───▼────┐
│Agent 1│ │Agent 2│ │Agent 3│  │Agent N  │  │Agent M │
│Role:  │ │Role:  │ │Role:  │  │Role:    │  │Role:   │
│Planner│ │Coder  │ │Tester │  │Reviewer │  │Custom  │
└───┬───┘ └──┬────┘ └──┬────┘  └───┬─────┘  └────┬───┘
    │        │         │           │             │
    └────────┴─────────┴───────────┴─────────────┘
                       │
            ┌──────────▼────────────┐
            │  Shared Knowledge Base│
            │  ┌──────────────────┐ │
            │  │ Context Store    │ │
            │  │ Task Results     │ │
            │  │ Agent Messages   │ │
            │  │ Consensus Data   │ │
            │  └──────────────────┘ │
            └───────────────────────┘
```

## Core Components

### 1. Agent Orchestrator

**Purpose**: Central coordinator managing the entire agent ecosystem

**Responsibilities**:
- Agent lifecycle management (spawn, monitor, terminate)
- Task distribution and load balancing
- Inter-agent communication routing
- Resource allocation (slots, memory)
- Health monitoring and recovery

**Key Features**:
- **Dynamic Agent Provisioning**: Spawn agents on-demand based on workload
- **Fault Tolerance**: Automatic agent restart and task reassignment
- **Priority Scheduling**: Task prioritization based on urgency and dependencies
- **Metrics Collection**: Track agent performance and system health

### 2. Agent Worker

**Purpose**: Individual autonomous agent instance

**Attributes**:
- `agent_id`: Unique identifier
- `role`: Agent specialization (planner, coder, tester, reviewer, etc.)
- `slot_id`: Assigned llama.cpp slot
- `capabilities`: List of supported operations
- `state`: Current state (idle, busy, waiting, failed)

**Capabilities**:
- Execute assigned tasks autonomously
- Request assistance from other agents
- Share findings to knowledge base
- Participate in consensus voting
- Tool/function calling for external operations

**States**:
```
INITIALIZING → IDLE → ASSIGNED → EXECUTING → REPORTING → IDLE
                ↓
              WAITING (for other agents)
                ↓
              FAILED → RECOVERING → IDLE
```

### 3. Shared Knowledge Base

**Purpose**: Centralized memory and context sharing

**Data Structures**:

```cpp
// Shared context entry
struct knowledge_entry {
    std::string key;              // Unique identifier
    std::string value;            // Content (JSON, text, etc.)
    std::string contributor_id;   // Agent that added this
    int64_t timestamp;           // Creation time
    int version;                 // Version number
    std::vector<std::string> tags; // Categorization
};

// Task result storage
struct task_result {
    std::string task_id;
    std::string agent_id;
    std::string result;
    bool success;
    std::string error_message;
    int64_t duration_ms;
};
```

**Operations**:
- `put(key, value, agent_id)`: Store knowledge
- `get(key)`: Retrieve knowledge
- `query(tags)`: Search by tags
- `update(key, value, agent_id)`: Update existing entry (creates new version)
- `get_history(key)`: Get all versions
- `subscribe(key, agent_id)`: Watch for updates

### 4. Task Scheduler

**Purpose**: Intelligent task distribution and dependency management

**Task Types**:
```cpp
enum agent_task_type {
    TASK_TYPE_ANALYZE,      // Analyze code/data
    TASK_TYPE_GENERATE,     // Generate code/content
    TASK_TYPE_TEST,         // Run tests
    TASK_TYPE_REVIEW,       // Review work
    TASK_TYPE_REFACTOR,     // Refactor code
    TASK_TYPE_DOCUMENT,     // Generate documentation
    TASK_TYPE_CONSENSUS,    // Participate in voting
    TASK_TYPE_CUSTOM        // User-defined
};
```

**Task Structure**:
```cpp
struct agent_task {
    std::string task_id;
    agent_task_type type;
    std::string description;
    json parameters;
    std::vector<std::string> dependencies; // Tasks that must complete first
    std::vector<std::string> required_roles; // Agent roles that can handle this
    int priority;                        // 0-10 (10 = highest)
    std::string parent_task_id;          // For subtask tracking
    int64_t created_at;
    int64_t deadline;                    // Optional timeout
};
```

**Scheduling Algorithms**:
- **Priority Queue**: High-priority tasks first
- **Dependency Resolution**: Topological sort for task graphs
- **Load Balancing**: Distribute tasks evenly across agents
- **Role Matching**: Assign tasks to specialized agents

### 5. Communication Protocol

**Message Types**:
```cpp
enum message_type {
    MSG_TYPE_REQUEST,       // Agent requests assistance
    MSG_TYPE_RESPONSE,      // Reply to request
    MSG_TYPE_BROADCAST,     // Message to all agents
    MSG_TYPE_DIRECT,        // Point-to-point message
    MSG_TYPE_EVENT,         // Event notification
    MSG_TYPE_CONSENSUS      // Voting message
};
```

**Message Structure**:
```cpp
struct agent_message {
    std::string message_id;
    std::string from_agent_id;
    std::string to_agent_id;     // Empty for broadcast
    message_type type;
    std::string subject;
    json payload;
    int64_t timestamp;
    std::string conversation_id; // Thread multiple messages
};
```

**Communication Patterns**:
- **Request-Response**: Agent A requests help, Agent B responds
- **Publish-Subscribe**: Agents subscribe to event topics
- **Broadcast**: Send to all agents
- **Multicast**: Send to agents with specific roles

### 6. Consensus Mechanisms

**Purpose**: Enable multi-agent decision making

**Voting Types**:
```cpp
enum consensus_type {
    CONSENSUS_SIMPLE_MAJORITY,    // >50% agreement
    CONSENSUS_SUPERMAJORITY,      // >=66% agreement
    CONSENSUS_UNANIMOUS,          // 100% agreement
    CONSENSUS_WEIGHTED            // Weighted by agent expertise
};
```

**Consensus Structure**:
```cpp
struct consensus_vote {
    std::string vote_id;
    std::string question;
    std::vector<std::string> options;
    consensus_type type;
    std::map<std::string, std::string> votes; // agent_id -> option
    std::map<std::string, float> weights;     // agent_id -> weight
    int64_t deadline;
    std::string result;
    bool finalized;
};
```

**Use Cases**:
- Code review approval (3+ agents agree code is good)
- Design decisions (which architecture to use)
- Test result validation (consensus on bug existence)
- Priority decisions (which task to tackle next)

## API Endpoints

### Agent Management

```http
POST /v1/agents/spawn
{
  "role": "coder",
  "capabilities": ["python", "javascript"],
  "config": {...}
}
Response: { "agent_id": "agent-abc123", "slot_id": 2 }

GET /v1/agents
Response: [{ "agent_id": "...", "role": "...", "state": "idle" }, ...]

DELETE /v1/agents/{agent_id}
Response: { "success": true }

GET /v1/agents/{agent_id}/status
Response: { "agent_id": "...", "state": "executing", "current_task": "..." }
```

### Task Management

```http
POST /v1/tasks/submit
{
  "type": "TASK_TYPE_GENERATE",
  "description": "Implement binary search function",
  "parameters": { "language": "python" },
  "priority": 8
}
Response: { "task_id": "task-xyz789" }

GET /v1/tasks/{task_id}
Response: { "task_id": "...", "status": "completed", "result": {...} }

POST /v1/tasks/workflow
{
  "tasks": [
    { "id": "t1", "type": "TASK_TYPE_GENERATE", ... },
    { "id": "t2", "type": "TASK_TYPE_TEST", "dependencies": ["t1"], ... },
    { "id": "t3", "type": "TASK_TYPE_REVIEW", "dependencies": ["t1", "t2"], ... }
  ]
}
Response: { "workflow_id": "wf-123", "status": "scheduled" }
```

### Knowledge Base

```http
POST /v1/knowledge
{
  "key": "project_architecture",
  "value": "{ ... }",
  "tags": ["architecture", "design"]
}

GET /v1/knowledge/{key}
Response: { "key": "...", "value": {...}, "version": 3 }

GET /v1/knowledge/query?tags=architecture,design
Response: [{ "key": "...", "value": {...} }, ...]

POST /v1/knowledge/{key}/subscribe
{ "agent_id": "agent-abc123" }
```

### Communication

```http
POST /v1/messages/send
{
  "from_agent_id": "agent-abc123",
  "to_agent_id": "agent-xyz789",
  "type": "MSG_TYPE_REQUEST",
  "subject": "Code review request",
  "payload": { "code": "...", "file": "main.py" }
}

GET /v1/messages/{agent_id}
Response: [{ "message_id": "...", "from": "...", "subject": "..." }, ...]

POST /v1/messages/broadcast
{
  "from_agent_id": "agent-abc123",
  "subject": "Build completed",
  "payload": { "status": "success" }
}
```

### Consensus

```http
POST /v1/consensus/vote/create
{
  "question": "Approve pull request #123?",
  "options": ["approve", "reject", "request_changes"],
  "type": "CONSENSUS_SIMPLE_MAJORITY",
  "deadline": 1700000000
}
Response: { "vote_id": "vote-abc" }

POST /v1/consensus/vote/{vote_id}/cast
{
  "agent_id": "agent-abc123",
  "option": "approve"
}

GET /v1/consensus/vote/{vote_id}
Response: { "vote_id": "...", "result": "approve", "finalized": true }
```

## Example Use Cases

### Use Case 1: Collaborative Code Development

```
1. User submits: "Implement a REST API for user management"
2. Orchestrator spawns agents:
   - Agent 1 (Planner): Creates task breakdown
   - Agent 2 (Coder): Implements endpoints
   - Agent 3 (Tester): Writes tests
   - Agent 4 (Reviewer): Reviews code
3. Workflow:
   - Planner → creates tasks in knowledge base
   - Coder → implements based on plan
   - Tester → writes/runs tests
   - Reviewer → reviews and requests changes or approves
   - Consensus vote for final approval
4. Result: Fully implemented, tested, reviewed API
```

### Use Case 2: Bug Investigation

```
1. User reports: "Login fails intermittently"
2. Orchestrator creates investigation workflow:
   - Agent 1 (Analyzer): Examines logs and code
   - Agent 2 (Tester): Attempts to reproduce bug
   - Agent 3 (Researcher): Searches knowledge base for similar issues
3. Agents collaborate:
   - Share findings in knowledge base
   - Request additional info from each other
   - Vote on root cause consensus
4. Agent 4 (Fixer): Implements fix based on consensus
5. Agent 5 (Verifier): Validates fix
```

### Use Case 3: Multi-Agent Discussion

```
1. User asks: "What's the best database for our use case?"
2. Orchestrator spawns expert agents:
   - Agent 1: PostgreSQL expert
   - Agent 2: MongoDB expert
   - Agent 3: Redis expert
3. Discussion protocol:
   - Each agent presents case for their specialty
   - Agents debate trade-offs via messaging
   - Agents vote on final recommendation
4. Orchestrator synthesizes discussion and recommendation
```

## Implementation Notes

### Leveraging Existing llama.cpp Infrastructure

1. **Slots → Agent Workers**
   - Each agent uses one server slot
   - Slot state machine tracks agent state
   - Slot save/restore for agent persistence

2. **Task Queue → Agent Task Scheduler**
   - Extend `server_queue` for agent tasks
   - Add dependency resolution logic
   - Implement priority-based scheduling

3. **Tool Calling → Agent Actions**
   - Use existing `chat-parser-xml-toolcall.cpp`
   - Define agent collaboration tools
   - Enable agents to call orchestrator functions

4. **HTTP Server → Agent API**
   - Add new routes to `tools/server/server.cpp`
   - Implement RESTful agent management
   - WebSocket support for real-time messaging

5. **RPC Backend → Distributed Agents**
   - Run agents on multiple machines
   - Load balance across hardware
   - Scale horizontally

### Thread Safety

- All shared data structures use `std::mutex`
- Message passing via thread-safe queues
- Knowledge base with read-write locks
- Atomic operations for agent state transitions

### Performance Considerations

- **Agent Pooling**: Reuse idle agents instead of spawning new ones
- **Batch Processing**: Group similar tasks for efficiency
- **Caching**: Cache frequent knowledge base queries
- **Lazy Loading**: Load agent models on-demand
- **Connection Pooling**: Reuse HTTP/RPC connections

## Configuration

```json
{
  "agent_collaboration": {
    "max_agents": 10,
    "default_agent_timeout": 300000,
    "knowledge_base": {
      "max_entries": 10000,
      "persistence": true,
      "storage_path": "./agent_kb.db"
    },
    "scheduler": {
      "algorithm": "priority_queue",
      "max_queue_size": 1000
    },
    "consensus": {
      "default_type": "CONSENSUS_SIMPLE_MAJORITY",
      "voting_timeout": 60000
    },
    "communication": {
      "message_retention": 86400000,
      "max_message_size": 1048576
    }
  }
}
```

## Future Enhancements

1. **Machine Learning Integration**
   - Learn optimal task assignments from history
   - Predict task duration and resource needs
   - Adaptive scheduling based on agent performance

2. **Advanced Consensus**
   - Delegated voting (agents can delegate votes)
   - Multi-round voting with discussion
   - Reputation-weighted consensus

3. **Agent Specialization**
   - Dynamic capability discovery
   - Skill-based routing
   - Agent training/fine-tuning

4. **Visualization Dashboard**
   - Real-time agent activity view
   - Task dependency graphs
   - Performance metrics and analytics

5. **Security**
   - Agent authentication and authorization
   - Sandboxed agent execution
   - Audit logging for all agent actions

---

**Version**: 1.0
**Status**: Design Document
**Last Updated**: 2025-11-20
