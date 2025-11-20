# Agent Collaboration System - Examples

This directory contains examples demonstrating the Agent Collaboration System for llama.cpp.

## Overview

The Agent Collaboration System enables multiple LLM agents to work together on complex tasks, inspired by Apache Spark's distributed computing architecture.

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Agent Orchestrator                       │
│  (Central coordinator for all agents)            │
└────────────┬────────────────────────────────────┘
             │
    ┌────────┼────────┬──────────┬─────────────┐
    │        │        │          │             │
┌───▼───┐ ┌─▼────┐ ┌─▼─────┐ ┌──▼──────┐  ┌───▼────┐
│Agent 1│ │Agent 2│ │Agent 3│ │Agent 4  │  │Agent N │
└───┬───┘ └──┬────┘ └──┬────┘ └───┬─────┘  └────┬───┘
    │        │         │          │             │
    └────────┴─────────┴──────────┴─────────────┘
                       │
            ┌──────────▼────────────┐
            │  Shared Knowledge Base│
            └───────────────────────┘
```

## Key Features

1. **Multi-Agent Coordination**: Spawn multiple agents with different roles (planner, coder, tester, reviewer)
2. **Task Scheduling**: Intelligent task distribution with dependency resolution
3. **Shared Knowledge Base**: Agents can share information and context
4. **Inter-Agent Communication**: Message passing between agents
5. **Consensus Mechanisms**: Multi-agent decision making through voting
6. **Workflow Management**: Define complex task workflows with dependencies

## Quick Start

### 1. Start the server with agent collaboration enabled:

```bash
./llama-server -m model.gguf --n-parallel 5
```

### 2. Spawn agents:

```bash
# Spawn a planner agent
curl -X POST http://localhost:8080/v1/agents/spawn \
  -H "Content-Type: application/json" \
  -d '{
    "role": "planner",
    "slot_id": 0,
    "capabilities": ["task_breakdown", "architecture_design"]
  }'

# Spawn a coder agent
curl -X POST http://localhost:8080/v1/agents/spawn \
  -H "Content-Type: application/json" \
  -d '{
    "role": "coder",
    "slot_id": 1,
    "capabilities": ["python", "javascript", "code_generation"]
  }'

# Spawn a tester agent
curl -X POST http://localhost:8080/v1/agents/spawn \
  -H "Content-Type: application/json" \
  -d '{
    "role": "tester",
    "slot_id": 2,
    "capabilities": ["unit_testing", "integration_testing"]
  }'
```

### 3. Submit a task workflow:

```bash
curl -X POST http://localhost:8080/v1/tasks/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [
      {
        "id": "plan",
        "type": "analyze",
        "description": "Create implementation plan for REST API",
        "required_roles": ["planner"],
        "priority": 10
      },
      {
        "id": "implement",
        "type": "generate",
        "description": "Implement REST API based on plan",
        "dependencies": ["plan"],
        "required_roles": ["coder"],
        "priority": 8
      },
      {
        "id": "test",
        "type": "test",
        "description": "Write and run tests for REST API",
        "dependencies": ["implement"],
        "required_roles": ["tester"],
        "priority": 8
      },
      {
        "id": "review",
        "type": "review",
        "description": "Review implementation and tests",
        "dependencies": ["implement", "test"],
        "required_roles": ["reviewer"],
        "priority": 7
      }
    ]
  }'
```

## Example Use Cases

### Use Case 1: Collaborative Code Development

Multiple agents work together to implement a feature:

1. **Planner Agent**: Breaks down the feature into tasks
2. **Coder Agent**: Implements the code
3. **Tester Agent**: Writes and runs tests
4. **Reviewer Agent**: Reviews the implementation
5. **Consensus**: All agents vote on whether to merge

See: [collaborative-coding.py](./collaborative-coding.py)

### Use Case 2: Bug Investigation

Agents collaborate to investigate and fix bugs:

1. **Analyzer Agent**: Examines logs and code
2. **Tester Agent**: Attempts to reproduce the bug
3. **Researcher Agent**: Searches knowledge base for similar issues
4. **Fixer Agent**: Implements the fix based on findings
5. **Verifier Agent**: Validates the fix

See: [bug-investigation.py](./bug-investigation.py)

### Use Case 3: Multi-Agent Discussion

Agents with different expertise discuss and recommend solutions:

1. Spawn expert agents (PostgreSQL, MongoDB, Redis experts)
2. Each agent presents their case
3. Agents debate via messaging
4. Consensus vote on final recommendation

See: [multi-agent-discussion.py](./multi-agent-discussion.py)

## API Reference

### Agent Management

#### Spawn Agent
```http
POST /v1/agents/spawn
{
  "role": "coder",
  "slot_id": 1,
  "capabilities": ["python", "javascript"],
  "config": {}
}
```

#### List Agents
```http
GET /v1/agents
```

#### Get Agent Info
```http
GET /v1/agents/{agent_id}
```

#### Terminate Agent
```http
DELETE /v1/agents/{agent_id}
```

### Task Management

#### Submit Task
```http
POST /v1/tasks/submit
{
  "type": "generate",
  "description": "Implement binary search",
  "parameters": {"language": "python"},
  "required_roles": ["coder"],
  "priority": 8
}
```

#### Get Task Status
```http
GET /v1/tasks/{task_id}
```

#### Submit Workflow
```http
POST /v1/tasks/workflow
{
  "tasks": [...]
}
```

### Knowledge Base

#### Store Knowledge
```http
POST /v1/knowledge
{
  "key": "api_design",
  "value": "{...}",
  "tags": ["architecture", "api"]
}
```

#### Retrieve Knowledge
```http
GET /v1/knowledge/{key}
```

#### Query Knowledge
```http
GET /v1/knowledge/query?tags=architecture,api
```

### Messaging

#### Send Message
```http
POST /v1/messages/send
{
  "from_agent_id": "agent-1",
  "to_agent_id": "agent-2",
  "type": "request",
  "subject": "Code review request",
  "payload": {"code": "..."}
}
```

#### Receive Messages
```http
GET /v1/messages/{agent_id}
```

#### Broadcast Message
```http
POST /v1/messages/broadcast
{
  "from_agent_id": "agent-1",
  "subject": "Build completed",
  "payload": {"status": "success"}
}
```

### Consensus

#### Create Vote
```http
POST /v1/consensus/vote/create
{
  "question": "Approve PR #123?",
  "options": ["approve", "reject", "request_changes"],
  "type": "simple_majority"
}
```

#### Cast Vote
```http
POST /v1/consensus/vote/{vote_id}/cast
{
  "agent_id": "agent-1",
  "option": "approve"
}
```

#### Get Vote Result
```http
GET /v1/consensus/vote/{vote_id}
```

## Configuration

The agent collaboration system can be configured via server parameters:

```bash
./llama-server \
  -m model.gguf \
  --n-parallel 10 \          # Max concurrent agents
  --ctx-size 4096            # Context size per agent
```

## Best Practices

1. **Agent Roles**: Define clear roles for each agent (planner, coder, tester, etc.)
2. **Task Dependencies**: Use dependencies to ensure tasks execute in the correct order
3. **Priority**: Set appropriate priorities for time-sensitive tasks
4. **Knowledge Sharing**: Use the knowledge base to share context between agents
5. **Consensus**: Use voting for important decisions that require multiple perspectives
6. **Resource Management**: Monitor agent count and adjust based on available slots

## Troubleshooting

### Issue: Agent not receiving tasks

**Solution**: Check that the agent's capabilities match the task's `required_roles`

### Issue: Tasks stuck in pending

**Solution**: Check task dependencies - ensure prerequisite tasks have completed

### Issue: Out of slots

**Solution**: Increase `--n-parallel` parameter or terminate idle agents

## Performance Tips

1. **Agent Pooling**: Reuse idle agents instead of spawning new ones
2. **Batch Processing**: Group similar tasks together
3. **Caching**: Use knowledge base to cache frequent queries
4. **Lazy Loading**: Spawn agents on-demand based on workload

## Further Reading

- [Architecture Documentation](../../docs/agent-collaboration-system.md)
- [API Reference](../../docs/api-reference.md)
- [Integration Guide](../../docs/integration-guide.md)

## Support

For issues and questions:
- GitHub Issues: https://github.com/ggerganov/llama.cpp/issues
- Documentation: https://github.com/ggerganov/llama.cpp/docs

## License

Same as llama.cpp - MIT License
