# Agent Collaboration System - Quick Start Guide

Get started with the Agent Collaboration System in 5 minutes!

## Step 1: Build llama-server (1 minute)

```bash
cd llama.cpp
mkdir build && cd build
cmake .. && cmake --build . --config Release --target llama-server
```

## Step 2: Start the Server (30 seconds)

```bash
./build/bin/llama-server \
  -m /path/to/your/model.gguf \
  --n-parallel 5
```

The server will start on `http://localhost:8080`

## Step 3: Spawn Your First Agent (30 seconds)

```bash
curl -X POST http://localhost:8080/v1/agents/spawn \
  -H "Content-Type: application/json" \
  -d '{
    "role": "assistant",
    "slot_id": 0,
    "capabilities": ["coding", "analysis"]
  }'
```

You'll get a response like:

```json
{
  "agent_id": "agent-abc123",
  "role": "assistant",
  "slot_id": 0,
  "status": "spawned"
}
```

## Step 4: Submit a Task (1 minute)

```bash
curl -X POST http://localhost:8080/v1/tasks/submit \
  -H "Content-Type: application/json" \
  -d '{
    "type": "generate",
    "description": "Write a Python function to calculate factorial",
    "required_roles": ["assistant"],
    "priority": 8
  }'
```

Response:

```json
{
  "task_id": "task-xyz789",
  "status": "submitted"
}
```

## Step 5: Check Task Status (30 seconds)

```bash
curl http://localhost:8080/v1/tasks/task-xyz789
```

Response:

```json
{
  "task_id": "task-xyz789",
  "status": "completed",
  "result": {
    "success": true,
    "output": "def factorial(n): ..."
  }
}
```

## What's Next?

### Try Multi-Agent Collaboration

Spawn multiple agents with different roles:

```bash
# Planner agent
curl -X POST http://localhost:8080/v1/agents/spawn \
  -d '{"role": "planner", "slot_id": 0, "capabilities": ["planning"]}'

# Coder agent
curl -X POST http://localhost:8080/v1/agents/spawn \
  -d '{"role": "coder", "slot_id": 1, "capabilities": ["python"]}'

# Tester agent
curl -X POST http://localhost:8080/v1/agents/spawn \
  -d '{"role": "tester", "slot_id": 2, "capabilities": ["testing"]}'
```

### Create a Workflow

```bash
curl -X POST http://localhost:8080/v1/tasks/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [
      {
        "id": "plan",
        "type": "analyze",
        "description": "Plan the implementation",
        "required_roles": ["planner"]
      },
      {
        "id": "code",
        "type": "generate",
        "description": "Implement the code",
        "dependencies": ["plan"],
        "required_roles": ["coder"]
      },
      {
        "id": "test",
        "type": "test",
        "description": "Test the implementation",
        "dependencies": ["code"],
        "required_roles": ["tester"]
      }
    ]
  }'
```

### Use Shared Knowledge

```bash
# Store knowledge
curl -X POST http://localhost:8080/v1/knowledge \
  -d '{
    "key": "project_architecture",
    "value": "Microservices with REST API",
    "tags": ["architecture", "design"]
  }'

# Retrieve knowledge
curl http://localhost:8080/v1/knowledge/project_architecture
```

### Create Consensus Votes

```bash
# Create vote
curl -X POST http://localhost:8080/v1/consensus/vote/create \
  -d '{
    "question": "Should we use async/await?",
    "options": ["yes", "no"],
    "type": "simple_majority"
  }'

# Cast votes
curl -X POST http://localhost:8080/v1/consensus/vote/{vote_id}/cast \
  -d '{"agent_id": "agent-1", "option": "yes"}'
```

## Python Example

```python
import requests

# Configuration
SERVER = "http://localhost:8080"

# Spawn agent
agent = requests.post(f"{SERVER}/v1/agents/spawn", json={
    "role": "assistant",
    "slot_id": 0,
    "capabilities": ["coding"]
}).json()

print(f"Agent ID: {agent['agent_id']}")

# Submit task
task = requests.post(f"{SERVER}/v1/tasks/submit", json={
    "type": "generate",
    "description": "Write hello world in Python",
    "required_roles": ["assistant"]
}).json()

print(f"Task ID: {task['task_id']}")

# Check status
status = requests.get(f"{SERVER}/v1/tasks/{task['task_id']}").json()
print(f"Status: {status['status']}")
```

## Useful Commands

```bash
# List all agents
curl http://localhost:8080/v1/agents

# List all tasks
curl http://localhost:8080/v1/tasks

# Get system stats
curl http://localhost:8080/v1/agents/stats

# Terminate agent
curl -X DELETE http://localhost:8080/v1/agents/{agent_id}

# Cancel task
curl -X DELETE http://localhost:8080/v1/tasks/{task_id}
```

## Common Patterns

### Pattern 1: Single Agent Task

```bash
# 1. Spawn agent
AGENT_ID=$(curl -s -X POST http://localhost:8080/v1/agents/spawn \
  -d '{"role": "assistant", "slot_id": 0}' | jq -r '.agent_id')

# 2. Submit task
TASK_ID=$(curl -s -X POST http://localhost:8080/v1/tasks/submit \
  -d "{\"description\": \"Your task\", \"required_roles\": [\"assistant\"]}" \
  | jq -r '.task_id')

# 3. Wait and check result
sleep 5
curl http://localhost:8080/v1/tasks/$TASK_ID
```

### Pattern 2: Parallel Tasks

```bash
# Submit multiple independent tasks
for i in {1..5}; do
  curl -X POST http://localhost:8080/v1/tasks/submit \
    -d "{\"description\": \"Task $i\"}" &
done
wait
```

### Pattern 3: Sequential Workflow

```bash
# Tasks execute in order due to dependencies
curl -X POST http://localhost:8080/v1/tasks/workflow \
  -d '{
    "tasks": [
      {"id": "step1", "description": "First"},
      {"id": "step2", "description": "Second", "dependencies": ["step1"]},
      {"id": "step3", "description": "Third", "dependencies": ["step2"]}
    ]
  }'
```

## Troubleshooting

### Server won't start
- Check model path is correct
- Ensure port 8080 is available
- Verify sufficient memory

### Agent not processing tasks
- Check agent is in `idle` state
- Verify task `required_roles` matches agent role
- Ensure slot is available

### Task stuck in pending
- Check task dependencies are completed
- Verify at least one agent has matching role
- Check system stats for bottlenecks

## Next Steps

- Read the [Full Documentation](agent-collaboration-system.md)
- Try [Example Scripts](../examples/agent-collaboration/)
- Explore [Integration Guide](agent-collaboration-integration.md)
- Check [API Reference](api-reference.md)

## Get Help

- **Issues**: https://github.com/ggerganov/llama.cpp/issues
- **Discord**: llama.cpp community
- **Docs**: https://github.com/ggerganov/llama.cpp/docs

---

**Ready to build something amazing with multi-agent AI? Let's go! 🚀**
