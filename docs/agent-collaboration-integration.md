# Agent Collaboration System - Integration Guide

This guide explains how to integrate the Agent Collaboration System into your llama.cpp server deployment.

## Overview

The Agent Collaboration System is **automatically included** in llama-server builds. No additional compilation flags or dependencies are required beyond the standard llama.cpp build requirements.

## Prerequisites

- llama.cpp built with server support (`LLAMA_BUILD_SERVER=ON`, default)
- HTTP library support enabled (`LLAMA_HTTPLIB=ON`, default)
- C++17 compatible compiler
- Sufficient slots for multiple concurrent agents (recommended: 4-10)

## Building

### Standard Build

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release --target llama-server
```

### Custom Build Options

```bash
cmake .. \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_HTTPLIB=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release --target llama-server
```

## Server Configuration

### Basic Configuration

Start the server with sufficient parallel slots for agents:

```bash
./llama-server \
  -m /path/to/model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --n-parallel 10 \
  --ctx-size 4096
```

**Key Parameters:**
- `--n-parallel`: Maximum concurrent agents (recommended: 4-10)
- `--ctx-size`: Context size per agent slot
- `--host`: Bind address (0.0.0.0 for all interfaces)
- `--port`: HTTP server port

### Production Configuration

For production deployments:

```bash
./llama-server \
  -m /path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --n-parallel 10 \
  --ctx-size 8192 \
  --n-gpu-layers 35 \
  --threads 8 \
  --log-file /var/log/llama-server.log
```

**Additional Parameters:**
- `--n-gpu-layers`: GPU acceleration for faster inference
- `--threads`: CPU threads for computation
- `--log-file`: Enable logging for debugging

## API Integration

### Python Client

```python
import requests

class AgentClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url

    def spawn_agent(self, role, slot_id, capabilities):
        response = requests.post(
            f"{self.base_url}/v1/agents/spawn",
            json={
                "role": role,
                "slot_id": slot_id,
                "capabilities": capabilities
            }
        )
        return response.json()["agent_id"]

    def submit_task(self, task_type, description, required_roles=None):
        response = requests.post(
            f"{self.base_url}/v1/tasks/submit",
            json={
                "type": task_type,
                "description": description,
                "required_roles": required_roles or [],
                "priority": 5
            }
        )
        return response.json()["task_id"]

# Usage
client = AgentClient()
agent_id = client.spawn_agent("coder", 0, ["python", "javascript"])
task_id = client.submit_task("generate", "Implement binary search", ["coder"])
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

class AgentClient {
    constructor(baseUrl = 'http://localhost:8080') {
        this.baseUrl = baseUrl;
    }

    async spawnAgent(role, slotId, capabilities) {
        const response = await axios.post(`${this.baseUrl}/v1/agents/spawn`, {
            role: role,
            slot_id: slotId,
            capabilities: capabilities
        });
        return response.data.agent_id;
    }

    async submitTask(taskType, description, requiredRoles = []) {
        const response = await axios.post(`${this.baseUrl}/v1/tasks/submit`, {
            type: taskType,
            description: description,
            required_roles: requiredRoles,
            priority: 5
        });
        return response.data.task_id;
    }
}

// Usage
const client = new AgentClient();
const agentId = await client.spawnAgent('coder', 0, ['python', 'javascript']);
const taskId = await client.submitTask('generate', 'Implement binary search', ['coder']);
```

### cURL Examples

```bash
# Spawn an agent
curl -X POST http://localhost:8080/v1/agents/spawn \
  -H "Content-Type: application/json" \
  -d '{
    "role": "coder",
    "slot_id": 0,
    "capabilities": ["python", "javascript"]
  }'

# Submit a task
curl -X POST http://localhost:8080/v1/tasks/submit \
  -H "Content-Type: application/json" \
  -d '{
    "type": "generate",
    "description": "Implement binary search",
    "required_roles": ["coder"],
    "priority": 8
  }'

# Check task status
curl http://localhost:8080/v1/tasks/{task_id}

# List all agents
curl http://localhost:8080/v1/agents

# Get system stats
curl http://localhost:8080/v1/agents/stats
```

## Architecture Integration

### Microservices Integration

```
┌─────────────────┐
│   Your App      │
│   (Client)      │
└────────┬────────┘
         │ HTTP REST
         ▼
┌─────────────────┐
│ llama-server    │
│ + Agent Collab  │
└────────┬────────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
  [Agent][Agent][Agent]
```

### Docker Deployment

```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget

# Clone and build llama.cpp
WORKDIR /app
COPY . /app
RUN mkdir build && cd build && \
    cmake .. && \
    cmake --build . --config Release --target llama-server

# Expose port
EXPOSE 8080

# Run server
CMD ["./build/bin/llama-server", \
     "-m", "/models/model.gguf", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--n-parallel", "10"]
```

Build and run:

```bash
docker build -t llama-agent-server .
docker run -p 8080:8080 -v /path/to/models:/models llama-agent-server
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-agent-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llama-agent-server
  template:
    metadata:
      labels:
        app: llama-agent-server
    spec:
      containers:
      - name: server
        image: llama-agent-server:latest
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_PATH
          value: "/models/model.gguf"
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: llama-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: llama-agent-service
spec:
  selector:
    app: llama-agent-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

## Enabling Agent Collaboration (Code-Level)

If you're modifying the server code, here's how to enable agent collaboration:

### 1. Include Headers

In `server.cpp`, add:

```cpp
#include "agent-collab.h"
#include "agent-collab-routes.h"
```

### 2. Initialize Orchestrator

Add to your server initialization:

```cpp
// Create agent orchestrator
agent_collab::agent_orchestrator orchestrator;
orchestrator.start();
```

### 3. Register Routes

Add routes to your HTTP server:

```cpp
// Create and register agent routes
agent_collab::agent_routes agent_api(orchestrator);
agent_api.register_routes(svr);
```

### 4. Cleanup on Shutdown

Add to cleanup code:

```cpp
orchestrator.stop();
```

## Configuration Options

### Environment Variables

```bash
# Server configuration
export LLAMA_SERVER_HOST="0.0.0.0"
export LLAMA_SERVER_PORT="8080"
export LLAMA_MAX_PARALLEL="10"

# Agent configuration
export AGENT_MAX_COUNT="10"
export AGENT_TIMEOUT_MS="300000"

# Knowledge base
export KB_MAX_ENTRIES="10000"
export KB_PERSISTENCE="true"
export KB_STORAGE_PATH="./agent_kb.json"
```

### Configuration File

Create `agent-config.json`:

```json
{
  "agent_collaboration": {
    "max_agents": 10,
    "default_agent_timeout": 300000,
    "knowledge_base": {
      "max_entries": 10000,
      "persistence": true,
      "storage_path": "./agent_kb.json"
    },
    "scheduler": {
      "algorithm": "priority_queue",
      "max_queue_size": 1000
    },
    "consensus": {
      "default_type": "simple_majority",
      "voting_timeout": 60000
    },
    "communication": {
      "message_retention": 86400000,
      "max_message_size": 1048576
    }
  }
}
```

## Monitoring and Observability

### Health Check

```bash
curl http://localhost:8080/health
```

### System Stats

```bash
curl http://localhost:8080/v1/agents/stats
```

Response:

```json
{
  "agents": {
    "total": 5,
    "idle": 2,
    "busy": 3
  },
  "tasks": {
    "total": 20,
    "pending": 5,
    "completed": 14,
    "failed": 1
  },
  "knowledge_base": {
    "entries": 42
  }
}
```

### Prometheus Metrics

Expose metrics endpoint:

```bash
curl http://localhost:8080/metrics
```

### Logging

Enable detailed logging:

```bash
./llama-server \
  -m model.gguf \
  --log-file /var/log/llama-server.log \
  --log-level debug
```

## Security Considerations

### 1. Authentication

Add API key authentication:

```bash
# Set API key
export LLAMA_API_KEY="your-secret-key"

# Use in requests
curl -H "Authorization: Bearer your-secret-key" \
  http://localhost:8080/v1/agents
```

### 2. Network Security

- Bind to localhost only in production: `--host 127.0.0.1`
- Use reverse proxy (nginx, Apache) for SSL/TLS
- Implement rate limiting
- Use firewall rules to restrict access

### 3. Resource Limits

- Set maximum parallel agents
- Implement task timeouts
- Monitor memory usage
- Set context size limits

## Troubleshooting

### Issue: Server Won't Start

**Symptoms**: Server fails to start with agent collaboration

**Solutions**:
1. Check build includes agent-collab files in CMakeLists.txt
2. Verify C++17 support
3. Check for missing dependencies (cpp-httplib)
4. Review build logs for compilation errors

### Issue: Agents Not Receiving Tasks

**Symptoms**: Tasks remain in pending state

**Solutions**:
1. Verify agent capabilities match task `required_roles`
2. Check agent state (should be `idle`)
3. Ensure sufficient slots available
4. Check task dependencies are met

### Issue: Out of Memory

**Symptoms**: Server crashes or becomes unresponsive

**Solutions**:
1. Reduce `--n-parallel` parameter
2. Decrease `--ctx-size`
3. Terminate idle agents
4. Monitor system resources

### Issue: Slow Response Times

**Symptoms**: API requests timeout or take too long

**Solutions**:
1. Enable GPU acceleration (`--n-gpu-layers`)
2. Increase thread count (`--threads`)
3. Use quantized models for faster inference
4. Scale horizontally with multiple server instances

## Performance Tuning

### Optimal Configuration for Different Workloads

**Lightweight (4GB RAM, 4 CPU cores)**:
```bash
./llama-server -m model-q4.gguf \
  --n-parallel 3 \
  --ctx-size 2048 \
  --threads 4
```

**Medium (16GB RAM, 8 CPU cores)**:
```bash
./llama-server -m model-q5.gguf \
  --n-parallel 6 \
  --ctx-size 4096 \
  --threads 8 \
  --n-gpu-layers 20
```

**Heavy (32GB+ RAM, GPU)**:
```bash
./llama-server -m model-f16.gguf \
  --n-parallel 10 \
  --ctx-size 8192 \
  --threads 16 \
  --n-gpu-layers 50
```

## Migration Guide

### From Standard llama-server

The agent collaboration system is **backward compatible**. Existing llama-server deployments will continue to work without changes. To enable agent features:

1. Update to latest version with agent collaboration
2. No configuration changes required
3. Start using agent API endpoints
4. Existing endpoints remain unchanged

### Gradual Adoption

```python
# Phase 1: Standard inference
response = requests.post(f"{server}/v1/completions", json={...})

# Phase 2: Spawn single agent
agent_id = spawn_agent("assistant", 0, [])

# Phase 3: Multi-agent workflows
workflow_id = submit_workflow([...])

# Phase 4: Full collaboration features
vote_id = create_consensus_vote(...)
```

## Support

- **Documentation**: `/docs/agent-collaboration-system.md`
- **Examples**: `/examples/agent-collaboration/`
- **Issues**: https://github.com/ggerganov/llama.cpp/issues
- **Discord**: llama.cpp community

## License

Same as llama.cpp - MIT License
