# Agent-to-Agent (A2A) Protocol Integration

This document describes the integration of the [Agent-to-Agent (A2A) Protocol](https://a2a-protocol.org) with the hierarchical agents framework.

## Overview

The A2A protocol enables standardized communication and interoperability between AI agents built on different frameworks. This implementation enhances the hierarchical agents system with A2A-compliant message passing, task lifecycle management, and agent discovery capabilities.

## What is A2A?

The Agent-to-Agent (A2A) protocol is an open standard for enabling AI agents to communicate and collaborate effectively. Key features include:

- **Standardized Communication**: JSON-RPC 2.0 over HTTP(S)
- **Agent Discovery**: Via Agent Cards detailing capabilities
- **Flexible Interaction**: Synchronous, streaming, and asynchronous patterns
- **Rich Data Exchange**: Handles text, files, and structured JSON data
- **Opacity Preservation**: Agents collaborate without exposing internal state

## Architecture

The A2A integration consists of three main components:

### 1. A2A Protocol Data Structures (`agents/a2a.py`)

Core A2A protocol types:

- **`A2AMessage`**: Single turn of communication with role and content Parts
- **`Task`**: Stateful unit of work with lifecycle management
- **`Artifact`**: Tangible output generated during task processing
- **`Part`** types: `TextPart`, `FilePart`, `DataPart` for flexible content
- **`AgentCard`**: Self-describing manifest for agent discovery
- **`TaskState`**: Enumeration of task states (pending, in-progress, completed, etc.)

### 2. Enhanced Shared Context (`agents/context.py`)

Extended with A2A support:

- **Task Management**: Create, track, and update A2A tasks
- **Context Grouping**: Group related tasks/messages via `contextId`
- **A2A Message Storage**: Store and retrieve A2A messages by context
- **State Transitions**: Track task lifecycle with full history

### 3. A2A-Enabled Hierarchical Agents (`agents/hierarchical.py`)

New methods for A2A protocol:

- **`create_a2a_task()`**: Create A2A tasks for the agent
- **`send_a2a_message()`**: Send A2A messages to other agents
- **`delegate_a2a_task()`**: Delegate tasks using A2A protocol
- **`get_agent_card()`**: Generate agent cards for discovery
- **`export_agent_card_json()`**: Export agent card as JSON

## Key Features

### 1. Standardized Message Format

Messages follow the A2A protocol specification with:

```python
from llama_cpp.agents import A2AMessage, TextPart, DataPart

# Create a text message
message = A2AMessage.from_text(
    text="Analyze this data",
    role="user",
    contextId="ctx-123"
)

# Create a message with multiple parts
message = A2AMessage(
    role="agent",
    parts=[
        TextPart(text="Here's the analysis:"),
        DataPart(data={"key": "value"})
    ],
    contextId="ctx-123",
    taskId="task-456"
)
```

### 2. Task Lifecycle Management

Tasks progress through well-defined states:

```python
from llama_cpp.agents import TaskState

# Task states
TaskState.PENDING          # Initial state
TaskState.IN_PROGRESS      # Agent processing
TaskState.INPUT_REQUIRED   # Needs user input
TaskState.COMPLETED        # Successfully finished
TaskState.FAILED           # Execution failed
TaskState.CANCELED         # Canceled by user
```

Example task creation and management:

```python
# Create task
task = agent.create_a2a_task(
    user_message="What is A2A protocol?",
    contextId="ctx-abc"
)

# Update task state
context.update_a2a_task(
    task_id=task.id,
    state=TaskState.IN_PROGRESS,
    message="Processing request"
)

# Add artifact to task
artifact = Artifact.from_text(
    text="A2A is an open protocol...",
    name="result.txt"
)
context.update_a2a_task(
    task_id=task.id,
    state=TaskState.COMPLETED,
    artifact=artifact
)
```

### 3. Context Grouping

The `contextId` groups related tasks and messages:

```python
# Create initial task
task1 = agent.create_a2a_task(
    user_message="What is A2A?",
    contextId=None  # Auto-generated
)

# Reuse context for related tasks
shared_context = task1.contextId
task2 = agent.create_a2a_task(
    user_message="How does A2A work?",
    contextId=shared_context
)

# Get all tasks in context
all_tasks = context.get_tasks_by_context(shared_context)
```

### 4. Agent Discovery via Agent Cards

Agents can generate A2A-compliant Agent Cards:

```python
# Generate agent card
card = agent.get_agent_card(
    url="http://localhost:8000/agents/researcher",
    version="1.0.0"
)

# Export as JSON
card_json = agent.export_agent_card_json()

# Card includes:
# - Agent identity and description
# - Skills and capabilities
# - Supported input/output modes
# - Protocol version
```

### 5. Artifacts for Task Results

Artifacts represent tangible outputs:

```python
from llama_cpp.agents import Artifact, TextPart, FilePart

# Text artifact
artifact = Artifact.from_text(
    text="Analysis results...",
    name="analysis.txt",
    description="Data analysis results"
)

# File artifact
artifact = Artifact(
    artifactId="art-123",
    name="chart.png",
    parts=[FilePart.from_bytes(
        data=image_bytes,
        name="chart.png",
        mime_type="image/png"
    )]
)
```

## Usage Examples

### Basic A2A Task

```python
from llama_cpp.agents import (
    HierarchicalAgent, SharedContext, OpenAIAgent,
    TaskState
)

# Create context with A2A enabled
context = SharedContext(enable_a2a=True)

# Create agent
agent = HierarchicalAgent(
    agent_id="researcher",
    provider=OpenAIAgent(model="gpt-4-turbo", api_key="..."),
    context=context,
    metadata={"role": "researcher"}
)

# Create and execute A2A task
task = agent.create_a2a_task(
    user_message="Explain A2A protocol benefits"
)

# Process task
context.update_a2a_task(task.id, state=TaskState.IN_PROGRESS)
response = agent.chat("Explain A2A protocol benefits")

# Complete with artifact
artifact = Artifact.from_text(
    text=response["choices"][0]["message"]["content"],
    name="benefits.txt"
)
context.update_a2a_task(
    task.id,
    state=TaskState.COMPLETED,
    artifact=artifact
)
```

### A2A Message Passing Between Agents

```python
# Create hierarchy
root = HierarchicalAgent("root", provider1, context)
child = root.add_child("child", provider2)

# Send A2A message
message = root.send_a2a_message(
    receiver_id="child",
    message_text="Process this request",
    contextId="ctx-shared"
)

# Get all messages in context
messages = context.get_a2a_messages_by_context("ctx-shared")
```

### A2A Task Delegation

```python
# Delegate task using A2A protocol
result = coordinator.delegate_a2a_task(
    child_id="worker1",
    task_description="Analyze data",
    contextId="ctx-analysis"
)

# Result includes:
# - task: Complete task object with state
# - artifact: Generated artifact
# - response: Raw LLM response
```

## Backward Compatibility

The A2A integration maintains full backward compatibility:

- **Legacy message passing** still works via `send_message()`, `get_messages()`
- **Legacy task delegation** via `delegate_to_child()` continues to function
- **A2A is optional**: If not enabled, hierarchical agents work as before
- **Graceful fallback**: A2A methods fall back to legacy mode if disabled

## Benefits of A2A Integration

### 1. Interoperability
- Agents can communicate using a standard protocol
- Easy integration with external A2A-compliant systems
- Protocol-defined message formats prevent ambiguity

### 2. Task Lifecycle Tracking
- Explicit task states and transitions
- Full history of state changes
- Clear success/failure indicators

### 3. Rich Content Exchange
- Support for text, files, and structured data via Parts
- Artifact mechanism for deliverables
- Metadata extensibility

### 4. Agent Discovery
- Agent Cards enable dynamic capability discovery
- Clients can query agent skills before interaction
- Standardized skill descriptions

### 5. Context Management
- `contextId` groups related work
- Easy to track multi-turn conversations
- Support for parallel task execution

## Critical Improvements Over Previous Implementation

### Before A2A:
- ❌ Ad-hoc message formats with arbitrary `message_type`
- ❌ No formal task state management
- ❌ Simple in-memory messages, no persistence model
- ❌ No standard for agent capability discovery
- ❌ Limited to text-only content
- ❌ No formal context grouping

### After A2A:
- ✅ Standardized message format following A2A spec
- ✅ Formal task lifecycle with state machines
- ✅ Rich content support (text, files, structured data)
- ✅ Agent Cards for capability discovery
- ✅ Context grouping via `contextId`
- ✅ Artifacts for tangible outputs
- ✅ State transition history tracking
- ✅ Backward compatible with legacy code

## Running Examples

The `examples/a2a_hierarchical_agents.py` file demonstrates all features:

```bash
# Set API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"  # Optional

# Run examples
python llama-py/examples/a2a_hierarchical_agents.py
```

Examples include:
1. A2A Task Lifecycle Management
2. A2A Message Passing Between Agents
3. A2A Task Delegation to Children
4. Agent Card Generation for Discovery
5. Context Grouping with contextId

## API Reference

### A2A Protocol Types

#### `A2AMessage`
```python
A2AMessage(
    role: Literal["user", "agent"],
    parts: List[Part],
    messageId: str = auto-generated,
    contextId: Optional[str] = None,
    taskId: Optional[str] = None,
    referenceTaskIds: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
)
```

#### `Task`
```python
Task(
    id: str,
    contextId: str,
    status: TaskStatus,
    history: Optional[List[A2AMessage]] = None,
    artifacts: Optional[List[Artifact]] = None,
    metadata: Optional[Dict[str, Any]] = None
)
```

#### `Artifact`
```python
Artifact(
    artifactId: str,
    parts: List[Part],
    name: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
)
```

### HierarchicalAgent A2A Methods

#### `create_a2a_task()`
Create a new A2A task for this agent.

#### `send_a2a_message()`
Send an A2A protocol message to another agent.

#### `delegate_a2a_task()`
Delegate a task to a child using A2A protocol.

#### `get_agent_card()`
Generate an A2A Agent Card for this agent.

#### `export_agent_card_json()`
Export agent card as JSON string.

### SharedContext A2A Methods

#### `create_a2a_task()`
Create a new A2A task with optional initial message.

#### `get_a2a_task()`
Retrieve a task by ID.

#### `update_a2a_task()`
Update task state or add artifacts.

#### `get_tasks_by_context()`
Get all tasks for a given contextId.

#### `get_a2a_messages_by_context()`
Get all A2A messages for a given contextId.

#### `send_a2a_message()`
Send an A2A message from one agent to another.

## References

- **A2A Protocol**: https://a2a-protocol.org
- **A2A Specification**: https://github.com/a2aproject/A2A
- **A2A Python SDK**: https://github.com/a2aproject/a2a-python

## Future Enhancements

Potential improvements for future versions:

1. **HTTP Server**: Add A2A JSON-RPC server for remote access
2. **Streaming Support**: Implement SSE for real-time updates
3. **Push Notifications**: Add webhook support for long-running tasks
4. **Task Persistence**: Store tasks in database for durability
5. **Authentication**: Implement security schemes from Agent Cards
6. **Skills System**: Expand skill definitions and matching
7. **Extensions**: Support custom A2A protocol extensions

## Conclusion

The A2A protocol integration significantly enhances the hierarchical agents framework by:

1. **Standardizing** inter-agent communication
2. **Formalizing** task lifecycle management
3. **Enabling** agent discovery via cards
4. **Supporting** rich content exchange
5. **Maintaining** backward compatibility

This makes the hierarchical agents framework ready for integration with the broader A2A ecosystem while preserving all existing functionality.
