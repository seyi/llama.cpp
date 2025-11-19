# Hierarchical Agent Trees

This document explains how to use the hierarchical agent tree system in llama-py.

## Overview

Hierarchical agents allow you to organize multiple AI agents in a tree structure with:

- **Parent-child relationships**: Agents can have children, forming a tree
- **Shared context**: State and data shared across the tree with scoping
- **Message passing**: Agents can send messages to each other
- **Task delegation**: Parents can delegate work to children
- **Result aggregation**: Collect and synthesize results from multiple agents
- **Multi-provider**: Use different AI providers at different levels

## Architecture

```
                    [Root Agent]
                     (GPT-4)
                         |
            ┌────────────┼────────────┐
            ▼            ▼            ▼
      [Researcher]  [Analyst]     [Writer]
       (Claude)      (Gemini)      (Local)
            |
         ┌──┴──┐
         ▼     ▼
    [WebSearch] [DocSearch]
     (GPT-3.5)   (Claude)
```

## Key Components

### 1. SharedContext

Manages shared state and communication across the agent tree.

```python
from llama_cpp.agents import SharedContext, ContextScope

context = SharedContext()

# Register agents (done automatically by HierarchicalAgent)
context.register_agent("agent1", parent_id=None)
context.register_agent("agent2", parent_id="agent1")

# Set/get values
context.set("key", "value", agent_id="agent1", scope=ContextScope.GLOBAL)
value = context.get("key", agent_id="agent2")

# Send messages
context.send_message(
    sender_id="agent1",
    receiver_id="agent2",
    message_type="task",
    content="Please analyze this data"
)

# Get messages
messages = context.get_messages(agent_id="agent2")
```

### 2. ContextScope

Defines visibility of context entries:

- **LOCAL**: Only the owning agent can see this
- **CHILDREN**: Owner and direct children can see this
- **SUBTREE**: Owner and all descendants can see this
- **GLOBAL**: All agents in the tree can see this

```python
from llama_cpp.agents import ContextScope

# Only this agent sees it
agent.set_context("private_data", value, scope=ContextScope.LOCAL)

# This agent and its direct children see it
agent.set_context("team_data", value, scope=ContextScope.CHILDREN)

# Entire subtree sees it
agent.set_context("project_data", value, scope=ContextScope.SUBTREE)

# All agents see it
agent.set_context("global_config", value, scope=ContextScope.GLOBAL)
```

### 3. HierarchicalAgent

Main class for creating agent hierarchies.

```python
from llama_cpp.agents import HierarchicalAgent, SharedContext, OpenAIAgent

# Create shared context
context = SharedContext()

# Create root agent
root = HierarchicalAgent(
    agent_id="coordinator",
    provider=OpenAIAgent(model="gpt-4-turbo", api_key="..."),
    context=context
)

# Add child agents
researcher = root.add_child(
    agent_id="researcher",
    provider=AnthropicAgent(model="claude-3-opus", api_key="...")
)

# Add grandchild
web_searcher = researcher.add_child(
    agent_id="web_searcher",
    provider=GoogleAgent(model="gemini-1.5-pro", api_key="...")
)
```

## Core Functionality

### Context Sharing

```python
# Set context at root level
root.set_context("project", "AI Research", scope=ContextScope.GLOBAL)
root.set_context("deadline", "2025-12-31", scope=ContextScope.SUBTREE)

# Children can access parent's context
project = researcher.get_context("project")  # "AI Research"
deadline = researcher.get_context("deadline")  # "2025-12-31"

# Get all accessible context
all_context = researcher.get_all_context()
# Returns: {"project": "AI Research", "deadline": "2025-12-31", ...}
```

### Message Passing

```python
# Send to specific agent
root.send_message(
    receiver_id="researcher",
    message_type="task_assignment",
    content="Research hierarchical agents"
)

# Broadcast to all children
root.broadcast_to_children(
    message_type="update",
    content="New phase starting"
)

# Broadcast to entire subtree
root.broadcast_to_subtree(
    message_type="announcement",
    content="Project milestone reached"
)

# Send to parent
researcher.send_to_parent(
    message_type="status",
    content="Research completed"
)

# Retrieve messages
messages = researcher.get_messages(message_type="task_assignment")
for msg in messages:
    print(f"From {msg.sender_id}: {msg.content}")
```

### Task Delegation

```python
# Delegate to specific child
result = root.delegate_to_child(
    child_id="researcher",
    task_description="Find papers on agent architectures",
    tools=[search_tool],
    max_tokens=500
)

print(result["choices"][0]["message"]["content"])
```

### Result Aggregation

```python
# Delegate same task to all children and aggregate results
aggregated = root.aggregate_from_children(
    task_description="What are the benefits of hierarchical agents?",
    aggregation_prompt="Synthesize the responses into a coherent summary",
    max_tokens=300
)

print(aggregated["choices"][0]["message"]["content"])
```

## Complete Example: Research Team

```python
from llama_cpp.agents import (
    HierarchicalAgent,
    SharedContext,
    ContextScope,
    Tool,
    OpenAIAgent,
    AnthropicAgent,
    GoogleAgent,
)
import os

# Create shared context
context = SharedContext()

# Define tools
search_tool = Tool(
    name="web_search",
    description="Search the web",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
)

analyze_tool = Tool(
    name="analyze",
    description="Analyze data",
    parameters={
        "type": "object",
        "properties": {
            "data": {"type": "string"},
            "focus": {"type": "string"}
        },
        "required": ["data"]
    }
)

# Create coordinator (GPT-4)
coordinator = HierarchicalAgent(
    agent_id="coordinator",
    provider=OpenAIAgent(
        model="gpt-4-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    ),
    context=context,
    metadata={"role": "team_lead"}
)

# Create researcher (Claude)
researcher = coordinator.add_child(
    agent_id="researcher",
    provider=AnthropicAgent(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    ),
    metadata={"role": "researcher"}
)

# Create analyst (Gemini)
analyst = coordinator.add_child(
    agent_id="analyst",
    provider=GoogleAgent(
        model="gemini-1.5-pro",
        api_key=os.getenv("GOOGLE_API_KEY")
    ),
    metadata={"role": "analyst"}
)

# Set shared project context
coordinator.set_context(
    "project_topic",
    "Hierarchical Multi-Agent Systems",
    scope=ContextScope.GLOBAL
)

coordinator.set_context(
    "project_deadline",
    "2025-01-31",
    scope=ContextScope.SUBTREE
)

# View tree structure
print("Team Structure:")
print(coordinator.get_tree_view())

# Delegate research task
print("\nDelegating research to researcher...")
research_result = coordinator.delegate_to_child(
    child_id="researcher",
    task_description="Find 3 key benefits of hierarchical agent systems",
    tools=[search_tool],
    max_tokens=300
)

# Delegate analysis task
print("Delegating analysis to analyst...")
analysis_result = coordinator.delegate_to_child(
    child_id="analyst",
    task_description="Analyze the trade-offs of hierarchical vs flat agent architectures",
    tools=[analyze_tool],
    max_tokens=300
)

# Aggregate results
print("\nAggregating results from team...")
final_report = coordinator.aggregate_from_children(
    task_description="Provide one insight about hierarchical agents",
    aggregation_prompt="""
    Synthesize the research and analysis into a concise executive summary.
    Focus on the most important findings.
    """,
    max_tokens=400
)

print("\nFinal Report:")
print(final_report["choices"][0]["message"]["content"])

# Get tree statistics
stats = context.get_tree_stats()
print(f"\nTree Statistics:")
print(f"  Total agents: {stats['total_agents']}")
print(f"  Context entries: {stats['total_context_entries']}")
print(f"  Messages: {stats['total_messages']}")
```

## Use Cases

### 1. Research Team

Organize specialists (researchers, analysts, writers) to collaborate on complex research:

```
[Research Coordinator] (GPT-4)
├─ [Literature Researcher] (Claude Opus)
├─ [Data Analyst] (Gemini Pro)
└─ [Technical Writer] (GPT-4)
```

### 2. Software Development Team

Coordinate different aspects of software development:

```
[Tech Lead] (GPT-4)
├─ [Backend Dev] (Claude)
│  ├─ [API Designer]
│  └─ [Database Expert]
├─ [Frontend Dev] (GPT-4)
│  ├─ [UI Designer]
│  └─ [Accessibility Expert]
└─ [QA Engineer] (Gemini)
```

### 3. Customer Support System

Multi-tier support with escalation:

```
[Support Manager] (GPT-4)
├─ [Tier 1 Support] (GPT-3.5)
│  ├─ [FAQ Bot]
│  └─ [Account Helper]
├─ [Tier 2 Support] (Claude)
└─ [Specialist Support] (GPT-4)
```

### 4. Content Creation Pipeline

Coordinate content research, creation, and editing:

```
[Content Director] (GPT-4)
├─ [Researcher] (Claude)
│  ├─ [Fact Checker]
│  └─ [Source Finder]
├─ [Writer] (GPT-4)
└─ [Editor] (Claude Opus)
```

## Best Practices

### 1. Choose Appropriate Scope

```python
# Project-wide config → GLOBAL
root.set_context("api_endpoint", "https://api.example.com", ContextScope.GLOBAL)

# Team-specific data → SUBTREE
team_lead.set_context("team_goal", "Complete feature X", ContextScope.SUBTREE)

# Direct reports only → CHILDREN
manager.set_context("team_meeting", "Monday 10am", ContextScope.CHILDREN)

# Private notes → LOCAL
agent.set_context("internal_notes", "Remember to...", ContextScope.LOCAL)
```

### 2. Structure for Specialization

Place specialized agents as children where their expertise is needed:

```python
# Good: Specialists under relevant parent
research_team = root.add_child("research", OpenAIAgent(...))
research_team.add_child("ml_expert", AnthropicAgent(...))
research_team.add_child("nlp_expert", GoogleAgent(...))

# Avoid: Flat structure loses organization
root.add_child("ml_expert", ...)
root.add_child("nlp_expert", ...)
root.add_child("cv_expert", ...)
```

### 3. Use Different Providers Strategically

```python
# Expensive, powerful model at top for coordination
coordinator = HierarchicalAgent(
    "coordinator",
    OpenAIAgent(model="gpt-4-turbo", ...),  # $$$
    context
)

# Cheaper models for simpler tasks
worker1 = coordinator.add_child(
    "worker1",
    OpenAIAgent(model="gpt-3.5-turbo", ...)  # $
)

# Specialized models for specific tasks
analyst = coordinator.add_child(
    "analyst",
    AnthropicAgent(model="claude-3-opus", ...)  # Best for analysis
)
```

### 4. Message Patterns

```python
# Command pattern: Parent → Child
parent.send_message("child1", "command", "Start task X")

# Status pattern: Child → Parent
child.send_to_parent("status", "Task completed")

# Broadcast pattern: Root → All
root.broadcast_to_subtree("announcement", "System update")

# Collaboration pattern: Peer → Peer
agent1.send_message("agent2", "collaboration", "Need your input on X")
```

## API Reference

### HierarchicalAgent

```python
class HierarchicalAgent:
    def __init__(
        agent_id: str,
        provider: BaseAgent,
        context: SharedContext,
        parent: Optional[HierarchicalAgent] = None,
        metadata: Optional[Dict] = None
    )

    # Tree management
    def add_child(agent_id, provider, metadata=None) -> HierarchicalAgent
    def remove_child(agent_id: str)

    # Context management
    def set_context(key, value, scope=ContextScope.SUBTREE, metadata=None)
    def get_context(key, default=None) -> Any
    def get_all_context(scope_filter=None) -> Dict
    def delete_context(key) -> bool

    # Message passing
    def send_message(receiver_id, message_type, content, metadata=None)
    def get_messages(message_type=None, clear=True) -> List[AgentMessage]
    def broadcast_to_children(message_type, content, metadata=None)
    def broadcast_to_subtree(message_type, content, metadata=None)
    def send_to_parent(message_type, content, metadata=None)

    # Task delegation
    def delegate_to_child(child_id, task_description, tools=None, **kwargs) -> Dict
    def aggregate_from_children(task_description, aggregation_prompt, tools=None, **kwargs) -> Dict

    # Chat (delegates to provider)
    def chat(message=None, tools=None, tool_choice="auto", **kwargs) -> Dict

    # Utilities
    def get_tree_view(indent=0) -> str
```

### SharedContext

```python
class SharedContext:
    def register_agent(agent_id, parent_id=None, metadata=None)
    def unregister_agent(agent_id)

    def set(key, value, agent_id, scope=ContextScope.SUBTREE, metadata=None)
    def get(key, agent_id, default=None) -> Any
    def get_all(agent_id, scope_filter=None) -> Dict
    def delete(key, agent_id) -> bool

    def send_message(sender_id, receiver_id, message_type, content, metadata=None)
    def get_messages(agent_id, message_type=None, clear=True) -> List

    def get_children(agent_id) -> Set[str]
    def get_descendants(agent_id) -> Set[str]
    def get_parent(agent_id) -> Optional[str]
    def get_ancestors(agent_id) -> List[str]

    def get_tree_stats() -> Dict
    def export_context(agent_id) -> Dict
    def import_context(agent_id, context_data)
```

## See Also

- [Multi-Provider Agents](examples/multi_provider_agents.py) - Basic multi-provider usage
- [Hierarchical Agent Example](examples/hierarchical_agent_tree.py) - Complete examples
- [Agent Function Calling](examples/agent_function_calling.py) - Tool use patterns
