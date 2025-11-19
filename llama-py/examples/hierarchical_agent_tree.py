#!/usr/bin/env python3
"""
Hierarchical Agent Tree Example

This example demonstrates how to organize agents in a tree structure with:
- Parent-child relationships
- Shared context with scoping
- Inter-agent message passing
- Task delegation and result aggregation
- Different AI providers at different levels

Tree Structure:
                    [Coordinator]
                     (GPT-4)
                         |
            ┌────────────┼────────────┐
            ▼            ▼            ▼
      [Researcher]  [Analyst]     [Writer]
       (Claude)      (Gemini)      (GPT-4)
            |
         ┌──┴──┐
         ▼     ▼
    [WebSearch] [PaperSearch]
     (Local)     (Claude)
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp.agents import (
    HierarchicalAgent,
    SharedContext,
    ContextScope,
    Tool,
    OpenAIAgent,
    AnthropicAgent,
    GoogleAgent,
    LlamaAgent,
)


def create_search_tool() -> Tool:
    """Create a web search tool"""
    return Tool(
        name="web_search",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    )


def create_analyze_tool() -> Tool:
    """Create an analysis tool"""
    return Tool(
        name="analyze_data",
        description="Analyze data and extract insights",
        parameters={
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Data to analyze"
                },
                "focus": {
                    "type": "string",
                    "description": "What to focus on in the analysis"
                }
            },
            "required": ["data"]
        }
    )


def example_basic_tree():
    """Example 1: Basic tree structure with context sharing"""
    print("\n" + "="*70)
    print("Example 1: Basic Tree Structure with Context Sharing")
    print("="*70)

    # Create shared context
    context = SharedContext()

    # Check if we have API keys (use mock agents if not)
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key:
        print("⚠️  OPENAI_API_KEY not set - skipping this example")
        return

    # Create root agent (Coordinator)
    print("\n1. Creating coordinator agent (GPT-4)...")
    coordinator = HierarchicalAgent(
        agent_id="coordinator",
        provider=OpenAIAgent(model="gpt-4-turbo", api_key=openai_key),
        context=context,
        metadata={"role": "coordinator", "level": 0}
    )

    # Create child agents
    print("2. Creating child agents...")

    if anthropic_key:
        researcher = coordinator.add_child(
            agent_id="researcher",
            provider=AnthropicAgent(
                model="claude-3-5-sonnet-20241022",
                api_key=anthropic_key
            ),
            metadata={"role": "researcher", "level": 1}
        )
        print("   - Researcher (Claude)")
    else:
        print("   ⚠️  Skipping researcher (ANTHROPIC_API_KEY not set)")

    # Set global context
    print("\n3. Setting shared context...")
    coordinator.set_context(
        "project",
        "Research on AI Agent Architectures",
        scope=ContextScope.GLOBAL
    )
    coordinator.set_context(
        "deadline",
        "2025-12-31",
        scope=ContextScope.SUBTREE
    )

    # View tree structure
    print("\n4. Agent Tree Structure:")
    print(coordinator.get_tree_view())

    # Check context access
    print("\n5. Context Accessibility:")
    coord_context = coordinator.get_all_context()
    print(f"   Coordinator can see: {list(coord_context.keys())}")

    if anthropic_key:
        researcher_context = researcher.get_all_context()
        print(f"   Researcher can see: {list(researcher_context.keys())}")

    # Get tree stats
    print("\n6. Tree Statistics:")
    stats = context.get_tree_stats()
    print(f"   Total agents: {stats['total_agents']}")
    print(f"   Total context entries: {stats['total_context_entries']}")
    print(f"   Agents: {', '.join(stats['agents'])}")


def example_message_passing():
    """Example 2: Inter-agent message passing"""
    print("\n" + "="*70)
    print("Example 2: Inter-Agent Message Passing")
    print("="*70)

    context = SharedContext()
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        print("⚠️  OPENAI_API_KEY not set - skipping this example")
        return

    # Create agents
    print("\n1. Creating agent hierarchy...")
    root = HierarchicalAgent(
        agent_id="root",
        provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key),
        context=context
    )

    child1 = root.add_child(
        agent_id="child1",
        provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key)
    )

    child2 = root.add_child(
        agent_id="child2",
        provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key)
    )

    # Message passing
    print("\n2. Sending messages...")

    # Broadcast to children
    root.broadcast_to_children(
        message_type="task_update",
        content="Starting new project phase"
    )
    print("   ✓ Root broadcast to children")

    # Child sends to parent
    child1.send_to_parent(
        message_type="status_update",
        content="Task 1 completed"
    )
    print("   ✓ Child1 sent message to parent")

    # Targeted message
    child1.send_message(
        receiver_id="child2",
        message_type="collaboration",
        content="Need help with analysis"
    )
    print("   ✓ Child1 sent message to Child2")

    # Retrieve messages
    print("\n3. Retrieving messages...")
    root_messages = root.get_messages()
    print(f"   Root received {len(root_messages)} messages")
    for msg in root_messages:
        print(f"     - From {msg.sender_id}: {msg.content}")

    child2_messages = child2.get_messages()
    print(f"   Child2 received {len(child2_messages)} messages")
    for msg in child2_messages:
        print(f"     - From {msg.sender_id}: {msg.content}")


def example_task_delegation():
    """Example 3: Task delegation and result aggregation"""
    print("\n" + "="*70)
    print("Example 3: Task Delegation and Result Aggregation")
    print("="*70)

    context = SharedContext()
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key:
        print("⚠️  OPENAI_API_KEY not set - skipping this example")
        return

    # Create coordinator
    print("\n1. Creating coordinator and worker agents...")
    coordinator = HierarchicalAgent(
        agent_id="coordinator",
        provider=OpenAIAgent(model="gpt-4-turbo", api_key=openai_key),
        context=context
    )

    # Add worker agents
    worker1 = coordinator.add_child(
        agent_id="worker1",
        provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key),
        metadata={"specialty": "analysis"}
    )

    if anthropic_key:
        worker2 = coordinator.add_child(
            agent_id="worker2",
            provider=AnthropicAgent(
                model="claude-3-haiku-20240307",
                api_key=anthropic_key
            ),
            metadata={"specialty": "research"}
        )
    else:
        worker2 = coordinator.add_child(
            agent_id="worker2",
            provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key),
            metadata={"specialty": "research"}
        )

    # Set shared context
    coordinator.set_context(
        "topic",
        "Benefits of hierarchical agent architectures",
        scope=ContextScope.SUBTREE
    )

    # Delegate task to specific child
    print("\n2. Delegating task to worker1...")
    result1 = coordinator.delegate_to_child(
        child_id="worker1",
        task_description="Analyze the key benefits of hierarchical agent systems",
        max_tokens=150
    )
    print(f"   ✓ Worker1 completed: {result1['choices'][0]['message']['content'][:100]}...")

    # Aggregate results from all children
    print("\n3. Aggregating results from all workers...")
    aggregated = coordinator.aggregate_from_children(
        task_description="Provide one benefit of hierarchical agents",
        aggregation_prompt="Synthesize the responses from workers into a coherent summary",
        max_tokens=200
    )
    print(f"   ✓ Aggregated result: {aggregated['choices'][0]['message']['content'][:150]}...")


def example_multi_level_hierarchy():
    """Example 4: Multi-level hierarchy with 3 levels"""
    print("\n" + "="*70)
    print("Example 4: Multi-Level Hierarchy (3 Levels)")
    print("="*70)

    context = SharedContext()
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        print("⚠️  OPENAI_API_KEY not set - skipping this example")
        return

    print("\n1. Creating 3-level hierarchy...")

    # Level 0: Root
    root = HierarchicalAgent(
        agent_id="ceo",
        provider=OpenAIAgent(model="gpt-4-turbo", api_key=openai_key),
        context=context,
        metadata={"level": 0, "role": "CEO"}
    )

    # Level 1: Department heads
    research_head = root.add_child(
        agent_id="research_head",
        provider=OpenAIAgent(model="gpt-4-turbo", api_key=openai_key),
        metadata={"level": 1, "role": "Research Head"}
    )

    engineering_head = root.add_child(
        agent_id="engineering_head",
        provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key),
        metadata={"level": 1, "role": "Engineering Head"}
    )

    # Level 2: Team members
    researcher1 = research_head.add_child(
        agent_id="researcher1",
        provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key),
        metadata={"level": 2, "role": "Researcher"}
    )

    researcher2 = research_head.add_child(
        agent_id="researcher2",
        provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key),
        metadata={"level": 2, "role": "Researcher"}
    )

    engineer1 = engineering_head.add_child(
        agent_id="engineer1",
        provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key),
        metadata={"level": 2, "role": "Engineer"}
    )

    # Display tree
    print("\n2. Organization Structure:")
    print(root.get_tree_view())

    # Set cascading context
    print("\n3. Setting cascading context...")
    root.set_context("company_goal", "Build best AI agents", ContextScope.GLOBAL)
    research_head.set_context("department_goal", "Research new architectures", ContextScope.SUBTREE)
    engineering_head.set_context("department_goal", "Implement prototypes", ContextScope.SUBTREE)

    # Check context visibility
    print("\n4. Context visibility at different levels:")
    print(f"   CEO sees: {list(root.get_all_context().keys())}")
    print(f"   Research Head sees: {list(research_head.get_all_context().keys())}")
    print(f"   Researcher1 sees: {list(researcher1.get_all_context().keys())}")

    # Broadcast from root
    print("\n5. Broadcasting message from CEO to entire organization...")
    root.broadcast_to_subtree(
        message_type="company_announcement",
        content="Great progress this quarter!"
    )

    # Check who received messages
    r1_messages = researcher1.get_messages(clear=False)
    e1_messages = engineer1.get_messages(clear=False)
    print(f"   Researcher1 received {len(r1_messages)} messages")
    print(f"   Engineer1 received {len(e1_messages)} messages")


def example_context_scoping():
    """Example 5: Different context scopes"""
    print("\n" + "="*70)
    print("Example 5: Context Scoping Demonstration")
    print("="*70)

    context = SharedContext()
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        print("⚠️  OPENAI_API_KEY not set - skipping this example")
        return

    # Create hierarchy
    print("\n1. Creating hierarchy...")
    root = HierarchicalAgent(
        agent_id="root",
        provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key),
        context=context
    )

    child1 = root.add_child("child1", OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key))
    child2 = root.add_child("child2", OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key))
    grandchild = child1.add_child("grandchild", OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key))

    # Set context with different scopes
    print("\n2. Setting context with different scopes...")

    root.set_context("global_data", "Visible to all", ContextScope.GLOBAL)
    root.set_context("local_data", "Only root sees this", ContextScope.LOCAL)
    root.set_context("children_data", "Root and direct children", ContextScope.CHILDREN)
    root.set_context("subtree_data", "Entire subtree", ContextScope.SUBTREE)

    # Check visibility
    print("\n3. Context visibility:")
    print(f"   Root sees: {list(root.get_all_context().keys())}")
    print(f"   Child1 sees: {list(child1.get_all_context().keys())}")
    print(f"   Child2 sees: {list(child2.get_all_context().keys())}")
    print(f"   Grandchild sees: {list(grandchild.get_all_context().keys())}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Hierarchical Agent Tree Examples")
    print("="*70)
    print("\nThese examples demonstrate organizing agents in a tree structure")
    print("with shared context, message passing, and task delegation.")
    print("\nRequired environment variables:")
    print("  - OPENAI_API_KEY (required for most examples)")
    print("  - ANTHROPIC_API_KEY (optional, for Claude examples)")
    print("  - GOOGLE_API_KEY (optional, for Gemini examples)")
    print()

    try:
        # Run examples
        example_basic_tree()
        example_message_passing()
        example_task_delegation()
        example_multi_level_hierarchy()
        example_context_scoping()

        print("\n" + "="*70)
        print("All examples completed!")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
