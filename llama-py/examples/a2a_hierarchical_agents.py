#!/usr/bin/env python3
"""
A2A Protocol with Hierarchical Agents Example

This example demonstrates how to use the Agent-to-Agent (A2A) protocol
with hierarchical agents for standardized inter-agent communication.

Features demonstrated:
- A2A protocol-compliant message passing
- Task lifecycle management with A2A states
- Context grouping with contextId
- Agent card generation for discovery
- Artifact creation from task results
- Task state transitions and history tracking

Tree Structure:
                    [Coordinator]
                     (GPT-4)
                         |
            ┌────────────┼────────────┐
            ▼            ▼            ▼
      [Researcher]  [Analyst]     [Writer]
       (Claude)      (Gemini)      (GPT-4)

A2A Protocol: https://a2a-protocol.org
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp.agents import (
    HierarchicalAgent,
    SharedContext,
    ContextScope,
    OpenAIAgent,
    AnthropicAgent,
    GoogleAgent,
    # A2A Protocol types
    A2AMessage,
    Task,
    TaskState,
    Artifact,
    TextPart,
    DataPart,
)


def example_a2a_task_lifecycle():
    """Example 1: A2A Task Lifecycle Management"""
    print("\n" + "="*70)
    print("Example 1: A2A Task Lifecycle Management")
    print("="*70)

    context = SharedContext(enable_a2a=True)
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        print("⚠️  OPENAI_API_KEY not set - skipping this example")
        return

    # Create coordinator agent
    print("\n1. Creating coordinator agent...")
    coordinator = HierarchicalAgent(
        agent_id="coordinator",
        provider=OpenAIAgent(model="gpt-4-turbo", api_key=openai_key),
        context=context,
        metadata={"role": "coordinator", "level": 0}
    )

    # Create an A2A task
    print("\n2. Creating A2A task...")
    task = coordinator.create_a2a_task(
        user_message="Explain the benefits of the A2A protocol in 50 words"
    )

    if task:
        print(f"   ✓ Task created: {task.id}")
        print(f"   ✓ Context ID: {task.contextId}")
        print(f"   ✓ Initial state: {task.status.state.value}")

        # Execute task
        print("\n3. Executing task...")
        context.update_a2a_task(
            task_id=task.id,
            state=TaskState.IN_PROGRESS,
            message="Processing user request"
        )

        response = coordinator.chat(
            message="Explain the benefits of the A2A protocol in 50 words",
            max_tokens=150
        )

        content = response["choices"][0]["message"]["content"]

        # Create artifact
        artifact = Artifact.from_text(
            text=content,
            name="a2a_benefits.txt",
            description="Explanation of A2A protocol benefits"
        )

        # Complete task
        context.update_a2a_task(
            task_id=task.id,
            state=TaskState.COMPLETED,
            message="Task completed successfully",
            artifact=artifact
        )

        print(f"   ✓ Task completed")
        print(f"   ✓ Result: {content[:100]}...")

        # Show task details
        print("\n4. Task Details:")
        final_task = context.get_a2a_task(task.id)
        print(f"   - State transitions: {len(final_task.status.stateTransitions or [])}")
        print(f"   - Artifacts: {len(final_task.artifacts or [])}")
        print(f"   - Messages in history: {len(final_task.history or [])}")

        # Export task as JSON
        print("\n5. Task JSON representation:")
        task_json = json.dumps(final_task.to_dict(), indent=2)
        print(f"   {task_json[:200]}...")


def example_a2a_message_passing():
    """Example 2: A2A Message Passing Between Agents"""
    print("\n" + "="*70)
    print("Example 2: A2A Message Passing Between Agents")
    print("="*70)

    context = SharedContext(enable_a2a=True)
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        print("⚠️  OPENAI_API_KEY not set - skipping this example")
        return

    print("\n1. Creating agent hierarchy...")
    root = HierarchicalAgent(
        agent_id="root",
        provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key),
        context=context
    )

    child1 = root.add_child(
        agent_id="child1",
        provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key),
        metadata={"role": "analyzer"}
    )

    child2 = root.add_child(
        agent_id="child2",
        provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key),
        metadata={"role": "summarizer"}
    )

    # Create shared context
    print("\n2. Creating shared A2A context...")
    shared_contextId = context.create_a2a_task(
        agent_id="root",
        initial_message=A2AMessage.from_text(
            "Collaborative analysis task",
            role="user"
        )
    ).contextId

    print(f"   ✓ Shared context ID: {shared_contextId}")

    # Send A2A messages
    print("\n3. Sending A2A messages...")

    # Root sends message to child1
    msg1 = root.send_a2a_message(
        receiver_id="child1",
        message_text="Please analyze the A2A protocol architecture",
        contextId=shared_contextId
    )
    print(f"   ✓ Root → Child1: Message sent")

    # Root sends message to child2
    msg2 = root.send_a2a_message(
        receiver_id="child2",
        message_text="Please summarize key A2A concepts",
        contextId=shared_contextId
    )
    print(f"   ✓ Root → Child2: Message sent")

    # Get all messages for the context
    print("\n4. Retrieving context messages...")
    context_messages = context.get_a2a_messages_by_context(shared_contextId)
    print(f"   ✓ Total messages in context: {len(context_messages)}")

    for i, msg in enumerate(context_messages, 1):
        text = msg.parts[0].text if msg.parts else ""
        print(f"   {i}. [{msg.role}] {text[:60]}...")


def example_a2a_task_delegation():
    """Example 3: A2A Task Delegation to Children"""
    print("\n" + "="*70)
    print("Example 3: A2A Task Delegation to Children")
    print("="*70)

    context = SharedContext(enable_a2a=True)
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key:
        print("⚠️  OPENAI_API_KEY not set - skipping this example")
        return

    print("\n1. Creating coordinator and workers...")
    coordinator = HierarchicalAgent(
        agent_id="coordinator",
        provider=OpenAIAgent(model="gpt-4-turbo", api_key=openai_key),
        context=context,
        metadata={"role": "coordinator"}
    )

    # Add workers
    if anthropic_key:
        worker1 = coordinator.add_child(
            agent_id="worker1",
            provider=AnthropicAgent(
                model="claude-3-5-sonnet-20241022",
                api_key=anthropic_key
            ),
            metadata={"role": "researcher", "specialty": "A2A protocol"}
        )
        print("   ✓ Worker1 (Claude) added")
    else:
        worker1 = coordinator.add_child(
            agent_id="worker1",
            provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key),
            metadata={"role": "researcher", "specialty": "A2A protocol"}
        )
        print("   ⚠️  Worker1 using OpenAI (ANTHROPIC_API_KEY not set)")

    # Delegate A2A task to worker
    print("\n2. Delegating A2A task to worker...")
    result = coordinator.delegate_a2a_task(
        child_id="worker1",
        task_description="List 3 key features of the A2A protocol",
        max_tokens=200
    )

    if result and "task" in result:
        task_data = result["task"]
        print(f"   ✓ Task ID: {task_data['id']}")
        print(f"   ✓ State: {task_data['status']['state']}")

        if "artifact" in result:
            artifact_data = result["artifact"]
            print(f"   ✓ Artifact created: {artifact_data['name']}")
            text_part = artifact_data['parts'][0]
            print(f"   ✓ Result: {text_part['text'][:150]}...")

        # Show state transitions
        if task_data['status'].get('stateTransitions'):
            print(f"\n3. State Transitions:")
            for transition in task_data['status']['stateTransitions']:
                print(f"   {transition['from']} → {transition['to']}: {transition.get('message', 'N/A')}")


def example_agent_cards():
    """Example 4: Agent Card Generation for Discovery"""
    print("\n" + "="*70)
    print("Example 4: Agent Card Generation for Discovery")
    print("="*70)

    context = SharedContext(enable_a2a=True)
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        print("⚠️  OPENAI_API_KEY not set - skipping this example")
        return

    print("\n1. Creating agents with different roles...")

    # Root coordinator
    coordinator = HierarchicalAgent(
        agent_id="coordinator",
        provider=OpenAIAgent(model="gpt-4-turbo", api_key=openai_key),
        context=context,
        metadata={"role": "coordinator"}
    )

    # Specialized workers
    researcher = coordinator.add_child(
        agent_id="researcher",
        provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key),
        metadata={"role": "researcher", "domain": "AI protocols"}
    )

    analyst = coordinator.add_child(
        agent_id="analyst",
        provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key),
        metadata={"role": "analyst", "domain": "data analysis"}
    )

    # Generate agent cards
    print("\n2. Generating A2A Agent Cards...")

    for agent in [coordinator, researcher, analyst]:
        card = agent.get_agent_card(
            url=f"http://localhost:8000/agents/{agent.agent_id}",
            version="1.0.0"
        )

        if card:
            print(f"\n   Agent: {agent.agent_id}")
            print(f"   - Name: {card.name}")
            print(f"   - Skills: {len(card.skills)}")
            for skill in card.skills:
                print(f"     • {skill.name}: {skill.description}")
            print(f"   - Capabilities:")
            print(f"     • Streaming: {card.capabilities.streaming}")
            print(f"     • State History: {card.capabilities.stateTransitionHistory}")

    # Export coordinator card as JSON
    print("\n3. Coordinator Agent Card (JSON):")
    card_json = coordinator.export_agent_card_json(
        url="http://localhost:8000/agents/coordinator",
        version="1.0.0"
    )
    if card_json:
        print(f"   {card_json[:400]}...")


def example_context_grouping():
    """Example 5: Context Grouping with A2A"""
    print("\n" + "="*70)
    print("Example 5: Context Grouping with contextId")
    print("="*70)

    context = SharedContext(enable_a2a=True)
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        print("⚠️  OPENAI_API_KEY not set - skipping this example")
        return

    print("\n1. Creating agents...")
    coordinator = HierarchicalAgent(
        agent_id="coordinator",
        provider=OpenAIAgent(model="gpt-3.5-turbo", api_key=openai_key),
        context=context
    )

    # Create multiple related tasks in the same context
    print("\n2. Creating related tasks in same context...")

    task1 = coordinator.create_a2a_task(
        user_message="What is A2A protocol?",
        contextId=None  # Will auto-generate
    )
    shared_context = task1.contextId

    task2 = coordinator.create_a2a_task(
        user_message="How does A2A handle task lifecycle?",
        contextId=shared_context  # Reuse same context
    )

    task3 = coordinator.create_a2a_task(
        user_message="What are A2A artifacts?",
        contextId=shared_context  # Reuse same context
    )

    print(f"   ✓ Shared context ID: {shared_context}")
    print(f"   ✓ Task 1 ID: {task1.id}")
    print(f"   ✓ Task 2 ID: {task2.id}")
    print(f"   ✓ Task 3 ID: {task3.id}")

    # Get all tasks for this context
    print("\n3. Retrieving all tasks for context...")
    all_tasks = context.get_tasks_by_context(shared_context)
    print(f"   ✓ Total tasks in context: {len(all_tasks)}")

    for i, task in enumerate(all_tasks, 1):
        if task.history:
            msg_text = task.history[0].parts[0].text if task.history[0].parts else ""
            print(f"   {i}. Task {task.id[:8]}... - {msg_text[:50]}...")


def main():
    """Run all A2A examples"""
    print("\n" + "="*70)
    print("A2A Protocol with Hierarchical Agents Examples")
    print("="*70)
    print("\nThese examples demonstrate the Agent-to-Agent (A2A) protocol")
    print("integration with hierarchical agents for standardized communication.")
    print("\nRequired environment variables:")
    print("  - OPENAI_API_KEY (required for most examples)")
    print("  - ANTHROPIC_API_KEY (optional, for Claude examples)")
    print("\nA2A Protocol: https://a2a-protocol.org")
    print()

    try:
        # Run examples
        example_a2a_task_lifecycle()
        example_a2a_message_passing()
        example_a2a_task_delegation()
        example_agent_cards()
        example_context_grouping()

        print("\n" + "="*70)
        print("All examples completed!")
        print("="*70)
        print("\n💡 Key A2A Features Demonstrated:")
        print("   ✓ Standardized message format with Parts (TextPart, DataPart, FilePart)")
        print("   ✓ Task lifecycle management with state transitions")
        print("   ✓ Context grouping for related interactions")
        print("   ✓ Agent Cards for capability discovery")
        print("   ✓ Artifacts for tangible task outputs")
        print("   ✓ Backward compatibility with legacy message passing")

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
