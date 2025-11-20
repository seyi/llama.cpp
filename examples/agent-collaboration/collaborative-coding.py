#!/usr/bin/env python3
"""
Collaborative Coding Example

This example demonstrates how multiple agents can collaborate to:
1. Plan a feature implementation
2. Write the code
3. Create tests
4. Review the implementation
5. Reach consensus on whether to merge

Requirements:
- llama-server running with agent collaboration enabled
- At least 4 available slots
"""

import requests
import json
import time
from typing import Dict, List

# Configuration
SERVER_URL = "http://localhost:8080"
POLLING_INTERVAL = 2  # seconds

class AgentCollabClient:
    def __init__(self, base_url: str = SERVER_URL):
        self.base_url = base_url
        self.agents = {}

    def spawn_agent(self, role: str, slot_id: int, capabilities: List[str]) -> str:
        """Spawn a new agent"""
        response = requests.post(
            f"{self.base_url}/v1/agents/spawn",
            json={
                "role": role,
                "slot_id": slot_id,
                "capabilities": capabilities
            }
        )
        response.raise_for_status()
        agent_id = response.json()["agent_id"]
        self.agents[role] = agent_id
        print(f"✓ Spawned {role} agent: {agent_id}")
        return agent_id

    def submit_workflow(self, tasks: List[Dict]) -> str:
        """Submit a task workflow"""
        response = requests.post(
            f"{self.base_url}/v1/tasks/workflow",
            json={"tasks": tasks}
        )
        response.raise_for_status()
        workflow_id = response.json()["workflow_id"]
        print(f"✓ Submitted workflow: {workflow_id}")
        return workflow_id

    def get_task_status(self, task_id: str) -> Dict:
        """Get task status"""
        response = requests.get(f"{self.base_url}/v1/tasks/{task_id}")
        response.raise_for_status()
        return response.json()

    def store_knowledge(self, key: str, value: str, agent_id: str, tags: List[str]):
        """Store knowledge in shared knowledge base"""
        response = requests.post(
            f"{self.base_url}/v1/knowledge",
            json={
                "key": key,
                "value": value,
                "agent_id": agent_id,
                "tags": tags
            }
        )
        response.raise_for_status()
        print(f"✓ Stored knowledge: {key}")

    def create_vote(self, question: str, options: List[str]) -> str:
        """Create a consensus vote"""
        response = requests.post(
            f"{self.base_url}/v1/consensus/vote/create",
            json={
                "question": question,
                "options": options,
                "type": "simple_majority"
            }
        )
        response.raise_for_status()
        vote_id = response.json()["vote_id"]
        print(f"✓ Created vote: {vote_id}")
        return vote_id

    def cast_vote(self, vote_id: str, agent_id: str, option: str):
        """Cast a vote"""
        response = requests.post(
            f"{self.base_url}/v1/consensus/vote/{vote_id}/cast",
            json={
                "agent_id": agent_id,
                "option": option
            }
        )
        response.raise_for_status()
        print(f"✓ Agent {agent_id} voted: {option}")

    def get_vote_result(self, vote_id: str) -> Dict:
        """Get vote result"""
        response = requests.get(f"{self.base_url}/v1/consensus/vote/{vote_id}")
        response.raise_for_status()
        return response.json()

    def send_message(self, from_agent: str, to_agent: str, subject: str, payload: Dict):
        """Send message between agents"""
        response = requests.post(
            f"{self.base_url}/v1/messages/send",
            json={
                "from_agent_id": from_agent,
                "to_agent_id": to_agent,
                "type": "direct",
                "subject": subject,
                "payload": payload
            }
        )
        response.raise_for_status()

    def list_agents(self) -> List[Dict]:
        """List all agents"""
        response = requests.get(f"{self.base_url}/v1/agents")
        response.raise_for_status()
        return response.json()["agents"]

    def get_stats(self) -> Dict:
        """Get system stats"""
        response = requests.get(f"{self.base_url}/v1/agents/stats")
        response.raise_for_status()
        return response.json()


def main():
    print("=" * 60)
    print("Agent Collaboration Example: Collaborative Coding")
    print("=" * 60)
    print()

    client = AgentCollabClient()

    # Step 1: Spawn agents
    print("Step 1: Spawning agents...")
    print("-" * 60)

    planner_id = client.spawn_agent("planner", 0, ["task_breakdown", "architecture_design"])
    coder_id = client.spawn_agent("coder", 1, ["python", "javascript", "code_generation"])
    tester_id = client.spawn_agent("tester", 2, ["unit_testing", "integration_testing"])
    reviewer_id = client.spawn_agent("reviewer", 3, ["code_review", "best_practices"])

    print()

    # Step 2: Define the feature to implement
    print("Step 2: Defining feature...")
    print("-" * 60)

    feature_description = """
    Implement a REST API endpoint for user authentication with the following requirements:
    - POST /api/auth/login (email, password)
    - Return JWT token on success
    - Include proper error handling
    - Hash passwords with bcrypt
    - Rate limiting (5 attempts per minute)
    """

    print(f"Feature: {feature_description.strip()}")
    print()

    # Step 3: Create workflow
    print("Step 3: Creating task workflow...")
    print("-" * 60)

    workflow_tasks = [
        {
            "id": "plan",
            "type": "analyze",
            "description": f"Create detailed implementation plan for: {feature_description}",
            "required_roles": ["planner"],
            "priority": 10,
            "parameters": {
                "feature": feature_description,
                "output": "Store implementation plan in knowledge base"
            }
        },
        {
            "id": "implement",
            "type": "generate",
            "description": "Implement the authentication endpoint based on plan",
            "dependencies": ["plan"],
            "required_roles": ["coder"],
            "priority": 8,
            "parameters": {
                "language": "python",
                "framework": "flask",
                "knowledge_key": "auth_implementation_plan"
            }
        },
        {
            "id": "test",
            "type": "test",
            "description": "Write comprehensive tests for authentication endpoint",
            "dependencies": ["implement"],
            "required_roles": ["tester"],
            "priority": 8,
            "parameters": {
                "test_framework": "pytest",
                "coverage_threshold": 80
            }
        },
        {
            "id": "review",
            "type": "review",
            "description": "Review implementation and tests",
            "dependencies": ["implement", "test"],
            "required_roles": ["reviewer"],
            "priority": 7,
            "parameters": {
                "check_security": True,
                "check_best_practices": True
            }
        }
    ]

    workflow_id = client.submit_workflow(workflow_tasks)
    print()

    # Step 4: Monitor workflow execution
    print("Step 4: Monitoring workflow execution...")
    print("-" * 60)

    task_ids = ["plan", "implement", "test", "review"]
    completed_tasks = set()

    while len(completed_tasks) < len(task_ids):
        for task_id in task_ids:
            if task_id in completed_tasks:
                continue

            try:
                status = client.get_task_status(task_id)
                task_status = status.get("status")

                if task_status == "completed" and task_id not in completed_tasks:
                    completed_tasks.add(task_id)
                    print(f"✓ Task '{task_id}' completed")

                    # Simulate storing results in knowledge base
                    if "result" in status:
                        result_key = f"{task_id}_result"
                        client.store_knowledge(
                            result_key,
                            json.dumps(status["result"]),
                            status.get("assigned_agent_id", "system"),
                            [task_id, "workflow", workflow_id]
                        )

                elif task_status == "failed":
                    print(f"✗ Task '{task_id}' failed")
                    completed_tasks.add(task_id)

            except requests.exceptions.HTTPError:
                # Task not found yet
                pass

        time.sleep(POLLING_INTERVAL)

    print()
    print("All tasks completed!")
    print()

    # Step 5: Consensus voting
    print("Step 5: Consensus voting on implementation...")
    print("-" * 60)

    vote_id = client.create_vote(
        "Should we merge this authentication implementation?",
        ["approve", "reject", "request_changes"]
    )

    # Simulate agents casting votes
    print("Agents casting votes...")
    client.cast_vote(vote_id, planner_id, "approve")
    client.cast_vote(vote_id, coder_id, "approve")
    client.cast_vote(vote_id, tester_id, "approve")
    client.cast_vote(vote_id, reviewer_id, "request_changes")

    print()

    # Get vote result
    vote_result = client.get_vote_result(vote_id)
    print(f"Vote Result: {vote_result['result']}")
    print(f"Votes: {vote_result['votes']}")
    print()

    # Step 6: Inter-agent communication
    print("Step 6: Inter-agent communication...")
    print("-" * 60)

    # Reviewer sends message to coder with feedback
    client.send_message(
        reviewer_id,
        coder_id,
        "Code review feedback",
        {
            "issues": [
                "Add input validation for email format",
                "Implement proper error messages",
                "Add logging for failed login attempts"
            ],
            "priority": "high"
        }
    )
    print("✓ Reviewer sent feedback to coder")
    print()

    # Step 7: Display final stats
    print("Step 7: Final statistics...")
    print("-" * 60)

    stats = client.get_stats()
    print(json.dumps(stats, indent=2))
    print()

    print("=" * 60)
    print("Collaborative coding workflow completed!")
    print("=" * 60)

    # Cleanup
    print("\nNote: To terminate agents, use:")
    print(f"curl -X DELETE {SERVER_URL}/v1/agents/{{agent_id}}")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to llama-server")
        print(f"Make sure the server is running at {SERVER_URL}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
