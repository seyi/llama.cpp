"""
Hierarchical agent implementation with tree structure and shared context.

This module provides HierarchicalAgent, which allows agents to be organized
in a tree structure with parent-child relationships, shared context, and
inter-agent communication.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import uuid

from .base import BaseAgent, Tool
from .context import SharedContext, ContextScope, AgentMessage


@dataclass
class AgentTask:
    """A task assigned to an agent"""
    task_id: str
    description: str
    tools: List[Tool]
    assigned_to: Optional[str] = None
    result: Optional[Any] = None
    status: str = "pending"  # pending, in_progress, completed, failed


class HierarchicalAgent(BaseAgent):
    """
    Hierarchical agent that can be organized in a tree structure.

    Features:
    - Parent-child relationships
    - Shared context with scoping
    - Inter-agent message passing
    - Task delegation to children
    - Result aggregation from children
    - Different AI providers at different levels

    Example:
        >>> context = SharedContext()
        >>>
        >>> # Create root agent
        >>> root = HierarchicalAgent(
        ...     agent_id="coordinator",
        ...     provider=OpenAIAgent(model="gpt-4-turbo", api_key="..."),
        ...     context=context
        ... )
        >>>
        >>> # Create child agents
        >>> researcher = HierarchicalAgent(
        ...     agent_id="researcher",
        ...     provider=AnthropicAgent(model="claude-3-opus", api_key="..."),
        ...     context=context,
        ...     parent=root
        ... )
        >>>
        >>> # Share context
        >>> root.set_context("task", "Research AI agents", scope=ContextScope.SUBTREE)
        >>> task = researcher.get_context("task")  # Can access parent's context
        >>>
        >>> # Delegate work
        >>> result = root.delegate_to_child("researcher", "Search for papers")
    """

    def __init__(
        self,
        agent_id: str,
        provider: BaseAgent,
        context: SharedContext,
        parent: Optional['HierarchicalAgent'] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize hierarchical agent.

        Args:
            agent_id: Unique identifier for this agent
            provider: Underlying agent provider (OpenAI, Anthropic, etc.)
            context: Shared context manager
            parent: Parent agent (None for root)
            metadata: Additional agent metadata
        """
        # Don't call BaseAgent.__init__ since we're wrapping a provider
        self.agent_id = agent_id
        self.provider = provider
        self.context = context
        self.parent = parent
        self.metadata = metadata or {}
        self.children: Dict[str, 'HierarchicalAgent'] = {}

        # Register with context
        parent_id = parent.agent_id if parent else None
        self.context.register_agent(agent_id, parent_id, metadata)

        # Add to parent's children
        if parent:
            parent.children[agent_id] = self

    @property
    def model(self) -> str:
        """Get the underlying provider's model"""
        return self.provider.model

    def add_child(
        self,
        agent_id: str,
        provider: BaseAgent,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'HierarchicalAgent':
        """
        Add a child agent.

        Args:
            agent_id: ID for the child agent
            provider: AI provider for the child
            metadata: Additional metadata

        Returns:
            The created child agent
        """
        child = HierarchicalAgent(
            agent_id=agent_id,
            provider=provider,
            context=self.context,
            parent=self,
            metadata=metadata,
        )
        return child

    def remove_child(self, agent_id: str):
        """Remove a child agent"""
        if agent_id in self.children:
            self.context.unregister_agent(agent_id)
            del self.children[agent_id]

    def set_context(
        self,
        key: str,
        value: Any,
        scope: ContextScope = ContextScope.SUBTREE,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Set a value in the shared context"""
        self.context.set(key, value, self.agent_id, scope, metadata)

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a value from the shared context"""
        return self.context.get(key, self.agent_id, default)

    def get_all_context(
        self,
        scope_filter: Optional[ContextScope] = None
    ) -> Dict[str, Any]:
        """Get all accessible context entries"""
        return self.context.get_all(self.agent_id, scope_filter)

    def delete_context(self, key: str) -> bool:
        """Delete a context entry (only if owner)"""
        return self.context.delete(key, self.agent_id)

    def send_message(
        self,
        receiver_id: Optional[str],
        message_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Send a message to another agent or broadcast.

        Args:
            receiver_id: Target agent ID (None for broadcast)
            message_type: Type of message
            content: Message content
            metadata: Additional metadata
        """
        self.context.send_message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            metadata=metadata,
        )

    def get_messages(
        self,
        message_type: Optional[str] = None,
        clear: bool = True
    ) -> List[AgentMessage]:
        """Get messages for this agent"""
        return self.context.get_messages(self.agent_id, message_type, clear)

    def broadcast_to_children(
        self,
        message_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Broadcast a message to all direct children"""
        for child_id in self.children:
            self.send_message(child_id, message_type, content, metadata)

    def broadcast_to_subtree(
        self,
        message_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Broadcast a message to entire subtree"""
        descendants = self.context.get_descendants(self.agent_id)
        for agent_id in descendants:
            self.send_message(agent_id, message_type, content, metadata)

    def send_to_parent(
        self,
        message_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Send a message to parent agent"""
        parent_id = self.context.get_parent(self.agent_id)
        if parent_id:
            self.send_message(parent_id, message_type, content, metadata)

    def chat(
        self,
        message: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: str = "auto",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat message using the underlying provider.

        This delegates to the wrapped provider's chat method.
        """
        # Inject context into system message if present
        context_data = self.get_all_context()
        if context_data and not kwargs.get("no_context_injection"):
            context_str = f"\n\nShared Context:\n{self._format_context(context_data)}"

            # Add to system message if provider has conversation history
            if hasattr(self.provider, 'conversation_history') and self.provider.conversation_history:
                # Find system message and append context
                for msg in self.provider.conversation_history:
                    if msg.role == "system" and msg.content:
                        msg.content += context_str
                        break
            elif message:
                # Prepend to message
                message = f"{context_str}\n\n{message}"

        return self.provider.chat(
            message=message,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

    def delegate_to_child(
        self,
        child_id: str,
        task_description: str,
        tools: Optional[List[Tool]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Delegate a task to a child agent.

        Args:
            child_id: ID of child agent
            task_description: Description of the task
            tools: Tools available for the task
            **kwargs: Additional chat parameters

        Returns:
            Response from the child agent
        """
        if child_id not in self.children:
            raise ValueError(f"Child agent '{child_id}' not found")

        child = self.children[child_id]

        # Create task
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            description=task_description,
            tools=tools or [],
        )

        # Store task in context
        self.set_context(
            f"task:{task.task_id}",
            task.__dict__,
            scope=ContextScope.CHILDREN
        )

        # Send task message
        self.send_message(
            receiver_id=child_id,
            message_type="task_assignment",
            content=task.__dict__,
        )

        # Execute task on child
        response = child.chat(
            message=task_description,
            tools=tools,
            **kwargs
        )

        # Update task status
        task.status = "completed"
        task.result = response
        self.set_context(
            f"task:{task.task_id}",
            task.__dict__,
            scope=ContextScope.CHILDREN
        )

        return response

    def aggregate_from_children(
        self,
        task_description: str,
        aggregation_prompt: str,
        tools: Optional[List[Tool]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Delegate a task to all children and aggregate results.

        Args:
            task_description: Task for children
            aggregation_prompt: Prompt for aggregating results
            tools: Tools for children
            **kwargs: Additional chat parameters

        Returns:
            Aggregated response
        """
        if not self.children:
            raise ValueError("No children to delegate to")

        # Delegate to all children
        child_results = {}
        for child_id, child in self.children.items():
            try:
                result = self.delegate_to_child(
                    child_id=child_id,
                    task_description=task_description,
                    tools=tools,
                    **kwargs
                )
                child_results[child_id] = result
            except Exception as e:
                child_results[child_id] = {"error": str(e)}

        # Store results in context
        self.set_context(
            "child_results",
            child_results,
            scope=ContextScope.LOCAL
        )

        # Aggregate results
        results_summary = self._format_child_results(child_results)
        aggregation_message = f"{aggregation_prompt}\n\nChild Results:\n{results_summary}"

        return self.chat(message=aggregation_message, **kwargs)

    def get_tree_view(self, indent: int = 0) -> str:
        """Get a text representation of the agent tree"""
        lines = []
        prefix = "  " * indent

        # Current agent
        provider_name = self.provider.__class__.__name__
        lines.append(f"{prefix}├─ {self.agent_id} ({provider_name} - {self.model})")

        # Children
        for child_id, child in self.children.items():
            lines.append(child.get_tree_view(indent + 1))

        return "\n".join(lines)

    def _format_context(self, context_data: Dict[str, Any]) -> str:
        """Format context data for injection into prompts"""
        lines = []
        for key, value in context_data.items():
            if isinstance(value, (str, int, float, bool)):
                lines.append(f"- {key}: {value}")
            else:
                lines.append(f"- {key}: {str(value)[:100]}...")
        return "\n".join(lines)

    def _format_child_results(self, results: Dict[str, Any]) -> str:
        """Format child results for aggregation"""
        lines = []
        for child_id, result in results.items():
            if isinstance(result, dict) and "error" in result:
                lines.append(f"\n{child_id}: ERROR - {result['error']}")
            elif isinstance(result, dict) and "choices" in result:
                content = result["choices"][0]["message"].get("content", "")
                lines.append(f"\n{child_id}:\n{content}")
            else:
                lines.append(f"\n{child_id}:\n{str(result)}")
        return "\n".join(lines)

    def _convert_tools_to_provider_format(
        self,
        tools: List[Tool]
    ) -> List[Dict[str, Any]]:
        """Convert tools to provider format (delegate to underlying provider)"""
        return self.provider._convert_tools_to_provider_format(tools)

    def __repr__(self) -> str:
        return f"HierarchicalAgent(id='{self.agent_id}', provider={self.provider.__class__.__name__}, children={len(self.children)})"

    def __del__(self):
        """Cleanup when agent is deleted"""
        try:
            self.context.unregister_agent(self.agent_id)
        except:
            pass
