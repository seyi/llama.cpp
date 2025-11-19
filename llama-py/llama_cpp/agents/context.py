"""
Shared context management for hierarchical agent trees.

This module provides a context management system that allows agents in a tree
structure to share state, communicate, and coordinate their work.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import json


class ContextScope(Enum):
    """Defines the scope of context visibility"""
    LOCAL = "local"          # Only this agent
    CHILDREN = "children"    # This agent and its children
    SUBTREE = "subtree"      # Entire subtree from this agent down
    GLOBAL = "global"        # All agents in the tree


@dataclass
class ContextEntry:
    """A single entry in the shared context"""
    key: str
    value: Any
    scope: ContextScope
    owner_id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "key": self.key,
            "value": self.value,
            "scope": self.scope.value,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class AgentMessage:
    """Message passed between agents in the tree"""
    sender_id: str
    receiver_id: Optional[str]  # None means broadcast
    message_type: str
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class SharedContext:
    """
    Shared context manager for hierarchical agent trees.

    Provides:
    - Key-value storage with scoping
    - Message passing between agents
    - Event notifications
    - Thread-safe operations
    """

    def __init__(self):
        self._context: Dict[str, ContextEntry] = {}
        self._messages: List[AgentMessage] = []
        self._agent_registry: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def register_agent(
        self,
        agent_id: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register an agent in the tree"""
        with self._lock:
            self._agent_registry[agent_id] = {
                "parent_id": parent_id,
                "children": set(),
                "metadata": metadata or {},
                "registered_at": datetime.now(),
            }

            # Add to parent's children
            if parent_id and parent_id in self._agent_registry:
                self._agent_registry[parent_id]["children"].add(agent_id)

    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the tree"""
        with self._lock:
            if agent_id not in self._agent_registry:
                return

            # Remove from parent's children
            parent_id = self._agent_registry[agent_id]["parent_id"]
            if parent_id and parent_id in self._agent_registry:
                self._agent_registry[parent_id]["children"].discard(agent_id)

            # Clean up children
            children = list(self._agent_registry[agent_id]["children"])
            for child_id in children:
                self.unregister_agent(child_id)

            # Remove context entries owned by this agent
            self._context = {
                k: v for k, v in self._context.items()
                if v.owner_id != agent_id
            }

            del self._agent_registry[agent_id]

    def set(
        self,
        key: str,
        value: Any,
        agent_id: str,
        scope: ContextScope = ContextScope.SUBTREE,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Set a value in the shared context"""
        with self._lock:
            now = datetime.now()
            if key in self._context:
                entry = self._context[key]
                entry.value = value
                entry.updated_at = now
                if metadata:
                    entry.metadata.update(metadata)
            else:
                self._context[key] = ContextEntry(
                    key=key,
                    value=value,
                    scope=scope,
                    owner_id=agent_id,
                    created_at=now,
                    updated_at=now,
                    metadata=metadata or {},
                )

    def get(
        self,
        key: str,
        agent_id: str,
        default: Any = None
    ) -> Any:
        """Get a value from the shared context"""
        with self._lock:
            if key not in self._context:
                return default

            entry = self._context[key]

            # Check if agent has access based on scope
            if self._has_access(agent_id, entry):
                return entry.value

            return default

    def get_all(
        self,
        agent_id: str,
        scope_filter: Optional[ContextScope] = None
    ) -> Dict[str, Any]:
        """Get all accessible context entries for an agent"""
        with self._lock:
            result = {}
            for key, entry in self._context.items():
                if self._has_access(agent_id, entry):
                    if scope_filter is None or entry.scope == scope_filter:
                        result[key] = entry.value
            return result

    def delete(self, key: str, agent_id: str) -> bool:
        """Delete a context entry (only owner can delete)"""
        with self._lock:
            if key not in self._context:
                return False

            entry = self._context[key]
            if entry.owner_id != agent_id:
                return False

            del self._context[key]
            return True

    def send_message(
        self,
        sender_id: str,
        receiver_id: Optional[str],
        message_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Send a message to another agent or broadcast"""
        with self._lock:
            message = AgentMessage(
                sender_id=sender_id,
                receiver_id=receiver_id,
                message_type=message_type,
                content=content,
                metadata=metadata or {},
            )
            self._messages.append(message)

    def get_messages(
        self,
        agent_id: str,
        message_type: Optional[str] = None,
        clear: bool = True
    ) -> List[AgentMessage]:
        """Get messages for an agent"""
        with self._lock:
            messages = []
            remaining = []

            for msg in self._messages:
                # Check if message is for this agent
                if msg.receiver_id is None or msg.receiver_id == agent_id:
                    if message_type is None or msg.message_type == message_type:
                        messages.append(msg)
                        if not clear:
                            remaining.append(msg)
                    else:
                        remaining.append(msg)
                else:
                    remaining.append(msg)

            if clear:
                self._messages = remaining

            return messages

    def get_children(self, agent_id: str) -> Set[str]:
        """Get direct children of an agent"""
        with self._lock:
            if agent_id not in self._agent_registry:
                return set()
            return self._agent_registry[agent_id]["children"].copy()

    def get_descendants(self, agent_id: str) -> Set[str]:
        """Get all descendants of an agent (subtree)"""
        with self._lock:
            descendants = set()
            to_visit = list(self.get_children(agent_id))

            while to_visit:
                child_id = to_visit.pop()
                descendants.add(child_id)
                to_visit.extend(self.get_children(child_id))

            return descendants

    def get_parent(self, agent_id: str) -> Optional[str]:
        """Get parent of an agent"""
        with self._lock:
            if agent_id not in self._agent_registry:
                return None
            return self._agent_registry[agent_id]["parent_id"]

    def get_ancestors(self, agent_id: str) -> List[str]:
        """Get all ancestors of an agent (path to root)"""
        with self._lock:
            ancestors = []
            current = agent_id

            while True:
                parent = self.get_parent(current)
                if parent is None:
                    break
                ancestors.append(parent)
                current = parent

            return ancestors

    def get_tree_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent tree"""
        with self._lock:
            return {
                "total_agents": len(self._agent_registry),
                "total_context_entries": len(self._context),
                "total_messages": len(self._messages),
                "agents": list(self._agent_registry.keys()),
            }

    def _has_access(self, agent_id: str, entry: ContextEntry) -> bool:
        """Check if an agent has access to a context entry"""
        if entry.scope == ContextScope.GLOBAL:
            return True

        if entry.scope == ContextScope.LOCAL:
            return agent_id == entry.owner_id

        if entry.scope == ContextScope.CHILDREN:
            # Owner and direct children have access
            if agent_id == entry.owner_id:
                return True
            return entry.owner_id in self.get_ancestors(agent_id)

        if entry.scope == ContextScope.SUBTREE:
            # Owner and all descendants have access
            if agent_id == entry.owner_id:
                return True
            return entry.owner_id in self.get_ancestors(agent_id)

        return False

    def export_context(self, agent_id: str) -> Dict[str, Any]:
        """Export all accessible context for an agent"""
        with self._lock:
            entries = {}
            for key, entry in self._context.items():
                if self._has_access(agent_id, entry):
                    entries[key] = entry.to_dict()

            return {
                "agent_id": agent_id,
                "entries": entries,
                "exported_at": datetime.now().isoformat(),
            }

    def import_context(
        self,
        agent_id: str,
        context_data: Dict[str, Any]
    ):
        """Import context entries for an agent"""
        with self._lock:
            entries = context_data.get("entries", {})
            for key, entry_dict in entries.items():
                self.set(
                    key=key,
                    value=entry_dict["value"],
                    agent_id=agent_id,
                    scope=ContextScope(entry_dict["scope"]),
                    metadata=entry_dict.get("metadata"),
                )
