"""
SharedContextPool — cross-agent session context.

All agents read from and write to this pool during a session.
Enables the statistical agent's findings to inform the code agent,
and failed task outputs to inform retry strategies.
"""
from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class ContextEntry:
    agent: str
    key: str
    value: Any
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ttl_seconds: Optional[int] = None


class SharedContextPool:
    """Thread-safe key-value store scoped to a session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._store: Dict[str, ContextEntry] = {}
        self._lock = threading.RLock()

    def set(self, key: str, value: Any, agent: str, ttl_seconds: Optional[int] = None) -> None:
        with self._lock:
            self._store[key] = ContextEntry(
                agent=agent, key=key, value=value, ttl_seconds=ttl_seconds
            )

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return default
            if entry.ttl_seconds:
                age = (datetime.utcnow() - datetime.fromisoformat(entry.timestamp)).total_seconds()
                if age > entry.ttl_seconds:
                    del self._store[key]
                    return default
            return entry.value

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            return {k: e.value for k, e in self._store.items()}

    def get_summary(self) -> str:
        """Return a natural language summary of all context for LLM injection."""
        with self._lock:
            parts = []
            for key, entry in self._store.items():
                parts.append(f"[{entry.agent}] {key}: {str(entry.value)[:200]}")
            return "\n".join(parts) if parts else "No shared context yet."

    def append_to_list(self, key: str, item: Any, agent: str) -> None:
        with self._lock:
            existing = self._store.get(key)
            if existing and isinstance(existing.value, list):
                existing.value.append(item)
            else:
                self.set(key, [item], agent)

# Session-scoped pool registry
_pools: Dict[str, SharedContextPool] = {}
_pools_lock = threading.Lock()


def get_context_pool(session_id: str) -> SharedContextPool:
    with _pools_lock:
        if session_id not in _pools:
            _pools[session_id] = SharedContextPool(session_id)
        return _pools[session_id]


def clear_context_pool(session_id: str) -> None:
    with _pools_lock:
        _pools.pop(session_id, None)
