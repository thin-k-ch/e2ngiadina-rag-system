"""
Memory Store – Persistent long-term memory per tenant.

Stores key-value notes as JSON files under MEMORY_PATH/<tenant_short_name>.json
Each memory has: id, content, created_at, tags (optional)

Used by the ReAct Agent's manage_memory tool and injected into system prompt.
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Optional


class MemoryStore:
    """Persistent memory storage per tenant."""

    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.getenv("MEMORY_PATH", "/app/memories")
        os.makedirs(self.base_path, exist_ok=True)

    def _path(self, tenant_id: str) -> str:
        safe = "".join([c for c in tenant_id if c.isalnum() or c in ("-", "_")])[:80]
        if not safe:
            safe = "default"
        return os.path.join(self.base_path, f"{safe}.json")

    def _load(self, tenant_id: str) -> List[Dict]:
        p = self._path(tenant_id)
        if not os.path.exists(p):
            return []
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save(self, tenant_id: str, memories: List[Dict]):
        p = self._path(tenant_id)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(memories, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)

    def add(self, tenant_id: str, content: str, tags: List[str] = None) -> Dict:
        """Add a new memory. Returns the created memory entry."""
        memories = self._load(tenant_id)
        mem_id = hashlib.sha1(f"{content}{time.time()}".encode()).hexdigest()[:12]
        entry = {
            "id": mem_id,
            "content": content.strip(),
            "tags": tags or [],
            "created_at": int(time.time()),
        }
        memories.append(entry)
        self._save(tenant_id, memories)
        return entry

    def list_all(self, tenant_id: str) -> List[Dict]:
        """List all memories for a tenant."""
        return self._load(tenant_id)

    def search(self, tenant_id: str, query: str) -> List[Dict]:
        """Simple keyword search in memories."""
        memories = self._load(tenant_id)
        q = query.lower()
        return [m for m in memories if q in m.get("content", "").lower()
                or any(q in t.lower() for t in m.get("tags", []))]

    def delete(self, tenant_id: str, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if found and deleted."""
        memories = self._load(tenant_id)
        before = len(memories)
        memories = [m for m in memories if m.get("id") != memory_id]
        if len(memories) < before:
            self._save(tenant_id, memories)
            return True
        return False

    def delete_by_content(self, tenant_id: str, keyword: str) -> int:
        """Delete all memories matching a keyword. Returns count of deleted."""
        memories = self._load(tenant_id)
        kw = keyword.lower()
        kept = [m for m in memories if kw not in m.get("content", "").lower()]
        deleted = len(memories) - len(kept)
        if deleted > 0:
            self._save(tenant_id, kept)
        return deleted

    def format_for_prompt(self, tenant_id: str, max_entries: int = 20) -> str:
        """Format memories as text block for system prompt injection."""
        memories = self._load(tenant_id)
        if not memories:
            return ""
        # Most recent first, limited
        recent = sorted(memories, key=lambda m: m.get("created_at", 0), reverse=True)[:max_entries]
        lines = []
        for m in recent:
            tags = f" [{', '.join(m['tags'])}]" if m.get("tags") else ""
            lines.append(f"- {m['content']}{tags}")
        return "\n".join(lines)


# Singleton
_store: Optional[MemoryStore] = None

def get_memory_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
