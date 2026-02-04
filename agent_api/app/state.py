import os
import json
import time
from typing import Any, Dict


def _now() -> int:
    return int(time.time())


class StateStore:
    """
    Simple JSON-per-conversation persistent store.
    Stored under STATE_PATH/<conv_id>.json

    Structure:
    {
      "summary": "...",
      "notes": "...",
      "updated_at": 1234567890
    }
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def _path(self, conv_id: str) -> str:
        # sanitize filename
        safe = "".join([c for c in conv_id if c.isalnum() or c in ("-", "_")])[:80]
        if not safe:
            safe = "conv"
        return os.path.join(self.base_path, f"{safe}.json")

    def load(self, conv_id: str) -> Dict[str, Any]:
        p = self._path(conv_id)
        if not os.path.exists(p):
            return {"summary": "", "notes": "", "updated_at": _now()}

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"summary": "", "notes": "", "updated_at": _now()}
            data.setdefault("summary", "")
            data.setdefault("notes", "")
            data.setdefault("updated_at", _now())
            return data
        except Exception:
            return {"summary": "", "notes": "", "updated_at": _now()}

    def save(self, conv_id: str, summary: str, notes: str) -> None:
        p = self._path(conv_id)
        tmp = p + ".tmp"
        data = {
            "summary": summary or "",
            "notes": notes or "",
            "updated_at": _now(),
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
