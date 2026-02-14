"""
Runtime Configuration – In-memory settings changeable via Admin UI.

Persisted to a JSON file so settings survive restarts.
"""

import os
import json
from typing import Optional


class RuntimeConfig:
    """Runtime-configurable settings, persisted to disk."""

    DEFAULTS = {
        "strategy_model": "",  # Empty = use same as answer model (no split)
        "num_batch": 1024,
        "num_ctx_max": 131072,
        # Online model for strategy calls (fast tool-routing, no doc content sent)
        "online_model_enabled": False,
        "online_api_url": "https://api.openai.com/v1",  # OpenAI-compatible endpoint
        "online_api_key": "",
        "online_model_name": "gpt-4o-mini",  # Fast + cheap for routing
        "online_strategy_mode": "routing",  # "routing" = simple tool choice, "planner" = full search plan
        "agent_mode_enabled": True,  # True = ReAct Agent, False = direct LLM (no tools)
    }

    def __init__(self, path: str = None):
        self._path = path or os.getenv("RUNTIME_CONFIG_PATH", "/state/runtime_config.json")
        self._data = dict(self.DEFAULTS)
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    saved = json.load(f)
                self._data.update(saved)
            except Exception:
                pass

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as e:
            print(f"⚠️ RuntimeConfig save failed: {e}")

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value):
        self._data[key] = value
        self._save()

    def get_all(self) -> dict:
        return dict(self._data)

    def update(self, data: dict):
        self._data.update(data)
        self._save()


# Singleton
_config: Optional[RuntimeConfig] = None

def get_runtime_config() -> RuntimeConfig:
    global _config
    if _config is None:
        _config = RuntimeConfig()
    return _config
