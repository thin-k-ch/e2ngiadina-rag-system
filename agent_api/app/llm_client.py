"""
Centralized LLM client supporting both Ollama and OpenAI-compatible backends.

This allows the Agent API to route answer generation to a GPU-accelerated
llama-server (OpenAI API) while keeping strategy/routing on Ollama or Groq.

Env vars:
    LLM_ANSWER_BACKEND   - "ollama" (default) or "openai"
    LLM_ANSWER_BASE_URL  - Base URL for OpenAI backend (e.g. http://host.docker.internal:8090)
    LLM_ANSWER_MODEL     - Model name for OpenAI backend (e.g. gpt-oss-120b)
"""

import os
import json
import httpx
from typing import AsyncGenerator, List, Dict, Optional

# ---------------------------------------------------------------------------
# Configuration (read once at import)
# ---------------------------------------------------------------------------

ANSWER_BACKEND = os.getenv("LLM_ANSWER_BACKEND", "ollama").lower()
ANSWER_BASE_URL = os.getenv("LLM_ANSWER_BASE_URL", "").rstrip("/")
ANSWER_MODEL = os.getenv("LLM_ANSWER_MODEL", "")


def get_answer_config() -> dict:
    """Return current answer backend configuration."""
    return {
        "backend": ANSWER_BACKEND,
        "base_url": ANSWER_BASE_URL,
        "model": ANSWER_MODEL,
    }


def is_openai_answer() -> bool:
    """Check if the answer model should use OpenAI-compatible backend."""
    return ANSWER_BACKEND == "openai" and bool(ANSWER_BASE_URL)


# ---------------------------------------------------------------------------
# OpenAI-compatible streaming
# ---------------------------------------------------------------------------

async def stream_chat_openai(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 8192,
    timeout: float = 300.0,
) -> AsyncGenerator[str, None]:
    """Stream tokens from an OpenAI-compatible API (llama-server, vLLM, etc.).

    Handles SSE format: ``data: {"choices":[{"delta":{"content":"..."}}]}``
    """
    # Clean messages: OpenAI API doesn't support "tool" role without tool_call_id
    clean = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "tool":
            clean.append({"role": "user", "content": f"[Tool-Ergebnis]:\n{content}"})
        elif "tool_calls" in m:
            tc = m.get("tool_calls", [{}])[0]
            func = tc.get("function", {})
            clean.append({"role": "assistant", "content": f"Tool: {func.get('name', '')}({json.dumps(func.get('arguments', {}))})"})
        else:
            clean.append({"role": role, "content": content})

    payload = {
        "model": model,
        "messages": clean,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=15.0, read=timeout)
    ) as client:
        async with client.stream(
            "POST", f"{base_url}/v1/chat/completions", json=payload
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass


# ---------------------------------------------------------------------------
# OpenAI-compatible completion (non-streaming)
# ---------------------------------------------------------------------------

async def complete_chat_openai(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 4096,
    timeout: float = 300.0,
) -> str:
    """Non-streaming chat completion from an OpenAI-compatible API."""
    # Clean messages
    clean = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "tool":
            clean.append({"role": "user", "content": f"[Tool-Ergebnis]:\n{content}"})
        elif "tool_calls" in m:
            tc = m.get("tool_calls", [{}])[0]
            func = tc.get("function", {})
            clean.append({"role": "assistant", "content": f"Tool: {func.get('name', '')}({json.dumps(func.get('arguments', {}))})"})
        else:
            clean.append({"role": role, "content": content})

    payload = {
        "model": model,
        "messages": clean,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=15.0, read=timeout)
    ) as client:
        r = await client.post(f"{base_url}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
