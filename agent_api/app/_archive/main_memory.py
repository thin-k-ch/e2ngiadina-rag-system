import os
import time
import hashlib
from fastapi import FastAPI, Header, Request
from pydantic import BaseModel

from .agent import Agent
from .state import StateStore

app = FastAPI(title="AGENTIC RAG API (OpenAI-compatible subset)")
agent = Agent()

STATE_PATH = os.getenv("STATE_PATH", "/state")
store = StateStore(STATE_PATH)

class Message(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    model: str | None = None
    messages: list[Message]
    stream: bool | None = False

def derive_conv_id(messages: list[Message], x_conversation_id: str | None) -> str:
    if x_conversation_id and x_conversation_id.strip():
        return x_conversation_id.strip()

    # stable-ish hash from the first ~2000 chars of conversation
    joined = "\n".join([f"{m.role}:{m.content}" for m in messages])[:2000]
    h = hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()
    return f"conv_{h[:16]}"

@app.get("/health")
def health():
    return {"ok": True, "service": "agent_api", "time": int(time.time())}

@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{"id": "agentic-rag", "object": "model", "created": 0, "owned_by": "local"}]
    }

@app.post("/v1/chat/completions")
async def chat(req: ChatReq, request: Request, x_conversation_id: str | None = Header(default=None)):
    if not req.messages:
        return {"error": "No messages provided"}

    conv_id = derive_conv_id(req.messages, x_conversation_id)
    state = store.load(conv_id)
    summary = state.get("summary", "") or ""
    notes = state.get("notes", "") or ""

    # take last user message (but keep full raw history for continuity)
    user_text = ""
    for m in req.messages[::-1]:
        if m.role == "user":
            user_text = m.content
            break
    
    # DEBUG: Log the actual query received
    print(f"🔍 OPENWEBUI_QUERY: '{user_text[:100]}'")
    print(f"🔍 OPENWEBUI_MESSAGES_COUNT: {len(req.messages)}")

    answer, new_summary, new_notes = await agent.answer(
        user_text=user_text,
        raw_messages=[m.model_dump() for m in req.messages],
        summary=summary,
        notes=notes,
    )

    store.save(conv_id, new_summary, new_notes)

    return {
        "id": f"agentic_{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or "agentic-rag",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        # debugging / optional visibility:
        "conversation_id": conv_id,
    }
