import os
import time
import hashlib
import json
import asyncio
from fastapi import FastAPI, Header, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .agent import Agent
from .state import StateStore
from .es_proxy import router as es_router

app = FastAPI(title="AGENTIC RAG API (OpenAI-compatible subset)")
agent = Agent()

STATE_PATH = os.getenv("STATE_PATH", "/state")

def _sse_chunk(rid, created, model, delta, finish_reason=None):
    chunk = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

def _sse_chat_completion(full_response: dict):
    """OpenAI SSE format: data: {...} and data: [DONE]"""
    rid = full_response.get("id", "agentic_stream")
    created = full_response.get("created", int(time.time()))
    model = full_response.get("model")
    
    content = ""
    try:
        content = full_response["choices"][0]["message"]["content"]
    except Exception:
        content = ""

    chunk = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
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

@app.get("/open")
async def open_file(path: str):
    """
    File proxy endpoint to serve local files via HTTP
    Security: Only serves files under FILE_BASE if set
    """
    file_base = os.getenv("FILE_BASE", "")
    
    # Security check: only allow files under base directory
    if file_base and not path.startswith(file_base):
        return {"error": "Access denied - path outside base directory"}
    
    # Normalize path
    normalized_path = os.path.normpath(path)
    
    # Check if file exists
    if not os.path.exists(normalized_path):
        return {"error": "File not found"}
    
    # Check if it's a file (not directory)
    if not os.path.isfile(normalized_path):
        return {"error": "Path is not a file"}
    
    return FileResponse(normalized_path)

async def chat_non_stream_impl(req: ChatReq, x_conversation_id: str | None = None):
    """Non-streaming chat implementation - reused by streaming and non-streaming paths"""
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

    answer, new_summary, new_notes, sources = await agent.answer(
        user_text=user_text,
        raw_messages=[m.model_dump() for m in req.messages],
        summary=summary,
        notes=notes,
    )

    store.save(conv_id, new_summary, new_notes)

    response = {
        "id": f"agentic_{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or "agentic-rag",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": answer
            },
            "finish_reason": "stop"
        }]
    }
    
    # Add sources if available
    if sources:
        response["sources"] = sources
    
    return response

@app.post("/v1/chat/completions")
async def chat(req: ChatReq, request: Request, x_conversation_id: str | None = Header(default=None)):
    if getattr(req, "stream", False):
        async def gen():
            rid = f"agentic_{int(time.time())}"
            created = int(time.time())
            model = req.model or "agentic-rag"

            # 1) send immediately (prevents OpenWebUI hanging)
            yield _sse_chunk(rid, created, model, {"role": "assistant"})
            await asyncio.sleep(0)

            # 2) compute FULL response using existing non-stream logic
            full = await chat_non_stream_impl(req, x_conversation_id)
            
            # Handle error case
            if "error" in full:
                yield _sse_chunk(rid, created, model, {"content": f"Error: {full['error']}"}, finish_reason="stop")
                yield "data: [DONE]\n\n"
                return

            content = ""
            try:
                content = full["choices"][0]["message"]["content"]
            except Exception:
                content = ""

            # 3) send content and done
            yield _sse_chunk(rid, created, model, {"content": content}, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")
    
    # else: existing non-stream return
    return await chat_non_stream_impl(req, x_conversation_id)

# Include ES Proxy Router
app.include_router(es_router, prefix="/proxy", tags=["es-proxy"])
