import os
import time
import hashlib
import json
import asyncio
import traceback
from fastapi import FastAPI, Header, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from .agent import Agent
from .state import StateStore
from .es_proxy import router as es_router

app = FastAPI(title="AGENTIC RAG API (OpenAI-compatible subset)")
agent = Agent()

ENABLE_DEBUG_ENDPOINTS = os.getenv("ENABLE_DEBUG_ENDPOINTS", "0") == "1"

STATE_PATH = os.getenv("STATE_PATH", "/state")
store = StateStore(STATE_PATH)

# ----------------------------
# OpenAI-compatible SSE helpers
# ----------------------------

def _normalize_delta_content(v) -> str:
    """
    OpenAI streaming requires: choices[0].delta.content is a STRING (or absent).
    Some internal code may emit dicts like {"type":"token","content":"..."}.
    We convert those to strings so OpenWebUI can render streaming properly.
    """
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="ignore")
        except Exception:
            return str(v)
    if isinstance(v, dict):
        # common internal shape: {"type": "...", "content": "..."}
        inner = v.get("content")
        if isinstance(inner, str):
            return inner
        if inner is None:
            # fallback: stringify dict (last resort)
            return json.dumps(v, ensure_ascii=False)
        return str(inner)
    # fallback: stringify any other type
    return str(v)

def _sse_chunk(rid: str, created: int, model: str, delta: dict, finish_reason=None) -> str:
    # Ensure OpenAI-compatible delta fields
    safe_delta = {}
    if "role" in delta and isinstance(delta["role"], str):
        safe_delta["role"] = delta["role"]
    if "content" in delta:
        safe_delta["content"] = _normalize_delta_content(delta["content"])

    chunk = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": safe_delta,
            "finish_reason": finish_reason
        }]
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

def derive_conv_id(messages, x_conversation_id: str | None) -> str:
    if x_conversation_id and x_conversation_id.strip():
        return x_conversation_id.strip()
    joined = "\n".join([f"{m.role}:{m.content}" for m in messages])[:2000]
    h = hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()
    return f"conv_{h[:16]}"

# ----------------------------
# OpenAI request models
# ----------------------------

class Message(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    model: str | None = None
    messages: list[Message]
    stream: bool | None = False

# ----------------------------
# Endpoints
# ----------------------------

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
    file_base = os.getenv("FILE_BASE", "")
    if file_base and not path.startswith(file_base):
        return {"error": "Access denied - path outside base directory"}

    normalized_path = os.path.normpath(path)
    if not os.path.exists(normalized_path):
        return {"error": "File not found"}
    if not os.path.isfile(normalized_path):
        return {"error": "Path is not a file"}

    return FileResponse(normalized_path)

async def chat_non_stream_impl(req: ChatReq, x_conversation_id: str | None = None):
    if not req.messages:
        return {"error": "No messages provided"}

    conv_id = derive_conv_id(req.messages, x_conversation_id)
    state = store.load(conv_id)
    summary = state.get("summary", "") or ""
    notes = state.get("notes", "") or ""

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
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop"
        }]
    }
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

            # Important: send first chunk immediately so OpenWebUI stops showing skeleton.
            yield _sse_chunk(rid, created, model, {"role": "assistant"})
            
            # Important: send TRACE immediately so OpenWebUI shows activity.
            yield _sse_chunk(rid, created, model, {"content": "[TRACE]\n"})

            try:
                # If agent has a native streaming generator, prefer it.
                if hasattr(agent, "answer_stream"):
                    async for part in agent.answer_stream(
                        user_text=next((m.content for m in req.messages[::-1] if m.role == "user"), ""),
                        raw_messages=[m.model_dump() for m in req.messages],
                        summary=(store.load(derive_conv_id(req.messages, x_conversation_id)).get("summary","") or ""),
                        notes=(store.load(derive_conv_id(req.messages, x_conversation_id)).get("notes","") or ""),
                    ):
                        # part may be a string, dict, etc. Normalize to string.
                        normalized_content = _normalize_delta_content(part)
                        yield _sse_chunk(rid, created, model, {"content": normalized_content})
                    yield "data: [DONE]\n\n"
                    return

                # Fallback: compute full and send as single content chunk.
                # Send heartbeat while computing
                yield _sse_chunk(rid, created, model, {"content": "[TRACE] Retrieving documents..."})
                
                # Background computation with periodic heartbeats
                async def compute_with_heartbeat():
                    return await chat_non_stream_impl(req, x_conversation_id)
                
                # Run computation with heartbeats every 2 seconds
                task = asyncio.create_task(compute_with_heartbeat())
                
                while not task.done():
                    yield _sse_chunk(rid, created, model, {"content": " "})  # Heartbeat space
                    await asyncio.sleep(2)
                
                full = await task

                if "error" in full:
                    yield _sse_chunk(rid, created, model, {"content": f"Error: {full['error']}"}, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return

                content = ""
                try:
                    content = full["choices"][0]["message"]["content"]
                except Exception:
                    content = ""

                yield _sse_chunk(rid, created, model, {"content": content}, finish_reason="stop")
                yield "data: [DONE]\n\n"

            except Exception as e:
                # Always end stream properly.
                yield _sse_chunk(rid, created, model, {"content": f"Error: {type(e).__name__}: {e}"}, finish_reason="stop")
                yield "data: [DONE]\n\n"

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

    # Non-stream path
    full = await chat_non_stream_impl(req, x_conversation_id)
    if isinstance(full, dict) and "error" in full:
        return JSONResponse(full, status_code=400)
    return full

# Include ES Proxy Router
app.include_router(es_router, prefix="/proxy", tags=["es-proxy"])

# Debug SSE endpoint (ChatGPT's debugging suggestion)
@app.get("/debug/sse")
async def debug_sse(request: Request):
    if not ENABLE_DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Not found")
    async def gen():
        try:
            # sofort was senden (verhindert OpenWebUI "graue Streifen")
            yield "data: [debug] hello\n\n"
            # heartbeat loop
            while True:
                if await request.is_disconnected():
                    print("[debug_sse] client disconnected")
                    return
                yield f"data: [hb] {int(time.time())}\n\n"
                await asyncio.sleep(2)
        except asyncio.CancelledError:
            print("[debug_sse] cancelled")
            return
        except Exception as e:
            print("[debug_sse] EXC:", repr(e))
            print(traceback.format_exc())
            # noch versuchen, Fehler als SSE zu senden
            yield f"data: [error] {repr(e)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
