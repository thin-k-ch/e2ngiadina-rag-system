"""
FastAPI routes for RAG endpoints with configurable pipelines.
MVP: SimpleRAG Pipeline (Query → Search → Snippets → LLM)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio

from ..rag_pipeline import create_pipeline, Event

router = APIRouter()

# Global pipeline instance - can be switched via API or env
_current_pipeline_type = "simple"  # MVP default
_pipeline_cache = {}

def _get_pipeline(pipeline_type: str = None):
    """Get or create pipeline instance"""
    global _current_pipeline_type, _pipeline_cache
    
    ptype = pipeline_type or _current_pipeline_type
    
    if ptype not in _pipeline_cache:
        _pipeline_cache[ptype] = create_pipeline(ptype)
    
    return _pipeline_cache[ptype]


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: str = "agentic-rag"
    stream: bool = False
    temperature: float = 0.3


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]


@router.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest):
    """
    OpenAI-compatible chat endpoint.
    
    Pipeline selection (via query prefixes):
    - Default: SimpleRAG (fast, MVP)
    - [ADVANCED] prefix: AgenticRAG (with strategy/analysis)
    """
    
    # Extract user text
    user_text = next(
        (m.get("content", "") for m in reversed(req.messages) if m.get("role") == "user"),
        ""
    )
    
    # Pipeline selection
    pipeline_type = "simple"  # Default: MVP
    if user_text.startswith("[ADVANCED]"):
        pipeline_type = "agentic"
        user_text = user_text.replace("[ADVANCED]", "").strip()
    
    pipeline = _get_pipeline(pipeline_type)
    
    if req.stream:
        return await _stream_response(req, pipeline, user_text)
    else:
        return await _complete_response(req, pipeline, user_text)


async def _complete_response(req: ChatRequest, pipeline, query: str):
    """Non-streaming response"""
    import time
    
    answer_parts = []
    sources = []
    
    async for event in pipeline.run(query):
        if event.type == "token":
            answer_parts.append(event.data.get("content", ""))
        elif event.type == "complete":
            sources = event.data.get("sources", [])
    
    answer = "".join(answer_parts) if answer_parts else "Keine Antwort generiert."
    
    # Add sources footer
    if sources:
        answer += "\n\n📚 Quellen:\n"
        for s in sources:
            n = s.get("n", "?")
            path = s.get("display_path", s.get("path", ""))
            answer += f"[{n}] {path}\n"
    
    return ChatResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=req.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop"
        }]
    )


async def _stream_response(req: ChatRequest, pipeline, query: str):
    """Streaming SSE response"""
    from fastapi.responses import StreamingResponse
    import time
    import json
    
    async def event_generator():
        rid = f"chatcmpl-{int(time.time())}"
        created = int(time.time())
        model = req.model
        
        answer_parts = []
        sources = []
        
        async for event in pipeline.run(query):
            if event.type == "phase_start":
                # Optional: yield phase markers for UI
                pass
            elif event.type == "progress":
                # Optional: yield progress for UI
                pass
            elif event.type == "token":
                content = event.data.get("content", "")
                answer_parts.append(content)
                
                chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": content}}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                
            elif event.type == "complete":
                sources = event.data.get("sources", [])
        
        # Add sources at end
        if sources:
            source_text = "\n\n📚 Quellen:\n" + "\n".join(
                f"[{s.get('n', '?')}] {s.get('display_path', s.get('path', ''))}"
                for s in sources
            )
            
            chunk = {
                "id": rid,
                "object": "chat.completion.chunk", 
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": source_text}}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # End marker
        yield f"data: {json.dumps({'choices': [{'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@router.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "pipeline": _current_pipeline_type}
