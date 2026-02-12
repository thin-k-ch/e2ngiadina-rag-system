import os
import time
import hashlib
import json
import asyncio
import traceback
from fastapi import FastAPI, Header, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from .agent_orchestrator import Agent, AgentOrchestrator
from .rag_pipeline import SimpleRAGPipeline
from .state import StateStore
from .es_proxy import router as es_router

app = FastAPI(title="AGENTIC RAG API")

# Agent instances
agent = Agent()
orchestrator = AgentOrchestrator()

# MVP: Simple RAG pipeline (Query → Search → Snippets → LLM)
simple_rag = SimpleRAGPipeline()

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

def _format_phase_for_ui(event: dict) -> str:
    """Format phase events for OpenWebUI display"""
    event_type = event.get("type")
    phase = event.get("phase", "")
    
    if event_type == "phase_start":
        phase_names = {
            "strategy": "🎯 Strategie",
            "retrieval": "🔍 Suche", 
            "analysis": "📄 Dokumenten-Analyse",
            "validation": "✓ Validierung",
            "answer": "💡 Antwort"
        }
        name = phase_names.get(phase, phase)
        return f"\n[{name}] wird gestartet...\n"
    
    elif event_type == "phase_progress":
        message = event.get("message", "")
        return f"  → {message}\n"
    
    elif event_type == "phase_complete":
        return f"  ✓ Abgeschlossen\n"
    
    elif event_type == "error":
        msg = event.get("message", "Unknown error")
        return f"  ⚠ Fehler: {msg}\n"
    
    return ""

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

class RAGConfig(BaseModel):
    max_context_docs: int | None = None
    max_sources: int | None = None
    search_top_k: int | None = None
    keyword_boost_path: float | None = None
    keyword_boost_snippet: float | None = None
    excel_penalty_relevant: float | None = None
    excel_penalty_irrelevant: float | None = None
    answer_temperature: float | None = None

class ChatReq(BaseModel):
    model: str | None = None
    messages: list[Message]
    stream: bool | None = False
    rag_config: RAGConfig | None = None

# ----------------------------
# Endpoints
# ----------------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "agent_api",
        "version": "2.0-multi-phase",
        "time": int(time.time()),
        "models": {
            "strategy": orchestrator.model_strategy,
            "answer": orchestrator.model_answer
        }
    }

@app.get("/v1/models")
async def models():
    """List available RAG models (ollama models with rag- prefix)"""
    try:
        # Fetch models from Ollama
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{ollama_base}/api/tags")
            ollama_models = r.json().get("models", [])
        
        # Format for OpenAI-compatible response with rag- prefix
        data = []
        for m in ollama_models:
            model_id = m.get("name", m.get("model", ""))
            # Add rag- prefix to distinguish from raw Ollama models
            rag_model_id = f"rag-{model_id}"
            data.append({
                "id": rag_model_id,
                "object": "model",
                "created": 0,
                "owned_by": "rag-pipeline"
            })
        
        return {"object": "list", "data": data}
    except Exception as e:
        # Fallback to static list if Ollama unreachable
        return {
            "object": "list",
            "data": [
                {"id": "rag-llama4:latest", "object": "model", "created": 0, "owned_by": "rag-pipeline"},
                {"id": "rag-apertus:70b-instruct-2509-q4_k_m", "object": "model", "created": 0, "owned_by": "rag-pipeline"},
                {"id": "rag-qwen2.5:3b", "object": "model", "created": 0, "owned_by": "rag-pipeline"},
                {"id": "rag-gpt-oss:latest", "object": "model", "created": 0, "owned_by": "rag-pipeline"},
            ]
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
    
    # Pipeline selection: Default = SimpleRAG (MVP), [ADVANCED] = Agentic
    use_simple_rag = True
    if user_text.startswith("[ADVANCED]"):
        use_simple_rag = False
        user_text = user_text.replace("[ADVANCED]", "").strip()
    
    if use_simple_rag:
        # MVP: Simple RAG Pipeline - use selected model (strip rag- prefix)
        selected_model = (req.model or "llama4:latest").replace("rag-", "", 1)
        
        # Build per-request config dict from rag_config
        run_config = {}
        if req.rag_config:
            if req.rag_config.max_context_docs is not None:
                run_config["max_context_docs"] = req.rag_config.max_context_docs
            if req.rag_config.max_sources is not None:
                run_config["max_sources"] = req.rag_config.max_sources
            if req.rag_config.search_top_k is not None:
                run_config["search_top_k"] = req.rag_config.search_top_k
            if req.rag_config.keyword_boost_path is not None:
                run_config["keyword_boost_path"] = req.rag_config.keyword_boost_path
            if req.rag_config.keyword_boost_snippet is not None:
                run_config["keyword_boost_snippet"] = req.rag_config.keyword_boost_snippet
            if req.rag_config.excel_penalty_relevant is not None:
                run_config["excel_penalty_relevant"] = req.rag_config.excel_penalty_relevant
            if req.rag_config.excel_penalty_irrelevant is not None:
                run_config["excel_penalty_irrelevant"] = req.rag_config.excel_penalty_irrelevant
            if req.rag_config.answer_temperature is not None:
                run_config["answer_temperature"] = req.rag_config.answer_temperature
        
        # Create pipeline with selected model
        from .rag_pipeline import create_pipeline
        pipeline = create_pipeline("simple", model=selected_model)
        
        answer_parts = []
        sources = []
        
        async for event in pipeline.run(user_text, summary, notes, config=run_config):
            if event.type == "token":
                answer_parts.append(event.data.get("content", ""))
            elif event.type == "complete":
                sources = event.data.get("sources", [])
        
        answer = "".join(answer_parts) if answer_parts else "Keine Antwort generiert."
        
        # Add sources directly to answer content as markdown links
        if sources:
            lines = ["", "", "Quellen:"]
            for s in sources:
                n = s.get("n", "?")
                dp = s.get("display_path", s.get("path", ""))
                url = s.get("local_url", "")
                if url:
                    lines.append(f"[{n}] [{dp}]({url})")
                else:
                    lines.append(f"[{n}] {dp}")
            answer += "\n" + "\n".join(lines)
        
        # Build sources for response
        formatted_sources = []
        for s in sources:
            formatted_sources.append({
                "n": s.get("n", "?"),
                "path": s.get("path", ""),
                "display_path": s.get("display_path", s.get("path", ""))
            })
        
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
        if formatted_sources:
            response["sources"] = formatted_sources
        
        # Save state (simple rag doesn't update summary/notes yet)
        store.save(conv_id, summary, notes)
        
        return response
    else:
        # Advanced: Agentic pipeline with full analysis
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
    rid = f"agentic_{int(time.time())}"
    created = int(time.time())
    model = req.model or "agentic-rag"
    
    if getattr(req, "stream", False):
        async def gen():
            # Send first chunk immediately
            yield _sse_chunk(rid, created, model, {"role": "assistant"})
            
            try:
                user_text = next((m.content for m in req.messages[::-1] if m.role == "user"), "")
                
                # MVP: SimpleRAG with selected model (strip rag- prefix)
                selected_model = model.replace("rag-", "", 1) if model.startswith("rag-") else (model or "llama4:latest")
                
                # Resolve conversation ID for session state
                conv_id = x_conversation_id or hashlib.md5(user_text.encode()).hexdigest()[:12]
                session = store.load(conv_id)
                summary = session.get("summary", "")
                notes = session.get("notes", "")
                
                # Check if user references a previous source: "Analysiere Quelle [1]"
                from .source_analyzer import detect_source_reference, fetch_document_text
                source_ref = detect_source_reference(user_text)
                
                if source_ref is not None:
                    # Load previous sources from global last_sources
                    last_session = store.load("last_sources")
                    prev_sources = last_session.get("sources", [])
                    
                    if prev_sources and 1 <= source_ref <= len(prev_sources):
                        ref_source = prev_sources[source_ref - 1]
                        ref_path = ref_source.get("path", "")
                        ref_display = ref_source.get("display_path", ref_path)
                        
                        yield _sse_chunk(rid, created, model, {"content": f"📄 Analysiere Dokument: **{ref_display}**\n\n"})
                        
                        # Fetch full text from ES
                        doc_text, doc_meta = await fetch_document_text(ref_path)
                        
                        if not doc_text:
                            yield _sse_chunk(rid, created, model, {"content": "⚠️ Dokumentinhalt konnte nicht aus ES geladen werden.\n"})
                        else:
                            # Truncate if too long for LLM context
                            max_chars = 12000
                            if len(doc_text) > max_chars:
                                doc_text = doc_text[:max_chars] + f"\n\n[... gekürzt, {len(doc_text)} Zeichen total]"
                            
                            # Stream LLM analysis
                            from .rag_pipeline import SimpleRAGPipeline
                            pipeline = SimpleRAGPipeline(model=selected_model)
                            
                            system_prompt = """DU BIST EIN DOKUMENTEN-ANALYSE-SYSTEM FÜR SCHWEIZER EISENBAHN-PROJEKTE (SBB TFK 2020).

REGELN:
1. Antworte IMMER auf Deutsch
2. Fasse den Dokumentinhalt strukturiert und vollständig zusammen
3. Hebe wichtige Fakten, Daten, Personen und Entscheidungen hervor
4. Nutze Aufzählungen und Überschriften für Übersichtlichkeit
5. Sei präzise und faktenbasiert - erfinde nichts"""

                            user_msg = f"""Hier ist das Dokument "{ref_display}":

{doc_text}

Aufgabe: {user_text}"""

                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_msg}
                            ]
                            
                            async for chunk in pipeline._llm_stream(messages):
                                yield _sse_chunk(rid, created, model, {"content": chunk})
                            
                            # Add source link at end
                            from urllib.parse import quote
                            url = ref_source.get("local_url", "")
                            yield _sse_chunk(rid, created, model, {"content": f"\n\nQuelle: [{ref_display}]({url})"})
                        
                        yield _sse_chunk(rid, created, model, {"content": ""}, finish_reason="stop")
                        yield "data: [DONE]\n\n"
                        return
                    else:
                        yield _sse_chunk(rid, created, model, {"content": f"⚠️ Quelle [{source_ref}] nicht gefunden. Bitte zuerst eine Suche durchführen.\n"})
                        yield _sse_chunk(rid, created, model, {"content": ""}, finish_reason="stop")
                        yield "data: [DONE]\n\n"
                        return
                
                # Normal RAG flow
                # Build per-request config from rag_config if provided
                run_config = {}
                if hasattr(req, 'rag_config') and req.rag_config:
                    if req.rag_config.max_context_docs is not None:
                        run_config["max_context_docs"] = req.rag_config.max_context_docs
                    if req.rag_config.max_sources is not None:
                        run_config["max_sources"] = req.rag_config.max_sources
                
                from .rag_pipeline import create_pipeline
                pipeline = create_pipeline("simple", model=selected_model)
                
                answer_parts = []
                sources = []
                
                async for event in pipeline.run(user_text, "", "", config=run_config):
                    if event.type == "token":
                        content = event.data.get("content", "")
                        answer_parts.append(content)
                        yield _sse_chunk(rid, created, model, {"content": content})
                    elif event.type == "complete":
                        sources = event.data.get("sources", [])
                
                # Add sources at end with clickable links
                if sources:
                    from urllib.parse import quote
                    source_lines = []
                    for s in sources:
                        n = s.get('n', '?')
                        dp = s.get('display_path', s.get('path', ''))
                        url = s.get('local_url', '')
                        if url:
                            source_lines.append(f"[{n}] [{dp}]({url})")
                        else:
                            source_lines.append(f"[{n}] {dp}")
                    source_text = "\n\nQuellen:\n" + "\n".join(source_lines)
                    yield _sse_chunk(rid, created, model, {"content": source_text})
                
                # Save sources globally for "Analysiere Quelle [N]" feature
                store.save(conv_id, summary, notes)
                if sources:
                    store.save("last_sources", "", "", sources=sources)
                
                # End marker
                yield _sse_chunk(rid, created, model, {"content": ""}, finish_reason="stop")
                yield "data: [DONE]\n\n"

            except Exception as e:
                yield _sse_chunk(rid, created, model, {"content": f"\n⚠ Fehler: {type(e).__name__}: {e}"}, finish_reason="stop")
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
