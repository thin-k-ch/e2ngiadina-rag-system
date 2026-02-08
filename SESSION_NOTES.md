# WINDSURF SESSION NOTES - CRITICAL LEARNINGS

## 🎯 **LLAMA4:LATEST IMPLEMENTATION - LESSONS LEARNED**

### 🚨 **CRITICAL ISSUES & SOLUTIONS**

#### **1. QUERY_PLANNER JSON PARSING CRASH**
**Problem:** `AttributeError: 'int' object has no attribute 'get'`
- **Root Cause:** Llama4 gibt manchmal malformed JSON zurück: `{"queries": 1}` statt `{"queries": ["text"]}`
- **Solution:** Robust helper `_safe_extract_queries()` implementiert
- **Code:** `/agent_api/app/query_planner.py` lines 4-37

#### **2. OPENWEBUI "NO RESPONSE" - STREAMING ISSUE**
**Problem:** OpenWebUI zeigt keine Antworten an, GPU arbeitet aber
- **Root Cause:** Agent API implementiert kein OpenAI SSE streaming
- **Expected:** `data: {...}` und `data: [DONE]` (Server-Sent Events)
- **Actual:** Komplettes JSON in einem Block
- **Solution:** StreamingResponse mit SSE Format implementiert
- **Code:** `/agent_api/app/main.py` lines 18-42, 144-147

#### **3. DOCKER COMPOSE CONFIGURATION**
**Problem:** Modelle nicht in OpenWebUI sichtbar
- **Root Cause:** `DEFAULT_MODELS` enthielt altes Modell
- **Solution:** Auf `llama4:latest` aktualisiert
- **File:** `/docker-compose.yml` line 155

### 🔧 **TECHNICAL IMPLEMENTATIONS**

#### **SSE Streaming Implementation**
```python
def _sse_chat_completion(full_response: dict):
    """OpenAI SSE format: data: {...} and data: [DONE]"""
    content = full_response["choices"][0]["message"]["content"]
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
```

#### **Robust JSON Parsing**
```python
def _safe_extract_queries(parsed, fallback_text: str) -> list[str]:
    """Handle dict, list, string, scalar, None"""
    if isinstance(parsed, dict):
        q = parsed.get("queries", [])
        if isinstance(q, list):
            return [str(x).strip() for x in q if str(x).strip()]
        if isinstance(q, str):
            return [q.strip()] if q.strip() else [fallback_text.strip()]
        # int/float/bool/None/other
        return [(fallback_text or "").strip()] if (fallback_text or "").strip() else []
    # ... additional type handling
```

### 📋 **VALIDATION CHECKLIST**

#### **Before Model Changes**
- [ ] Check current model compatibility with query_planner
- [ ] Verify Docker compose environment variables
- [ ] Test non-streaming responses first
- [ ] Test streaming responses with SSE format

#### **After Model Changes**  
- [ ] Validate `docker compose ps` shows all containers running
- [ ] Test Agent API health: `curl http://localhost:11436/health`
- [ ] Test non-streaming: `"stream": false`
- [ ] Test streaming: `"stream": true` (must show `data: {...}`)
- [ ] Verify OpenWebUI shows models and responses
- [ ] Check GPU utilization during requests

### 🚨 **COMMON TRAPS TO AVOID**

#### **1. Python Script Hanging**
- **Problem:** Complex Python patches hang in bash
- **Solution:** Use manual edits with `edit` tool instead
- **Pattern:** Break complex patches into small manual changes

#### **2. Container Not Rebuilt**
- **Problem:** Code changes not active after restart
- **Solution:** Use `docker compose build --no-cache` then `up -d`
- **Check:** Verify changes in container logs

#### **3. Streaming Format Issues**
- **Problem:** OpenWebUI expects SSE, gets JSON
- **Solution:** Always test with `curl -H 'Accept: text/event-stream'`
- **Validation:** Must see `data: {...}` lines, not JSON block

### 🔄 **RECOVERY PROCEDURES**

#### **If OpenWebUI Shows No Models**
1. Check `DEFAULT_MODELS` in docker-compose.yml
2. Verify Agent API: `curl http://localhost:11436/v1/models`
3. Restart OpenWebUI: `docker compose restart openwebui`
4. Force recreate: `docker compose up -d --force-recreate openwebui`

#### **If Agent API Crashes**
1. Check logs: `docker compose logs agent_api --tail 20`
2. Verify query_planner fixes are applied
3. Rebuild container: `docker compose build --no-cache agent_api`
4. Test with simple query first

#### **If Streaming Fails**
1. Test non-streaming: `"stream": false`
2. Test streaming: `"stream": true` with curl
3. Verify SSE format: `data: {...}` and `data: [DONE]`
4. Check imports: `StreamingResponse` in main.py

### 📊 **CURRENT WORKING CONFIGURATION**

#### **Models**
- **Primary:** `llama4:latest` (108.6B params, 67GB)
- **Fallback:** `qwen2.5:14b` (if needed)
- **Embedding:** `mxbai-embed-large:latest`

#### **Services**
- **Agent API:** http://localhost:11436 (with SSE streaming)
- **OpenWebUI:** http://localhost:8086 (compatible with streaming)
- **Ollama:** http://localhost:11434 (llama4:latest loaded)
- **Elasticsearch:** http://localhost:9200 (phrase search)
- **ChromaDB:** Embedded in Agent API (hybrid search)

#### **Git Tags**
- **Current:** `v1.2.0-llama4-streaming-fix`
- **Repository:** https://github.com/thin-k-ch/e2ngiadina-rag-system

### 🎯 **NEXT STEPS FOR FUTURE DEVELOPMENT**

1. **Real Streaming:** Implement true token-by-token streaming instead of single chunk
2. **Error Handling:** Better error messages for malformed responses
3. **Performance:** Optimize hybrid search for large documents
4. **Monitoring:** Add health checks and metrics
5. **Documentation:** Update API documentation with streaming examples

---
**Last Updated:** 2026-02-08 01:45 UTC
**Status:** All services stable with llama4:latest and full streaming support
