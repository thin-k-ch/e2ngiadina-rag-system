"""
ReAct Agent – Autonomous Tool-Calling Loop
=============================================

Replaces the hardcoded 5-path router (A/B/C/D/E) with an LLM-driven
tool-calling loop. The LLM decides which tools to use and in what order.

Flow:
    User query → LLM (with tools) → tool_call? → execute → LLM → ... → final answer (streamed)

Requires a tool-calling-capable model (qwen2.5:72b, llama3.3:70b, llama4).
Falls back to single-shot answer if model doesn't emit tool_calls.
"""

import os
import json
import asyncio
import time
from typing import AsyncGenerator, Dict, Any, List, Optional

# ---------------------------------------------------------------------------
# Tool Definitions (Ollama/OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Durchsucht das Projektarchiv (Elasticsearch + ChromaDB) nach Dokumenten. "
                           "Gibt Pfade und Textausschnitte zurück. Nutze dies für jede Frage, die sich "
                           "auf Projektdokumente, Verträge, E-Mails, Protokolle etc. bezieht.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Suchbegriffe oder Frage (deutsch)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Liest den Volltext eines bestimmten Dokuments aus Elasticsearch. "
                           "Nutze dies, wenn du ein Dokument im Detail analysieren musst "
                           "(z.B. nach search_documents einen bestimmten Treffer vertiefen).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Dokumentpfad (aus search_documents Ergebnis)"
                    }
                },
                "required": ["path"]
            }
        }
    },
]

# ---------------------------------------------------------------------------
# System Prompt for the ReAct Agent
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """DU BIST EIN AUTONOMER DOKUMENTEN-ANALYST für Schweizer Eisenbahn-Projekte (SBB TFK 2020 - Tunnelfunk).

FACHBEGRIFFE: FAT=Werksabnahme, SAT=Standortabnahme, TFK=Tunnelfunk, GBT=Gotthard Basistunnel, RBT=Rhomberg Bahntechnik

DU HAST ZUGRIFF AUF TOOLS:
- search_documents: Durchsucht das Projektarchiv nach Dokumenten
- read_document: Liest ein Dokument vollständig (Volltext)

ARBEITSWEISE:
1. Analysiere die Frage und entscheide welche Tools du brauchst
2. Suche zuerst mit search_documents nach relevanten Dokumenten
3. Wenn nötig, lies einzelne Dokumente mit read_document im Detail
4. Du kannst mehrere Suchen durchführen um verschiedene Aspekte abzudecken
5. Wenn du genug Informationen hast, antworte dem Benutzer

ANTWORT-FORMAT:
- Antworte auf Deutsch
- Starte DIREKT mit Fakten – KEINE Einleitung wie "Basierend auf..."
- Zitiere jede Aussage mit [Pfad] oder [N]
- Nutze Aufzählungen und kurze Absätze
- Sei gründlich und vollständig

WICHTIG: Nutze die Tools aktiv! Vermute nicht – suche und lies stattdessen."""

# ---------------------------------------------------------------------------
# Tool Execution
# ---------------------------------------------------------------------------

async def _execute_search(args: dict) -> str:
    """Execute search_documents tool"""
    query = args.get("query", "")
    if not query:
        return "Fehler: Kein Suchbegriff angegeben."
    
    from .tools import Tools
    from .rag_pipeline import SimpleRAGPipeline
    
    tools = Tools()
    pipeline = SimpleRAGPipeline()
    
    # Apply glossary rewrite
    from .glossary import rewrite_query
    rewritten, _ = rewrite_query(query)
    
    # Hybrid search
    result = await asyncio.to_thread(
        tools.search_hybrid,
        query=rewritten,
        es_size=40
    )
    
    hits = result.get("merged_hits", [])
    
    # Normalize and rank
    normalized = []
    for h in hits:
        path = h.get("file", {}).get("path", "")
        snippet = h.get("snippet", "")
        score = h.get("score", 0)
        normalized.append({"path": path, "snippet": snippet, "score": score})
    
    ranked = pipeline._rank_hits(normalized, query)
    
    # Format top results for the LLM
    top_n = ranked[:10]
    if not top_n:
        return f"Keine Treffer für '{query}'."
    
    parts = [f"Suche '{query}': {len(ranked)} Treffer. Top {len(top_n)}:\n"]
    for i, h in enumerate(top_n, 1):
        snippet = h.get("snippet", "")[:500]
        parts.append(f"[{i}] {h['path']}\n{snippet}\n")
    
    return "\n".join(parts)


async def _execute_read_document(args: dict) -> str:
    """Execute read_document tool"""
    path = args.get("path", "")
    if not path:
        return "Fehler: Kein Dokumentpfad angegeben."
    
    from .source_analyzer import fetch_document_text
    
    content, metadata = await fetch_document_text(path)
    
    if not content:
        return f"Dokument nicht gefunden: {path}"
    
    # Truncate very long documents
    max_chars = 12000
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[... gekürzt, {len(content)} Zeichen total]"
    
    return f"=== {path} ===\n{content}"


TOOL_EXECUTORS = {
    "search_documents": _execute_search,
    "read_document": _execute_read_document,
}

# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

class ReactAgent:
    """
    Autonomous agent with tool-calling loop.
    
    Uses Ollama's native tool-calling format. Falls back to
    direct answer if model doesn't support tool calling.
    """
    
    def __init__(self, model: str = None, ollama_base: str = None):
        self.model = model or os.getenv("OLLAMA_MODEL_ANSWER", "llama4:latest")
        self.ollama_base = (ollama_base or os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")).rstrip("/")
        self.max_steps = 6
    
    async def run(
        self,
        query: str,
        chat_history: list = None,
        system_prompt_extra: str = "",
        max_steps: int = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Run the ReAct loop. Yields dicts:
            {"type": "phase", "content": "🔍 Suche..."}
            {"type": "tool_call", "name": "search_documents", "args": {...}}
            {"type": "tool_result", "name": "search_documents", "summary": "10 Treffer"}
            {"type": "token", "content": "..."}  # Final answer streaming
            {"type": "sources", "sources": [...]}
            {"type": "done"}
        """
        max_steps = max_steps or self.max_steps
        
        # Build initial messages
        system_content = REACT_SYSTEM_PROMPT
        if system_prompt_extra:
            system_content += "\n\n" + system_prompt_extra
        
        messages = [{"role": "system", "content": system_content}]
        
        # Add chat history (last 3 turns)
        if chat_history:
            messages.extend(chat_history[-6:])
        
        messages.append({"role": "user", "content": query})
        
        # Collect sources for linking
        all_sources = []
        
        # --- ReAct Loop: non-streaming tool steps ---
        for step in range(max_steps):
            print(f"🤖 ReAct step {step+1}/{max_steps}")
            
            # Call LLM with tools (non-streaming)
            response = await self._llm_with_tools(messages)
            
            tool_calls = response.get("message", {}).get("tool_calls")
            content = response.get("message", {}).get("content", "")
            
            if tool_calls:
                # LLM wants to use a tool
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    tool_args = func.get("arguments", {})
                    
                    # Parse arguments if string
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except:
                            tool_args = {"query": tool_args}
                    
                    print(f"🔧 Tool call: {tool_name}({tool_args})")
                    yield {"type": "phase", "content": self._phase_label(tool_name, tool_args)}
                    yield {"type": "tool_call", "name": tool_name, "args": tool_args}
                    
                    # Execute tool
                    executor = TOOL_EXECUTORS.get(tool_name)
                    if executor:
                        try:
                            result = await executor(tool_args)
                            # Collect sources from search results
                            if tool_name == "search_documents":
                                all_sources.extend(self._extract_sources(result))
                        except Exception as e:
                            result = f"Fehler bei {tool_name}: {str(e)}"
                            print(f"❌ Tool error: {e}")
                    else:
                        result = f"Unbekanntes Tool: {tool_name}"
                    
                    summary = f"{len(result)} Zeichen" if len(result) > 100 else result[:100]
                    yield {"type": "tool_result", "name": tool_name, "summary": summary}
                    
                    # Add assistant tool_call + tool result to conversation
                    messages.append({
                        "role": "assistant",
                        "content": content or "",
                        "tool_calls": [tc]
                    })
                    messages.append({
                        "role": "tool",
                        "content": result
                    })
            else:
                # LLM wants to answer directly (no tool call)
                # If we already have tool context, stream the final answer
                if step > 0:
                    # Re-stream: the content from the non-streaming call
                    # is already the final answer, but we want streaming.
                    # So we do one more streaming call with the full context.
                    yield {"type": "phase", "content": "✍️ Erstelle Antwort...\n\n"}
                    async for token in self._llm_stream_final(messages):
                        yield {"type": "token", "content": token}
                elif content:
                    # First step, model answered directly (no tools used)
                    # This happens with models that don't support tool calling
                    # or for simple questions. Stream the response.
                    yield {"type": "phase", "content": ""}
                    async for token in self._llm_stream_final(messages):
                        yield {"type": "token", "content": token}
                else:
                    # Empty response - fallback to streaming
                    async for token in self._llm_stream_final(messages):
                        yield {"type": "token", "content": token}
                
                # Yield sources
                if all_sources:
                    yield {"type": "sources", "sources": all_sources}
                
                yield {"type": "done"}
                return
        
        # Max steps reached - generate final answer with what we have
        yield {"type": "phase", "content": "✍️ Maximale Schritte erreicht, erstelle Antwort...\n\n"}
        async for token in self._llm_stream_final(messages):
            yield {"type": "token", "content": token}
        
        if all_sources:
            yield {"type": "sources", "sources": all_sources}
        yield {"type": "done"}
    
    # ------------------------------------------------------------------
    # LLM Calls
    # ------------------------------------------------------------------
    
    async def _llm_with_tools(self, messages: list) -> dict:
        """Non-streaming LLM call with tool definitions"""
        import httpx
        
        # Estimate tokens for dynamic context window
        total_chars = sum(len(m.get("content", "")) for m in messages)
        est_tokens = total_chars // 3
        num_ctx = max(4096, est_tokens + 8192 + 512)
        num_ctx = min(num_ctx, 131072)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
            "options": {
                "num_ctx": num_ctx,
                "temperature": 0.2,
            }
        }
        
        timeout = 120.0
        if total_chars > 20000:
            timeout = 300.0
        
        print(f"🔧 ReAct LLM: {total_chars} chars, num_ctx={num_ctx}, model={self.model}")
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
            r = await client.post(f"{self.ollama_base}/api/chat", json=payload)
            r.raise_for_status()
            return r.json()
    
    async def _llm_stream_final(self, messages: list) -> AsyncGenerator[str, None]:
        """Streaming LLM call for final answer (no tools)"""
        import httpx
        
        total_chars = sum(len(m.get("content", "")) for m in messages)
        est_tokens = total_chars // 3
        num_ctx = max(4096, est_tokens + 8192 + 512)
        num_ctx = min(num_ctx, 131072)
        
        # Remove tool_calls from messages for clean streaming
        clean_messages = []
        for m in messages:
            clean = {"role": m["role"], "content": m.get("content", "")}
            clean_messages.append(clean)
        
        payload = {
            "model": self.model,
            "messages": clean_messages,
            "stream": True,
            "options": {
                "num_ctx": num_ctx,
                "temperature": 0.3,
                "num_predict": 8192,
            }
        }
        
        timeout = 300.0
        if total_chars > 20000:
            timeout = 600.0
        
        print(f"🔧 ReAct stream: {total_chars} chars, num_ctx={num_ctx}, model={self.model}")
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
            async with client.stream(
                "POST",
                f"{self.ollama_base}/api/chat",
                json=payload
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        content = obj.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except:
                        pass
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    
    def _phase_label(self, tool_name: str, args: dict) -> str:
        """Human-readable phase label for UI"""
        if tool_name == "search_documents":
            q = args.get("query", "")[:60]
            return f"🔍 Suche: *{q}*...\n\n"
        elif tool_name == "read_document":
            p = args.get("path", "").split("/")[-1][:50]
            return f"📄 Lese: *{p}*...\n\n"
        return f"🔧 {tool_name}...\n\n"
    
    def _extract_sources(self, search_result: str) -> list:
        """Extract source paths from search result text"""
        import re
        sources = []
        file_base = os.getenv("FILE_BASE", "/media/felix/RAG/1")
        
        for match in re.finditer(r'\[(\d+)\]\s+(.+?)(?:\n|$)', search_result):
            n = int(match.group(1))
            path = match.group(2).strip()
            if path:
                from urllib.parse import quote
                encoded = quote(f"{file_base}/{path}", safe="/:@")
                sources.append({
                    "n": n,
                    "path": path,
                    "display_path": f"/{path}",
                    "local_url": f"http://localhost:11436/open?path={encoded}"
                })
        
        return sources
