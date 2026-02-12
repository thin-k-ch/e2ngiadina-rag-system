"""
agent_streaming_final.py

Drop-in replacement for agent_api/app/agent.py.

Goals:
- Never crash on unexpected hit formats ("str has no attribute get").
- Provide deterministic [TRACE] first, then [FINAL] with real token streaming.
- Keep API compatible with app.main: Agent.answer(...) -> (answer, new_summary, new_notes, sources)

Notes:
- This intentionally avoids importing non-existent symbols (e.g. format_sources_markdown).
- Uses Tools.search_hybrid() if available; otherwise returns empty context.
- Uses Ollama /api/chat for both non-stream and streaming.
"""

from __future__ import annotations

import os
import json
import time
import asyncio
from typing import Any, Dict, List, Tuple, AsyncGenerator, Optional

import httpx
import sys

# Ensure /app is in path for glossary import
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

# Optional imports from the repo. Keep the agent resilient if some modules drift.
try:
    from .tools import Tools  # type: ignore
except Exception:
    Tools = None  # type: ignore

try:
    from .format_links import make_clickable_path  # type: ignore
except Exception:
    make_clickable_path = None  # type: ignore

try:
    from app.glossary import rewrite_query
    print(f"✅ Glossary import OK: {rewrite_query is not None}")
except Exception as e:
    print(f"❌ Glossary import failed: {e}")
    rewrite_query = None


def _now() -> int:
    return int(time.time())


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return ""


def _safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def _normalize_hit(hit: Any) -> Dict[str, Any]:
    """
    Normalizes a retrieval hit into:
      {
        "path": str,
        "text": str,
        "score": float|int|None,
        "metadata": dict,
      }
    Accepts dict hits (preferred) OR str (treated as text).
    """
    if isinstance(hit, str):
        return {"path": "", "text": hit, "score": None, "metadata": {}}
    if not isinstance(hit, dict):
        return {"path": "", "text": _safe_str(hit), "score": None, "metadata": {}}

    md = _safe_get(hit, "metadata", {}) or {}
    md = md if isinstance(md, dict) else {}

    # Common keys across variants
    path = _safe_get(hit, "path", "") or _safe_get(md, "path", "") or _safe_get(md, "source", "")
    text = _safe_get(hit, "text", "") or _safe_get(hit, "content", "") or _safe_get(md, "text", "")

    # Sometimes text is nested / not a string
    if not isinstance(text, str):
        text = _safe_str(text)

    score = _safe_get(hit, "score", None)
    return {
        "path": _safe_str(path),
        "text": text,
        "score": score,
        "metadata": md,
    }


def _dedupe_hits(hits: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for h in hits:
        path = (h.get("path") or "").strip()
        # primary key: path + first 80 chars of text
        k = (path, (h.get("text") or "")[:80])
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
        if len(out) >= limit:
            break
    return out


def _build_sources(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    file_base = os.getenv("FILE_BASE", "") or ""
    sources: List[Dict[str, Any]] = []
    n = 1
    for h in hits:
        # Handle both old format (h.get("path")) and new format (h.get("file", {}).get("path"))
        if isinstance(h, str):
            # Old format: string hit
            path = h.strip()
            score = 0
            snippet = h
            file_info = {}
        else:
            # New format: dictionary hit
            path = (h.get("file", {}).get("path") or "").strip()
            if not path:
                path = (h.get("path") or "").strip()
            
            if not path:
                continue
                
            score = h.get("score", 0)
            snippet = h.get("snippet", "")
            file_info = h.get("file", {})
        
        if not path:
            continue
            
        display_path = path
        url = ""
        file_url = ""  # file:// URL for local opening
        
        # DEBUG
        import sys
        sys.stderr.write(f"[DEBUG] Processing path: {path}, file_base: {file_base}\n")
        
        if make_clickable_path:
            display_path, url = make_clickable_path(path, file_base=file_base, use_http_proxy=True)
            sys.stderr.write(f"[DEBUG] make_clickable_path returned: {url}\n")
        else:
            # fallback: still try to use /open endpoint with URL encoding
            try:
                from urllib.parse import quote
                full_path = path
                if file_base and not path.startswith("/"):
                    full_path = os.path.join(file_base, path)
                url = f"http://localhost:11436/open?path={quote(full_path)}"
                sys.stderr.write(f"[DEBUG] Fallback URL: {url}\n")
            except Exception as e:
                sys.stderr.write(f"[DEBUG] URL generation error: {e}\n")
                url = ""
        
        # Create file:// URL for local file opening
        try:
            from urllib.parse import quote
            full_path = path
            if file_base and not path.startswith("/"):
                full_path = os.path.join(file_base, path)
            # Ensure absolute path for file:// URL
            if not full_path.startswith("/"):
                full_path = "/" + full_path
            file_url = f"file://{quote(full_path)}"
        except Exception:
            file_url = ""
        
        sources.append({
            "n": n,
            "path": path,
            "display_path": display_path,
            "local_url": url,
            "file_url": file_url,  # file:// URL for clicking
            "score": score,
            "snippet": snippet,
            "file": file_info,
        })
        n += 1
    return sources


def _format_context(hits: List[Dict[str, Any]], max_chars: int = 9000) -> str:
    """
    Builds a compact context block from top hits.
    """
    parts: List[str] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        # Handle both old format (h.get("text")) and new format (h.get("snippet"))
        if isinstance(h, str):
            # Old format: string hit
            txt = h.strip()
            path = ""
        else:
            # New format: dictionary hit
            txt = (h.get("snippet") or "").strip()
            path = (h.get("path") or "").strip()
            if not path:
                path = (h.get("file", {}).get("path") or "").strip()
        
        if not txt:
            continue
            
        header = f"[{i}] {path}" if path else f"[{i}]"
        block = header + "\n" + txt
        if used + len(block) + 2 > max_chars:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


class Agent:
    def __init__(self):
        self.ollama_base = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
        self.model_default = os.getenv("OLLAMA_MODEL", "llama4:latest")
        self.trace_enabled = os.getenv("AGENT_TRACE", "1").lower() not in ("0", "false", "no")
        # Retrieval
        self.tools = Tools() if Tools else None

    # --------------------------
    # Ollama calls
    # --------------------------
    async def llm_text(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.2) -> str:
        payload = {
            "model": model or self.model_default,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
            r = await client.post(f"{self.ollama_base}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        return _safe_get(_safe_get(data, "message", {}), "content", "")

    async def llm_stream(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.2) -> AsyncGenerator[str, None]:
        """
        Yields incremental token strings.
        """
        payload = {
            "model": model or self.model_default,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
            async with client.stream("POST", f"{self.ollama_base}/api/chat", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    msg = _safe_get(obj, "message", {})
                    chunk = _safe_get(msg, "content", "")
                    if chunk:
                        yield chunk

    # --------------------------
    # Retrieval with Policy
    # --------------------------
    async def _retrieve_with_policy(self, user_text: str) -> Dict[str, Any]:
        """
        New retrieval method using ChatGPT's Policy:
        - Tool-Gate: decide_gate() 
        - Exact Phrase: search_exact_phrase()
        - Hybrid: search_hybrid()
        - Stop Rules: max 2 rounds
        - Guardrails: can_claim_absence()
        """
        import sys
        sys.stderr.write(f"[DEBUG] _retrieve_with_policy called with: {user_text}\n")
        sys.stderr.flush()
        
        if not self.tools:
            sys.stderr.write("[DEBUG] NO TOOLS!\n")
            sys.stderr.flush()
            return {"mode": "no_tools", "hits": [], "total_hits": 0}

        # Apply glossary rewrite FIRST for domain knowledge
        search_text = user_text
        try:
            import sys
            if '/app' not in sys.path:
                sys.path.insert(0, '/app')
            from app.glossary import rewrite_query as g_rewrite
            rewritten, meta = g_rewrite(user_text)
            if meta.get('expansions'):
                sys.stderr.write(f"🔧 Glossary rewrite: '{user_text}' -> '{rewritten[:80]}...'\n")
                sys.stderr.flush()
                search_text = rewritten
        except Exception as e:
            sys.stderr.write(f"⚠️ Glossary error: {e}\n")
            sys.stderr.flush()

        # Step 1: Tool-Gate
        gate = self.tools.decide_gate(search_text)
        print(f"🚪 TOOL-GATE: {gate.mode} - {gate.reason}")

        if not gate.require_rag:
            return {"mode": "no_rag", "hits": [], "total_hits": 0}

        # Step 2: Execute search based on mode
        if gate.mode == "exact_phrase":
            result = self.tools.search_exact_phrase(gate.phrase or search_text)
            print(f"🎯 EXACT PHRASE RESULT: {result['total_hits']} hits")
            return result
        elif gate.mode == "hybrid":
            result = self.tools.search_hybrid(search_text)
            print(f"🎯 HYBRID RESULT: {len(result.get('merged_hits', []))} hits")
            return result
        else:
            return {"mode": "no_rag", "hits": [], "total_hits": 0}

    async def _retrieve(self, user_text: str) -> List[Dict[str, Any]]:
        """
        Legacy method - now uses policy-based retrieval
        """
        result = await self._retrieve_with_policy(user_text)
        
        # Convert to legacy format
        if result.get("mode") == "exact_phrase":
            return result.get("best_hits", [])
        elif result.get("mode") == "hybrid":
            return result.get("merged_hits", [])
        else:
            return []

    # --------------------------
    # Public API used by app.main
    # --------------------------
    async def answer(
        self,
        user_text: str,
        raw_messages: Optional[List[Dict[str, Any]]] = None,
        summary: str = "",
        notes: str = "",
    ) -> Tuple[str, str, str, List[Dict[str, Any]]]:
        """
        Non-streaming answer for /v1/chat/completions with stream=false.
        Returns: (answer_text, new_summary, new_notes, sources)
        """
        import sys
        sys.stderr.write(f"[DEBUG] answer() called with: {user_text[:50]}\n")
        sys.stderr.flush()
        
        hits = await self._retrieve_with_policy(user_text)
        
        # Apply guardrails
        if isinstance(hits, dict) and hits.get("mode") == "exact_phrase":
            if not self.tools.can_claim_absence("exact_phrase", True, hits.get("total_hits", 0), 1):
                # If we can't claim absence, we must have evidence
                pass  # hits already contain evidence
            else:
                # We can claim absence - but only if truly 0 hits
                if hits.get("total_hits", 0) == 0:
                    return "0 Treffer in rag_files_v1 (exact phrase search)", "", "", []
        
        # Convert to legacy format for _build_sources
        if isinstance(hits, dict):
            if hits.get("mode") == "exact_phrase":
                legacy_hits = hits.get("best_hits", [])
            elif hits.get("mode") == "hybrid":
                legacy_hits = hits.get("merged_hits", [])
            else:
                legacy_hits = []
        else:
            legacy_hits = []
        
        sources = _build_sources(legacy_hits)
        
        print(f"🔍 DEBUG: legacy_hits count={len(legacy_hits)}")
        if legacy_hits:
            print(f"🔍 DEBUG: first hit keys={list(legacy_hits[0].keys())}")
            print(f"🔍 DEBUG: first hit snippet={legacy_hits[0].get('snippet', 'NO SNIPPET')[:100]}")

        context = _format_context(legacy_hits)
        print(f"🔍 DEBUG: context length={len(context)}")
        print(f"🔍 DEBUG: context preview={context[:200]}")
        sys = (
            "DU BIST EIN RAG-AGENT. KORREKTE ANTWORT-STRUKTUR:\n"
            "1. Starte DIREKT mit: 'Laut Dokument [N]:' gefolgt vom Zitat\n"
            "2. MAXIMAL 2-3 konkrete Zitate aus dem Kontext\n"
            "3. Quellen-Liste am Ende: [1] [Pfad](URL)\n"
            "\n"
            "VERBOT - Diese Phrasen sind STRENG UNTERAGT:\n"
            "- 'Ich habe...'\n"
            "- 'Hier sind...'\n"
            "- 'Es scheint...'\n"
            "- 'Wenn Sie...'\n"
            "\n"
            "BEISPIEL für korrekte Antwort:\n"
            "Laut Dokument [1]: 'Projektleitung Konzepthase umfasst Planung und Steuerung.'\n"
            "Laut Dokument [3]: 'Verantwortlichkeiten in Phase 1: Konzeptentwicklung.'\n"
            "\n"
            "Quellen:\n"
            "[1] [/Pfad/zum/Dokument.pdf](http://localhost:11436/open?path=...)\n"
            "[3] [/Pfad/zum/Dokument.pdf](http://localhost:11436/open?path=...)\n"
            "\n"
            "WENN keine Treffer: 'Keine Dokumente mit [Suchbegriff] gefunden.'\n"
        )

        msgs: List[Dict[str, str]] = [{"role": "system", "content": sys}]
        if summary:
            msgs.append({"role": "system", "content": f"Konversations-Zusammenfassung:\n{summary}"})
        if notes:
            msgs.append({"role": "system", "content": f"Notizen:\n{notes}"})
        if context:
            msgs.append({"role": "system", "content": f"RETRIEVED CONTEXT:\n{context}"})
        msgs.append({"role": "user", "content": user_text})

        # Skip slow LLM for now - just return search results summary
        if not context:
            return "Keine Dokumente gefunden.", summary, notes, sources
        
        # Quick summary of findings with clickable links
        answer_lines = ["Gefundene Dokumente:"]
        for i, s in enumerate(sources[:5], 1):
            path = s.get('display_path', s.get('path', ''))
            url = s.get('local_url', '')
            if url:
                answer_lines.append(f"[{i}] [{path}]({url})")
            else:
                answer_lines.append(f"[{i}] {path}")
        
        out = "\n".join(answer_lines)

        # Append sources markdown with HTTP links (file:// blocked by browsers)
        if sources:
            lines = ["", "Quellen:"]
            for s in sources:
                dp = s.get("display_path", s.get("path", ""))
                url = s.get("local_url", "")  # HTTP URL to /open endpoint
                n = s.get("n", "?")
                if url:
                    lines.append(f"[{n}] [{dp}]({url})")
                else:
                    lines.append(f"[{n}] {dp}")
            out = out.rstrip() + "\n" + "\n".join(lines) + "\n"
        elif hits.get("mode") == "exact_phrase" and hits.get("total_hits", 0) == 0:
            out += "\n\n0 Treffer in rag_files_v1 (exact phrase search)"

        # Keep summary/notes stable for now (no auto-summarize in this safe build)
        return out, summary, notes, sources

    async def answer_stream(
        self,
        user_text: str,
        raw_messages: Optional[List[Dict[str, Any]]] = None,
        summary: str = "",
        notes: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming generator that yields dict events:
          {"type":"trace","content": "..."} and {"type":"token","content":"..."} and {"type":"final_sources","sources":[...]}
        app.main should map these to SSE chunks.
        """
        # 1) TRACE immediately
        if self.trace_enabled:
            yield {"type": "trace", "content": "[TRACE]\n"}
            await asyncio.sleep(0)

        # 2) Retrieval step
        hits: List[Dict[str, Any]] = []
        try:
            hits = await self._retrieve(user_text)
        except Exception as e:
            if self.trace_enabled:
                yield {"type": "trace", "content": f"[TRACE] Retrieval error: {_safe_str(e)}\n"}
            hits = []

        if self.trace_enabled:
            yield {"type": "trace", "content": f"- retrieval_hits: {len(hits)}\n"}
            await asyncio.sleep(0)

        sources = _build_sources(hits)
        context = _format_context(hits)

        # 3) Compose prompt
        sys = (
            "Du bist ein lokaler RAG-Agent. Antworte auf Deutsch.\n"
            "Wenn Kontext vorhanden ist, stütze dich darauf.\n"
            "Gib am Ende eine kurze Quellenliste (lokal) aus.\n"
            "Wichtig: Schreibe KEINE internen Gedankengänge (kein chain-of-thought), "
            "nur eine knappe, nachvollziehbare Antwort.\n"
        )
        msgs: List[Dict[str, str]] = [{"role": "system", "content": sys}]
        if summary:
            msgs.append({"role": "system", "content": f"Konversations-Zusammenfassung:\n{summary}"})
        if notes:
            msgs.append({"role": "system", "content": f"Notizen:\n{notes}"})
        if context:
            msgs.append({"role": "system", "content": f"RETRIEVED CONTEXT:\n{context}"})
        msgs.append({"role": "user", "content": user_text})

        # 4) Stream tokens
        if self.trace_enabled:
            yield {"type": "trace", "content": "[/TRACE]\n[FINAL]\n"}
            await asyncio.sleep(0)

        try:
            async for tok in self.llm_stream(msgs):
                yield {"type": "token", "content": tok}
        except Exception as e:
            yield {"type": "token", "content": f"\nFehler beim Streaming: {_safe_str(e)}\n"}

        # 5) Append sources with HTTP links (browser-compatible)
        if sources:
            lines = ["", "\nQuellen:"]
            for s in sources:
                dp = s.get("display_path", s.get("path", ""))
                url = s.get("local_url", "")  # HTTP URL
                n = s.get("n", "?")
                if url:
                    lines.append(f"[{n}] [{dp}]({url})")
                else:
                    lines.append(f"[{n}] {dp}")
            yield {"type": "token", "content": "\n" + "\n".join(lines) + "\n"}

        yield {"type": "final_sources", "sources": sources}
