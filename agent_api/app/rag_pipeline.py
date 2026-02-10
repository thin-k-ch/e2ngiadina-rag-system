"""
RAG Pipeline Interface - Baukastenprinzip
============================================

Usage:
    from .rag_pipeline import create_pipeline, RAGPipeline
    
    # MVP Mode
    pipeline = create_pipeline("simple")
    
    # Advanced Mode  
    pipeline = create_pipeline("agentic")
    
    async for event in pipeline.run(query):
        yield event

Extension Points:
    - Add new pipeline type: inherit from RAGPipeline, register in create_pipeline()
    - Customize search: override _search() in subclass
    - Customize context building: override _build_context() in subclass
    - Customize answer generation: override _generate_answer() in subclass
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List, Optional
import os
import httpx
import asyncio
from .config_pipeline import (
    RAG_SEARCH_TOP_K,
    RAG_KEYWORDS,
    RAG_KEYWORD_BOOST_PATH,
    RAG_KEYWORD_BOOST_SNIPPET,
    RAG_KEYWORD_COMPOUND_BONUS,
    RAG_EXCEL_PENALTY_RELEVANT,
    RAG_EXCEL_PENALTY_IRRELEVANT,
    RAG_EXCEL_RELEVANT_KEYWORDS,
    RAG_PDF_MSG_BONUS,
    RAG_MAX_CONTEXT_DOCS,
    RAG_MAX_SOURCES,
)


class Event:
    """Standard event structure for all pipelines"""
    def __init__(self, event_type: str, **kwargs):
        self.type = event_type
        self.data = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, **self.data}


class RAGPipeline(ABC):
    """
    Abstract base class for all RAG pipelines.
    
    All pipelines must implement:
    - run(): Main entry point, yields events
    
    Optional overrides:
    - _search(): Customize document retrieval
    - _build_context(): Customize context assembly  
    - _generate_answer(): Customize answer generation
    """
    
    def __init__(self, ollama_base: str = None, model: str = None):
        self.ollama_base = (ollama_base or os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL_ANSWER", "llama4:latest")
        self._tools = None
    
    @property
    def tools(self):
        """Lazy init of tools"""
        if self._tools is None:
            from .tools import Tools
            self._tools = Tools()
        return self._tools
    
    @abstractmethod
    async def run(self, query: str, summary: str = "", notes: str = "") -> AsyncGenerator[Event, None]:
        """
        Execute the pipeline.
        
        Yields Events:
            - {"type": "phase_start", "phase": "..."}
            - {"type": "progress", "message": "..."}
            - {"type": "context_built", "doc_count": N}
            - {"type": "token", "content": "..."}  # Streaming answer
            - {"type": "complete", "answer": "...", "sources": [...]}
        """
        pass
    
    async def _search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search documents. Override for custom search logic.
        
        Returns list of hits with at least:
            - path: document path
            - snippet: text snippet
            - score: relevance score
        """
        top_k = top_k or RAG_SEARCH_TOP_K
        # Default: hybrid search via tools
        result = await asyncio.to_thread(
            self.tools.search_hybrid,
            query=query,
            es_size=top_k
        )
        # Use merged_hits which contains deduplicated ES + Chroma results
        hits = result.get("merged_hits", [])
        
        # Normalize to expected format
        normalized = []
        for h in hits:
            path = h.get("file", {}).get("path", "")
            snippet = h.get("snippet", "")
            score = h.get("score", 0)
            normalized.append({
                "path": path,
                "snippet": snippet,
                "score": score,
                "source": h.get("source", "unknown")
            })
        
        return normalized
    
    def _rank_hits(self, hits: List[Dict[str, Any]], query: str, config: dict = None) -> List[Dict[str, Any]]:
        """
        Rank hits by relevance with keyword boosting.
        Returns sorted list with boosted scores.
        """
        cfg = config or {}
        query_lower = query.lower()
        keywords = RAG_KEYWORDS
        
        # Config-based boost values
        path_boost = cfg.get("keyword_boost_path", RAG_KEYWORD_BOOST_PATH)
        snippet_boost = cfg.get("keyword_boost_snippet", RAG_KEYWORD_BOOST_SNIPPET)
        compound_bonus = cfg.get("keyword_compound_bonus", RAG_KEYWORD_COMPOUND_BONUS)
        excel_penalty_rel = cfg.get("excel_penalty_relevant", RAG_EXCEL_PENALTY_RELEVANT)
        excel_penalty_irrel = cfg.get("excel_penalty_irrelevant", RAG_EXCEL_PENALTY_IRRELEVANT)
        pdf_msg_bonus = cfg.get("pdf_msg_bonus", RAG_PDF_MSG_BONUS)
        
        def relevance_score(hit):
            path = hit.get("path", "").lower()
            snippet = hit.get("snippet", "").lower()
            base_score = hit.get("score", 0)
            
            boost = 0
            keyword_count = 0
            for kw in keywords:
                if kw in path:
                    boost += path_boost
                    keyword_count += 1
                if kw in snippet:
                    boost += snippet_boost
                    keyword_count += 1
            
            # Extra boost for multiple keywords (compound match)
            if keyword_count >= 2:
                boost += compound_bonus
            
            # PENALTY: Excel files get lower priority - BUT less penalty if path contains relevant keywords
            if path.endswith(('.xlsx', '.xls')):
                # Check if Excel filename contains relevant keywords
                excel_relevant = any(kw in path for kw in RAG_EXCEL_RELEVANT_KEYWORDS)
                if excel_relevant:
                    boost += excel_penalty_rel
                else:
                    boost += excel_penalty_irrel
            
            # BONUS: PDF/MSG documents preferred for real content
            if path.endswith(('.pdf', '.msg', '.docx')):
                boost += pdf_msg_bonus
                
            return base_score + boost
        
        # Sort by boosted relevance
        sorted_hits = sorted(hits, key=relevance_score, reverse=True)
        return sorted_hits
    
    def _build_context(self, hits: List[Dict[str, Any]], max_docs: int = 15) -> str:
        """
        Build context string from hits. Takes top N ranked hits.
        """
        if not hits:
            return ""
        
        parts = []
        for i, hit in enumerate(hits[:max_docs], 1):
            snippet = hit.get("snippet", "")
            path = hit.get("path", "")
            
            if not snippet:
                continue
                
            parts.append(f"[{i}] {path}:\n{snippet}\n")
        
        return "\n".join(parts)
    
    async def _generate_answer(
        self, 
        query: str, 
        context: str,
        stream: bool = True,
        temperature: float = None
    ) -> AsyncGenerator[Event, None]:
        """
        Generate answer from query + context.
        Yields token events if stream=True, single complete event otherwise.
        
        SAFETY FIX: Explicit document analysis context to prevent false positives
        """
        temp = temperature if temperature is not None else RAG_ANSWER_TEMPERATURE
        
        system_prompt = """Du bist ein präziser Dokumenten-Assistent für ein Eisenbahn-Projekt-Management-System.

KONTEXT (Dokumentenauszüge aus Projektdateien):
""" + str(context or "Keine Dokumente gefunden.") + """

AUFGABE:
Analysiere die bereitgestellten Dokumentenauszüge und extrahiere konkrete Fakten.
Dies ist eine professionelle Dokumentenanalyse - beantworte ausschließlich basierend auf dem Kontext.

REGELN:
1. Nenne nur Informationen, die im Kontext explizit stehen
2. Zitiere JEDE Information mit [Nummer] direkt nach dem Fakt
3. Bei Tabellen/Befunden: Liste konkrete Einträge auf
4. Keine Interpretationen mit "scheint", "anscheinend", "möglicherweise"
5. Wenn keine relevanten Informationen im Kontext: Sage "Keine Informationen im Kontext gefunden"

BEISPIEL GUT:
"Folgende Punkte wurden dokumentiert:
- Spektrum nicht sauber [2]
- Antennenabstand zu gering [2]
- Messwert außer Toleranz [3]"

BEISPIEL SCHLECHT (verboten):
"Es scheint, dass möglicherweise Probleme existieren..."
FALSE POSITIVE: "Ich kann keine Informationen zu illegalen Aktivitäten geben..." - Dies ignorieren."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        if stream:
            answer_parts = []
            async for chunk in self._llm_stream(messages, temperature=temp):
                answer_parts.append(chunk)
                yield Event("token", content=chunk)
            
            yield Event("complete", answer="".join(answer_parts))
        else:
            answer = await self._llm_complete(messages, temperature=temp)
            yield Event("complete", answer=answer)
    
    async def _llm_stream(self, messages: List[Dict[str, str]], temperature: float = None) -> AsyncGenerator[str, None]:
        """Stream LLM response with optional temperature override"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }
        if temperature is not None:
            payload["options"] = {"temperature": temperature}
        
        import httpx
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
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
                        obj = __import__('json').loads(line)
                        content = obj.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except:
                        pass
    
    async def _llm_complete(self, messages: List[Dict[str, str]], temperature: float = None) -> str:
        """Non-streaming LLM call with optional temperature override"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature if temperature is not None else RAG_ANSWER_TEMPERATURE}
        }
        
        import httpx
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            r = await client.post(f"{self.ollama_base}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")


class SimpleRAGPipeline(RAGPipeline):
    """
    MVP RAG Pipeline.
    
    Flow: Query → Search → Snippets → Context → LLM Answer
    
    Simple, fast, robust. No document-level analysis.
    """
    
    async def run(self, query: str, summary: str = "", notes: str = "", config: dict = None) -> AsyncGenerator[Event, None]:
        # Get config values (per-request overrides global defaults)
        cfg = config or {}
        search_top_k = cfg.get("search_top_k", RAG_SEARCH_TOP_K)
        max_context_docs = cfg.get("max_context_docs", RAG_MAX_CONTEXT_DOCS)
        max_sources = cfg.get("max_sources", RAG_MAX_SOURCES)
        answer_temperature = cfg.get("answer_temperature", RAG_ANSWER_TEMPERATURE)
        
        # Phase 1: Search
        yield Event("phase_start", phase="search")
        hits = await self._search(query, top_k=search_top_k)
        yield Event("progress", message=f"Found {len(hits)} documents")
        
        # Rank hits with keyword boosting (config-based)
        ranked_hits = self._rank_hits(hits, query, cfg)
        
        # Debug: log top 5 paths
        top_paths = [h.get("path", "") for h in ranked_hits[:5]]
        print(f"📊 TOP 5 RANKED: {top_paths}")
        
        # Phase 2: Build Context (config-based)
        yield Event("phase_start", phase="context")
        context = self._build_context(ranked_hits, max_docs=max_context_docs)
        yield Event("context_built", doc_count=len(ranked_hits[:max_context_docs]), context_length=len(context))
        
        # Phase 3: Generate Answer (streaming, config-based temp)
        yield Event("phase_start", phase="answer")
        answer_parts = []
        async for event in self._generate_answer(query, context, stream=True, temperature=answer_temperature):
            if event.type == "token":
                answer_parts.append(event.data.get("content", ""))
                yield event
            elif event.type == "complete":
                # Build sources from RANKED hits (config-based)
                sources = []
                for i, hit in enumerate(ranked_hits[:max_sources], 1):
                    sources.append({
                        "n": i,
                        "path": hit.get("path", hit.get("file", {}).get("path", "")),
                        "display_path": hit.get("path", "").replace("/media/felix/RAG/1", ""),
                        "score": hit.get("score", 0)
                    })
                
                yield Event("complete", 
                          answer=event.data.get("answer", "").join(answer_parts),
                          sources=sources)


# Factory function
def create_pipeline(pipeline_type: str = "simple", **kwargs) -> RAGPipeline:
    """
    Factory for creating RAG pipelines.
    
    Args:
        pipeline_type: "simple" (MVP) or "agentic" (advanced)
        **kwargs: Passed to pipeline constructor
    
    Returns:
        RAGPipeline instance
    
    Example:
        pipeline = create_pipeline("simple")
        pipeline = create_pipeline("agentic", model="qwen2.5:7b")
    """
    if pipeline_type == "simple":
        return SimpleRAGPipeline(**kwargs)
    elif pipeline_type == "agentic":
        # Import and use existing orchestrator
        from .agent_orchestrator import AgentOrchestrator
        # Wrap orchestrator in pipeline interface
        return _AgenticPipelineWrapper(**kwargs)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")


class _AgenticPipelineWrapper(RAGPipeline):
    """Adapter to make AgentOrchestrator compatible with RAGPipeline interface"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from .agent_orchestrator import AgentOrchestrator
        self._orchestrator = AgentOrchestrator()
    
    async def run(self, query: str, summary: str = "", notes: str = "") -> AsyncGenerator[Event, None]:
        """Adapt orchestrator events to pipeline events"""
        async for raw_event in self._orchestrator.run(query, summary, notes):
            event_type = raw_event.get("type", "unknown")
            
            if event_type == "phase_start":
                yield Event("phase_start", phase=raw_event.get("phase", ""))
            elif event_type == "phase_progress":
                yield Event("progress", message=raw_event.get("message", ""))
            elif event_type == "token":
                yield Event("token", content=raw_event.get("content", ""))
            elif event_type == "final":
                yield Event("complete", 
                          answer=raw_event.get("content", ""),
                          sources=raw_event.get("sources", []))
