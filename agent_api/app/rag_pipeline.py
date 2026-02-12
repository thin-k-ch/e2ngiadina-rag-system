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
    RAG_ANSWER_TEMPERATURE,
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
        
        # Extract query terms for relevance boosting (primary signal)
        query_terms = [t.strip().lower() for t in query_lower.split() if len(t.strip()) >= 3]
        
        def relevance_score(hit):
            path = hit.get("path", "").lower()
            snippet = hit.get("snippet", "").lower()
            base_score = hit.get("score", 0)
            
            boost = 0
            
            # PRIMARY: Query term matching (strongest signal)
            query_match_count = 0
            for qt in query_terms:
                if qt in path:
                    boost += 20  # Strong boost for query term in path
                    query_match_count += 1
                if qt in snippet:
                    boost += 10  # Good boost for query term in snippet
                    query_match_count += 1
            
            # Bonus for multiple query terms matching
            if query_match_count >= 2:
                boost += 15
            
            # SECONDARY: Domain keyword matching (weaker signal)
            keyword_count = 0
            for kw in keywords:
                if kw in path:
                    boost += path_boost
                    keyword_count += 1
                if kw in snippet:
                    boost += snippet_boost
                    keyword_count += 1
            
            if keyword_count >= 2:
                boost += compound_bonus
            
            # PENALTY: Excel files
            if path.endswith(('.xlsx', '.xls')):
                excel_relevant = any(kw in path for kw in RAG_EXCEL_RELEVANT_KEYWORDS)
                boost += excel_penalty_rel if excel_relevant else excel_penalty_irrel
            
            # BONUS: PDF/MSG/DOCX preferred
            if path.endswith(('.pdf', '.msg', '.docx')):
                boost += pdf_msg_bonus
                
            return base_score + boost
        
        # Compute and store scores
        for hit in hits:
            hit["relevance_score"] = relevance_score(hit)
        
        # Sort by boosted relevance
        sorted_hits = sorted(hits, key=lambda h: h["relevance_score"], reverse=True)
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
        
        system_prompt = """DU BIST EIN DOKUMENTEN-ANALYST FÜR SCHWEIZER EISENBAHN-PROJEKTE (SBB TFK 2020 - Tunnelfunk).
Fachgebiete: Projektleitung, Programmleitung, Funktechnik, Tunnelfunk.

FACHBEGRIFFE: FAT=Werksabnahme, SAT=Standortabnahme, TFK=Tunnelfunk, GBT=Gotthard Basistunnel, RBT=Rhomberg Bahntechnik

DEINE AUFGABE: Extrahiere konkrete Fakten aus den Dokumenten und präsentiere sie strukturiert.

ANTWORT-FORMAT (ZWINGEND):
1. Antworte auf Deutsch
2. Starte DIREKT mit Fakten aus den Dokumenten - KEINE Einleitung
3. Zitiere jede Aussage mit [N] (Quellennummer)
4. Nutze Aufzählungen und kurze Absätze
5. Wenn Dokumente Informationen enthalten, ZITIERE sie - sage NIEMALS "die Dokumente sind allgemein"

BEISPIEL einer korrekten Antwort:
---
Laut [1] findet das Supplier Board monatlich statt. Teilnehmer sind gemäss [2]:
- Projektleitung SBB
- Rhomberg Bahntechnik (Lieferant)
Hauptthemen laut [3]: Lieferstatus, offene Punkte, Terminplan.
---

WENN KEINE relevanten Dokumente: Sage "Keine relevanten Dokumente gefunden."
"""

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        if context:
            messages.append({"role": "system", "content": f"DOKUMENT-KONTEXT:\n{context}"})
        messages.append({"role": "user", "content": query})
        
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
        # Apply glossary query rewrite for domain disambiguation
        from .glossary import rewrite_query
        rewritten_query, query_meta = rewrite_query(query)
        if rewritten_query != query:
            print(f"📝 Query rewritten: '{query}' -> '{rewritten_query[:80]}...'")
            print(f"   Expansions: {query_meta.get('expansions', [])}")
        query = rewritten_query  # Use expanded query
        
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
                # Build sources from RANKED hits with dynamic relevance cutoff
                file_base = os.getenv("FILE_BASE", "/media/felix/RAG/1")
                from urllib.parse import quote
                
                # Dynamic source filtering - two tiers:
                # Tier 1: Query terms in PATH (document is about the topic) → all shown
                # Tier 2: Query terms only in snippet (mentions topic) → max 5
                query_terms = [t.strip().lower() for t in query.lower().split() if len(t.strip()) >= 3]
                max_snippet_only = 5  # Max sources that only match in snippet
                
                tier1 = []  # Path matches
                tier2 = []  # Snippet-only matches
                
                for hit in ranked_hits:
                    path = hit.get("path", hit.get("file", {}).get("path", ""))
                    snippet = hit.get("snippet", "")
                    path_lower = path.lower()
                    snippet_lower = snippet.lower()
                    
                    in_path = any(qt in path_lower for qt in query_terms) if query_terms else False
                    in_snippet = any(qt in snippet_lower for qt in query_terms) if query_terms else True
                    
                    if in_path:
                        tier1.append(hit)
                    elif in_snippet:
                        tier2.append(hit)
                
                # Combine: all path matches + limited snippet matches
                relevant_hits = tier1 + tier2[:max_snippet_only]
                relevant_hits = relevant_hits[:max_sources]
                
                sources = []
                for i, hit in enumerate(relevant_hits, 1):
                    path = hit.get("path", hit.get("file", {}).get("path", ""))
                    display_path = path.replace("/media/felix/RAG/1", "")
                    full_path = os.path.join(file_base, path.lstrip("/"))
                    url = f"http://localhost:11436/open?path={quote(full_path)}"
                    
                    sources.append({
                        "n": i,
                        "path": path,
                        "display_path": display_path,
                        "local_url": url,
                        "score": round(hit.get("relevance_score", 0), 4)
                    })
                
                print(f"📊 Sources: {len(sources)} (path={len(tier1)}, snippet={len(tier2[:max_snippet_only])}) von {len(ranked_hits)} total")
                
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
