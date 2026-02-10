"""
agent_orchestrator.py

Multi-Stage Agentic RAG Orchestrator with 5 Phases:
1. Strategy Agent (7B) - JSON Search-Spec generation
2. Retrieval Agent - ES + Chroma with pre-validation
3. Document Analysis Agent - PDF/DOCX/EML reading
4. Validation Agent - Hard verification of results
5. Answer Agent (13B+) - Final streaming answer

Streaming: Phase events + token streaming for OpenWebUI visibility.
"""

from __future__ import annotations

import os
import json
import time
import asyncio
import re
from typing import Any, Dict, List, Tuple, AsyncGenerator, Optional
from dataclasses import dataclass, field
from enum import Enum
import httpx

# Phase definitions
class Phase(Enum):
    STRATEGY = "strategy"
    RETRIEVAL = "retrieval"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    ANSWER = "answer"

@dataclass
class AgentState:
    """Shared state across all phases"""
    query: str = ""
    strategy: Dict[str, Any] = field(default_factory=dict)
    retrieval_hits: List[Dict[str, Any]] = field(default_factory=list)
    analyzed_documents: List[Dict[str, Any]] = field(default_factory=list)
    validation_result: Dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "retrieval_hits_count": len(self.retrieval_hits),
            "analyzed_docs_count": len(self.analyzed_documents),
            "validation": self.validation_result,
            "iteration": self.iteration,
        }

class AgentOrchestrator:
    """Domain Glossary for Query Disambiguation and Rewriting"""
    """Main orchestrator managing the 5-phase pipeline"""
    
    # Inline glossary rewrite to avoid import issues
    _GLOSSARY_ACRONYMS = {
        "FAT": {
            "meaning": "Factory Acceptance Test",
            "synonyms_de": ["Werksabnahme", "Abnahmetest", "FAT Test", "FAT-Protokoll", "Werksabnahmetest"],
            "context_signals": ["SBB", "TFK", "Tunnelfunk", "Abnahme", "Test", "Protokoll", "Prüfung", "Manteldokument", "Befund"]
        },
        "SAT": {
            "meaning": "Site Acceptance Test", 
            "synonyms_de": ["Standortabnahme", "Bauabnahme", "SAT Test", "Abnahme vor Ort"],
            "context_signals": ["Installation", "Vor-Ort", "Betrieb", "Inbetriebnahme", "Site"]
        },
        "TFK": {
            "meaning": "Tunnelfunkkonzept",
            "synonyms_de": ["TFK 2020", "Tunnelfunk Konzept", "SBB TFK", "Tunnel-Funk-Konzept"],
            "context_signals": ["SBB", "Tunnel", "Funk", "BOS-Funk", "TETRA"]
        }
    }

    def _rewrite_query(self, query: str) -> tuple[str, Dict[str, Any]]:
        """Rewrite query with domain knowledge - inline to avoid import issues"""
        query_upper = query.upper()
        rewritten = query
        expansions = []
        
        for acronym, definition in self._GLOSSARY_ACRONYMS.items():
            pattern = r'\b' + re.escape(acronym) + r'\b'
            if re.search(pattern, query_upper):
                # Check context
                has_context = any(
                    signal.lower() in query.lower() 
                    for signal in definition["context_signals"]
                )
                # Always expand in this domain
                expansion_terms = [definition["meaning"]] + definition["synonyms_de"]
                expansion_str = " OR ".join(f'"{t}"' for t in expansion_terms[:3])
                rewritten = re.sub(pattern, f"({acronym} {expansion_str})", rewritten, flags=re.IGNORECASE)
                expansions.append({"acronym": acronym, "meaning": definition["meaning"]})
        
        return rewritten, {"expansions": expansions}

    def __init__(self):
        self.ollama_base = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
        self.model_strategy = os.getenv("OLLAMA_MODEL_STRATEGY", "qwen2.5:7b")
        self.model_answer = os.getenv("OLLAMA_MODEL_ANSWER", "llama4:latest")
        self.model_analysis = os.getenv("OLLAMA_MODEL_ANALYSIS", "qwen2.5:3b")  # Fast model for doc analysis
        self.trace_enabled = os.getenv("AGENT_TRACE", "1").lower() not in ("0", "false", "no")
        
        # Lazy imports to avoid circular deps
        self._tools = None
        self._phase_agents = {}
        
    @property
    def tools(self):
        if self._tools is None:
            try:
                from .tools import Tools
                self._tools = Tools()
            except Exception:
                self._tools = None
        return self._tools
    
    def _get_phase_agent(self, phase: Phase):
        """Lazy initialization of phase agents"""
        if phase not in self._phase_agents:
            if phase == Phase.STRATEGY:
                from .phase_strategy import StrategyAgent
                self._phase_agents[phase] = StrategyAgent(self.ollama_base, self.model_strategy)
            elif phase == Phase.RETRIEVAL:
                from .phase_retrieval import RetrievalAgent
                self._phase_agents[phase] = RetrievalAgent(self.tools)
            elif phase == Phase.ANALYSIS:
                from .phase_analysis import AnalysisAgent
                self._phase_agents[phase] = AnalysisAgent(self.ollama_base, self.model_analysis)  # Fast 3B model
            elif phase == Phase.VALIDATION:
                from .phase_validation import ValidationAgent
                self._phase_agents[phase] = ValidationAgent(self.ollama_base, self.model_strategy)
            elif phase == Phase.ANSWER:
                from .phase_answer import AnswerAgent
                self._phase_agents[phase] = AnswerAgent(self.ollama_base, self.model_answer)
        return self._phase_agents.get(phase)
    
    async def run(
        self,
        query: str,
        summary: str = "",
        notes: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main orchestration loop with streaming events.
        Yields: {"type": "phase_start", "phase": "...", "data": {...}}
                {"type": "phase_progress", "phase": "...", "message": "..."}
                {"type": "phase_complete", "phase": "...", "result": {...}}
                {"type": "token", "content": "..."}
                {"type": "final", "content": "...", "sources": [...]}
        """
        import sys
        
        # Apply glossary rewrite FIRST for domain knowledge
        try:
            rewritten, meta = self._rewrite_query(query)
            if meta.get('expansions'):
                sys.stderr.write(f"🔧 Glossary rewrite: '{query}' -> '{rewritten[:80]}...'\n")
                sys.stderr.flush()
                query = rewritten
        except Exception as e:
            sys.stderr.write(f"⚠️ Glossary error: {e}\n")
            sys.stderr.flush()
        
        state = AgentState(query=query, iteration=0)
        
        # Phase 1: Strategy
        async for event in self._run_phase(Phase.STRATEGY, state, summary, notes):
            yield event
        
        # Iteration loop: Retrieval -> Analysis -> Validation
        while state.iteration < state.max_iterations:
            state.iteration += 1
            
            # Phase 2: Retrieval
            async for event in self._run_phase(Phase.RETRIEVAL, state, summary, notes):
                yield event
            
            if not state.retrieval_hits:
                yield {"type": "phase_progress", "phase": "retrieval", "message": "Keine Treffer gefunden"}
                break
            
            # Phase 3: Document Analysis
            async for event in self._run_phase(Phase.ANALYSIS, state, summary, notes):
                yield event
            
            # Phase 4: Validation
            async for event in self._run_phase(Phase.VALIDATION, state, summary, notes):
                yield event
            
            # Check if we need another iteration
            if state.validation_result.get("needs_iteration", False) and state.iteration < state.max_iterations:
                yield {
                    "type": "phase_progress",
                    "phase": "orchestrator",
                    "message": f"Iteration {state.iteration + 1}: Strategie-Anpassung..."
                }
                # Update strategy for next iteration
                state.strategy = state.validation_result.get("revised_strategy", state.strategy)
            else:
                break
        
        # Phase 5: Answer
        async for event in self._run_phase(Phase.ANSWER, state, summary, notes):
            yield event
        
        yield {
            "type": "final",
            "content": state.final_answer,
            "sources": state.sources,
            "state": state.to_dict()
        }
    
    async def _run_phase(
        self,
        phase: Phase,
        state: AgentState,
        summary: str,
        notes: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a single phase with event wrapping"""
        
        yield {
            "type": "phase_start",
            "phase": phase.value,
            "timestamp": time.time()
        }
        
        agent = self._get_phase_agent(phase)
        if agent is None:
            yield {"type": "error", "message": f"Agent for phase {phase} not available"}
            return
        
        try:
            if phase == Phase.STRATEGY:
                result = await agent.run(state.query)
                state.strategy = result
                yield {
                    "type": "phase_progress",
                    "phase": phase.value,
                    "message": f"Strategie: {result.get('intent', 'unknown')}, Sprachen: {result.get('languages', [])}"
                }
                
            elif phase == Phase.RETRIEVAL:
                result = await agent.run(state.strategy)
                state.retrieval_hits = result.get("hits", [])
                yield {
                    "type": "phase_progress",
                    "phase": phase.value,
                    "message": f"ES: {result.get('es_hits', 0)} Treffer, Chroma: {result.get('chroma_hits', 0)} Treffer"
                }
                
            elif phase == Phase.ANALYSIS:
                # DEBUG
                import sys
                sys.stderr.write(f"🔬 ANALYSIS PHASE START: {len(state.retrieval_hits)} hits, strategy={state.strategy.get('intent', 'unknown')}\n")
                sys.stderr.flush()
                
                # Progress updates during document reading
                doc_count = 0
                async for progress in agent.run_streaming(state.retrieval_hits, state.strategy):
                    if progress.get("type") == "document_complete":
                        doc_count += 1
                        sys.stderr.write(f"📄 Doc {doc_count} complete: {progress.get('path', 'unknown')[:50]}...\n")
                        sys.stderr.flush()
                        yield {
                            "type": "phase_progress",
                            "phase": phase.value,
                            "message": f"Dokument {doc_count}/{len(state.retrieval_hits)} analysiert: {progress.get('path', 'unknown')}"
                        }
                    elif progress.get("type") == "extraction_complete":
                        state.analyzed_documents = progress.get("documents", [])
                        sys.stderr.write(f"✅ ANALYSIS complete: {len(state.analyzed_documents)} docs analyzed\n")
                        sys.stderr.flush()
                
            elif phase == Phase.VALIDATION:
                result = await agent.run(state.analyzed_documents, state.strategy, state.query)
                state.validation_result = result
                if result.get("needs_iteration", False):
                    yield {
                        "type": "phase_progress",
                        "phase": phase.value,
                        "message": f"Validierung: Treffer unzureichend ({result.get('reason', 'unknown')}) - Neuer Versuch..."
                    }
                else:
                    yield {
                        "type": "phase_progress",
                        "phase": phase.value,
                        "message": f"Validierung: {len(state.analyzed_documents)} Dokumente bestätigt"
                    }
                    
            elif phase == Phase.ANSWER:
                # Stream tokens from answer generation
                answer_parts = []
                async for token in agent.run_streaming(state.analyzed_documents, state.strategy, state.query):
                    if token.get("type") == "token":
                        answer_parts.append(token.get("content", ""))
                        yield token  # Pass through token events
                    elif token.get("type") == "sources":
                        state.sources = token.get("sources", [])
                state.final_answer = "".join(answer_parts)
            
            yield {
                "type": "phase_complete",
                "phase": phase.value,
                "timestamp": time.time()
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "phase": phase.value,
                "message": str(e)
            }

# Backwards-compatible wrapper for existing Agent interface
class Agent:
    """Backwards-compatible wrapper that uses the orchestrator"""
    
    def __init__(self):
        self.orchestrator = AgentOrchestrator()
        self.trace_enabled = self.orchestrator.trace_enabled
    
    async def answer(
        self,
        user_text: str,
        raw_messages: Optional[List[Dict[str, Any]]] = None,
        summary: str = "",
        notes: str = "",
    ) -> Tuple[str, str, str, List[Dict[str, Any]]]:
        """Non-streaming API for backwards compatibility"""
        answer = ""
        sources = []
        
        async for event in self.orchestrator.run(user_text, summary, notes):
            if event.get("type") == "token":
                answer += event.get("content", "")
            elif event.get("type") == "final":
                answer = event.get("content", answer)
                sources = event.get("sources", sources)
        
        return answer, summary, notes, sources
    
    async def answer_stream(
        self,
        user_text: str,
        raw_messages: Optional[List[Dict[str, Any]]] = None,
        summary: str = "",
        notes: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming API with phase events"""
        async for event in self.orchestrator.run(user_text, summary, notes):
            yield event
