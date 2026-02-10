"""
thinking_agent.py

Phase 2: True Thinking Mode Agent
Replaces the 5-phase pipeline with a unified reasoning agent that:
1. Develops its own search strategy through Chain of Thought
2. Analyzes documents with full context understanding
3. Self-corrects and iterates on its own reasoning
4. Streams the thinking process to the UI

Inspired by ChatGPT o1/o3 - the LLM does the reasoning, not the code.
"""

from __future__ import annotations

import os
import json
import httpx
import asyncio
from typing import Any, Dict, List, Optional, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum

from app.glossary import DomainGlossary, rewrite_query


class ThoughtType(Enum):
    PLANNING = "planning"           # Strategy development
    SEARCHING = "searching"         # Document retrieval
    READING = "reading"             # Document analysis
    REASONING = "reasoning"         # Logical deduction
    CRITIQUE = "critique"           # Self-criticism
    REFINEMENT = "refinement"       # Strategy adjustment
    SYNTHESIS = "synthesis"         # Answer construction
    FINAL = "final"                 # Final answer


@dataclass
class Thought:
    type: ThoughtType
    content: str
    step: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    tool_name: str
    parameters: Dict[str, Any]
    result: Any = None
    error: str = None


class ThinkingAgent:
    """
    Unified reasoning agent that replaces the 5-phase pipeline.
    The LLM controls the flow through structured thinking.
    """
    
    def __init__(
        self,
        ollama_base: str,
        model: str,  # Should be 13B+ for reasoning
        tools: Any  # Tools instance for ES/Chroma/Document reading
    ):
        self.ollama_base = ollama_base
        self.model = model
        self.tools = tools
        
        # Configuration
        self.max_thinking_steps = 15
        self.max_tool_calls = 10
        self.thinking_temperature = 0.7  # Higher for creative reasoning
        self.answer_temperature = 0.3    # Lower for factual answers
        
        # System prompts for different phases
        self._thinking_prompt = self._load_thinking_prompt()
        self._answer_prompt = self._load_answer_prompt()
    
    async def run(
        self,
        query: str,
        conversation_context: str = "",
        notes: str = ""
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main entry point. Streams thinking process and final answer.
        
        Yields:
            {"type": "thought", "thought_type": "...", "content": "...", "step": N}
            {"type": "tool_call", "tool": "...", "params": {...}}
            {"type": "tool_result", "result": ...}
            {"type": "token", "content": "..."}  # Final answer tokens
            {"type": "final", "content": "...", "sources": [...]}
        """
        
        # Initialize thinking state
        state = {
            "query": query,
            "context": conversation_context,
            "notes": notes,
            "thoughts": [],
            "tool_calls": [],
            "documents": [],  # Read documents
            "findings": [],   # Extracted findings
            "iteration": 0,
            "max_iterations": 3
        }
        
        # Phase 1: Strategic Thinking - SKIP for speed, use glossary-based search directly
        yield {
            "type": "thought",
            "thought_type": "planning",
            "content": "Using glossary-based search strategy (fast mode)",
            "step": 1
        }
        
        # Phase 2: Document Discovery (glossary-enhanced)
        while state["iteration"] < state["max_iterations"]:
            state["iteration"] += 1
            
            # Search and retrieve documents
            async for event in self._discover_documents(state):
                yield event
            
            if not state["documents"]:
                break
            
            # Analyze documents - SKIP slow LLM analysis, just collect snippets
            for doc in state["documents"][:5]:
                path = (doc.get("path") or 
                       doc.get("file", {}).get("path") or 
                       doc.get("_source", {}).get("meta", {}).get("real", {}).get("path") or
                       doc.get("metadata", {}).get("path") or
                       "unknown")
                
                content = doc.get("text") or doc.get("snippet") or doc.get("content", "")
                
                # Quick extraction without LLM
                if content:
                    state["findings"].append({
                        "source": path,
                        "type": "snippet",
                        "summary": content[:200],
                        "details": content[:1000],
                        "confidence": "medium"
                    })
            
            # Skip self-critique for speed
            break  # Exit after first iteration
        
        # Phase 3: Synthesize Final Answer (ALWAYS run this)
        yield {
            "type": "thought",
            "thought_type": "synthesis",
            "content": f"Synthesizing answer from {len(state['findings'])} findings...",
            "step": len(state["thoughts"]) + 1
        }
        
        async for event in self._synthesize_answer(state):
            yield event
    
    async def _strategic_thinking(
        self,
        state: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Step 1: The LLM develops a search and analysis strategy.
        This is the core "thinking" phase.
        """
        
        # Send start event immediately to keep stream alive
        yield {
            "type": "thought",
            "thought_type": "planning",
            "content": "Developing search strategy...",
            "step": 1
        }
        
        messages = [
            {"role": "system", "content": self._thinking_prompt},
            {"role": "user", "content": self._build_thinking_user_prompt(state)}
        ]
        
        # Stream the thinking process - MINIMAL tokens for speed
        thinking_content = []
        step = 2
        
        async for chunk in self._llm_stream(
            messages,
            temperature=self.thinking_temperature,
            max_tokens=512  # Drastically reduced from 2048 - just get search queries fast
        ):
            thinking_content.append(chunk)
            
            # Parse structured thoughts from stream
            # Look for patterns like "[THOUGHT: type] content"
            thought = self._parse_thought_chunk(chunk, step)
            if thought:
                yield {
                    "type": "thought",
                    "thought_type": thought.type.value,
                    "content": thought.content,
                    "step": step
                }
                state["thoughts"].append(thought)
                step += 1
        
        # If LLM didn't generate any tool calls, fall back to direct glossary-based search
        full_thinking = "".join(thinking_content)
        tool_calls = self._extract_tool_calls(full_thinking)
        
        if not tool_calls:
            # Fallback: generate search queries directly using glossary
            yield {
                "type": "thought",
                "thought_type": "planning",
                "content": "Using glossary-based search (LLM planning was empty)",
                "step": step
            }
            search_queries = await self._generate_search_queries(state)
            for sq in search_queries:
                tool_calls.append(ToolCall(
                    tool_name="search",
                    parameters={"query": sq["query"]}
                ))
        
        for tc in tool_calls:
            yield {
                "type": "tool_call",
                "tool": tc.tool_name,
                "params": tc.parameters
            }
            
            # Execute tool call
            result = await self._execute_tool_call(tc)
            tc.result = result
            
            yield {
                "type": "tool_result",
                "tool": tc.tool_name,
                "result": result
            }
            
            state["tool_calls"].append(tc)
    
    async def _discover_documents(
        self,
        state: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Step 2: Execute searches based on the developed strategy.
        The LLM decides which searches to run.
        """
        
        # Let the LLM decide what to search for
        search_strategy = await self._generate_search_queries(state)
        
        for query_info in search_strategy:
            query = query_info["query"]
            reason = query_info.get("reason", "")
            
            yield {
                "type": "thought",
                "thought_type": "searching",
                "content": f"Searching: {query} (Reason: {reason})",
                "step": len(state["thoughts"]) + 1
            }
            
            # Execute hybrid search WITH explicit PDF/DOCX extensions
            result = self.tools.search_hybrid(
                query, 
                ext_filter=["pdf", "docx", "doc", "txt", "md", "msg", "eml", "xlsx", "pptx"]
            )
            
            hits = result.get("merged_hits", [])
            es_count = len(result.get("es_hits", [])) if isinstance(result.get("es_hits"), list) else result.get("es_hits", 0)
            chroma_count = len(result.get("chroma_hits", [])) if isinstance(result.get("chroma_hits"), list) else result.get("chroma_hits", 0)
            
            yield {
                "type": "thought",
                "thought_type": "searching",
                "content": f"Found {len(hits)} hits (ES: {es_count}, Chroma: {chroma_count})",
                "step": len(state["thoughts"]) + 1
            }
            
            # Add to state
            for hit in hits[:5]:  # Top 5 per query
                if hit not in state["documents"]:
                    state["documents"].append(hit)
    
    async def _analyze_documents(
        self,
        state: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Step 3: Deep analysis of retrieved documents.
        The LLM reads and understands the documents in context.
        """
        
        # Prioritize and limit documents to prevent timeout
        prioritized = self._prioritize_documents(state["documents"], state["query"])
        
        # Analyze top 5 documents only
        for doc in prioritized[:5]:
            # Get document path from various possible locations
            path = (doc.get("path") or 
                   doc.get("file", {}).get("path") or 
                   doc.get("_source", {}).get("meta", {}).get("real", {}).get("path") or
                   doc.get("metadata", {}).get("path") or
                   "unknown")
            
            yield {
                "type": "thought",
                "thought_type": "reading",
                "content": f"Reading document: {path}",
                "step": len(state["thoughts"]) + 1
            }
            
            # Read full document content
            content = await self._read_document(doc)
            
            # Let LLM extract findings
            findings = await self._extract_findings(content, state["query"], path)
            
            for finding in findings:
                state["findings"].append({
                    "source": path,
                    **finding
                })
                
                yield {
                    "type": "thought",
                    "thought_type": "reasoning",
                    "content": f"Finding: {finding.get('summary', '')[:100]}...",
                    "step": len(state["thoughts"]) + 1
                }
    
    async def _self_critique(self, state: Dict[str, Any]) -> bool:
        """
        Step 4: Self-critique. Do we need more information?
        Returns True if we should search for more documents.
        """
        
        if not state["findings"]:
            return True  # Need to search more
        
        # Ask LLM if we have enough information
        critique_prompt = f"""Based on the following findings, do we have enough information to answer the query comprehensively?

Query: {state['query']}

Findings ({len(state['findings'])}):
"""
        for i, f in enumerate(state["findings"][:10], 1):
            critique_prompt += f"{i}. {f.get('summary', 'N/A')[:200]}\n"
        
        critique_prompt += """

Respond with JSON:
{
    "sufficient": true/false,
    "reason": "explanation",
    "gaps": ["what information is missing"]
}"""
        
        response = await self._llm_complete([
            {"role": "system", "content": "You are a critical analyst. Be honest about information gaps."},
            {"role": "user", "content": critique_prompt}
        ])
        
        try:
            result = json.loads(response)
            return not result.get("sufficient", True)
        except:
            return False
    
    async def _refine_strategy(
        self,
        state: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Step 5: Refine search strategy based on critique.
        """
        
        yield {
            "type": "thought",
            "thought_type": "refinement",
            "content": "Refining search strategy based on findings...",
            "step": len(state["thoughts"]) + 1
        }
        
        # Let LLM suggest new search directions
        # This updates state for the next iteration
    
    async def _synthesize_answer(
        self,
        state: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Step 6: Synthesize final answer from all findings.
        """
        
        # Build context from findings
        context = self._build_findings_context(state["findings"])
        
        messages = [
            {"role": "system", "content": self._answer_prompt},
            {"role": "user", "content": f"""Query: {state['query']}

Findings from {len(state['findings'])} sources:
{context}

Provide a comprehensive answer based ONLY on the above findings. Cite sources with [Source: path]."""}
        ]
        
        # Stream final answer
        full_answer = []
        async for chunk in self._llm_stream(
            messages,
            temperature=self.answer_temperature,
            max_tokens=4096
        ):
            full_answer.append(chunk)
            yield {
                "type": "token",
                "content": chunk
            }
        
        # Yield final with sources
        yield {
            "type": "final",
            "content": "".join(full_answer),
            "sources": list({f["source"] for f in state["findings"]})
        }
    
    # --- Helper Methods ---
    
    async def _llm_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from Ollama"""
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
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
                        obj = json.loads(line)
                        content = obj.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except:
                        continue
    
    async def _llm_complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024  # Reduced for speed
    ) -> str:
        """Non-streaming LLM call with shorter timeout"""
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            r = await client.post(f"{self.ollama_base}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")
    
    def _load_thinking_prompt(self) -> str:
        """System prompt for the thinking phase - clean, no domain append"""
        return """You are a strategic research assistant.

THINKING PROCESS:
1. Analyze the query - what information is needed?
2. Develop search queries - be specific and targeted
3. Plan document analysis - identify key sections to read
4. Consider gaps - what if initial searches don't find enough?

OUTPUT FORMAT:
Use structured thinking markers:
[THOUGHT: planning] Your strategic planning here
[THOUGHT: searching] When you want to search: [SEARCH: "query"]
[THOUGHT: reasoning] Logical deductions
[THOUGHT: critique] Self-criticism of the approach
[THOUGHT: refinement] Adjusting strategy

TOOL CALLS:
To search documents, use: [SEARCH: "your search query"]
To read a specific document, use: [READ: "/path/to/file.pdf"]

Be thorough but focused. Think step by step."""
    
    def _load_answer_prompt(self) -> str:
        """System prompt for final answer synthesis - clean, factual"""
        return """You are a factual research assistant.

RULES:
1. Answer ONLY based on the provided findings
2. Cite sources explicitly: [Source: filename.pdf]
3. Be definitive - "The documents show..." not "It seems..."
4. If findings are insufficient: "The available documents do not contain information about..."
5. Structure the answer clearly with sections if needed

FORBIDDEN:
- "It appears that..."
- "It seems..."
- "Based on my analysis..." (not "my", cite the documents)
- Speculation beyond the findings"""
    
    def _build_thinking_user_prompt(self, state: Dict[str, Any]) -> str:
        """Build user prompt for thinking phase"""
        prompt = f"""QUERY TO RESEARCH: {state['query']}

CONTEXT:
- This is iteration {state['iteration']} of {state['max_iterations']}
- Previous thoughts: {len(state['thoughts'])}
- Documents found so far: {len(state['documents'])}
- Findings extracted: {len(state['findings'])}"""
        
        if state["notes"]:
            prompt += f"\n\nNOTES:\n{state['notes']}"
        
        prompt += """\n\nDevelop your search strategy. What will you search for? How will you analyze the documents?"""
        
        return prompt
    
    def _parse_thought_chunk(self, chunk: str, step: int) -> Optional[Thought]:
        """Parse thought markers from LLM output"""
        
        import time
        
        # Look for [THOUGHT: type] pattern
        if "[THOUGHT:" in chunk:
            try:
                start = chunk.find("[THOUGHT:") + 9
                end = chunk.find("]", start)
                thought_type_str = chunk[start:end].strip().lower()
                
                # Map to enum
                type_mapping = {
                    "planning": ThoughtType.PLANNING,
                    "searching": ThoughtType.SEARCHING,
                    "reading": ThoughtType.READING,
                    "reasoning": ThoughtType.REASONING,
                    "critique": ThoughtType.CRITIQUE,
                    "refinement": ThoughtType.REFINEMENT,
                    "synthesis": ThoughtType.SYNTHESIS,
                }
                
                thought_type = type_mapping.get(thought_type_str, ThoughtType.REASONING)
                
                # Extract content after the marker
                content_start = end + 1
                content = chunk[content_start:].strip()
                
                return Thought(
                    type=thought_type,
                    content=content,
                    step=step,
                    timestamp=time.time()
                )
            except:
                pass
        
        return None
    
    def _extract_tool_calls(self, thinking_output: str) -> List[ToolCall]:
        """Extract tool calls from thinking output"""
        
        tool_calls = []
        
        # Search for [SEARCH: "query"] pattern
        import re
        search_pattern = r'\[SEARCH:\s*"([^"]+)"\]'
        for match in re.finditer(search_pattern, thinking_output):
            tool_calls.append(ToolCall(
                tool_name="search",
                parameters={"query": match.group(1)}
            ))
        
        # Search for [READ: "path"] pattern
        read_pattern = r'\[READ:\s*"([^"]+)"\]'
        for match in re.finditer(read_pattern, thinking_output):
            tool_calls.append(ToolCall(
                tool_name="read_document",
                parameters={"path": match.group(1)}
            ))
        
        return tool_calls
    
    async def _execute_tool_call(self, tc: ToolCall) -> Any:
        """Execute a tool call with glossary rewrite applied to searches"""
        
        if tc.tool_name == "search":
            query = tc.parameters.get("query", "")
            
            # Apply glossary rewrite to LLM-generated search queries
            rewritten_query, meta = rewrite_query(query)
            if meta['expansions']:
                print(f"🔧 Tool search rewrite: '{query}' → '{rewritten_query[:80]}...'")
                query = rewritten_query
            
            return self.tools.search_hybrid(
                query,
                ext_filter=["pdf", "docx", "doc", "txt", "md", "msg", "eml", "xlsx", "pptx"]
            )
        
        elif tc.tool_name == "read_document":
            # Would need to implement document reading
            path = tc.parameters.get("path", "")
            return {"path": path, "content": "Document content would be read here"}
        
        return None
    
    async def _generate_search_queries(
        self,
        state: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate search queries with domain glossary applied"""
        
        # Apply glossary disambiguation FIRST
        original_query = state['query']
        rewritten_query, meta = rewrite_query(original_query)
        
        # Log the rewrite for transparency
        if meta['expansions']:
            print(f"🔧 Query rewrite: '{original_query}' → '{rewritten_query}'")
            print(f"   Expansions: {[e['acronym'] + '=' + e['meaning'] for e in meta['expansions']]}")
        
        # If we have expansions, use the rewritten query directly
        if meta['expansions']:
            return [
                {"query": rewritten_query, "reason": f"Domain-expanded: {meta['expansions'][0]['acronym']}={meta['expansions'][0]['meaning']}"},
                {"query": original_query, "reason": "Original query as fallback"}
            ]
        
        # Otherwise let LLM generate variations
        prompt = f"""Based on the query "{original_query}", generate 2-3 search queries to find relevant documents.

Respond with JSON array:
[
    {{"query": "search term 1", "reason": "why this query"}},
    {{"query": "search term 2", "reason": "why this query"}}
]"""
        
        response = await self._llm_complete([
            {"role": "user", "content": prompt}
        ], temperature=0.5)
        
        try:
            return json.loads(response)
        except:
            # Fallback: use original
            return [
                {"query": original_query, "reason": "Original query"}
            ]
    
    async def _read_document(self, doc: Dict[str, Any]) -> str:
        """Read full document content - prefer ES content over filesystem"""
        
        # Try ES content first (fastest)
        try:
            path = doc.get("path") or doc.get("file", {}).get("path")
            if path:
                es_doc = self.tools.es.es_get_document_content(file_path=path)
                if es_doc and "content" in es_doc and es_doc["content"]:
                    print(f"📄 READ FROM ES: {path[:60]}... ({len(es_doc['content'])} chars)")
                    return es_doc["content"]
        except Exception as e:
            print(f"⚠️ ES read failed: {e}")
        
        # Fallback to snippet/text in doc
        content = doc.get("text") or doc.get("snippet") or doc.get("content")
        if content:
            print(f"📄 READ FROM snippet: {len(content)} chars")
            return content
            
        # Last resort: try _source from ES hit
        if "_source" in doc:
            return doc["_source"].get("content", "")
            
        return ""
    
    async def _extract_findings(
        self,
        content: str,
        query: str,
        source: str
    ) -> List[Dict[str, Any]]:
        """Extract relevant findings from document"""
        
        prompt = f"""Analyze this document excerpt and extract findings relevant to the query.

Query: {query}
Source: {source}

Document excerpt (first 3000 chars):
{content[:3000]}

Extract findings in JSON format:
[
    {{
        "type": "finding|fact|summary",
        "category": "for findings: A-error|B-error|other",
        "summary": "brief summary",
        "details": "relevant details",
        "confidence": "high|medium|low"
    }}
]

Return empty array [] if no relevant findings."""
        
        response = await self._llm_complete([
            {"role": "user", "content": prompt}
        ])
        
        try:
            return json.loads(response)
        except:
            return []
    
    def _prioritize_documents(
        self,
        documents: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Prioritize documents by relevance"""
        
        # Sort by score (ES) or inverse distance (Chroma)
        def score(doc):
            if doc.get("source") == "es":
                return doc.get("score", 0)
            else:
                # Chroma: convert distance to score
                dist = doc.get("distance", 1.0) or 1.0
                return 1.0 - dist
        
        return sorted(documents, key=score, reverse=True)
    
    def _build_findings_context(self, findings: List[Dict[str, Any]]) -> str:
        """Build context string from findings"""
        
        parts = []
        for i, f in enumerate(findings, 1):
            source = f.get("source", "unknown")
            summary = f.get("summary", "")
            details = f.get("details", "")
            category = f.get("category", "")
            
            part = f"[{i}] Source: {source}\n"
            if category:
                part += f"Category: {category}\n"
            part += f"Summary: {summary}\n"
            if details:
                part += f"Details: {details[:500]}\n"
            parts.append(part)
        
        return "\n---\n".join(parts)
