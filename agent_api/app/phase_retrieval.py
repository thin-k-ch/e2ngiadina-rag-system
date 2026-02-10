"""
phase_retrieval.py

Phase 2: Retrieval Agent
Executes ES BM25 + Chroma vector search with strategy-based query expansion.
Includes pre-validation (hard thresholds) for hit quality.
"""

from __future__ import annotations

import os
import asyncio
from typing import Any, Dict, List, Optional


class RetrievalAgent:
    """Hybrid retrieval with pre-validation"""
    
    def __init__(self, tools):
        self.tools = tools
        self.es_min_score = 0.3  # Minimum BM25 score
        self.chroma_max_distance = 1.5  # Maximum vector distance
        self.min_diversity = 0.3  # Minimum diversity score
        self.max_hits_per_source = 3  # Max hits from same file
    
    async def run(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute retrieval based on strategy.
        Returns: {"hits": [...], "es_hits": N, "chroma_hits": N, "pre_validated": bool}
        """
        # Get expanded queries
        expanded_queries = strategy.get("expanded_queries", [])
        if not expanded_queries:
            keywords = strategy.get("keywords", [])
            expanded_queries = [" ".join(keywords)] if keywords else [""]
        
        all_hits = []
        es_total = 0
        chroma_total = 0
        
        # Search with each expanded query variant
        for query in expanded_queries[:2]:  # Max 2 variants to avoid explosion
            if not query:
                continue
                
            result = await self._search_with_policy(query, strategy)
            
            if result.get("mode") == "exact_phrase":
                hits = result.get("best_hits", [])
                es_total += int(result.get("total_hits", 0) or 0)
            elif result.get("mode") == "hybrid":
                hits = result.get("merged_hits", [])
                es_result = result.get("es_hits", 0)
                chroma_result = result.get("chroma_hits", 0)
                # Handle both int and list returns
                es_total += len(es_result) if isinstance(es_result, list) else int(es_result or 0)
                chroma_total += len(chroma_result) if isinstance(chroma_result, list) else int(chroma_result or 0)
            else:
                hits = []
            
            all_hits.extend(hits)
        
        # Deduplicate and limit
        deduped = self._deduplicate_hits(all_hits, limit=20)
        
        # Pre-validation: hard thresholds
        validated, pre_validation_result = self._pre_validate(deduped, strategy)
        
        return {
            "hits": validated[:12],  # Return top 12
            "es_hits": es_total,
            "chroma_hits": chroma_total,
            "total_before_dedup": len(all_hits),
            "total_after_dedup": len(deduped),
            "pre_validated_count": len(validated),
            "pre_validation": pre_validation_result,
            "needs_iteration": pre_validation_result.get("needs_iteration", False)
        }
    
    async def _search_with_policy(
        self,
        query: str,
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use existing policy-based search from tools"""
        if not self.tools:
            return {"mode": "no_tools", "hits": [], "total_hits": 0}
        
        # Apply tool gate
        gate = self.tools.decide_gate(query)
        
        if not gate.require_rag:
            return {"mode": "no_rag", "hits": [], "total_hits": 0}
        
        # Execute based on mode
        if gate.mode == "exact_phrase":
            result = self.tools.search_exact_phrase(gate.phrase or query)
            return {
                "mode": "exact_phrase",
                "best_hits": result.get("best_hits", []),
                "total_hits": result.get("total_hits", 0)
            }
        elif gate.mode == "hybrid":
            result = self.tools.search_hybrid(query)
            return {
                "mode": "hybrid",
                "merged_hits": result.get("merged_hits", []),
                "es_hits": result.get("es_hits", 0),
                "chroma_hits": result.get("chroma_hits", 0)
            }
        else:
            return {"mode": "no_rag", "hits": [], "total_hits": 0}
    
    def _deduplicate_hits(
        self,
        hits: List[Dict[str, Any]],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Deduplicate hits by path + snippet fingerprint"""
        seen = set()
        out = []
        
        for h in hits:
            # Get path
            if isinstance(h, str):
                path = h[:100]
                text = h[:80]
            else:
                path = h.get("path", "")
                if not path:
                    file_info = h.get("file", {})
                    path = file_info.get("path", "") if isinstance(file_info, dict) else ""
                text = (h.get("text", "") or h.get("snippet", ""))[:80]
            
            # Create fingerprint
            fp = (path, text)
            if fp in seen:
                continue
            seen.add(fp)
            
            # Normalize to common format
            normalized = self._normalize_hit(h)
            if normalized:
                out.append(normalized)
            
            if len(out) >= limit:
                break
        
        return out
    
    def _normalize_hit(self, hit: Any) -> Optional[Dict[str, Any]]:
        """Normalize hit to common format"""
        if isinstance(hit, str):
            return {
                "path": hit,
                "text": hit,
                "snippet": hit[:200],
                "score": 0.0,
                "source": "unknown",
                "metadata": {}
            }
        
        if not isinstance(hit, dict):
            return None
        
        # Extract path
        path = hit.get("path", "")
        if not path:
            file_info = hit.get("file", {})
            if isinstance(file_info, dict):
                path = file_info.get("path", "")
        
        # Extract text
        text = hit.get("text", "") or hit.get("snippet", "") or hit.get("content", "")
        
        # Extract score
        score = hit.get("score", 0)
        if score is None:
            score = 0.0
        
        # Determine source
        source = "unknown"
        if "_es" in str(hit) or hit.get("es_score"):
            source = "elasticsearch"
        elif "_chroma" in str(hit) or hit.get("distance"):
            source = "chroma"
        
        return {
            "path": path,
            "text": text,
            "snippet": text[:300] if text else "",
            "score": float(score),
            "source": source,
            "metadata": hit.get("metadata", {}),
            "es_score": hit.get("es_score"),
            "chroma_distance": hit.get("distance")
        }
    
    def _pre_validate(
        self,
        hits: List[Dict[str, Any]],
        strategy: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Hard threshold validation with content filtering.
        Removes irrelevant documents (config files, etc.).
        """
        if not hits:
            return [], {
                "needs_iteration": True,
                "reason": "no_hits",
                "suggestion": "Expand search terms or remove filters"
            }
        
        # Get search keywords for content checking
        keywords = strategy.get("keywords", [])
        keyword_set = set(kw.lower() for kw in keywords)
        # Add common domain terms
        keyword_set.update(["fehler", "befund", "abnahme", "fat", "sat", "a-", "b-", "error", "defect"])
        
        # Filter out irrelevant documents
        filtered_hits = []
        rejected = []
        
        for h in hits:
            path = h.get("path", "").lower()
            text = (h.get("text", "") or "").lower()
            snippet = (h.get("snippet", "") or "").lower()
            combined_text = text + " " + snippet
            
            # Reject obvious non-content files
            reject_patterns = [
                ".config", "config.", "settings.", "setup.", "install.", 
                "requirements.txt", "docker-compose", ".yaml", ".yml",
                "readme", "license", ".git", ".env", ".log"  # unless they contain actual findings
            ]
            
            is_reject = any(pattern in path for pattern in reject_patterns)
            
            # For reject candidates, check if they actually contain relevant content
            if is_reject:
                # Only keep if strong evidence of relevant content
                has_strong_evidence = any(
                    term in combined_text[:500] 
                    for term in ["fehler", "befund", "abweichung", "error", "defect", "finding"]
                )
                if not has_strong_evidence:
                    rejected.append({"path": path, "reason": "config/system file without findings"})
                    continue
            
            # Check if content matches search intent (at least one keyword or domain term)
            has_relevant_content = any(kw in combined_text for kw in keyword_set)
            
            # Accept if ES score is high enough OR content matches
            es_score = h.get("score", 0) or 0
            if es_score >= 2.0 or has_relevant_content:
                filtered_hits.append(h)
            else:
                rejected.append({"path": path, "reason": "low relevance score + no keyword match"})
        
        # Check minimum scores
        es_hits = [h for h in filtered_hits if h.get("source") == "elasticsearch"]
        chroma_hits = [h for h in filtered_hits if h.get("source") == "chroma"]
        
        low_quality_es = sum(1 for h in es_hits if h.get("score", 0) < self.es_min_score)
        low_quality_chroma = sum(1 for h in chroma_hits if h.get("chroma_distance", 999) > self.chroma_max_distance)
        
        # Check source diversity
        unique_paths = set(h.get("path", "") for h in filtered_hits if h.get("path"))
        diversity = len(unique_paths) / len(filtered_hits) if filtered_hits else 0
        
        # Check for customer matches if specified
        target_customer = strategy.get("filters", {}).get("customer")
        customer_matches = 0
        if target_customer:
            for h in filtered_hits:
                path = h.get("path", "").lower()
                text = (h.get("text", "") or "").lower()
                combined = path + " " + text
                if target_customer.lower() in combined:
                    customer_matches += 1
        
        # Build validation result
        validation = {
            "total_hits": len(hits),
            "filtered_hits": len(filtered_hits),
            "rejected_count": len(rejected),
            "rejected_reasons": rejected[:5],  # Show first 5 for debugging
            "es_hits": len(es_hits),
            "chroma_hits": len(chroma_hits),
            "unique_sources": len(unique_paths),
            "diversity_score": diversity,
            "low_quality_es": low_quality_es,
            "low_quality_chroma": low_quality_chroma,
            "customer_matches": customer_matches if target_customer else None,
            "needs_iteration": False,
            "reason": None,
            "suggestion": None
        }
        
        # Determine if iteration needed
        needs_iteration = False
        reason = None
        suggestion = None
        
        if len(filtered_hits) < 3:
            needs_iteration = True
            reason = "too_few_hits_after_filter"
            suggestion = "Expand search with synonyms - many hits were filtered as irrelevant"
        elif diversity < self.min_diversity:
            needs_iteration = True
            reason = "low_diversity"
            suggestion = "Hits too similar - broaden search"
        elif target_customer and customer_matches < 2:
            needs_iteration = True
            reason = "customer_mismatch"
            suggestion = f"Add more terms related to customer {target_customer}"
        
        validation["needs_iteration"] = needs_iteration
        validation["reason"] = reason
        validation["suggestion"] = suggestion
        
        # Filter out very low quality hits
        validated = [
            h for h in filtered_hits
            if (h.get("source") != "elasticsearch" or h.get("score", 0) >= self.es_min_score * 0.5)
            and (h.get("source") != "chroma" or h.get("chroma_distance", 999) <= self.chroma_max_distance * 1.5)
        ]
        
        return validated, validation
