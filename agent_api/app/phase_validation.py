"""
phase_validation.py

Phase 4: Validation Agent
Hard verification of analysis results against original query.
Can trigger iteration if results are insufficient.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class ValidationAgent:
    """Validates if analyzed documents answer the query sufficiently"""
    
    def __init__(self, ollama_base: str, model: str):
        self.ollama_base = ollama_base
        self.model = model
        self.min_relevance_score = 0.6
        self.min_coverage = 0.4  # At least 40% of query aspects covered
    
    async def run(
        self,
        analyzed_documents: List[Dict[str, Any]],
        strategy: Dict[str, Any],
        original_query: str
    ) -> Dict[str, Any]:
        """
        Validate analysis results.
        Returns: {"valid": bool, "needs_iteration": bool, "revised_strategy": {...}, "reason": "..."}
        """
        if not analyzed_documents:
            return {
                "valid": False,
                "needs_iteration": True,
                "revised_strategy": self._revise_strategy(strategy, "no_documents"),
                "reason": "no_documents",
                "explanation": "Keine Dokumente zur Analyse vorhanden"
            }
        
        # Check if documents have findings
        docs_with_findings = sum(1 for d in analyzed_documents if d.get("extracted_findings"))
        
        # Quick heuristics first
        heuristic_result = self._heuristic_validation(analyzed_documents, strategy, original_query)
        
        # If heuristics pass, do LLM validation
        if heuristic_result.get("valid", False):
            llm_result = await self._llm_validation(analyzed_documents, strategy, original_query)
            
            # Combine results
            final_valid = heuristic_result["valid"] and llm_result.get("valid", False)
            
            if not final_valid:
                return {
                    "valid": False,
                    "needs_iteration": True,
                    "revised_strategy": self._revise_strategy(strategy, llm_result.get("reason", "unknown")),
                    "reason": llm_result.get("reason", "validation_failed"),
                    "explanation": llm_result.get("explanation", "Validierung fehlgeschlagen"),
                    "relevance_scores": llm_result.get("relevance_scores", {}),
                    "coverage": llm_result.get("coverage", 0)
                }
        
        # Validation passed
        return {
            "valid": True,
            "needs_iteration": False,
            "reason": None,
            "explanation": f"{docs_with_findings} Dokumente mit relevanten Befunden bestätigt",
            "document_count": len(analyzed_documents),
            "docs_with_findings": docs_with_findings,
            "revised_strategy": None
        }
    
    def _heuristic_validation(
        self,
        documents: List[Dict[str, Any]],
        strategy: Dict[str, Any],
        query: str
    ) -> Dict[str, Any]:
        """Quick heuristic checks"""
        
        # Check 1: Document count
        if len(documents) < 2:
            return {"valid": False, "reason": "too_few_documents"}
        
        # Check 2: Findings present
        total_findings = sum(
            len(d.get("extracted_findings", [])) 
            for d in documents
        )
        if total_findings == 0:
            return {"valid": False, "reason": "no_findings"}
        
        # Check 3: Customer match (if specified)
        target_customer = strategy.get("filters", {}).get("customer")
        if target_customer:
            customer_mentions = 0
            target_lower = target_customer.lower()
            
            for doc in documents:
                findings = doc.get("extracted_findings", [])
                for f in findings:
                    content = str(f.get("content", "")).lower()
                    if target_lower in content:
                        customer_mentions += 1
            
            if customer_mentions < 2:
                return {"valid": False, "reason": "customer_not_found"}
        
        # Check 4: Keyword coverage
        keywords = strategy.get("keywords", [])
        if keywords:
            keyword_hits = {kw: 0 for kw in keywords}
            
            for doc in documents:
                findings = doc.get("extracted_findings", [])
                for f in findings:
                    content = str(f.get("content", "")).lower()
                    for kw in keywords:
                        if kw.lower() in content:
                            keyword_hits[kw] += 1
            
            coverage = sum(1 for v in keyword_hits.values() if v > 0) / len(keywords)
            if coverage < self.min_coverage:
                return {"valid": False, "reason": "low_keyword_coverage", "coverage": coverage}
        
        return {"valid": True, "reason": None}
    
    async def _llm_validation(
        self,
        documents: List[Dict[str, Any]],
        strategy: Dict[str, Any],
        query: str
    ) -> Dict[str, Any]:
        """LLM-based validation of result quality"""
        
        # Prepare findings summary
        findings_summary = []
        for doc in documents[:5]:  # Limit for LLM context
            findings = doc.get("extracted_findings", [])
            path = doc.get("path", "unknown")
            
            doc_findings = []
            for f in findings[:5]:  # Limit per doc
                f_type = f.get("type", "unknown")
                content = f.get("content", "") or f.get("description", "")
                if content:
                    doc_findings.append(f"[{f_type}] {content[:200]}")
            
            if doc_findings:
                findings_summary.append(f"Dokument {path}:\n" + "\n".join(doc_findings))
        
        if not findings_summary:
            return {"valid": False, "reason": "no_extractable_findings"}
        
        prompt = f"""Validiere ob die analysierten Dokumente die folgende Anfrage beantworten:

ORIGINAL ANFRAGE: {query}

STRATEGIE: {json.dumps(strategy, ensure_ascii=False)}

GEFUNDENE BEFUNDE:
{chr(10).join(findings_summary)}

Bewerte folgende Aspekte (JSON-Format):
{{
    "valid": true|false,
    "reason": "why_invalid_or_valid",
    "explanation": "Detaillierte Erklärung",
    "relevance_scores": {{
        "query_match": 0.0-1.0,
        "completeness": 0.0-1.0,
        "quality": 0.0-1.0
    }},
    "coverage": 0.0-1.0,
    "missing_aspects": ["was fehlt"],
    "recommendation": "retry|expand_search|proceed"
}}

Regeln:
- valid=true nur wenn Dokumente wirklich zur Query passen
- Bei FAT/SAT/TIB: Prüfe ob Kunden und Befund-Kategorien korrekt erkannt wurden
- Bei Unklarheit: valid=false mit retry-Empfehlung
- coverage = Anteil der Query-Aspekte die abgedeckt sind

Antworte nur mit JSON!"""

        messages = [
            {"role": "system", "content": "Du bist ein Validierungs-Agent. Prüfe ob Dokumenten-Analyse zur Query passt. Antworte nur mit JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._call_llm(messages)
            result = json.loads(response.strip())
            
            # Ensure required fields
            return {
                "valid": result.get("valid", False),
                "reason": result.get("reason", "unknown"),
                "explanation": result.get("explanation", ""),
                "relevance_scores": result.get("relevance_scores", {}),
                "coverage": result.get("coverage", 0),
                "missing_aspects": result.get("missing_aspects", []),
                "recommendation": result.get("recommendation", "retry")
            }
            
        except Exception as e:
            # On error, be conservative and suggest retry
            return {
                "valid": False,
                "reason": "validation_error",
                "explanation": f"Fehler bei Validierung: {str(e)}",
                "recommendation": "retry"
            }
    
    def _revise_strategy(
        self,
        old_strategy: Dict[str, Any],
        reason: str
    ) -> Dict[str, Any]:
        """Revise strategy for next iteration"""
        
        new_strategy = dict(old_strategy)
        keywords = list(new_strategy.get("keywords", []))
        synonyms = dict(new_strategy.get("synonyms", {}))
        expanded = list(new_strategy.get("expanded_queries", []))
        
        if reason == "no_documents":
            # Remove filters, broaden keywords
            new_strategy["filters"] = {"doctype": [], "date_range": None, "customer": None}
            # Add broader synonyms
            for kw in keywords:
                if kw not in synonyms:
                    synonyms[kw] = []
                # Add wilder variations
                synonyms[kw].extend([f"*{kw}*", kw.lower(), kw.upper()])
            
        elif reason == "customer_not_found":
            # Keep customer but add alternative names
            customer = new_strategy.get("filters", {}).get("customer")
            if customer:
                # Add customer to keywords
                if customer not in keywords:
                    keywords.append(customer)
                # Expand customer name
                synonyms[customer] = [
                    customer.lower(),
                    customer.upper(),
                    customer.replace(" ", ""),
                    customer.replace("-", " ")
                ]
            
        elif reason == "low_keyword_coverage":
            # Add more synonyms for uncovered keywords
            for kw in keywords:
                if kw not in synonyms or not synonyms[kw]:
                    synonyms[kw] = [f"{kw}*", f"*{kw}", f"*{kw}*"]
        
        elif reason == "no_findings" or reason == "no_extractable_findings":
            # Try different document types
            current_types = new_strategy.get("filters", {}).get("doctype", [])
            if not current_types:
                new_strategy["filters"]["doctype"] = ["pdf", "docx"]
            # Add broader terms
            new_terms = ["report", "test", "protocol", "findings"]
            for term in new_terms:
                if term not in keywords:
                    keywords.append(term)
        
        # Generate new expanded queries
        if keywords:
            expanded = self._generate_expanded_queries(keywords, synonyms)
        
        new_strategy["keywords"] = keywords
        new_strategy["synonyms"] = synonyms
        new_strategy["expanded_queries"] = expanded[:4]
        new_strategy["iteration"] = old_strategy.get("iteration", 0) + 1
        
        return new_strategy
    
    def _generate_expanded_queries(
        self,
        keywords: List[str],
        synonyms: Dict[str, List[str]]
    ) -> List[str]:
        """Generate expanded query variants"""
        queries = []
        
        # Original
        queries.append(" ".join(keywords))
        
        # With first synonym for each
        if synonyms:
            expanded = []
            for kw in keywords:
                syns = synonyms.get(kw, [])
                if syns:
                    expanded.append(syns[0])
                else:
                    expanded.append(kw)
            queries.append(" ".join(expanded))
        
        # With wildcards
        wildcard = [f"*{kw}*" for kw in keywords[:3]]
        queries.append(" ".join(wildcard))
        
        return queries[:4]
    
    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call Ollama"""
        import httpx
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 1024}
        }
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            r = await client.post(f"{self.ollama_base}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")
