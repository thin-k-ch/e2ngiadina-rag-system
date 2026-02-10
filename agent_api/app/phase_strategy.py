"""
phase_strategy.py

Phase 1: Strategy Agent
Generates structured search strategy from user query.
Uses smaller model (7B) for speed.

Output: JSON with intent, languages, keywords, synonyms, filters
"""

from __future__ import annotations

import os
import json
import httpx
from typing import Any, Dict, List, Optional


class StrategyAgent:
    """Generates multilingual search strategies with synonym expansion"""
    
    def __init__(self, ollama_base: str, model: str):
        self.ollama_base = ollama_base
        self.model = model
        
        # Common DE-EN business term mappings
        self._term_mappings = {
            # Document types
            "rechnung": ["invoice", "bill", "billing"],
            "vertrag": ["contract", "agreement"],
            "angebot": ["offer", "proposal", "quotation"],
            "bestellung": ["order", "purchase order", "po"],
            "lieferschein": ["delivery note", "shipping document"],
            "mahnung": ["reminder", "dunning notice"],
            "quittung": ["receipt"],
            "zeugnis": ["report", "certificate"],
            "protokoll": ["protocol", "minutes", "log"],
            
            # FAT/SAT/TIB terms
            "fat": ["factory acceptance test", "werkabnahme", "abnahme"],
            "sat": ["site acceptance test", "kundenabnahme"],
            "tib": ["test in building"],
            "abnahme": ["acceptance", "approval", "sign-off"],
            "befund": ["finding", "issue", "defect", "observation"],
            "fehler": ["error", "defect", "fault", "bug"],
            "mangel": ["deficiency", "defect", "shortcoming"],
            "prüfung": ["test", "inspection", "verification"],
            
            # Customer/Project terms
            "kunde": ["customer", "client"],
            "projekt": ["project"],
            "auftrag": ["order", "job", "assignment"],
            "mandat": ["mandate", "contract"],
            
            # Mantel/Wrapper document terms - CRITICAL for this domain
            "mantel": ["wrapper", "envelope", "hülle", "cover"],
            "manteldokument": ["wrapper document", "cover sheet", "übersichtsdokument"],
            "a-fehler": ["a error", "a defect", "critical defect", "schwere abweichung"],
            "b-fehler": ["b error", "b defect", "minor defect", "geringe abweichung"],
            "a fehler": ["a error", "a defect", "critical defect"],
            "b fehler": ["b error", "b defect", "minor defect"],
            "abweichung": ["deviation", "discrepancy", "non-conformance"],
            "nachweis": ["proof", "evidence", "certification"],
            "dokumentation": ["documentation", "records", "papers"],
        }
    
    async def run(self, query: str) -> Dict[str, Any]:
        """Generate search strategy from query"""
        
        # Build prompt for strategy generation
        prompt = self._build_strategy_prompt(query)
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._call_llm(messages)
            strategy = self._parse_response(response)
            
            # Post-process: add synonym expansions
            strategy = self._expand_synonyms(strategy)
            
            return strategy
            
        except Exception as e:
            # Fallback: basic strategy
            return {
                "intent": "fact_lookup",
                "languages": ["de", "en"],
                "keywords": [query],
                "synonyms": self._get_basic_synonyms(query),
                "filters": {"doctype": [], "date_range": None, "customer": None},
                "expanded_queries": [query],
                "confidence": 0.5
            }
    
    def _get_system_prompt(self) -> str:
        return """Du bist ein Search-Strategie-Agent für technische Dokumente (FAT/SAT/TIB).

ANALYSE-REGELN:
1. "Manteldokument" = Wrapper/Deckblatt mit Zusammenfassung aller Befunde
2. "A-Fehler" = Kritische Abweichungen (schwerwiegend, blocker)
3. "B-Fehler" = Geringfügige Abweichungen (kosmetisch, minor)
4. Suchbegriffe MÜSSEN Synonyme für DE und EN enthalten

WICHTIG: Antworte AUSSCHLIESSLICH mit validem JSON:

{
    "intent": "fact_lookup|summary|comparison|analysis",
    "languages": ["de", "en"],
    "keywords": ["keyword1", "keyword2"],
    "synonyms": {
        "keyword1": ["synonym1", "synonym2"],
        "keyword2": ["synonym1"]
    },
    "filters": {
        "doctype": ["pdf", "docx", "eml"],
        "date_range": null,
        "customer": null
    },
    "expanded_queries": ["query variant 1", "query variant 2"],
    "confidence": 0.9
}

BEISPIEL für "A-Fehler in Manteldokumenten":
- keywords: ["A-Fehler", "Manteldokument"]
- synonyms: {"A-Fehler": ["A error", "critical defect", "schwere Abweichung"], "Manteldokument": ["wrapper", "Übersichtsdokument"]}
- expanded_queries: ["A-Fehler Manteldokument", "critical defect wrapper document"]

Keine Erklärungen, nur JSON!"""
    
    def _build_strategy_prompt(self, query: str) -> str:
        return f"""Analysiere folgende Anfrage und erzeuge eine Suchstrategie:

ANFRAGE: {query}

Extrahiere:
1. Was sucht der Benutzer konkret?
2. Welche Dokumententypen könnten relevant sein?
3. Welche Kunden/Projekte werden erwähnt?
4. Welche Synonyme könnten helfen (DE + EN)?

Antworte mit JSON (siehe System-Prompt für Format)."""
    
    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call Ollama with structured output"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 1024},
            "format": "json"  # Request JSON mode if supported
        }
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            r = await client.post(f"{self.ollama_base}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate JSON response"""
        try:
            # Try direct JSON parse
            data = json.loads(response.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                data = json.loads(response[start:end].strip())
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                data = json.loads(response[start:end].strip())
            else:
                raise
        
        # Ensure required fields
        return {
            "intent": data.get("intent", "fact_lookup"),
            "languages": data.get("languages", ["de", "en"]),
            "keywords": data.get("keywords", []),
            "synonyms": data.get("synonyms", {}),
            "filters": data.get("filters", {"doctype": [], "date_range": None, "customer": None}),
            "expanded_queries": data.get("expanded_queries", []),
            "confidence": data.get("confidence", 0.5)
        }
    
    def _expand_synonyms(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Add known synonym mappings to strategy"""
        keywords = strategy.get("keywords", [])
        synonyms = strategy.get("synonyms", {})
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self._term_mappings:
                # Add known mappings
                if keyword not in synonyms:
                    synonyms[keyword] = []
                for syn in self._term_mappings[keyword_lower]:
                    if syn not in synonyms[keyword]:
                        synonyms[keyword].append(syn)
        
        # Generate expanded queries if not provided
        expanded = strategy.get("expanded_queries", [])
        if not expanded and keywords:
            # Create 2-4 query variants with synonyms
            base_query = " ".join(keywords)
            expanded.append(base_query)
            
            # Add EN variant
            en_keywords = []
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in self._term_mappings:
                    en_keywords.append(self._term_mappings[kw_lower][0])
                else:
                    en_keywords.append(kw)
            if en_keywords != keywords:
                expanded.append(" ".join(en_keywords))
        
        strategy["synonyms"] = synonyms
        strategy["expanded_queries"] = expanded[:4]  # Max 4 variants
        
        return strategy
    
    def _get_basic_synonyms(self, query: str) -> Dict[str, List[str]]:
        """Generate basic synonyms for fallback"""
        synonyms = {}
        for word in query.split():
            word_lower = word.lower()
            if word_lower in self._term_mappings:
                synonyms[word] = self._term_mappings[word_lower]
        return synonyms
