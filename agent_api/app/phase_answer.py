"""
phase_answer.py

Phase 5: Answer Agent
Generates final streaming answer based on analyzed documents.
Uses large model (13B+) for high quality responses.
Includes structured citations.
"""

from __future__ import annotations

import json
import httpx
import os
from typing import Any, Dict, List, AsyncGenerator, Optional


class AnswerAgent:
    """Generates final streaming answer with citations"""
    
    def __init__(self, ollama_base: str, model: str):
        self.ollama_base = ollama_base
        self.model = model
    
    async def run_streaming(
        self,
        analyzed_documents: List[Dict[str, Any]],
        strategy: Dict[str, Any],
        original_query: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream final answer with token-by-token output.
        Yields: {"type": "token", "content": "..."} and {"type": "sources", "sources": [...]}
        """
        # Build context from analyzed documents
        context = self._build_context(analyzed_documents)
        
        # Build prompt
        messages = self._build_messages(original_query, context, strategy)
        
        # Stream tokens
        full_response = []
        try:
            async for token in self._llm_stream(messages):
                full_response.append(token)
                yield {"type": "token", "content": token}
        except Exception as e:
            yield {"type": "token", "content": f"\n[Fehler bei Antwort-Generierung: {str(e)}]\n"}
        
        # Extract and yield sources
        sources = self._extract_sources(analyzed_documents)
        yield {"type": "sources", "sources": sources}
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from analyzed documents"""
        context_parts = []
        
        for i, doc in enumerate(documents[:8], 1):  # Max 8 documents
            path = doc.get("path", "unknown")
            findings = doc.get("extracted_findings", [])
            
            if not findings:
                continue
            
            doc_section = [f"\n[Quelle {i}: {path}]"]
            
            # Add key findings
            for finding in findings[:5]:  # Max 5 findings per doc
                f_type = finding.get("type", "unknown")
                
                if f_type == "finding":
                    category = finding.get("category", "")
                    desc = finding.get("description", "") or finding.get("content", "")
                    severity = finding.get("severity", "")
                    doc_section.append(f"  Befund [{category}] ({severity}): {desc[:300]}")
                    
                elif f_type == "fact":
                    content = finding.get("content", "")
                    doc_section.append(f"  Fakt: {content[:300]}")
                    
                elif f_type == "summary":
                    content = finding.get("content", "")
                    doc_section.append(f"  Zusammenfassung: {content[:400]}")
                    
                else:
                    content = finding.get("content", "") or finding.get("description", "")
                    if content:
                        doc_section.append(f"  {content[:300]}")
            
            context_parts.append("\n".join(doc_section))
        
        return "\n".join(context_parts)
    
    def _build_messages(
        self,
        query: str,
        context: str,
        strategy: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Build message list for LLM"""
        
        intent = strategy.get("intent", "fact_lookup")
        
        # System prompt based on intent
        if intent == "analysis":
            system_prompt = self._get_analysis_prompt()
        elif intent == "summary":
            system_prompt = self._get_summary_prompt()
        elif intent == "comparison":
            system_prompt = self._get_comparison_prompt()
        else:  # fact_lookup
            system_prompt = self._get_fact_prompt()
        
        # User prompt with context
        user_prompt = f"""Anfrage: {query}

Kontext aus analysierten Dokumenten:
{context}

Antworte basierend auf den obigen Dokumenten. Zitiere die Quellen mit [Quelle N] Format."""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _get_fact_prompt(self) -> str:
        return """Du bist ein Fakten-basierter Informations-Agent. 

REGEL 1 - ENTSCHEIDUNGSHIERARCHIE:
• Wenn Dokumente klare Informationen enthalten: DEFINITIVE Aussagen machen
• Wenn Dokumente widersprüchlich sind: Widerspruch explizit nennen
• Wenn keine Informationen vorhanden: "Keine Informationen in den Dokumenten gefunden"
• NIE "es scheint", "möglicherweise", "anscheinend" verwenden

REGEL 2 - ANTWORTFORMAT:
1. Beginne DIREKT mit dem Gesamtbild/Ergebnis (1-2 Sätze)
2. Dann detaillierte Belege mit Quellen
3. Jede konkrete Aussage muss [Quelle N] zitieren

REGEL 3 - VERBOTENE FORMULIERUNGEN:
- "Es scheint, dass..."
- "Möglicherweise..." / "Vielleicht..."
- "Anschließend..." / "Offenbar..."
- "Ich habe analysiert..."
- "Basierend auf meiner Analyse..."
- "Laut Dokument [N] scheint..." → Stattdessen: "Dokument [N] zeigt:..."

BEISPIEL für korrekte Antwort:
"Der FAT-Bericht vom März 2024 enthält 3 A-Befunde und 2 B-Befunde [Quelle 1].

A-Befunde (kritisch):
1. Heat Exchanger Leak in Modul X [Quelle 1, Seite 4]
2. Control System Failure [Quelle 1, Seite 5]  
3. Pressure Valve malfunction [Quelle 1, Seite 6]

B-Befunde (moderat):
1. Dokumentation unvollständig [Quelle 1, Seite 8]

Status: Alle A-Befunde wurden behoben [Quelle 2]."

BEISPIEL für "nicht gefunden":
"In den vorliegenden Dokumenten wurden keine FAT-Befunde für Kunde X gefunden."""  
    
    def _get_analysis_prompt(self) -> str:
        return """Du bist ein analytischer Dokumenten-Experte. Entscheidungshierarchie gilt auch hier.

ANTWORTFORMAT:
1. Executive Summary: Das Gesamtbild in 1-2 Sätzen (DEFINITIV, keine Floskeln)
2. Detaillierte Analyse strukturiert nach:
   - Kategorie/Schweregrad
   - Zeitraum
   - Status
3. Tabellen für komplexe Daten
4. Jede Aussage mit [Quelle N] belegen
5. Widersprüche explizit markieren

VERBOTEN:
- "Es könnte sein, dass..."
- "Die Analyse zeigt möglicherweise..."
- Eingängige Phrasen ohne Fakten-Backup
- Interpretationen ohne Textbeleg"""
    
    def _get_summary_prompt(self) -> str:
        return """Du bist ein Zusammenfassungs-Experte. Erstelle prägnante, faktenbasierte Zusammenfassungen.

ANTWORTFORMAT:
1. Kern-Ergebnis (2-3 Sätze, definitiv)
2. Wichtige Details mit Quellen
3. Offene Punkte (falls vorhanden)

LÄNGE: 150-400 Wörter je nach Komplexität

VERBOTEN:
- "Zusammenfassend lässt sich sagen..."
- "Es scheint, als ob..."
- Vage Verweise auf "die Dokumente" ohne konkrete Aussagen"""
    
    def _get_comparison_prompt(self) -> str:
        return """Du bist ein Vergleichs-Analyst.

ANTWORTFORMAT:
1. Klare Vergleichsstruktur (seitlich nebeneinander oder tabellarisch)
2. Alle Vergleichspunkte mit [Quelle N] belegen
3. Unterschiede UND Gemeinsamkeiten nennen
4. Abschließende Bewertung

BEISPIEL:
"VERGLEICH: FAT vs SAT Ergebnisse

| Aspekt | FAT [Quelle 1] | SAT [Quelle 2] |
|--------|---------------|---------------|
| Datum | 10.03.2024 | 15.04.2024 |
| A-Befunde | 3 | 1 |
| Status | 2 offen | Alle geschlossen |

Wichtige Unterschiede:
- FAT fand 3 kritische Fehler, SAT nur 1 [Quellen 1,2]
- Reparaturen zwischen FAT und SAT erfolgreich [Quelle 3]

Gemeinsamkeiten:
- Beide Tests durch Kunde X abgenommen [Quellen 1,2]"

VERBOTEN:
- Nicht belegte Vergleiche
- Subjektive Bewertungen ohne Faktenbasis"""
    
    async def _llm_stream(
        self,
        messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from LLM"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.3, "num_predict": 4096}
        }
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            async with client.stream("POST", f"{self.ollama_base}/api/chat", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        msg = obj.get("message", {})
                        content = msg.get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue
    
    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information for final attribution"""
        sources = []
        file_base = os.getenv("FILE_BASE", "")
        
        for i, doc in enumerate(documents[:10], 1):
            path = doc.get("path", "")
            if not path:
                continue
            
            # Try to create clickable URL
            url = ""
            try:
                from urllib.parse import quote
                full_path = path
                if file_base and not path.startswith("/"):
                    full_path = f"{file_base}/{path}"
                url = f"http://localhost:11436/open?path={quote(full_path)}"
            except Exception:
                pass
            
            # Count findings
            findings = doc.get("extracted_findings", [])
            finding_count = len(findings)
            
            # Get document type
            doc_type = doc.get("type", "unknown")
            
            sources.append({
                "n": i,
                "path": path,
                "url": url,
                "type": doc_type,
                "finding_count": finding_count,
                "display": f"[{i}] {path}"
            })
        
        return sources
