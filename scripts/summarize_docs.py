#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Summarizer – OnePager + Erkenntnisse (Management Insights)
===================================================================

Pre-computes structured summaries for all PDF/EML/DOC/DOCX documents
in the Elasticsearch index.  Each summary contains:
  - OnePager (Kontext, Kernaussagen, Zahlen, Risiken, Pendenzen, Entitäten)
  - Erkenntnisse für das Management (systemische, übertragbare Empfehlungen)

Resumable:  writes one JSONL line per document; skips already-processed docs.

Usage:
  # Status check
  python summarize_docs.py --status

  # Pilot: first 50 docs
  python summarize_docs.py --limit 50

  # Continue (auto-resume)
  python summarize_docs.py --limit 2000

  # Full run (all remaining)
  python summarize_docs.py

  # Custom model / ES
  python summarize_docs.py --model gpt-oss:latest --es-url http://localhost:9200
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    backend: str = "ollama"              # "ollama" or "openai" (llama-server)
    ollama_base: str = "http://localhost:11434"
    openai_base: str = "http://localhost:8090"
    ollama_model: str = "gpt-oss:latest"
    es_url: str = "http://localhost:9200"
    es_index: str = "rag_tfk18_v1"
    extensions: Tuple[str, ...] = ("pdf", "eml", "doc", "docx")
    min_content_chars: int = 200          # skip docs with less text
    output_dir: str = ""                  # set via --out
    output_file: str = "summaries.jsonl"
    timeout: int = 600                    # per LLM call
    limit: int = 0                        # 0 = unlimited
    log_level: str = "INFO"

log = logging.getLogger("summarize_docs")

# ---------------------------------------------------------------------------
# Elasticsearch helpers
# ---------------------------------------------------------------------------

def list_documents(cfg: Config) -> List[Dict[str, Any]]:
    """Get list of unique documents from ES with their metadata."""
    docs: Dict[str, Dict[str, Any]] = {}
    
    for ext in cfg.extensions:
        scroll_url = f"{cfg.es_url}/{cfg.es_index}/_search?scroll=2m"
        # Match both "pdf" and ".pdf" (different ES indices use different conventions)
        ext_variants = [ext.lstrip("."), f".{ext.lstrip('.')}"]
        query = {
            "size": 500,
            "query": {"terms": {"file.extension": ext_variants}},
            "_source": ["file.filename", "file.extension", "file.filesize", "path.virtual"],
            "sort": [{"file.filename": "asc"}],
        }
        r = requests.post(scroll_url, json=query, timeout=30)
        r.raise_for_status()
        data = r.json()
        scroll_id = data.get("_scroll_id")
        
        while True:
            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break
            for h in hits:
                src = h["_source"]
                fname = src.get("file", {}).get("filename", "")
                if fname and fname not in docs:
                    docs[fname] = {
                        "filename": fname,
                        "extension": src.get("file", {}).get("extension", ""),
                        "filesize": src.get("file", {}).get("filesize", 0),
                        "path": src.get("path", {}).get("virtual", ""),
                    }
            # Scroll next
            if not scroll_id:
                break
            r = requests.post(
                f"{cfg.es_url}/_search/scroll",
                json={"scroll": "2m", "scroll_id": scroll_id},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        
        # Clear scroll
        if scroll_id:
            try:
                requests.delete(
                    f"{cfg.es_url}/_search/scroll",
                    json={"scroll_id": scroll_id},
                    timeout=10,
                )
            except Exception:
                pass
    
    # Sort by extension priority: eml > doc/docx > pdf (management-relevant first)
    ext_priority = {"eml": 0, "doc": 1, "docx": 1, "pdf": 2}
    return sorted(docs.values(), key=lambda d: (ext_priority.get(d.get("extension", ""), 9), d["filename"]))


def fetch_document_chunks(cfg: Config, filename: str) -> List[str]:
    """Fetch all text chunks for a document, ordered by position."""
    url = f"{cfg.es_url}/{cfg.es_index}/_search"
    query = {
        "size": 200,
        "query": {"term": {"file.filename": filename}},
        "_source": ["content"],
        "sort": ["_doc"],
    }
    r = requests.post(url, json=query, timeout=30)
    r.raise_for_status()
    hits = r.json().get("hits", {}).get("hits", [])
    chunks = []
    for h in hits:
        content = h.get("_source", {}).get("content", "").strip()
        if content:
            chunks.append(content)
    return chunks


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def openai_chat(cfg: Config, messages: List[Dict[str, str]],
                temperature: float = 0.2, num_predict: int = 4096) -> str:
    """OpenAI-compatible /v1/chat/completions (llama-server, vLLM, etc.)."""
    url = cfg.openai_base.rstrip("/") + "/v1/chat/completions"
    timeout = cfg.timeout
    if num_predict > 4096:
        timeout = max(timeout, int(num_predict / 4) + 600)
    payload = {
        "model": cfg.ollama_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": num_predict,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ollama_chat(cfg: Config, messages: List[Dict[str, str]],
                temperature: float = 0.2, num_predict: int = 4096) -> str:
    """Call LLM via Ollama or OpenAI-compatible backend."""
    if cfg.backend == "openai":
        return openai_chat(cfg, messages, temperature, num_predict)
    url = cfg.ollama_base.rstrip("/") + "/api/chat"
    total_chars = sum(len(m.get("content", "")) for m in messages)
    num_ctx = max(4096, int(total_chars / 3) + num_predict + 512)
    num_ctx = min(num_ctx, 65536)
    
    payload = {
        "model": cfg.ollama_model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    r = requests.post(url, json=payload, timeout=cfg.timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def extract_json_from_text(text: str) -> Optional[Any]:
    """Robustly extract JSON from LLM output."""
    text = text.strip()
    # Strip <think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in text and "</think>" not in text:
        for ch in ["{", "["]:
            idx = text.find(ch)
            if idx >= 0:
                text = text[idx:]
                break
    
    for strategy in [
        lambda t: json.loads(t),
        lambda t: json.loads(re.sub(r"```(?:json)?\s*\n?", "", t).rstrip("`").strip()),
        lambda t: json.loads(t[t.find("{"):t.rfind("}") + 1]),
        lambda t: json.loads(t[t.find("["):t.rfind("]") + 1]),
    ]:
        try:
            return strategy(text)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# Document classification
# ---------------------------------------------------------------------------

_CONTRACTUAL_KW = {
    "vertrag", "werkvertrag", "agb", "haftung", "sla", "datenschutz",
    "pönale", "konventionalstrafe", "sia", "vergütung", "abnahme",
    "gewährleistung", "kündigung", "schadenersatz", "offerte", "nachtrag",
}

_EMAIL_KW = {
    "von:", "an:", "betreff:", "gesendet:", "from:", "to:", "subject:", "sent:",
    "cc:", "bcc:", "date:",
}

def classify_document(filename: str, text_start: str) -> str:
    """Classify as technical / contractual / email / minutes."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    lower = text_start[:5000].lower()
    
    if ext == "eml":
        return "email"
    
    # Check for meeting minutes
    minutes_kw = {"protokoll", "sitzung", "traktand", "meeting", "minutes",
                  "teilnehmer", "pendenzen", "beschluss", "agenda"}
    if sum(1 for kw in minutes_kw if kw in lower) >= 3:
        return "minutes"
    
    # Check contractual
    if sum(1 for kw in _CONTRACTUAL_KW if kw in lower) >= 2:
        return "contractual"
    
    return "technical"


# ---------------------------------------------------------------------------
# MAP phase – extract facts per chunk
# ---------------------------------------------------------------------------

MAP_SYSTEM = """Du bist ein exakter Analyst für Schweizer Infrastrukturprojekte.
Extrahiere strukturierte Fakten aus dem Textchunk als JSON.
KEINE Halluzinationen: Wenn etwas nicht im Text steht, lass das Feld leer.
Gib NUR valides JSON zurück, keine Markdown-Fences, kein Fliesstext."""

MAP_USER_TEMPLATE = """Dokumenttyp: {doc_type}
Datei: {filename}
Chunk {chunk_idx}/{total_chunks}

Extrahiere aus dem folgenden TEXTCHUNK die wichtigsten Fakten als JSON:
{{
  "key_points": ["max 5 Kernaussagen"],
  "entities": ["Firmen, Personen, Systeme, Organisationen"],
  "dates_numbers": ["Daten, Beträge, Fristen, Kennzahlen"],
  "risks_issues": ["Risiken, Probleme, offene Punkte"],
  "actions": ["Konkrete To-dos, Pendenzen, Entscheide"],
  "management_insights": [
    "Systemische Prozessprobleme, Governance-Lücken oder übertragbare Erkenntnisse für das Top-Management. "
    "NUR Themen die auf andere Projekte/Organisationen übertragbar sind. "
    "NICHT: technische Einzeldetails, vendor-spezifische Positionen, reine Tatsachenfeststellungen ohne Handlungsempfehlung."
  ]
}}

TEXTCHUNK:
---
{text}
---"""


def map_chunk(cfg: Config, filename: str, doc_type: str,
              chunk_text: str, chunk_idx: int, total_chunks: int) -> Dict[str, Any]:
    """MAP a single chunk to structured facts."""
    user_msg = MAP_USER_TEMPLATE.format(
        doc_type=doc_type,
        filename=filename,
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
        text=chunk_text[:40000],
    )
    
    try:
        resp = ollama_chat(
            cfg,
            [{"role": "system", "content": MAP_SYSTEM},
             {"role": "user", "content": user_msg}],
            temperature=0.1,
            num_predict=2048,
        )
        obj = extract_json_from_text(resp)
        if obj is None:
            obj = {"key_points": [resp[:500]], "_parse_error": True}
        obj["_chunk"] = chunk_idx
        return obj
    except Exception as e:
        log.warning(f"  MAP chunk {chunk_idx} failed: {e}")
        return {"_chunk": chunk_idx, "key_points": [f"Fehler: {e}"], "_error": True}


# ---------------------------------------------------------------------------
# MAP batch – process multiple chunks in a single LLM call
# ---------------------------------------------------------------------------

MAP_BATCH_USER_TEMPLATE = """Dokumenttyp: {doc_type}
Datei: {filename}
Chunks {chunk_range} von {total_chunks}

Extrahiere aus den folgenden {n_batch} TEXTCHUNKS die wichtigsten Fakten als JSON-Array.
Gib ein JSON-Array zurück mit einem Objekt pro Chunk:
[
  {{
    "chunk": <chunk_nummer>,
    "key_points": ["max 5 Kernaussagen"],
    "entities": ["Firmen, Personen, Systeme, Organisationen"],
    "dates_numbers": ["Daten, Beträge, Fristen, Kennzahlen"],
    "risks_issues": ["Risiken, Probleme, offene Punkte"],
    "actions": ["Konkrete To-dos, Pendenzen, Entscheide"],
    "management_insights": ["Systemische Erkenntnisse für das Top-Management"]
  }}
]

{chunks_text}"""


def _split_text_blocks(text: str, target_size: int = 15000,
                       max_blocks: int = 16) -> List[str]:
    """Split text into blocks of ~target_size chars, breaking at paragraph boundaries."""
    if len(text) <= target_size:
        return [text]
    blocks = []
    remaining = text
    while remaining and len(blocks) < max_blocks:
        if len(remaining) <= target_size:
            blocks.append(remaining)
            break
        # Find a paragraph break near the target size
        cut = target_size
        # Look for double-newline (paragraph break) within 20% window
        search_start = int(target_size * 0.8)
        best = remaining.rfind("\n\n", search_start, target_size + 500)
        if best > 0:
            cut = best + 2
        else:
            # Fall back to single newline
            best = remaining.rfind("\n", search_start, target_size + 500)
            if best > 0:
                cut = best + 1
        blocks.append(remaining[:cut])
        remaining = remaining[cut:]
    return blocks


def _batch_chunks(chunks: List[str], batch_size: int = 3,
                  max_chars: int = 24000) -> List[List[tuple]]:
    """Group (index, chunk_text) into batches respecting size limits."""
    batches: List[List[tuple]] = []
    current: List[tuple] = []
    current_chars = 0
    for i, text in enumerate(chunks):
        t = text[:12000]
        if current and (len(current) >= batch_size
                        or current_chars + len(t) > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append((i + 1, t))  # 1-indexed
        current_chars += len(t)
    if current:
        batches.append(current)
    return batches


def map_chunk_batch(cfg: Config, filename: str, doc_type: str,
                    batch: List[tuple], total_chunks: int) -> List[Dict[str, Any]]:
    """MAP multiple chunks in a single LLM call. Returns list of fact dicts."""
    indices = [idx for idx, _ in batch]
    chunk_range = f"{indices[0]}-{indices[-1]}"

    chunks_text = ""
    for idx, text in batch:
        chunks_text += f"\n--- CHUNK {idx} ---\n{text}\n"

    user_msg = MAP_BATCH_USER_TEMPLATE.format(
        doc_type=doc_type,
        filename=filename,
        chunk_range=chunk_range,
        total_chunks=total_chunks,
        n_batch=len(batch),
        chunks_text=chunks_text,
    )

    try:
        resp = ollama_chat(
            cfg,
            [{"role": "system", "content": MAP_SYSTEM},
             {"role": "user", "content": user_msg}],
            temperature=0.1,
            num_predict=2048 * len(batch),
        )
        # Try to parse as JSON array
        parsed = extract_json_from_text(resp)
        if isinstance(parsed, list):
            # Got array — tag each with chunk index
            results = []
            for j, obj in enumerate(parsed):
                if not isinstance(obj, dict):
                    obj = {"key_points": [str(obj)]}
                obj["_chunk"] = indices[j] if j < len(indices) else j + 1
                results.append(obj)
            return results
        elif isinstance(parsed, dict):
            # Got single object — treat as merged facts for all chunks
            parsed["_chunk"] = indices[0]
            return [parsed]
        else:
            # Fallback: couldn't parse
            return [{"_chunk": indices[0], "key_points": [resp[:500]],
                     "_parse_error": True}]
    except Exception as e:
        log.warning(f"  MAP batch {chunk_range} failed: {e}")
        return [{"_chunk": idx, "key_points": [f"Fehler: {e}"],
                 "_error": True} for idx in indices]


# ---------------------------------------------------------------------------
# REDUCE phase – synthesize OnePager + Erkenntnisse
# ---------------------------------------------------------------------------

REDUCE_SYSTEM = """Du bist ein Senior Technical Writer für Schweizer Infrastrukturprojekte.
Erzeuge eine strukturierte Zusammenfassung auf Deutsch (450-650 Wörter).

REGELN:
- KEINE erfundenen Details, KEINE Vermutungen
- ENTFERNE Sektionen VOLLSTÄNDIG (inkl. Überschrift), wenn das Dokument dazu keine Informationen enthält. Schreibe NIEMALS Platzhalter wie "Nicht im Dokument erwähnt", "Keine explizit genannt", "Keine im Dokument definiert", "Keine Risiken … genannt", "Keine Pendenzen … erwähnt" oder ähnliche Leer-Sätze. Wenn nichts da ist, lösche die Sektion komplett.
- Erfinde KEINE Handlungsempfehlungen die nicht im Dokument stehen oder sich nicht logisch ableiten lassen
- Verwende Fachbegriffe korrekt (SIA, FAT, SAT, TFK, etc.)

BESONDERS WICHTIG für "Erkenntnisse für das Management":
- Nur systemische, auf andere Projekte übertragbare Erkenntnisse
- Jede Erkenntnis muss eine konkrete Handlungsempfehlung enthalten
- NICHT: technische Einzeldetails, vendor-/produktspezifische Punkte
- NICHT: reine Tatsachenfeststellungen ohne Empfehlung
- Wenn keine Management-Erkenntnisse ableitbar sind: "Keine übertragbaren Erkenntnisse in diesem Dokument"
"""

REDUCE_USER_TEMPLATE = """Dokumenttyp: {doc_type}
Quelle: {filename}

Verdichte die folgenden extrahierten Chunk-Fakten zu EINEM strukturierten OnePager.

Ausgabeformat (genau so, mit Überschriften):

# {filename}

## Kontext & Zweck
2-3 Sätze

## Kernaussagen
- 5-8 Bullets

## Wichtige Zahlen / Daten / Fristen
- Bullets (falls vorhanden; Sektion weglassen wenn nichts vorhanden)

{extra_section}

## Risiken & offene Punkte
- NUR was im Dokument steht (Sektion weglassen wenn nichts vorhanden)

## Pendenzen / Nächste Schritte
- NUR wenn im Dokument explizit erwähnt (Sektion weglassen wenn nichts vorhanden)

## Erkenntnisse für das Management
- Systemische Prozessprobleme, Governance-Lücken, übertragbare Lessons Learned
- Jede Erkenntnis mit konkreter Handlungsempfehlung für zukünftige Projekte
- NICHT: technische Details, vendor-spezifische Punkte, reine Fakten ohne Empfehlung
- Falls keine: Sektion weglassen

## Relevante Entitäten
Kompakt, komma-separiert

Chunk-Fakten ({n_chunks} Abschnitte):
{facts_json}"""

# ---------------------------------------------------------------------------
# DIRECT mode – single-call text → OnePager (skips MAP for short docs)
# ---------------------------------------------------------------------------

DIRECT_USER_TEMPLATE = """Dokumenttyp: {doc_type}
Quelle: {filename}

WICHTIG: Lies den Text ZUERST vollständig und identifiziere dabei insbesondere:
- Systemische Prozessprobleme, Governance-Lücken, übertragbare Erkenntnisse
- Konkrete Handlungsempfehlungen, die sich für zukünftige Projekte ableiten lassen

Erstelle dann aus dem Dokumenttext EINEN strukturierten OnePager.

Ausgabeformat (genau so, mit Überschriften):

# {filename}

## Kontext & Zweck
2-3 Sätze

## Kernaussagen
- 5-8 Bullets

## Wichtige Zahlen / Daten / Fristen
- Bullets (falls vorhanden; Sektion weglassen wenn nichts vorhanden)

{extra_section}

## Risiken & offene Punkte
- NUR was im Dokument steht (Sektion weglassen wenn nichts vorhanden)

## Pendenzen / Nächste Schritte
- NUR wenn im Dokument explizit erwähnt (Sektion weglassen wenn nichts vorhanden)

## Erkenntnisse für das Management
- Systemische Prozessprobleme, Governance-Lücken, übertragbare Lessons Learned
- Jede Erkenntnis mit konkreter Handlungsempfehlung für zukünftige Projekte
- NICHT: technische Details, vendor-spezifische Punkte, reine Fakten ohne Empfehlung
- Falls keine: Sektion weglassen

## Relevante Entitäten
Kompakt, komma-separiert

DOKUMENTTEXT:
---
{text}
---"""

DIRECT_LIGHT_TEMPLATE = """Quelle: {filename}

Erstelle eine kompakte technische Zusammenfassung.

Ausgabeformat:

# {filename}

## Kurzbeschreibung
2-3 Sätze: Was ist das Dokument, was beschreibt es?

## Technische Eckdaten
- Wichtigste Zahlen, Masse, Konfigurationen

## Relevante Entitäten
Kompakt, komma-separiert

DOKUMENTTEXT:
---
{text}
---"""


def direct_to_onepager(cfg: Config, filename: str, doc_type: str,
                       full_text: str) -> str:
    """Single-call: raw text → OnePager directly (no MAP phase)."""
    if _is_light_doc(filename, doc_type):
        user_msg = DIRECT_LIGHT_TEMPLATE.format(
            filename=filename, text=full_text[:40000])
        return ollama_chat(
            cfg,
            [{"role": "system", "content": REDUCE_LIGHT_SYSTEM},
             {"role": "user", "content": user_msg}],
            temperature=0.2, num_predict=1024)

    if doc_type == "contractual":
        extra_section = "## Vertragliche Pflichten & Konditionen\n- Wesentliche Pflichten beider Parteien (NUR was im Vertrag steht)"
    elif doc_type == "email":
        extra_section = "## Kommunikationskontext\n- Absender/Empfänger-Beziehung, Tonalität, Eskalationsstufe"
    elif doc_type == "minutes":
        extra_section = "## Beschlüsse & Zuweisungen\n- Entscheide mit Verantwortlichen und Fristen"
    else:
        extra_section = "## Technische Kernpunkte\n- Wichtigste technische Aussagen"

    user_msg = DIRECT_USER_TEMPLATE.format(
        doc_type=doc_type, filename=filename,
        extra_section=extra_section, text=full_text[:40000])
    return ollama_chat(
        cfg,
        [{"role": "system", "content": REDUCE_SYSTEM},
         {"role": "user", "content": user_msg}],
        temperature=0.2, num_predict=4096)


# Light template for technical specs (rack layouts, cable measurements, etc.)
REDUCE_LIGHT_SYSTEM = """Du bist ein technischer Dokumentarist. Erzeuge eine KURZE Zusammenfassung (150-250 Wörter) auf Deutsch.
KEINE erfundenen Details. Lasse Sektionen weg, wenn keine Informationen vorhanden sind."""

REDUCE_LIGHT_TEMPLATE = """Quelle: {filename}

Erstelle eine kompakte technische Zusammenfassung.

Ausgabeformat:

# {filename}

## Kurzbeschreibung
2-3 Sätze: Was ist das Dokument, was beschreibt es?

## Technische Eckdaten
- Wichtigste Zahlen, Masse, Konfigurationen

## Relevante Entitäten
Kompakt, komma-separiert

Chunk-Fakten ({n_chunks} Abschnitte):
{facts_json}"""


_TECH_SPEC_KEYWORDS = {
    "racklayout", "raumlayout", "stellplätze", "kabellayout", "kabelplan",
    "scheitelkabel", "lageplan", "grundriss", "schaltplan", "strangschema",
    "montageanleitung", "datenblatt", "datasheet", "messwerte", "messprotokoll",
}

def _is_light_doc(filename: str, doc_type: str) -> bool:
    """Detect technical specs that get the light summary template."""
    if doc_type != "technical":
        return False
    fname_lower = filename.lower()
    return any(kw in fname_lower for kw in _TECH_SPEC_KEYWORDS)


def reduce_to_onepager(cfg: Config, filename: str, doc_type: str,
                       facts: List[Dict[str, Any]]) -> str:
    """REDUCE: synthesize all facts into OnePager + Erkenntnisse."""
    facts_json = json.dumps(facts, ensure_ascii=False, indent=1)
    if len(facts_json) > 30000:
        facts_json = facts_json[:30000] + "\n... (gekürzt)"
    
    # Use light template for technical specs
    if _is_light_doc(filename, doc_type):
        user_msg = REDUCE_LIGHT_TEMPLATE.format(
            filename=filename,
            n_chunks=len(facts),
            facts_json=facts_json,
        )
        return ollama_chat(
            cfg,
            [{"role": "system", "content": REDUCE_LIGHT_SYSTEM},
             {"role": "user", "content": user_msg}],
            temperature=0.2,
            num_predict=1024,
        )
    
    # Full template for management-relevant docs
    if doc_type == "contractual":
        extra_section = "## Vertragliche Pflichten & Konditionen\n- Wesentliche Pflichten beider Parteien (NUR was im Vertrag steht)"
    elif doc_type == "email":
        extra_section = "## Kommunikationskontext\n- Absender/Empfänger-Beziehung, Tonalität, Eskalationsstufe"
    elif doc_type == "minutes":
        extra_section = "## Beschlüsse & Zuweisungen\n- Entscheide mit Verantwortlichen und Fristen"
    else:
        extra_section = "## Technische Kernpunkte\n- Wichtigste technische Aussagen"
    
    user_msg = REDUCE_USER_TEMPLATE.format(
        doc_type=doc_type,
        filename=filename,
        extra_section=extra_section,
        n_chunks=len(facts),
        facts_json=facts_json,
    )
    
    return ollama_chat(
        cfg,
        [{"role": "system", "content": REDUCE_SYSTEM},
         {"role": "user", "content": user_msg}],
        temperature=0.2,
        num_predict=4096,
    )


# ---------------------------------------------------------------------------
# Extract management insights from OnePager text
# ---------------------------------------------------------------------------

def extract_erkenntnisse(onepager_text: str) -> List[str]:
    """Parse the Erkenntnisse section from the OnePager markdown.
    
    Groups multi-line insights (title + erkenntnis + empfehlung) into
    single logical blocks.  Typical LLM output pattern:
      1. **Title**
         - *Erkenntnis*: ...
         - *Handlungsempfehlung*: ...
    """
    insights = []
    in_section = False
    current_block: List[str] = []
    
    for line in onepager_text.split("\n"):
        if "Erkenntnisse" in line and line.strip().startswith("#"):
            in_section = True
            continue
        if not in_section:
            continue
        
        stripped = line.strip()
        
        # Next markdown section → stop
        if stripped.startswith("#"):
            break
        
        # Skip empty lines
        if not stripped:
            continue
        
        # Check for "keine übertragbaren Erkenntnisse"
        clean = stripped.lstrip("- •").strip()
        if "keine" in clean.lower()[:30] and "erkenntnis" in clean.lower()[:60]:
            break
        
        # Detect start of a new numbered insight (e.g. "1. **Title**" or "- **Title**")
        is_new_item = bool(re.match(r"^(\d+[\.\)]\s|\-\s)\*\*", stripped))
        
        if is_new_item:
            # Save previous block
            if current_block:
                insights.append("\n".join(current_block))
            current_block = [clean]
        elif current_block:
            # Continuation of current block
            current_block.append(clean)
        else:
            # Standalone line before any numbered item
            if clean:
                current_block = [clean]
    
    # Don't forget last block
    if current_block:
        insights.append("\n".join(current_block))
    
    return insights


# ---------------------------------------------------------------------------
# Main pipeline: process one document
# ---------------------------------------------------------------------------

def process_document(cfg: Config, doc: Dict[str, Any]) -> Dict[str, Any]:
    """Full MAP-REDUCE pipeline for a single document."""
    filename = doc["filename"]
    t0 = time.time()
    
    # 1. Fetch chunks from ES
    chunks = fetch_document_chunks(cfg, filename)
    if not chunks:
        return {"filename": filename, "status": "no_content", "elapsed_s": 0}
    
    full_text = "\n\n".join(chunks)
    if len(full_text.strip()) < cfg.min_content_chars:
        return {"filename": filename, "status": "too_short",
                "chars": len(full_text), "elapsed_s": 0}
    
    # 2. Classify
    doc_type = classify_document(filename, full_text)
    
    # 3. Generate OnePager
    DIRECT_LIMIT = 40000  # chars ≈ 12k tokens — single call for short docs
    n_es_chunks = len(chunks)
    log.info(f"  {n_es_chunks} ES-chunks, {len(full_text)} chars, type={doc_type}")
    
    all_facts = []
    try:
        if len(full_text) <= DIRECT_LIMIT:
            # DIRECT mode: text → OnePager in 1 call (covers ~99% of docs)
            log.info(f"  DIRECT: 1 call")
            onepager = direct_to_onepager(cfg, filename, doc_type, full_text)
        else:
            # LONG docs: MAP-REDUCE pipeline
            blocks = _split_text_blocks(full_text, target_size=15000, max_blocks=16)
            log.info(f"  MAP-REDUCE: {len(full_text)} chars → {len(blocks)} blocks")
            if len(blocks) <= 2:
                for i, block in enumerate(blocks):
                    facts = map_chunk(cfg, filename, doc_type, block, i + 1, len(blocks))
                    all_facts.append(facts)
            else:
                batches = _batch_chunks(blocks, batch_size=3, max_chars=24000)
                log.info(f"  MAP batched: {len(blocks)} blocks → {len(batches)} calls")
                for batch in batches:
                    facts_list = map_chunk_batch(cfg, filename, doc_type, batch, len(blocks))
                    all_facts.extend(facts_list)
            log.info(f"  REDUCE: synthesizing OnePager...")
            onepager = reduce_to_onepager(cfg, filename, doc_type, all_facts)
    except Exception as e:
        log.error(f"  FAILED: {e}")
        onepager = f"# {filename}\n\n(Verarbeitung fehlgeschlagen: {e})"
    
    # 5. Extract structured management insights
    erkenntnisse = extract_erkenntnisse(onepager)
    
    elapsed = time.time() - t0
    log.info(f"  Done: {len(onepager)} chars, {len(erkenntnisse)} Erkenntnisse, {elapsed:.1f}s")
    
    return {
        "filename": filename,
        "path": doc.get("path", ""),
        "extension": doc.get("extension", ""),
        "filesize": doc.get("filesize", 0),
        "doc_type": doc_type,
        "n_chunks": len(chunks),
        "n_map_chunks": len(all_facts),
        "onepager": onepager,
        "erkenntnisse": erkenntnisse,
        "map_facts": all_facts,
        "status": "ok",
        "elapsed_s": round(elapsed, 1),
        "model": cfg.ollama_model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

def load_processed(output_path: str) -> Set[str]:
    """Load set of already-processed filenames from JSONL."""
    processed = set()
    if not os.path.exists(output_path):
        return processed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("filename"):
                    processed.add(obj["filename"])
            except json.JSONDecodeError:
                continue
    return processed


def show_status(output_path: str, total_docs: int):
    """Print summary status of the summarization run."""
    processed = load_processed(output_path)
    ok = 0
    skipped = 0
    errors = 0
    erkenntnisse_total = 0
    
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    s = obj.get("status", "")
                    if s == "ok":
                        ok += 1
                        erkenntnisse_total += len(obj.get("erkenntnisse", []))
                    elif s in ("no_content", "too_short"):
                        skipped += 1
                    else:
                        errors += 1
                except json.JSONDecodeError:
                    errors += 1
    
    remaining = total_docs - len(processed)
    pct = len(processed) / max(total_docs, 1) * 100
    
    print(f"\n{'='*60}")
    print(f"Summarization Status: {output_path}")
    print(f"{'='*60}")
    print(f"  Total documents:     {total_docs:>6}")
    print(f"  Processed:           {len(processed):>6} ({pct:.1f}%)")
    print(f"    OK:                {ok:>6}")
    print(f"    Skipped (short):   {skipped:>6}")
    print(f"    Errors:            {errors:>6}")
    print(f"  Remaining:           {remaining:>6}")
    print(f"  Erkenntnisse total:  {erkenntnisse_total:>6}")
    if ok > 0:
        print(f"  Ø Erkenntnisse/doc:  {erkenntnisse_total/ok:>6.1f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute document summaries (OnePager + Erkenntnisse)"
    )
    parser.add_argument("--model", default="gpt-oss-120b",
                        help="Model name (default: gpt-oss-120b)")
    parser.add_argument("--backend", default="openai", choices=["ollama", "openai"],
                        help="LLM backend: ollama or openai/llama-server (default: openai)")
    parser.add_argument("--ollama", default="http://localhost:11434",
                        help="Ollama base URL")
    parser.add_argument("--openai-base", default="http://localhost:8090",
                        help="OpenAI-compatible API base URL for llama-server (default: http://localhost:8090)")
    parser.add_argument("--es-url", default="http://localhost:9200",
                        help="Elasticsearch URL")
    parser.add_argument("--es-index", default="rag_tfk18_v1",
                        help="Elasticsearch index")
    parser.add_argument("--out", default="/media/felix/RAG/AGENTIC/runs/summaries",
                        help="Output directory")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max documents to process (0=unlimited)")
    parser.add_argument("--extensions", nargs="+", default=["pdf", "eml", "doc", "docx"],
                        help="File extensions to process")
    parser.add_argument("--min-chars", type=int, default=200,
                        help="Minimum content chars (skip shorter docs)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="LLM call timeout in seconds")
    parser.add_argument("--status", action="store_true",
                        help="Show progress status and exit")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    cfg = Config(
        backend=args.backend,
        ollama_base=args.ollama,
        openai_base=getattr(args, 'openai_base', 'http://localhost:8090'),
        ollama_model=args.model,
        es_url=args.es_url,
        es_index=args.es_index,
        extensions=tuple(args.extensions),
        min_content_chars=args.min_chars,
        output_dir=args.out,
        timeout=args.timeout,
        limit=args.limit,
    )
    
    # Ensure output dir exists
    os.makedirs(cfg.output_dir, exist_ok=True)
    output_path = os.path.join(cfg.output_dir, cfg.output_file)
    
    # List all documents from ES
    log.info("Fetching document list from Elasticsearch...")
    all_docs = list_documents(cfg)
    log.info(f"Found {len(all_docs)} unique documents")
    
    # Status mode
    if args.status:
        show_status(output_path, len(all_docs))
        return
    
    # Load already processed
    processed = load_processed(output_path)
    log.info(f"Already processed: {len(processed)} documents")
    
    # Filter to remaining
    remaining = [d for d in all_docs if d["filename"] not in processed]
    log.info(f"Remaining: {len(remaining)} documents")
    
    if cfg.limit > 0:
        remaining = remaining[:cfg.limit]
        log.info(f"Limited to: {len(remaining)} documents")
    
    if not remaining:
        log.info("Nothing to do — all documents already processed.")
        show_status(output_path, len(all_docs))
        return
    
    # Process documents
    log.info(f"Processing {len(remaining)} documents with {cfg.ollama_model}...")
    log.info(f"Output: {output_path}")
    log.info("="*60)
    
    ok_count = 0
    skip_count = 0
    err_count = 0
    erkenntnisse_count = 0
    t_start = time.time()
    
    for i, doc in enumerate(remaining):
        log.info(f"[{i+1}/{len(remaining)}] {doc['filename']} ({doc.get('filesize',0)//1024}KB, .{doc.get('extension','')})")
        
        try:
            result = process_document(cfg, doc)
            
            # Write to JSONL immediately (crash-safe)
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            status = result.get("status", "error")
            if status == "ok":
                ok_count += 1
                n_erk = len(result.get("erkenntnisse", []))
                erkenntnisse_count += n_erk
                if n_erk > 0:
                    log.info(f"  Erkenntnisse: {n_erk}")
            elif status in ("no_content", "too_short"):
                skip_count += 1
            else:
                err_count += 1
                
        except KeyboardInterrupt:
            log.info("\nInterrupted by user. Progress saved.")
            break
        except Exception as e:
            err_count += 1
            log.error(f"  FAILED: {e}")
            # Write error record so we don't retry broken docs forever
            error_record = {
                "filename": doc["filename"],
                "status": "error",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
        
        # Progress every 10 docs
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_remaining = (len(remaining) - i - 1) / rate if rate > 0 else 0
            log.info(
                f"  Progress: {i+1}/{len(remaining)} | "
                f"OK={ok_count} Skip={skip_count} Err={err_count} | "
                f"Erkenntnisse={erkenntnisse_count} | "
                f"Rate={rate*3600:.0f}/h | "
                f"ETA={eta_remaining/3600:.1f}h"
            )
    
    # Final summary
    elapsed = time.time() - t_start
    log.info("="*60)
    log.info(f"DONE in {elapsed/3600:.1f}h")
    log.info(f"  OK: {ok_count} | Skipped: {skip_count} | Errors: {err_count}")
    log.info(f"  Erkenntnisse: {erkenntnisse_count} total")
    log.info(f"  Output: {output_path}")
    
    show_status(output_path, len(all_docs))


if __name__ == "__main__":
    main()
