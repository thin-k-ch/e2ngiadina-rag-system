#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resumable batch pipeline for RAG-based project report enrichment.
=================================================================

Designed for the TFK18 Schlussbericht: Extracts structured claims from
all indexed documents, clusters them into findings, and generates
report suggestions aligned with your existing draft outline.

Phases:
  0) topics.yaml  – search topics (optional, defaults provided)
  1) candidates   – retrieve relevant chunks from ES (+ optional Chroma)
  2) MAP          – extract structured claims per chunk via Ollama
  3) REDUCE       – hierarchical clustering of claims into findings
  4) DRAFT        – generate report suggestions from findings + outline

All artifacts are written to ./runs/<run_id>/. Each phase checks for
existing outputs and resumes safely.

Usage:
  # Full run with TFK18 index:
  python batch_report.py --es-index rag_tfk18_v1 --model gpt-oss:latest

  # With custom topics and outline:
  python batch_report.py --es-index rag_tfk18_v1 --topics topics.yaml --outline mein_bericht.md

  # Resume from MAP phase:
  python batch_report.py --run-id 20260217_120000_abc123 --start-at map

  # Use smaller model for speed:
  python batch_report.py --es-index rag_tfk18_v1 --model qwen2.5:3b --max-candidates 500

Requires: Python 3.10+, pip install requests pyyaml tqdm elasticsearch
"""

import os
import re
import sys
import json
import time
import uuid
import hashlib
import logging
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    run_id: str
    out_dir: str

    # LLM backend
    backend: str = "ollama"  # "ollama" or "openai" (llama-server, vLLM, etc.)
    ollama_base: str = "http://localhost:11434"
    ollama_model: str = "glm-4.7-flash:latest"
    ollama_timeout_s: int = 1800  # Base timeout; REDUCE/DRAFT get extra time dynamically
    openai_base: str = "http://localhost:8090"  # llama-server default

    # Elasticsearch
    es_url: str = "http://localhost:9200"
    es_index: str = "rag_tfk18_v1"

    # Candidate limits
    max_candidates: int = 3000
    min_chars: int = 200
    max_input_chars: int = 6000

    # File extension filter (skip noise: images, measurement data, CAD, binaries)
    include_extensions: tuple = (
        "pdf", "eml", "docx", "doc", "xlsx", "xls", "txt", "md",
        "msg", "pptx", "ppt", "csv", "html", "htm", "rtf",
    )

    # MAP phase
    map_retry: int = 2
    map_sleep_s: float = 0.1
    map_workers: int = 4  # Concurrent Ollama requests for MAP phase
    map_model: str = ""  # If set, use this model for MAP (e.g. qwen2.5:3b); otherwise use ollama_model

    # REDUCE phase
    reduce_batch_size: int = 20   # Claims per hierarchical REDUCE batch (smaller = more reliable with large models)
    max_findings: int = 120

    # Verbosity
    log_level: str = "INFO"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(out_dir: str, level: str = "INFO") -> logging.Logger:
    ensure_dir(out_dir)
    logger = logging.getLogger("batch_report")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(out_dir, "run.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_jsonl(path: str, row: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# ---------------------------------------------------------------------------
# Ollama client (/api/chat with dynamic num_ctx)
# ---------------------------------------------------------------------------

def openai_chat(cfg: Config, messages: List[Dict[str, str]],
                temperature: float = 0.2, num_predict: int = 4096,
                model_override: str = "") -> str:
    """OpenAI-compatible /v1/chat/completions (llama-server, vLLM, etc.)."""
    url = cfg.openai_base.rstrip("/") + "/v1/chat/completions"
    timeout = cfg.ollama_timeout_s
    if num_predict > 4096:
        timeout = max(timeout, int(num_predict / 4) + 600)
    payload = {
        "model": model_override or cfg.ollama_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": num_predict,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ollama_chat(cfg: Config, messages: List[Dict[str, str]],
                temperature: float = 0.2, num_predict: int = 4096,
                model_override: str = "") -> str:
    """Ollama /api/chat with dynamic num_ctx and timeout based on input size."""
    if cfg.backend == "openai":
        return openai_chat(cfg, messages, temperature, num_predict, model_override)
    url = cfg.ollama_base.rstrip("/") + "/api/chat"
    total_chars = sum(len(m.get("content", "")) for m in messages)
    num_ctx = max(4096, int(total_chars / 3) + num_predict + 512)
    num_ctx = min(num_ctx, 65536)

    # Dynamic timeout: base + extra for large outputs (REDUCE/DRAFT need more)
    timeout = cfg.ollama_timeout_s
    if num_predict > 4096:
        timeout = max(timeout, int(num_predict / 4) + 600)  # ~30min for 8K output

    payload = {
        "model": model_override or cfg.ollama_model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "num_predict": num_predict,
        }
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def extract_json_from_text(text: str) -> Optional[Any]:
    """Robustly extract JSON object or array from model output."""
    text = text.strip()
    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Strategy 2: strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", text)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Strategy 3: find first { or [
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        first = text.find(start_char)
        last = text.rfind(end_char)
        if first != -1 and last > first:
            try:
                return json.loads(text[first:last + 1])
            except Exception:
                pass
    # Strategy 4: repair truncated JSON array by finding last complete top-level item
    arr_start = text.find("[")
    if arr_start != -1:
        # Walk through tracking bracket depth to find complete top-level objects
        s = text[arr_start:]
        depth = 0
        in_string = False
        escape = False
        last_complete_end = -1  # position after last complete top-level }
        for i, ch in enumerate(s):
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in '{[':
                depth += 1
            elif ch in '}]':
                depth -= 1
                if depth == 1 and ch == '}':
                    # Just closed a top-level object inside the array
                    last_complete_end = i + 1
        if last_complete_end > 0:
            try:
                return json.loads(s[:last_complete_end] + "]")
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Default topics (used if no --topics file provided)
# ---------------------------------------------------------------------------

DEFAULT_TOPICS = [
    # --- Aligned to Schlussbericht PROFUMO Findings ---
    {"id": "abnahme_qualitaet", "label": "Abnahmefähigkeit & Qualitätsgates",
     "queries": ["Abnahme Qualität Gate Kriterien", "bereit zur Abnahme Freigabe",
                 "Abnahmeprotokoll Lieferobjekt Dossier", "First-Time-Right Nacharbeit Rework",
                 "Definition of Done Abnahmekriterien", "Qualitätsmangel Zurückweisung"]},
    {"id": "traceability_nachweis", "label": "Nachweis-Disziplin & Traceability",
     "queries": ["Traceability Nachvollziehbarkeit Rohdaten", "Messbericht Auswertung Korrektur",
                 "Datenbereinigung Neuauswertung zurückgewiesen", "Dokumentation unvollständig fehlend",
                 "Nachweisführung Messung Protokoll"]},
    {"id": "change_control_freeze", "label": "Design-Freeze & Change Control",
     "queries": ["Design Freeze Änderung Change", "Nachmessung Anpassung ungeplant",
                 "Change Control Board Änderungsmanagement", "Rework Spirale Korrekturschleifen",
                 "Änderungsindex Konfiguration Versionsstand"]},
    {"id": "schnittstellen_interfaces", "label": "Schnittstellenverantwortung & Interfaces",
     "queries": ["Schnittstelle Verantwortung Ownership", "Handover Zone Koordination",
                 "Interface Register RACI Zuständigkeit", "Systemübergang Integration Abstimmung"]},
    {"id": "externe_abhaengigkeiten", "label": "Externe Abhängigkeiten (SBB/BAV)",
     "queries": ["SBB Messzug Abhängigkeit Verfügbarkeit", "BAV Zulassung Genehmigung",
                 "Abnahmegeschwindigkeit Vmax externe Instanz", "Fallback Szenario Alternative",
                 "Lokführer Bewilligung EVU"]},
    {"id": "lbt_krise_technik", "label": "LBT-Krise & HF-Technik",
     "queries": ["Signalschwund Interferenz Leckkabel", "LBT Lötschberg Basistunnel Krise",
                 "B-Kabel A-Kabel Wandabstand Reflexion", "PIM Passive Intermodulation Messung",
                 "GSM-R Frequenz Dämpfung Abdeckung", "Enotrac Simulation Wellenausbreitung"]},
    {"id": "kosten_finanzen", "label": "Finanzielle Eskalation & Budget",
     "queries": ["Kredit Aufstockung Nachkredit Budget", "Kostenüberschreitung Endkostenprognose",
                 "Mehrkosten Zusatzaufwand Nachtrag", "Vergabe freihändig Krisenintervention",
                 "Kostentreiber Rework Messfahrten ungeplant"]},
    {"id": "lieferanten_commscope", "label": "Lieferanten & CommScope",
     "queries": ["CommScope Lieferant Mangel Kriterium", "Lieferantensteuerung Vertrag Pflicht",
                 "Muss-Kriterium nicht erfüllt Lieferant", "Gewährleistung Mängelrüge Nachbesserung"]},
    {"id": "organisation_governance", "label": "Organisation & Governance",
     "queries": ["Eskalation Geschäftsleitung Management", "Governance Steuerung Projektorganisation",
                 "Kernteam Kommunikation Koordination", "PMO Projektmanagement Standard"]},
    {"id": "entscheidungen_timeline", "label": "Schlüsselentscheidungen & Timeline",
     "queries": ["Entscheidung Beschluss Genehmigung Meilenstein", "Wendepunkt Eskalation kritisch",
                 "Timeline Chronologie Phasenübergang", "Statusbericht Fortschritt Prognose"]},
    {"id": "was_gut_lief", "label": "Was gut funktioniert hat",
     "queries": ["Erfolg wirksam bewährt funktioniert", "Best Practice Standardisierung Vorbild",
                 "Krisenhebel Massnahme wirksam Lösung", "Verbesserung gelungen Ergebnis positiv"]},
    {"id": "empfehlungen_zukunft", "label": "Empfehlungen & Lessons Learned",
     "queries": ["Empfehlung Verbesserung künftig vermeiden", "Lessons Learned Erkenntnis",
                 "sollte muss zukünftig verbessert", "Standardisierung Institutionalisierung Prozess"]},
]


def load_topics(path: Optional[str]) -> List[Dict[str, Any]]:
    """Load topics from YAML or use defaults."""
    if path and os.path.exists(path):
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "topics" in data:
            return data["topics"]
    return DEFAULT_TOPICS


# ---------------------------------------------------------------------------
# Phase 1: Candidate Selection (ES)
# ---------------------------------------------------------------------------

def search_es(cfg: Config, query: str, size: int = 100) -> List[Dict[str, Any]]:
    """BM25 search against ES index. Returns list of {doc_id, path, content, score}.
    Filters by file extension to skip noise (images, measurements, CAD)."""
    url = f"{cfg.es_url}/{cfg.es_index}/_search"

    # Build extension filter: match any of the allowed extensions in the path
    ext_should = [{"wildcard": {"path.virtual": f"*.{ext}"}} for ext in cfg.include_extensions]

    payload = {
        "size": size,
        "query": {
            "bool": {
                "must": [{"match": {"content": {"query": query, "operator": "or"}}}],
                "filter": [
                    {"bool": {"should": ext_should, "minimum_should_match": 1}},
                ],
            }
        },
        "_source": ["content", "path"],
    }
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return []

    results = []
    for hit in data.get("hits", {}).get("hits", []):
        src = hit.get("_source", {})
        path_obj = src.get("path", {})
        if isinstance(path_obj, dict):
            path_str = path_obj.get("virtual", "") or path_obj.get("real", "")
        else:
            path_str = str(path_obj)
        content = str(src.get("content", ""))
        if len(content) < cfg.min_chars:
            continue
        results.append({
            "doc_id": hit["_id"],
            "path": path_str,
            "content": content[:cfg.max_input_chars],
            "score": hit.get("_score", 0),
        })
    return results


def step_candidates(cfg: Config, logger: logging.Logger,
                    topics: List[Dict], candidates_file: Optional[str]) -> str:
    """Phase 1: Collect candidate chunks from ES (topic-guided) or from file."""
    out_path = os.path.join(cfg.out_dir, "candidates.jsonl")
    if os.path.exists(out_path) and count_jsonl(out_path) > 0:
        n = count_jsonl(out_path)
        logger.info(f"[resume] candidates exists: {n} entries")
        return out_path

    if candidates_file:
        logger.info(f"Loading candidates from file: {candidates_file}")
        for row in iter_jsonl(candidates_file):
            txt = (row.get("text") or row.get("content") or "").strip()
            if len(txt) >= cfg.min_chars:
                row["content"] = txt[:cfg.max_input_chars]
                row.setdefault("doc_id", sha1(txt))
                row.setdefault("path", "")
                append_jsonl(out_path, row)
        n = count_jsonl(out_path)
        logger.info(f"Loaded {n} candidates from file")
        return out_path

    # Topic-guided ES search with fair distribution across topics
    seen_ids = set()
    total = 0
    per_topic_budget = max(50, cfg.max_candidates // len(topics)) if topics else cfg.max_candidates

    for topic in topics:
        queries = topic.get("queries", [])
        topic_id = topic.get("id", "unknown")
        topic_hits = 0

        for query in queries:
            if total >= cfg.max_candidates or topic_hits >= per_topic_budget:
                break
            results = search_es(cfg, query, size=200)
            for r in results:
                if total >= cfg.max_candidates or topic_hits >= per_topic_budget:
                    break
                if r["doc_id"] in seen_ids:
                    continue
                seen_ids.add(r["doc_id"])
                r["topic"] = topic_id
                r["query"] = query
                append_jsonl(out_path, r)
                total += 1
                topic_hits += 1

        logger.info(f"  Topic '{topic_id}': {topic_hits} candidates ({len(queries)} queries)")

    logger.info(f"CANDIDATES: {total} total (deduped by doc_id)")
    return out_path


# ---------------------------------------------------------------------------
# Phase 2: MAP – Extract structured claims per chunk
# ---------------------------------------------------------------------------

MAP_SYSTEM = """Du bist ein präziser Analyst für Schweizer Infrastrukturprojekte.
Du extrahierst nur belegbare Aussagen aus Textfragmenten.
Du erfindest nichts. Wenn etwas nicht im Text steht, setze es auf null.
Antworte AUSSCHLIESSLICH mit einem JSON-Objekt (kein Fliesstext, keine Markdown-Fences)."""

MAP_USER_TEMPLATE = """Extrahiere aus diesem Dokument-Fragment strukturierte Claims für einen Schlussbericht.

Antworte als JSON:
{{
  "finding_candidate": "Kernaussage als 1-2 Sätze (oder null wenn keine relevante Aussage)",
  "recommendation_candidate": "Daraus abgeleitete Empfehlung (oder null)",
  "category": "Tech|Process|Org|Security|Cost|Operations|Other",
  "impact": "High|Med|Low|Unknown",
  "signals": ["issue|decision|incident|root_cause|risk|mitigation|metric|milestone"],
  "evidence_quote": "Wörtliches Zitat, max 2 Sätze (oder null)",
  "time_ref": "Datum/Zeitraum falls erwähnt (oder null)",
  "confidence": 0.0 bis 1.0
}}

Dokument: {path}
Topic: {topic}

TEXT:
\"\"\"{text}\"\"\"
"""


def _map_one_candidate(cfg: Config, cand: Dict[str, Any]) -> Tuple[str, Optional[Dict], Optional[str]]:
    """Process a single candidate through MAP. Returns (doc_id, claim_obj_or_None, error_or_None).
    Thread-safe: no shared mutable state."""
    doc_id = cand.get("doc_id", "")
    path = cand.get("path", "")
    text = cand.get("content", "")
    topic = cand.get("topic", "")

    prompt = MAP_USER_TEMPLATE.format(
        path=path, topic=topic, text=text[:cfg.max_input_chars]
    )

    last_err = None
    for attempt in range(cfg.map_retry + 1):
        try:
            raw = ollama_chat(
                cfg,
                messages=[
                    {"role": "system", "content": MAP_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                num_predict=2048,
                model_override=cfg.map_model,
            )
            obj = extract_json_from_text(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"Not a JSON object: {raw[:200]}")

            obj["doc_id"] = doc_id
            obj["path"] = path
            obj["topic"] = topic
            obj.setdefault("confidence", 0.3)
            obj.setdefault("finding_candidate", None)
            obj.setdefault("recommendation_candidate", None)

            if obj.get("finding_candidate") or obj.get("recommendation_candidate"):
                return (doc_id, obj, None)
            return (doc_id, None, None)  # Empty claim, but no error
        except Exception as e:
            last_err = str(e)
            time.sleep(0.5 + attempt)

    return (doc_id, None, last_err)


def step_map(cfg: Config, logger: logging.Logger, candidates_path: str) -> str:
    """Phase 2: MAP – Extract one claim per candidate chunk (concurrent)."""
    out_path = os.path.join(cfg.out_dir, "claims.jsonl")
    done_path = os.path.join(cfg.out_dir, "map_done.json")
    errors_path = os.path.join(cfg.out_dir, "map_errors.jsonl")

    done_ids = set()
    if os.path.exists(done_path):
        done_ids = set(read_json(done_path))

    candidates = list(iter_jsonl(candidates_path))
    remaining = [c for c in candidates if c.get("doc_id", "") not in done_ids]

    map_model_name = cfg.map_model or cfg.ollama_model
    logger.info(f"MAP: {len(candidates)} candidates, {len(done_ids)} already done, {len(remaining)} remaining")
    logger.info(f"MAP model: {map_model_name} | workers: {cfg.map_workers}")

    if not remaining:
        logger.info("[resume] MAP already complete")
        return out_path

    success = 0
    fail = 0
    t_start = time.time()
    write_lock = threading.Lock()

    def on_result(doc_id, claim_obj, error, path=""):
        nonlocal success, fail
        with write_lock:
            if error:
                fail += 1
                append_jsonl(errors_path, {"doc_id": doc_id, "path": path, "error": error})
            else:
                success += 1
                if claim_obj:
                    append_jsonl(out_path, claim_obj)
            done_ids.add(doc_id)
            total = success + fail
            # Checkpoint every 50
            if total % 50 == 0 and total > 0:
                write_json(done_path, sorted(done_ids))
                elapsed = time.time() - t_start
                rate = total / elapsed if elapsed > 0 else 0
                eta_s = (len(remaining) - total) / rate if rate > 0 else 0
                logger.info(
                    f"MAP checkpoint: {success}✅ {fail}❌ | "
                    f"{rate:.2f} docs/s | ETA: {eta_s/60:.0f} min"
                )

    with ThreadPoolExecutor(max_workers=cfg.map_workers) as pool:
        futures = {}
        for cand in remaining:
            f = pool.submit(_map_one_candidate, cfg, cand)
            futures[f] = cand

        pbar = tqdm(total=len(remaining), desc=f"MAP ({cfg.map_workers}w)", initial=0)
        for future in as_completed(futures):
            cand = futures[future]
            try:
                doc_id, claim_obj, error = future.result()
                on_result(doc_id, claim_obj, error, path=cand.get("path", ""))
            except Exception as e:
                doc_id = cand.get("doc_id", "")
                on_result(doc_id, None, str(e), path=cand.get("path", ""))
            pbar.update(1)
        pbar.close()

    write_json(done_path, sorted(done_ids))
    total_claims = count_jsonl(out_path)
    elapsed = time.time() - t_start
    logger.info(f"MAP done: {success}✅ {fail}❌ | {total_claims} claims | {elapsed/60:.1f} min | {cfg.map_workers} workers")
    return out_path


# ---------------------------------------------------------------------------
# Phase 3: REDUCE – Hierarchical clustering of claims into findings
# ---------------------------------------------------------------------------

REDUCE_SYSTEM = """Du bist ein Senior-Reviewer für Schlussberichte in Schweizer Infrastrukturprojekten.
Du verdichtest Claims zu managementtauglichen Findings.

KRITISCHE REGELN:
1. DEDUPLIZIERUNG: Wenn mehrere Claims dasselbe Thema beschreiben, fasse sie zu EINEM Finding zusammen.
   NIEMALS zwei Findings mit demselben Kerninhalt erstellen.
2. EVIDENZ-ZUORDNUNG: Jedes Finding MUSS die spezifischen Dokumente zitieren, aus denen es stammt.
   Kopiere doc_id und path EXAKT aus den Claims. Erfinde KEINE Pfade.
3. EIGENSTÄNDIGE FORMULIERUNG: Formuliere statement und recommendation in eigenen Worten als Synthese.
   Kopiere NICHT einfach den Claim-Text. Verdichte und abstrahiere.
4. KONKRETHEIT: Nenne konkrete Zahlen, Daten, Firmennamen wenn in Claims vorhanden.
5. Antworte NUR als JSON-Liste. Kein Markdown, kein Kommentar davor oder danach."""

REDUCE_USER_TEMPLATE = """Verdichte diese {n} Claims zu maximal {max_f} Findings.

SCHRITT 1 – GRUPPIEREN: Identifiziere thematische Cluster (z.B. alle Claims zu Abnahme, alle zu Kosten).
SCHRITT 2 – DEDUPLIZIEREN: Claims mit gleichem Kerninhalt → EIN Finding. Zähle wie viele Claims zusammengefasst wurden.
SCHRITT 3 – SYNTHETISIEREN: Formuliere pro Cluster ein eigenständiges Finding mit:
  - title: Prägnanter Titel (max 10 Wörter), der das Kernthema benennt
  - category: Genau EINE aus [Tech, Operations, Cost, Process, Management, Risk, Other]
  - impact: Genau EINES aus [High, Med, Low]
  - statement: 2-4 Sätze Synthese. Nenne konkrete Fakten (Zahlen, Daten, Namen) aus den Claims.
  - recommendation: 1-2 Sätze konkrete Handlungsempfehlung. NICHT generisch.
  - evidence: Liste mit doc_id + path + quote AUS DEN CLAIMS (nicht erfinden!)
    → Nimm die VERSCHIEDENEN Dokumente aus den zusammengefassten Claims
    → NICHT alle Evidenzen aus demselben Dokument
  - confidence: Durchschnitt der Claim-Confidences (0.0-1.0)

WICHTIG:
- Jedes Finding muss Evidenz aus MINDESTENS 2 verschiedenen Dokumenten haben (wenn verfügbar)
- Titel müssen EINZIGARTIG sein – keine zwei Findings mit gleichem/ähnlichem Titel
- Generische Findings wie "Allgemeine Projektprobleme" vermeiden

CLAIMS:
{claims_json}

Antworte als JSON-Liste: [{{"title": "...", "category": "...", "impact": "High|Med|Low", "statement": "...", "recommendation": "...", "evidence": [{{"doc_id": "...", "path": "...", "quote": "..."}}], "confidence": 0.0-1.0}}]"""


def step_reduce(cfg: Config, logger: logging.Logger, claims_path: str) -> str:
    """Phase 3: Hierarchical REDUCE – batch claims, then merge batches."""
    out_path = os.path.join(cfg.out_dir, "findings.json")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10:
        logger.info(f"[resume] findings exists: {out_path}")
        return out_path

    claims = list(iter_jsonl(claims_path))
    # Filter: only claims with actual content
    useful = [c for c in claims if c.get("finding_candidate") or c.get("recommendation_candidate")]
    # Sort by confidence descending
    useful.sort(key=lambda x: float(x.get("confidence", 0) or 0), reverse=True)

    logger.info(f"REDUCE: {len(claims)} total claims, {len(useful)} useful")

    if not useful:
        write_json(out_path, [])
        return out_path

    # --- Hierarchical REDUCE: batch → intermediate findings → final merge ---
    batch_size = cfg.reduce_batch_size
    batches = [useful[i:i + batch_size] for i in range(0, len(useful), batch_size)]
    logger.info(f"REDUCE: {len(batches)} batches of ~{batch_size} claims each")

    all_intermediate = []

    for bi, batch in enumerate(batches):
        # Compact claims for prompt (only essential fields)
        compact = []
        for c in batch:
            compact.append({
                "doc_id": c.get("doc_id", ""),
                "path": str(c.get("path", ""))[-80:],  # Last 80 chars of path
                "category": c.get("category", ""),
                "impact": c.get("impact", ""),
                "finding": c.get("finding_candidate", ""),
                "recommendation": c.get("recommendation_candidate", ""),
                "evidence": c.get("evidence_quote", ""),
                "confidence": c.get("confidence", 0),
            })

        max_f = max(3, min(15, len(batch) // 3))
        prompt = REDUCE_USER_TEMPLATE.format(
            n=len(compact), max_f=max_f,
            claims_json=json.dumps(compact, ensure_ascii=False, indent=1)
        )

        try:
            raw = ollama_chat(
                cfg,
                messages=[
                    {"role": "system", "content": REDUCE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                num_predict=8192,
            )
            findings = extract_json_from_text(raw)
            if isinstance(findings, list):
                all_intermediate.extend(findings)
                logger.info(f"  Batch {bi+1}/{len(batches)}: {len(findings)} findings")
            elif isinstance(findings, dict):
                all_intermediate.append(findings)
                logger.info(f"  Batch {bi+1}/{len(batches)}: 1 finding (dict)")
            else:
                logger.warning(f"  Batch {bi+1}: invalid JSON, saving raw for debug")
                # Save raw for debugging
                write_json(os.path.join(cfg.out_dir, f"reduce_batch_{bi}_raw.json"), {"raw": raw})
        except Exception as e:
            logger.error(f"  Batch {bi+1} failed: {e}")

    logger.info(f"REDUCE intermediate: {len(all_intermediate)} findings from {len(batches)} batches")

    # --- Final merge if too many intermediate findings ---
    if len(all_intermediate) > cfg.max_findings:
        logger.info(f"REDUCE final merge: {len(all_intermediate)} → max {cfg.max_findings}")
        # Re-reduce the intermediate findings
        compact_findings = json.dumps(all_intermediate, ensure_ascii=False, indent=1)
        if len(compact_findings) > 50000:
            compact_findings = compact_findings[:50000] + "\n..."

        merge_prompt = f"""Verdichte diese {len(all_intermediate)} Zwischen-Findings zu maximal {cfg.max_findings} finale Findings.

REGELN:
1. DEDUPLIZIERUNG hat höchste Priorität: Findings mit gleichem/ähnlichem Kernthema → zusammenfassen
2. Evidenz-Referenzen aus ALLEN zusammengefassten Findings übernehmen (verschiedene Dokumente!)
3. Statement eigenständig formulieren als Synthese aller zusammengefassten Findings
4. Jedes finale Finding braucht: title, category, impact, statement, recommendation, evidence, confidence
5. Titel müssen EINZIGARTIG sein
6. Sprache: Deutsch

ZWISCHEN-FINDINGS:
{compact_findings}

Antworte NUR als JSON-Liste."""

        try:
            raw = ollama_chat(
                cfg,
                messages=[
                    {"role": "system", "content": REDUCE_SYSTEM},
                    {"role": "user", "content": merge_prompt},
                ],
                temperature=0.2,
                num_predict=8192,
            )
            merged = extract_json_from_text(raw)
            if isinstance(merged, list):
                all_intermediate = merged
                logger.info(f"REDUCE final: {len(merged)} findings")
        except Exception as e:
            logger.error(f"Final merge failed: {e} (keeping intermediate findings)")

    write_json(out_path, all_intermediate)
    logger.info(f"REDUCE done: {len(all_intermediate)} findings → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Phase 4: DRAFT – Generate report suggestions
# ---------------------------------------------------------------------------

DRAFT_SYSTEM = """Du bist ein Senior-Analyst für kritische Infrastruktur-Schlussberichte bei der BLS AG (Schweiz).
Du reichst einen bestehenden Schlussbericht mit neuen Evidenzien und Findings an.
Du schreibst prägnant, entscheidungsorientiert, mit klaren Empfehlungen.
Du nutzt AUSSCHLIESSLICH die extrahierten Findings als Faktenbasis – erfinde nichts.
Wenn ein Finding den bestehenden Bericht bestätigt, verweise explizit darauf.
Wenn ein Finding NEU ist (nicht im bestehenden Bericht), markiere es klar als [NEU].
Sprache: Deutsch. Format: Markdown."""

DRAFT_USER_TEMPLATE = """Analysiere die folgenden Findings und ergänze meinen bestehenden Schlussbericht.

{outline_section}

Findings aus der Dokumentenanalyse ({n} Stück):
{findings_json}

Erstelle folgende Abschnitte (jeweils als Markdown):

## 1. Neue Evidenzien für bestehende Befunde
Für jedes der 14 PROFUMO-Findings im bestehenden Bericht: Welche neuen Belege/Zitate aus den Findings stützen oder widersprechen ihm?
Format: Finding-Nr → Neue Evidenz (Dokumentname, Zitat)

## 2. Neue Findings [NEU]
Findings die im bestehenden Bericht NICHT vorkommen. Gruppiert nach Kategorie.
Pro Finding: Titel, Statement, Evidenz (Dokument + Zitat), Empfehlung.

## 3. Ergänzte Lessons Learned
15-25 Bulletpoints, gruppiert nach Kategorie (Tech, Process, Org, Cost)
Pro Lesson: Was ist passiert → Was haben wir gelernt → Was empfehlen wir
Markiere [NEU] wenn nicht im bestehenden Bericht enthalten.

## 4. Ergänzte Empfehlungen ans Management
10-20 konkrete Massnahmen als Tabelle:
| # | Empfehlung | Kategorie | Nutzen | Aufwand | Priorität | Neu? |
Markiere mit [NEU] falls nicht im bestehenden Bericht.

## 5. Konkrete Textvorschläge pro Kapitel
Für jedes Kapitel des Berichts: 1-3 kurze Absätze mit konkretem Inhalt.
Verweise auf Quell-Dokumente wo immer möglich."""


def step_draft(cfg: Config, logger: logging.Logger,
               findings_path: str, outline_path: Optional[str]) -> str:
    """Phase 4: Generate report suggestions from findings + outline."""
    out_path = os.path.join(cfg.out_dir, "report_suggestions.md")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
        logger.info(f"[resume] draft exists: {out_path}")
        return out_path

    findings = read_json(findings_path)
    outline = ""
    if outline_path and os.path.exists(outline_path):
        if outline_path.lower().endswith(".pdf"):
            # Extract text from PDF via pdftotext
            import subprocess
            try:
                result = subprocess.run(
                    ["pdftotext", "-layout", outline_path, "-"],
                    capture_output=True, text=True, timeout=60
                )
                outline = result.stdout
                logger.info(f"DRAFT: extracted {len(outline):,} chars from PDF outline")
            except Exception as e:
                logger.warning(f"DRAFT: pdftotext failed: {e}, treating as text")
                outline = open(outline_path, "r", encoding="utf-8").read()
        else:
            outline = open(outline_path, "r", encoding="utf-8").read()
    # Truncate outline if too long (keep first ~20K chars to fit in context)
    if len(outline) > 20000:
        outline = outline[:20000] + "\n\n[... Rest des Berichts gekürzt ...]\n"

    outline_section = f"Mein bestehender Schlussbericht (Auszug):\n{outline}" if outline else \
        "Kein bestehender Bericht vorhanden – erstelle eine Standard-Kapitelstruktur."

    # If too many findings for one prompt, take top by confidence
    if len(findings) > 80:
        # Sort by confidence if available
        findings.sort(key=lambda f: float(f.get("confidence", 0) or 0), reverse=True)
        findings = findings[:80]
        logger.info(f"DRAFT: truncated to top 80 findings by confidence")

    findings_json = json.dumps(findings, ensure_ascii=False, indent=1)
    if len(findings_json) > 40000:
        findings_json = findings_json[:40000] + "\n... (gekürzt)"

    prompt = DRAFT_USER_TEMPLATE.format(
        outline_section=outline_section,
        n=len(findings),
        findings_json=findings_json,
    )

    logger.info(f"DRAFT: generating report suggestions ({len(findings)} findings, {len(prompt)} chars prompt)")

    try:
        md = ollama_chat(
            cfg,
            messages=[
                {"role": "system", "content": DRAFT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            num_predict=16384,
        )

        # Prepend metadata header
        header = (
            f"<!-- Schlussbericht-Entwurf -->\n"
            f"<!-- Generiert: {now_ts()} | Modell: {cfg.ollama_model} -->\n"
            f"<!-- Findings: {len(findings)} | ES-Index: {cfg.es_index} -->\n"
            f"<!-- Run: {cfg.run_id} -->\n\n"
            f"# Schlussbericht – Ergänzungen & Empfehlungen\n\n"
        )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(header + md)

        logger.info(f"DRAFT done: {len(md)} chars → {out_path}")
    except Exception as e:
        logger.error(f"DRAFT failed: {e}")
        raise

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Batch-Pipeline für RAG-basierten Schlussbericht",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Beispiele:
  python batch_report.py --es-index rag_tfk18_v1
  python batch_report.py --es-index rag_tfk18_v1 --model qwen2.5:3b --max-candidates 500
  python batch_report.py --es-index rag_tfk18_v1 --topics topics.yaml --outline bericht.md
  python batch_report.py --run-id 20260217_120000_abc --start-at reduce"""
    )
    ap.add_argument("--run-id", default=None, help="Run-ID (für Resume)")
    ap.add_argument("--out", default="runs", help="Basis-Ausgabeverzeichnis (default: runs/)")
    ap.add_argument("--model", default="glm-4.7-flash:latest", help="LLM model name")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "openai"],
                    help="LLM backend: 'ollama' or 'openai' for llama-server/vLLM (default: ollama)")
    ap.add_argument("--ollama", default="http://localhost:11434", help="Ollama Base URL")
    ap.add_argument("--openai-base", default="http://localhost:8090",
                    help="OpenAI-compatible API base URL for llama-server/vLLM (default: http://localhost:8090)")
    ap.add_argument("--es-url", default="http://localhost:9200", help="Elasticsearch URL")
    ap.add_argument("--es-index", default="rag_tfk18_v1", help="ES Index")
    ap.add_argument("--topics", default=None, help="YAML mit Suchthemen (optional)")
    ap.add_argument("--outline", default=None, help="Pfad zu deinem bestehenden Bericht/Outline (Markdown/Text)")
    ap.add_argument("--candidates-file", default=None, help="JSONL mit vorbereiteten Candidates (statt ES-Suche)")
    ap.add_argument("--max-candidates", type=int, default=3000)
    ap.add_argument("--max-findings", type=int, default=120)
    ap.add_argument("--reduce-batch-size", type=int, default=20,
                    help="Claims per REDUCE batch (default: 20, use smaller for large models)")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="Base timeout in seconds for Ollama calls (default: 1800)")
    ap.add_argument("--map-model", default="",
                    help="Separate model for MAP phase (e.g. qwen2.5:3b). If empty, uses --model")
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent workers for MAP phase (default: 4)")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--start-at", default="candidates",
                    choices=["candidates", "map", "reduce", "draft"])
    ap.add_argument("--stop-after", default="draft",
                    choices=["candidates", "map", "reduce", "draft"],
                    help="Stop after this phase (default: draft = run all)")
    return ap.parse_args()


def _log_quality(cfg: Config, logger: logging.Logger, phase: str):
    """Run quality analysis after MAP or REDUCE and log key metrics."""
    try:
        from batch_quality import analyze_claims, analyze_findings
    except ImportError:
        # Try relative import
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
        from batch_quality import analyze_claims, analyze_findings

    logger.info(f"── Quality Check ({phase.upper()}) ──")
    try:
        if phase == "map":
            metrics = analyze_claims(os.path.join(cfg.out_dir, "claims.jsonl"))
            fc = metrics.get("field_completeness", {})
            div = metrics.get("diversity", {})
            conf = metrics.get("confidence", {})
            logger.info(
                f"  Claims: {metrics.get('total_claims',0)} | "
                f"finding={fc.get('has_finding_pct',0)}% | "
                f"evidence={fc.get('has_evidence_pct',0)}% | "
                f"confidence={conf.get('avg',0):.2f}"
            )
            logger.info(
                f"  Diversity: {div.get('unique_documents',0)} docs | "
                f"{div.get('duplicate_pct',0)}% duplicates"
            )
        elif phase == "reduce":
            metrics = analyze_findings(os.path.join(cfg.out_dir, "findings.json"))
            fc = metrics.get("field_completeness", {})
            eq = metrics.get("evidence_quality", {})
            dup = metrics.get("duplicates", {})
            logger.info(
                f"  Findings: {metrics.get('total_findings',0)} | "
                f"statement={fc.get('has_statement_pct',0)}% | "
                f"evidence={fc.get('has_evidence_pct',0)}% | "
                f"recommendation={fc.get('has_recommendation_pct',0)}%"
            )
            logger.info(
                f"  Evidence: avg {eq.get('avg_evidence_per_finding',0)}/finding | "
                f"{eq.get('unique_evidence_documents',0)} unique docs"
            )
            logger.info(
                f"  Duplicates: titles={dup.get('duplicate_titles_pct',0)}% | "
                f"statements={dup.get('duplicate_statements_pct',0)}%"
            )
        # Save full metrics
        metrics_path = os.path.join(cfg.out_dir, f"quality_{phase}.json")
        write_json(metrics_path, metrics)
        logger.info(f"  Saved: {metrics_path}")
    except Exception as e:
        logger.warning(f"  Quality check failed: {e}")


def main():
    args = parse_args()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_dir = os.path.join(args.out, run_id)
    ensure_dir(out_dir)

    cfg = Config(
        run_id=run_id,
        out_dir=out_dir,
        backend=args.backend,
        ollama_base=args.ollama,
        openai_base=args.openai_base,
        ollama_model=args.model,
        ollama_timeout_s=args.timeout,
        es_url=args.es_url,
        es_index=args.es_index,
        max_candidates=args.max_candidates,
        max_findings=args.max_findings,
        reduce_batch_size=args.reduce_batch_size,
        map_model=args.map_model,
        map_workers=args.workers,
        log_level=args.log_level,
    )

    logger = setup_logger(out_dir, cfg.log_level)
    map_model_display = cfg.map_model or cfg.ollama_model
    logger.info(f"{'='*60}")
    logger.info(f"RUN: {cfg.run_id}")
    logger.info(f"Ollama: {cfg.ollama_base}")
    logger.info(f"  REDUCE/DRAFT model: {cfg.ollama_model}")
    logger.info(f"  MAP model:          {map_model_display}")
    logger.info(f"  MAP workers:        {cfg.map_workers}")
    logger.info(f"ES: {cfg.es_url}/{cfg.es_index}")
    logger.info(f"Out: {cfg.out_dir}")
    logger.info(f"Max candidates: {cfg.max_candidates} | Max findings: {cfg.max_findings}")
    logger.info(f"{'='*60}")

    # Save config for reproducibility
    write_json(os.path.join(out_dir, "config.json"), {
        "run_id": cfg.run_id,
        "model": cfg.ollama_model,
        "map_model": cfg.map_model or cfg.ollama_model,
        "map_workers": cfg.map_workers,
        "es_index": cfg.es_index,
        "max_candidates": cfg.max_candidates,
        "max_findings": cfg.max_findings,
        "reduce_batch_size": cfg.reduce_batch_size,
        "started_at": now_ts(),
        "topics_file": args.topics,
        "outline_file": args.outline,
    })

    topics = load_topics(args.topics)
    logger.info(f"Topics: {len(topics)} ({', '.join(t['id'] for t in topics)})")

    # Phase routing with resume support
    candidates_path = os.path.join(cfg.out_dir, "candidates.jsonl")
    claims_path = os.path.join(cfg.out_dir, "claims.jsonl")
    findings_path = os.path.join(cfg.out_dir, "findings.json")

    phases = ["candidates", "map", "reduce", "draft"]
    start_idx = phases.index(args.start_at)
    stop_idx = phases.index(args.stop_after)

    for phase in phases[start_idx:stop_idx + 1]:
        logger.info(f"\n{'─'*40}")
        logger.info(f"PHASE: {phase.upper()}")
        logger.info(f"{'─'*40}")
        t0 = time.time()

        if phase == "candidates":
            candidates_path = step_candidates(cfg, logger, topics, args.candidates_file)
        elif phase == "map":
            if not os.path.exists(candidates_path) or count_jsonl(candidates_path) == 0:
                candidates_path = step_candidates(cfg, logger, topics, args.candidates_file)
            claims_path = step_map(cfg, logger, candidates_path)
            _log_quality(cfg, logger, "map")
        elif phase == "reduce":
            if not os.path.exists(claims_path) or count_jsonl(claims_path) == 0:
                logger.error("No claims found. Run MAP phase first.")
                break
            findings_path = step_reduce(cfg, logger, claims_path)
            _log_quality(cfg, logger, "reduce")
        elif phase == "draft":
            if not os.path.exists(findings_path):
                logger.error("No findings found. Run REDUCE phase first.")
                break
            step_draft(cfg, logger, findings_path, args.outline)

        elapsed = time.time() - t0
        logger.info(f"Phase {phase} done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    logger.info(f"\n{'='*60}")
    logger.info(f"RUN {cfg.run_id} COMPLETE at {now_ts()}")
    logger.info(f"Artifacts: {cfg.out_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Abgebrochen (Ctrl+C). Run kann mit --start-at fortgesetzt werden.")
    except Exception as e:
        logging.getLogger("batch_report").error(f"Fatal: {e}", exc_info=True)
        raise
