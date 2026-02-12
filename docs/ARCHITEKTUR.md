# 🏗️ Agentic RAG System – Architekturübersicht

> **Stand:** 2025-02-12 | **Version:** Phase 4
> **Zweck:** Vollständige technische Dokumentation zum Nachbauen des Systems

---

## 1. System-Überblick

```
┌─────────────────────────────────────────────────────────────┐
│                    Host: NVIDIA DGX Spark                    │
│                 /media/felix/RAG/1 = Projektarchiv           │
│                                                              │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │  OpenWebUI   │  │  Kibana    │  │  Ollama (GPU)        │  │
│  │  :8086       │  │  :5601     │  │  :11434              │  │
│  └──────┬───────┘  └─────┬──────┘  └──────────┬───────────┘  │
│         │                │                     │              │
│         ▼                ▼                     │              │
│  ┌──────────────────────────────┐              │              │
│  │      Agent API (FastAPI)     │◄─────────────┘              │
│  │      :11436                  │                             │
│  │                              │                             │
│  │  ┌──────────┐ ┌───────────┐ │  ┌───────────┐             │
│  │  │ES Client │ │Chroma Cli.│ │  │ PyRunner  │             │
│  │  └────┬─────┘ └─────┬─────┘ │  │ :9000     │             │
│  └───────┼─────────────┼───────┘  └─────┬─────┘             │
│          ▼             ▼                 │                    │
│  ┌──────────────┐ ┌──────────┐           │                   │
│  │Elasticsearch │ │ ChromaDB │     /data:ro                  │
│  │  :9200       │ │ (embedded│     (Projektarchiv)           │
│  └──────────────┘ └──────────┘                               │
│                                                              │
│  ┌──────────────┐                                            │
│  │   Indexer     │ (einmalig / on-demand)                    │
│  └──────────────┘                                            │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Docker Services (docker-compose.yml)

### 2.1 ollama
| Parameter | Wert |
|-----------|------|
| **Image** | `ollama/ollama:latest` |
| **Port** | 11434 |
| **Volume** | `/media/felix/RAG/ollama:/root/.ollama` |
| **GPU** | NVIDIA, 1x |
| **Keep-Alive** | 24h (Modelle bleiben im RAM) |

Modelle werden über `ollama pull` geladen und unter `/media/felix/RAG/ollama` persistiert.

### 2.2 agent_api
| Parameter | Wert |
|-----------|------|
| **Build** | `./agent_api/Dockerfile` |
| **Port** | 11436 |
| **Base Image** | `python:3.11-slim` |
| **Entrypoint** | `uvicorn app.main:app --host 0.0.0.0 --port 11436` |

**Volumes:**
```
/media/felix/RAG/1/volumes/chroma → /chroma      (ChromaDB Daten)
/media/felix/RAG/1/volumes/logs   → /logs         (Logfiles)
/media/felix/RAG/1/volumes/state  → /state        (Session State JSON)
/media/felix/RAG/1                → /media/felix/RAG/1  (Datei-Links)
```

**Wichtige Env-Variablen:**
```bash
OLLAMA_BASE_URL=http://ollama:11434
LLM_MODEL=llama4:latest
ES_URL=http://elasticsearch:9200
ES_INDEX=rag_files_v1
CHROMA_PATH=/chroma
COLLECTION=documents
FILE_BASE=/media/felix/RAG/1
PYRUNNER_URL=http://runner:9000/run
STATE_PATH=/state
EMBED_MODEL=all-MiniLM-L6-v2
```

**Python Dependencies** (`requirements.txt`):
```
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.4
chromadb==0.5.7
sentence-transformers==2.7.0
httpx==0.27.0
rank-bm25==0.2.2
rapidfuzz==3.10.1
numpy==1.26.4
elasticsearch==8.12.1
```

### 2.3 runner (PyRunner)
| Parameter | Wert |
|-----------|------|
| **Build** | `./runner/Dockerfile` |
| **Port** | 9000 |
| **Base Image** | `python:3.11-slim` |
| **Entrypoint** | `uvicorn app.run:app --host 0.0.0.0 --port 9000` |

**Volumes:**
```
/media/felix/RAG/1 → /data:ro    (Projektarchiv, READ-ONLY)
```

**Env-Variablen:**
```bash
NO_INTERNET=1
TIMEOUT_SECONDS=25
DATA_ROOT=/data
```

**Python Dependencies** (`requirements.txt`):
```
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.4
pandas==2.2.2
openpyxl==3.1.5
tabulate==0.9.0
python-dateutil==2.9.0
```

**API:**
- `GET /health` → Status + Timeout + DATA_ROOT
- `POST /run` → `{ code: str, locals?: dict, timeout?: int }` → `{ ok, stdout, stderr, result, locals, error }`

### 2.4 elasticsearch
| Parameter | Wert |
|-----------|------|
| **Image** | `docker.elastic.co/elasticsearch/elasticsearch:8.12.2` |
| **Port** | 9200 |
| **Volume** | `/media/felix/RAG/1/volumes/esdata` |
| **Config** | Single-node, Security disabled, 2GB Heap |

### 2.5 kibana
| Parameter | Wert |
|-----------|------|
| **Image** | `docker.elastic.co/kibana/kibana:8.12.2` |
| **Port** | 5601 |

### 2.6 openwebui
| Parameter | Wert |
|-----------|------|
| **Image** | `ghcr.io/open-webui/open-webui:v0.3.18` |
| **Port** | 8086 |

**Wichtige Env-Variablen:**
```bash
OLLAMA_BASE_URL=http://ollama:11434           # Direktzugriff auf Ollama
OPENAI_API_BASE_URLS=http://agent_api:11436/v1 # RAG API als "OpenAI"
OPENAI_API_KEYS=local
DEFAULT_MODELS=agentic-rag,llama4:latest
```

### 2.7 indexer (On-Demand)
| Parameter | Wert |
|-----------|------|
| **Build** | `./indexer/Dockerfile` |
| **restart** | `no` (manuell starten) |

Indexiert Dokumente aus `/media/felix/RAG/1` nach ES + ChromaDB.
Unterstützte Formate: PDF, DOCX, TXT, MSG, EML, XLSX, PPTX

---

## 3. Agent API – Dateistruktur

```
agent_api/app/
├── main.py                 # FastAPI App, /v1/chat/completions, Routing-Logik
├── rag_pipeline.py         # SimpleRAGPipeline: Search → Context → LLM Answer
├── tools.py                # Tools-Klasse: Hybrid-Suche (ES + Chroma), Gate-Logik
├── tools_es.py             # ESTools: BM25 Search, Exact Phrase, AND-Fallback
├── chroma_client.py        # ChromaDB Client (PersistentClient)
├── source_analyzer.py      # Quellen-Referenz-Erkennung + Dokument-Volltext-Abruf
├── code_executor.py        # PyRunner Client (Code-Ausführung)
├── glossary.py             # Domain-Glossar (Akronyme, Fachbegriffe)
├── glossary.yaml           # Glossar-Definitionen
├── config_rag.py           # ES-Indices, Extension-Filter, Trigger-Patterns
├── config_pipeline.py      # Pipeline-Tuning-Parameter (Top-K, Boost, etc.)
├── state.py                # Session State (JSON per Conversation)
├── format_links.py         # Quellen-Link Formatierung
├── rerank.py               # Relevance Reranking
├── thinking_agent.py       # (Phase 2, experimentell) Multi-Step Thinking Agent
├── agent.py                # (Legacy) Alter Agent-Code
└── agent_orchestrator.py   # (Legacy) Orchestrator
```

---

## 4. Datenfluss im Detail

### 4.1 Erstanfrage (Pfad C: Normaler RAG-Flow)

```
User-Frage (OpenWebUI)
    │
    ▼
POST /v1/chat/completions (SSE Stream)
    │
    ├─ 1. Modell bestimmen: "rag-gpt-oss:latest" → "gpt-oss:latest" (strip rag-)
    ├─ 2. Thinking-Mode? Nur wenn Modellname "-think" enthält
    ├─ 3. Multi-Source-Check: Referenziert "diese Dokumente"? → Pfad A
    ├─ 4. Single-Source-Check: Referenziert "[N]"? → Pfad B
    │
    ▼ (nichts erkannt → normaler RAG)
    │
    ├─ 5. Chat-History extrahieren (letzte 3 Turns = 6 Messages)
    ├─ 6. Follow-up-Kontext laden (vorherige Top 3 Quellen als Volltext)
    │
    ▼
SimpleRAGPipeline.run()
    │
    ├─ 7. Glossar-Rewrite: "GBT" → "GBT Gotthard Basistunnel"
    ├─ 8. Query-Expansion (bei Follow-ups): Keywords aus History anhängen
    │
    ├─ 9. Hybrid-Suche:
    │     ├─ ES BM25:  tools_es.es_bm25_search_content()
    │     │             Index: rag_files_v1
    │     │             Filter: DEFAULT_EXT_FILTER
    │     │             Felder: content (BM25), path (Boost)
    │     │
    │     └─ ChromaDB: 5 Collections parallel durchsucht
    │                  (documents, documents_docx, documents_txt,
    │                   documents_msg, documents_mail_ews)
    │                  Embedding: all-MiniLM-L6-v2
    │
    ├─ 10. Dedup + Merge: Pfad-basierte Deduplizierung, ES bevorzugt
    ├─ 11. Ranking: Keyword-Boost (Pfad +2.0, Snippet +1.0, Compound +3.0)
    │               Excel-Penalty, PDF/MSG-Bonus
    │
    ├─ 12. Context-Aufbau: Top 10 Snippets (max 2000 Zeichen/Snippet)
    │      + Follow-up: Vorherige Dokumente vorangestellt
    │
    ├─ 13. LLM-Antwort (Ollama):
    │      ├─ System-Prompt (Dokumenten-Analyst, Fachbegriffe, Antwortformat)
    │      ├─ Chat-History (falls vorhanden)
    │      ├─ Kontext-Dokumente
    │      └─ User-Frage
    │      → Stream Tokens via SSE
    │
    ├─ 14. Code-Erkennung: Falls ```python Block in Antwort
    │      → POST runner:9000/run
    │      → Ergebnis inline anhängen
    │
    └─ 15. Quellen-Links: Klickbare Markdown-Links
           Quellen werden in last_sources gespeichert
```

### 4.2 Nachfrage auf vorherige Quellen (Pfad A/B)

```
"Analysiere Quelle [2]"  oder  "Vergleiche diese Dokumente"
    │
    ├─ detect_source_reference() → Quellennummer (Pfad B)
    │  ODER
    ├─ detect_multi_source_reference() → "all" (Pfad A)
    │
    ▼
Lade last_sources aus StateStore
    │
    ▼
fetch_document_text() → ES _search by path
    │  (Volltext, max 8000-12000 Zeichen/Dok)
    │
    ▼
Dedizierter System-Prompt (exhaustive Analyse)
    + Chat-History
    + Dokument-Volltext
    │
    ▼
pipeline._llm_stream() → Streame Antwort
    + Quellen-Links
```

---

## 5. Elasticsearch Index

### Index: `rag_files_v1`

**Wichtige Felder:**
| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `content` | text | Dokumentinhalt (BM25-suchbar) |
| `file.filename` | keyword | Dateiname |
| `file.extension` | keyword | Extension (`.pdf`, `.eml`, etc.) |
| `path.virtual` | text/keyword | Relativer Pfad (Display) |
| `path.real` | text/keyword | Absoluter Pfad |
| `meta.real.path` | text | Pfad-Metadaten |

**Extension-Filter:**
```
md, txt, rst, log, json, yaml, yml,
pdf, docx, doc, msg, eml, .eml,
xlsx, xls, pptx, ppt
```

> **Achtung:** `file.extension` speichert teils mit Punkt (`.eml`), teils ohne (`eml`).
> Daher stehen beide Varianten im Filter.

---

## 6. ChromaDB Collections

| Collection | Inhalt | Embedding |
|------------|--------|-----------|
| `documents` | PDFs | all-MiniLM-L6-v2 |
| `documents_docx` | DOCX | all-MiniLM-L6-v2 |
| `documents_txt` | TXT/RST/LOG | all-MiniLM-L6-v2 |
| `documents_msg` | MSG (Outlook) | all-MiniLM-L6-v2 |
| `documents_mail_ews` | EML/Mails | all-MiniLM-L6-v2 |

**Chunk-Konfiguration:** 1200 Zeichen, 180 Overlap

---

## 7. Session State

**Speicherort:** `/state/<conv_id>.json`

```json
{
  "summary": "...",
  "notes": "...",
  "sources": [
    {
      "n": 1,
      "path": "SBB TFK.../Dokument.pdf",
      "display_path": "/SBB TFK.../Dokument.pdf",
      "local_url": "http://localhost:11436/open?path=..."
    }
  ],
  "updated_at": 1234567890
}
```

Spezial-Key `last_sources` speichert die Quellen der letzten Suche global (für Pfad A/B).

---

## 8. Pipeline-Tuning-Parameter

Alle per Env-Variable oder `rag_config` im Request setzbar:

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `RAG_SEARCH_TOP_K` | 40 | Anzahl Treffer aus ES/Chroma |
| `RAG_MAX_CONTEXT_DOCS` | 10 | Dokumente im LLM-Kontext |
| `RAG_MAX_SOURCES` | 40 | Quellen in der Antwort |
| `RAG_MAX_SNIPPET_LENGTH` | 2000 | Max Zeichen pro Snippet |
| `RAG_ANSWER_TEMPERATURE` | 0.3 | LLM Temperatur |
| `RAG_ANSWER_MAX_TOKENS` | 4000 | Max Antwort-Tokens |
| `RAG_KEYWORD_BOOST_PATH` | 2.0 | Pfad-Keyword-Boost |
| `RAG_KEYWORD_BOOST_SNIPPET` | 1.0 | Snippet-Keyword-Boost |
| `RAG_KEYWORD_COMPOUND_BONUS` | 3.0 | Multi-Keyword-Bonus |
| `RAG_EXCEL_PENALTY_RELEVANT` | -1.0 | Excel-Penalty (relevant) |
| `RAG_EXCEL_PENALTY_IRRELEVANT` | -4.0 | Excel-Penalty (irrelevant) |
| `RAG_PDF_MSG_BONUS` | 1.0 | PDF/MSG/DOCX Bonus |

---

## 9. LLM System-Prompt

Das LLM erhält immer diesen Basis-Prompt:

```
DU BIST EIN DOKUMENTEN-ANALYST FÜR SCHWEIZER EISENBAHN-PROJEKTE
(SBB TFK 2020 - Tunnelfunk).

Fachgebiete: Projektleitung, Programmleitung, Funktechnik, Tunnelfunk.
Fachbegriffe: FAT=Werksabnahme, SAT=Standortabnahme, TFK=Tunnelfunk,
              GBT=Gotthard Basistunnel, RBT=Rhomberg Bahntechnik

Antwort-Format:
1. Deutsch
2. Direkt mit Fakten starten
3. Jede Aussage mit [N] zitieren
4. Aufzählungen und kurze Absätze

Code-Ausführung:
- Python-Code in ```python Blöcken wird automatisch ausgeführt
- Verfügbar: pandas, tabulate, csv, os, json
- Dateien unter DATA_ROOT='/data'
```

---

## 10. Thinking Mode (optional)

Wird aktiviert durch `-think` im Modellnamen.

```
Schritt 1: Analyse (in <think> Tags, einklappbar in OpenWebUI)
  - Welche Dokumente sind relevant?
  - Was sind die Kernfakten?
  - Gibt es Widersprüche?

Schritt 2: Finale Antwort (normal gestreamt)
  - Basierend auf Analyse + Kontext
```

---

## 11. Indexer

### Indexer-Container (`indexer/`)

Wird manuell gestartet:
```bash
docker compose run --rm indexer
```

**Unterstützte Formate und Loader:**
| Format | Loader | Modul |
|--------|--------|-------|
| PDF | PyMuPDF | `index_pdfs.py` |
| DOCX | python-docx | `index_docx.py` |
| TXT/RST/LOG | Plaintext | `index_txt.py` |
| MSG | extract-msg | `index_msg.py` |
| EML | email.parser | `index_eml.py` + `index_eml_to_es.py` |
| XLSX | openpyxl/pandas | via text_loaders |
| PPTX | python-pptx | via text_loaders |

**Ablauf:**
1. Scan `/data` rekursiv
2. Manifest-Check (SQLite) → nur neue/geänderte Dateien
3. Text extrahieren
4. Chunking (1200 Zeichen, 180 Overlap)
5. Embedding (all-MiniLM-L6-v2)
6. Upsert nach ChromaDB (Collection je nach Typ)
7. Upsert nach Elasticsearch (rag_files_v1)

---

## 12. Datenverzeichnisse auf dem Host

```
/media/felix/RAG/
├── 1/                              # Projektarchiv (Quelldokumente)
│   ├── SBB TFK 2020 PJ - 1 Projekte/
│   ├── SBB TFK 2020 PJ - 2 Kommunikation/
│   ├── SBB TFK 2020 PJ - 3 Beschaffung/
│   ├── SBB TFK 2020 PJ - 4 Technik Planung/
│   ├── SBB TFK 2020 PJ - 5 Projektablauf/
│   ├── SBB TFK 2020 PJ - 6 Projektorganisation/
│   ├── SBB TFK 2020 PJ - 7 Finanzen/
│   ├── SBB TFK 2020 PJ - 8 Qualitätsmanagement/
│   ├── SBB TFK 2020 PJ - 9 Medien/
│   ├── MailsFEA/                   # E-Mail Archiv (.eml)
│   └── volumes/
│       ├── chroma/                 # ChromaDB Daten
│       ├── esdata/                 # Elasticsearch Daten
│       ├── state/                  # Session State JSONs
│       ├── logs/                   # Application Logs
│       └── manifest/               # Indexer Manifest (SQLite)
│
├── ollama/                         # Ollama Model Store
│
└── AGENTIC/                        # Git Repository (dieses Projekt)
    ├── agent_api/                  # RAG Backend
    ├── runner/                     # Python Sandbox
    ├── indexer/                    # Dokument-Indexer
    ├── docs/                       # Dokumentation
    ├── docker-compose.yml
    ├── START.sh / STOP.sh
    └── ...
```

---

## 13. Netzwerk

Alle Container laufen im Docker-Netzwerk `agentic_default`.

| Service | Interner Hostname | Port |
|---------|-------------------|------|
| Ollama | `ollama` | 11434 |
| Agent API | `agent_api` | 11436 |
| PyRunner | `runner` | 9000 |
| Elasticsearch | `elasticsearch` | 9200 |
| Kibana | `kibana` | 5601 |
| OpenWebUI | `openwebui` | 8086 |

---

## 14. Startup / Shutdown

```bash
# Starten
cd /media/felix/RAG/AGENTIC
docker compose up -d

# Stoppen
docker compose down

# Einzelnen Service neu bauen
docker compose build agent_api
docker compose up -d agent_api

# Logs
docker logs e2ngiadina-api --tail 50 -f

# Runner neu bauen
docker compose build runner
docker compose up -d runner
```

---

## 15. Versionierung

| Tag | Datum | Inhalt |
|-----|-------|--------|
| `v2025.02.12-phase4` | 2025-02-12 | Code Execution + .eml Fix + Generic Follow-up |
