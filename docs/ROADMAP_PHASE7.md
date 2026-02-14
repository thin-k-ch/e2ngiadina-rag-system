# 🚀 Roadmap Phase 7+8: Qualität, UX & Erweiterungen

> **Stand:** 2026-02-14 | **Status:** Phase 8a abgeschlossen
> **Ausgangslage:** Phase 6 stabil (ReAct Agent, 7 Tools, Multi-Tenant, File-Upload Protokoll)
> **Git-Tag Baseline:** `v2026.02.13-phase6-hotfix`
> **Aktueller Tag:** `v2026.02.14-phase8a`
> **Aktueller Stand:** Thinking Mode, Cross-Encoder Reranking, verbesserte Quellen, Metadaten-Extraktion

---

## 1. Was ist erledigt (Phase 1–6)

| Phase | Features | Status |
|-------|----------|--------|
| 1–3 | ES+Chroma Indexierung, Hybrid-Suche, RAG-Pipeline | ✅ |
| 4 | Python Code Execution (PyRunner Sandbox) | ✅ |
| 5 | Transcript→Protocol, Dynamic num_ctx/Timeout, OpenWebUI File-Upload | ✅ |
| 6 | ReAct Agent (7 Tools), Multi-Tenant, Forced-Step, Phase-Indikatoren | ✅ |
| Hotfixes | Title-Bypass, example.com URLs, File-Upload Context-Extraktion | ✅ |

---

## 2. Offene Punkte aus Phase 6

### 2.1 ~~Web-Suche (braucht API-Key)~~ ✅ Erledigt
- SearXNG als self-hosted Meta-Suchmaschine integriert (kein API-Key nötig)
- Container: `searxng/searxng:latest`, Config: `searxng/settings.yml`
- Fallback-Kette: SearXNG → Brave API → Serper.dev
- Getestet unter `rag-llama4:latest` ✅

### 2.2 ~~Fess-Plugins aus ES entfernen~~ ✅ Erledigt
- ES neu gebaut ohne Plugins (`docker compose build --no-cache elasticsearch`)
- 0 Plugins installiert, ES läuft sauber (34 Shards, yellow/single-node)

### 2.3 Zweiter Mandant testen
- Tenant-System ist gebaut (`tenants/_template.yaml`), aber nur SBB TFK aktiv
- Zum Testen: Template kopieren, ausfüllen, Daten indexieren

---

## 3. Verbesserungskatalog (priorisiert)

### Prio 1: Qualität & Stabilität

#### 3.1 ~~Antwortqualität verbessern~~ ✅ Erledigt (System-Prompt)
- REACT_SYSTEM_PROMPT überarbeitet: Indikativ statt Konjunktiv, exakte Zitate, Seitenzahlen
- Greeting-Fix: Keine Tool-Liste mehr bei "Hallo"
- Forced search → read_document Hint: LLM bekommt Extra-Step um Dokumente vollständig zu lesen
- **Offen:** Reranking (Cross-Encoder), Context-Qualität (ganze Absätze)

#### 3.2 ~~Fehlertoleranz~~ ✅ Erledigt
- `LLMError` Exception + Retry-Logik (1x Retry, +60s Timeout)
- `_llm_with_tools` und `_llm_stream_final`: Retry bei ReadTimeout, ConnectTimeout, ConnectError
- User-freundliche Fehlermeldung mit Tipps (Modell wechseln, erneut versuchen)

#### 3.2b ~~Modell-Architektur~~ ✅ Erledigt
- **Auto-Discovery**: `REACT_MODELS` Whitelist entfernt – ALLE Modelle gehen durch ReAct Agent
- **Kein `rag-` Prefix mehr**: Modelle erscheinen unter ihrem echten Ollama-Namen
- **Ollama Proxy**: agent_api proxied `/api/tags`, `/api/pull`, `/api/delete`, `/api/show`, `/api/chat`
- **OpenWebUI Single-Connection**: Nur noch über agent_api (kein direktes Ollama mehr)
- **Modell-Management via UI**: Neue Modelle pullen/löschen direkt in OpenWebUI
- **Embedding-Schutz**: `mxbai-embed-large` kann nicht gelöscht werden (technisch nötig für Vektorsuche)
- **Aufgeräumte Modelle**: qwen2.5:72b, qwen2.5:14b, llama3.1 entfernt (~100 GB frei)
- **Aktiver Kern**: llama4 (67GB), deepseek-r1:70b (42GB), gpt-oss (13GB), qwen2.5:3b (1.9GB), apertus:70b (43GB)
- **Prompt-basiertes Tool-Calling**: Reasoning-Modelle (DeepSeek-R1, QwQ, etc.) ohne native Tool-API werden automatisch erkannt und nutzen Prompt-basierten Fallback mit `<tool_call>` Tags
- **Greeting-Shortcut**: Prompt-Tool-Modelle überspringen den teuren Tool-Prompt bei einfachen Fragen
- **Timeouts**: Reasoning-Modelle erhalten 600s statt 300s (lange `<think>`-Phase)

#### 3.3 ~~Indexer-Verbesserungen~~ ✅ Teilweise erledigt (Metadaten)
- **Inkrementelles Re-Indexing**: Nur geänderte Dateien neu indexieren (Manifest existiert)
- **Neue Formate**: HTML, RTF, CSV nativ (nicht nur via Python)
- ~~**Metadaten-Extraktion**~~: ✅ Autor, Datum, Betreff aus E-Mails (.eml/.msg), PDFs und DOCX
  - `text_loaders.py`: `extract_pdf_metadata()`, `extract_docx_metadata()`, `extract_eml_metadata()`, `extract_msg_metadata()`
  - PDF: Author, Title, CreationDate, Producer, PageCount (via PyMuPDF)
  - DOCX: Author, Title, Subject, Created, Modified, LastModifiedBy
  - EML/MSG: From, To, Cc, Subject, Date
  - Chroma→ES Transfer: Metadaten-Felder werden übertragen

### Prio 2: UX-Verbesserungen

#### 3.4 ~~Bessere Quellen-Links~~ ✅ Erledigt
- ✅ Quellen als klickbare Markdown-Links mit `/open` Endpoint
- ✅ Saubere Dateinamen (Ordner/Datei statt voller Pfad)
- ✅ Snippet-Preview unter jedem Link
- ✅ Deduplizierung (gleiche Datei nur 1x gelistet)
- ✅ Visueller Separator (`---`) und **Fettdruck** für Quellen-Nummern

#### 3.5 Chat-Kontext verbessern
- Längerer Konversationsverlauf (aktuell 3 Turns)
- Session-Zusammenfassung für lange Gespräche
- "Merke dir X" → Persistenter Notiz-Speicher pro Mandant

#### 3.6 ~~Fortschrittsanzeige / Thinking Mode~~ ✅ Erledigt
- ✅ **Thinking Mode**: Collapsible `<details>` Blöcke für ReAct-Schritte
  - `🧠 Recherche & Analyse`: Tool-Calls, Ergebnisse, Schrittanzahl + Dauer
  - `💭 Überlegung`: `<think>`-Blöcke von Reasoning-Modellen (DeepSeek-R1)
- ✅ **Keepalive**: Alle 5s `⏳ Denke nach... (Xs)` während LLM-Calls
- ✅ **Timeout erhöht**: `AIOHTTP_CLIENT_TIMEOUT=900` (15 Min) in OpenWebUI
- ✅ **Cross-Encoder Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB)
  - Rerankt Top-20 Suchergebnisse semantisch nach Hybrid-Merge
  - Lazy-Loading, ~0.2s für 20 Hits
  - Konfigurierbar: `RERANKER_ENABLED`, `RERANKER_MODEL`, `RERANKER_TOP_N`
- ✅ **System-Prompt**: Ausführlichere, detailliertere Antworten mit Markdown-Struktur

### Prio 3: Neue Features

#### 3.7 Dokumenten-Vergleich
- Automatischer Diff/Vergleich zweier Dokumente
- Änderungstracking zwischen Versionsständen
- "Was hat sich zwischen Version 1 und 2 des Werkvertrags geändert?"

#### 3.8 Zusammenfassungs-Modus
- Ganzes Dokument zusammenfassen (nicht nur Suchtreffer-Snippets)
- Mehrseitige PDFs kapitelweise zusammenfassen
- Executive Summary für Vertragsdokumente

#### 3.9 Automatische Reports
- "Erstelle einen Statusbericht über alle offenen Pendenzen"
- Aggregation über mehrere Dokumente
- Export als Markdown oder PDF

#### 3.10 Whisper-Integration
- Direkte Audio-Upload → Transkription → Protokoll
- Whisper lokal auf DGX Spark (GPU verfügbar)
- Speaker Diarization für automatische Sprecher-Erkennung

### Prio 4: Infrastruktur

#### 3.11 Monitoring & Logging
- Strukturiertes Logging (JSON) statt Print-Statements
- Request-Tracing (Request-ID durch alle Services)
- Metriken: Antwortzeit, Token-Verbrauch, Tool-Nutzung

#### 3.12 Backup & Recovery
- ES-Snapshots automatisiert
- ChromaDB-Backup
- State-Backup

#### 3.13 Security
- API-Key-Authentifizierung (aktuell offen)
- Rate-Limiting
- Audit-Log (wer hat was gefragt)

---

## 4. Empfohlene Reihenfolge (aktualisiert 2026-02-13)

### ✅ Erledigt (Phase 7)
1. ~~Web-Search~~ → SearXNG self-hosted ✅
2. ~~ES ohne Fess-Plugins~~ ✅
3. ~~Antwortqualität: System-Prompt Tuning~~ ✅
4. ~~Fehlertoleranz: Retry + Fehlermeldungen~~ ✅
5. ~~Modell-Architektur: Auto-Discovery, kein rag- Prefix~~ ✅
6. ~~DeepSeek-R1: Prompt-basiertes Tool-Calling~~ ✅

### ✅ Erledigt (Phase 8a – 2026-02-14)
7. ~~Cross-Encoder Reranking~~ → `reranker.py`, ms-marco-MiniLM-L-6-v2 ✅
8. ~~Indexer: Metadaten-Extraktion~~ → PDF/DOCX/EML/MSG Metadaten ✅
9. ~~Quellen-Links verbessern~~ → klickbare Links, Snippet-Preview, Dedup ✅
10. ~~Thinking Mode~~ → `<details>` Blöcke, Keepalive, Reasoning-Events ✅
11. ~~Ausführlichere Antworten~~ → System-Prompt erweitert ✅
12. ~~Timeout erhöhen~~ → AIOHTTP_CLIENT_TIMEOUT=900 ✅

### Nächste Session (Phase 8b – Autonomie)
10. **Whisper-Integration** → Audio-Upload → Transkription → Protokoll (lokal auf GPU)
11. **Langzeit-Gedächtnis** → Persistenter Notiz-Speicher pro Mandant/Session
12. **Multi-Dokument-Vergleich** → Diff zwischen Versionen, Änderungstracking
13. **Zusammenfassungs-Modus** → Ganzes Dokument kapitelweise zusammenfassen

### Langfristig (Phase 9 – Produktionsreife)
14. Monitoring & strukturiertes Logging (JSON, Request-Tracing)
15. Security (API-Keys, Rate-Limiting, Audit-Log)
16. Automatische Reports (Statusberichte, Export als PDF)
17. Zweiter Mandant aufsetzen + testen

---

## 5. Hardware-Potential (DGX Spark)

Die NVIDIA DGX Spark hat 128GB unified RAM (CPU+GPU shared). Aktuell genutzt:
- Ollama: ~125GB auf Disk (llama4:67GB, deepseek-r1:42GB, gpt-oss:13GB, qwen2.5:3b, apertus:43GB, mxbai-embed)
- ES: ~2GB RAM
- ChromaDB + Services: ~4GB RAM
- **Hinweis**: Nicht alle Modelle gleichzeitig im VRAM – Ollama lädt/entlädt automatisch

Möglichkeiten:
- **Whisper Large V3**: ~3GB GPU RAM → lokale Transkription
- **Cross-Encoder Reranking**: ~1GB → bessere Suchergebnisse
- **Hinweis zu DeepSeek-R1:70b**: Reasoning-Modell mit `<think>`-Phase, ~3 Min/Step, ideal für komplexe Analysen aber langsam

---

## 6. 🔮 Vision: Ausbaustufen zum Autonomen Agent-System

### Stufe 1: Intelligenter Assistent (✅ AKTUELL)
> *"Frag mich was über deine Dokumente"*
- ReAct Agent mit 7 Tools, autonome Recherche
- Hybrid-Suche (ES + ChromaDB), Code-Execution, Web-Suche
- Prompt-basiertes Tool-Calling für alle Modelltypen

### Stufe 2: Proaktiver Analyst
> *"Ich erkenne Muster und mache Vorschläge"*
- **Cross-Encoder Reranking**: Deutlich bessere Trefferqualität
- **Zusammenfassungen on-demand**: Ganzes Dokument → Executive Summary
- **Dokumenten-Vergleich**: "Was hat sich geändert zwischen V1 und V2?"
- **Automatische Klassifikation**: Neue Dokumente werden beim Indexieren kategorisiert
- **Follow-up-Vorschläge**: Agent schlägt nach Antwort relevante Folgefragen vor

### Stufe 3: Kollaborativer Wissensarbeiter
> *"Ich merke mir was du brauchst und arbeite über Sessions hinweg"*
- **Langzeit-Gedächtnis**: Persistenter Speicher pro User/Mandant (wichtige Fakten, Präferenzen)
- **Whisper-Pipeline**: Audio → Transkription → Protokoll → Pendenzenliste (End-to-End)
- **Multi-Step-Planung**: Komplexe Aufgaben in Teilschritte zerlegen mit Checkpoints
- **Report-Generator**: Automatische Statusberichte über mehrere Dokumente/Themen
- **Benachrichtigungen**: "Neues Dokument indexiert das zu deiner letzten Frage passt"

### Stufe 4: Autonomer Projektassistent
> *"Ich überwache, analysiere und handle proaktiv"*
- **Scheduled Tasks**: Regelmässige Reports, Änderungs-Monitoring
- **Multi-Agent-Architektur**: Spezialisierte Sub-Agenten (Recherche, Analyse, Redaktion)
- **Workflow-Automation**: Ketten von Aktionen (Index → Analyse → Report → Versand)
- **Versionierte Wissensbasis**: Änderungshistorie, Rollback, Diff
- **API-Schnittstelle**: Externe Systeme können den Agent ansprechen (Webhook, REST)
