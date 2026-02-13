# 🚀 Roadmap Phase 7: Qualität, UX & Erweiterungen

> **Stand:** 2026-02-13 | **Status:** In Arbeit
> **Ausgangslage:** Phase 6 stabil (ReAct Agent, 7 Tools, Multi-Tenant, File-Upload Protokoll)
> **Git-Tag Baseline:** `v2026.02.13-phase6-hotfix`

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

#### 3.3 Indexer-Verbesserungen
- **Inkrementelles Re-Indexing**: Nur geänderte Dateien neu indexieren (Manifest existiert)
- **Neue Formate**: HTML, RTF, CSV nativ (nicht nur via Python)
- **Metadaten-Extraktion**: Autor, Datum, Betreff aus E-Mails und Office-Dokumenten

### Prio 2: UX-Verbesserungen

#### 3.4 Bessere Quellen-Links
- Quellen als klickbare Links die das Dokument direkt öffnen
- PDF: Direkt zur Seite springen (wenn Seitennummer bekannt)
- Quellen-Preview: Kurzes Snippet unter jedem Link

#### 3.5 Chat-Kontext verbessern
- Längerer Konversationsverlauf (aktuell 3 Turns)
- Session-Zusammenfassung für lange Gespräche
- "Merke dir X" → Persistenter Notiz-Speicher pro Mandant

#### 3.6 Fortschrittsanzeige
- Bessere Phase-Indikatoren: "Schritt 2/4: Lese Dokument..."
- Geschätzte Wartezeit bei langen Operationen
- Token-Zähler / Context-Window-Auslastung (für Debugging)

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

## 4. Empfohlene Reihenfolge

### Sofort (heute Abend)
1. Web-Search API-Key einrichten (5 Min)
2. ES neu bauen ohne Fess-Plugins (5 Min)

### Kurzfristig (nächste Session)
3. Antwortqualität: System-Prompt Tuning
4. Fehlertoleranz: Retry + bessere Fehlermeldungen
5. Quellen-Links: Direkt-Öffnung verbessern

### Mittelfristig
6. Whisper-Integration (Audio → Transkript → Protokoll)
7. Dokumenten-Vergleich
8. Zusammenfassungs-Modus
9. Zweiter Mandant aufsetzen

### Langfristig
10. Monitoring & strukturiertes Logging
11. Security (API-Keys, Rate-Limiting)
12. Automatische Reports

---

## 5. Hardware-Potential (DGX Spark)

Die NVIDIA DGX Spark hat 128GB RAM und GPU. Aktuell genutzt:
- Ollama: ~125GB gesamt (llama4:67GB, deepseek-r1:42GB, gpt-oss:13GB, qwen2.5:3b, apertus:43GB, mxbai-embed)
- ES: ~2GB
- ChromaDB + Services: ~4GB
- **Hinweis**: Nicht alle Modelle gleichzeitig im VRAM – Ollama lädt/entlädt automatisch

Möglichkeiten:
- **Whisper Large V3**: ~3GB GPU RAM → lokale Transkription
- **Cross-Encoder Reranking**: ~1GB → bessere Suchergebnisse
- **Hinweis zu DeepSeek-R1:70b**: Reasoning-Modell mit `<think>`-Phase, ~3 Min/Step, ideal für komplexe Analysen aber langsam
