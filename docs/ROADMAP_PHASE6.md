# 🚀 Roadmap Phase 6: ReAct Agent + Mandantenfähigkeit

> **Stand:** 2025-02-12 | **Status:** Geplant
> **Ausgangslage:** Phase 5 stabil (5 Pfade, RAG, Code Execution, Transcript→Protocol)
> **Git-Tag Baseline:** `v2025.02.12-phase5`

---

## 1. Ziel-Vision: Sparring-Partner

Ein lokaler AI-Assistent der:
- **Die Welt kennt** – Allgemeinwissen + optional Web-Suche
- **Dokumente analysiert** – Verträge, Protokolle, Korrespondenz durchsuchen und verstehen
- **Auswertungen macht** – Python-Code, Tabellen, Vergleiche
- **Protokolle schreibt** – Transkripte → strukturierte Sitzungsprotokolle
- **Autonom recherchiert** – Suchen → Lesen → Vertiefen → Antworten (Multi-Step)
- **Mandantenfähig ist** – Zwischen Projekten/Repositories umschalten (<10 Min)

---

## 2. Entscheid: Option A (Evolutionär)

**Begründung:**
- Die Kernkomponenten (Suche, Indexierung, Code Execution, Ranking) sind solide und modular
- Nur die Routing-Schicht (main.py) muss refactored werden
- Kein Neuanfang nötig – Investition in Tools bleibt erhalten
- Git-Tags sichern jeden Zwischenstand ab

**Absicherung:**
- Feature-Branch `feature/react-agent` für alles Neue
- `main` bleibt stabil auf Phase 5
- ReAct-Loop mit `max_steps=1` verhält sich wie heute (Fallback)
- Inkrementelle Migration: Ein Tool nach dem anderen

---

## 3. Architektur-Refactoring

### 3.1 Von Pfad-Router zu ReAct-Loop

```
PHASE 5 (heute):                     PHASE 6 (neu):

main.py (750 Zeilen)                 main.py (schlank, ~200 Zeilen)
├─ Pfad A (Multi-Doc)    ──→         ├─ Request entgegennehmen
├─ Pfad B (Single-Doc)   ──→         ├─ ReAct-Loop starten
├─ Pfad C (RAG)          ──→         └─ SSE streamen
├─ Pfad D (Code)         ──→
└─ Pfad E (Transcript)   ──→         react_agent.py (NEU, Kernstück)
                                      ├─ ReAct-Loop (Denken → Tool → Denken)
                                      ├─ Tool-Registry (alle verfügbaren Tools)
                                      └─ Streaming der Zwischen-/Endergebnisse

tools.py                  ──→         tools/ (Verzeichnis, je 1 File pro Tool)
rag_pipeline.py           ──→         ├─ search.py       ← Hybrid-Suche (ES+Chroma)
source_analyzer.py        ──→         ├─ read_doc.py     ← Dokument-Volltext laden
code_executor.py          ──→         ├─ execute.py      ← Python Sandbox (PyRunner)
transcript_processor.py   ──→         ├─ protocol.py     ← Transkript→Protokoll
                                      ├─ web_search.py   ← NEU: Internet-Recherche
                                      └─ list_files.py   ← NEU: Dateibaum erkunden
```

### 3.2 ReAct-Loop (Kernmechanismus)

```python
async def react_loop(user_query, tools, max_steps=6):
    messages = [system_prompt_with_tool_descriptions, user_query]
    
    for step in range(max_steps):
        response = await llm_call_with_tools(messages)
        
        if response.has_tool_call:
            # LLM will ein Tool aufrufen
            tool_name = response.tool_call.name
            tool_args = response.tool_call.arguments
            result = await execute_tool(tool_name, tool_args)
            messages.append(tool_result(result))
            yield phase_update(f"Schritt {step+1}: {tool_name}...")
            # → nächste Iteration: LLM sieht das Ergebnis
        else:
            yield response.text  # Finale Antwort
            return
```

### 3.3 Tool-Definition (Schema für LLM)

```python
TOOLS = [
    {
        "name": "search_documents",
        "description": "Durchsucht das Projektarchiv (ES + Chroma). Nutze dies für jede Frage die sich auf Dokumente, Verträge, E-Mails etc. bezieht.",
        "parameters": {
            "query": "Suchbegriffe",
            "file_types": "Optional: pdf, docx, eml, msg etc."
        }
    },
    {
        "name": "read_document",
        "description": "Liest ein ganzes Dokument (Volltext). Nutze dies wenn du ein spezifisches Dokument im Detail analysieren musst.",
        "parameters": {
            "path": "Pfad zum Dokument (aus search_documents Ergebnis)"
        }
    },
    {
        "name": "execute_python",
        "description": "Führt Python-Code aus. Zugriff auf /data (Projektarchiv, read-only). Verfügbar: pandas, tabulate, os, json, csv.",
        "parameters": {
            "code": "Python-Code"
        }
    },
    {
        "name": "create_protocol",
        "description": "Erstellt ein strukturiertes Sitzungsprotokoll aus einem Transkript-Text.",
        "parameters": {
            "transcript": "Transkript-Text oder Dateipfad",
            "speakers": "Optional: Speaker-Mapping (SPEAKER_00: Name)"
        }
    },
    {
        "name": "web_search",
        "description": "Internet-Suche für allgemeines Wissen, Standards, Normen etc.",
        "parameters": {
            "query": "Suchanfrage"
        }
    }
]
```

---

## 4. Mandantenfähigkeit (Multi-Tenant / Multi-Repository)

### 4.1 Konzept

Ein "Mandant" = ein Projekt-Repository mit eigenen Dokumenten, eigenem Index, eigener Konfiguration.

```
/media/felix/RAG/
├── repos/
│   ├── sbb-tfk-2020/          # Mandant 1 (aktuell: /media/felix/RAG/1)
│   │   ├── documents/          # Quelldokumente
│   │   └── config.yaml         # Mandant-Konfiguration
│   │
│   ├── projekt-alpha/          # Mandant 2
│   │   ├── documents/
│   │   └── config.yaml
│   │
│   └── privat/                 # Mandant 3
│       ├── documents/
│       └── config.yaml
│
├── volumes/                    # Shared infrastructure
│   ├── esdata/                 # ES (alle Mandanten, getrennte Indices)
│   ├── chroma/                 # Chroma (getrennte Collections)
│   └── state/                  # Session State (pro Mandant)
│
└── AGENTIC/                    # Code (mandantenunabhängig)
```

### 4.2 Mandant-Konfiguration (`config.yaml`)

```yaml
# /media/felix/RAG/repos/sbb-tfk-2020/config.yaml
name: "SBB TFK 2020 – Tunnelfunk"
short_name: "sbb-tfk"

# Pfade
document_root: /media/felix/RAG/repos/sbb-tfk-2020/documents

# Elasticsearch
es_index: "rag_sbb_tfk_v1"

# ChromaDB Collections (Prefix)
chroma_prefix: "sbb_tfk"
# → sbb_tfk_documents, sbb_tfk_docx, sbb_tfk_txt, sbb_tfk_msg, sbb_tfk_mail

# Domain-spezifisch
glossary:
  GBT: "Gotthard Basistunnel"
  TFK: "Tunnelfunk"
  RBT: "Rhomberg Bahntechnik"
  FAT: "Werksabnahme (Factory Acceptance Test)"
  SAT: "Standortabnahme (Site Acceptance Test)"

system_prompt_extra: |
  Du bist Spezialist für Schweizer Eisenbahn-Projekte (SBB TFK 2020 - Tunnelfunk).
  Fachgebiete: Projektleitung, Funktechnik, Tunnelfunk.

# Whisper Auto-Korrekturen
transcript_corrections:
  Adnova: Atnova
  Reticum: Rhäticom
  Eppenberg: Dettenberg
```

### 4.3 Umschalten zwischen Mandanten

**Ziel: <10 Minuten**

```
Schritt 1: API-Call oder CLI-Befehl
  POST /v1/tenant/switch  {"tenant": "sbb-tfk"}
  ODER: ./switch-tenant.sh sbb-tfk

Schritt 2: Was passiert automatisch:
  - config.yaml wird geladen
  - ES_INDEX wird umgestellt
  - Chroma Collections werden umgestellt
  - FILE_BASE wird umgestellt
  - Glossar wird geladen
  - System-Prompt wird angepasst

Schritt 3: Falls noch nicht indexiert:
  - docker compose run --rm indexer  (einmalig pro Mandant)
  - Dauer: je nach Dokumentenmenge (5-30 Min)
```

**Nach dem ersten Indexieren = Switch in Sekunden** (nur Config-Reload).

### 4.4 Implementierung

```python
# tenant_manager.py
class TenantManager:
    def __init__(self, repos_dir="/media/felix/RAG/repos"):
        self.repos_dir = repos_dir
        self.current = None
    
    def list_tenants(self) -> list:
        """Alle verfügbaren Mandanten auflisten"""
        ...
    
    def switch(self, tenant_name: str) -> dict:
        """Mandant wechseln – lädt config.yaml, setzt Env-Variablen"""
        config_path = f"{self.repos_dir}/{tenant_name}/config.yaml"
        config = yaml.safe_load(open(config_path))
        
        # Globale Konfiguration umstellen
        os.environ["ES_INDEX"] = config["es_index"]
        os.environ["FILE_BASE"] = config["document_root"]
        os.environ["COLLECTION"] = f"{config['chroma_prefix']}_documents"
        
        # Glossar + Prompt laden
        self.current = config
        return config
    
    def get_glossary(self) -> dict:
        return self.current.get("glossary", {})
    
    def get_system_prompt_extra(self) -> str:
        return self.current.get("system_prompt_extra", "")
    
    def get_transcript_corrections(self) -> dict:
        return self.current.get("transcript_corrections", {})
```

### 4.5 UI-Integration

In OpenWebUI: Mandant als "Model" abbilden:
- `rag-sbb-tfk:latest` → Mandant SBB TFK
- `rag-projekt-alpha:latest` → Mandant Projekt Alpha

Oder: Über Chat-Kommando wechseln:
```
/tenant sbb-tfk
→ "✅ Mandant gewechselt: SBB TFK 2020 – Tunnelfunk (12'450 Dokumente)"
```

---

## 5. Priorisierte Umsetzungsreihenfolge

### Tag 1: Fundament
1. Feature-Branch `feature/react-agent` erstellen
2. `qwen2.5:72b` pullen und als Tool-Calling-Modell testen
3. Minimaler ReAct-Loop mit 2 Tools: `search_documents` + `read_document`
4. Test: "Welche Back-to-Back Regelungen gibt es in unseren Verträgen?"

### Tag 2: Tools migrieren
5. `execute_python` Tool (← code_executor.py)
6. `create_protocol` Tool (← transcript_processor.py)
7. Alle 5 bisherigen Pfade als Tool-Calls funktionsfähig
8. Regressionstests: Alle bisherigen Use Cases müssen weiter funktionieren

### Tag 3: Neue Capabilities
9. `web_search` Tool (Brave API oder Serper.dev)
10. `list_files` / `read_file` Tools für Datei-Exploration
11. Mandant-Konfiguration (config.yaml) Struktur aufsetzen

### Tag 4: Mandantenfähigkeit
12. `tenant_manager.py` implementieren
13. Switch-Mechanismus (API + CLI)
14. Indexer mandantenfähig machen (getrennte Indices/Collections)
15. Test: Zweiten Mandanten anlegen und umschalten

### Tag 5: Polish
16. Phase-Indikatoren im Streaming ("🔍 Suche...", "📄 Lese Dokument...")
17. Merge `feature/react-agent` → `main`
18. Dokumentation aktualisieren
19. Git-Tag `v2025.02.13-phase6`

---

## 6. Modell-Empfehlung für Phase 6

| Modell | Tool-Calling | Qualität | RAM (DGX Spark 128GB) |
|--------|-------------|----------|----------------------|
| `qwen2.5:72b` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ~45GB (Q4) ✅ |
| `llama3.3:70b` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ~45GB (Q4) ✅ |
| `mistral-large:123b` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~75GB (Q4) ✅ |
| `gpt-oss:latest` (~20B) | ⭐⭐⭐ | ⭐⭐⭐ | ~12GB ✅ |
| `llama4:latest` (Scout 108B) | ⭐⭐⭐ | ⭐⭐⭐ | ~65GB ✅ |

**Empfehlung:** `qwen2.5:72b` als Default – bestes Verhältnis Tool-Calling / Qualität / RAM.

---

## 7. Risiken und Mitigationen

| Risiko | Mitigation |
|--------|-----------|
| ReAct-Loop halluziniert Tool-Calls | Strikte Tool-Schema-Validierung + Max Steps |
| Modell ignoriert Tools | Few-Shot-Beispiele im System-Prompt |
| Regression bestehender Features | Testfälle vor Refactoring dokumentieren |
| Mandanten-Switch bricht Suche | Getrennte ES-Indices, kein Shared State |
| Performance bei 72B Modell | Keep-Alive 24h, erste Anfrage ~30s, dann schnell |

---

## 8. Testfälle (vor Refactoring dokumentieren)

### Bestehende Features (müssen weiter funktionieren)
1. **RAG-Suche:** "Suche Manteldokumente GBT Z5O" → Treffer mit Quellen
2. **Follow-up:** "Was steht in Quelle [2]?" → Volltext-Analyse
3. **Multi-Doc:** "Vergleiche diese Dokumente" → Alle vorherigen Quellen
4. **Code Execution:** "Zähle alle .eml Dateien" → Python + Ergebnis
5. **Transkript→Protokoll:** Inline-Text → Strukturiertes Protokoll
6. **Dateipfad-Referenz:** "Protokoll aus /datei.txt" → Datei laden + Protokoll

### Neue Features (Phase 6)
7. **Multi-Step-Recherche:** "Welche Vertragsklauseln zu Back-to-Back?" → Suchen → Lesen → Vergleichen
8. **Web-Suche:** "Was verlangt ISO 9001 Kapitel 7?" → Internet + Antwort
9. **Mandant-Switch:** `/tenant sbb-tfk` → Config geladen, Index gewechselt
10. **Autonome Exploration:** "Welche Ordnerstruktur haben wir?" → list_files → Übersicht
