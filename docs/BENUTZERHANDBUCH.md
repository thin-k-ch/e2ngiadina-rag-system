# 🔧 Agentic RAG System – Benutzerhandbuch

> **Stand:** 2025-02-12 | **Version:** Phase 4 (Code Execution + Follow-up Context)

---

## Architektur-Überblick

```
OpenWebUI (Port 8086)
    ↓ OpenAI-kompatible API (SSE Streaming)
Agent API (Port 11436)  ←→  PyRunner (Port 9000)
    ↓                          ↑ Python-Code Sandbox
    ├→ Elasticsearch (BM25 Keyword-Suche)
    └→ ChromaDB (Vektor/Semantik-Suche)
    ↓
Ollama (Port 11434) – LLM Inference (GPU)
```

---

## 1. Modelle in OpenWebUI

In OpenWebUI gibt es zwei Gruppen von Modellen:

| Modell | Typ | Beschreibung |
|--------|-----|--------------|
| `rag-llama4:latest` | RAG | Llama 4 mit Dokumentensuche |
| `rag-gpt-oss:latest` | RAG | GPT-OSS mit Dokumentensuche |
| `rag-qwen2.5:3b` | RAG | Kleines, schnelles Modell |
| `rag-apertus:70b-...` | RAG | Grosses 70B Modell |
| `llama4:latest` | Direkt | Ollama direkt, **OHNE RAG** |

### Wichtige Regel:
- **`rag-*` Modelle** → Die Frage geht durch die RAG-Pipeline (Suche + Dokumente + LLM)
- **Modelle ohne `rag-`** → Gehen direkt an Ollama, **keine Dokumentensuche!**

### Thinking Mode (optional):
- `-think` Suffix (z.B. `rag-gpt-oss:latest-think`) aktiviert einen **Zwei-Schritt-Analysemodus**:
  1. LLM analysiert erst die Dokumente (sichtbar in einklappbarem `<think>`-Block)
  2. Dann schreibt es die finale Antwort
- **Nicht nötig** für Code-Ausführung oder normale Suchen
- Nur nützlich für komplexe Analysefragen, wo der Denkprozess sichtbar sein soll

---

## 2. Die 4 Verarbeitungspfade

Wenn eine Frage gestellt wird, prüft das System der Reihe nach:

### Pfad A: Multi-Dokument-Analyse
**Trigger:** Frage referenziert "diese Dokumente", "den Quellen", "diesen Unterlagen" etc.

Beispiele:
```
"Liste mir jeden Fehler in diesen Dokumenten auf"
"Vergleiche die Quellen miteinander"
"Was steht in allen Dokumenten zum Thema X?"
```

**Was passiert:**
1. System lädt die Quellen der **letzten Suche** (max. 5 Dokumente, Volltext aus ES)
2. Schickt alle Dokumente + Frage ans LLM
3. LLM analysiert exhaustiv alle Dokumente

**Voraussetzung:** Vorher muss eine Suche stattgefunden haben, deren Quellen referenziert werden.

---

### Pfad B: Einzel-Dokument-Analyse
**Trigger:** Spezifische Quelle wird mit `[N]` referenziert.

Beispiele:
```
"Analysiere Quelle [1]"
"Was steht in Dokument [3]?"
"Fasse [2] zusammen"
```

**Was passiert:**
1. System lädt Quelle N aus der letzten Suche (Volltext aus ES)
2. Schickt das Dokument + Frage ans LLM
3. LLM analysiert das Einzeldokument im Detail

---

### Pfad C: Normaler RAG-Flow (Standard)
**Trigger:** Jede "normale" Frage, die nicht unter A oder B fällt.

Beispiele:
```
"Suche alle Manteldokumente für den GBT Z5O"
"Welche .eml Dateien gibt es?"
"Was sind die Eignungskriterien?"
```

**Ablauf Schritt für Schritt:**

1. **Glossar-Rewrite** – Fachbegriffe werden expandiert (z.B. "GBT" → "Gotthard Basistunnel")
2. **Query-Expansion** (nur bei Follow-ups) – Keywords aus vorherigen Fragen werden automatisch hinzugefügt
3. **Hybrid-Suche** – Parallel in ES (Keyword/BM25) + ChromaDB (Vektor/Semantik)
4. **Ranking** – Treffer werden nach Relevanz sortiert (Keyword-Boosting)
5. **Kontext-Aufbau** – Top-Dokument-Snippets werden als Kontext zusammengefasst
6. **Follow-up-Kontext** (bei Nachfragen) – Vorherige Quellen (Volltext, max. 3) werden dem Kontext vorangestellt
7. **LLM-Antwort** – Streamt die Antwort basierend auf den Dokumenten
8. **Code-Ausführung** (automatisch) – Falls das LLM einen ```python Block generiert, wird dieser im Sandbox-Runner ausgeführt und das Ergebnis angehängt
9. **Quellen-Links** – Klickbare Links zu den Quelldokumenten

---

### Pfad D: Python-Code-Ausführung (in Pfad C integriert)
**Trigger:** Fragen die Berechnung/Zählung/Dateioperationen erfordern.

Beispiele:
```
"Zähle alle .eml Dateien im Archiv"
"Erstelle eine Tabelle der PDF-Dateien pro Ordner"
"Berechne die Gesamtgrösse aller Dokumente"
"Schreibe einen Python-Code der alle Verträge auflistet"
```

**Was passiert:**
1. Normaler RAG-Flow (Pfad C) läuft
2. Das LLM weiss, dass es Python schreiben kann (steht im System-Prompt)
3. Wenn es einen ```python Block generiert → wird automatisch im **PyRunner** ausgeführt
4. PyRunner hat:
   - **Zugriff** auf das gesamte Projektarchiv (read-only) unter `/data`
   - **Bibliotheken:** `pandas`, `tabulate`, `csv`, `os`, `json`
   - **Timeout:** 25 Sekunden
   - **Kein Internet** – nur lokale Dateien
5. Ergebnis erscheint als `📊 Ergebnis:` Block in der Antwort

**Tipp:** Wenn das LLM von sich aus keinen Code schreibt, explizit sagen: *"Schreibe Python-Code dafür"*

---

## 3. Follow-up-Kontext (Gesprächsverlauf)

Das System merkt sich den Konversationsverlauf **innerhalb eines Chats**:

| Feature | Was passiert |
|---------|-------------|
| **Chat-History** | Die letzten 3 Frage-Antwort-Paare werden dem LLM als Kontext mitgegeben |
| **Query-Expansion** | Keywords aus vorherigen Fragen werden automatisch zur Suche hinzugefügt |
| **Prev-Doc-Context** | Die Top 3 Quellen der letzten Suche werden als Volltext dem Kontext beigefügt |
| **Quellen-Speicher** | Die Quellen jeder Suche werden gespeichert für "Analysiere Quelle [N]" |

### Typischer 3-Schritt-Workflow:
```
1. "Suche mir alle Manteldokumente für den GBT Z5O"
   → Normale Suche, findet Dokumente, zeigt Quellen [1]-[5]

2. "Sind Quellen 1 bis 3 identisch?"
   → Multi-Dokument-Analyse (Pfad A), lädt [1]-[3] als Volltext

3. "Liste mir jeden einzelnen Fehler in diesen Dokumenten auf"
   → Multi-Dokument-Analyse (Pfad A), exhaustive Fehleranalyse
```

---

## 4. Dateitypen die durchsucht werden

```
md, txt, rst, log, json, yaml, yml,
pdf, docx, doc, msg, eml, .eml,
xlsx, xls, pptx, ppt
```

---

## 5. Tipps für optimale Ergebnisse

| Situation | Empfehlung |
|-----------|------------|
| **Erste Suche** | Spezifische Begriffe verwenden: *"Manteldokumente GBT Z5O"* statt *"Dokumente suchen"* |
| **Nachfragen** | Im **gleichen Chat** bleiben – Kontext wird automatisch übernommen |
| **Detailanalyse** | *"Analysiere Quelle [2] im Detail"* → lädt das ganze Dokument |
| **Vergleich** | *"Vergleiche diese Dokumente"* → Pfad A, alle vorherigen Quellen |
| **Dateien zählen/listen** | Explizit *"Schreibe Python-Code"* oder *"Zähle alle..."* |
| **Tabellen** | *"Erstelle eine Tabelle mit..."* → LLM kann Markdown-Tabellen oder Python/pandas nutzen |
| **Komplexe Analyse** | `-think` Modell wählen → sichtbarer Analyseschritt vor der Antwort |
| **Neues Thema** | **Neuen Chat** starten – sonst wird alter Kontext mitgeschleppt |

---

## 6. Grenzen des Systems

- **Kein Schreiben/Ändern** von Dateien (nur read-only)
- **Kein Internet-Zugriff** (kein Web-Search, kein Download)
- **Keine Bild/Scan-Analyse** (nur Text-Extrakt aus PDFs)
- **Kein Chat-übergreifendes Gedächtnis** – jeder Chat ist eine eigene Session
- **Max ~12'000 Zeichen** pro Dokument im Kontext (wird gekürzt)
- **Max 5 Dokumente** bei Multi-Dokument-Analyse
- **Max 3 vorherige Quellen** als Follow-up-Kontext
