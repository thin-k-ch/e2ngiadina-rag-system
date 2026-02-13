# 🔧 Agentic RAG System – Benutzerhandbuch

> **Stand:** 2026-02-13 | **Version:** Phase 6 (ReAct Agent + Multi-Tenant)

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
| `rag-llama4:latest` | ReAct | Llama 4 mit autonomer Tool-Nutzung |
| `rag-qwen2.5:72b` | ReAct | Qwen 72B – bestes Tool-Calling |
| `rag-llama3.3:70b` | ReAct | Llama 3.3 70B |
| `rag-gpt-oss:latest` | RAG | GPT-OSS mit klassischer Dokumentensuche |
| `llama4:latest` | Direkt | Ollama direkt, **OHNE RAG** |

### Wichtige Regel:
- **`rag-*` Modelle** → Die Frage geht durch die RAG-Pipeline (Suche + Dokumente + LLM)
- **Modelle ohne `rag-`** → Gehen direkt an Ollama, **keine Dokumentensuche!**

### ReAct-Modelle (empfohlen):
- `rag-llama4:latest`, `rag-qwen2.5:72b`, `rag-llama3.3:70b` nutzen den **ReAct Agent** (Pfad F)
- Der Agent entscheidet **autonom** welche Tools er braucht: Suchen, Lesen, Code ausführen, etc.
- Mehrstufige Recherche: Suchen → Dokument lesen → Vertiefen → Antworten

### Thinking Mode (optional):
- `-think` Suffix (z.B. `rag-gpt-oss:latest-think`) aktiviert einen **Zwei-Schritt-Analysemodus**
- Nur nützlich für komplexe Analysefragen, wo der Denkprozess sichtbar sein soll

---

## 2. Die 6 Verarbeitungspfade

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

### Pfad E: Transkript → Protokoll (NEU)
**Trigger:** Keywords wie "Protokoll", "Transkript", "Pendenzen", "Whisper", "Sitzung" etc.

Beispiele:
```
"Erstelle ein Protokoll aus diesem Transkript: [Text]"
"Erstelle ein Sitzungsprotokoll aus der Datei /AuditVorbereitung.txt"
"Verarbeite dieses Meeting-Transkript zu einem Protokoll mit Pendenzenliste"
```

**Was passiert:**
1. RAG-Suche wird **komplett übersprungen** – Volltext geht direkt ans LLM
2. **Vorverarbeitung:**
   - Header-Mappings werden angewendet (z.B. `SPEAKER_00: Felix`)
   - Auto-Korrekturen für bekannte Whisper-Fehler (Adnova→Atnova, etc.)
3. Spezialisierter **Protokoll-Prompt** generiert:
   - Sitzungskopf (Datum, Teilnehmende, Thema)
   - Traktanden mit Diskussion, Aussagen, Entscheidungen
   - Beschlüsse und Entscheidungen (nummeriert)
   - **Pendenzenliste** als Tabelle (Nr, Pendenz, Verantwortlich, Termin, Status)
   - Nächste Schritte
4. **Dynamisches Context Window** (num_ctx) – passt sich automatisch an die Textlänge an
5. **Dynamischer Timeout** – bis 10 Minuten für lange Transkripte

**3 Eingabemethoden:**

| Methode | Anleitung | Qualität |
|---------|-----------|----------|
| **📎 Datei-Upload** (empfohlen) | Datei in OpenWebUI hochladen + "Erstelle Protokoll" | ✅ Bis ~200K Zeichen |
| **Text einfügen** | Text direkt in den Chat einfügen | ✅ Voll |
| **Dateipfad** | `Erstelle Protokoll aus der Datei /mein_transkript.txt` | ✅ Voll |

**Header-Format für Speaker-Ersetzung:**

Am Anfang der Transkript-Datei manuell hinzufügen:
```
SPEAKER_00: Felix
SPEAKER_01: Stefano

SPEAKER_00 [0.00-5.02]:
Entschuldigung, so...
```
→ Alle `SPEAKER_00` werden automatisch durch `Felix` ersetzt.

---

### Pfad F: ReAct Agent – Autonome Recherche (NEU Phase 6)
**Trigger:** Modelle `llama4:latest`, `qwen2.5:72b`, `llama3.3:70b` (automatisch)

Beispiele:
```
"Was steht im Werkvertrag über Gewährleistung?"
"Welche Back-to-Back Regelungen gibt es in unseren Verträgen?"
"Wie viele PDF-Dateien haben wir pro Ordner?"
"Welche Ordnerstruktur haben wir?"
```

**Was passiert:**
1. Der Agent analysiert die Frage und entscheidet **autonom** welche Tools er braucht
2. **Mehrstufig:** Suchen → Dokument lesen → ggf. Code ausführen → Antworten
3. Max. 6 Schritte, dann wird mit dem was vorhanden ist geantwortet
4. Quellen-Links werden automatisch am Ende angehängt

**7 verfügbare Tools:**

| Tool | Beschreibung |
|------|-------------|
| `search_documents` | Hybrid-Suche (ES + Chroma) im Projektarchiv |
| `read_document` | Dokument vollständig aus ES laden |
| `execute_python` | Python-Code im Sandbox-Runner ausführen |
| `create_protocol` | Transkript → strukturiertes Protokoll |
| `list_files` | Verzeichnisinhalt auflisten |
| `read_file` | Datei direkt lesen (CSV, TXT, Log) |
| `web_search` | Internet-Suche (braucht API-Key) |

**Forced-Step-Mechanismus:**
- Dateisystem-Fragen ("wie viele Dateien", "Ordnerstruktur") → automatisch Python-Code
- Dokument-Fragen → erzwungene Suche falls LLM kein Tool aufruft
- Grüsse/Chat → direkte Antwort ohne Tool

---

### Pfad C: Normaler RAG-Flow (Fallback)
**Trigger:** Jede Frage mit einem Modell das NICHT in den ReAct-Modellen ist.

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
- **Web-Suche** nur mit API-Key (BRAVE_API_KEY oder SERPER_API_KEY in docker-compose.yml)
- **Keine Bild/Scan-Analyse** (nur Text-Extrakt aus PDFs)
- **Kein Chat-übergreifendes Gedächtnis** – jeder Chat ist eine eigene Session
- **Max ~12'000 Zeichen** pro Dokument im Kontext (wird gekürzt bei RAG-Pfad)
- **Max 5 Dokumente** bei Multi-Dokument-Analyse
- **Max 3 vorherige Quellen** als Follow-up-Kontext
- **Datei-Upload via OpenWebUI** funktioniert bis ~200K Zeichen (RAG_TOP_K=50, CHUNK_SIZE=4000)
- **Context Window:** Dynamisch bis 128K Tokens (modellabhängig)
- **ReAct Agent:** Max 6 Schritte pro Anfrage
