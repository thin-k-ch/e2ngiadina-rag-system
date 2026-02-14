# Phase 8: Document Intelligence Layer (DOCOPS)

## Status: Geplant (nächste Session)

## Kernidee

Ein Batch-Worker der **offline** (Nächte/Wochenenden) durch alle Dokumente im Archiv iteriert
und strukturierte Zusammenfassungen erstellt. Diese bilden ein "Wissens-Layer" über dem
rohen ES-Volltext und ChromaDB-Embeddings.

## Architektur

```
Bestehend:
  ES (rag_files_v1)   ← Volltext + Metadaten
  ChromaDB             ← Embeddings/Chunks

NEU: Batch-Worker (Hintergrund)
  1. Nächstes unbearbeitetes Dokument aus ES
  2. Chunking mit stabilen IDs ({doc_hash}_{page}_{chunk_nr})
  3. MAP-Extraktion via LLM (summary, facts, people, dates, topics)
  4. JSON-Validierung + Quality Gates
  5. Speichern in doc_intel_v1

NEU: doc_intel_v1 (ES-Index oder SQLite)
  - doc_id, doc_path
  - summary (DE, ~200 Wörter)
  - key_facts[] (mit chunk_evidence)
  - decisions[]
  - dates[] (mit Kontext)
  - people[] (Rollen/Zuordnung)
  - topics/tags[]
  - open_issues[]
  - cross_refs[] (Verweise auf andere Dokumente)
  - quality_score (0-100)
  - processed_at, model_used
```

## Rechenbeispiel (~5000 Dokumente)

| Methode              | Pro Dok | Total    | Kosten    |
|----------------------|---------|----------|-----------|
| llama4:8b lokal      | ~30s    | ~42h     | 0 CHF     |
| qwen2.5:72b lokal    | ~3min   | ~10 Tage | 0 CHF     |
| Groq (llama-3.1-8b)  | ~3s     | ~4h      | ~5 USD    |
| Claude API           | ~5s     | ~7h      | ~50-100 USD |

## Phasen

### Phase 8a: Minimal Viable (1 Session)
- SQLite-DB für `doc_intel_v1`
- Batch-Script: iteriert durch ES, extrahiert pro Dokument
- Einfaches Schema: `{doc_id, summary, key_facts[], people[], dates[]}`
- Lokales Modell (llama4) mit simplem Extraktions-Prompt
- Inkrementell: nur unbearbeitete Dokumente (Timestamp-Check)

### Phase 8b: Quality
- DOCOPS-artige MAP-Extraktion für wichtige Dokumente
- Quality Gates als Python-Postprocessing (JSON-Validierung, Snippet-Check)
- Cross-Referenz-Erkennung zwischen Dokumenten

### Phase 8c: Integration
- Neues ReAct-Tool: `search_intelligence` 
- Sucht zuerst in doc_intel, dann Fallback auf ES-Volltext
- Antworten werden strukturierter und präziser

## Vorteile gegenüber aktuellem System

| Jetzt                                | Mit doc_intel                              |
|--------------------------------------|--------------------------------------------|
| Suche in 5000 Volltexten             | Suche in strukturierten Records            |
| LLM muss aus 20K Chars extrahieren   | Vorab extrahierte Facts + Evidence         |
| Keine Cross-Refs                     | Automatische Querverweise                  |
| Antwortqualität modellabhängig       | Vorstrukturiert = konsistenter             |

## Kritische Punkte

1. **Stabile Chunk-IDs**: `{doc_hash}_{page}_{chunk_nr}` - deterministisch
2. **Aktualität**: File-Watcher oder Timestamp-Check für neue/geänderte Dokumente
3. **Halluzination im Batch**: JSON-Validierung + Snippet-Verification Pflicht
4. **80/20-Regel**: 80% brauchbar mit einfachem Prompt, 20% brauchen Spezialbehandlung

## DOCOPS Prompt (Referenz)

Der vollständige DOCOPS_v1 System-Prompt ist unten dokumentiert.
Er ist für **Batch-Verarbeitung** mit Cloud-Modellen oder grossen lokalen Modellen (72B+) gedacht,
NICHT als Live-System-Prompt für 8B-Modelle.

### Grundprinzipien
- **P0 Belegpflicht**: Jede Aussage verweist auf genau einen chunk_id
- **P1 Keine neuen Fakten**: Was nicht im Text steht → "Nicht angegeben"
- **P2 Konflikte**: Explizit markieren, nicht glätten
- **P3 Zahlen**: Nur übernehmen wenn im Chunk vorhanden, mit Einheit
- **P4 Strukturtreue**: section_path nutzen wenn vorhanden
- **P5 Stil**: Knapp, sachlich, keine Floskeln

### 4-Phasen-Modell
1. **PLAN**: Strategie als JSON (task_type, queries, filters, output_spec)
2. **EXECUTE**: Tool-Calls + MAP-Extraktion (generic_records_v1)
3. **VERIFY**: Quality Gates prüfen, max 2 Repair-Runden
4. **DELIVER**: Endergebnis mit Belegpflicht

### Quality Gates
- G1: refs_present_exactly_one (jede Zeile genau ein chunk_id)
- G2: schema_conformance (Output entspricht output_spec)
- G3: coverage (jede relevante Section hat Records)
- G4: conflicts_marked (Widersprüche explizit listen)

### MAP-Schema: generic_records_v1
```json
{
  "doc_id": "...",
  "records": [
    {
      "record_type": "issue|clause|risk|decision|milestone|requirement|other",
      "fields": {
        "id": "...", "title": "...", "description": "...",
        "status": "...", "severity": "...", "date": "...",
        "owner": "...", "component": "...", "tags": ["..."]
      },
      "evidence": {
        "chunk_id": "c000123",
        "page_range": "12-13",
        "snippet": "max 20 Wörter"
      }
    }
  ]
}
```
