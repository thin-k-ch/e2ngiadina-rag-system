# WINDSURF Runbook (minimal, reproducible)

Stand: 2026-02-07 (Europe/Zurich)

Ziel: Nach Reboot/Neu-Deploy **ohne ES-Index-Änderungen** wieder sauber starten und mit einem **Small Smoke Test** verifizieren.

## Endpoints / Ports
- Elasticsearch: http://localhost:9200
- Agent API:      http://localhost:11436
- OpenWebUI:      http://localhost:8086
- Ollama:         http://localhost:11434

## Wichtige Leitplanken
- **Keine ES-Index-Änderungen** (DELETE/PUT Mapping/Settings, Reindex, Templates) ohne explizite Freigabe.
- Test-Skripte sind **read-only**: nur GET und POST `_search`/`_count` bzw. Agent Chat.

## Bekannte IST-Fakten aus den bisherigen Tests
- ES HTTP = 200, Version ~ 8.12.2
- Cluster status: yellow (1 node)
- Index: `rag_files_v1`
- Dokumente: `54,844`
- Content-Qualität (Stichprobe): ~0% leere `content` Felder (200er Sample)
- Agent API: OpenAI-kompatibel:
  - `/health` (200, Body ggf. leer)
  - `/v1/models`
  - `/v1/chat/completions`
  - `/open` **GET** mit Query-Parameter `path` (required)
- Ollama `/api/tags` liefert Modelle (z.B. `llama4:latest`, `qwen2.5:14b`)
- WebUI (8086) liefert 200 (uvicorn)

## FSCrawler (Start / Persistenz)
**Golden path** (Beispiel):
```bash
FSCRAWLER_HOME="/media/felix/RAG/AGENTIC/volumes/fscrawler" ./bin/fscrawler rag1 --loop 1
```

Hinweis: Wenn `FSCRAWLER_HOME` nicht gesetzt ist, kann FSCrawler Settings aus `~/.fscrawler/...` laden.

## Small Smoke Test (unter 1 Minute)
Siehe: `scripts/smoke_small.sh`

## Release Train Suite (umfangreicher)
Siehe: `scripts/smoke_release_train.sh`
