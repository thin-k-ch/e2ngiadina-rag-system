# WINDSURF Standard Testcases Pack (Small + Release)

This pack contains a **read-only** test-suite for your WINDSURF stack:

- Elasticsearch (default: http://localhost:9200)
- Agent API (default: http://localhost:11436)
- OpenWebUI (default: http://localhost:8086)
- Ollama (default: http://localhost:11434)

Goals:
- **Small suite**: quick checks after single implementation steps (<< 1 minute).
- **Release suite**: matrix + stronger checks for release-train verification.

## Hard rules
- **NO Elasticsearch index modifications** (no DELETE/PUT of indices/templates/mappings).
- Tests use **curl only** (no jq dependency).
- All HTTP calls enforce **hard timeouts** (avoid hangs).
- Artifacts are written to: `/tmp/windsurf_tests/`

## Run
From repo root (e.g. `/media/felix/RAG/AGENTIC`):

```bash
bash tests/small/run.sh
bash tests/release/run.sh
```

## Definition of Done (DoD)
**DONE** when:
1) Smoke services: ES/AGENT/WEBUI/OLLAMA respond with HTTP 200
2) ES index count is > 0 (and plausible)
3) `tests/small/run.sh` exits 0
4) `tests/release/run.sh` exits 0

If 1-2 pass but tests fail → status is **degraded** (not done).

## Configuration
All scripts support environment overrides:

- `ES` (default `http://localhost:9200`)
- `IDX` (default `rag_files_v1`)
- `AGENT` (default `http://localhost:11436`)
- `WEBUI` (default `http://localhost:8086`)
- `OLLAMA` (default `http://localhost:11434`)
- `MODEL` (default `llama4:latest`)
- `OUTDIR` (default `/tmp/windsurf_tests`)
