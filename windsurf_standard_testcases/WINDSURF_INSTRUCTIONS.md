# WINDSURF – Copy/Paste Runbook (Pragmatic)

> Purpose: run the standard tests autonomously and reproducibly.
> Rule: **READ-ONLY** – do not modify ES indices.

## 0) Quick smoke (30s)
```bash
cd /media/felix/RAG/AGENTIC

curl -sS -o /dev/null -w "ES HTTP=%{http_code}\n" http://localhost:9200/
curl -sS -o /dev/null -w "AGENT HTTP=%{http_code}\n" http://localhost:11436/health
curl -sS -o /dev/null -w "WEBUI HTTP=%{http_code}\n" http://localhost:8086/
curl -sS -o /dev/null -w "OLLAMA HTTP=%{http_code}\n" http://localhost:11434/api/tags
curl -sS http://localhost:9200/rag_files_v1/_count?pretty | head -c 200 ; echo
```

## 1) Small suite
```bash
cd /media/felix/RAG/AGENTIC
bash tests/small/run.sh
echo "SMALL_RC=$?"
```

## 2) Release suite
```bash
cd /media/felix/RAG/AGENTIC
bash tests/release/run.sh
echo "RELEASE_RC=$?"
```

## Interpretation
- RC=0: pass
- RC!=0: fail; check the printed failing test section and the artifacts in `/tmp/windsurf_tests/`.
