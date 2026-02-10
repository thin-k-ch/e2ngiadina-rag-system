#!/usr/bin/env bash
# WINDSURF Small Smoke Suite (read-only)
# Ziel: < 1 Minute, robuste Ausgabe, kein Abbruch bei Einzel-Fehlern.

ES="http://localhost:9200"
IDX="rag_files_v1"
AGENT="http://localhost:11436"
WEBUI="http://localhost:8086"
OLLAMA="http://localhost:11434"

echo "=== SMALL SUITE: Services HTTP codes ==="
curl -sS -o /dev/null -w "ES=%{http_code}\n"      "$ES/" || true
curl -sS -o /dev/null -w "WEBUI=%{http_code}\n"   "$WEBUI/" || true
curl -sS -o /dev/null -w "AGENT=%{http_code}\n"   "$AGENT/health" || true
curl -sS -o /dev/null -w "OLLAMA=%{http_code}\n"  "$OLLAMA/api/tags" || true
echo

echo "=== SMALL SUITE: ES basic facts ==="
curl -sS "$ES/" | head -c 500; echo; echo
curl -sS "$ES/_cluster/health?pretty" | head -c 900; echo; echo

echo "=== SMALL SUITE: ES count + sample (read-only) ==="
curl -sS "$ES/$IDX/_count?pretty" | head -c 400; echo; echo
curl -sS "$ES/$IDX/_search?size=1&pretty" | head -c 1200; echo; echo

echo "=== SMALL SUITE: Ollama tags (truncated) ==="
curl -sS "$OLLAMA/api/tags" | head -c 1200; echo; echo

echo "=== SMALL SUITE: Agent models (truncated) ==="
curl -sS "$AGENT/v1/models" | head -c 1200; echo; echo

echo "=== SMALL SUITE: Agent chat (LLM-only) ==="
curl -sS "$AGENT/v1/chat/completions"   -H "Content-Type: application/json"   -d '{
    "model":"llama4:latest",
    "messages":[{"role":"user","content":"Antworte nur mit: Ich bin erreichbar."}],
    "stream": false
  }' | head -c 1600; echo; echo

echo "=== SMALL SUITE: Agent chat (RAG probe) ==="
curl -sS "$AGENT/v1/chat/completions"   -H "Content-Type: application/json"   -d '{
    "model":"llama4:latest",
    "messages":[{"role":"user","content":"Suche in den Dokumenten nach \"Projektleitung Konzepthase\" und gib mir den Dateinamen und den relevanten Ausschnitt. Wenn du Quellen hast, gib sie als /open URLs an."}],
    "stream": false
  }' | head -c 2200; echo; echo

echo "=== SMALL SUITE: DONE ==="
exit 0
