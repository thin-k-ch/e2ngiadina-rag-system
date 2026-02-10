#!/usr/bin/env bash
# tests/release/02_services_headers.sh
set +e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/tests/lib/http_helper.sh"

ES="${ES:-http://localhost:9200}"
AGENT="${AGENT:-http://localhost:11436}"
WEBUI="${WEBUI:-http://localhost:8086}"
OLLAMA="${OLLAMA:-http://localhost:11434}"
OUTDIR="${OUTDIR:-/tmp/windsurf_tests}"
mkdir -p "$OUTDIR"

echo "== Headers (top 20) =="

h1="$OUTDIR/h_es.txt"
http_get_head "$ES/" "$h1"
echo "--- ES ---"
head -n 20 "$h1" || true

h2="$OUTDIR/h_webui.txt"
http_get_head "$WEBUI/" "$h2"
echo "--- WEBUI ---"
head -n 20 "$h2" || true

h3="$OUTDIR/h_agent.txt"
http_get_head "$AGENT/health" "$h3"
echo "--- AGENT ---"
head -n 20 "$h3" || true

h4="$OUTDIR/h_ollama.txt"
http_get_head "$OLLAMA/api/tags" "$h4"
echo "--- OLLAMA ---"
head -n 20 "$h4" || true

exit 0
