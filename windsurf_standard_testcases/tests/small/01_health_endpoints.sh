#!/usr/bin/env bash
# tests/small/01_health_endpoints.sh
set +e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/tests/lib/http_helper.sh"

ES="${ES:-http://localhost:9200}"
IDX="${IDX:-rag_files_v1}"
AGENT="${AGENT:-http://localhost:11436}"
WEBUI="${WEBUI:-http://localhost:8086}"
OLLAMA="${OLLAMA:-http://localhost:11434}"
OUTDIR="${OUTDIR:-/tmp/windsurf_tests}"
mkdir -p "$OUTDIR"

echo "== A) Service HTTP codes =="
es_code="$(http_code "$ES/")"
agent_code="$(http_code "$AGENT/health")"
webui_code="$(http_code "$WEBUI/")"
ollama_code="$(http_code "$OLLAMA/api/tags")"

echo "ES=$es_code  AGENT=$agent_code  WEBUI=$webui_code  OLLAMA=$ollama_code"

[ "$es_code" = "200" ] || exit 10
[ "$agent_code" = "200" ] || exit 11
[ "$webui_code" = "200" ] || exit 12
[ "$ollama_code" = "200" ] || exit 13

echo "== B) ES index count > 0 =="
count_out="$OUTDIR/es_count.json"
http_get_body "$ES/$IDX/_count?pretty" "$count_out"
cat "$count_out" | head -c 300; echo

compact="$(json_compact "$count_out")"
echo "$compact" | grep -q '"count":' || exit 20
count="$(echo "$compact" | sed -n 's/.*"count":\([0-9][0-9]*\).*/\1/p')"
[ -n "$count" ] || exit 21
[ "$count" -gt 0 ] || exit 22
echo "Count=$count"
exit 0
