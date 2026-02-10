#!/usr/bin/env bash
# tests/small/02_exact_phrase_es_vs_agent.sh
set +e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/tests/lib/http_helper.sh"

ES="${ES:-http://localhost:9200}"
IDX="${IDX:-rag_files_v1}"
AGENT="${AGENT:-http://localhost:11436}"
MODEL="${MODEL:-llama4:latest}"
OUTDIR="${OUTDIR:-/tmp/windsurf_tests}"
mkdir -p "$OUTDIR"

PHRASE="Projektleitung Konzepthase"
EXPECTED_FILE="Sockelkosten Konzeptphase.xlsx"

echo "== Ground truth: ES match_phrase should return 1 hit for [$PHRASE] =="
es_body="$OUTDIR/es_phrase.json"
cat > "$OUTDIR/es_phrase_req.json" <<JSON
{
  "size": 1,
  "_source": ["file.filename","path.real","file.url"],
  "query": { "match_phrase": { "content": "${PHRASE}" } }
}
JSON

http_post_json "$ES/$IDX/_search?pretty" "$OUTDIR/es_phrase_req.json" "$es_body"
cat "$es_body" | head -c 800; echo

es_compact="$(json_compact "$es_body")"
echo "$es_compact" | grep -q '"total":{"value":1' || exit 30
echo "$es_compact" | grep -q "$EXPECTED_FILE" || exit 31

echo "== Agent exact phrase mode should return the same filename =="
agent_req="$OUTDIR/agent_phrase_req.json"
agent_body="$OUTDIR/agent_phrase_resp.json"

cat > "$agent_req" <<JSON
{
  "model": "${MODEL}",
  "stream": false,
  "messages": [
    {"role":"user","content":"Suche exakt die Phrase: ${PHRASE}. Gib nur die Dateinamen der besten Treffer."}
  ]
}
JSON

http_post_json "$AGENT/v1/chat/completions" "$agent_req" "$agent_body"
cat "$agent_body" | head -c 1200; echo

agent_compact="$(json_compact "$agent_body")"
echo "$agent_compact" | grep -q "$EXPECTED_FILE" || exit 40
echo "$agent_compact" | grep -q "/open?path=" || exit 41

exit 0
