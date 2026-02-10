#!/usr/bin/env bash
# tests/release/01_matrix_exact_phrases.sh
set +e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/tests/lib/http_helper.sh"

ES="${ES:-http://localhost:9200}"
IDX="${IDX:-rag_files_v1}"
AGENT="${AGENT:-http://localhost:11436}"
MODEL="${MODEL:-llama4:latest}"
OUTDIR="${OUTDIR:-/tmp/windsurf_tests}"
mkdir -p "$OUTDIR"

phrases=(
  "Projektleitung Konzepthase|Sockelkosten Konzeptphase.xlsx|1"
  "Tabelle1|.xlsx|1"
)

echo "== Matrix exact phrase checks (ES vs Agent) =="

i=0
for row in "${phrases[@]}"; do
  i=$((i+1))
  phrase="${row%%|*}"
  rest="${row#*|}"
  expected="${rest%%|*}"
  expected_hits="${row##*|}"

  echo
  echo "--- Case $i: phrase=[$phrase] expected=[$expected] hits=$expected_hits ---"

  es_req="$OUTDIR/release_es_req_${i}.json"
  es_body="$OUTDIR/release_es_resp_${i}.json"
  cat > "$es_req" <<JSON
{
  "size": 3,
  "_source": ["file.filename","path.real","file.url"],
  "query": { "match_phrase": { "content": "${phrase}" } }
}
JSON
  http_post_json "$ES/$IDX/_search?pretty" "$es_req" "$es_body"
  es_compact="$(json_compact "$es_body")"

  if [ "$expected_hits" = "1" ]; then
    echo "$es_compact" | grep -q '"total":{"value":1' || exit 30
  else
    echo "$es_compact" | grep -q '"total":{"value":' || exit 31
  fi

  echo "$es_compact" | grep -qi "$expected" || exit 32

  agent_req="$OUTDIR/release_agent_req_${i}.json"
  agent_body="$OUTDIR/release_agent_resp_${i}.json"
  cat > "$agent_req" <<JSON
{
  "model": "${MODEL}",
  "stream": false,
  "messages": [
    {"role":"user","content":"Suche exakt die Phrase: ${phrase}. Gib nur die Dateinamen der besten Treffer."}
  ]
}
JSON
  http_post_json "$AGENT/v1/chat/completions" "$agent_req" "$agent_body"
  agent_compact="$(json_compact "$agent_body")"

  echo "$agent_compact" | grep -qi "$expected" || exit 40
  echo "$agent_compact" | grep -q "/open?path=" || exit 41
done

echo
echo "Matrix OK."
exit 0
