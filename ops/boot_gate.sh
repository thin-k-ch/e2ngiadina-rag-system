#!/usr/bin/env bash
set -euo pipefail

# AGENTIC Boot-Gate
# - Fails fast if any core contract is broken:
#   1) ES ready + count
#   2) Agent non-stream LLM chat must answer "OK"
#   3) Agent SSE stream must emit immediate "data:" + eventually "[DONE]"

COMPOSE_DIR="${COMPOSE_DIR:-/media/felix/RAG/AGENTIC}"
ES_URL="${ES_URL:-http://localhost:9200}"
AGENT_URL="${AGENT_URL:-http://localhost:11436}"
INDEX_NAME="${INDEX_NAME:-rag_files_v1}"
LLM_MODEL="${LLM_MODEL:-llama4:latest}"

CONNECT_TIMEOUT=2
MAX_TIME=15

fail() { echo "❌ BOOT-GATE FAIL: $*" >&2; exit 1; }
ok()   { echo "✅ $*"; }

cd "$COMPOSE_DIR" || fail "Cannot cd to $COMPOSE_DIR"

echo "=== BOOT-GATE: start stack (if needed) ==="
docker compose config >/dev/null || fail "docker compose config invalid"
docker compose up -d || fail "docker compose up -d failed"

echo "=== Wait: ES ready (HTTP 200) ==="
for i in {1..20}; do
  if curl -fsS --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" "$ES_URL" >/dev/null 2>&1; then
    ok "Elasticsearch HTTP 200 ($ES_URL)"
    break
  fi
  sleep 1
  [[ $i -eq 20 ]] && fail "Elasticsearch not ready after 20s"
done

echo "=== Check: ES cluster health (yellow/green) ==="
HEALTH_JSON="$(curl -fsS --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" "$ES_URL/_cluster/health?pretty")" || fail "ES health endpoint failed"
echo "$HEALTH_JSON" | grep -q '"status" : "yellow"\|"status" : "green"' || fail "ES cluster not yellow/green"
ok "ES cluster status is yellow/green"

echo "=== Check: ES index count ($INDEX_NAME) ==="
COUNT_JSON="$(curl -fsS --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" "$ES_URL/$INDEX_NAME/_count?pretty")" || fail "ES count endpoint failed"
echo "$COUNT_JSON"
COUNT_VAL="$(echo "$COUNT_JSON" | grep -E '"count"\s*:\s*' | head -n1 | sed -E 's/.*"count"\s*:\s*([0-9]+).*/\1/')"
[[ -n "${COUNT_VAL:-}" ]] || fail "Could not parse ES count"
[[ "$COUNT_VAL" -ge 1000 ]] || fail "ES count too low ($COUNT_VAL) — index missing?"
ok "ES count looks sane ($COUNT_VAL)"

echo "=== Wait: Agent API /health ==="
for i in {1..20}; do
  if curl -fsS --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" "$AGENT_URL/health" >/dev/null 2>&1; then
    ok "Agent API HTTP 200 ($AGENT_URL/health)"
    break
  fi
  sleep 1
  [[ $i -eq 20 ]] && fail "Agent API not ready after 20s"
done

echo "=== Check: Agent non-stream chat should return OK ==="
NONSTREAM="$(curl -fsS --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" \
  -H 'Content-Type: application/json' \
  "$AGENT_URL/v1/chat/completions" \
  -d "{\"model\":\"$LLM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Sag nur OK.\"}],\"stream\":false}" )" \
  || fail "Agent non-stream call failed"

echo "$NONSTREAM" | head -c 800; echo
echo "$NONSTREAM" | grep -q '"content":"OK"\|"content":"OK.' || fail "Non-stream did not return OK (routing broken?)"
echo "$NONSTREAM" | grep -q 'Nicht in den Dokumenten gefunden' && fail "Non-stream routed to RAG incorrectly"
ok "Non-stream chat returns OK"

echo "=== Check: Agent SSE streaming emits immediate data: and [DONE] ==="
# Expect within 1s at least one "data:" line; then within MAX_TIME we must see [DONE]
set +e
STREAM_OUT="$(curl -sS --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" \
  -H 'Content-Type: application/json' \
  "$AGENT_URL/v1/chat/completions" \
  -d "{\"model\":\"$LLM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Sag nur OK.\"}],\"stream\":true}" )"
CURL_RC=$?
set -e
[[ $CURL_RC -eq 0 ]] || fail "Streaming request timed out / failed (curl rc=$CURL_RC)"

echo "$STREAM_OUT" | head -n 20
echo "$STREAM_OUT" | grep -q '^data: ' || fail "Streaming output missing 'data:' lines (SSE broken)"
echo "$STREAM_OUT" | grep -q 'data: \[DONE\]' || fail "Streaming output missing [DONE]"
ok "SSE streaming format OK"

echo "=== Check: Chroma volume presence (file exists) ==="
# We only verify persistence presence; functional chroma is indirectly exercised by agent in later tests.
# If you want a functional chroma check, add a small query that must hit rag_files_v1_chunks.
docker compose exec -T agent_api sh -lc 'ls -lh /chroma/chroma.sqlite3 2>/dev/null' || fail "Chroma sqlite not found inside agent container"
ok "Chroma sqlite present in container"

echo
echo "🎉 BOOT-GATE PASS — Stack is up and core contracts hold."
