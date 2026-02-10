#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"

echo "=== START_SAFE: AGENTIC stack boot ==="
echo "Project: ${ROOT_DIR}"

# --- Ensure docker daemon is ready (wait up to 90s) ---
echo "Checking Docker daemon readiness..."
if ! docker info >/dev/null 2>&1; then
  echo "Docker not ready yet. Waiting up to 90s..."
  for i in $(seq 1 90); do
    if docker info >/dev/null 2>&1; then
      echo "OK: Docker is ready."
      break
    fi
    sleep 1
  done
fi

if ! docker info >/dev/null 2>&1; then
  echo "❌ START_SAFE FAIL: Docker daemon not reachable."
  echo "Fix: sudo systemctl enable --now docker && sudo usermod -aG docker \$USER (then relogin)"
  exit 1
fi

# --- Validate compose ---
docker compose -f "$COMPOSE_FILE" config >/dev/null
echo "OK: docker compose config valid"

# --- Start stack ---
echo "Bringing stack up..."
docker compose -f "$COMPOSE_FILE" up -d

echo "Waiting for services (with timeouts)..."

# Helper: wait for HTTP 200
wait_http_200() {
  local name="$1"
  local url="$2"
  local tries="${3:-40}"
  local sleep_s="${4:-2}"

  echo "-> Waiting for ${name}: ${url}"
  for i in $(seq 1 "$tries"); do
    if curl -fsS --connect-timeout 2 --max-time 5 "$url" >/dev/null 2>&1; then
      echo "OK: ${name} is HTTP 200"
      return 0
    fi
    sleep "$sleep_s"
  done
  echo "❌ FAIL: ${name} not ready: ${url}"
  return 1
}

# ES root should be 200 once ready
wait_http_200 "Elasticsearch" "http://localhost:9200" 60 2

# ES cluster health check (yellow/green)
echo "=== Sanity: Elasticsearch cluster health ==="
HEALTH="$(curl -fsS --connect-timeout 2 --max-time 10 http://localhost:9200/_cluster/health?pretty)"
echo "$HEALTH" | head -n 40
if ! echo "$HEALTH" | grep -qE '"status"\s*:\s*"(yellow|green)"'; then
  echo "❌ FAIL: Elasticsearch cluster status not yellow/green"
  exit 1
fi
echo "OK: Elasticsearch cluster status yellow/green"

# ES count check (rag_files_v1)
echo
echo "=== Sanity: Elasticsearch count (rag_files_v1) ==="
COUNT_JSON="$(curl -fsS --connect-timeout 2 --max-time 10 http://localhost:9200/rag_files_v1/_count?pretty)"
echo "$COUNT_JSON"
if ! echo "$COUNT_JSON" | grep -q '"count"'; then
  echo "❌ FAIL: ES count endpoint did not return count"
  exit 1
fi

# Agent health
echo
wait_http_200 "Agent API" "http://localhost:11436/health" 60 2

# Chroma volume exists (as seen from agent container mount)
# We check via docker exec into agent container by service name, not container name.
echo
echo "=== Sanity: Chroma volume presence (inside agent_api) ==="
AGENT_CID="$(docker compose -f "$COMPOSE_FILE" ps -q agent_api || true)"
if [[ -z "${AGENT_CID}" ]]; then
  echo "❌ FAIL: agent_api container not found"
  exit 1
fi
docker exec "$AGENT_CID" sh -lc 'ls -lh /chroma/chroma.sqlite3 || true'

# Agent non-stream exact phrase check (fast, deterministic)
echo
echo "=== Sanity: Agent API non-stream (exact phrase) ==="
RESP="$(curl -fsS --connect-timeout 2 --max-time 20 \
  -H 'Content-Type: application/json' \
  http://localhost:11436/v1/chat/completions \
  -d '{"model":"llama4:latest","messages":[{"role":"user","content":"Suche exakt die Phrase: Projektleitung Konzepthase. Gib nur die Dateinamen der besten Treffer."}],"stream":false}')"
echo "$RESP" | head -c 1500; echo
echo "$RESP" | grep -q "Sockelkosten Konzeptphase.xlsx" && echo "OK: exact phrase returned expected file" || {
  echo "❌ FAIL: agent did not return expected file"
  exit 1
}

echo
echo "=== START_SAFE OK ==="
echo "Stack is up and passed basic sanity checks."

