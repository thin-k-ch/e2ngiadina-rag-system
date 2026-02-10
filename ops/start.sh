#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"

echo "=== START: AGENTIC stack boot & sanity ==="
cd "$ROOT_DIR"

docker compose -f "$COMPOSE_FILE" config >/dev/null
echo "OK: docker compose config valid"

echo "Bringing stack up..."
docker compose -f "$COMPOSE_FILE" up -d

echo "Waiting a few seconds for services..."
sleep 6

echo
echo "=== Sanity: Elasticsearch ==="
curl -fsS --connect-timeout 2 --max-time 10 http://localhost:9200 >/dev/null
echo "OK: Elasticsearch is HTTP 200 (http://localhost:9200)"

echo
echo "=== Sanity: Elasticsearch count (rag_files_v1) ==="
curl -fsS --connect-timeout 2 --max-time 10 http://localhost:9200/rag_files_v1/_count?pretty

echo
echo "=== Sanity: Agent health ==="
curl -fsS --connect-timeout 2 --max-time 10 http://localhost:11436/health >/dev/null
echo "OK: Agent API is HTTP 200 (http://localhost:11436/health)"

echo
echo "=== START OK ==="
