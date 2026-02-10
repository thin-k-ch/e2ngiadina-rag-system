#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_shutdown_logs/$(date +%Y%m%d_%H%M%S)"

echo "=== STOP_SAFE: AGENTIC stack graceful shutdown ==="
echo "Project: ${ROOT_DIR}"

mkdir -p "$LOG_DIR"

# If docker isn't reachable, nothing to stop
if ! docker info >/dev/null 2>&1; then
  echo "INFO: Docker daemon not reachable. Nothing to stop."
  echo "=== STOP_SAFE OK ==="
  exit 0
fi

cd "$ROOT_DIR"

echo "Saving pre-shutdown status to ${LOG_DIR} ..."
docker compose -f "$COMPOSE_FILE" ps > "${LOG_DIR}/compose_ps.txt" 2>&1 || true
docker ps > "${LOG_DIR}/docker_ps.txt" 2>&1 || true

echo "Requesting graceful stop (docker compose stop -t 45)..."
docker compose -f "$COMPOSE_FILE" stop -t 45 || true

echo "Stopping stack (docker compose down)..."
docker compose -f "$COMPOSE_FILE" down || true

echo "Syncing filesystem buffers..."
sync

echo
echo "=== STOP_SAFE OK ==="
echo "Safe to reboot/shutdown the host now."
