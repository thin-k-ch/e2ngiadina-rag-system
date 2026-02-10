#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"

echo "=== STOP: AGENTIC stack graceful shutdown ==="
cd "$ROOT_DIR"

echo "Current compose services:"
docker compose -f "$COMPOSE_FILE" ps || true

echo
echo "Stopping stack (docker compose down)..."
docker compose -f "$COMPOSE_FILE" down

sync
echo
echo "=== STOP OK ==="
echo "Safe to reboot/shutdown the host now."
