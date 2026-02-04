#!/bin/bash
set -e
cd "$(dirname "$0")/.."
docker compose up -d --build ollama runner agent_api
echo "Agent API health: http://localhost:11436/health"
