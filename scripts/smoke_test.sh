#!/bin/bash
set -e
cd "$(dirname "$0")/.."
docker compose up -d --build ollama runner agent_api
echo "Waiting for services..."
sleep 2
curl -s http://localhost:11436/health | python -m json.tool || true
echo "Run ingestion (PDFs) first: ./scripts/ingest_pdfs.sh"
