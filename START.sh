#!/bin/bash
set -e
cd "$(dirname "$0")"

# E2NGIADINA RAG System - One-Click Startup
echo "🚀 E2NGIADINA RAG System – Startup"
echo "==================================="

# Check Docker daemon
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker daemon not running. Starting..."
    sudo systemctl start docker
    sleep 3
fi

# Step 1: Build code services (agent_api + runner) from latest source
echo ""
echo "� Building code services (agent_api, runner)..."
docker compose build agent_api runner 2>&1 | tail -5

# Step 2: Start infrastructure (Ollama, ES, Kibana, OpenWebUI) – these auto-restart
# Step 3: Start code services with force-recreate (always fresh from latest build)
echo ""
echo "🔄 Starting all services..."
docker compose up -d
# Force-recreate code services to ensure latest image is used
docker compose up -d --force-recreate agent_api runner

# Step 4: Wait for services
echo ""
echo "⏳ Waiting for services..."
sleep 8

# Step 5: Health checks
echo ""
echo "🏥 Health Checks:"
echo "─────────────────"

# Ollama
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    MODELS=$(curl -sf http://localhost:11434/api/tags | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('models',[])))" 2>/dev/null)
    echo "  ✅ Ollama        – $MODELS Modelle"
else
    echo "  ❌ Ollama        – nicht erreichbar"
fi

# Agent API
if curl -sf http://localhost:11436/v1/models >/dev/null 2>&1; then
    echo "  ✅ Agent API     – OK"
else
    echo "  ❌ Agent API     – nicht erreichbar"
fi

# PyRunner
if curl -sf http://localhost:9000/health >/dev/null 2>&1; then
    echo "  ✅ PyRunner      – OK"
else
    echo "  ❌ PyRunner      – nicht erreichbar"
fi

# Elasticsearch
if curl -sf http://localhost:9200/_cat/health >/dev/null 2>&1; then
    DOCS=$(curl -sf "http://localhost:9200/_cat/indices/rag_files_v1?h=docs.count" 2>/dev/null | tr -d ' ')
    echo "  ✅ Elasticsearch – $DOCS Dokumente"
else
    echo "  ❌ Elasticsearch – nicht erreichbar"
fi

# OpenWebUI
if curl -sf http://localhost:8086/ >/dev/null 2>&1; then
    echo "  ✅ OpenWebUI     – OK"
else
    echo "  ❌ OpenWebUI     – nicht erreichbar"
fi

echo ""
echo "════════════════════════════════════════"
echo "✅ System bereit!"
echo ""
echo "  🌐 OpenWebUI:  http://localhost:8086"
echo "  🔌 Agent API:  http://localhost:11436"
echo "  🤖 Ollama:     http://localhost:11434"
echo "════════════════════════════════════════"
