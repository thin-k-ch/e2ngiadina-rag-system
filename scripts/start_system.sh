#!/bin/bash
# WINDSURF SYSTEM START - GPU OPTIMIZED
# Sichert konsistenten Start mit GPU Support

set -e

echo "🚀 WINDSURF SYSTEM START - GPU OPTIMIZED"
echo "========================================="

# 1) Snap Ollama deaktivieren (falls vorhanden)
echo "1️⃣  Deactivating Snap Ollama (if exists)..."
sudo systemctl stop snap.ollama.listener.service 2>/dev/null || true
sudo systemctl disable snap.ollama.listener.service 2>/dev/null || true
echo "✅ Snap Ollama handled"

# 2) Docker GPU Runtime sicherstellen
echo "2️⃣  Ensuring Docker GPU Runtime..."
sudo systemctl restart docker 2>/dev/null || true
echo "✅ Docker restarted"

# 3) Docker Compose starten
echo "3️⃣  Starting Docker Compose services..."
cd /media/felix/RAG/AGENTIC
docker compose up -d
echo "✅ Docker Compose started"

# 4) Warten auf Services
echo "4️⃣  Waiting for services to be ready..."
sleep 15

# 5) GPU Support verifizieren
echo "5️⃣  Verifying GPU Support..."
if docker exec e2ngiadina-ollama nvidia-smi >/dev/null 2>&1; then
    echo "✅ GPU Support verified"
else
    echo "❌ GPU Support failed - restarting Ollama..."
    docker compose restart ollama
    sleep 10
fi

# 6) Smoke Test ausführen
echo "6️⃣  Running Smoke Test..."
if bash testing/scripts/smoke_small.sh >/dev/null 2>&1; then
    echo "✅ Smoke Test passed"
else
    echo "❌ Smoke Test failed - check logs"
fi

echo ""
echo "🎉 WINDSURF SYSTEM READY!"
echo "=========================="
echo "📊 Elasticsearch: http://localhost:9200"
echo "🤖 Agent API:    http://localhost:11436"
echo "🌐 OpenWebUI:    http://localhost:8086"
echo "🧠 Ollama:       http://localhost:11434"
echo "📖 Runbook:      README_RUNBOOK.md"
