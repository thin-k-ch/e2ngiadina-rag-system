#!/bin/bash
set -e

echo "=== WINDSURF RAG System - Complete Startup ==="

# Step 1: Set execute permissions
echo "Setting execute permissions..."
chmod +x /media/felix/RAG/AGENTIC/scripts/*.sh

# Step 2: Start all services
echo "Starting all services..."
cd /media/felix/RAG/AGENTIC
docker compose up -d --build

# Step 3: Wait for services to be ready
echo "Waiting for services to start..."
sleep 15

# Step 4: Pull LLM model
echo "Pulling llama4:latest model..."
curl -s http://localhost:11434/api/pull -d '{"name":"llama4:latest"}' || echo "Model might already exist or pull failed"

# Step 5: Check service health
echo "Checking service health..."
echo "=== Ollama ==="
curl -s http://localhost:11434/api/tags || echo "Ollama not responding"

echo "=== Agent API ==="
curl -s http://localhost:11436/health || echo "Agent API not responding"

echo "=== OpenWebUI ==="
curl -s -I http://localhost:8086 | head -1 || echo "OpenWebUI not responding"

echo ""
echo "=== System Status ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "=== Access Information ==="
echo "OpenWebUI: http://localhost:8086"
echo "API Health: http://localhost:11436/health"
echo "Ollama API: http://localhost:11434"
echo ""
echo "=== Next Steps ==="
echo "1. Run extended ingestion: docker compose run --rm indexer"
echo "2. Test the system in OpenWebUI at http://localhost:8086"
echo "3. Check logs: tail -f ./volumes/logs/indexer.log"
echo ""
echo "=== Startup Complete ==="
