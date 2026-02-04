#!/bin/bash
set -e

echo "=== WINDSURF RAG System - Status Check ==="

echo ""
echo "=== Docker Services Status ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "=== Service Health Checks ==="

# Ollama
echo "Ollama (LLM):"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama API responding"
    curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | head -3
else
    echo "❌ Ollama API not responding"
fi

# Agent API
echo ""
echo "Agent API:"
if curl -s http://localhost:11436/health > /dev/null 2>&1; then
    echo "✅ Agent API responding"
    curl -s http://localhost:11436/health
else
    echo "❌ Agent API not responding"
fi

# OpenWebUI
echo ""
echo "OpenWebUI:"
if curl -s -I http://localhost:8086 | grep -q "200 OK"; then
    echo "✅ OpenWebUI responding"
    echo "🌐 http://localhost:8086"
else
    echo "❌ OpenWebUI not responding"
fi

echo ""
echo "=== GPU Status ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
else
    echo "❌ nvidia-smi not available"
fi

echo ""
echo "=== ChromaDB Document Count ==="
docker compose run --rm indexer python -c "
import chromadb
try:
    c=chromadb.PersistentClient('/chroma')
    col=c.get_or_create_collection('documents')
    count = col.count()
    print(f'📊 Documents in ChromaDB: {count}')
except Exception as e:
    print(f'❌ ChromaDB error: {e}')
" 2>/dev/null || echo "❌ Could not check ChromaDB count"

echo ""
echo "=== Recent Logs ==="
echo "Indexer (last 5 lines):"
tail -n 5 ./volumes/logs/indexer.log 2>/dev/null || echo "No indexer logs found"

echo ""
echo "Agent API (last 5 lines):"
tail -n 5 ./volumes/logs/agent_api.log 2>/dev/null || echo "No agent API logs found"

echo ""
echo "=== Storage Usage ==="
echo "ChromaDB:"
du -sh ./volumes/chroma 2>/dev/null || echo "Not found"

echo "Ollama Models:"
du -sh ./volumes/ollama 2>/dev/null || echo "Not found"

echo "Logs:"
du -sh ./volumes/logs 2>/dev/null || echo "Not found"

echo ""
echo "=== Quick Test Commands ==="
echo "# Test API:"
echo "curl -s -X POST http://localhost:11436/v1/chat/completions -H \"Content-Type: application/json\" -d '{\"model\":\"agentic-rag\",\"messages\":[{\"role\":\"user\",\"content\":\"test query\"}]}'"
echo ""
echo "# Run ingestion:"
echo "docker compose run --rm indexer"
echo ""
echo "# View logs:"
echo "tail -f ./volumes/logs/indexer.log"

echo ""
echo "=== Status Check Complete ==="
