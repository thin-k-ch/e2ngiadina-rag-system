#!/bin/bash
set -e

echo "=== WINDSURF RAG System - Reset ==="
echo "⚠️  This will stop all services and clear data!"
echo ""
read -p "Are you sure you want to continue? (yes/no): " -r
if [[ ! $REPLY =~ ^yes$ ]]; then
    echo "Reset cancelled."
    exit 1
fi

echo ""
echo "Stopping all services..."
docker compose down

echo ""
echo "Cleaning up Docker resources..."
docker system prune -f
docker volume prune -f

echo ""
echo "Clearing application data (optional)..."
read -p "Clear ChromaDB data? (yes/no): " -r
if [[ $REPLY =~ ^yes$ ]]; then
    echo "Clearing ChromaDB..."
    rm -rf ./volumes/chroma/*
fi

read -p "Clear logs? (yes/no): " -r
if [[ $REPLY =~ ^yes$ ]]; then
    echo "Clearing logs..."
    rm -rf ./volumes/logs/*
fi

read -p "Clear Ollama models? (yes/no): " -r
if [[ $REPLY =~ ^yes$ ]]; then
    echo "Clearing Ollama models..."
    rm -rf ./volumes/ollama/*
fi

read -p "Clear manifest? (yes/no): " -r
if [[ $REPLY =~ ^yes$ ]]; then
    echo "Clearing manifest..."
    rm -rf ./volumes/manifest/*
fi

echo ""
echo "Rebuilding containers..."
docker compose build --no-cache

echo ""
echo "=== Reset Complete ==="
echo "Run './scripts/start_all.sh' to restart the system"
