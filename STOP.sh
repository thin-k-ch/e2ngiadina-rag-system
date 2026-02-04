#!/bin/bash
set -e

# E2NGIADINA RAG System - One-Click Shutdown
echo "🛑 Stopping E2NGIADINA RAG System..."

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Please run from /media/felix/RAG/AGENTIC"
    exit 1
fi

# Stop all services
echo "🔄 Stopping Docker services..."
docker compose down

# Also stop external postgres if running
if docker ps --format "table {{.Names}}" | grep -q "docker-postgres-1"; then
    echo "🔄 Stopping external PostgreSQL..."
    docker stop docker-postgres-1
fi

echo ""
echo "✅ All WINDSURF services stopped!"
echo ""
echo "📊 System Status:"
docker ps --format "table {{.Names}}\t{{.Status}}"

echo ""
echo "🚀 To restart: ./START.sh"
echo "📊 To check status: ./scripts/status_check.sh"
echo "🔄 To reset: ./scripts/reset_system.sh"
