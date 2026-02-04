#!/bin/bash
set -e

# E2NGIADINA RAG System - One-Click Startup
echo "🚀 Starting E2NGIADINA RAG System..."

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Please run from /media/felix/RAG/AGENTIC"
    exit 1
fi

# Set execute permissions
echo "🔧 Setting permissions..."
chmod +x scripts/*.sh

# Start the system
echo "🔄 Starting all services..."
./scripts/start_all.sh

echo ""
echo "✅ WINDSURF RAG System started successfully!"
echo ""
echo "📱 Access Points:"
echo "   Web Interface: http://localhost:8086"
echo "   API Health:    http://localhost:11436/health"
echo "   Ollama API:    http://localhost:11434"
echo ""
echo "📊 Next Steps:"
echo "   1. Check status: ./scripts/status_check.sh"
echo "   2. Index data:  docker compose run --rm indexer"
echo "   3. Test system: Open http://localhost:8086"
echo ""
echo "📖 Documentation: cat README.md"
