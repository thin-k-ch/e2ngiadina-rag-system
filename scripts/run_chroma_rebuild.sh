#!/bin/bash
# WINDSURF CHROMA REBUILD RUNNER
# Stellt sicher dass das Skript mit venv Python läuft

set -e

echo "🔥 WINDSURF CHROMA REBUILD"
echo "=========================="

# Venv Python sicherstellen
VENV_PYTHON="/media/felix/RAG/AGENTIC/venv/bin/python"
REBUILD_SCRIPT="/media/felix/RAG/AGENTIC/scripts/chroma_rebuild_from_es.py"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Venv Python nicht gefunden: $VENV_PYTHON"
    exit 1
fi

if [ ! -f "$REBUILD_SCRIPT" ]; then
    echo "❌ Rebuild Script nicht gefunden: $REBUILD_SCRIPT"
    exit 1
fi

echo "✅ Venv Python: $VENV_PYTHON"
echo "✅ Rebuild Script: $REBUILD_SCRIPT"

# Environment setzen
export ES_URL="http://localhost:9200"
export ES_INDEX="rag_files_v1"
export CHROMA_PERSIST_DIR="/media/felix/RAG/1/volumes/chroma"
export COLLECTION_NAME="rag_files_v1_chunks"
export BATCH_DOCS="200"
export BATCH_UPSERT="1000"
export CHUNK_SIZE_CHARS="1200"
export OVERLAP_CHARS="200"

echo "📊 Configuration:"
echo "   ES_URL: $ES_URL"
echo "   ES_INDEX: $ES_INDEX"
echo "   CHROMA_PERSIST_DIR: $CHROMA_PERSIST_DIR"
echo "   COLLECTION_NAME: $COLLECTION_NAME"
echo "   BATCH_DOCS: $BATCH_DOCS"
echo "   BATCH_UPSERT: $BATCH_UPSERT"
echo "   CHUNK_SIZE_CHARS: $CHUNK_SIZE_CHARS"
echo "   OVERLAP_CHARS: $OVERLAP_CHARS"

echo ""
echo "🚀 Starting rebuild with venv Python..."
echo "   Log file: $CHROMA_PERSIST_DIR/rebuild.log"
echo ""

# Mit venv Python ausführen
exec "$VENV_PYTHON" "$REBUILD_SCRIPT"
