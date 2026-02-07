#!/bin/bash
# WINDSURF FSCRAWLER ONE-LINER WRAPPER
# Garantiert FSCRAWLER_HOME = volumes/fscrawler (verhindert drift)

set -e

# FSCRAWLER_HOME auf volumes forcieren
export FSCRAWLER_HOME="/media/felix/RAG/AGENTIC/volumes/fscrawler"

# In FSCrawler Verzeichnis wechseln
cd /media/felix/RAG/AGENTIC/tools/fscrawler

# Alle Argumente an FSCrawler durchreichen
echo "🚀 Starting FSCrawler with FSCRAWLER_HOME=$FSCRAWLER_HOME"
echo "Command: ./bin/fscrawler $*"
echo ""

./bin/fscrawler "$@"
