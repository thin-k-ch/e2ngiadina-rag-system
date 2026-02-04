#!/bin/bash
set -e
cd "$(dirname "$0")/.."
docker compose run --rm indexer
echo "Indexer completed. Chroma persisted in ./volumes/chroma"
