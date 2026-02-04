# WINDSURF RAG System - Complete Setup Guide

## System Overview
- **Purpose**: RAG (Retrieval-Augmented Generation) System mit GPU-Unterstützung
- **LLM**: llama4:latest (GPU-beschleunigt)
- **Vector DB**: ChromaDB
- **Web Interface**: OpenWebUI
- **Data Source**: /media/felix/RAG/1 (PDF, DOCX, XLSX, MSG, PPTX, TXT, HTML, CSV, JSON, XML, YAML, ZIP)

## Services & Ports
| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| Ollama | agentic-ollama | 11434 | LLM Inference (GPU) |
| Agent API | agentic-api | 11436 | RAG API |
| Runner | agentic-runner | 9000 | Code Execution |
| OpenWebUI | agentic-openwebui | 8086 | Web Interface |
| Indexer | agentic-indexer | - | Data Ingestion |

## Quick Start (Single Command)
```bash
cd /media/felix/RAG/AGENTIC
chmod +x scripts/*.sh
docker compose up -d --build
```

## Detailed Setup Steps

### 1. Execute Permissions
```bash
chmod +x /media/felix/RAG/AGENTIC/scripts/*.sh
```

### 2. Start All Services
```bash
cd /media/felix/RAG/AGENTIC
docker compose up -d --build
```

### 3. Pull LLM Model (GPU)
```bash
curl -s http://localhost:11434/api/pull -d '{"name":"llama4:latest"}'
```

### 4. Run Extended Ingestion
```bash
docker compose run --rm indexer
```

### 5. Verify System Health
```bash
# Health Check
curl -s http://localhost:11436/health

# Test API
curl -s -X POST http://localhost:11436/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"agentic-rag","messages":[{"role":"user","content":"test query"}]}'

# Check WebUI
# Open http://localhost:8086 in browser
```

## Configuration Files

### docker-compose.yml
- GPU support enabled for Ollama
- Extended file format support in indexer
- ZIP_MAX_DEPTH=2 configured
- llama4:latest as default model

### Key Environment Variables
```yaml
OLLAMA_KEEP_ALIVE=24h
LLM_MODEL=llama4:latest
ZIP_MAX_DEPTH=2
OPENAI_API_BASE_URL=http://agent_api:11436/v1
```

## Data Processing

### Supported File Formats
- PDF (via PyMuPDF)
- DOCX (via python-docx)
- XLSX (via openpyxl/pandas)
- MSG (Outlook, via extract-msg)
- PPTX (via python-pptx)
- TXT, MD, CSV, JSON, XML, YAML
- HTML (via BeautifulSoup)
- ZIP (recursive, depth=2)

### Processing Pipeline
1. File discovery in /media/felix/RAG/1
2. Content extraction via text_loaders.py
3. Text chunking (1200 chars, 180 overlap)
4. Embedding generation (all-MiniLM-L6-v2)
5. Storage in ChromaDB
6. Manifest tracking for incremental updates

## Storage Locations
- **Vector DB**: ./volumes/chroma
- **Manifest**: ./volumes/manifest/manifest.sqlite3
- **Logs**: ./volumes/logs/
- **Ollama Models**: ./volumes/ollama

## Monitoring & Debugging

### Check Service Status
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### View Logs
```bash
# All services
docker compose logs --tail 100

# Specific service
docker logs --tail 100 agentic-api
docker logs --tail 100 agentic-ollama
docker logs --tail 100 agentic-openwebui

# Application logs
tail -n 100 ./volumes/logs/indexer.log
tail -n 100 ./volumes/logs/agent_api.log
```

### GPU Usage
```bash
nvidia-smi
```

### ChromaDB Count
```bash
docker compose run --rm indexer python -c "
import chromadb
c=chromadb.PersistentClient('/chroma')
col=c.get_or_create_collection('documents')
print('Documents in ChromaDB:', col.count())
"
```

## Troubleshooting

### Common Issues
1. **Port conflicts**: Ensure ports 8086, 9000, 11434, 11436 are free
2. **GPU not detected**: Check nvidia-smi and Docker GPU support
3. **Memory issues**: Monitor RAM usage during ingestion
4. **Permission errors**: Check execute permissions on scripts

### Reset Procedures
```bash
# Stop all services
docker compose down

# Clear ChromaDB (if needed)
rm -rf ./volumes/chroma/*

# Clear logs (if needed)
rm -rf ./volumes/logs/*

# Restart
docker compose up -d --build
```

## Performance Notes
- **GPU Acceleration**: llama4:latest runs on NVIDIA GB10 (CUDA 12.1)
- **Concurrent Processing**: 6 workers for ingestion
- **Batch Size**: 256 documents per ChromaDB upsert
- **Memory Management**: 24h model keep-alive in Ollama

## Security Notes
- CORS configured for development (*)
- No internet access for runner container
- Data directory mounted read-only
- API keys configured as empty for local setup

## Next Steps
1. Monitor ingestion progress
2. Test queries in OpenWebUI
3. Adjust chunking parameters if needed
4. Configure additional models if required
5. Set up monitoring/alerting for production

## Contact & Support
- Check logs first for troubleshooting
- GPU: NVIDIA GB10, CUDA 12.1
- Data: 7,985+ files in /media/felix/RAG/1
- Expected ChromaDB count: 50,000+ documents after full ingestion
