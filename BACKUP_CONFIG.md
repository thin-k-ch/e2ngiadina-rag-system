# WINDSURF RAG System - Configuration Backup

## 📋 Critical Configuration Files

### Core Services
- **docker-compose.yml** - Complete service configuration
- **agent_api/app/agent.py** - Memory-enabled RAG agent
- **agent_api/app/main.py** - Memory-enabled API endpoints
- **agent_api/app/state.py** - Persistent state management

### Indexing Scripts
- **indexer/app/index_pdfs.py** - PDF indexing (unlimited)
- **indexer/app/index_docx.py** - DOCX indexing (unlimited)
- **indexer/app/text_loaders.py** - Multi-format loaders

### Management Scripts
- **START.sh** - One-click system start
- **STOP.sh** - One-click system stop
- **scripts/status_check.sh** - System monitoring

## 🔧 Key Configuration Values

### Environment Variables
```yaml
agent_api:
  LLM_MODEL: llama4:latest
  CONTEXT_MAX_TOKENS: 12000
  CONTEXT_SUMMARY_TOKENS: 1200
  CONTEXT_RECENT_TOKENS: 7000
  NOTES_MAX_TOKENS: 600
  SUMMARY_UPDATE_TRIGGER_TOKENS: 9000
  STATE_PATH: /state

indexer:
  COLLECTION: documents
  COLLECTION_DOCX: documents_docx
  MIN_TEXT_CHARS: 200
  EMBED_MODEL: all-MiniLM-L6-v2
```

### File Limits
- **PDFs:** Unlimited (all found files)
- **DOCXs:** Unlimited (all found files)
- **Memory:** Per conversation, no global limits

## 📊 Current Data Status

### Collections
- **documents (PDFs):** 12,077 chunks
- **documents_docx (DOCXs):** 2,067 chunks
- **Total:** 14,144 chunks

### Storage Paths
- **ChromaDB:** `/volumes/chroma/`
- **Memory State:** `/volumes/state/`
- **Manifest:** `/volumes/manifest/`
- **Logs:** `/volumes/logs/`

## 🚀 Startup Sequence

### 1. Start Services
```bash
cd /media/felix/RAG/AGENTIC
./START.sh
```

### 2. Verify Status
```bash
./scripts/status_check.sh
```

### 3. Test RAG
```bash
curl -s -X POST http://localhost:11436/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Conversation-Id: test" \
  -d '{"model":"agentic-rag","messages":[{"role":"user","content":"Test query"}]}'
```

## 📁 Important File Locations

### Must Backup
```
/media/felix/RAG/AGENTIC/
├── docker-compose.yml
├── agent_api/app/
│   ├── agent.py
│   ├── main.py
│   └── state.py
├── indexer/app/
│   ├── index_pdfs.py
│   ├── index_docx.py
│   └── text_loaders.py
├── START.sh
├── STOP.sh
└── scripts/status_check.sh
```

### Data (Optional Backup)
```
/media/felix/RAG/AGENTIC/volumes/
├── chroma/          # Vector database
├── state/           # Memory state
├── manifest/        # Index manifest
└── logs/            # System logs
```

## 🔍 Verification Checklist

### Services Running
- [ ] agentic-ollama (Port 11434)
- [ ] agentic-api (Port 11436)
- [ ] agentic-runner (Port 9000)
- [ ] agentic-openwebui (Port 8086)

### Functionality Tests
- [ ] RAG with PDFs works
- [ ] RAG with DOCXs works
- [ ] Memory persistence works
- [ ] Web interface accessible
- [ ] API endpoints responding

### Data Verification
- [ ] PDF collection count: 12,077+
- [ ] DOCX collection count: 2,067+
- [ ] Memory state files created
- [ ] Logs writing correctly

## 🚨 Recovery Procedures

### Service Recovery
```bash
# Restart all services
./STOP.sh && ./START.sh

# Rebuild specific service
docker compose build agent_api
docker compose up -d agent_api
```

### Data Recovery
```bash
# Check ChromaDB
docker compose run --rm indexer python -c "
import chromadb
c=chromadb.PersistentClient('/chroma')
print('PDFs:', c.get_or_create_collection('documents').count())
print('DOCXs:', c.get_or_create_collection('documents_docx').count())
"

# Re-index if needed
docker compose run --rm indexer python -m app.index_pdfs
docker compose run --rm indexer python -m app.index_docx
```

---

**Configuration Backup Complete - System Ready for Production**
