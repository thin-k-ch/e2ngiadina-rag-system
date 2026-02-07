# WINDSURF RAG SYSTEM - RUNBOOK

## 🚀 STARTREIHENFOLGE

### 1. Elasticsearch (Port 9200)
```bash
# Health Check
curl -s http://localhost:9200/_cluster/health | jq '.status'
# Expected: "green" or "yellow"
```

### 2. Ollama (Port 11434)
```bash
# Health Check
curl -s http://localhost:11434/api/tags | jq -r '.models[0].name'
# Expected: Model name (e.g., "llama4:latest")
```

### 3. Agent (Port 11436)
```bash
# Health Check
curl -s http://localhost:11436/health
# Expected: HTTP 200
```

### 4. OpenWebUI (Port 8086)
```bash
# Health Check
curl -s -o /dev/null -w "%{http_code}" http://localhost:8086/
# Expected: 200
```

### 5. FSCrawler (Optional - für neue Indexierung)
```bash
# Goldenes Startkommando
cd /media/felix/RAG/AGENTIC/tools/fscrawler
FSCRAWLER_HOME="/media/felix/RAG/AGENTIC/volumes/fscrawler" \
./bin/fscrawler rag1 --loop 1
```

## 📡 PORTS + ERWARTETE HTTP CODES

| Service | Port | Endpoint | Expected Code |
|---------|------|----------|---------------|
| Elasticsearch | 9200 | `/_cluster/health` | 200 |
| Ollama | 11434 | `/api/tags` | 200 |
| Agent | 11436 | `/health` | 200 |
| OpenWebUI | 8086 | `/` | 200 |
| Agent OpenAPI | 11436 | `/openapi.json` | 200 |

## ✅ DEFINITION OF DONE (v0)

### Small Suite grün = System produktiv

**Checks:**
1. ✅ Elasticsearch health + count > 50k docs
2. ✅ Agent health + 1 successful chat completion
3. ✅ `/open` GET funktioniert mit bekannter Datei
4. ✅ FSCrawler Config persistent in `volumes/fscrawler`

**Erfolgreicher Test:**
```bash
./scripts/smoke_small.sh
# Expected: All checks PASS
```

## 🔧 WICHTIGE KONFIGURATION

### FSCrawler Persistent Config
```bash
# Config Location
/media/felix/RAG/AGENTIC/volumes/fscrawler/rag1/_settings.yaml

# Goldenes Startkommando (immer verwenden!)
FSCRAWLER_HOME="/media/felix/RAG/AGENTIC/volumes/fscrawler" \
./bin/fscrawler rag1 --loop 1
```

### Agent API
- OpenAI-kompatibel: `/v1/chat/completions`
- File Proxy: `/open?path=/media/felix/...`
- OpenAPI: `/openapi.json`

## 🚨 TROUBLESHOOTING

### FSCrawler Config Drift
```bash
# Fix: Config syncen
cp /media/felix/RAG/AGENTIC/volumes/fscrawler/rag1/_settings.yaml ~/.fscrawler/rag1/_settings.yaml
```

### Agent findet keine Treffer
```bash
# Check: Agent Config vs ES Index
curl -s http://localhost:11436/openapi.json | jq '.paths."/open".get.parameters'
```

## 📊 SYSTEM STATUS

**Letzter Test:** 2026-02-07
**Dokumente:** 54,844
**Index:** rag_files_v1
**Status:** ✅ Produktiv
