# WINDSURF RAG SYSTEM - RUNBOOK

## 🚀 STARTREIHENFOLGE

### Automatischer Start (Systemd)
```bash
# Beim System-Boot automatisch
sudo systemctl enable windsurf-rag.service
# Startet das gesamte System mit GPU Support
```

### Manueller Start (GPU Optimiert)
```bash
# GPU-optimierter Start
./scripts/start_system.sh

# Oder klassisch
docker compose up -d
```

### Service-Reihenfolge
1. **Docker** (mit GPU Runtime)
2. **Snap Ollama** wird deaktiviert
3. **Elasticsearch** (Port 9200)
4. **Ollama** (Port 11434) mit GPU
5. **Agent API** (Port 11436)
6. **OpenWebUI** (Port 8086)

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
