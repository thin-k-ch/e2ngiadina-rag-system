# Morgen-Runbook - MCP ChatGPT Verbindung

## 🚀 Quick Start (Zero Brain)

### 1. Chromium starten
```bash
/snap/chromium/current/usr/lib/chromium-browser/chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=/home/felix/automation-profile \
  --no-sandbox \
  https://chatgpt.com/c/69889256-088c-8331-8b7c-1c90523e4478
```

### 2. RAG System starten
```bash
cd /media/felix/RAG/AGENTIC
docker compose up -d
```

### 3. MVP Status prüfen
```bash
./scripts/smoke_stream.sh
```

### 4. MCP-Server starten
```bash
node /home/felix/chatgpt-mcp-v2.js
```

## 📋 Persistenz-Status

### ✅ Was ist bereits stabil:
- **Chromium Profile**: `/home/felix/automation-profile/` (gespeichert)
- **ChatGPT Session**: Authentifiziert und persistent
- **MCP-Server**: `/home/felix/chatgpt-mcp-v2.js` (toStringSafe fix)
- **Windsurf Config**: `.windsurf/mcp_config.json` zeigt auf v2

### 🔍 Tab-Identifikation (automatisch)
Der MCP-Server findet den ChatGPT Tab automatisch über:
- `type === "page"`
- `URL enthält chatgpt.com`
- `Title enthält "ChatGPT"`

### 🛡️ Fehler-Prävention
- **Port 9222 muss frei sein** (sonst hängt alles)
- **Kein frisches Profil** (sonst Logout)
- **Gleiche JS-Datei verwenden** (chatgpt-mcp-v2.js)

## 🧪 MVP Test
Nach Start sollte `./scripts/smoke_stream.sh` zeigen:
- ✅ Health check passed
- ✅ Debug endpoint correctly disabled (404)
- ✅ Streaming Contract mit sofortigem TRACE

## 📞 Wenn Probleme
1. **Chromium nicht erreichbar**: `ps aux | grep chromium`
2. **Port 9222 belegt**: `sudo netstat -tulpn | grep 9222`
3. **MCP nicht verbunden**: Windsurf MCP Status prüfen

---
**Status**: MVP v0.1.0 - Production Ready 🎉
