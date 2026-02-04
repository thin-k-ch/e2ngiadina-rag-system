# E2NGIADINA RAG System

Ein vollständiges RAG (Retrieval-Augmented Generation) System mit GPU-Unterstützung und erweiterter Datei-Indexierung.

## 🚀 Quick Start

**Einzelner Befehl zum Starten des gesamten Systems:**
```bash
cd /media/felix/RAG/AGENTIC
./scripts/start_all.sh
```

## 📋 Systemübersicht

### Features
- **GPU-beschleunigtes LLM**: llama4:latest auf NVIDIA GB10
- **Multi-Format Indexierung**: PDF, DOCX, XLSX, MSG, PPTX, TXT, HTML, CSV, JSON, XML, YAML, ZIP
- **Vector Database**: ChromaDB mit 50,000+ Dokumenten
- **Web Interface**: OpenWebUI auf Port 8086
- **REST API**: OpenAI-kompatibel auf Port 11436

### Services
| Service | Port | Beschreibung |
|---------|------|-------------|
| OpenWebUI | 8086 | Web Interface |
| Agent API | 11436 | RAG API |
| Ollama | 11434 | LLM Inference (GPU) |
| Runner | 9000 | Code Execution |

## 🛠️ Installation & Setup

### 1. Voraussetzungen
- Docker & Docker Compose
- NVIDIA GPU mit CUDA 12.1+
- 7,985+ Dateien in `/media/felix/RAG/1`

### 2. System starten
```bash
cd /media/felix/RAG/AGENTIC
chmod +x scripts/*.sh
./scripts/start_all.sh
```

### 3. Daten indexieren
```bash
docker compose run --rm indexer
```

### 4. System testen
```bash
# Health Check
curl -s http://localhost:11436/health

# API Test
curl -s -X POST http://localhost:11436/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"agentic-rag","messages":[{"role":"user","content":"test query"}]}'

# Web Interface
# Öffne http://localhost:8086 im Browser
```

## 📁 Projektstruktur

```
/media/felix/RAG/AGENTIC/
├── README.md                    # Diese Datei
├── WINDSURF_SETUP.md           # Detaillierte Dokumentation
├── docker-compose.yml          # Service-Konfiguration
├── scripts/                    # Automatisierungsskripte
│   ├── start_all.sh           # Komplett-Start
│   ├── status_check.sh        # System-Status
│   ├── reset_system.sh        # Komplett-Reset
│   ├── ingest_pdfs.sh         # PDF-Indexierung (legacy)
│   └── smoke_test.sh          # API-Tests
├── agent_api/                  # RAG API Service
├── indexer/                    # Multi-Format Indexierung
├── runner/                     # Code Execution Service
└── volumes/                    # Persistente Daten
    ├── chroma/                 # Vector Database
    ├── ollama/                 # LLM Models
    ├── manifest/               # Index Manifest
    └── logs/                   # System Logs
```

## 🔧 Management Scripts

### System starten
```bash
./scripts/start_all.sh
```
- Setzt Berechtigungen
- Startet alle Services
- Lädt llama4:latest Modell
- Führt Health-Checks durch

### System-Status prüfen
```bash
./scripts/status_check.sh
```
- Zeigt Service-Status
- GPU-Auslastung
- ChromaDB Dokumentenzahl
- Speichernutzung

### System zurücksetzen
```bash
./scripts/status_check.sh
```
- Stoppt alle Services
- Bereinigt Docker-Ressourcen
- Optionales Löschen von Daten

## 📊 Monitoring & Debugging

### Logs ansehen
```bash
# Alle Services
docker compose logs --tail 100

# Spezifische Services
docker logs --tail 100 agentic-api
docker logs --tail 100 agentic-ollama

# Anwendungslogs
tail -f ./volumes/logs/indexer.log
tail -f ./volumes/logs/agent_api.log
```

### GPU-Status
```bash
nvidia-smi
```

### ChromaDB Statistik
```bash
docker compose run --rm indexer python -c "
import chromadb
c=chromadb.PersistentClient('/chroma')
col=c.get_or_create_collection('documents')
print('Dokumente:', col.count())
"
```

## 🔄 Datenverarbeitung

### Unterstützte Formate
- **PDF**: PyMuPDF
- **Office**: DOCX, XLSX, PPTX
- **Email**: MSG (Outlook)
- **Web**: HTML, XML
- **Daten**: CSV, JSON, YAML
- **Text**: TXT, MD
- **Archive**: ZIP (recursive, depth=2)

### Verarbeitungs-Pipeline
1. Datei-Discovery in `/media/felix/RAG/1`
2. Content-Extraktion via `text_loaders.py`
3. Text-Chunking (1200 chars, 180 overlap)
4. Embedding-Generierung (all-MiniLM-L6-v2)
5. Speicherung in ChromaDB
6. Manifest-Tracking für Updates

## 🌐 API-Nutzung

### OpenAI-kompatibles Endpunkt
```bash
curl -X POST http://localhost:11436/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agentic-rag",
    "messages": [{"role": "user", "content": "Ihre Frage hier"}]
  }'
```

### Web Interface
- **URL**: http://localhost:8086
- **Modell**: agentic-rag
- **Funktionen**: Chat mit RAG, Dokumenten-Suche

## 🛠️ Fehlersuche

### Häufige Probleme
1. **Port-Konflikte**: Ports 8086, 9000, 11434, 11436 frei?
2. **GPU nicht erkannt**: `nvidia-smi` prüfen
3. **Speicherprobleme**: RAM während Indexierung überwachen
4. **Berechtigungen**: `chmod +x scripts/*.sh` ausführen

### Kompletter Reset
```bash
./scripts/reset_system.sh
./scripts/start_all.sh
```

## 📈 Performance

- **GPU**: NVIDIA GB10, CUDA 12.1
- **Modell**: llama4:latest (GPU-beschleunigt)
- **Verarbeitung**: 6 Worker, Batch-Size 256
- **Speicher**: 24h Model-Keep-Alive
- **Dokumente**: 50,000+ in ChromaDB

## 🔐 Sicherheit

- CORS für Entwicklung konfiguriert
- Kein Internet-Zugriff für Runner-Container
- Daten-Verzeichnis read-only gemountet
- API-Keys für lokalen Setup leer konfiguriert

## 📞 Support

1. **Logs prüfen**: `./scripts/status_check.sh`
2. **GPU prüfen**: `nvidia-smi`
3. **Services neustarten**: `docker compose restart`
4. **Kompletter Reset**: `./scripts/reset_system.sh`

---

**WINDSURF RAG System** - Production-ready mit GPU-Beschleunigung und erweiterter Datei-Indexierung.
