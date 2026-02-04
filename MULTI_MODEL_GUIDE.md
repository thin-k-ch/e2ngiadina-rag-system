# 🎯 Multi-Modell Guide für OpenWebUI

## 📋 Aktuelle Konfiguration

Ihr OpenWebUI ist jetzt konfiguriert für die gleichzeitige Nutzung von:
- **RAG Agent:** `agentic-rag` (mit Memory und 37.624 Dokumenten)
- **Direkte Ollama Modelle:** `llama4:latest` und `qwen2.5:14b`

## 🌐 Zugriff

**Web Interface:** http://localhost:8086

## 🔄 Modell-Wechsel

### Option 1: Über das Web Interface
1. Öffnen Sie http://localhost:8086
2. Klicken Sie auf das Modell-Dropdown oben links
3. Wählen Sie zwischen:
   - `agentic-rag` → Für RAG mit Memory und Dokumenten
   - `llama4:latest` → Für direkte LLM-Anfragen
   - `qwen2.5:14b` → Für schnellere Antworten

### Option 2: Automatische Erkennung
OpenWebUI sollte automatisch alle verfügbaren Modelle anzeigen.

## 🎯 Anwendungsfälle

### 📚 RAG Agent (`agentic-rag`)
**Verwenden für:**
- 📄 Dokumenten-basierte Fragen
- 🧠 Persistente Conversations
- 🔍 Kontextbezogene Antworten mit Zitaten
- 📊 Business Intelligence

**Beispiel:**
```
"Welche Offerten hat Rhomberg Bahntechnik für den Gotthard Basistunnel abgegeben?"
```

### 🚀 Direkte LLMs (`llama4:latest`, `qwen2.5:14b`)
**Verwenden für:**
- 💬 Allgemeine Konversationen
- 🔧 Code-Generierung
- 📝 Text-Erstellung
- 🤖 Kreative Aufgaben

**Beispiel:**
```
"Schreibe ein Python-Skript für Datenanalyse"
```

## ⚙️ Konfigurations-Details

### Environment Variablen
```yaml
openwebui:
  environment:
    - OPENAI_API_BASE_URL=http://agent_api:11436/v1  # RAG Agent
    - OLLAMA_BASE_URL=http://ollama:11434            # Direkte Ollama Modelle
    - ENABLE_OLLAMA_API=true                         # Ollama aktivieren
    - SHOW_OLLAMA_MODELS=true                        # Modelle anzeigen
    - DEFAULT_MODELS=agentic-rag                     # Standard-Modell
```

### Verfügbare Modelle
```bash
# Ollama Modelle
docker compose exec ollama ollama list
# → llama4:latest (67 GB, GPU)
# → qwen2.5:14b (9.0 GB, GPU)

# RAG Agent
curl http://localhost:11436/v1/models
# → agentic-rag (mit Memory + 37.624 chunks)
```

## 🔧 Fehlersuche

### Falls Modelle nicht angezeigt werden:
1. **OpenWebUI neustarten:**
   ```bash
   docker compose restart openwebui
   ```

2. **Ollama Status prüfen:**
   ```bash
   docker compose exec ollama ollama ps
   ```

3. **Agent API Status prüfen:**
   ```bash
   curl http://localhost:11436/health
   ```

4. **Browser Cache leeren:**
   - Strg+F5 (Windows/Linux)
   - Cmd+Shift+R (Mac)

### Falls RAG nicht funktioniert:
1. **ChromaDB prüfen:**
   ```bash
   docker compose exec agent_api python -c "
   import chromadb
   c=chromadb.PersistentClient('/chroma')
   print('PDFs:', c.get_or_create_collection('documents').count())
   print('DOCXs:', c.get_or_create_collection('documents_docx').count())
   "
   ```

2. **Memory System prüfen:**
   ```bash
   ls -la /media/felix/RAG/AGENTIC/volumes/state/
   ```

## 📊 Performance-Vergleich

| Modell | Geschwindigkeit | Qualität | Spezial | Anwendungsfall |
|--------|---------------|----------|----------|---------------|
| `agentic-rag` | 3-5s | ⭐⭐⭐⭐⭐ | 📚 Dokumente | Business Fragen |
| `llama4:latest` | 2-4s | ⭐⭐⭐⭐⭐ | 🧠 Allgemein | Komplexe Aufgaben |
| `qwen2.5:14b` | 1-2s | ⭐⭐⭐⭐ | ⚡ Schnell | Einfache Fragen |

## 🎯 Best Practices

### 1. Richtige Modellwahl
- **Dokumenten-Fragen** → Immer `agentic-rag`
- **Allgemeine Konversation** → `llama4:latest`
- **Schnelle Antworten** → `qwen2.5:14b`

### 2. Conversation Management
- **RAG Conversations** werden automatisch gespeichert
- **Direkte LLM Conversations** sind session-basiert
- **Wechsel zwischen Modellen** ist jederzeit möglich

### 3. Memory Nutzung
- **RAG Agent** merkt sich frühere Gespräche
- **Conversation ID** für Kontext-Persistenz
- **Private Notes** für Agent-Working-Memory

## 🚀 Zukunftsoptionen

### Weitere Modelle hinzufügen:
```bash
# Neues Modell pullen
docker compose exec ollama ollama pull model_name

# OpenWebUI neustarten
docker compose restart openwebui
```

### Custom Modelle:
- **Fine-tuned Modelle** für spezifische Domänen
- **Spezialisierte Modelle** für bestimmte Aufgaben
- **Multi-Modal Modelle** für Bilder + Text

---

**🎯 Ihr E2NGIADINA RAG System unterstützt jetzt flexible Multi-Modell-Nutzung!**

*Web Interface: http://localhost:8086*
