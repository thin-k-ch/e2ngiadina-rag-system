# 🚀 E2NGIADINA RAG System

**Production-Ready RAG System with Memory, Multi-Format Indexing, and GPU Acceleration**

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)
[![LLM](https://img.shields.io/badge/LLM-llama4:latest-orange.svg)](https://ollama.com/library/llama4)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-success.svg)](https://developer.nvidia.com/cuda-zone)

## 🎯 Features

- **🧠 Persistent Memory System** - Conversation summaries and notes
- **📚 Multi-Format Indexing** - PDF + DOCX with separate collections
- **🚀 GPU Acceleration** - llama4:latest with NVIDIA CUDA
- **🌐 Web Interface** - OpenWebUI for easy interaction
- **📊 Production Ready** - Docker Compose, monitoring, logging
- **🔍 High-Quality RAG** - Citations, context-aware responses
- **📈 Scalable** - Unlimited document processing
- **🛠️ Easy Management** - One-click start/stop scripts

## 📊 Current Status

**Indexed Content:**
- **PDFs:** 35,557 chunks (5,000 documents)
- **DOCXs:** 2,067 chunks (1,130 documents)
- **Total:** 37,624 searchable chunks

**System Performance:**
- **Response Time:** <5 seconds
- **Memory:** Per conversation persistence
- **GPU:** NVIDIA CUDA 12.1
- **Model:** llama4:latest

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with CUDA drivers
- Git

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/windsurf-rag-system.git
cd windsurf-rag-system
```

### 2. One-Click Start
```bash
./START.sh
```

### 3. Access Interfaces
- **Web UI:** http://localhost:8086
- **API:** http://localhost:11436
- **Ollama:** http://localhost:11434

### 4. Test the System
```bash
curl -s -X POST http://localhost:11436/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Conversation-Id: test" \
  -d '{
    "model": "agentic-rag",
    "messages": [{"role": "user", "content": "What documents are indexed?"}]
  }'
```

## 📁 Project Structure

```
windsurf-rag-system/
├── 📄 README.md                 # This file
├── 🚀 START.sh                  # One-click start
├── 🛑 STOP.sh                   # One-click stop
├── 🐳 docker-compose.yml        # Service configuration
├── 📚 agent_api/                # RAG API with Memory
│   ├── app/
│   │   ├── agent.py            # Memory-enabled agent
│   │   ├── main.py             # FastAPI endpoints
│   │   └── state.py            # Persistent state
│   └── Dockerfile
├── 🔍 indexer/                  # Multi-format indexing
│   ├── app/
│   │   ├── index_pdfs.py       # PDF indexing
│   │   ├── index_docx.py       # DOCX indexing
│   │   └── text_loaders.py     # Format loaders
│   └── Dockerfile
├── 🏃 runner/                   # Python execution service
├── 📊 scripts/                  # Management utilities
│   ├── status_check.sh
│   └── reset_system.sh
└── 📦 volumes/                  # Persistent data
    ├── chroma/                 # Vector database
    ├── state/                  # Memory state
    └── logs/                   # System logs
```

## 🔧 Configuration

### Environment Variables
```yaml
agent_api:
  LLM_MODEL: llama4:latest
  CONTEXT_MAX_TOKENS: 12000
  MEMORY_ENABLED: true
  STATE_PATH: /state

indexer:
  COLLECTION_PDF: documents
  COLLECTION_DOCX: documents_docx
  EMBED_MODEL: all-MiniLM-L6-v2
  MIN_TEXT_CHARS: 200
```

### Customization
- **Models:** Change `LLM_MODEL` in docker-compose.yml
- **Collections:** Modify `COLLECTION_*` variables
- **Token Limits:** Adjust `CONTEXT_*` variables
- **Quality:** Change `MIN_TEXT_CHARS` filter

## 📚 Document Indexing

### Supported Formats
- **PDFs:** Full text extraction with metadata
- **DOCX:** Business documents, contracts, reports
- **Future:** Excel, CSV, MSG, PPTX (extendable)

### Index All Documents
```bash
# Index PDFs
docker compose run --rm indexer python -m app.index_pdfs

# Index DOCXs
docker compose run --rm indexer python -m app.index_docx
```

### Quality Filtering
- Minimum text length: 200 characters
- Duplicate detection via SHA1 hashing
- Metadata extraction from file paths
- Batch processing for efficiency

## 🧠 Memory System

### Features
- **Persistent Conversations:** Per conversation ID
- **Automatic Summaries:** Generated when needed
- **Private Notes:** Working memory for the agent
- **Context Management:** Sliding window with token limits

### Usage
```bash
# With conversation ID
curl -H "X-Conversation-Id: my_conversation" ...

# Automatic ID generation (hash-based)
curl # No header needed
```

## 🌐 API Documentation

### Endpoints
- **Health:** `GET /health`
- **Models:** `GET /v1/models`
- **Chat:** `POST /v1/chat/completions`

### OpenAI Compatible
```bash
curl -s -X POST http://localhost:11436/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agentic-rag",
    "messages": [{"role": "user", "content": "Your question"}]
  }'
```

## 📊 Monitoring

### System Status
```bash
./scripts/status_check.sh
```

### Logs
```bash
# Agent API logs
docker compose logs agent_api

# Indexer logs
docker compose logs indexer

# All services
docker compose logs
```

### Performance Metrics
- **Response Time:** Track API latency
- **Memory Usage:** Monitor conversation state
- **Indexing Speed:** Documents per second
- **GPU Utilization:** CUDA performance

## 🛠️ Development

### Adding New Document Types
1. Update `text_loaders.py` with new parser
2. Create `index_newformat.py` script
3. Add environment variables
4. Update documentation

### Extending Memory System
1. Modify `state.py` for new storage
2. Update `agent.py` for new features
3. Adjust token limits in docker-compose.yml

### Custom Models
1. Pull model in Ollama: `docker compose exec ollama ollama pull model_name`
2. Update `LLM_MODEL` environment variable
3. Restart services

## 🔒 Security Considerations

### Production Deployment
- **Network Security:** Use internal networks
- **Authentication:** Add API keys or OAuth
- **Data Privacy:** Encrypt sensitive volumes
- **Access Control:** Limit container permissions

### Data Protection
- **Local Processing:** No external API calls
- **Volume Isolation:** Separate data containers
- **Log Sanitization:** Remove sensitive information

## 📈 Scaling

### Horizontal Scaling
- **Load Balancer:** Multiple API instances
- **Database Cluster:** Distributed ChromaDB
- **Queue System:** Redis for task management

### Vertical Scaling
- **GPU Clusters:** Multiple NVIDIA cards
- **Memory:** Increase container limits
- **Storage:** SSD for ChromaDB

## 🐛 Troubleshooting

### Common Issues
1. **GPU not detected:** Check NVIDIA drivers
2. **Port conflicts:** Verify port availability
3. **Memory issues:** Check volume permissions
4. **Indexing errors:** Review logs in `/volumes/logs/`

### Reset System
```bash
./scripts/reset_system.sh
```

### Health Check
```bash
curl http://localhost:11436/health
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### Development Guidelines
- **Code Style:** Follow PEP 8
- **Testing:** Add unit tests for new features
- **Documentation:** Update README and inline docs
- **Docker:** Use multi-stage builds for efficiency

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** - LLM serving platform
- **ChromaDB** - Vector database
- **Sentence Transformers** - Embedding models
- **FastAPI** - High-performance API framework
- **OpenWebUI** - Web interface for LLMs

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/windsurf-rag-system/issues)
- **Discussions:** [GitHub Discussions](https://github.com/YOUR_USERNAME/windsurf-rag-system/discussions)
- **Documentation:** [Wiki](https://github.com/YOUR_USERNAME/windsurf-rag-system/wiki)

---

**🚀 WINDSURF RAG System - Production-Ready RAG with Memory**

*Built with ❤️ for advanced document intelligence*
