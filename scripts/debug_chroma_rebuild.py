#!/usr/bin/env python3
"""
Debug Chroma rebuild - Enhanced logging
"""

import os
import sys
import json
import traceback
from pathlib import Path

def main():
    print("=== CHROMA REBUILD DEBUG ===")
    
    # 1. Check Python environment
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # 2. Check venv
    venv_python = "/media/felix/RAG/AGENTIC/venv/bin/python"
    if os.path.exists(venv_python):
        print(f"✅ Venv exists: {venv_python}")
    else:
        print(f"❌ Venv missing: {venv_python}")
    
    # 3. Check directories
    chroma_dir = Path("/media/felix/RAG/1/volumes/chroma")
    print(f"Chroma dir: {chroma_dir}")
    print(f"Chroma dir exists: {chroma_dir.exists()}")
    print(f"Chroma dir writable: {os.access(chroma_dir, os.W_OK)}")
    
    # 4. Check imports
    try:
        import requests
        print(f"✅ requests: {requests.__version__}")
    except ImportError as e:
        print(f"❌ requests: {e}")
    
    try:
        import chromadb
        print(f"✅ chromadb: {chromadb.__version__}")
    except ImportError as e:
        print(f"❌ chromadb: {e}")
    
    try:
        import tiktoken
        print(f"✅ tiktoken: {tiktoken.__version__}")
    except ImportError as e:
        print(f"❌ tiktoken: {e}")
    
    # 5. Test ES connection
    try:
        import requests
        response = requests.get("http://localhost:9200/_cluster/health", timeout=5)
        print(f"✅ ES: {response.json()['status']}")
    except Exception as e:
        print(f"❌ ES: {e}")
    
    # 6. Test Chroma setup
    try:
        import chromadb
        client = chromadb.PersistentClient(
            path="/media/felix/RAG/1/volumes/chroma",
            settings=chromadb.config.Settings(allow_reset=True)
        )
        print("✅ Chroma client created")
        
        # Try to create test collection
        test_coll = client.create_collection("debug_test")
        client.delete_collection("debug_test")
        print("✅ Chroma operations work")
        
    except Exception as e:
        print(f"❌ Chroma: {e}")
        traceback.print_exc()
    
    # 7. Write test log
    try:
        log_file = chroma_dir / "debug.log"
        with open(log_file, 'w') as f:
            f.write(f"Debug test at {os.times()}\n")
            f.write(f"Python: {sys.executable}\n")
            f.write(f"Working dir: {os.getcwd()}\n")
        print(f"✅ Log written to: {log_file}")
    except Exception as e:
        print(f"❌ Log write: {e}")

if __name__ == "__main__":
    main()
