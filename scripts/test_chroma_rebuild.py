#!/usr/bin/env python3
"""
Test Chroma rebuild with small sample
"""

import os
import sys
import json
import requests
from pathlib import Path

# Add venv to path
venv_python = "/media/felix/RAG/AGENTIC/venv/bin/python"
if os.path.exists(venv_python):
    os.execv(venv_python, [venv_python] + sys.argv)

# Configuration
ES_URL = "http://localhost:9200"
ES_INDEX = "rag_files_v1"
CHROMA_PERSIST_DIR = "/media/felix/RAG/1/volumes/chroma"

def test_es_connection():
    """Test Elasticsearch connection"""
    try:
        response = requests.get(f"{ES_URL}/_cluster/health")
        response.raise_for_status()
        print(f"✅ ES Connection: {response.json()['status']}")
        return True
    except Exception as e:
        print(f"❌ ES Connection failed: {e}")
        return False

def test_sample_data():
    """Test with small sample"""
    try:
        response = requests.post(
            f"{ES_URL}/{ES_INDEX}/_search",
            json={
                "size": 3,
                "_source": ["content", "file", "path"],
                "query": {"match_all": {}}
            }
        )
        response.raise_for_status()
        data = response.json()
        
        hits = data.get("hits", {}).get("hits", [])
        print(f"✅ Sample data: {len(hits)} documents")
        
        for i, hit in enumerate(hits[:2]):
            source = hit["_source"]
            content_len = len(source.get("content", ""))
            filename = source.get("file", {}).get("filename", "unknown")
            print(f"   Doc {i+1}: {filename} ({content_len} chars)")
        
        return len(hits) > 0
    except Exception as e:
        print(f"❌ Sample data failed: {e}")
        return False

def test_chroma_setup():
    """Test Chroma setup"""
    try:
        import chromadb
        from chromadb.config import Settings
        
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(allow_reset=True)
        )
        
        # Test collection creation
        test_collection = client.create_collection("test_collection")
        client.delete_collection("test_collection")
        
        print("✅ Chroma setup: OK")
        return True
    except Exception as e:
        print(f"❌ Chroma setup failed: {e}")
        return False

def main():
    """Main test function"""
    print("=== CHROMA REBUILD TEST ===")
    
    tests = [
        ("ES Connection", test_es_connection),
        ("Sample Data", test_sample_data),
        ("Chroma Setup", test_chroma_setup)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n{name}:")
        if test_func():
            passed += 1
    
    print(f"\n=== RESULTS ===")
    print(f"Passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✅ All tests passed - ready for rebuild!")
        return 0
    else:
        print("❌ Some tests failed - fix issues first")
        return 1

if __name__ == "__main__":
    sys.exit(main())
