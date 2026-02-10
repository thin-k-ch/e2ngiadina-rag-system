#!/usr/bin/env python3
import requests
import json

# Test Query
query_text = "rechnung"
print(f"🔍 TESTING HYBRID SEARCH FOR: '{query_text}'")
print("=" * 50)

# Test 1: Direct Elasticsearch
print("\n1️⃣ ELASTICSEARCH DIRECT:")
try:
    es_response = requests.post(
        "http://localhost:9200/rag_chunks_v1/_search",
        headers={"Content-Type": "application/json"},
        json={
            "query": {"match": {"text": query_text}},
            "size": 3
        }
    )
    es_hits = es_response.json()["hits"]["hits"]
    print(f"   ✅ ES Hits: {len(es_hits)}")
    for i, hit in enumerate(es_hits[:2]):
        source = hit["_source"]
        print(f"   📄 {i+1}: {source.get('document_type', 'unknown')} - {source.get('original_path', 'no path')[:80]}...")
except Exception as e:
    print(f"   ❌ ES Error: {e}")

# Test 2: Direct Chroma
print("\n2️⃣ CHROMA DIRECT:")
try:
    chroma_response = requests.post(
        "http://localhost:6000/api/v1/collections/rag_chunks/query",
        headers={"Content-Type": "application/json"},
        json={
            "query_texts": [query_text],
            "n_results": 3
        }
    )
    chroma_data = chroma_response.json()
    chroma_hits = chroma_data.get("ids", [[]])[0]
    print(f"   ✅ Chroma Hits: {len(chroma_hits)}")
    for i, hit_id in enumerate(chroma_hits[:2]):
        print(f"   📄 {i+1}: {hit_id}")
except Exception as e:
    print(f"   ❌ Chroma Error: {e}")

# Test 3: Agent Hybrid
print("\n3️⃣ AGENT HYBRID:")
try:
    agent_response = requests.post(
        "http://localhost:11436/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "agentic-rag",
            "messages": [{"role": "user", "content": f"Finde Informationen zu {query_text}"}],
            "temperature": 0.1
        }
    )
    result = agent_response.json()
    answer = result["choices"][0]["message"]["content"]
    print(f"   ✅ Agent Response Length: {len(answer)} chars")
    print(f"   📝 First 200 chars: {answer[:200]}...")
except Exception as e:
    print(f"   ❌ Agent Error: {e}")

print("\n" + "=" * 50)
print("🎯 HYBRID VERIFICATION COMPLETE!")
