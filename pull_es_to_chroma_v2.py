#!/usr/bin/env python3
"""
Pull ES documents to ChromaDB using HTTP requests
Compatible with ES 8.x
"""
import os
import sys
import hashlib
import json
import requests
from pathlib import Path

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_files_v1")
CHROMA_PATH = os.getenv("CHROMA_PATH", "/media/felix/RAG/chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents_from_es")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
BATCH = int(os.getenv("BATCH", "32"))

def chunk_text(text, size=1200, overlap=180):
    text = (text or "").strip()
    if not text:
        return []
    out = []
    i = 0
    n = len(text)
    step = max(1, size - overlap)
    while i < n:
        out.append(text[i:i+size])
        i += step
    return out

def stable_id(path, chunk_index):
    h = hashlib.sha1(path.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"es:{h}:{chunk_index}"

def es_search(scroll_id=None, size=100):
    """Search ES with scroll"""
    if scroll_id:
        url = f"{ES_URL}/_search/scroll"
        body = {"scroll": "2m", "scroll_id": scroll_id}
    else:
        url = f"{ES_URL}/{ES_INDEX}/_search?scroll=2m"
        body = {
            "size": size,
            "query": {"match_all": {}},
            "_source": ["content", "path.virtual", "file.filename"]
        }
    
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=body, headers=headers, timeout=60)
    return resp.json()

def main():
    print("=" * 60)
    print("Pull ES → ChromaDB")
    print("=" * 60)
    print(f"ES: {ES_URL}/{ES_INDEX}")
    print(f"Chroma: {CHROMA_PATH}/{CHROMA_COLLECTION}")
    
    # Check ES
    try:
        r = requests.get(ES_URL, timeout=5)
        print(f"✅ ES connected: {r.status_code}")
    except Exception as e:
        print(f"❌ ES error: {e}")
        return
    
    # Get ES count
    r = requests.get(f"{ES_URL}/{ES_INDEX}/_count", timeout=10)
    es_count = r.json().get("count", 0)
    print(f"📊 ES docs: {es_count:,}")
    
    # Setup Chroma
    sys.path.insert(0, '/media/felix/RAG/AGENTIC/venv/lib/python3.12/site-packages')
    import chromadb
    from sentence_transformers import SentenceTransformer
    
    client = chromadb.PersistentClient(CHROMA_PATH)
    try:
        col = client.get_collection(CHROMA_COLLECTION)
        print(f"✅ Collection exists: {CHROMA_COLLECTION}")
    except Exception:
        col = client.create_collection(CHROMA_COLLECTION)
        print(f"✅ Created collection: {CHROMA_COLLECTION}")
    
    chroma_count = col.count()
    print(f"📊 Chroma docs: {chroma_count:,}")
    
    if chroma_count >= es_count:
        print(f"\n✅ Chroma already up to date ({chroma_count} >= {es_count})")
        return
    
    # Get already indexed IDs
    existing_ids = set()
    try:
        result = col.get(limit=100000)
        existing_ids = set(result.get("ids", []))
        print(f"📋 Existing Chroma IDs: {len(existing_ids):,}")
    except Exception as e:
        print(f"⚠️ Could not get existing IDs: {e}")
    
    # Pull from ES
    print(f"\n🚀 Pulling documents...")
    embedder = SentenceTransformer(EMBED_MODEL)
    
    scroll_id = None
    processed = 0
    indexed = 0
    batch_ids = []
    batch_docs = []
    batch_metas = []
    batch_embeddings = []
    
    while True:
        result = es_search(scroll_id=scroll_id)
        
        if scroll_id is None:
            scroll_id = result.get("_scroll_id")
        
        hits = result.get("hits", {}).get("hits", [])
        if not hits:
            break
        
        for hit in hits:
            source = hit.get("_source", {})
            path = source.get("path", {}).get("virtual", "")
            content = source.get("content", "")
            filename = source.get("file", {}).get("filename", "")
            
            if not content or not path:
                continue
            
            # Chunk
            chunks = chunk_text(content)
            for i, chunk in enumerate(chunks):
                doc_id = stable_id(path, i)
                
                if doc_id in existing_ids:
                    continue
                
                batch_ids.append(doc_id)
                batch_docs.append(chunk)
                batch_metas.append({
                    "path": path,
                    "filename": filename,
                    "chunk_index": i
                })
                batch_embeddings.append(embedder.encode(chunk).tolist())
                
                if len(batch_ids) >= BATCH:
                    try:
                        col.add(
                            ids=batch_ids,
                            documents=batch_docs,
                            metadatas=batch_metas,
                            embeddings=batch_embeddings
                        )
                        indexed += len(batch_ids)
                    except Exception as e:
                        print(f"\n⚠️ Batch error: {e}")
                    
                    batch_ids = []
                    batch_docs = []
                    batch_metas = []
                    batch_embeddings = []
                    
                    if indexed % 500 == 0:
                        print(f"   Indexed: {indexed:,} / {es_count:,}")
            
            processed += 1
    
    # Final batch
    if batch_ids:
        try:
            col.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=batch_embeddings
            )
            indexed += len(batch_ids)
        except Exception as e:
            print(f"\n⚠️ Final batch error: {e}")
    
    final_count = col.count()
    print(f"\n" + "=" * 60)
    print(f"✅ ChromaDB Update Complete:")
    print(f"   Processed: {processed:,} ES docs")
    print(f"   Indexed: {indexed:,} chunks")
    print(f"   Chroma total: {final_count:,}")
    print("=" * 60)

if __name__ == "__main__":
    main()
