#!/usr/bin/env python3
"""
Sync ONLY new EML files from ES to ChromaDB (fast update)
Filters by extension = .eml or .msg
"""
import os
import sys
import hashlib
import requests

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_files_v1")
CHROMA_PATH = os.getenv("CHROMA_PATH", "/media/felix/RAG/chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents_from_es")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
BATCH = int(os.getenv("BATCH", "64"))

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

def es_search_eml(scroll_id=None, size=100):
    """Search ES for EML files only"""
    if scroll_id:
        url = f"{ES_URL}/_search/scroll"
        body = {"scroll": "2m", "scroll_id": scroll_id}
    else:
        url = f"{ES_URL}/{ES_INDEX}/_search?scroll=2m"
        body = {
            "size": size,
            "query": {
                "bool": {
                    "filter": [
                        {"terms": {"file.extension": [".eml", ".msg"]}}
                    ]
                }
            },
            "_source": ["content", "path.virtual", "file.filename", "file.extension"]
        }
    
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=body, headers=headers, timeout=60)
    return resp.json()

def main():
    print("=" * 60)
    print("Sync EML only: ES → ChromaDB")
    print("=" * 60)
    
    sys.path.insert(0, '/media/felix/RAG/AGENTIC/venv/lib/python3.12/site-packages')
    import chromadb
    from sentence_transformers import SentenceTransformer
    
    # Setup Chroma
    client = chromadb.PersistentClient(CHROMA_PATH)
    try:
        col = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        col = client.create_collection(CHROMA_COLLECTION)
    
    initial_count = col.count()
    print(f"📊 Initial Chroma docs: {initial_count:,}")
    
    # Get existing IDs
    existing_ids = set()
    try:
        result = col.get(limit=200000)
        existing_ids = set(result.get("ids", []))
        print(f"📋 Existing IDs: {len(existing_ids):,}")
    except Exception as e:
        print(f"⚠️ Could not get existing IDs: {e}")
    
    # Load embedder
    print(f"🔄 Loading embedder: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    
    # Pull EML from ES
    print(f"\n🚀 Pulling EML documents from ES...")
    
    scroll_id = None
    processed = 0
    indexed = 0
    batch_ids = []
    batch_docs = []
    batch_metas = []
    batch_embeddings = []
    
    while True:
        result = es_search_eml(scroll_id=scroll_id)
        
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
            ext = source.get("file", {}).get("extension", "")
            
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
                    "extension": ext,
                    "chunk_index": i,
                    "source": "eml_sync"
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
                    
                    if indexed % 1000 == 0:
                        print(f"   Indexed: {indexed:,}")
            
            processed += 1
            if processed % 100 == 0:
                print(f"   Processed: {processed:,} EML files")
    
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
    print(f"✅ EML Sync Complete:")
    print(f"   Processed: {processed:,} EML files")
    print(f"   Indexed: {indexed:,} new chunks")
    print(f"   Chroma total: {final_count:,} (was {initial_count:,})")
    print("=" * 60)

if __name__ == "__main__":
    main()
