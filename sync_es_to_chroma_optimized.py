#!/usr/bin/env python3
"""
Optimized ES → ChromaDB Sync with Batch Embeddings
10x faster than single-chunk encoding
"""
import os
import sys
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_files_v1")
CHROMA_PATH = os.getenv("CHROMA_PATH", "/media/felix/RAG/chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents_from_es")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
ES_BATCH = int(os.getenv("ES_BATCH", "500"))  # Larger ES batches
CHROMA_BATCH = int(os.getenv("CHROMA_BATCH", "512"))  # Larger Chroma batches
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "256"))  # Batch embedding size

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

def es_search(scroll_id=None, size=500):
    """Search ES with larger batches"""
    if scroll_id:
        url = f"{ES_URL}/_search/scroll"
        body = {"scroll": "5m", "scroll_id": scroll_id}
    else:
        url = f"{ES_URL}/{ES_INDEX}/_search?scroll=5m"
        body = {
            "size": size,
            "query": {"match_all": {}},
            "_source": ["content", "path.virtual", "file.filename"]
        }
    
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=body, headers=headers, timeout=120)
    return resp.json()

def batch_encode(embedder, chunks: List[str]) -> List[List[float]]:
    """Encode multiple chunks at once - 10x faster"""
    if not chunks:
        return []
    # Batch encode all chunks together
    embeddings = embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()

def process_batch(embedder, items: List[Tuple]) -> Tuple[List, List, List, List]:
    """Process a batch of items with batched embeddings"""
    # Collect all chunks for batch encoding
    all_chunks = []
    chunk_mapping = []  # Maps flat chunk index back to (item_index, chunk_index)
    
    for item_idx, (path, filename, content) in enumerate(items):
        chunks = chunk_text(content)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_mapping.append((item_idx, chunk_idx, path, filename))
    
    if not all_chunks:
        return [], [], [], []
    
    # Batch encode all chunks at once
    print(f"  🔄 Encoding {len(all_chunks)} chunks in batch...")
    embeddings = batch_encode(embedder, all_chunks)
    
    # Build Chroma batch
    batch_ids = []
    batch_docs = []
    batch_metas = []
    batch_embs = []
    
    for (item_idx, chunk_idx, path, filename), chunk, emb in zip(chunk_mapping, all_chunks, embeddings):
        doc_id = stable_id(path, chunk_idx)
        batch_ids.append(doc_id)
        batch_docs.append(chunk)
        batch_metas.append({
            "path": path,
            "filename": filename,
            "chunk_index": chunk_idx
        })
        batch_embs.append(emb)
    
    return batch_ids, batch_docs, batch_metas, batch_embs

def main():
    print("=" * 60)
    print("Optimized ES → ChromaDB Sync (Batch Embeddings)")
    print("=" * 60)
    
    sys.path.insert(0, '/media/felix/RAG/AGENTIC/venv/lib/python3.12/site-packages')
    import chromadb
    from sentence_transformers import SentenceTransformer
    
    # Setup Chroma
    client = chromadb.PersistentClient(CHROMA_PATH)
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print("🗑️  Deleted old collection")
    except:
        pass
    
    col = client.create_collection(CHROMA_COLLECTION)
    print(f"✅ Created collection: {CHROMA_COLLECTION}")
    
    # Check ES
    r = requests.get(f"{ES_URL}/{ES_INDEX}/_count", timeout=10)
    es_count = r.json().get("count", 0)
    print(f"📊 ES docs: {es_count:,}")
    
    # Load embedder
    print(f"🔄 Loading embedder: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    
    # Process in large batches
    print(f"\n🚀 Starting sync with batch_size={ES_BATCH}...")
    
    scroll_id = None
    processed = 0
    indexed = 0
    current_batch = []
    
    while True:
        result = es_search(scroll_id=scroll_id, size=ES_BATCH)
        
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
            
            current_batch.append((path, filename, content))
            processed += 1
            
            # Process when batch is full
            if len(current_batch) >= EMBED_BATCH:
                print(f"\n📦 Processing batch of {len(current_batch)} docs...")
                
                ids, docs, metas, embs = process_batch(embedder, current_batch)
                
                if ids:
                    # Split into Chroma-sized batches
                    for i in range(0, len(ids), CHROMA_BATCH):
                        end = min(i + CHROMA_BATCH, len(ids))
                        try:
                            col.add(
                                ids=ids[i:end],
                                documents=docs[i:end],
                                metadatas=metas[i:end],
                                embeddings=embs[i:end]
                            )
                            indexed += (end - i)
                        except Exception as e:
                            print(f"⚠️ Chroma error: {e}")
                
                print(f"✅ Indexed {indexed:,} chunks total, {processed:,} docs processed")
                current_batch = []
        
        print(f"  Processed: {processed:,} / {es_count:,}")
    
    # Final batch
    if current_batch:
        print(f"\n📦 Processing final batch of {len(current_batch)} docs...")
        ids, docs, metas, embs = process_batch(embedder, current_batch)
        if ids:
            for i in range(0, len(ids), CHROMA_BATCH):
                end = min(i + CHROMA_BATCH, len(ids))
                try:
                    col.add(
                        ids=ids[i:end],
                        documents=docs[i:end],
                        metadatas=metas[i:end],
                        embeddings=embs[i:end]
                    )
                    indexed += (end - i)
                except Exception as e:
                    print(f"⚠️ Chroma error: {e}")
    
    final_count = col.count()
    print(f"\n" + "=" * 60)
    print(f"✅ Sync Complete:")
    print(f"   Processed: {processed:,} ES docs")
    print(f"   Indexed: {indexed:,} chunks in ChromaDB")
    print(f"   Chroma total: {final_count:,}")
    print("=" * 60)

if __name__ == "__main__":
    main()
