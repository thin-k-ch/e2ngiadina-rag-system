#!/usr/bin/env python3
"""
Index TFK18 data into self-contained volumes directory.
Creates: TFK18/volumes/esdata, chroma, manifest
"""
import os
import sys
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

# Paths relative to TFK18 root
TFK18_ROOT = "/media/felix/RAG/TFK18"
TFK18_VOLUMES = os.path.join(TFK18_ROOT, "volumes")
TFK18_DATA = TFK18_ROOT  # Data files directly in TFK18/

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = "rag_files_tfk18_v1"  # Separate index name
CHROMA_PATH = os.path.join(TFK18_VOLUMES, "chroma")
MANIFEST_PATH = os.path.join(TFK18_VOLUMES, "manifest", "manifest.sqlite3")

# Ensure directories exist
os.makedirs(os.path.join(TFK18_VOLUMES, "esdata"), exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(os.path.join(TFK18_VOLUMES, "manifest"), exist_ok=True)
os.makedirs(os.path.join(TFK18_VOLUMES, "logs"), exist_ok=True)

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
ES_BATCH = int(os.getenv("ES_BATCH", "100"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "128"))
CHROMA_BATCH = int(os.getenv("CHROMA_BATCH", "256"))

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
    return f"tfk18:{h}:{chunk_index}"

def get_file_hash(filepath):
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except:
        return None

def scan_files(root_dir):
    """Recursively scan all files in directory"""
    files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            files.append(filepath)
    return files

def extract_text_basic(filepath):
    """Basic text extraction for various file types"""
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if ext in ['.txt', '.md', '.py', '.js', '.html', '.xml', '.json', '.csv']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext in ['.pdf']:
            # Use pdfplumber if available
            try:
                import pdfplumber
                with pdfplumber.open(filepath) as pdf:
                    return "\n".join(page.extract_text() or "" for page in pdf.pages)
            except:
                return None
        elif ext in ['.docx']:
            try:
                from docx import Document
                doc = Document(filepath)
                return "\n".join(p.text for p in doc.paragraphs)
            except:
                return None
        elif ext in ['.pptx']:
            try:
                from pptx import Presentation
                prs = Presentation(filepath)
                texts = []
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            texts.append(shape.text)
                return "\n".join(texts)
            except:
                return None
        elif ext in ['.eml', '.msg']:
            # Use existing text_loaders
            sys.path.insert(0, '/media/felix/RAG/AGENTIC/indexer/app')
            from text_loaders import read_eml_with_attachments, read_msg
            if ext == '.eml':
                return read_eml_with_attachments(filepath)
            else:
                return read_msg(filepath)
        else:
            return None
    except Exception as e:
        print(f"⚠️ Error extracting {filepath}: {e}")
        return None

def es_create_index():
    """Create ES index if not exists"""
    url = f"{ES_URL}/{ES_INDEX}"
    try:
        r = requests.head(url, timeout=10)
        if r.status_code == 200:
            print(f"✅ ES index {ES_INDEX} already exists")
            return
    except:
        pass
    
    # Create index with mapping
    mapping = {
        "mappings": {
            "properties": {
                "content": {"type": "text", "analyzer": "standard"},
                "file.filename": {"type": "keyword"},
                "file.extension": {"type": "keyword"},
                "path.virtual": {"type": "keyword"},
                "file.hash": {"type": "keyword"},
                "indexed_at": {"type": "date"}
            }
        }
    }
    
    try:
        r = requests.put(url, json=mapping, timeout=30)
        if r.status_code in [200, 201]:
            print(f"✅ Created ES index: {ES_INDEX}")
        else:
            print(f"⚠️ ES index creation: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"❌ Failed to create ES index: {e}")

def es_index_batch(docs):
    """Bulk index to ES"""
    if not docs:
        return 0
    
    url = f"{ES_URL}/{ES_INDEX}/_bulk"
    lines = []
    for doc in docs:
        lines.append(json.dumps({"index": {"_index": ES_INDEX}}))
        lines.append(json.dumps(doc))
    
    body = "\n".join(lines) + "\n"
    
    try:
        headers = {"Content-Type": "application/json"}
        r = requests.post(url, data=body, headers=headers, timeout=60)
        if r.status_code == 200:
            result = r.json()
            errors = result.get("errors", False)
            if errors:
                print(f"⚠️ ES bulk had errors: {result.get('items', [])[:2]}")
            return len(docs)
        else:
            print(f"⚠️ ES bulk failed: {r.status_code}")
            return 0
    except Exception as e:
        print(f"⚠️ ES bulk error: {e}")
        return 0

def batch_encode(embedder, chunks: List[str]) -> List[List[float]]:
    if not chunks:
        return []
    embeddings = embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()

def main():
    print("=" * 60)
    print("TFK18 → Self-Contained Index Builder")
    print("=" * 60)
    print(f"Data: {TFK18_DATA}")
    print(f"Volumes: {TFK18_VOLUMES}")
    print(f"ES Index: {ES_INDEX}")
    print(f"Chroma: {CHROMA_PATH}")
    print("=" * 60)
    
    # Setup imports
    sys.path.insert(0, '/media/felix/RAG/AGENTIC/venv/lib/python3.12/site-packages')
    import chromadb
    from sentence_transformers import SentenceTransformer
    import json
    
    # Scan files
    print("\n🔍 Scanning files...")
    files = scan_files(TFK18_DATA)
    # Exclude volumes directory
    files = [f for f in files if not f.startswith(TFK18_VOLUMES)]
    print(f"📁 Found {len(files):,} files")
    
    if not files:
        print("❌ No files found!")
        return
    
    # Setup ES
    es_create_index()
    
    # Setup Chroma
    client = chromadb.PersistentClient(CHROMA_PATH)
    try:
        client.delete_collection("documents")
        print("🗑️  Deleted old Chroma collection")
    except:
        pass
    
    col = client.create_collection("documents")
    print(f"✅ Created Chroma collection: documents")
    
    # Load embedder
    print(f"🔄 Loading embedder: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    
    # Process files
    print(f"\n🚀 Processing {len(files):,} files...")
    es_docs = []
    chroma_batches = []
    processed = 0
    es_indexed = 0
    chroma_chunks = 0
    
    for filepath in files:
        rel_path = os.path.relpath(filepath, TFK18_DATA)
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        
        # Extract text
        content = extract_text_basic(filepath)
        if not content:
            continue
        
        file_hash = get_file_hash(filepath)
        
        # ES document
        es_doc = {
            "content": content,
            "file.filename": filename,
            "file.extension": ext,
            "path.virtual": rel_path,
            "file.hash": file_hash,
            "indexed_at": None  # Will be set by ES
        }
        es_docs.append(es_doc)
        
        # Chroma chunks
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            doc_id = stable_id(rel_path, i)
            chroma_batches.append({
                "id": doc_id,
                "text": chunk,
                "meta": {
                    "path": rel_path,
                    "filename": filename,
                    "chunk_index": i,
                    "extension": ext
                }
            })
        
        processed += 1
        
        # Batch ES index
        if len(es_docs) >= ES_BATCH:
            indexed = es_index_batch(es_docs)
            es_indexed += indexed
            es_docs = []
            print(f"  ES: {es_indexed:,} docs indexed")
        
        # Batch Chroma (accumulate and process in larger batches)
        if len(chroma_batches) >= EMBED_BATCH * 2:
            # Encode batch
            texts = [b["text"] for b in chroma_batches[:EMBED_BATCH]]
            embeddings = batch_encode(embedder, texts)
            
            # Add to Chroma
            batch = chroma_batches[:EMBED_BATCH]
            try:
                col.add(
                    ids=[b["id"] for b in batch],
                    documents=[b["text"] for b in batch],
                    metadatas=[b["meta"] for b in batch],
                    embeddings=embeddings
                )
                chroma_chunks += len(batch)
            except Exception as e:
                print(f"⚠️ Chroma error: {e}")
            
            chroma_batches = chroma_batches[EMBED_BATCH:]
            print(f"  Chroma: {chroma_chunks:,} chunks indexed, {processed:,} files processed")
    
    # Final ES batch
    if es_docs:
        indexed = es_index_batch(es_docs)
        es_indexed += indexed
    
    # Final Chroma batches
    while chroma_batches:
        batch_size = min(EMBED_BATCH, len(chroma_batches))
        batch = chroma_batches[:batch_size]
        texts = [b["text"] for b in batch]
        embeddings = batch_encode(embedder, texts)
        
        try:
            col.add(
                ids=[b["id"] for b in batch],
                documents=[b["text"] for b in batch],
                metadatas=[b["meta"] for b in batch],
                embeddings=embeddings
            )
            chroma_chunks += len(batch)
        except Exception as e:
            print(f"⚠️ Chroma error: {e}")
        
        chroma_batches = chroma_batches[batch_size:]
    
    final_chroma_count = col.count()
    
    print("\n" + "=" * 60)
    print("✅ TFK18 Index Complete:")
    print(f"   Files processed: {processed:,}")
    print(f"   ES docs: {es_indexed:,}")
    print(f"   Chroma chunks: {chroma_chunks:,} (count: {final_chroma_count:,})")
    print(f"   ES Index: {ES_INDEX}")
    print(f"   Chroma Path: {CHROMA_PATH}")
    print("=" * 60)
    print("\nTo switch to TFK18:")
    print("1. docker-compose down")
    print("2. mv /media/felix/RAG/1 /media/felix/RAG/1_backup")
    print("3. mv /media/felix/RAG/TFK18 /media/felix/RAG/1")
    print("4. docker-compose up -d")
    print("\nTo switch back:")
    print("Reverse the above steps.")

if __name__ == "__main__":
    main()
