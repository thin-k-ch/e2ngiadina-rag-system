#!/usr/bin/env python3
"""
Sync TFK18 documents from Elasticsearch to ChromaDB.
Runs INSIDE the Docker container for chromadb version compatibility.

Usage (from host):
  docker cp scripts/sync_tfk18_es_to_chroma.py e2ngiadina-api:/tmp/
  docker exec -d e2ngiadina-api python3 /tmp/sync_tfk18_es_to_chroma.py

Monitor:
  docker exec e2ngiadina-api tail -f /tmp/sync_tfk18.log
"""

import hashlib
import json
import os
import sys
import time
import logging

import requests
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# --- Configuration ---
ES_URL = "http://elasticsearch:9200"
ES_INDEX = "rag_tfk18_v1"
CHROMA_PATH = "/chroma/tfk18"
CHROMA_COLLECTION = "tfk18_documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
OVERLAP = 180
ES_BATCH = 500
EMBED_BATCH = 64
CHROMA_BATCH = 5000

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/sync_tfk18.log"),
    ]
)
log = logging.getLogger(__name__)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    text = (text or "").strip()
    if not text:
        return []
    out = []
    step = max(1, size - overlap)
    i = 0
    n = len(text)
    while i < n:
        out.append(text[i:i + size])
        i += step
    return out


def stable_id(path: str, chunk_index: int):
    h = hashlib.sha1(path.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"tfk18:{h}:{chunk_index}"


def es_scroll(scroll_id=None, size=500):
    if scroll_id:
        r = requests.post(f"{ES_URL}/_search/scroll",
                          json={"scroll": "5m", "scroll_id": scroll_id}, timeout=60)
    else:
        r = requests.post(f"{ES_URL}/{ES_INDEX}/_search?scroll=5m",
                          json={"size": size, "query": {"match_all": {}}, "_source": True},
                          timeout=60)
    return r.json()


def batch_encode(embedder, texts):
    return embedder.encode(texts, convert_to_tensor=False, show_progress_bar=False, batch_size=EMBED_BATCH).tolist()


def main():
    log.info("=" * 60)
    log.info("TFK18 ES → ChromaDB Sync")
    log.info(f"ES: {ES_URL}/{ES_INDEX}")
    log.info(f"Chroma: {CHROMA_PATH}/{CHROMA_COLLECTION}")
    log.info("=" * 60)

    # Check ES
    r = requests.get(f"{ES_URL}/{ES_INDEX}/_count", timeout=10)
    es_count = r.json().get("count", 0)
    log.info(f"ES docs: {es_count:,}")

    # Init Chroma
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    # Get or create (don't delete existing!)
    col = client.get_or_create_collection(CHROMA_COLLECTION)
    existing = col.count()
    log.info(f"Chroma collection '{CHROMA_COLLECTION}': {existing} existing")

    # Load embedder (CPU)
    log.info(f"Loading embedder: {EMBED_MODEL} (CPU)")
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

    # Phase 1: Scroll ALL docs from ES first (fast, no embedding yet)
    log.info("Phase 1: Collecting all documents from ES...")
    scroll_id = None
    all_chunks = []  # list of (id, text, metadata)
    processed_docs = 0
    skipped = 0
    t0 = time.time()

    while True:
        result = es_scroll(scroll_id=scroll_id, size=ES_BATCH)
        scroll_id = result.get("_scroll_id", scroll_id)

        hits = result.get("hits", {}).get("hits", [])
        if not hits:
            break

        for hit in hits:
            src = hit.get("_source", {})
            content = src.get("content", "")
            path_obj = src.get("path", {})
            path = path_obj.get("virtual", "") if isinstance(path_obj, dict) else str(path_obj)
            file_obj = src.get("file", {})
            filename = file_obj.get("filename", "") if isinstance(file_obj, dict) else ""
            extension = file_obj.get("extension", "") if isinstance(file_obj, dict) else ""

            if not content or len(content.strip()) < 200:
                skipped += 1
                continue

            chunks = chunk_text(content)
            if not chunks:
                skipped += 1
                continue

            for ci, ch in enumerate(chunks):
                doc_id = stable_id(path or hit["_id"], ci)
                all_chunks.append((doc_id, ch, {
                    "path": (path or "")[:500],
                    "filename": filename,
                    "extension": extension,
                    "chunk_index": ci,
                    "source_type": "es_chunk",
                }))

            processed_docs += 1

        log.info(f"  Scrolled: {processed_docs:,} docs, {len(all_chunks):,} chunks, {skipped} skipped")

    # Close scroll
    if scroll_id:
        try:
            requests.delete(f"{ES_URL}/_search/scroll",
                            json={"scroll_id": scroll_id}, timeout=10)
        except Exception:
            pass

    log.info(f"Phase 1 done: {processed_docs:,} docs → {len(all_chunks):,} chunks ({skipped} skipped)")

    # Phase 2: Embed and upsert in batches
    log.info(f"Phase 2: Encoding + upserting {len(all_chunks):,} chunks...")
    indexed_chunks = 0

    for i in range(0, len(all_chunks), CHROMA_BATCH):
        batch = all_chunks[i:i + CHROMA_BATCH]
        batch_ids = [c[0] for c in batch]
        batch_docs = [c[1] for c in batch]
        batch_metas = [c[2] for c in batch]

        log.info(f"  Batch {i//CHROMA_BATCH + 1}: encoding {len(batch)} chunks...")
        batch_embs = batch_encode(embedder, batch_docs)
        log.info(f"  Upserting to Chroma...")
        col.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_embs,
        )
        indexed_chunks += len(batch)
        elapsed = time.time() - t0
        pct = indexed_chunks / len(all_chunks) * 100
        log.info(f"  Progress: {indexed_chunks:,}/{len(all_chunks):,} ({pct:.0f}%), {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    final_count = col.count()
    log.info("=" * 60)
    log.info(f"✅ Sync Complete:")
    log.info(f"   ES docs processed: {processed_docs:,}")
    log.info(f"   Chunks indexed: {indexed_chunks:,}")
    log.info(f"   Skipped: {skipped}")
    log.info(f"   Chroma total: {final_count:,}")
    log.info(f"   Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
