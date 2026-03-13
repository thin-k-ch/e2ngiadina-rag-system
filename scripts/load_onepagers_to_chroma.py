#!/usr/bin/env python3
"""
Load OnePager summaries into ChromaDB as whole documents (no chunking).
Must run INSIDE the Docker container (chromadb version compatibility).

Usage (from host):
  docker cp scripts/load_onepagers_to_chroma.py e2ngiadina-api:/tmp/
  docker cp runs/summaries/summaries.jsonl e2ngiadina-api:/tmp/
  docker exec e2ngiadina-api python3 /tmp/load_onepagers_to_chroma.py \
      --chroma-path /chroma/tfk18 \
      --collection tfk18_onepagers \
      /tmp/summaries.jsonl
"""

import argparse
import json
import sys
import time

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def main():
    ap = argparse.ArgumentParser(description="Load OnePager summaries into Chroma")
    ap.add_argument("jsonl_file", help="Path to summaries.jsonl")
    ap.add_argument("--chroma-path", default="/chroma/tfk18",
                    help="Chroma persistent storage path")
    ap.add_argument("--collection", default="tfk18_onepagers",
                    help="Chroma collection name")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Embedding model")
    ap.add_argument("--replace", action="store_true",
                    help="Delete existing collection before loading")
    ap.add_argument("--batch-size", type=int, default=50,
                    help="Upsert batch size")
    args = ap.parse_args()

    # Load summaries
    print(f"Loading summaries from {args.jsonl_file}...")
    docs = []
    with open(args.jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            if not r.get("onepager", "").strip():
                continue
            docs.append(r)
    print(f"  {len(docs)} OnePager loaded")

    # Init Chroma
    emb_fn = SentenceTransformerEmbeddingFunction(model_name=args.embed_model)
    client = chromadb.PersistentClient(
        path=args.chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )

    if args.replace:
        try:
            client.delete_collection(args.collection)
            print(f"  Deleted existing collection '{args.collection}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=emb_fn,
    )
    existing = collection.count()
    print(f"  Collection '{args.collection}': {existing} existing docs")

    # Prepare documents
    ids = []
    documents = []
    metadatas = []

    for r in docs:
        filename = r.get("filename", "unknown")
        doc_id = f"op:{filename}"

        # Build searchable text: OnePager + Erkenntnisse
        text_parts = [r["onepager"]]
        erkenntnisse = r.get("erkenntnisse", [])
        if erkenntnisse:
            text_parts.append("\n\nERKENNTNISSE FÜR DAS MANAGEMENT:")
            for e in erkenntnisse:
                text_parts.append(f"- {e}")
        doc_text = "\n".join(text_parts)

        meta = {
            "filename": filename,
            "path": str(r.get("path", ""))[:500],
            "extension": r.get("extension", ""),
            "doc_type": r.get("doc_type", ""),
            "n_erkenntnisse": len(erkenntnisse),
            "onepager_chars": len(r.get("onepager", "")),
            "source_type": "onepager",
        }

        ids.append(doc_id)
        documents.append(doc_text)
        metadatas.append(meta)

    if not documents:
        print("No valid OnePagers to load.")
        sys.exit(0)

    # Upsert in batches
    t0 = time.time()
    loaded = 0
    for start in range(0, len(documents), args.batch_size):
        end = min(start + args.batch_size, len(documents))
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        loaded += (end - start)
        if loaded % 500 == 0 or loaded == len(documents):
            elapsed = time.time() - t0
            rate = loaded / elapsed if elapsed > 0 else 0
            print(f"  {loaded}/{len(documents)} loaded ({rate:.0f}/s)")

    elapsed = time.time() - t0
    print(f"\n✅ Loaded {len(documents)} OnePagers into '{args.collection}'")
    print(f"   Path: {args.chroma_path}")
    print(f"   Collection count: {collection.count()}")
    print(f"   Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
