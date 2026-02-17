#!/usr/bin/env python3
"""
Load batch-pipeline findings into a Chroma collection for use as
a pre-computed knowledge layer in the RAG agent.

Usage:
  python load_findings_to_chroma.py findings.json
  python load_findings_to_chroma.py findings.json --collection tfk18_findings
  python load_findings_to_chroma.py findings.json --chroma-path /chroma
"""

import argparse
import json
import os
import sys

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def build_document_text(finding: dict) -> str:
    """Build a searchable text representation of a finding."""
    parts = []
    if finding.get("title"):
        parts.append(f"Titel: {finding['title']}")
    if finding.get("category"):
        parts.append(f"Kategorie: {finding['category']}")
    if finding.get("statement"):
        parts.append(f"Befund: {finding['statement']}")
    if finding.get("recommendation"):
        parts.append(f"Empfehlung: {finding['recommendation']}")
    if finding.get("evidence"):
        ev = finding["evidence"]
        if isinstance(ev, list):
            for e in ev[:5]:  # limit to 5 evidence items
                if isinstance(e, dict):
                    src = e.get('path', e.get('doc', 'unbekannt'))
                    # Use just filename for readability
                    if '/' in src:
                        src = src.rsplit('/', 1)[-1]
                    quote = e.get('quote', '')
                    if quote:
                        parts.append(f"Evidenz ({src}): {quote}")
                else:
                    parts.append(f"Evidenz: {e}")
        elif isinstance(ev, str):
            parts.append(f"Evidenz: {ev}")
    return "\n".join(parts)


def build_metadata(finding: dict, index: int) -> dict:
    """Build metadata dict for a finding (Chroma requires flat str/int/float values)."""
    meta = {
        "finding_index": index,
        "title": str(finding.get("title", ""))[:500],
        "category": str(finding.get("category", "unknown")),
        "impact": str(finding.get("impact", "unknown")),
        "confidence": float(finding.get("confidence", 0) or 0),
        "source_type": "batch_finding",
    }
    # Flatten evidence sources
    ev = finding.get("evidence", [])
    if isinstance(ev, list):
        docs = []
        for e in ev[:5]:
            if isinstance(e, dict):
                p = e.get('path', e.get('doc', ''))
                if '/' in p:
                    p = p.rsplit('/', 1)[-1]
                docs.append(p)
            else:
                docs.append(str(e))
        meta["evidence_docs"] = "; ".join(d for d in docs if d)
    elif isinstance(ev, str):
        meta["evidence_docs"] = ev[:500]
    return meta


def main():
    ap = argparse.ArgumentParser(description="Load findings into Chroma collection")
    ap.add_argument("findings_file", help="Path to findings.json")
    ap.add_argument("--collection", default="tfk18_findings",
                    help="Chroma collection name (default: tfk18_findings)")
    ap.add_argument("--chroma-path", default=os.getenv("CHROMA_PATH", "/chroma"),
                    help="Chroma persistent storage path")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Embedding model (must match existing collections)")
    ap.add_argument("--replace", action="store_true",
                    help="Delete existing collection before loading")
    args = ap.parse_args()

    # Load findings
    with open(args.findings_file, "r", encoding="utf-8") as f:
        findings = json.load(f)

    if not isinstance(findings, list):
        print(f"Error: expected JSON array, got {type(findings).__name__}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(findings)} findings from {args.findings_file}")

    # Init Chroma
    emb_fn = SentenceTransformerEmbeddingFunction(model_name=args.embed_model)
    client = chromadb.PersistentClient(
        path=args.chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )

    if args.replace:
        try:
            client.delete_collection(args.collection)
            print(f"Deleted existing collection '{args.collection}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=emb_fn,
    )

    # Prepare documents
    ids = []
    documents = []
    metadatas = []

    for i, finding in enumerate(findings):
        doc_text = build_document_text(finding)
        if not doc_text.strip():
            continue
        ids.append(f"finding_{i:04d}")
        documents.append(doc_text)
        metadatas.append(build_metadata(finding, i))

    if not documents:
        print("No valid findings to load.")
        sys.exit(0)

    # Upsert (idempotent)
    batch_size = 100
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"✅ Loaded {len(documents)} findings into Chroma collection '{args.collection}'")
    print(f"   Path: {args.chroma_path}")
    print(f"   Embedding: {args.embed_model}")
    print(f"   Collection count: {collection.count()}")


if __name__ == "__main__":
    main()
