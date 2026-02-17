#!/usr/bin/env python3
"""
Targeted PDF text extraction for TFK18 ES index.
Finds PDF docs with placeholder content and replaces with pdfplumber-extracted text.
Uses parallel extraction + bulk ES updates for speed.
"""
import os
import json
import time
import requests
import pdfplumber
from concurrent.futures import ThreadPoolExecutor, as_completed

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = "rag_tfk18_v1"
TFK18_ROOT = "/media/felix/RAG/TFK18"
SCROLL_SIZE = 200
WORKERS = 6


def extract_pdf_text(filepath, max_pages=200):
    """Extract text from PDF using pdfplumber."""
    try:
        pages = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages[:max_pages]:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)
        return "\n\n".join(pages) if pages else None
    except Exception:
        return None


def process_one(doc_id, vpath, fname):
    """Extract text for one PDF. Returns (doc_id, text) or (doc_id, None)."""
    rel = vpath.lstrip("/")
    full = os.path.join(TFK18_ROOT, rel)
    if not os.path.isfile(full):
        return (doc_id, None, "not_found")
    text = extract_pdf_text(full)
    if not text or len(text.strip()) < 10:
        return (doc_id, None, "no_text")
    return (doc_id, text, "ok")


def bulk_update_es(updates):
    """Bulk update ES documents with extracted text."""
    if not updates:
        return 0
    lines = []
    for doc_id, text in updates:
        lines.append(json.dumps({"update": {"_index": ES_INDEX, "_id": doc_id}}))
        lines.append(json.dumps({"doc": {"content": text}}))
    body = "\n".join(lines) + "\n"
    try:
        r = requests.post(f"{ES_URL}/_bulk", data=body,
                          headers={"Content-Type": "application/json"}, timeout=120)
        if r.status_code == 200:
            result = r.json()
            return sum(1 for item in result.get("items", [])
                       if item.get("update", {}).get("status") == 200)
    except Exception as e:
        print(f"  ⚠️ Bulk error: {e}")
    return 0


def main():
    print("=" * 60)
    print("TFK18 PDF Text Extraction (parallel + bulk)")
    print(f"ES Index: {ES_INDEX} | Workers: {WORKERS}")
    print("=" * 60)

    # Scroll through all PDF docs with placeholder content
    scroll_body = {
        "size": SCROLL_SIZE,
        "query": {
            "bool": {
                "must": [
                    {"term": {"file.extension": "pdf"}},
                    {"match_phrase": {"content": "[File:"}}
                ]
            }
        },
        "_source": ["file.filename", "path.virtual"]
    }

    r = requests.post(f"{ES_URL}/{ES_INDEX}/_search?scroll=30m", json=scroll_body, timeout=60)
    data = r.json()
    scroll_id = data.get("_scroll_id")
    hits = data.get("hits", {}).get("hits", [])
    total = data.get("hits", {}).get("total", {}).get("value", 0)
    print(f"\n📄 PDFs to process: {total:,}")

    updated = 0
    failed = 0
    not_found = 0
    start = time.time()

    while hits:
        # Parallel extraction
        batch_updates = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {}
            for h in hits:
                doc_id = h["_id"]
                src = h["_source"]
                vpath = src.get("path", {}).get("virtual", "")
                fname = src.get("file", {}).get("filename", "")
                futures[ex.submit(process_one, doc_id, vpath, fname)] = doc_id

            for fut in as_completed(futures):
                doc_id, text, status = fut.result()
                if status == "ok":
                    batch_updates.append((doc_id, text))
                elif status == "not_found":
                    not_found += 1
                else:
                    failed += 1

        # Bulk update ES
        bulk_ok = bulk_update_es(batch_updates)
        updated += bulk_ok
        failed += len(batch_updates) - bulk_ok

        processed = updated + failed + not_found
        elapsed = time.time() - start
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (total - processed) / rate if rate > 0 else 0
        print(f"  ⏳ {processed:,}/{total:,} ({rate:.1f}/s) | ✅ {updated:,} | ❌ {failed:,} | 🔍 {not_found:,} | ETA: {remaining/60:.0f}min")

        # Next scroll
        r = requests.post(f"{ES_URL}/_search/scroll",
                          json={"scroll": "30m", "scroll_id": scroll_id}, timeout=60)
        data = r.json()
        scroll_id = data.get("_scroll_id")
        hits = data.get("hits", {}).get("hits", [])

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"✅ Done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"   Updated:   {updated:,}")
    print(f"   Failed:    {failed:,}")
    print(f"   Not found: {not_found:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
