#!/usr/bin/env python3
"""
Incremental EML Indexer for Elasticsearch (Host Execution)
Only indexes new/changed EML files with attachments + OCR
Run from host: python3 index_eml_to_es_host.py
"""
import os
import sys
import hashlib
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# Add local paths for imports
sys.path.insert(0, '/media/felix/RAG/AGENTIC')
sys.path.insert(0, '/media/felix/RAG/AGENTIC/indexer')

try:
    from app.text_loaders import read_eml_with_attachments, read_text_bytes
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run from /media/felix/RAG/AGENTIC with venv activated")
    sys.exit(1)

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_files_v1")
DATA_DIR = os.getenv("DATA_DIR", "/media/felix/RAG/1")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
WORKERS = int(os.getenv("WORKERS", "4"))

def get_file_hash(path):
    """Get SHA256 hash of file"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_already_indexed_eml():
    """Query ES for already indexed EML files"""
    try:
        query = {
            "size": 10000,
            "query": {
                "bool": {
                    "filter": [
                        {"terms": {"file.extension": [".eml", ".msg"]}}
                    ]
                }
            },
            "_source": ["path.virtual", "meta.sha256"]
        }
        
        response = requests.post(
            f"{ES_URL}/{ES_INDEX}/_search?scroll=2m",
            json=query,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"⚠️ ES query failed: {response.status_code}")
            return {}
        
        result = response.json()
        scroll_id = result.get("_scroll_id")
        hits = result["hits"]["hits"]
        
        indexed = {}
        for hit in hits:
            source = hit.get("_source", {})
            path = source.get("path", {}).get("virtual", "")
            sha = source.get("meta", {}).get("sha256", "")
            if path:
                indexed[path] = sha
        
        print(f"   Found {len(indexed)} EML files already in ES")
        return indexed
        
    except Exception as e:
        print(f"⚠️ Could not query ES: {e}")
        return {}

def process_eml_file(filepath, relative_path):
    """Process EML with attachments and OCR"""
    try:
        result = read_eml_with_attachments(str(filepath))
        if not result or not result.get("text"):
            return None, "NO_TEXT"
        
        text = result["text"]
        attachments = result.get("attachments", [])
        
        if len(text) < 100:
            return None, "TOO_SHORT"
        
        file_hash = get_file_hash(filepath)
        stat = filepath.stat()
        
        doc = {
            "content": text,
            "path": {
                "real": str(filepath),
                "virtual": str(relative_path)
            },
            "file": {
                "filename": filepath.name,
                "extension": filepath.suffix.lower(),
                "size": stat.st_size
            },
            "meta": {
                "sha256": file_hash,
                "mtime": int(stat.st_mtime),
                "indexed_at": datetime.now().isoformat(),
                "attachment_count": len([a for a in attachments if a.get("text")]),
                "attachment_names": [a.get("filename", "unknown") for a in attachments if a.get("text")]
            }
        }
        
        doc_id = file_hash[:16]
        return {"_id": doc_id, "_source": doc}, None
        
    except Exception as e:
        return None, f"ERROR: {e}"

def bulk_index(actions):
    """Bulk index to ES using _bulk API"""
    if not actions:
        return True
    
    bulk_body = ""
    for action in actions:
        meta = json.dumps({"index": {"_index": ES_INDEX, "_id": action["_id"]}})
        source = json.dumps(action["_source"])
        bulk_body += meta + "\n" + source + "\n"
    
    try:
        response = requests.post(
            f"{ES_URL}/_bulk",
            data=bulk_body,
            headers={"Content-Type": "application/x-ndjson"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("errors"):
                errors = [item for item in result["items"] if item.get("index", {}).get("error")]
                print(f"   ⚠️ {len(errors)} bulk errors")
                return False
            return True
        else:
            print(f"   ❌ Bulk request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Bulk error: {e}")
        return False

def main():
    print("=" * 60)
    print("Incremental EML Indexer for Elasticsearch")
    print("=" * 60)
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"🎯 ES index: {ES_INDEX}")
    print(f"🔗 ES URL: {ES_URL}")
    
    # Check ES connection
    try:
        r = requests.get(ES_URL, timeout=5)
        if r.status_code == 200:
            print("✅ Connected to Elasticsearch")
        else:
            print(f"❌ ES not ready: {r.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to ES: {e}")
        return
    
    # Get already indexed files
    print("\n🔍 Checking already indexed EML files...")
    indexed_files = get_already_indexed_eml()
    
    # Find all EML files on disk
    print("\n🔍 Scanning disk for EML files...")
    data_path = Path(DATA_DIR)
    disk_files = list(data_path.rglob("*.eml"))
    print(f"   Found {len(disk_files)} EML files on disk")
    
    # Determine which files need indexing
    print("\n⚖️  Comparing disk vs ES...")
    to_index = []
    for filepath in disk_files:
        relative = filepath.relative_to(data_path)
        current_hash = get_file_hash(filepath)
        existing_hash = indexed_files.get(str(relative), "")
        
        if str(relative) not in indexed_files:
            to_index.append((filepath, relative, "NEW"))
        elif current_hash != existing_hash:
            to_index.append((filepath, relative, "CHANGED"))
    
    new_count = sum(1 for _, _, reason in to_index if reason == "NEW")
    changed_count = sum(1 for _, _, reason in to_index if reason == "CHANGED")
    
    print(f"   NEW files: {new_count}")
    print(f"   CHANGED files: {changed_count}")
    print(f"   SKIPPED (unchanged): {len(disk_files) - len(to_index)}")
    
    if not to_index:
        print("\n✅ No new EML files to index. All up to date!")
        return
    
    # Process files
    print(f"\n🚀 Indexing {len(to_index)} EML files...")
    
    actions = []
    processed = 0
    errors = 0
    
    def process_one(args):
        filepath, relative, reason = args
        action, err = process_eml_file(filepath, relative)
        return action, err, str(relative)
    
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process_one, item): item for item in to_index}
        
        for i, fut in enumerate(futures):
            action, err, path = fut.result()
            
            if err:
                errors += 1
                if err != "NO_TEXT" and err != "TOO_SHORT":
                    print(f"   ⚠️  {path}: {err}")
            else:
                actions.append(action)
                processed += 1
            
            # Bulk index every BATCH_SIZE
            if len(actions) >= BATCH_SIZE:
                print(f"   📤 Indexing batch {i//BATCH_SIZE + 1}...")
                bulk_index(actions)
                actions = []
    
    # Final batch
    if actions:
        print(f"   📤 Indexing final batch...")
        bulk_index(actions)
    
    print("\n" + "=" * 60)
    print(f"✅ EML Indexing Complete:")
    print(f"   Processed: {processed} files")
    print(f"   Errors: {errors}")
    print(f"   Total EML in ES: {len(indexed_files) + processed}")
    print("=" * 60)

if __name__ == "__main__":
    main()
