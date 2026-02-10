"""
Incremental EML Indexer for Elasticsearch
Only indexes new/changed EML files (with attachments + OCR)
"""
import os
import hashlib
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers

from indexer.app.text_loaders import read_eml_with_attachments, read_text_bytes

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_files_v1")
ROOT_DIR = os.getenv("ROOT_DIR", "/media/felix/RAG/1")
BATCH_SIZE = int(os.getenv("ES_BATCH", "100"))
WORKERS = int(os.getenv("WORKERS", "4"))
MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "100"))

def get_file_hash(path):
    """Get SHA256 hash of file for change detection"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_file_stat(path):
    """Get file stats for change detection"""
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime": int(stat.st_mtime)}

def get_already_indexed_files(es, index):
    """Query ES for already indexed EML files with their hashes"""
    try:
        # Scroll query to get all EML files
        query = {
            "size": 10000,
            "query": {
                "bool": {
                    "filter": [
                        {"terms": {"file.extension": [".eml", ".msg"]}}
                    ]
                }
            },
            "_source": ["path", "meta.sha256", "file.filename"]
        }
        
        result = es.search(index=index, body=query, scroll="2m")
        scroll_id = result.get("_scroll_id")
        hits = result["hits"]["hits"]
        
        indexed = {}
        for hit in hits:
            source = hit.get("_source", {})
            path = source.get("path", {}).get("real", "")
            sha = source.get("meta", {}).get("sha256", "")
            if path:
                indexed[path] = sha
        
        # Get remaining scroll results
        while len(hits) > 0:
            result = es.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = result.get("_scroll_id")
            hits = result["hits"]["hits"]
            for hit in hits:
                source = hit.get("_source", {})
                path = source.get("path", {}).get("real", "")
                sha = source.get("meta", {}).get("sha256", "")
                if path:
                    indexed[path] = sha
        
        return indexed
    except Exception as e:
        print(f"Warning: Could not query existing EML files: {e}")
        return {}

def process_eml_file(path, root_dir):
    """Process single EML file with attachments and OCR"""
    try:
        # Read EML with attachment OCR
        result = read_eml_with_attachments(path)
        if not result or not result.get("text"):
            return None, f"NO_TEXT: {path}"
        
        text = result["text"]
        attachments = result.get("attachments", [])
        
        if len(text) < MIN_TEXT_CHARS:
            return None, f"TOO_SHORT: {path} ({len(text)} chars)"
        
        # Get file metadata
        rel_path = os.path.relpath(path, root_dir)
        file_hash = get_file_hash(path)
        stat = get_file_stat(path)
        
        # Build ES document
        doc = {
            "content": text,
            "path": {"real": path, "virtual": rel_path},
            "file": {
                "filename": os.path.basename(path),
                "extension": ".eml",
                "size": stat["size"]
            },
            "meta": {
                "sha256": file_hash,
                "mtime": stat["mtime"],
                "indexed_at": __import__('time').time(),
                "attachment_count": len(attachments),
                "attachment_names": [a.get("filename", "unknown") for a in attachments if a.get("text")]
            }
        }
        
        return doc, None
        
    except Exception as e:
        return None, f"ERROR: {path} - {e}"

def main():
    print("=" * 60)
    print("Incremental EML Indexer for Elasticsearch")
    print("=" * 60)
    
    # Connect to ES
    es = Elasticsearch(ES_URL)
    
    # Check connection
    if not es.ping():
        print(f"❌ Cannot connect to Elasticsearch at {ES_URL}")
        return
    print(f"✅ Connected to Elasticsearch: {ES_URL}")
    print(f"📁 Target index: {ES_INDEX}")
    print(f"📂 Root directory: {ROOT_DIR}")
    
    # Get already indexed EML files
    print("\n🔍 Checking already indexed EML files...")
    indexed_files = get_already_indexed_files(es, ES_INDEX)
    print(f"   Found {len(indexed_files)} EML files already in ES")
    
    # Find all EML files on disk
    print("\n🔍 Scanning disk for EML files...")
    disk_files = []
    for dirpath, _, filenames in os.walk(ROOT_DIR):
        for fn in filenames:
            if fn.lower().endswith('.eml'):
                disk_files.append(os.path.join(dirpath, fn))
    print(f"   Found {len(disk_files)} EML files on disk")
    
    # Determine which files need indexing
    print("\n⚖️  Comparing disk vs ES...")
    to_index = []
    for path in disk_files:
        current_hash = get_file_hash(path)
        existing_hash = indexed_files.get(path, "")
        
        if path not in indexed_files:
            to_index.append((path, "NEW"))
        elif current_hash != existing_hash:
            to_index.append((path, "CHANGED"))
    
    new_count = sum(1 for _, reason in to_index if reason == "NEW")
    changed_count = sum(1 for _, reason in to_index if reason == "CHANGED")
    
    print(f"   NEW files: {new_count}")
    print(f"   CHANGED files: {changed_count}")
    print(f"   SKIPPED (unchanged): {len(disk_files) - len(to_index)}")
    
    if not to_index:
        print("\n✅ No new EML files to index. All up to date!")
        return
    
    # Process files in parallel
    print(f"\n🚀 Indexing {len(to_index)} EML files with attachments...")
    
    def process_one(args):
        path, reason = args
        doc, err = process_eml_file(path, ROOT_DIR)
        if err:
            return None, err
        
        # Create ES action
        file_hash = doc["meta"]["sha256"]
        action = {
            "_index": ES_INDEX,
            "_id": file_hash,  # Use hash as ID for deduplication
            "_source": doc
        }
        return action, None
    
    actions = []
    errors = []
    processed = 0
    
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process_one, item): item for item in to_index}
        for fut in tqdm(as_completed(futures), total=len(to_index), desc="EML+OCR"):
            action, err = fut.result()
            if err:
                errors.append(err)
                print(f"   ⚠️  {err}")
                continue
            
            actions.append(action)
            processed += 1
            
            # Batch index to ES
            if len(actions) >= BATCH_SIZE:
                try:
                    helpers.bulk(es, actions, request_timeout=120)
                    print(f"   📤 Indexed batch of {len(actions)} documents")
                except Exception as e:
                    print(f"   ❌ Batch indexing error: {e}")
                actions = []
    
    # Final batch
    if actions:
        try:
            helpers.bulk(es, actions, request_timeout=120)
            print(f"   📤 Indexed final batch of {len(actions)} documents")
        except Exception as e:
            print(f"   ❌ Final batch error: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ EML Indexing Complete:")
    print(f"   Processed: {processed} files")
    print(f"   Errors: {len(errors)}")
    print(f"   Total EML in ES: {len(indexed_files) + processed}")
    print("=" * 60)
    
    if errors:
        print(f"\n⚠️  {len(errors)} errors occurred (see above)")

if __name__ == "__main__":
    main()
