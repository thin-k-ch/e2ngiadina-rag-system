"""
EML Indexer with Attachment OCR Support
Indexes .eml files including attachments with text extraction
"""
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .chunking import chunk_text
from .hashing import sha1_file, file_stat
from .manifest import Manifest, ManifestRow
from .chroma_store import ChromaStore
from .text_loaders import read_eml_with_attachments

SUPPORTED = {".eml"}

def log(path, msg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")

def rel_meta(root: str, full_path: str) -> dict:
    rel = os.path.relpath(full_path, root)
    parts = rel.split(os.sep)
    dirs = parts[:-1]
    fn = parts[-1]
    ext = os.path.splitext(fn)[1].lower()
    meta = {"original_path": rel, "filename": fn, "ext": ext, "document_type": "eml"}
    for i in range(8):
        meta[f"dir_{i+1}"] = dirs[i] if i < len(dirs) else ""
    meta["category"] = meta["dir_1"] or "root"
    return meta

def main():
    ROOT = os.getenv("ROOT_DIR", "/data")
    CHROMA_PATH = os.getenv("CHROMA_PATH", "/chroma")
    COLLECTION = os.getenv("COLLECTION_EML", "documents_eml")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
    OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
    MANIFEST_PATH = os.getenv("MANIFEST_PATH", "/manifest/manifest.sqlite3")
    LOG_PATH = os.getenv("LOG_PATH", "/logs/indexer_eml.log")
    WORKERS = int(os.getenv("WORKERS", "6"))
    BATCH_UPSERT = int(os.getenv("BATCH_UPSERT", "256"))
    MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "200"))

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    log(LOG_PATH, f"START EML indexer root={ROOT} collection={COLLECTION} model={EMBED_MODEL}")

    manifest = Manifest(MANIFEST_PATH)
    store = ChromaStore(CHROMA_PATH, COLLECTION)
    embedder = SentenceTransformer(EMBED_MODEL)

    # Find all EML files
    files = []
    for dirpath, _, filenames in os.walk(ROOT):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in SUPPORTED:
                files.append(os.path.join(dirpath, fn))

    log(LOG_PATH, f"SCAN found {len(files)} EML files")

    # Filter: only new or changed files (check manifest)
    new_files = []
    for f in files:
        stat = file_stat(f)
        row = manifest.get(f)
        if row is None or row.mtime != stat["mtime"] or row.size != stat["size"]:
            new_files.append(f)

    log(LOG_PATH, f"NEW/CHANGED {len(new_files)} EML files to index")

    if not new_files:
        log(LOG_PATH, "No new EML files to index. Exiting.")
        return

    # Process files with attachments
    def process_one(path: str):
        try:
            # Read EML with attachment OCR
            result = read_eml_with_attachments(path)
            if not result or not result.get("text"):
                return None, f"NO_TEXT: {path}"

            text = result["text"]
            attachments = result.get("attachments", [])
            
            if len(text) < MIN_TEXT_CHARS:
                return None, f"TOO_SHORT: {path} ({len(text)} chars)"

            # Create metadata
            meta = rel_meta(ROOT, path)
            meta["attachment_count"] = len(attachments)
            meta["attachment_names"] = [a.get("filename", "unknown") for a in attachments]

            # Chunk the combined text
            chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
            if not chunks:
                return None, f"NO_CHUNKS: {path}"

            # Embeddings
            embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

            return {
                "path": path,
                "meta": meta,
                "chunks": chunks,
                "embeddings": embeddings,
                "attachments": attachments
            }, None

        except Exception as e:
            return None, f"ERROR: {path} - {e}"

    # Parallel processing
    buffer = []
    errors = []
    processed = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process_one, f): f for f in new_files}
        for fut in tqdm(as_completed(futures), total=len(new_files), desc="EML+OCR"):
            result, err = fut.result()
            if err:
                errors.append(err)
                log(LOG_PATH, err)
                continue

            buffer.append(result)
            processed += 1

            # Batch upsert to Chroma
            if len(buffer) >= BATCH_UPSERT:
                _flush_buffer(store, manifest, buffer, LOG_PATH)
                buffer = []

    # Final flush
    if buffer:
        _flush_buffer(store, manifest, buffer, LOG_PATH)

    log(LOG_PATH, f"DONE EML: processed={processed}, errors={len(errors)}")
    print(f"EML Indexing complete: {processed} files, {len(errors)} errors")

def _flush_buffer(store, manifest, buffer, log_path):
    """Upsert batch to Chroma and update manifest"""
    for item in buffer:
        try:
            # Store in Chroma
            store.upsert_document(
                path=item["path"],
                chunks=item["chunks"],
                embeddings=item["embeddings"],
                meta=item["meta"]
            )

            # Update manifest
            stat = file_stat(item["path"])
            manifest_row = ManifestRow(
                path=item["path"],
                sha1=sha1_file(item["path"]),
                mtime=stat["mtime"],
                size=stat["size"],
                chunk_count=len(item["chunks"]),
                indexed_at=__import__('time').time()
            )
            manifest.upsert(manifest_row)

        except Exception as e:
            log(log_path, f"UPSERT_ERROR: {item['path']} - {e}")

if __name__ == "__main__":
    main()
