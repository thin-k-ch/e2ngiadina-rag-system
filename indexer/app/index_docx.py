import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .chunking import chunk_text
from .hashing import sha1_file, file_stat
from .manifest import Manifest, ManifestRow
from .chroma_store import ChromaStore
from .text_loaders import read_docx

def log(path, msg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")

def rel_parts(root, full):
    rel = os.path.relpath(full, root)
    parts = rel.split(os.sep)
    dirs = parts[:-1]
    meta = {
        "original_path": rel,
        "filename": parts[-1],
        "ext": os.path.splitext(parts[-1])[1].lower(),
    }
    for i in range(8):
        meta[f"dir_{i+1}"] = dirs[i] if i < len(dirs) else ""
    meta["category"] = meta["dir_1"] or "root"
    return meta

def process_docx(root, path, chunk_size, overlap):
    try:
        mtime, size = file_stat(path)
        h = sha1_file(path)
        text = read_docx(path)
        if not text or len(text.strip()) < 200:
            return ("empty", path, 0)
        
        chunks = chunk_text(text, chunk_size, overlap)
        base_meta = rel_parts(root, path)
        items = []
        for ci, chunk in enumerate(chunks):
            chunk_id = f"{h}:0:{ci}"
            meta = dict(base_meta)
            meta["sha1"] = h
            meta["page"] = 0
            meta["chunk_index"] = ci
            items.append((chunk_id, chunk, meta))
        return ("ok", path, items)
    except Exception:
        log(LOG_PATH, f"ERROR path={path}\n{traceback.format_exc()}")
        return ("err", path, [])

def main():
    ROOT = os.getenv("ROOT_DIR", "/data")
    CHROMA_PATH = os.getenv("CHROMA_PATH", "/chroma")
    COLLECTION = os.getenv("COLLECTION_DOCX", "documents_docx")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
    OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
    MANIFEST_PATH = os.getenv("MANIFEST_PATH", "/manifest/manifest.sqlite3")
    LOG_PATH = os.getenv("LOG_PATH", "/logs/indexer_docx.log")
    WORKERS = int(os.getenv("WORKERS", "6"))
    BATCH_UPSERT = int(os.getenv("BATCH_UPSERT", "256"))
    MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "200"))

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    log(LOG_PATH, f"START DOCX indexer root={ROOT} collection={COLLECTION} model={EMBED_MODEL}")

    manifest = Manifest(MANIFEST_PATH)
    store = ChromaStore(CHROMA_PATH, COLLECTION)
    embedder = SentenceTransformer(EMBED_MODEL)

    files = []
    for dirpath, _, fns in os.walk(ROOT):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext == ".docx":
                files.append(os.path.join(dirpath, fn))
    files.sort()
    log(LOG_PATH, f"FOUND docx={len(files)}")

    def handle_one(path):
        try:
            mtime, size = file_stat(path)
            old = manifest.get(path)
            if old and old.mtime == mtime and old.size == size:
                return ("skipped", path, 0)

            h = sha1_file(path)
            items = process_docx(ROOT, path, CHUNK_SIZE, OVERLAP)
            if items[0] != "ok":
                return items

            _, _, chunks = items
            ids, docs, metas = [], [], []
            for chunk_id, chunk, meta in chunks:
                ids.append(chunk_id)
                docs.append(chunk)
                metas.append(meta)

            for start in range(0, len(docs), BATCH_UPSERT):
                ds = docs[start:start + BATCH_UPSERT]
                es = embedder.encode(ds, convert_to_tensor=False, show_progress_bar=False).tolist()
                store.upsert(
                    ids=ids[start:start + BATCH_UPSERT],
                    documents=ds,
                    metadatas=metas[start:start + BATCH_UPSERT],
                    embeddings=es,
                )

            manifest.upsert(ManifestRow(path=path, sha1=h, mtime=mtime, size=size))
            return ("ok", path, len(docs))
        except Exception:
            log(LOG_PATH, f"ERROR path={path}\n{traceback.format_exc()}")
            return ("err", path, 0)

    total_chunks = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(handle_one, p) for p in files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="DOCX"):
            status, path, n = fut.result()
            total_chunks += n
            if status in ("empty", "err"):
                log(LOG_PATH, f"{status.upper()} {path}")

    log(LOG_PATH, f"DONE total_chunks={total_chunks} chroma_count={store.count()}")

if __name__ == "__main__":
    main()
