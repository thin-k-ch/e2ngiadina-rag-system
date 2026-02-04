import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .chunking import chunk_text
from .hashing import sha1_file, file_stat
from .manifest import Manifest, ManifestRow
from .chroma_store import ChromaStore
from .text_loaders import read_msg

SUPPORTED = {".msg"}

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
    meta = {"original_path": rel, "filename": fn, "ext": ext, "document_type": "msg"}
    for i in range(8):
        meta[f"dir_{i+1}"] = dirs[i] if i < len(dirs) else ""
    meta["category"] = meta["dir_1"] or "root"
    return meta

def main():
    ROOT = os.getenv("ROOT_DIR", "/data")
    CHROMA_PATH = os.getenv("CHROMA_PATH", "/chroma")
    COLLECTION = os.getenv("COLLECTION_MSG", "documents_msg")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
    OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
    MANIFEST_PATH = os.getenv("MANIFEST_PATH", "/manifest/manifest.sqlite3")
    LOG_PATH = os.getenv("LOG_PATH", "/logs/indexer_msg.log")
    WORKERS = int(os.getenv("WORKERS", "6"))
    BATCH_UPSERT = int(os.getenv("BATCH_UPSERT", "256"))
    MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "200"))

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    log(LOG_PATH, f"START MSG indexer root={ROOT} collection={COLLECTION} model={EMBED_MODEL}")

    manifest = Manifest(MANIFEST_PATH)
    store = ChromaStore(CHROMA_PATH, COLLECTION)
    embedder = SentenceTransformer(EMBED_MODEL)

    files = []
    for dirpath, _, fns in os.walk(ROOT):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SUPPORTED:
                files.append(os.path.join(dirpath, fn))
    files.sort()
    log(LOG_PATH, f"FOUND msg files={len(files)}")

    def handle_one(abs_path: str):
        try:
            mtime, size = file_stat(abs_path)
            old = manifest.get(abs_path)
            if old and old.mtime == mtime and old.size == size:
                return ("skipped", abs_path, 0)

            sha1 = sha1_file(abs_path)
            base_meta = rel_meta(ROOT, abs_path)
            base_meta["sha1"] = sha1

            text = (read_msg(abs_path) or "").strip()
            if len(text) < MIN_TEXT_CHARS:
                return ("empty", abs_path, 0)

            chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
            if not chunks:
                return ("empty", abs_path, 0)

            total = 0
            for start in range(0, len(chunks), BATCH_UPSERT):
                batch = chunks[start:start + BATCH_UPSERT]
                ids = [f"{sha1}:0:{start+i}" for i in range(len(batch))]
                metas = []
                for i in range(len(batch)):
                    md = dict(base_meta)
                    md["page"] = 0
                    md["chunk_index"] = start + i
                    metas.append(md)

                embs = embedder.encode(batch, convert_to_tensor=False, show_progress_bar=False).tolist()
                store.upsert(ids=ids, documents=batch, metadatas=metas, embeddings=embs)
                total += len(batch)

            manifest.upsert(ManifestRow(path=abs_path, sha1=sha1, mtime=mtime, size=size))
            return ("ok", abs_path, total)

        except Exception:
            log(LOG_PATH, f"ERROR path={abs_path}\n{traceback.format_exc()}")
            return ("err", abs_path, 0)

    total_chunks = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(handle_one, p) for p in files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="MSG files"):
            status, path, n = fut.result()
            total_chunks += n
            if status in ("empty", "err"):
                log(LOG_PATH, f"{status.upper()} {path}")

    log(LOG_PATH, f"DONE total_chunks={total_chunks} count={store.count()}")

if __name__ == "__main__":
    main()
