# /media/felix/RAG/AGENTIC/indexer/app/index_all.py
import os
import traceback
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .pdf_extract import extract_pdf_text
from .chunking import chunk_text
from .hashing import sha1_file, file_stat
from .manifest import Manifest, ManifestRow
from .chroma_store import ChromaStore

# Reuse loaders you created in text_loaders.py
from .text_loaders import (
    read_text_file,
    read_docx,
    read_pptx,
    read_xlsx,
    read_msg,
    read_html,
    read_text_bytes,
)

# --------- helpers ---------
SUPPORTED_EXT = {
    ".pdf",
    ".txt", ".md", ".csv",
    ".docx",
    ".pptx",
    ".xlsx",
    ".msg",
    ".zip",
    ".html", ".htm",
    ".json", ".xml", ".yaml", ".yml",
}

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

    meta = {
        "original_path": rel,
        "filename": fn,
        "ext": ext,
    }
    for i in range(8):
        meta[f"dir_{i+1}"] = dirs[i] if i < len(dirs) else ""
    meta["category"] = meta["dir_1"] or "root"
    return meta

def safe_zip_members(z: zipfile.ZipFile):
    for info in z.infolist():
        name = info.filename
        if info.is_dir():
            continue
        # block absolute and traversal
        if name.startswith("/") or name.startswith("\\"):
            continue
        if ".." in name.replace("\\", "/").split("/"):
            continue
        yield info

def extract_text_by_ext(path: str, ext: str) -> tuple[str, dict]:
    """
    Returns (text, extra_meta). extra_meta includes document_type etc.
    """
    ext = ext.lower()
    extra = {"document_type": ext.lstrip(".")}

    if ext in (".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml"):
        return read_text_file(path), extra
    if ext == ".docx":
        return read_docx(path), extra
    if ext == ".pptx":
        return read_pptx(path), extra
    if ext == ".xlsx":
        return read_xlsx(path), extra
    if ext == ".msg":
        return read_msg(path), extra
    if ext in (".html", ".htm"):
        return read_html(path), extra

    return "", extra

def extract_text_from_zip_bytes(inner_path: str, b: bytes) -> tuple[str, dict, str] | None:
    """
    Returns (text, extra_meta, inner_ext) for supported inner files.
    For office formats it writes to a temp file and reuses extract_text_by_ext/pdf.
    """
    inner_ext = os.path.splitext(inner_path)[1].lower()
    extra = {"zip_inner_path": inner_path, "document_type": inner_ext.lstrip(".")}

    # Quick path for text-like
    if inner_ext in (".txt", ".md", ".csv", ".html", ".htm", ".json", ".xml", ".yaml", ".yml"):
        txt = read_text_bytes(b)
        if inner_ext in (".html", ".htm"):
            # reuse HTML parsing by writing a temp file not needed; do simple soup here
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(txt, "lxml")
            return soup.get_text("\n"), extra, inner_ext
        return txt, extra, inner_ext

    # For binary formats: write temp file and parse
    if inner_ext in (".pdf", ".docx", ".pptx", ".xlsx", ".msg"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=inner_ext) as tf:
            tf.write(b)
            tmp_path = tf.name
        try:
            if inner_ext == ".pdf":
                # handled by process_pdf below; return sentinel text empty is not helpful
                # We'll extract here to simplify the pipeline:
                pages = extract_pdf_text(tmp_path)
                text = "\n\n".join((p.get("text") or "") for p in pages)
                return text, extra | {"document_type": "pdf"}, inner_ext
            text, m2 = extract_text_by_ext(tmp_path, inner_ext)
            return text, extra | m2, inner_ext
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if inner_ext == ".zip":
        # handled by recursion at the ZIP level
        return None

    return None

def chunks_from_text(text: str, chunk_size: int, overlap: int):
    # Treat as "page 0" for non-PDF
    chs = chunk_text(text or "", chunk_size, overlap)
    out = []
    for ci, ch in enumerate(chs):
        out.append((0, ci, ch))
    return out

def chunks_from_pdf(path: str, chunk_size: int, overlap: int):
    pages = extract_pdf_text(path)
    out = []
    for p in pages:
        page_no = int(p["page"])
        for ci, ch in enumerate(chunk_text(p["text"] or "", chunk_size, overlap)):
            out.append((page_no, ci, ch))
    return out

def process_zip(
    root: str,
    outer_abs_path: str,
    sha1: str,
    base_meta: dict,
    chunk_size: int,
    overlap: int,
    max_depth: int,
    depth: int = 0,
):
    """
    Returns list of records: (chunk_id, chunk_text, metadata)
    """
    records = []
    if depth > max_depth:
        return records

    with zipfile.ZipFile(outer_abs_path, "r") as z:
        for info in safe_zip_members(z):
            inner_path = info.filename
            inner_ext = os.path.splitext(inner_path)[1].lower()

            # If inner is a ZIP, recurse by extracting bytes to temp file
            if inner_ext == ".zip":
                if depth == max_depth:
                    continue
                inner_bytes = z.read(info)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tf:
                    tf.write(inner_bytes)
                    tmp_zip = tf.name
                try:
                    # recurse: treat the "outer file" sha1 as stable anchor
                    # include inner_path in metadata so it stays traceable
                    inner_base = dict(base_meta)
                    inner_base["document_type"] = "zip"
                    inner_base["zip_inner_path"] = inner_path
                    records.extend(
                        process_zip(
                            root=root,
                            outer_abs_path=tmp_zip,
                            sha1=sha1,
                            base_meta=inner_base,
                            chunk_size=chunk_size,
                            overlap=overlap,
                            max_depth=max_depth,
                            depth=depth + 1,
                        )
                    )
                finally:
                    try:
                        os.unlink(tmp_zip)
                    except Exception:
                        pass
                continue

            if inner_ext not in SUPPORTED_EXT:
                continue

            try:
                b = z.read(info)
                extracted = extract_text_from_zip_bytes(inner_path, b)
                if not extracted:
                    continue
                text, extra_meta, _ = extracted
                if not (text or "").strip():
                    continue

                for page, ci, ch in chunks_from_text(text, chunk_size, overlap):
                    meta = dict(base_meta)
                    meta.update(extra_meta)
                    meta["sha1"] = sha1
                    meta["page"] = page
                    meta["chunk_index"] = ci
                    # stable ID includes inner path
                    chunk_id = f"{sha1}:{inner_path}:{ci}"
                    records.append((chunk_id, ch, meta))
            except Exception:
                # skip inner errors, continue
                continue

    return records

# --------- main indexing ---------
def main():
    ROOT = os.getenv("ROOT_DIR", "/data")
    CHROMA_PATH = os.getenv("CHROMA_PATH", "/chroma")
    COLLECTION = os.getenv("COLLECTION", "documents")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
    OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
    MANIFEST_PATH = os.getenv("MANIFEST_PATH", "/manifest/manifest.sqlite3")
    LOG_PATH = os.getenv("LOG_PATH", "/logs/indexer.log")
    WORKERS = int(os.getenv("WORKERS", "6"))
    BATCH_UPSERT = int(os.getenv("BATCH_UPSERT", "256"))
    ZIP_MAX_DEPTH = int(os.getenv("ZIP_MAX_DEPTH", "2"))

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    log(LOG_PATH, f"START indexer root={ROOT} chroma={CHROMA_PATH} model={EMBED_MODEL}")
    log(LOG_PATH, f"SUPPORTED={sorted(SUPPORTED_EXT)} zip_max_depth={ZIP_MAX_DEPTH}")

    manifest = Manifest(MANIFEST_PATH)
    store = ChromaStore(CHROMA_PATH, COLLECTION)
    embedder = SentenceTransformer(EMBED_MODEL)

    # scan files
    files = []
    for dirpath, _, fns in os.walk(ROOT):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SUPPORTED_EXT:
                files.append(os.path.join(dirpath, fn))
    files.sort()
    log(LOG_PATH, f"FOUND files={len(files)}")

    def handle_one(abs_path: str):
        try:
            mtime, size = file_stat(abs_path)
            old = manifest.get(abs_path)
            if old and old.mtime == mtime and old.size == size:
                return ("skipped", abs_path, 0)

            ext = os.path.splitext(abs_path)[1].lower()
            sha1 = sha1_file(abs_path)
            base_meta = rel_meta(ROOT, abs_path)
            base_meta["sha1"] = sha1

            records = []  # list of (id, text, meta)

            if ext == ".pdf":
                chunks = chunks_from_pdf(abs_path, CHUNK_SIZE, OVERLAP)
                for page, ci, ch in chunks:
                    if not (ch or "").strip():
                        continue
                    meta = dict(base_meta)
                    meta["document_type"] = "pdf"
                    meta["page"] = page
                    meta["chunk_index"] = ci
                    chunk_id = f"{sha1}:{page}:{ci}"
                    records.append((chunk_id, ch, meta))

            elif ext == ".zip":
                # recurse inside zip, preserve outer original_path in meta + inner paths
                base_meta["document_type"] = "zip"
                records = process_zip(
                    root=ROOT,
                    outer_abs_path=abs_path,
                    sha1=sha1,
                    base_meta=base_meta,
                    chunk_size=CHUNK_SIZE,
                    overlap=OVERLAP,
                    max_depth=ZIP_MAX_DEPTH,
                    depth=0,
                )

            else:
                text, extra = extract_text_by_ext(abs_path, ext)
                if not (text or "").strip():
                    return ("empty", abs_path, 0)
                for page, ci, ch in chunks_from_text(text, CHUNK_SIZE, OVERLAP):
                    if not (ch or "").strip():
                        continue
                    meta = dict(base_meta)
                    meta.update(extra)
                    meta["page"] = page
                    meta["chunk_index"] = ci
                    chunk_id = f"{sha1}:0:{ci}"
                    records.append((chunk_id, ch, meta))

            if not records:
                return ("empty", abs_path, 0)

            # embeddings + upsert in batches
            total = 0
            for start in range(0, len(records), BATCH_UPSERT):
                batch = records[start:start + BATCH_UPSERT]
                ids = [r[0] for r in batch]
                docs = [r[1] for r in batch]
                metas = [r[2] for r in batch]
                embs = embedder.encode(docs, convert_to_tensor=False, show_progress_bar=False).tolist()
                store.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                total += len(docs)

            manifest.upsert(ManifestRow(path=abs_path, sha1=sha1, mtime=mtime, size=size))
            return ("ok", abs_path, total)

        except Exception:
            log(LOG_PATH, f"ERROR path={abs_path}\n{traceback.format_exc()}")
            return ("err", abs_path, 0)

    total_chunks = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(handle_one, p) for p in files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Files"):
            status, path, n = fut.result()
            total_chunks += n
            if status in ("empty", "err"):
                log(LOG_PATH, f"{status.upper()} {path}")

    log(LOG_PATH, f"DONE total_chunks={total_chunks} chroma_count={store.count()}")

if __name__ == "__main__":
    main()
