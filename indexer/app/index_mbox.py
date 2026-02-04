import os
import re
import traceback
import mailbox
import email
from email.header import decode_header
from bs4 import BeautifulSoup

from sentence_transformers import SentenceTransformer
from .chunking import chunk_text
from .hashing import sha1_file, file_stat
from .manifest import Manifest, ManifestRow
from .chroma_store import ChromaStore

def log(path, msg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")

def decode_mime(s: str) -> str:
    if not s:
        return ""
    out = []
    for part, enc in decode_header(s):
        if isinstance(part, bytes):
            try:
                out.append(part.decode(enc or "utf-8", errors="ignore"))
            except Exception:
                out.append(part.decode("utf-8", errors="ignore"))
        else:
            out.append(part)
    return "".join(out).strip()

def extract_body(msg: email.message.Message) -> str:
    # Prefer text/plain, fallback to html
    text_plain = []
    text_html = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = (part.get_content_type() or "").lower()
            disp = (part.get("Content-Disposition") or "").lower()
            if "attachment" in disp:
                continue
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            try:
                s = payload.decode(charset, errors="ignore")
            except Exception:
                s = payload.decode("utf-8", errors="ignore")

            if ctype == "text/plain" and s.strip():
                text_plain.append(s)
            elif ctype == "text/html" and s.strip():
                text_html.append(s)
    else:
        payload = msg.get_payload(decode=True) or b""
        charset = msg.get_content_charset() or "utf-8"
        try:
            s = payload.decode(charset, errors="ignore")
        except Exception:
            s = payload.decode("utf-8", errors="ignore")
        ctype = (msg.get_content_type() or "").lower()
        if ctype == "text/html":
            text_html.append(s)
        else:
            text_plain.append(s)

    if text_plain:
        return "\n".join(text_plain).strip()

    if text_html:
        soup = BeautifulSoup("\n".join(text_html), "lxml")
        return soup.get_text("\n").strip()

    return ""

def extract_attachments_meta(msg: email.message.Message):
    files = []
    for part in msg.walk():
        disp = (part.get("Content-Disposition") or "").lower()
        if "attachment" not in disp:
            continue
        fn = part.get_filename()
        fn = decode_mime(fn) if fn else ""
        files.append({
            "filename": fn,
            "content_type": part.get_content_type(),
        })
    return files

def find_mbox_files(mail_root: str):
    # Thunderbird mbox: file without extension, paired with .msf; also nested in .sbd dirs
    out = []
    for dirpath, _, fns in os.walk(mail_root):
        for fn in fns:
            if fn.endswith(".msf"):
                continue
            if fn.endswith(".sqlite"):
                continue
            if fn.endswith(".dat"):
                continue
            p = os.path.join(dirpath, fn)
            # Skip obvious attachment storage dirs
            if "mailbox-attachments" in p:
                continue
            # Heuristic: mbox files are usually large and have no dot-extension
            base, ext = os.path.splitext(fn)
            if ext != "":
                continue
            out.append(p)
    out.sort()
    return out

def main():
    MAIL_ROOT = os.getenv("MAIL_ROOT", "/mail")
    CHROMA_PATH = os.getenv("CHROMA_PATH", "/chroma")
    COLLECTION = os.getenv("COLLECTION_MAIL", "documents_mail")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
    OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
    MANIFEST_PATH = os.getenv("MANIFEST_PATH", "/manifest/manifest.sqlite3")
    LOG_PATH = os.getenv("LOG_PATH", "/logs/indexer_mail.log")
    BATCH_UPSERT = int(os.getenv("BATCH_UPSERT", "256"))
    MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "200"))

    log(LOG_PATH, f"START MBOX indexer root={MAIL_ROOT} collection={COLLECTION} model={EMBED_MODEL}")
    manifest = Manifest(MANIFEST_PATH)
    store = ChromaStore(CHROMA_PATH, COLLECTION)
    embedder = SentenceTransformer(EMBED_MODEL)

    mbox_files = find_mbox_files(MAIL_ROOT)
    log(LOG_PATH, f"FOUND mbox_files={len(mbox_files)}")

    total_chunks = 0
    for mbox_path in mbox_files:
        try:
            mtime, size = file_stat(mbox_path)
            old = manifest.get(mbox_path)
            if old and old.mtime == mtime and old.size == size:
                continue

            folder_rel = os.path.relpath(mbox_path, MAIL_ROOT)

            mbox = mailbox.mbox(mbox_path)
            # index each message
            records = []
            for i, msg in enumerate(mbox):
                try:
                    mid = (msg.get("Message-ID") or "").strip()
                    subj = decode_mime(msg.get("Subject") or "")
                    frm = decode_mime(msg.get("From") or "")
                    to = decode_mime(msg.get("To") or "")
                    date = decode_mime(msg.get("Date") or "")

                    body = extract_body(msg)
                    if len((body or "").strip()) < MIN_TEXT_CHARS:
                        continue

                    att = extract_attachments_meta(msg)

                    header_block = f"Subject: {subj}\nFrom: {frm}\nTo: {to}\nDate: {date}\nMessage-ID: {mid}\n"
                    full_text = (header_block + "\n" + body).strip()

                    chunks = chunk_text(full_text, CHUNK_SIZE, OVERLAP)
                    if not chunks:
                        continue

                    # stable mail id
                    stable = mid if mid else f"{folder_rel}:{i}"
                    sha = re.sub(r"[^a-zA-Z0-9_-]+", "_", stable)[:120]

                    for ci, ch in enumerate(chunks):
                        meta = {
                            "document_type": "email",
                            "mail_folder": folder_rel,
                            "message_id": mid,
                            "subject": subj,
                            "from": frm,
                            "to": to,
                            "date": date,
                            "chunk_index": ci,
                            "attachments": att,
                        }
                        chunk_id = f"mail:{sha}:{ci}"
                        records.append((chunk_id, ch, meta))
                except Exception:
                    continue

            # embed + upsert
            for start in range(0, len(records), BATCH_UPSERT):
                batch = records[start:start+BATCH_UPSERT]
                ids = [r[0] for r in batch]
                docs = [r[1] for r in batch]
                metas = [r[2] for r in batch]
                embs = embedder.encode(docs, convert_to_tensor=False, show_progress_bar=False).tolist()
                store.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                total_chunks += len(docs)

            # track mbox file itself
            sha1 = sha1_file(mbox_path)
            manifest.upsert(ManifestRow(path=mbox_path, sha1=sha1, mtime=mtime, size=size))

        except Exception:
            log(LOG_PATH, f"ERROR mbox={mbox_path}\n{traceback.format_exc()}")

    log(LOG_PATH, f"DONE total_chunks={total_chunks} count={store.count()}")

if __name__ == "__main__":
    main()
