import os, re, sqlite3, traceback, base64
from sentence_transformers import SentenceTransformer
from .chunking import chunk_text
from .chroma_store import ChromaStore
from .manifest import Manifest, ManifestRow
from .hashing import file_stat, sha1_file

def log(path, msg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip()+"\n")

def is_textish(colname: str) -> bool:
    c = colname.lower()
    return any(k in c for k in ["body", "text", "html", "plain", "content", "snippet"])

def pick_best_text_column(cols):
    # prioritize html/plain/body/content
    prefs = ["body", "plain", "text", "content", "html"]
    lc = [c.lower() for c in cols]
    for p in prefs:
        for i,c in enumerate(lc):
            if p in c:
                return cols[i]
    # fallback: first "textish"
    for c in cols:
        if is_textish(c):
            return c
    return None

def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, bytes):
        # try utf-8, else latin-1
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return x.decode("latin-1", errors="ignore")
    return str(x)

def normalize_body(s: str) -> str:
    s = s or ""
    # strip huge runs of binary-ish garbage
    s = s.replace("\x00"," ")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def extract_email_metadata(properties_xml: str) -> dict:
    """Extract email metadata from base64-encoded MIME content in properties XML"""
    metadata = {
        "email_from": "",
        "email_to": "",
        "email_cc": "",
        "email_subject": "",
        "email_date": "",
        "mail_folder": ""
    }
    
    if not properties_xml:
        return metadata
    
    try:
        # Extract MimeContent from XML
        mime_match = re.search(r'<MimeContent[^>]*>([^<]+)</MimeContent>', properties_xml)
        if not mime_match:
            return metadata
            
        mime_b64 = mime_match.group(1)
        mime_decoded = base64.b64decode(mime_b64).decode('utf-8', errors='ignore')
        
        # Parse email headers
        lines = mime_decoded.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('From:'):
                metadata["email_from"] = line[5:].strip()
            elif line.startswith('To:'):
                metadata["email_to"] = line[3:].strip()
            elif line.startswith('Cc:'):
                metadata["email_cc"] = line[3:].strip()
            elif line.startswith('Subject:'):
                metadata["email_subject"] = line[8:].strip()
            elif line.startswith('Date:'):
                metadata["email_date"] = line[5:].strip()
                
    except Exception as e:
        # Use global log path or create a temporary one
        log_path = os.getenv("LOG_PATH", "/logs/indexer_ews_mail.log")
        log(log_path, f"MIME parsing error: {e}")
    
    return metadata

def main():
    ROOT = os.getenv("ROOT_DIR", "/data")  # unused
    BODIES_DB = os.getenv("EWS_BODIES_DB", "/mail/ews-bodies.sqlite")
    META_DB = os.getenv("EWS_META_DB", "/mail/ews-db.sqlite")
    CHROMA_PATH = os.getenv("CHROMA_PATH", "/chroma")
    COLLECTION = os.getenv("COLLECTION_MAIL_EWS", "documents_mail_ews")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE","1200"))
    OVERLAP = int(os.getenv("CHUNK_OVERLAP","180"))
    MANIFEST_PATH = os.getenv("MANIFEST_PATH","/manifest/manifest.sqlite3")
    LOG_PATH = os.getenv("LOG_PATH","/logs/indexer_ews_mail.log")
    BATCH_UPSERT = int(os.getenv("BATCH_UPSERT","256"))
    MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS","200"))

    log(LOG_PATH, f"START EWS sqlite indexer bodies={BODIES_DB} meta={META_DB} collection={COLLECTION} model={EMBED_MODEL}")

    # manifest keyed by DB file mtime/size (reindex only if DB changed)
    manifest = Manifest(MANIFEST_PATH)
    mtime, size = file_stat(BODIES_DB)
    old = manifest.get(BODIES_DB)
    if old and old.mtime == mtime and old.size == size:
        log(LOG_PATH, "SKIP: sqlite unchanged (manifest hit)")
        return

    store = ChromaStore(CHROMA_PATH, COLLECTION)
    embedder = SentenceTransformer(EMBED_MODEL)

    # Connect to both databases
    bodies_con = sqlite3.connect(BODIES_DB)
    bodies_con.row_factory = sqlite3.Row
    bodies_cur = bodies_con.cursor()
    
    meta_con = sqlite3.connect(META_DB)
    meta_con.row_factory = sqlite3.Row
    meta_cur = meta_con.cursor()

    # Get bodies table
    tables = [r[0] for r in bodies_cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    log(LOG_PATH, f"TABLES={len(tables)}")

    total_chunks = 0
    total_rows_used = 0

    for t in tables:
        try:
            # columns
            cols = [r["name"] for r in bodies_cur.execute(f"PRAGMA table_info('{t}')").fetchall()]
            if not cols:
                continue

            text_col = pick_best_text_column(cols)
            if not text_col:
                continue

            # pick an id-ish column for stable ids
            id_col = None
            for c in cols:
                cl = c.lower()
                if cl in ("id","rowid","msgid","messageid","message_id","key","uid","itemid"):
                    id_col = c
                    break
            if not id_col:
                # fallback: first integer column
                for r in bodies_cur.execute(f"PRAGMA table_info('{t}')").fetchall():
                    if (r["type"] or "").lower().startswith("int"):
                        id_col = r["name"]
                        break

            # count quickly; skip tiny tables
            try:
                n = bodies_cur.execute(f"SELECT count(*) AS c FROM '{t}'").fetchone()["c"]
            except Exception:
                n = None
            if n is not None and n < 5:
                continue

            log(LOG_PATH, f"SCAN table={t} rows={n} text_col={text_col} id_col={id_col}")

            # fetch rows with JOIN to metadata (using separate connections)
            q = f"SELECT * FROM '{t}'"
            for row in bodies_cur.execute(q):
                raw = row[text_col]
                body = normalize_body(safe_str(raw))
                if len(body) < MIN_TEXT_CHARS:
                    continue

                # Get metadata from separate database
                item_id = row["itemId"] if "itemId" in row.keys() else ""
                properties = ""
                item_class = ""
                folder_id = ""
                
                if item_id:
                    meta_cur.execute('SELECT properties, itemClass, folderId FROM itemMetadata WHERE itemId = ?', (item_id,))
                    meta_row = meta_cur.fetchone()
                    if meta_row:
                        properties = meta_row["properties"] or ""
                        item_class = meta_row["itemClass"] or ""
                        folder_id = meta_row["folderId"] or ""

                # Extract metadata from properties
                email_meta = extract_email_metadata(properties)
                
                # ONLY INDEX EMAILS WITH VALID METADATA OR WITH BODY
                # This ensures we index both emails with proper metadata and emails with just content
                if not (email_meta["email_from"] and email_meta["email_to"]) and len(body) < MIN_TEXT_CHARS:
                    continue  # Skip emails without metadata AND insufficient body content
                
                # build metadata with extracted email info
                meta = {
                    "document_type": "email",
                    "source": "exquilla_ews_sqlite",
                    "table": t,
                    "text_column": text_col,
                    # Extracted email metadata
                    "email_from": email_meta["email_from"],
                    "email_to": email_meta["email_to"],
                    "email_cc": email_meta["email_cc"],
                    "email_subject": email_meta["email_subject"],
                    "email_date": email_meta["email_date"],
                    "mail_folder": email_meta["mail_folder"],
                    # Additional metadata
                    "item_id": safe_str(item_id),
                    "item_class": safe_str(item_class),
                    "folder_id": safe_str(folder_id),
                }

                # stable base id - use itemId directly
                base = safe_str(row["itemId"]) if "itemId" in row.keys() else f"{t}:{total_rows_used}"

                base = re.sub(r"[^a-zA-Z0-9_-]+", "_", base)[:120]
                total_rows_used += 1

                chunks = chunk_text(body, CHUNK_SIZE, OVERLAP)
                if not chunks:
                    continue

                # embed + upsert in batches
                for start in range(0, len(chunks), BATCH_UPSERT):
                    batch = chunks[start:start+BATCH_UPSERT]
                    ids = [f"ews:{t}:{base}:{start+i}" for i in range(len(batch))]
                    metas = []
                    for i in range(len(batch)):
                        md = dict(meta)
                        md["chunk_index"] = start+i
                        metas.append(md)
                    embs = embedder.encode(batch, convert_to_tensor=False, show_progress_bar=False).tolist()
                    store.upsert(ids=ids, documents=batch, metadatas=metas, embeddings=embs)
                    total_chunks += len(batch)

        except Exception:
            log(LOG_PATH, f"ERROR table={t}\n{traceback.format_exc()}")

    # close connections
    bodies_con.close()
    meta_con.close()

    # mark manifest for DB file
    sha = sha1_file(BODIES_DB)
    manifest.upsert(ManifestRow(path=BODIES_DB, sha1=sha, mtime=mtime, size=size))

    log(LOG_PATH, f"DONE rows_used={total_rows_used} total_chunks={total_chunks} count={store.count()}")

if __name__ == "__main__":
    main()
