import os
import re
import hashlib
from elasticsearch import Elasticsearch
import chromadb
from sentence_transformers import SentenceTransformer

ES_URL = os.getenv("ES_URL","http://elasticsearch:9200")
ES_INDEX = os.getenv("ES_INDEX","rag_files_v1")

CHROMA_PATH = os.getenv("CHROMA_PATH","/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION","documents_from_es")

EMBED_MODEL = os.getenv("EMBED_MODEL","all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE","1200"))
OVERLAP = int(os.getenv("CHUNK_OVERLAP","180"))

MAX_DOCS = int(os.getenv("MAX_DOCS","0"))  # 0 = no limit
BATCH = int(os.getenv("BATCH","64"))

def chunk_text(text: str, size: int, overlap: int):
    text = (text or "").strip()
    if not text:
        return []
    out=[]
    i=0
    n=len(text)
    step=max(1, size-overlap)
    while i < n:
        out.append(text[i:i+size])
        i += step
    return out

def stable_id(path: str, chunk_index: int):
    h = hashlib.sha1(path.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"es:{h}:{chunk_index}"

def normalize_path(p: str):
    return (p or "").replace("\\","/").strip()

def main():
    es = Elasticsearch(ES_URL)
    client = chromadb.PersistentClient(CHROMA_PATH)
    try:
        col = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        col = client.create_collection(CHROMA_COLLECTION)

    embedder = SentenceTransformer(EMBED_MODEL)

    # PIT for stable paging
    pit = es.open_point_in_time(index=ES_INDEX, keep_alive="5m")["id"]
    search_after = None
    processed_docs = 0

    try:
        while True:
            body = {
              "size": 200,
              "sort": [{"_shard_doc": "asc"}],
              "pit": {"id": pit, "keep_alive": "5m"},
              "_source": True,
              "query": {"match_all": {}}
            }
            if search_after is not None:
                body["search_after"] = search_after

            res = es.search(body=body, request_timeout=120)
            hits = res["hits"]["hits"]
            if not hits:
                break

            for h in hits:
                src = h.get("_source") or {}
                # FSCrawler typically stores extracted text in "content"
                text = src.get("content") or ""
                # path can be in multiple shapes depending on fscrawler version
                path = (
                  (src.get("path") or {}).get("real") if isinstance(src.get("path"), dict) else src.get("path")
                ) or src.get("file", {}).get("filename") or ""
                path = normalize_path(path)

                # Basic quality gate (avoid binary/empty)
                if not text or len(text.strip()) < 200:
                    continue

                chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
                if not chunks:
                    continue

                metas=[]
                ids=[]
                docs=[]
                for ci, ch in enumerate(chunks):
                    ids.append(stable_id(path or h["_id"], ci))
                    docs.append(ch)
                    metas.append({
                      "original_path": path or h["_id"],
                      "document_type": "from_es",
                      "es_index": ES_INDEX,
                      "es_id": h["_id"],
                      "chunk_index": ci,
                    })

                    if len(docs) >= BATCH:
                        embs = embedder.encode(docs, convert_to_tensor=False, show_progress_bar=False).tolist()
                        col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                        ids, docs, metas = [], [], []

                if docs:
                    embs = embedder.encode(docs, convert_to_tensor=False, show_progress_bar=False).tolist()
                    col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

                processed_docs += 1
                if MAX_DOCS and processed_docs >= MAX_DOCS:
                    print("Reached MAX_DOCS, stopping.")
                    return

            search_after = hits[-1]["sort"]
            print("paged, processed_docs=", processed_docs)

    finally:
        try:
            es.close_point_in_time(body={"id": pit})
        except Exception:
            pass

if __name__ == "__main__":
    main()
# NICHT AUSFÜHREN – nur commit/ablage.
# (Morgen/Später: wir machen dafür einen kontrollierten Run + Audit.)
