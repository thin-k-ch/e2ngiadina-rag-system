import os
from elasticsearch import Elasticsearch, helpers
import chromadb

ES_URL = os.getenv("ES_URL", "http://elasticsearch:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_chunks_v1")
CHROMA_PATH = os.getenv("CHROMA_PATH", "/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")
BATCH = int(os.getenv("ES_BATCH", "1000"))

def derive_project(path: str) -> str:
    # simple: first folder name
    if not path:
        return ""
    p = path.replace("\\","/").strip("/")
    return p.split("/")[0] if "/" in p else ""

def main():
    es = Elasticsearch(ES_URL)
    client = chromadb.PersistentClient(CHROMA_PATH)
    col = client.get_collection(CHROMA_COLLECTION)

    total = col.count()
    print("Chroma count:", total)

    # Chroma pagination: get all ids in chunks
    offset = 0
    indexed = 0

    while offset < total:
        # Chroma get supports limit/offset
        res = col.get(include=["documents","metadatas"], limit=BATCH, offset=offset)
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        ids = res.get("ids", [])

        actions = []
        for i in range(len(ids)):
            cid = ids[i]
            text = docs[i] or ""
            md = metas[i] or {}
            path = md.get("original_path") or md.get("file_path") or ""
            dt = md.get("document_type") or md.get("type") or ""
            chunk_index = md.get("chunk_index", 0)
            project = md.get("project") or derive_project(path)
            folder = md.get("mail_folder") or md.get("folder") or ""

            # Carry over metadata fields from Chroma (author, title, email headers, etc.)
            meta_fields = {}
            for mk in ("author", "title", "subject", "creation_date", "modified_date",
                        "last_modified_by", "producer", "page_count",
                        "email_from", "email_to", "email_cc", "email_subject", "email_date"):
                if md.get(mk):
                    meta_fields[mk] = md[mk]

            actions.append({
              "_op_type": "index",
              "_index": ES_INDEX,
              "_id": cid,
              "chunk_id": cid,
              "text": text,
              "original_path": path,
              "path_text": path,
              "document_type": dt,
              "project": project,
              "folder": folder,
              "chunk_index": int(chunk_index) if str(chunk_index).isdigit() else 0,
              **meta_fields
            })

        if actions:
            helpers.bulk(es, actions, request_timeout=120)
            indexed += len(actions)

        offset += BATCH
        print("indexed", indexed, "/", total)

    print("DONE indexed", indexed)

if __name__ == "__main__":
    main()
