import os
import httpx
from sentence_transformers import SentenceTransformer
from .chroma_client import ChromaClient

class Tools:
    def __init__(self):
        self.embed_model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        self.chroma_path = os.getenv("CHROMA_PATH", "/chroma")
        self.collection = os.getenv("COLLECTION", "documents")
        self.collection_docx = os.getenv("COLLECTION_DOCX", "documents_docx")
        self.top_k = int(os.getenv("TOP_K", "10"))
        self.runner_url = os.getenv("PYRUNNER_URL", "http://runner:9000/run")
        self.embedder = SentenceTransformer(self.embed_model_name)
        self.chroma = ChromaClient(self.chroma_path, self.collection)
        self.chroma_docx = ChromaClient(self.chroma_path, self.collection_docx)

    def search_chunks(self, query: str, top_k: int | None = None):
        k = top_k or self.top_k
        emb = self.embedder.encode(query, convert_to_tensor=False).tolist()
        
        # Suche in beiden Collections (PDFs und DOCXs)
        res_pdf = self.chroma.search(emb, top_k=k)
        res_docx = self.chroma_docx.search(emb, top_k=k)
        
        # Ergebnisse kombinieren
        docs = res_pdf.get("documents", [[]])[0] + res_docx.get("documents", [[]])[0]
        metas = res_pdf.get("metadatas", [[]])[0] + res_docx.get("metadatas", [[]])[0]
        ids = res_pdf.get("ids", [[]])[0] + res_docx.get("ids", [[]])[0]
        dists = res_pdf.get("distances", [[]])[0] + res_docx.get("distances", [[]])[0]
        
        # Nach Distanz sortieren (bessere Ergebnisse zuerst)
        combined = list(zip(docs, metas, ids, dists))
        combined.sort(key=lambda x: x[3] if x[3] is not None else 1.0)
        
        out = []
        for i, (doc, meta, id_val, dist) in enumerate(combined[:k]):
            out.append({
                "id": id_val,
                "distance": dist,
                "text": doc,
                "metadata": meta,
            })
        return out

    async def python_exec(self, code: str, locals: dict | None = None):
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(self.runner_url, json={"code": code, "locals": locals or {}})
            r.raise_for_status()
            return r.json()
