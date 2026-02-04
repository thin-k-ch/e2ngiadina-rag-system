import os
import httpx
from sentence_transformers import SentenceTransformer
from .chroma_client import ChromaClient

class Tools:
    def __init__(self):
        self.embed_model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        self.chroma_path = os.getenv("CHROMA_PATH", "/chroma")
        self.collection = os.getenv("COLLECTION", "documents")
        self.top_k = int(os.getenv("TOP_K", "10"))
        self.runner_url = os.getenv("PYRUNNER_URL", "http://runner:9000/run")
        self.embedder = SentenceTransformer(self.embed_model_name)
        self.chroma = ChromaClient(self.chroma_path, self.collection)

    def search_chunks(self, query: str, top_k: int | None = None):
        k = top_k or self.top_k
        emb = self.embedder.encode(query, convert_to_tensor=False).tolist()
        res = self.chroma.search(emb, top_k=k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out = []
        for i in range(len(docs)):
            out.append({
                "id": ids[i],
                "distance": dists[i] if i < len(dists) else None,
                "text": docs[i],
                "metadata": metas[i],
            })
        return out

    async def python_exec(self, code: str, locals: dict | None = None):
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(self.runner_url, json={"code": code, "locals": locals or {}})
            r.raise_for_status()
            return r.json()
