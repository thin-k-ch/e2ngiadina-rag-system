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
        self.collection_txt = os.getenv("COLLECTION_TXT", "documents_txt")
        self.collection_msg = os.getenv("COLLECTION_MSG", "documents_msg")
        self.collection_mail = os.getenv("COLLECTION_MAIL_EWS", "documents_mail_ews")
        self.top_k = int(os.getenv("TOP_K", "10"))
        self.runner_url = os.getenv("PYRUNNER_URL", "http://runner:9000/run")
        self.embedder = SentenceTransformer(self.embed_model_name)
        self.chroma = ChromaClient(self.chroma_path, self.collection)
        self.chroma_docx = ChromaClient(self.chroma_path, self.collection_docx)
        self.chroma_txt = ChromaClient(self.chroma_path, self.collection_txt)
        self.chroma_msg = ChromaClient(self.chroma_path, self.collection_msg)
        self.chroma_mail = ChromaClient(self.chroma_path, self.collection_mail)

    def search_chunks(self, query: str, top_k: int | None = None):
        k = top_k or self.top_k
        emb = self.embedder.encode(query, convert_to_tensor=False).tolist()
        
        # Suche in allen Collections (PDFs, DOCXs, TXTs, MSGs)
        res_pdf = self.chroma.search(emb, top_k=k)
        res_docx = self.chroma_docx.search(emb, top_k=k)
        res_txt = self.chroma_txt.search(emb, top_k=k)
        res_msg = self.chroma_msg.search(emb, top_k=k)
        
        # Ergebnisse kombinieren
        docs = (res_pdf.get("documents", [[]])[0] + 
                res_docx.get("documents", [[]])[0] + 
                res_txt.get("documents", [[]])[0] + 
                res_msg.get("documents", [[]])[0])
        metas = (res_pdf.get("metadatas", [[]])[0] + 
                 res_docx.get("metadatas", [[]])[0] + 
                 res_txt.get("metadatas", [[]])[0] + 
                 res_msg.get("metadatas", [[]])[0])
        ids = (res_pdf.get("ids", [[]])[0] + 
               res_docx.get("ids", [[]])[0] + 
               res_txt.get("ids", [[]])[0] + 
               res_msg.get("ids", [[]])[0])
        dists = (res_pdf.get("distances", [[]])[0] + 
                 res_docx.get("distances", [[]])[0] + 
                 res_txt.get("distances", [[]])[0] + 
                 res_msg.get("distances", [[]])[0])
        
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

    def search_chunks_multi(self, queries: list[str], top_k_each: int = 6, max_total: int = 12):
        seen = set()
        merged = []
        for q in queries:
            hits = self.search_chunks(q, top_k=top_k_each)
            for h in hits:
                hid = h.get("id")
                if not hid or hid in seen:
                    continue
                seen.add(hid)
                merged.append(h)
        # sort by distance (lower is better); None goes last
        merged.sort(key=lambda x: (x.get("distance") is None, x.get("distance", 999999)))
        return merged[:max_total]

    def search_mail(self, query: str, top_k: int = 12):
        """Search only in email collection"""
        emb = self.embedder.encode(query, convert_to_tensor=False).tolist()
        res = self.chroma_mail.search(emb, top_k=top_k)
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
                "metadata": metas[i]
            })
        return out

    def filter_by_people(self, hits, sender=None, recipient=None):
        """Filter email hits by sender and/or recipient"""
        def norm(s): return (s or "").lower()
        sN = norm(sender)
        rN = norm(recipient)
        out = []
        for h in hits:
            md = h.get("metadata") or {}
            f = norm(md.get("email_from"))
            t = norm(md.get("email_to"))
            ok = True
            if sN: ok = ok and (sN in f)
            if rN: ok = ok and (rN in t)
            if ok: out.append(h)
        return out

    def search_multi(self, queries: list[str], top_k_each: int = 8, max_total: int = 24):
        seen=set()
        merged=[]
        for q in queries:
            for h in self.search_chunks(q, top_k=top_k_each):
                hid=h.get("id")
                if not hid or hid in seen:
                    continue
                seen.add(hid)
                merged.append(h)
        return merged[:max_total]

    def filter_must_include(self, hits: list, must_include: list[str]):
        if not must_include:
            return hits
        mi=[m.lower() for m in must_include if m]
        out=[]
        for h in hits:
            t=(h.get("text") or "").lower()
            md=h.get("metadata") or {}
            p=(md.get("original_path") or md.get("file_path") or "").lower()
            ok=True
            for m in mi:
                if m not in t and m not in p:
                    ok=False; break
            if ok:
                out.append(h)
        return out

    async def python_exec(self, code: str, locals: dict | None = None):
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(self.runner_url, json={"code": code, "locals": locals or {}})
            r.raise_for_status()
            return r.json()
