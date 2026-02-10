import os
import httpx
import re
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .chroma_client import ChromaClient
from .tools_es import ESTools
from .config_rag import (
    RAG_FILES_INDICES,
    ES_CONTENT_FIELD,
    EXACT_TRIGGERS,
    SEARCH_TRIGGERS,
    INTERNAL_TRIGGERS,
    STOP,
)

@dataclass
class Gate:
    require_rag: bool
    mode: str            # "exact_phrase" | "hybrid" | "no_rag"
    phrase: str | None   # for exact_phrase
    reason: str

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
        self.es = ESTools()

    def _quoted(self, q: str) -> str | None:
        m = re.search(r"\"([^\"]+)\"", q)
        return m.group(1) if m else None

    def decide_gate(self, user_query: str) -> Gate:
        q = " ".join(user_query.strip().split())
        exact = any(re.search(p, q, re.I) for p in EXACT_TRIGGERS)
        no_rag = any(re.search(p, q, re.I) for p in [
            r"\bbrainstorm\b",
            r"\bidee\b",
            r"\bkreativ\b",
            r"\bschreib\b", 
            r"\btext\b.*\bschreib\b",
            r"\bpython\b.*\bskill\b",
            r"\breines schreiben\b",
            r"\bohne quellen\b",
        ])

        phrase = self._quoted(q)
        
        # If exact trigger found but no quotes, treat whole query as phrase
        if exact and not phrase:
            phrase = q

        # HARD RULE 1: explicit no_rag request (brainstorm, pure writing)
        if no_rag:
            return Gate(False, "no_rag", None, "Pure writing/brainstorming - no evidence needed")

        # HARD RULE 2: exact request => exact_phrase mode
        if exact:
            return Gate(True, "exact_phrase", phrase or q, "Exact phrase requested")

        # DEFAULT: ALWAYS RAG (hybrid mode)
        return Gate(True, "hybrid", None, "Always search for evidence")

    def _get(self, d: dict, path: str, default=None):
        cur = d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def _es_to_hits(self, es_resp: Dict[str, Any], *, phrase: Optional[str] = None, exact_level: str = "bm25") -> Tuple[List[Dict[str, Any]], int]:
        hits = []
        total = self._get(es_resp, "hits.total.value", 0) or 0
        hits_list = self._get(es_resp, "hits.hits", []) or []
        print(f"🔍 ES_TO_HITS DEBUG: total={total}, hits_list len={len(hits_list)}")
        if hits_list:
            print(f"🔍 ES_TO_HITS DEBUG: first hit keys={list(hits_list[0].keys())}")
        for h in hits_list:
            src = h.get("_source", {}) or {}
            highlight = (h.get("highlight", {}) or {}).get(ES_CONTENT_FIELD, [])
            snippet = " ".join(highlight) if highlight else ""

            path = self._get(src, "meta.real.path", None) or self._get(src, "path.real", None)
            filename = self._get(src, "file.filename", None)
            ext = self._get(src, "file.extension", None)

            # Determine "exact_match" in our no-reindex constraints:
            exact_match = False
            if exact_level == "phrase":
                # If highlight contains literal phrase, that's strongest.
                if phrase and snippet and phrase in snippet:
                    exact_match = True
                else:
                    # still treat phrase query hit as exact-ish (phrase-level)
                    exact_match = True

            hits.append({
                "source": "es",
                "doc_id": h.get("_id"),
                "score": float(h.get("_score") or 0.0),
                "file": {
                    "filename": filename,
                    "extension": ext,
                    "path": path,
                },
                "snippet": snippet,
                "exact_level": exact_level,     # "phrase" | "and_fallback" | "bm25"
                "exact_match": exact_match,
                "raw": {"_index": h.get("_index")}
            })
        return hits, total

    def search_exact_phrase(self, phrase: str, *, size: int = 10, indices: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Runs exact phrase search against rag_files_v1 content with slop=0.
        If 0 hits, runs AND fallback once.
        Returns unified dict with rounds and best hits.
        """
        print(f" EXACT PHRASE SEARCH: '{phrase}'")
        
        idxs = indices or RAG_FILES_INDICES

        # Round 1: match_phrase slop=0
        resp1_raw = self.es.es_exact_phrase_content(phrase, indices=idxs, size=size)
        resp1 = dict(resp1_raw) if hasattr(resp1_raw, '__iter__') else {}
        hits1, total1 = self._es_to_hits(resp1, phrase=phrase, exact_level="phrase")
        print(f" ES EXACT PHRASE: {total1} hits")

        rounds = [{"kind": "phrase", "total": total1, "hits": hits1}]

        # If hits exist, stop immediately (deterministic)
        if total1 > 0:
            return {
                "mode": "exact_phrase",
                "phrase": phrase,
                "indices": idxs,
                "rounds": rounds,
                "best_hits": hits1,
                "total_hits": total1,
            }

        # Round 2 fallback: strict AND match (still no fuzziness)
        resp2_raw = self.es.es_exact_fallback_and(phrase, indices=idxs, size=size)
        resp2 = dict(resp2_raw) if hasattr(resp2_raw, '__iter__') else {}
        hits2, total2 = self._es_to_hits(resp2, phrase=phrase, exact_level="and_fallback")
        print(f"📊 ES AND FALLBACK: {total2} hits")
        rounds.append({"kind": "and_fallback", "total": total2, "hits": hits2})

        # Round 3: Chroma vector search (semantic similarity for phrase)
        chroma_hits = self.search_chunks(phrase, top_k=10)
        print(f"📊 CHROMA PHRASE: {len(chroma_hits)} hits")
        if chroma_hits:
            rounds.append({"kind": "chroma_phrase", "total": len(chroma_hits), "hits": chroma_hits})
            # Merge ES and Chroma results
            merged_hits = self._dedup_merge(hits2, chroma_hits)
            return {
                "mode": "exact_phrase",
                "phrase": phrase,
                "indices": idxs,
                "rounds": rounds,
                "best_hits": merged_hits[:size],
                "total_hits": len(merged_hits),
            }

        return {
            "mode": "exact_phrase",
            "phrase": phrase,
            "indices": idxs,
            "rounds": rounds,
            "best_hits": hits2,
            "total_hits": total2,
        }

    def _dedup_merge(self, es_hits: List[Dict[str, Any]], chroma_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Key: prefer real path, else ES id
        def key(h):
            p = ((h.get("file") or {}).get("path") or "").strip()
            return p if p else f"{h.get('source')}:{h.get('doc_id')}"

        merged = {}
        for h in es_hits + chroma_hits:
            k = key(h)
            if k not in merged:
                merged[k] = h
            else:
                # keep the higher score within same source; if mix, prefer ES snippet if present
                if (h.get("score", 0) or 0) > (merged[k].get("score", 0) or 0):
                    merged[k] = h
                elif (merged[k].get("snippet") or "") == "" and (h.get("snippet") or ""):
                    merged[k]["snippet"] = h["snippet"]
        return list(merged.values())

    def search_hybrid(
        self,
        query: str,
        *,
        es_size: int = 50,
        chroma_k: int = 30,
        indices: Optional[List[str]] = None,
        ext_filter: Optional[List[str]] = None,
        chroma_queries: Optional[List[str]] = None,
        fuzzy_rerank_fn=None,  # optional callable: (query, hits)->hits
    ) -> Dict[str, Any]:
        """
        Hybrid search: ES BM25 + Chroma vector. Merge+dedup. Optional fuzzy rerank AFTER merge.
        """
        print(f"🔍 HYBRID SEARCH: {query}")
        
        idxs = indices or RAG_FILES_INDICES

        # ES BM25
        es_resp_raw = self.es.es_bm25_search_content(query, indices=idxs, size=es_size, ext_filter=ext_filter)
        # Convert ObjectApiResponse to dict
        es_resp = dict(es_resp_raw) if hasattr(es_resp_raw, '__iter__') else {}
        es_hits, es_total = self._es_to_hits(es_resp, exact_level="bm25")
        print(f"📊 ES BM25: {es_total} hits")

        # Chroma multi-query (optional)
        cq = chroma_queries or [query]
        chroma_all = []
        for q in cq:
            # Use existing chroma search
            chroma_all.extend(self.search_chunks(q, top_k=chroma_k))
        print(f"📊 CHROMA: {len(chroma_all)} hits")

        merged = self._dedup_merge(es_hits, chroma_all)

        if fuzzy_rerank_fn is not None:
            merged = fuzzy_rerank_fn(query, merged)

        # Sort: ES bm25 score is raw; chroma score scale differs.
        def sort_key(h):
            src_boost = 1 if h.get("source") == "es" else 0
            return (src_boost, float(h.get("score") or 0.0))
        merged.sort(key=sort_key, reverse=True)

        print(f"🎯 HYBRID RESULT: {len(merged)} unique hits")

        return {
            "mode": "hybrid",
            "query": query,
            "indices": idxs,
            "es_total": es_total,
            "es_hits": es_hits,
            "chroma_hits": chroma_all,
            "merged_hits": merged,
        }

    def can_claim_absence(self, mode: str, es_exact_ran: bool, es_total_hits: int, round_idx: int) -> bool:
        if mode != "exact_phrase":
            return False
        if not es_exact_ran:
            return False
        if es_total_hits > 0:
            return False
        return round_idx >= STOP["max_rounds"]  # after rewrite/fallback round

    # Keep existing methods for compatibility
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
            # Convert Chroma format to legacy format expected by _build_sources
            path = (meta.get("original_path") or meta.get("file_path") or meta.get("path") or "") if meta else ""
            filename = (meta.get("filename") or meta.get("file", {}).get("filename") or "") if meta else ""
            out.append({
                "id": id_val,
                "distance": dist,
                "text": doc,
                "metadata": meta,
                # Legacy format fields for _build_sources
                "file": {"path": path, "filename": filename},
                "snippet": doc[:500] if doc else "",
                "score": 1.0 - (dist if dist else 0.0),  # Convert distance to score
                "source": "chroma",
            })
        return out

    async def python_exec(self, code: str, locals: dict | None = None):
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(self.runner_url, json={"code": code, "locals": locals or {}})
            r.raise_for_status()
            return r.json()
