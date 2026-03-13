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
        self.top_k = int(os.getenv("TOP_K", "10"))
        self.runner_url = os.getenv("PYRUNNER_URL", "http://runner:9000/run")
        self.embedder = SentenceTransformer(self.embed_model_name)
        self.es = ESTools()
        # Current tenant state
        self._current_chroma_path = None
        self._current_chroma_prefix = None
        # Init Chroma with defaults (will be overridden by configure_for_tenant)
        self._init_chroma(
            os.getenv("CHROMA_PATH", "/chroma"),
            os.getenv("COLLECTION", "documents"),
        )

    def _init_chroma(self, chroma_path: str, prefix: str):
        """(Re-)initialize all ChromaDB clients for a given path and collection prefix."""
        self._current_chroma_path = chroma_path
        self._current_chroma_prefix = prefix
        self.chroma_path = chroma_path
        self.collection = prefix
        self.collection_docx = f"{prefix}_docx"
        self.collection_txt = f"{prefix}_txt"
        self.collection_msg = f"{prefix}_msg"
        self.collection_mail = f"{prefix}_mail_ews"
        self.chroma = ChromaClient(chroma_path, self.collection)
        self.chroma_docx = ChromaClient(chroma_path, self.collection_docx)
        self.chroma_txt = ChromaClient(chroma_path, self.collection_txt)
        self.chroma_msg = ChromaClient(chroma_path, self.collection_msg)
        self.chroma_mail = ChromaClient(chroma_path, self.collection_mail)
        # OnePagers collection (tenant-specific naming: {prefix}_onepagers)
        onepagers_name = f"{prefix}_onepagers"
        try:
            self.chroma_onepagers = ChromaClient(chroma_path, onepagers_name)
            _oc = self.chroma_onepagers.collection.count()
            if _oc > 0:
                print(f"📄 OnePagers collection '{onepagers_name}': {_oc} entries")
                self._has_onepagers = True
            else:
                self._has_onepagers = False
        except Exception:
            self.chroma_onepagers = None
            self._has_onepagers = False
        # Findings collection (tenant-specific naming: {prefix}_findings)
        findings_name = f"{prefix}_findings"
        try:
            self.chroma_findings = ChromaClient(chroma_path, findings_name)
            _fc = self.chroma_findings.collection.count()
            if _fc > 0:
                print(f"📋 Findings collection '{findings_name}': {_fc} entries")
                self._has_findings = True
            else:
                self._has_findings = False
        except Exception:
            self.chroma_findings = None
            self._has_findings = False
        print(f"🗄️ Chroma: path={chroma_path}, prefix={prefix}, onepagers={'✅' if self._has_onepagers else '❌'}, findings={'✅' if self._has_findings else '❌'}")

    def configure_for_tenant(self, chroma_path: str, prefix: str):
        """Switch Chroma to a different tenant. Skips if already on the same tenant."""
        if chroma_path == self._current_chroma_path and prefix == self._current_chroma_prefix:
            return
        print(f"🔄 Tools: Switching Chroma → path={chroma_path}, prefix={prefix}")
        self._init_chroma(chroma_path, prefix)

    def search_findings(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search pre-computed findings collection. Returns list of finding dicts."""
        if not self._has_findings or not self.chroma_findings:
            return []
        try:
            emb = self.embedder.encode(query, convert_to_tensor=False).tolist()
            res = self.chroma_findings.search(emb, top_k=top_k)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            out = []
            for doc, meta, dist in zip(docs, metas, dists):
                if dist is not None and dist > 1.2:  # skip low-relevance
                    continue
                out.append({
                    "text": doc,
                    "title": meta.get("title", ""),
                    "category": meta.get("category", ""),
                    "impact": meta.get("impact", ""),
                    "evidence_docs": meta.get("evidence_docs", ""),
                    "distance": dist,
                    "source_type": "finding",
                })
            return out
        except Exception as e:
            print(f"⚠️ Findings search error: {e}")
            return []

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

            path = self._get(src, "path.virtual", None) or self._get(src, "meta.real.path", None) or self._get(src, "path.real", None)
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
        search_mode, _ = self._get_search_config()
        print(f"🔍 HYBRID SEARCH: {query} [mode={search_mode}]")
        
        idxs = indices or RAG_FILES_INDICES

        # ES BM25 (skip in onepagers_only mode)
        if search_mode == "onepagers_only":
            es_hits, es_total = [], 0
            print(f"📊 ES BM25: skipped (onepagers_only)")
        else:
            es_resp_raw = self.es.es_bm25_search_content(query, indices=idxs, size=es_size, ext_filter=ext_filter)
            # Convert ObjectApiResponse to dict
            es_resp = dict(es_resp_raw) if hasattr(es_resp_raw, '__iter__') else {}
            es_hits, es_total = self._es_to_hits(es_resp, exact_level="bm25")
            print(f"📊 ES BM25: {es_total} hits")

        # Chroma multi-query (optional)
        cq = chroma_queries or [query]
        chroma_all = []
        for q in cq:
            chroma_all.extend(self.search_chunks(q, top_k=chroma_k))
        print(f"📊 CHROMA: {len(chroma_all)} hits [mode={search_mode}]")

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

    def _get_search_config(self):
        """Read search_mode and onepager_boost from RuntimeConfig."""
        try:
            from .runtime_config import get_runtime_config
            cfg = get_runtime_config()
            mode = cfg.get("search_mode", "full_search")
            boost = float(cfg.get("onepager_boost", 0.7))
            return mode, max(0.3, min(1.0, boost))
        except Exception:
            return "full_search", 0.7

    def _collect_results(self, *result_sets):
        """Merge multiple Chroma search results into flat lists."""
        docs, metas, ids, dists = [], [], [], []
        for res in result_sets:
            if not res:
                continue
            docs.extend(res.get("documents", [[]])[0])
            metas.extend(res.get("metadatas", [[]])[0])
            ids.extend(res.get("ids", [[]])[0])
            dists.extend(res.get("distances", [[]])[0])
        return docs, metas, ids, dists

    def _to_legacy_hits(self, combined, k, source_tag="chroma"):
        """Convert (doc, meta, id, dist) tuples to legacy format."""
        out = []
        for doc, meta, id_val, dist in combined[:k]:
            path = (meta.get("original_path") or meta.get("file_path") or meta.get("path") or "") if meta else ""
            filename = (meta.get("filename") or meta.get("file", {}).get("filename") or "") if meta else ""
            is_onepager = (meta or {}).get("source_type") == "onepager"
            out.append({
                "id": id_val,
                "distance": dist,
                "text": doc,
                "metadata": meta,
                "file": {"path": path, "filename": filename},
                "snippet": doc[:500] if doc else "",
                "score": 1.0 - (dist if dist else 0.0),
                "source": source_tag,
                "is_onepager": is_onepager,
            })
        return out

    # Keep existing methods for compatibility
    def search_chunks(self, query: str, top_k: int | None = None):
        k = top_k or self.top_k
        search_mode, boost = self._get_search_config()
        emb = self.embedder.encode(query, convert_to_tensor=False).tolist()

        if search_mode == "onepagers_only":
            return self._search_onepagers_only(emb, k)
        elif search_mode == "onepagers_first":
            return self._search_onepagers_first(emb, k, boost)
        else:
            return self._search_full(emb, k, boost)

    def _search_full(self, emb, k, boost):
        """Full search: all collections, OnePager hits boosted by distance multiplier."""
        res_pdf = self.chroma.search(emb, top_k=k)
        res_docx = self.chroma_docx.search(emb, top_k=k)
        res_txt = self.chroma_txt.search(emb, top_k=k)
        res_msg = self.chroma_msg.search(emb, top_k=k)
        res_op = self.chroma_onepagers.search(emb, top_k=k) if self._has_onepagers else {}

        docs, metas, ids, dists = self._collect_results(res_pdf, res_docx, res_txt, res_msg)
        op_docs, op_metas, op_ids, op_dists = self._collect_results(res_op)

        # Apply boost to OnePager distances (lower distance = higher rank)
        op_dists_boosted = [d * boost if d is not None else 1.0 for d in op_dists]

        all_docs = docs + op_docs
        all_metas = metas + op_metas
        all_ids = ids + op_ids
        all_dists = dists + op_dists_boosted

        combined = list(zip(all_docs, all_metas, all_ids, all_dists))
        combined.sort(key=lambda x: x[3] if x[3] is not None else 1.0)
        return self._to_legacy_hits(combined, k)

    def _search_onepagers_only(self, emb, k):
        """Only search OnePager summaries."""
        if not self._has_onepagers:
            return []
        res_op = self.chroma_onepagers.search(emb, top_k=k)
        docs, metas, ids, dists = self._collect_results(res_op)
        combined = list(zip(docs, metas, ids, dists))
        combined.sort(key=lambda x: x[3] if x[3] is not None else 1.0)
        return self._to_legacy_hits(combined, k, source_tag="onepager")

    def _search_onepagers_first(self, emb, k, boost):
        """Two-phase: OnePagers first, then detail chunks from relevant documents."""
        # Phase 1: Search OnePagers
        if self._has_onepagers:
            res_op = self.chroma_onepagers.search(emb, top_k=k)
            op_docs, op_metas, op_ids, op_dists = self._collect_results(res_op)
        else:
            op_docs, op_metas, op_ids, op_dists = [], [], [], []

        # Phase 2: Search raw document chunks
        res_pdf = self.chroma.search(emb, top_k=k)
        res_docx = self.chroma_docx.search(emb, top_k=k)
        res_txt = self.chroma_txt.search(emb, top_k=k)
        res_msg = self.chroma_msg.search(emb, top_k=k)
        raw_docs, raw_metas, raw_ids, raw_dists = self._collect_results(res_pdf, res_docx, res_txt, res_msg)

        # Boost OnePager distances
        op_dists_boosted = [d * boost if d is not None else 1.0 for d in op_dists]

        # Combine: OnePagers (boosted) + raw chunks
        all_docs = op_docs + raw_docs
        all_metas = op_metas + raw_metas
        all_ids = op_ids + raw_ids
        all_dists = op_dists_boosted + raw_dists

        combined = list(zip(all_docs, all_metas, all_ids, all_dists))
        combined.sort(key=lambda x: x[3] if x[3] is not None else 1.0)

        # Ensure at least half the results are OnePagers (if available)
        op_count = sum(1 for _, m, _, _ in combined[:k] if (m or {}).get("source_type") == "onepager")
        if op_count < k // 2 and len(op_docs) >= k // 2:
            # Force top OnePagers into results
            op_combined = list(zip(op_docs, op_metas, op_ids, op_dists_boosted))
            op_combined.sort(key=lambda x: x[3] if x[3] is not None else 1.0)
            raw_combined = list(zip(raw_docs, raw_metas, raw_ids, raw_dists))
            raw_combined.sort(key=lambda x: x[3] if x[3] is not None else 1.0)
            combined = op_combined[:k // 2] + raw_combined[:k - k // 2]
            combined.sort(key=lambda x: x[3] if x[3] is not None else 1.0)

        return self._to_legacy_hits(combined, k)

    async def python_exec(self, code: str, locals: dict | None = None):
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(self.runner_url, json={"code": code, "locals": locals or {}})
            r.raise_for_status()
            return r.json()
