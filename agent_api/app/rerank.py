from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

def _tok(s: str):
    return [t for t in (s or "").lower().split() if t]

def rerank(query: str, hits: list, top_n: int = 8):
    # hits: [{"text":..., "metadata":..., "distance":...}]
    texts = [(h.get("text") or "") for h in hits]
    corpus = [_tok(t) for t in texts]
    if not corpus:
        return hits[:top_n]

    bm25 = BM25Okapi(corpus)
    qtok = _tok(query)
    scores = bm25.get_scores(qtok)

    ranked = []
    for i,h in enumerate(hits):
        md = h.get("metadata") or {}
        path = (md.get("original_path") or md.get("file_path") or "")
        # fuzzy boost for filenames/paths containing keywords
        fb = fuzz.partial_ratio(query.lower(), path.lower()) / 100.0
        # combine: bm25 + small fuzzy boost - small distance penalty
        dist = h.get("distance")
        dist_pen = 0.0 if dist is None else min(0.5, float(dist) / 4.0)
        score = float(scores[i]) + 0.3*fb - 0.2*dist_pen
        ranked.append((score, h))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [h for _,h in ranked[:top_n]]
