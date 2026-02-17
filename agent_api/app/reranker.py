"""
Cross-Encoder Reranking
========================

Uses a lightweight cross-encoder model to rerank search results
by semantic relevance to the query. Runs on CPU/GPU (~1 GB).

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- 22M params, ~90 MB on disk
- Designed for passage reranking
- Language-agnostic enough for German text

The model is loaded lazily on first use and cached for subsequent calls.
"""

import os
import time
from typing import List, Dict, Any

# Module-level singleton – loaded once, reused across requests
_cross_encoder = None
_MODEL_NAME = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "1") == "1"
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "20"))  # How many hits to rerank (rest kept in original order)


def _get_model():
    """Lazy-load the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        print(f"🔄 Loading Cross-Encoder model: {_MODEL_NAME}...")
        start = time.time()
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(_MODEL_NAME)
        elapsed = time.time() - start
        print(f"✅ Cross-Encoder loaded in {elapsed:.1f}s")
    return _cross_encoder


def rerank(query: str, hits: List[Dict[str, Any]], top_n: int = None) -> List[Dict[str, Any]]:
    """
    Rerank hits using cross-encoder semantic similarity.
    
    Args:
        query: The search query
        hits: List of hit dicts with at least 'path' and 'snippet' keys
        top_n: How many top hits to rerank (rest kept in original order after)
    
    Returns:
        Reranked list of hits with 'rerank_score' added
    """
    # Check runtime config toggle (takes precedence over env var)
    try:
        from .runtime_config import get_runtime_config
        if not get_runtime_config().get("reranker_enabled", True):
            return hits
    except Exception:
        pass
    if not RERANKER_ENABLED or not hits:
        return hits
    
    top_n = top_n or RERANKER_TOP_N
    
    # Split: rerank top_n, keep rest as-is
    to_rerank = hits[:top_n]
    remainder = hits[top_n:]
    
    # Build query-passage pairs
    pairs = []
    for h in to_rerank:
        snippet = h.get("snippet", "")[:512]  # Cross-encoder max ~512 tokens
        path = h.get("path", "").split("/")[-1]  # Filename for context
        passage = f"{path}: {snippet}" if path else snippet
        pairs.append([query, passage])
    
    if not pairs:
        return hits
    
    try:
        model = _get_model()
        start = time.time()
        scores = model.predict(pairs)
        elapsed = time.time() - start
        
        # Attach scores
        for i, h in enumerate(to_rerank):
            h["rerank_score"] = float(scores[i])
        
        # Sort reranked portion by cross-encoder score (descending)
        to_rerank.sort(key=lambda h: h.get("rerank_score", 0), reverse=True)
        
        top_score = to_rerank[0]["rerank_score"] if to_rerank else 0
        bottom_score = to_rerank[-1]["rerank_score"] if to_rerank else 0
        print(f"🎯 Cross-Encoder reranked {len(to_rerank)} hits in {elapsed:.2f}s "
              f"(top={top_score:.3f}, bottom={bottom_score:.3f})")
        
        return to_rerank + remainder
        
    except Exception as e:
        print(f"⚠️ Cross-Encoder reranking failed: {e}")
        return hits  # Fallback: return original order
