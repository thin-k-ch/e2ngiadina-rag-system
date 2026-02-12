"""
Source Analyzer - "Analysiere Quelle [N]" Feature
===================================================
Detects when a user references a previously returned source
and fetches the full document text from Elasticsearch for LLM analysis.
"""

import re
import os
import json
from typing import Optional, Tuple, Dict, Any
import httpx

# Pattern: "Analysiere Quelle [1]", "Fasse [3] zusammen", "Was steht in Quelle [2]?", etc.
SOURCE_REF_PATTERNS = [
    r'(?:analysiere|zusammenfass|erkl[äa]r|beschreib|zeig|lies|lese|öffne|inhalt|detail)\w*\s+.*?\[(\d+)\]',
    r'\[(\d+)\]\s*(?:analysier|zusammenfass|erkl[äa]r|beschreib|zeig|lies|lese|öffne|inhalt|detail)',
    r'(?:quelle|dokument|source|doc)\s*\[?(\d+)\]?',
    r'(?:analysiere|fasse|erkläre|beschreibe|zeige)\s+(?:quelle|dokument|source)\s*\[?(\d+)\]?',
    r'(?:was steht in|was enthält|inhalt von)\s+.*?\[(\d+)\]',
]


def detect_source_reference(query: str) -> Optional[int]:
    """
    Detect if user query references a source number like [1], [2], etc.
    Returns the source number (1-indexed) or None.
    """
    query_lower = query.lower().strip()
    
    for pattern in SOURCE_REF_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            return int(match.group(1))
    
    # Fallback: simple [N] detection if query is short and contains a number reference
    if len(query_lower) < 80:
        match = re.search(r'\[(\d+)\]', query)
        if match:
            return int(match.group(1))
    
    return None


async def fetch_document_text(path: str, es_url: str = None, es_index: str = None) -> Tuple[str, Dict[str, Any]]:
    """
    Fetch full document text from Elasticsearch by path.
    Returns (text, metadata) tuple.
    """
    es_url = es_url or os.getenv("ES_URL", "http://elasticsearch:9200")
    es_index = es_index or os.getenv("ES_INDEX", "rag_files_v1")
    
    # Search by path.virtual (display path) or path.real (full path)
    query = {
        "size": 1,
        "_source": ["content", "path", "meta", "file"],
        "query": {
            "bool": {
                "should": [
                    {"term": {"path.virtual.keyword": path}},
                    {"term": {"path.real.keyword": path}},
                    {"match_phrase": {"path.virtual": path}},
                ]
            }
        }
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{es_url}/{es_index}/_search",
            json=query,
            headers={"Content-Type": "application/json"}
        )
        data = r.json()
    
    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        return "", {}
    
    src = hits[0].get("_source", {})
    content = src.get("content", "")
    metadata = {
        "path": src.get("path", {}),
        "meta": src.get("meta", {}),
        "file": src.get("file", {}),
    }
    
    return content, metadata
