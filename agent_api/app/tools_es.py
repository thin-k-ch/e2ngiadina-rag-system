import os
from elasticsearch import Elasticsearch
from typing import Any, Dict, List, Optional

from .config_rag import (
    RAG_FILES_INDICES,
    ES_CONTENT_FIELD,
    ES_SOURCE_FIELDS,
    DEFAULT_EXT_FILTER,
)

ES_URL = os.getenv("ES_URL","http://elasticsearch:9200")
ES_INDEX = os.getenv("ES_INDEX","rag_files_v1")  # FIXED: use rag_files_v1

class ESTools:
    def __init__(self):
        self.es = None
        self.es_url = ES_URL
        
    def _get_es(self):
        if self.es is None:
            try:
                self.es = Elasticsearch(
                    self.es_url,
                    timeout=30,
                    max_retries=10,
                    retry_on_timeout=True
                )
            except Exception as e:
                print(f"Failed to initialize ES client: {e}")
                self.es = None
        return self.es

    def es_exact_phrase_content(
        self,
        phrase: str,
        *,
        indices: Optional[List[str]] = None,
        size: int = 10,
    ) -> Dict[str, Any]:
        """
        Exact-ish phrase search using match_phrase slop=0 on analyzed `content`.
        No fuzziness. No should. Deterministic.
        """
        es = self._get_es()
        if es is None:
            print("ES client not available")
            return {}
        
        idx = ",".join(indices or RAG_FILES_INDICES)

        body = {
            "size": size,
            "_source": ES_SOURCE_FIELDS,
            "query": {
                "match_phrase": {
                    ES_CONTENT_FIELD: {
                        "query": phrase,
                        "slop": 0
                    }
                }
            },
            "highlight": {
                "fields": {
                    ES_CONTENT_FIELD: {"number_of_fragments": 1, "fragment_size": 240}
                }
            }
        }
        try:
            return es.search(index=idx, body=body)
        except Exception as e:
            print(f"ES exact phrase search failed: {e}")
            return {}

    def es_exact_fallback_and(
        self,
        phrase: str,
        *,
        indices: Optional[List[str]] = None,
        size: int = 10,
    ) -> Dict[str, Any]:
        """
        Fallback ONLY if exact phrase returns 0 hits.
        Still strict: AND operator. No fuzziness.
        """
        es = self._get_es()
        if es is None:
            print("ES client not available")
            return {}
        
        idx = ",".join(indices or RAG_FILES_INDICES)

        body = {
            "size": size,
            "_source": ES_SOURCE_FIELDS,
            "query": {
                "match": {
                    ES_CONTENT_FIELD: {
                        "query": phrase,
                        "operator": "AND"
                    }
                }
            },
            "highlight": {
                "fields": {
                    ES_CONTENT_FIELD: {"number_of_fragments": 1, "fragment_size": 240}
                }
            }
        }
        try:
            return es.search(index=idx, body=body)
        except Exception as e:
            print(f"ES exact fallback and search failed: {e}")
            return {}

    def es_bm25_search_content(
        self,
        query: str,
        *,
        indices: Optional[List[str]] = None,
        size: int = 50,
        ext_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Hybrid BM25 stage on `content` with AND operator and extension filter.
        """
        es = self._get_es()
        if es is None:
            print("ES client not available")
            return {}
        
        idx = ",".join(indices or RAG_FILES_INDICES)
        exts = ext_filter or DEFAULT_EXT_FILTER

        body = {
            "size": size,
            "_source": ES_SOURCE_FIELDS,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "content^3",
                                    "file.filename^5",
                                    "path.virtual^2"
                                ],
                                "type": "best_fields",
                                "operator": "OR"
                            }
                        }
                    ],
                    "filter": [
                        {"terms": {"file.extension": exts}}
                    ]
                }
            },
            "highlight": {
                "fields": {
                    ES_CONTENT_FIELD: {"number_of_fragments": 1, "fragment_size": 240}
                }
            }
        }
        print(f"🔍 ES DEBUG: Query='{query}', Exts={exts[:5]}..., Index={idx}")
        try:
            result = es.search(index=idx, body=body)
            total = result.get("hits", {}).get("total", {}).get("value", 0)
            print(f"🔍 ES DEBUG: Found {total} hits")
            return result
        except Exception as e:
            print(f"ES BM25 search failed: {e}")
            return {}
    
    def es_get_document_content(
        self,
        doc_id: Optional[str] = None,
        file_path: Optional[str] = None,
        index: str = "rag_files_v1"
    ) -> Dict[str, Any]:
        """
        Retrieve full document content from ES by doc_id or file_path.
        Returns: {"content": "...", "path": "...", "meta": {...}}
        """
        es = self._get_es()
        if es is None:
            return {"error": "ES not available"}
        
        try:
            if doc_id:
                result = es.get(index=index, id=doc_id)
                source = result.get("_source", {})
                return {
                    "doc_id": doc_id,
                    "content": source.get("content", ""),
                    "path": source.get("meta", {}).get("real", {}).get("path", "") or 
                           source.get("path", {}).get("real", ""),
                    "meta": source.get("meta", {}),
                    "file": source.get("file", {})
                }
            elif file_path:
                body = {
                    "size": 1,
                    "query": {
                        "term": {"meta.real.path.keyword": file_path}
                    }
                }
                result = es.search(index=index, body=body)
                hits = result.get("hits", {}).get("hits", [])
                if hits:
                    source = hits[0].get("_source", {})
                    return {
                        "doc_id": hits[0].get("_id"),
                        "content": source.get("content", ""),
                        "path": file_path,
                        "meta": source.get("meta", {}),
                        "file": source.get("file", {})
                    }
                return {"error": f"Document not found: {file_path}"}
            else:
                return {"error": "Either doc_id or file_path required"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def es_search_by_path_prefix(
        self,
        path_prefix: str,
        query: str = "*",
        size: int = 10,
        index: str = "rag_files_v1"
    ) -> List[Dict[str, Any]]:
        """
        Find all documents under a specific path prefix.
        Useful for finding all Manteldokumente for a customer.
        """
        es = self._get_es()
        if es is None:
            return []
        
        body = {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {"wildcard": {"meta.real.path": f"{path_prefix}*"}},
                        {"match": {"content": query}} if query != "*" else {"match_all": {}}
                    ]
                }
            },
            "_source": ["content", "meta", "file"]
        }
        
        try:
            result = es.search(index=index, body=body)
            hits = result.get("hits", {}).get("hits", [])
            return [
                {
                    "doc_id": h.get("_id"),
                    "content": h.get("_source", {}).get("content", ""),
                    "path": h.get("_source", {}).get("meta", {}).get("real", {}).get("path", ""),
                    "score": h.get("_score"),
                    "file": h.get("_source", {}).get("file", {})
                }
                for h in hits
            ]
        except Exception as e:
            print(f"ES path prefix search failed: {e}")
            return []
