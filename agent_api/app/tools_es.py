import os
from elasticsearch import Elasticsearch

ES_URL = os.getenv("ES_URL","http://elasticsearch:9200")
ES_INDEX = os.getenv("ES_INDEX","rag_chunks_v1")

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

    def search_es(self, query: str, k: int = 12, filters: dict | None = None):
        es = self._get_es()
        if es is None:
            print("ES client not available")
            return []
            
        filters = filters or {}
        must = [{"multi_match": {"query": query, "fields": ["text^2","path_text"]}}]
        f = []
        if filters.get("document_type"):
            f.append({"term":{"document_type": filters["document_type"]}})
        if filters.get("project"):
            f.append({"term":{"project": filters["project"]}})
        if filters.get("folder"):
            f.append({"term":{"folder": filters["folder"]}})

        body = {
          "size": k,
          "query": {"bool": {"must": must, "filter": f}},
          "highlight": {"fields": {"text": {"number_of_fragments": 1, "fragment_size": 180}}}
        }

        try:
            res = es.search(index=ES_INDEX, body=body)
        except Exception as e:
            print(f"ES search error: {e}")
            return []

        hits = []
        for h in res["hits"]["hits"]:
            src = h["_source"]
            md = {
              "original_path": src.get("original_path",""),
              "document_type": src.get("document_type",""),
              "project": src.get("project",""),
              "folder": src.get("folder","")
            }
            hits.append({
                "id": h["_id"],
                "score": h["_score"],
                "text": src.get("text",""),
                "metadata": md,
                "highlight": h.get("highlight",{})
            })
        return hits

    def es_exact_phrase_content(self, phrase: str, size: int = 5):
        """Exact phrase search in ES content field"""
        es = self._get_es()
        if es is None:
            print("ES client not available")
            return []
        
        # DEBUG LOGGING
        print(f" EXACT_PHRASE DEBUG: es_url={self.es_url} index={ES_INDEX} phrase='{phrase}'")
            
        body = {
            "size": size,
            "_source": ["file.filename", "path.real", "file.url", "content"],
            "query": {
                "match_phrase": {
                    "content": phrase
                }
            }
        }
        
        print(f" EXACT_PHRASE DEBUG: query_body={body}")

        try:
            res = es.search(index=ES_INDEX, body=body)
            hits = []
            for hit in res["hits"]["hits"]:
                source = hit["_source"]
                hits.append({
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "filename": source.get("file", {}).get("filename", ""),
                    "path_real": source.get("path", {}).get("real", ""),
                    "file_url": source.get("file", {}).get("url", ""),
                    "content": source.get("content", "")
                })
            print(f" EXACT_PHRASE DEBUG: found {len(hits)} hits")
            for i, hit in enumerate(hits[:3]):
                print(f" EXACT_PHRASE DEBUG: hit[{i}] filename='{hit.get('filename', 'NO_FILENAME')}'")
            return hits
        except Exception as e:
            print(f" EXACT_PHRASE DEBUG: ERROR = {e}")
            print(f"ES exact phrase search failed: {e}")
            return []
