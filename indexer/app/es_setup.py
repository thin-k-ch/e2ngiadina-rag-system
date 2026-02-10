import os
from elasticsearch import Elasticsearch

ES_URL = os.getenv("ES_URL", "http://elasticsearch:9200")
INDEX = os.getenv("ES_INDEX", "rag_chunks_v1")

def main():
    es = Elasticsearch(ES_URL)

    # delete old index (optional)
    if es.indices.exists(index=INDEX):
        es.indices.delete(index=INDEX)

    body = {
      "settings": {
        "analysis": {
          "filter": {
            "de_en_synonyms": {
              "type": "synonym",
              "lenient": True,
              "synonyms": [
                "rechnung, invoice, invoices, bill, billing",
                "vertrag, contract, agreement",
                "offerte, offer, quotation, quote",
                "lieferant, supplier, vendor"
              ]
            }
          },
          "analyzer": {
            "de_en": {
              "tokenizer": "standard",
              "filter": ["lowercase", "de_en_synonyms"]
            }
          }
        }
      },
      "mappings": {
        "properties": {
          "chunk_id": {"type":"keyword"},
          "text": {"type":"text", "analyzer":"de_en"},
          "original_path": {"type":"keyword"},
          "path_text": {"type":"text", "analyzer":"de_en"},
          "document_type": {"type":"keyword"},
          "project": {"type":"keyword"},
          "folder": {"type":"keyword"},
          "chunk_index": {"type":"integer"}
        }
      }
    }

    es.indices.create(index=INDEX, body=body)
    print("OK created index", INDEX)

if __name__ == "__main__":
    main()
