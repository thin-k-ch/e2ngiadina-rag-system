"""
RAG Configuration - Central settings for ES indices, fields, and policies
"""
from typing import List

# Elasticsearch Configuration
RAG_FILES_INDEX = "rag_files_v1"
RAG_FILES_INDICES = [RAG_FILES_INDEX]  # optional: add "rag1_folder"

ES_CONTENT_FIELD = "content"
ES_SOURCE_FIELDS = [
    "file.filename",
    "file.extension", 
    "meta.real.path",
    "meta.real.virtual",
    "path.virtual",
    "path.real",
]

DEFAULT_EXT_FILTER = ["md", "txt", "rst", "log", "json", "yaml", "yml", "pdf", "docx", "doc", "msg", "eml", "xlsx", "xls", "pptx", "ppt"]

# Policy Configuration
EXACT_TRIGGERS = [
    r"\bexact phrase\b",
    r"\bexakt\b",
    r"\bexakt(e|es|genau(e|es)?)\b",
    r"\bwortlaut\b",
    r"\bliteral\b",
    r"\".+?\"",          # quoted text
]

SEARCH_TRIGGERS = [
    r"\bsuche\b",
    r"\bsuchen\b",
    r"\bfinde\b", 
    r"\bfinden\b",
    r"\bsuch(e|en|t)?\b",
    r"\bphrase\b",
]

INTERNAL_TRIGGERS = [
    r"\brag\b", r"\belasticsearch\b", r"\bchroma\b", r"\bmcp\b", r"\bsse\b",
    r"\bopenwebui\b", r"\bapi\b", r"\bendpoint\b", r"\bnon-stream\b", r"\bstream\b",
    r"\banswer_stream\b", r"\/v1\/chat\/completions", r"\/v1\/models",
    r"\bsanity\b", r"\bindex\b", r"\bdocs?\b", r"\blog\b", r"\bconfig\b",
]

# Stop Rules
STOP = {
    "max_rounds": 2,
    
    # Exact mode
    "min_phrase_hits": 1,           # match_phrase slop=0
    "min_and_fallback_hits": 1,     # acceptable only if phrase=0
    "require_es_executed": True,
    
    # Hybrid mode (raw BM25)
    "min_unique_docs": 2,
    "min_hits": 5,
    "min_top_bm25": 6.0,            # empirisch gut bei raw BM25
    "min_top_bm25_short_query": 4.0 # wenn Query sehr kurz
}
