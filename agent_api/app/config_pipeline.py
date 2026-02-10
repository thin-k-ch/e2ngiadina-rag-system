"""
Pipeline Configuration - Tunable Parameters
===========================================

All parameters can be set via environment variables:
    export RAG_MAX_CONTEXT_DOCS=40
    export RAG_MAX_SOURCES=40
    export RAG_SEARCH_TOP_K=40

Or in docker-compose.yml:
    environment:
      - RAG_MAX_CONTEXT_DOCS=40
"""

import os

# Search Phase
RAG_SEARCH_TOP_K = int(os.getenv("RAG_SEARCH_TOP_K", "40"))  # How many docs to fetch from ES/Chroma

# Ranking Phase  
RAG_KEYWORD_BOOST_PATH = float(os.getenv("RAG_KEYWORD_BOOST_PATH", "2.0"))  # Boost for keywords in path
RAG_KEYWORD_BOOST_SNIPPET = float(os.getenv("RAG_KEYWORD_BOOST_SNIPPET", "1.0"))  # Boost for keywords in content
RAG_KEYWORD_COMPOUND_BONUS = float(os.getenv("RAG_KEYWORD_COMPOUND_BONUS", "3.0"))  # Extra boost for multiple matches

# Excel Penalty
RAG_EXCEL_PENALTY_RELEVANT = float(os.getenv("RAG_EXCEL_PENALTY_RELEVANT", "-1.0"))  # Small penalty for relevant Excel
RAG_EXCEL_PENALTY_IRRELEVANT = float(os.getenv("RAG_EXCEL_PENALTY_IRRELEVANT", "-4.0"))  # Large penalty for irrelevant Excel
RAG_PDF_MSG_BONUS = float(os.getenv("RAG_PDF_MSG_BONUS", "1.0"))  # Bonus for PDF/MSG/DOCX

# Context Phase
RAG_MAX_CONTEXT_DOCS = int(os.getenv("RAG_MAX_CONTEXT_DOCS", "10"))  # How many docs to include in LLM context
RAG_MAX_SNIPPET_LENGTH = int(os.getenv("RAG_MAX_SNIPPET_LENGTH", "2000"))  # Max chars per snippet

# Sources Phase
RAG_MAX_SOURCES = int(os.getenv("RAG_MAX_SOURCES", "40"))  # How many sources to return in response

# Answer Phase
RAG_ANSWER_TEMPERATURE = float(os.getenv("RAG_ANSWER_TEMPERATURE", "0.3"))
RAG_ANSWER_MAX_TOKENS = int(os.getenv("RAG_ANSWER_MAX_TOKENS", "4000"))

# Keywords for relevance scoring
RAG_KEYWORDS = os.getenv("RAG_KEYWORDS", "fat,sat,befund,abnahme,test,prüfung,mängel").split(",")
RAG_EXCEL_RELEVANT_KEYWORDS = os.getenv("RAG_EXCEL_RELEVANT_KEYWORDS", "befund,fat,sat,abnahme,test,prüfung").split(",")


def print_config():
    """Print current configuration for debugging"""
    print("=" * 50)
    print("RAG Pipeline Configuration")
    print("=" * 50)
    print(f"Search: TOP_K={RAG_SEARCH_TOP_K}")
    print(f"Ranking: PATH_BOOST={RAG_KEYWORD_BOOST_PATH}, SNIPPET_BOOST={RAG_KEYWORD_BOOST_SNIPPET}")
    print(f"Excel: RELEVANT_PENALTY={RAG_EXCEL_PENALTY_RELEVANT}, IRRELEVANT_PENALTY={RAG_EXCEL_PENALTY_IRRELEVANT}")
    print(f"Context: MAX_DOCS={RAG_MAX_CONTEXT_DOCS}, MAX_SNIPPET={RAG_MAX_SNIPPET_LENGTH}")
    print(f"Sources: MAX_SOURCES={RAG_MAX_SOURCES}")
    print(f"Answer: TEMP={RAG_ANSWER_TEMPERATURE}, MAX_TOKENS={RAG_ANSWER_MAX_TOKENS}")
    print(f"Keywords: {RAG_KEYWORDS}")
    print("=" * 50)
