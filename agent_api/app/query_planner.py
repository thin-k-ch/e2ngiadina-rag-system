import json
import re

def _safe_extract_queries(parsed, fallback_text: str) -> list[str]:
    """
    Robustly normalize LLM output into a list[str].
    Accepts:
      - {"queries": ["..."]} or {"queries": "..."} or {"queries": 1}
      - ["..."]
      - "..."
      - 1 / None / other -> fallback
    """
    # dict style
    if isinstance(parsed, dict):
        q = parsed.get("queries", [])
        if isinstance(q, list):
            return [str(x).strip() for x in q if str(x).strip()]
        if isinstance(q, str):
            q = q.strip()
            return [q] if q else [fallback_text.strip()]
        # int/float/bool/None/other
        ft = (fallback_text or "").strip()
        return [ft] if ft else []
    # list style
    if isinstance(parsed, list):
        out = [str(x).strip() for x in parsed if str(x).strip()]
        if out:
            return out
        ft = (fallback_text or "").strip()
        return [ft] if ft else []
    # string style
    if isinstance(parsed, str):
        t = parsed.strip()
        return [t] if t else [(fallback_text or "").strip()]
    # scalar / unknown
    ft = (fallback_text or "").strip()
    return [ft] if ft else []

SYSTEM = """Du bist ein Retrieval-Planer für ein RAG-System.
Aufgabe: Erzeuge Suchanfragen-Varianten (Deutsch/Englisch/Singular/Plural/Synonyme), um relevante Dokumente zu finden.
Gib AUSSCHLIESSLICH valides JSON zurück.

JSON Schema:
{
  "queries": ["..."],
  "must_include": ["..."],    // optionale Begriffe/Entitäten
  "must_not": ["..."],        // optionale Negativbegriffe
  "mode": "rag" | "extract"   // "extract" wenn Nutzer nach Tabelle/Liste/Extraktion fragt
}

Regeln:
- Max 12 queries.
- Nutze deutsche und englische Varianten.
- Wenn im Prompt Begriffe wie "Tabelle", "Liste", "Übersicht", "extrahiere", "alle", "Beträge", "Datum" vorkommen -> mode="extract".
- Wenn Entitäten wie Firmennamen vorkommen: in must_include aufnehmen.
- Wenn du bereits Top-Treffer siehst, darfst du aus deren Stichwörtern neue Queries ableiten (Refinement).
"""

def _json_sanitize(s: str) -> str:
    # strip code fences
    s = re.sub(r"^```(json)?", "", s.strip(), flags=re.I).strip()
    s = re.sub(r"```$", "", s.strip()).strip()
    return s

def detect_retrieval_mode(user_text: str) -> dict:
    """Detect special retrieval modes from user text"""
    import re
    
    # Exact phrase mode detection with regex
    # Pattern: "suche exakt die phrase: <phrase>" (case-insensitive, stops at punctuation or end)
    pattern = r"(?i)\bsuche\s+exakt\s+die\s+phrase:\s*(.+?)(?:[.\!\?\;\n\r]|$)"
    match = re.search(pattern, user_text)
    
    if match:
        # Extract phrase and clean it
        phrase = match.group(1).strip()
        
        # Remove surrounding quotes if present
        if (phrase.startswith('"') and phrase.endswith('"')) or (phrase.startswith("'") and phrase.endswith("'")):
            phrase = phrase[1:-1].strip()
        
        return {
            "mode": "exact_phrase",
            "phrase": phrase,
            "queries": [phrase],  # Single query for exact phrase
            "must_include": [],
            "must_not": []
        }
    
    # Default RAG mode
    return {
        "mode": "rag",
        "queries": [user_text.strip()],  # Include user query as default
        "must_include": [],
        "must_not": []
    }

async def plan_queries(llm_call, user_text: str) -> dict:
    messages = [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":user_text}
    ]
    out = await llm_call(messages, temperature=0.1)
    s = _json_sanitize(out)
    try:
        parsed = json.loads(s)
    except Exception:
        # fallback: naive variants
        parsed = None
    queries = _safe_extract_queries(parsed, user_text)
    # hard limits + cleanup
    qs = []
    for x in queries[:12]:
        x = (x or "").strip()
        if x and x not in qs:
            qs.append(x)
    data = {
        "queries": qs if qs else [user_text.strip()],
        "must_include": [],
        "must_not": [],
        "mode": "rag"
    }
    return data

async def refine_queries(llm_call, user_text: str, top_hits_preview: str) -> dict:
    messages = [
        {"role":"system","content":SYSTEM + "\n\nDu bekommst jetzt Top-Hits Preview. Erzeuge bessere/konkretere Suchqueries."},
        {"role":"user","content": "USER FRAGE:\n" + user_text + "\n\nTOP-HITS PREVIEW:\n" + top_hits_preview}
    ]
    out = await llm_call(messages, temperature=0.1)
    s = _json_sanitize(out)
    try:
        parsed = json.loads(s)
    except Exception:
        parsed = None
    queries = _safe_extract_queries(parsed, user_text)
    # cleanup wie in plan_queries
    qs=[]
    for x in queries[:12]:
        x=(x or "").strip()
        if x and x not in qs:
            qs.append(x)
    data = {
        "queries": qs if qs else [user_text.strip()],
        "must_include": [],
        "must_not": [],
        "mode": "rag"
    }
    return data
