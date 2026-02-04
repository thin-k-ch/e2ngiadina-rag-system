import json
import re

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
"""

def _json_sanitize(s: str) -> str:
    # strip code fences
    s = re.sub(r"^```(json)?", "", s.strip(), flags=re.I).strip()
    s = re.sub(r"```$", "", s.strip()).strip()
    return s

async def plan_queries(llm_call, user_text: str) -> dict:
    messages = [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":user_text}
    ]
    out = await llm_call(messages, temperature=0.1)
    s = _json_sanitize(out)
    try:
        data = json.loads(s)
    except Exception:
        # fallback: naive variants
        q = user_text.strip()
        data = {"queries":[q], "must_include":[], "must_not":[], "mode":"rag"}
    # hard limits + cleanup
    qs = []
    for x in data.get("queries", [])[:12]:
        x = (x or "").strip()
        if x and x not in qs:
            qs.append(x)
    data["queries"] = qs if qs else [user_text.strip()]
    data["must_include"] = [str(x).strip() for x in data.get("must_include", []) if str(x).strip()][:6]
    data["must_not"] = [str(x).strip() for x in data.get("must_not", []) if str(x).strip()][:6]
    data["mode"] = "extract" if data.get("mode") == "extract" else "rag"
    return data
