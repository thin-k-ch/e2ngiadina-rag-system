#!/usr/bin/env bash
# WINDSURF Release Train Suite (read-only, umfangreicher)
# Ziel: breitere Qualitätschecks, kann mehrere Minuten dauern.
# KEINE ES-Änderungen. Nur GET/POST _search/_count und Service Checks.

ES="http://localhost:9200"
IDX="rag_files_v1"
AGENT="http://localhost:11436"
WEBUI="http://localhost:8086"
OLLAMA="http://localhost:11434"

echo "=== RELEASE TRAIN: Service status ==="
curl -sS -o /dev/null -w "ES=%{http_code}\n"      "$ES/" || true
curl -sS -o /dev/null -w "WEBUI=%{http_code}\n"   "$WEBUI/" || true
curl -sS -o /dev/null -w "AGENT=%{http_code}\n"   "$AGENT/health" || true
curl -sS -o /dev/null -w "OLLAMA=%{http_code}\n"  "$OLLAMA/api/tags" || true
echo

echo "=== RELEASE TRAIN: ES health + index health ==="
curl -sS "$ES/_cluster/health?pretty" | head -c 1200; echo; echo
curl -sS "$ES/_cluster/health/$IDX?pretty" | head -c 1200; echo; echo

echo "=== RELEASE TRAIN: ES count ==="
curl -sS "$ES/$IDX/_count?pretty" | head -c 600; echo; echo

echo "=== RELEASE TRAIN: Aggs (ext + mime) ==="
curl -sS "$ES/$IDX/_search"   -H "Content-Type: application/json"   -d '{
    "size": 0,
    "aggs": {
      "ext":  { "terms": { "field": "file.extension.keyword", "size": 20 } },
      "mime": { "terms": { "field": "file.content_type.keyword", "size": 20 } }
    }
  }' | head -c 8000; echo; echo

echo "=== RELEASE TRAIN: Content sanity (200 sample, estimate empty content) ==="
curl -sS "$ES/$IDX/_search"   -H "Content-Type: application/json"   -d '{
    "size": 200,
    "_source": ["file.filename","file.extension","file.url","path.real","content"],
    "query": { "match_all": {} }
  }' | python3 - <<'PY'
import json,sys
data=json.load(sys.stdin)
hits=data.get("hits",{}).get("hits",[])
total=len(hits)
empty=0
examples=[]
for h in hits:
    src=h.get("_source",{})
    c=src.get("content")
    if c is None or (isinstance(c,str) and len(c.strip())==0):
        empty += 1
        if len(examples) < 5:
            examples.append(src.get("file",{}).get("filename") or src.get("path",{}).get("real") or "unknown")
pct = (empty/total*100) if total else 0.0
print(f"sample={total} empty={empty} empty_pct={pct:.1f}%")
if examples:
    print("empty_examples:")
    for e in examples:
        print(" -", e)
PY
echo

echo "=== RELEASE TRAIN: ES query probes ==="
echo "--- Probe 1: match 'Tabelle1' ---"
curl -sS "$ES/$IDX/_search" -H "Content-Type: application/json" -d '{
  "size": 3,
  "_source": ["file.filename","path.real"],
  "query": { "match": { "content": "Tabelle1" } }
}' | head -c 2200; echo; echo

echo "--- Probe 2: phrase 'Projektleitung Konzepthase' ---"
curl -sS "$ES/$IDX/_search" -H "Content-Type: application/json" -d '{
  "size": 3,
  "_source": ["file.filename","path.real"],
  "query": { "match_phrase": { "content": "Projektleitung Konzepthase" } }
}' | head -c 2200; echo; echo

echo "=== RELEASE TRAIN: Agent API (OpenAPI + endpoints) ==="
curl -sS "$AGENT/openapi.json" | head -c 1600; echo; echo

echo "=== RELEASE TRAIN: Agent RAG probes (3 prompts) ==="
for q in   "Suche nach \"Sockelkosten Konzeptphase\" und nenne Datei + Pfad. Gib /open URLs als Quellen an."   "Suche nach \"Projektleitung Konzepthase\" und gib Ausschnitt + Quelle."   "Gib mir eine kurze Liste der 5 häufigsten Dateitypen, mit Quellenlinks."
do
  echo "--- Prompt: $q"
  json_q=$(python3 - <<PY
import json,sys
print(json.dumps(sys.argv[1]))
PY
"$q")
  curl -sS "$AGENT/v1/chat/completions"     -H "Content-Type: application/json"     -d "{
      "model":"llama4:latest",
      "messages":[{"role":"user","content":$json_q}],
      "stream": false
    }" | head -c 2600
  echo; echo
done

echo "=== RELEASE TRAIN: DONE ==="
exit 0
