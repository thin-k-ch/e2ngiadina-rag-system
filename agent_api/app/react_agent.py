"""
ReAct Agent – Autonomous Tool-Calling Loop
=============================================

Replaces the hardcoded 5-path router (A/B/C/D/E) with an LLM-driven
tool-calling loop. The LLM decides which tools to use and in what order.

Flow:
    User query → LLM (with tools) → tool_call? → execute → LLM → ... → final answer (streamed)

Requires a tool-calling-capable model (qwen2.5:72b, llama3.3:70b, llama4).
Falls back to single-shot answer if model doesn't emit tool_calls.
"""

import os
import json
import asyncio
import time
from typing import AsyncGenerator, Dict, Any, List, Optional


class LLMError(Exception):
    """Raised when LLM calls fail after retries (timeout, connection error)."""
    pass

# ---------------------------------------------------------------------------
# Tool Definitions (Ollama/OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Durchsucht das Projektarchiv (Elasticsearch + ChromaDB) nach Dokumenten. "
                           "Gibt Pfade und Textausschnitte zurück. Nutze dies für jede Frage, die sich "
                           "auf Projektdokumente, Verträge, E-Mails, Protokolle etc. bezieht.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Suchbegriffe oder Frage (deutsch)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Liest den Volltext eines bestimmten Dokuments aus Elasticsearch. "
                           "Nutze dies, wenn du ein Dokument im Detail analysieren musst "
                           "(z.B. nach search_documents einen bestimmten Treffer vertiefen).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Dokumentpfad (aus search_documents Ergebnis)"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Führt Python-Code in einer Sandbox aus. Nutze dies für: "
                           "Dateien zählen/auflisten, Datenanalyse (CSV/Excel), Berechnungen, "
                           "Statistiken. Verfügbare Bibliotheken: pandas, tabulate, csv, os, json. "
                           "Dateien liegen unter DATA_ROOT='/data'. Nutze print() für Ausgaben.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python-Code zur Ausführung. Nutze print() und setze result='...' für das Hauptergebnis."
                    },
                    "description": {
                        "type": "string",
                        "description": "Kurze Beschreibung was der Code tut"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_protocol",
            "description": "Erstellt ein strukturiertes Sitzungsprotokoll aus einem Transkript oder Gesprächstext. "
                           "Nutze dies wenn der Benutzer ein Transkript, eine Mitschrift oder einen "
                           "Besprechungstext in ein professionelles Protokoll umwandeln möchte.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transcript": {
                        "type": "string",
                        "description": "Der Transkript-/Gesprächstext"
                    },
                    "instruction": {
                        "type": "string",
                        "description": "Zusätzliche Anweisungen (z.B. 'Fokus auf Pendenzen', 'Englisches Protokoll')"
                    }
                },
                "required": ["transcript"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Listet Dateien und Unterordner in einem Verzeichnis auf. "
                           "Basispfad ist '/data' (= Projektarchiv). Nutze dies um die Ordnerstruktur "
                           "zu erkunden, bevor du Dateien liest oder analysierst.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Verzeichnispfad relativ zu /data (z.B. 'SBB TFK 2020 PJ - 1 Projekte/14 Werkvertrag')"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Optionaler Dateifilter (z.B. '*.pdf', '*.eml')"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Liest den Inhalt einer Datei direkt vom Dateisystem (nicht aus ES-Index). "
                           "Nutze dies f\u00fcr Dateien die nicht indexiert sind, oder wenn du den "
                           "exakten Dateiinhalt brauchst (z.B. CSV, TXT, Log-Dateien).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Dateipfad relativ zu /data (z.B. 'SBB TFK 2020 PJ - 1 Projekte/README.md')"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Sucht im Internet nach aktuellen Informationen. "
                           "Nutze dies f\u00fcr Fragen die NICHT aus dem Projektarchiv beantwortet werden k\u00f6nnen: "
                           "aktuelle Normen, Technologien, allgemeines Fachwissen, Preise, Nachrichten.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Suchbegriffe (idealerweise auf Englisch f\u00fcr bessere Ergebnisse)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "manage_memory",
            "description": "Verwaltet das Langzeit-Gedächtnis. Speichert Notizen, Fakten oder Anweisungen die sich der Agent "
                           "über Sitzungen hinweg merken soll. Nutze dies wenn der Benutzer sagt 'Merke dir...', "
                           "'Vergiss...', 'Erinnerung...', oder wenn wichtige Projektfakten festgehalten werden sollen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Aktion: 'save' (neue Notiz speichern), 'list' (alle Notizen anzeigen), 'search' (Notizen suchen), 'delete' (Notiz löschen)",
                        "enum": ["save", "list", "search", "delete"]
                    },
                    "content": {
                        "type": "string",
                        "description": "Bei 'save': Text der Notiz. Bei 'search'/'delete': Suchbegriff oder ID."
                    },
                    "tags": {
                        "type": "string",
                        "description": "Optionale Tags, kommagetrennt (z.B. 'projekt,kontakt,termin')"
                    }
                },
                "required": ["action"]
            }
        }
    },
]

# ---------------------------------------------------------------------------
# System Prompt for the ReAct Agent
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """DU BIST EIN AUTONOMER DOKUMENTEN-ANALYST für Schweizer Eisenbahn-Projekte (SBB TFK 2020 - Tunnelfunk).

FACHBEGRIFFE: FAT=Werksabnahme, SAT=Standortabnahme, TFK=Tunnelfunk, GBT=Gotthard Basistunnel, RBT=Rhomberg Bahntechnik

TOOLS (nutze sie aktiv – vermute nicht, suche und lies!):
- search_documents: Projektarchiv durchsuchen (Elasticsearch + ChromaDB)
- read_document: Dokument vollständig lesen (Volltext aus Index)
- execute_python: Python-Code ausführen (Dateien zählen, Datenanalyse, pandas)
- create_protocol: Sitzungsprotokoll aus Transkript erstellen
- list_files: Dateien/Ordner im Projektarchiv auflisten
- read_file: Datei direkt vom Dateisystem lesen (CSV, TXT, Log)
- web_search: Im Internet suchen (Normen, Technologie, Nachrichten)
- manage_memory: Langzeit-Gedächtnis verwalten (Notizen speichern/abrufen/löschen über Sitzungen hinweg)

ARBEITSWEISE:
1. Frage analysieren → passende Tools wählen
2. Dokumentenfragen: search_documents → DANN read_document für die relevantesten Treffer!
   Die Suche liefert nur kurze Snippets. Lies das VOLLSTÄNDIGE Dokument mit read_document um exakte Details, Definitionen und Vertragsklauseln zu finden.
3. IMMER mindestens 1x read_document aufrufen bevor du antwortest – Snippets allein reichen NICHT.
4. Dateisystem (zählen, listen): execute_python oder list_files
5. Datenanalyse (CSV, Excel): execute_python mit pandas
6. Transkript → Protokoll: create_protocol (GESAMTEN Text übergeben, nicht kürzen)
7. Externes Wissen (Normen, Preise, Nachrichten): web_search
8. Mehrere Tools kombinieren und mehrere Schritte machen

ANTWORT-REGELN:
- Antworte auf Deutsch
- Starte DIREKT mit der konkreten Antwort – KEINE Einleitungen ("Basierend auf...", "Gerne...", "Es scheint...")
- ZITIERE exakte Textpassagen aus den Dokumenten in Anführungszeichen: "exakter Text" [Dateiname]
- Nenne Seitenzahlen, Datumswerte und Kapitelnummern wenn verfügbar
- Verwende Indikativ, nicht Konjunktiv ("Der Vertrag regelt..." statt "Der Vertrag könnte regeln...")
- Nenne die KONKRETEN Dokumente, in denen die Information steht (Vertrag, Protokoll, E-Mail etc.)
- Strukturiere mit Markdown: Überschriften (##), Aufzählungen, **Fettdruck** für Schlüsselbegriffe
- Wenn du ein Hauptdokument findest (z.B. Werkvertrag, Pflichtenheft), nenne es PROMINENT am Anfang
- Bei Begrüssungen (Hallo, Hi): antworte kurz und freundlich, liste NICHT deine Tools auf

VERBOTEN:
- KEINE vagen Aussagen wie "scheint zu sein", "könnte sein", "es gibt Hinweise"
- KEINE allgemeinen Beschreibungen wenn du konkrete Fakten aus den Dokumenten hast
- Erfinde NIEMALS URLs oder Links. Quellen-Links werden automatisch angehängt. Verweise nur mit [Dateiname] oder [N].
- Für Dateisystem-Fragen IMMER execute_python nutzen, NICHT search_documents.
- Sage NICHT "Ich konnte leider keine Informationen finden" wenn du noch nicht alle Tools versucht hast.
- Keine Vermutungen – wenn du unsicher bist, lies das Dokument mit read_document oder suche weiter.

KRITISCH – NIEMALS SAGEN:
- "Ich habe keinen Zugriff auf das Dokument" → Du HAST Zugriff über read_document!
- "Ich bin eine KI und kann nicht auf Dokumente zugreifen" → FALSCH. Du hast Tools dafür.
- "Bitte teilen Sie mir mit, welches Dokument..." → SUCHE SELBST mit search_documents!
- "Ich habe keinen direkten Zugriff auf proprietäre Informationen" → Du bist direkt an das Projektarchiv angeschlossen.
Du bist KEIN generischer Chatbot. Du bist ein spezialisierter Dokumenten-Analyst MIT Zugriff auf das komplette Projektarchiv über deine Tools. Wenn der User nach einem Dokument fragt, NUTZE deine Tools um es zu finden und zu lesen."""

# ---------------------------------------------------------------------------
# Tool Execution
# ---------------------------------------------------------------------------

async def _execute_search(args: dict, tenant=None) -> str:
    """Execute search_documents tool"""
    query = args.get("query", "")
    if not query:
        return "Fehler: Kein Suchbegriff angegeben."
    
    from .tools import Tools
    from .rag_pipeline import SimpleRAGPipeline
    
    # Use tenant-specific ES index if available
    es_index = tenant.es_index if tenant else os.getenv("ES_INDEX", "rag_files_v1")
    ext_filter = tenant.ext_filter if tenant else None
    tools = Tools()
    pipeline = SimpleRAGPipeline()
    
    # Apply glossary rewrite
    from .glossary import rewrite_query
    rewritten, _ = rewrite_query(query)
    
    print(f"🔍 _execute_search: query='{rewritten}', es_index={es_index}, tenant={tenant.short_name if tenant else 'none'}")
    
    # Hybrid search – pass tenant-specific ES index and ext_filter
    result = await asyncio.to_thread(
        tools.search_hybrid,
        query=rewritten,
        es_size=40,
        indices=[es_index],
        ext_filter=ext_filter,
    )
    
    hits = result.get("merged_hits", [])
    
    # Normalize and rank
    normalized = []
    for h in hits:
        path = h.get("file", {}).get("path", "")
        snippet = h.get("snippet", "")
        score = h.get("score", 0)
        normalized.append({"path": path, "snippet": snippet, "score": score})
    
    ranked = pipeline._rank_hits(normalized, query)
    
    # Cross-Encoder Reranking (semantic rerank of top hits)
    from .reranker import rerank
    ranked = await asyncio.to_thread(rerank, query, ranked)
    
    # Format top results for the LLM
    top_n = ranked[:10]
    if not top_n:
        return f"Keine Treffer für '{query}'."
    
    parts = [f"Suche '{query}': {len(ranked)} Treffer. Top {len(top_n)}:\n"]
    for i, h in enumerate(top_n, 1):
        snippet = h.get("snippet", "")[:500]
        parts.append(f"[{i}] {h['path']}\n{snippet}\n")
    
    return "\n".join(parts)


async def _execute_read_document(args: dict, tenant=None) -> str:
    """Execute read_document tool"""
    path = args.get("path", "")
    if not path:
        return "Fehler: Kein Dokumentpfad angegeben."
    
    from .source_analyzer import fetch_document_text
    
    # Pass tenant-specific ES index
    es_index = tenant.es_index if tenant else None
    content, metadata = await fetch_document_text(path, es_index=es_index)
    
    if not content:
        return f"Dokument nicht gefunden: {path}"
    
    max_chars = 20000
    total_len = len(content)
    
    if total_len > max_chars:
        # Smart truncation: try to find the relevant section using the query
        query_hint = args.get("query_hint", "")
        best_section = _find_relevant_section(content, query_hint, max_chars)
        if best_section:
            content = best_section
        else:
            content = content[:max_chars] + f"\n\n[... gekürzt, {total_len} Zeichen total]"
    
    return f"=== {path} ({total_len} Zeichen) ===\n{content}"


def _find_relevant_section(content: str, query_hint: str, max_chars: int) -> str:
    """Find the most relevant section in a long document based on query keywords.
    Returns a section of max_chars around the best match, or None if no good match."""
    if not query_hint:
        return None
    
    import re
    # Extract meaningful keywords from query (skip very short words)
    words = [w.lower() for w in re.findall(r'\w+', query_hint) if len(w) >= 4]
    if not words:
        return None
    
    # Score each position by keyword density in a sliding window
    content_lower = content.lower()
    best_score = 0
    best_pos = 0
    window = 3000  # Score window
    step = 500
    
    for pos in range(0, len(content) - window, step):
        chunk = content_lower[pos:pos + window]
        score = sum(chunk.count(w) for w in words)
        if score > best_score:
            best_score = score
            best_pos = pos
    
    if best_score == 0:
        return None
    
    # Extract section centered on best position, with some context before
    half = max_chars // 2
    start = max(0, best_pos - half // 3)  # More text after the match than before
    end = min(len(content), start + max_chars)
    start = max(0, end - max_chars)
    
    section = content[start:end]
    
    # Add markers
    prefix = f"[... Dokument ab Position {start}/{len(content)}]\n" if start > 0 else ""
    suffix = f"\n[... gekürzt, {len(content)} Zeichen total]" if end < len(content) else ""
    
    print(f"📌 Smart-Truncation: found relevant section at pos {best_pos}, score={best_score}, showing {start}-{end}/{len(content)}")
    return prefix + section + suffix


async def _execute_python(args: dict, tenant=None) -> str:
    """Execute execute_python tool"""
    code = args.get("code", "")
    desc = args.get("description", "")
    if not code:
        return "Fehler: Kein Code angegeben."
    
    from .code_executor import execute_code, format_execution_result
    
    print(f"⚙️ execute_python: {desc or code[:80]}...")
    result = await execute_code(code)
    formatted = format_execution_result(result)
    
    return f"Code-Ergebnis ({desc}):\n{formatted}" if desc else f"Code-Ergebnis:\n{formatted}"


async def _execute_create_protocol(args: dict, tenant=None) -> str:
    """Execute create_protocol tool – streams protocol via LLM"""
    transcript = args.get("transcript", "")
    instruction = args.get("instruction", "Erstelle ein vollständiges Protokoll mit Pendenzenliste.")
    
    if not transcript:
        return "Fehler: Kein Transkript angegeben."
    
    from .transcript_processor import preprocess_transcript, PROTOCOL_SYSTEM_PROMPT, PROTOCOL_USER_TEMPLATE
    from .rag_pipeline import SimpleRAGPipeline
    
    # Preprocess
    transcript = preprocess_transcript(transcript)
    
    # Build LLM messages
    user_msg = PROTOCOL_USER_TEMPLATE.format(
        instruction=instruction,
        transcript=transcript
    )
    
    messages = [
        {"role": "system", "content": PROTOCOL_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg}
    ]
    
    # Use non-streaming LLM call (result goes back into the ReAct loop)
    model = os.getenv("OLLAMA_MODEL_ANSWER", "llama4:latest")
    pipeline = SimpleRAGPipeline(model=model)
    
    print(f"📝 create_protocol: {len(transcript)} chars transcript, instruction: {instruction[:80]}")
    protocol = await pipeline._llm_complete(messages)
    
    return protocol


async def _execute_list_files(args: dict, tenant=None) -> str:
    """Execute list_files tool – list directory contents via PyRunner"""
    path = args.get("path", "")
    pattern = args.get("pattern", "")
    
    filter_line = ""
    if pattern:
        filter_line = f"\nimport fnmatch\nentries = [e for e in entries if os.path.isdir(os.path.join(full_path, e)) or fnmatch.fnmatch(e, {repr(pattern)})]\n"
    
    code = f"""import os

full_path = os.path.join(DATA_ROOT, {repr(path.strip('/'))})
if not os.path.isdir(full_path):
    print(f"Verzeichnis nicht gefunden: {{full_path}}")
    result = "nicht gefunden"
else:
    entries = sorted(os.listdir(full_path))
    {filter_line}
    dirs = []
    files = []
    for e in entries:
        ep = os.path.join(full_path, e)
        if os.path.isdir(ep):
            try:
                sub_count = len(os.listdir(ep))
            except:
                sub_count = 0
            dirs.append(f"📁 {{e}}/ ({{sub_count}} Einträge)")
        else:
            size = os.path.getsize(ep)
            if size > 1048576:
                size_str = f"{{size/1048576:.1f}} MB"
            elif size > 1024:
                size_str = f"{{size/1024:.0f}} KB"
            else:
                size_str = f"{{size}} B"
            files.append(f"📄 {{e}} ({{size_str}})")
    
    print(f"Verzeichnis: {{full_path}}")
    print(f"{{len(dirs)}} Ordner, {{len(files)}} Dateien")
    print()
    for d in dirs[:50]:
        print(f"  {{d}}")
    for f in files[:50]:
        print(f"  {{f}}")
    if len(dirs) + len(files) > 100:
        print(f"  ... und {{len(dirs) + len(files) - 100}} weitere")
    result = f"{{len(dirs)}} Ordner, {{len(files)}} Dateien"
"""
    
    from .code_executor import execute_code, format_execution_result
    res = await execute_code(code)
    return format_execution_result(res)


async def _execute_read_file(args: dict, tenant=None) -> str:
    """Execute read_file tool – read file content via PyRunner"""
    path = args.get("path", "")
    if not path:
        return "Fehler: Kein Dateipfad angegeben."
    
    code = f"""import os

full_path = os.path.join(DATA_ROOT, {repr(path.strip('/'))})
if not os.path.isfile(full_path):
    print(f"Datei nicht gefunden: {{full_path}}")
    result = None
else:
    size = os.path.getsize(full_path)
    ext = os.path.splitext(full_path)[1].lower()
    if ext in ('.pdf', '.docx', '.xlsx', '.pptx', '.msg', '.zip', '.jpg', '.png'):
        print(f"Binärdatei: {{full_path}} ({{size}} bytes, {{ext}})")
        print("Hinweis: Nutze read_document für indexierte Dokumente oder execute_python für Datenanalyse.")
        result = f"Binärdatei {{ext}}, {{size}} bytes"
    else:
        max_chars = 15000
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_chars + 1)
        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars]
        print(f"=== {{os.path.basename(full_path)}} ({{size}} bytes) ===")
        print(content)
        if truncated:
            print(f"\\n[... gekürzt, {{size}} bytes total]")
        result = f"{{len(content)}} Zeichen gelesen"
"""
    
    from .code_executor import execute_code, format_execution_result
    res = await execute_code(code, timeout=15)
    return format_execution_result(res)


async def _execute_web_search(args: dict, tenant=None) -> str:
    """Execute web_search tool – SearXNG (self-hosted) → Brave → Serper fallback chain"""
    query = args.get("query", "")
    if not query:
        return "Fehler: Kein Suchbegriff angegeben."
    
    import httpx
    
    def _format_results(results: list[dict], source: str) -> str:
        parts = [f"Web-Suche '{query}' ({source}): {len(results)} Ergebnisse\n"]
        for i, res in enumerate(results, 1):
            title = res.get("title", "")
            url = res.get("url", res.get("link", ""))
            desc = res.get("content", res.get("description", res.get("snippet", "")))[:300]
            parts.append(f"[{i}] {title}\n    {url}\n    {desc}\n")
        return "\n".join(parts)
    
    # --- Priority 1: SearXNG (self-hosted, no API key needed) ---
    searxng_url = os.getenv("SEARXNG_URL", "")
    if searxng_url:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(
                    f"{searxng_url}/search",
                    params={"q": query, "format": "json", "language": "de", "pageno": 1},
                    headers={"Accept": "application/json"}
                )
                data = r.json()
                results = data.get("results", [])[:8]
                if results:
                    print(f"🌐 SearXNG: {len(results)} results for '{query}'")
                    return _format_results(results, "SearXNG")
                else:
                    print(f"⚠️ SearXNG: 0 results for '{query}'")
        except Exception as e:
            print(f"⚠️ SearXNG error: {e}")
    
    # --- Priority 2: Brave Search API (needs BRAVE_API_KEY) ---
    brave_key = os.getenv("BRAVE_API_KEY", "")
    if brave_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": 5},
                    headers={"X-Subscription-Token": brave_key, "Accept": "application/json"}
                )
                data = r.json()
                results = data.get("web", {}).get("results", [])
                if results:
                    return _format_results(results, "Brave")
        except Exception as e:
            print(f"⚠️ Brave Search error: {e}")
    
    # --- Priority 3: Serper.dev (needs SERPER_API_KEY) ---
    serper_key = os.getenv("SERPER_API_KEY", "")
    if serper_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(
                    "https://google.serper.dev/search",
                    json={"q": query, "num": 5},
                    headers={"X-API-KEY": serper_key, "Content-Type": "application/json"}
                )
                data = r.json()
                results = data.get("organic", [])
                if results:
                    return _format_results(results, "Serper")
        except Exception as e:
            print(f"⚠️ Serper Search error: {e}")
    
    return ("Web-Suche nicht verfügbar. Weder SearXNG noch API-Keys konfiguriert.\n"
            "Beantworte die Frage basierend auf deinem Trainings-Wissen.")


async def _execute_manage_memory(args: dict, tenant=None) -> str:
    """Execute manage_memory tool – persistent notes per tenant."""
    from .memory_store import get_memory_store
    
    store = get_memory_store()
    tenant_id = tenant.short_name if tenant else "default"
    action = args.get("action", "list")
    content = args.get("content", "")
    tags_str = args.get("tags", "")
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
    
    if action == "save":
        if not content:
            return "Fehler: Kein Inhalt zum Speichern angegeben."
        entry = store.add(tenant_id, content, tags)
        return f"✅ Notiz gespeichert (ID: {entry['id']}): {content}"
    
    elif action == "list":
        memories = store.list_all(tenant_id)
        if not memories:
            return "📋 Keine Notizen vorhanden."
        lines = [f"📋 {len(memories)} Notizen gespeichert:\n"]
        for m in memories:
            tags_info = f" [{', '.join(m['tags'])}]" if m.get("tags") else ""
            lines.append(f"- [{m['id']}] {m['content']}{tags_info}")
        return "\n".join(lines)
    
    elif action == "search":
        if not content:
            return "Fehler: Kein Suchbegriff angegeben."
        results = store.search(tenant_id, content)
        if not results:
            return f"🔍 Keine Notizen gefunden für '{content}'."
        lines = [f"🔍 {len(results)} Notizen gefunden für '{content}':\n"]
        for m in results:
            lines.append(f"- [{m['id']}] {m['content']}")
        return "\n".join(lines)
    
    elif action == "delete":
        if not content:
            return "Fehler: Kein Suchbegriff oder ID zum Löschen angegeben."
        # Try delete by ID first
        if store.delete(tenant_id, content):
            return f"🗑️ Notiz {content} gelöscht."
        # Try delete by keyword
        count = store.delete_by_content(tenant_id, content)
        if count > 0:
            return f"🗑️ {count} Notiz(en) mit '{content}' gelöscht."
        return f"❌ Keine Notiz gefunden mit ID oder Stichwort '{content}'."
    
    return f"Unbekannte Aktion: {action}. Verwende 'save', 'list', 'search' oder 'delete'."


TOOL_EXECUTORS = {
    "search_documents": _execute_search,
    "read_document": _execute_read_document,
    "execute_python": _execute_python,
    "create_protocol": _execute_create_protocol,
    "list_files": _execute_list_files,
    "read_file": _execute_read_file,
    "web_search": _execute_web_search,
    "manage_memory": _execute_manage_memory,
}

# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

# Module-level cache: models that returned 400 on native tool-calling.
# Persists across requests (ReactAgent instances are per-request).
# Pre-seeded with known reasoning models without native tool support.
_PROMPT_TOOLS_MODELS: set[str] = set()
_PROMPT_TOOLS_PREFIXES = ["deepseek-r1", "deepseek-r2", "qwq", "phi4-reasoning"]


def _needs_prompt_tools(model: str) -> bool:
    """Check if model needs prompt-based tool calling (cached or known prefix)."""
    if model in _PROMPT_TOOLS_MODELS:
        return True
    model_base = model.split(":")[0].lower()
    return any(prefix in model_base for prefix in _PROMPT_TOOLS_PREFIXES)


def _mark_prompt_tools(model: str):
    """Cache a model as needing prompt-based tool calling."""
    if model not in _PROMPT_TOOLS_MODELS:
        _PROMPT_TOOLS_MODELS.add(model)
        print(f"💾 Cached {model} as prompt-tools model (will skip native tools on future requests)")


class ReactAgent:
    """
    Autonomous agent with tool-calling loop.
    
    Uses Ollama's native tool-calling format. Falls back to
    direct answer if model doesn't support tool calling.
    """
    
    def __init__(self, model: str = None, ollama_base: str = None, tenant=None):
        self.model_answer = model or os.getenv("OLLAMA_MODEL_ANSWER", "llama4:latest")
        # Strategy model: runtime config > env var > same as answer model
        from .runtime_config import get_runtime_config
        cfg = get_runtime_config()
        strategy = cfg.get("strategy_model", "")
        if not strategy:
            strategy = os.getenv("OLLAMA_MODEL_STRATEGY", "")
        self.model_strategy = strategy if strategy else self.model_answer
        self.num_batch = cfg.get("num_batch", 1024)
        self.num_ctx_max = cfg.get("num_ctx_max", 131072)
        # Online model for strategy (fast cloud routing, no doc content sent)
        self._online_enabled = bool(cfg.get("online_model_enabled", False))
        self._online_api_url = cfg.get("online_api_url", "https://api.openai.com/v1")
        self._online_api_key = cfg.get("online_api_key", "")
        self._online_model = cfg.get("online_model_name", "gpt-4o-mini")
        self._online_strategy_mode = cfg.get("online_strategy_mode", "routing")  # "routing" or "planner"
        if self._online_enabled and self._online_api_key:
            print(f"☁️ Online strategy: {self._online_model} via {self._online_api_url} (mode={self._online_strategy_mode})")
        elif self._online_enabled:
            print(f"⚠️ Online strategy enabled but no API key set → falling back to local")
            self._online_enabled = False
        # Expose .model for backward compat (used in _stream_with_thinking etc.)
        self.model = self.model_answer
        self.ollama_base = (ollama_base or os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")).rstrip("/")
        self.max_steps = 6
        self.tenant = tenant  # TenantConfig or None
        # Check module-level cache + known prefixes for prompt-based tool calling
        self._use_prompt_tools_answer = _needs_prompt_tools(self.model_answer)
        self._use_prompt_tools_strategy = _needs_prompt_tools(self.model_strategy)
        if self.model_strategy != self.model_answer:
            print(f"🧠 Model split: strategy={self.model_strategy}, answer={self.model_answer}")
        if self._use_prompt_tools_answer:
            print(f"🧠 Model {self.model_answer}: using prompt-based tool calling (no native tools)")
    
    async def _get_search_plan(self, query: str, chat_history: list = None) -> dict:
        """Ask cloud model to generate a structured search plan. No doc content sent."""
        import httpx
        
        planner_prompt = """Du bist ein Such-Stratege für ein RAG-System mit Schweizer Bau-/Infrastruktur-Dokumenten.

Analysiere die Benutzer-Frage und erstelle einen strukturierten Suchplan als JSON.

Regeln:
- Generiere 1-3 gezielte Suchanfragen (deutsch, mit Synonymen/Fachbegriffen)
- Entscheide ob Dokumente gelesen werden sollen (read_top_n: 0-3)
- Wenn der User WORTGENAUE Zitate oder spezifische Kapitel verlangt: read_top_n=1-2 (gezielt lesen!)
- Weniger Dokumente = bessere Zitatqualität. Lieber 1 richtiges als 5 falsche.
- Gib Fokus-Hinweise für die finale Antwort
- Bei einfachen Fragen: weniger Schritte. Bei komplexen: mehr.
- Bei Grüssen/Small-Talk: setze "skip": true

Antworte NUR mit validem JSON, kein anderer Text:
{
  "skip": false,
  "queries": ["Suchanfrage 1", "Suchanfrage 2"],
  "read_top_n": 2,
  "focus": "Worauf die Antwort fokussieren soll",
  "answer_format": "Prosa / Tabelle / Liste / Aufzählung",
  "answer_hint": "z.B. 'Zitiere WORTGENAU aus dem Dokument' oder 'Vergleiche die Quellen'"
}

WICHTIG für answer_hint:
- Wenn der User "wortgenau", "exakt", "originaltext" oder "zitiere" schreibt: answer_hint MUSS "Zitiere WORTGENAU und VOLLSTÄNDIG aus dem Dokument. Kein Paraphrasieren!" enthalten
- Wenn der User ein spezifisches Kapitel nennt: answer_hint MUSS "Fokussiere auf das genannte Kapitel" enthalten"""
        
        messages = [{"role": "system", "content": planner_prompt}]
        if chat_history:
            for m in chat_history[-4:]:
                messages.append({"role": m["role"], "content": m.get("content", "")[:1500]})
        messages.append({"role": "user", "content": query})
        
        total_chars = sum(len(m.get("content", "")) for m in messages)
        print(f"📋 Planner (online): {total_chars} chars, model={self._online_model}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    f"{self._online_api_url.rstrip('/')}/chat/completions",
                    json={
                        "model": self._online_model,
                        "messages": messages,
                        "temperature": 0.1,
                        "max_tokens": 512,
                    },
                    headers={
                        "Authorization": f"Bearer {self._online_api_key}",
                        "Content-Type": "application/json",
                    }
                )
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            print(f"⚠️ Planner call failed: {e}")
            return None
        
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        print(f"📋 Planner response: {len(content)} chars, tokens={usage.get('total_tokens', '?')}")
        
        # Parse JSON from response (handle markdown code blocks)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        
        try:
            plan = json.loads(content)
            print(f"📋 Plan: {len(plan.get('queries', []))} queries, read_top_n={plan.get('read_top_n', 0)}, focus={plan.get('focus', '?')[:60]}")
            return plan
        except json.JSONDecodeError:
            print(f"⚠️ Could not parse planner JSON: {content[:200]}")
            return None
    
    async def _run_planner_mode(
        self, query: str, chat_history: list = None,
        system_prompt_extra: str = "",
    ) -> AsyncGenerator[dict, None]:
        """Planner mode: Cloud model creates search plan, local model only writes the answer."""
        
        yield {"type": "thinking_start"}
        yield {"type": "phase", "content": "📋 Erstelle Suchstrategie...\n\n"}
        
        start_time = time.time()
        plan = await self._get_search_plan(query, chat_history)
        
        # Fallback to ReAct if planner fails or says skip
        if not plan:
            yield {"type": "phase", "content": "⚠️ Planner-Fallback → ReAct\n\n"}
            yield {"type": "thinking_end", "steps": 0, "elapsed": int(time.time() - start_time)}
            # Fall through to normal run handled by caller
            return
        
        if plan.get("skip"):
            yield {"type": "thinking_end", "steps": 0, "elapsed": int(time.time() - start_time)}
            # Simple query, no search needed – stream direct answer
            messages = self._build_system_messages(query, chat_history, system_prompt_extra)
            async for evt in self._stream_with_thinking(messages):
                yield evt
            yield {"type": "done"}
            return
        
        # --- Execute plan mechanically ---
        all_sources = []
        search_context = []
        read_context = []
        read_sources = []  # Only docs that were actually read
        steps = 0
        
        # Step 1: Run all search queries
        queries = plan.get("queries", [query])
        for sq in queries[:3]:  # max 3 search queries
            steps += 1
            yield {"type": "phase", "content": f"🔍 Suche: *{sq[:60]}*...\n\n"}
            yield {"type": "tool_call", "name": "search_documents", "args": {"query": sq}}
            
            result = await _execute_search({"query": sq}, tenant=self.tenant)
            search_context.append(f"--- Suche '{sq}' ---\n{result}")
            all_sources.extend(self._extract_sources(result))
            
            summary = f"{len(result)} Zeichen"
            yield {"type": "tool_result", "name": "search_documents", "summary": summary}
        
        # Step 2: Read top documents if requested (max 3, dedup by path)
        read_n = min(plan.get("read_top_n", 0), 3)
        if read_n > 0 and all_sources:
            seen_paths = set()
            for src in all_sources:
                if len(read_sources) >= read_n:
                    break
                path = src.get("path", "")
                if not path or path in seen_paths:
                    continue
                seen_paths.add(path)
                steps += 1
                yield {"type": "phase", "content": f"📄 Lese: *{src.get('display_name', path)[:50]}*...\n\n"}
                yield {"type": "tool_call", "name": "read_document", "args": {"path": path}}
                
                doc_content = await _execute_read_document({"path": path, "query_hint": query}, tenant=self.tenant)
                read_context.append(f"--- Dokument: {path} ---\n{doc_content}")
                read_sources.append(src)
                
                summary = f"{len(doc_content)} Zeichen"
                yield {"type": "tool_result", "name": "read_document", "summary": summary}
        
        elapsed = int(time.time() - start_time)
        yield {"type": "thinking_end", "steps": steps, "elapsed": elapsed}
        
        # Step 3: Generate final answer – use same format as ReAct loop
        focus = plan.get("focus", "")
        answer_format = plan.get("answer_format", "")
        answer_hint = plan.get("answer_hint", "")
        
        messages = self._build_system_messages(query, chat_history, system_prompt_extra)
        user_msg = messages.pop()  # Remove user query (re-add at end)
        
        # If documents were read, use ONLY their full content (no search snippet noise)
        if read_context:
            for ctx in read_context:
                truncated = ctx[:20000] if len(ctx) > 20000 else ctx
                messages.append({"role": "assistant", "content": "Vollständiger Dokument-Inhalt:"})
                messages.append({"role": "tool", "content": truncated})
        else:
            # No docs read – use search snippets only
            for ctx in search_context:
                truncated = ctx[:12000] if len(ctx) > 12000 else ctx
                messages.append({"role": "assistant", "content": "Ich habe folgende Informationen gefunden:"})
                messages.append({"role": "tool", "content": truncated})
        
        # Add focus/format hints
        hints = []
        if focus: hints.append(f"Fokus: {focus}")
        if answer_format: hints.append(f"Format: {answer_format}")
        if answer_hint: hints.append(answer_hint)
        if hints:
            messages.append({"role": "assistant", "content": " | ".join(hints)})
        
        messages.append(user_msg)
        
        # Stream final answer from local model
        async for evt in self._stream_with_thinking(messages):
            yield evt
        
        # Emit sources: only READ docs if available, otherwise search sources
        if read_sources:
            yield {"type": "sources", "sources": read_sources}
        else:
            seen = set()
            unique_sources = []
            for s in all_sources:
                fn = s.get("display_name", s.get("path", ""))
                if fn not in seen:
                    seen.add(fn)
                    unique_sources.append(s)
            if unique_sources:
                yield {"type": "sources", "sources": unique_sources[:5]}
        
        yield {"type": "done"}
    
    def _build_system_messages(self, query, chat_history, system_prompt_extra=""):
        """Build messages list with system prompt, memories, chat history, and query."""
        system_content = REACT_SYSTEM_PROMPT
        if self.tenant:
            if self.tenant.glossary_line:
                system_content = system_content.replace(
                    "FACHBEGRIFFE: FAT=Werksabnahme, SAT=Standortabnahme, TFK=Tunnelfunk, GBT=Gotthard Basistunnel, RBT=Rhomberg Bahntechnik",
                    self.tenant.glossary_line
                )
            if self.tenant.system_prompt_extra:
                system_content += "\n\n" + self.tenant.system_prompt_extra.strip()
        if system_prompt_extra:
            system_content += "\n\n" + system_prompt_extra
        from .memory_store import get_memory_store
        tenant_id = self.tenant.short_name if self.tenant else "default"
        memory_text = get_memory_store().format_for_prompt(tenant_id)
        if memory_text:
            system_content += f"\n\nLANGZEIT-GEDÄCHTNIS:\n{memory_text}"
        messages = [{"role": "system", "content": system_content}]
        if chat_history:
            messages.extend(chat_history[-6:])
        messages.append({"role": "user", "content": query})
        return messages
    
    async def run(
        self,
        query: str,
        chat_history: list = None,
        system_prompt_extra: str = "",
        max_steps: int = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Run the ReAct loop. Yields dicts:
            {"type": "phase", "content": "🔍 Suche..."}
            {"type": "tool_call", "name": "search_documents", "args": {...}}
            {"type": "tool_result", "name": "search_documents", "summary": "10 Treffer"}
            {"type": "token", "content": "..."}  # Final answer streaming
            {"type": "sources", "sources": [...]}
            {"type": "done"}
        """
        max_steps = max_steps or self.max_steps
        
        # --- Planner mode: Cloud generates search plan, local only writes answer ---
        if self._online_enabled and self._online_strategy_mode == "planner":
            planner_done = False
            async for evt in self._run_planner_mode(query, chat_history, system_prompt_extra):
                yield evt
                if evt.get("type") == "done":
                    planner_done = True
            if planner_done:
                return
            # If planner returned without "done", it failed → fall through to ReAct
            print(f"🔄 Planner mode failed, falling back to ReAct loop")
        
        # Build initial messages – inject tenant context
        system_content = REACT_SYSTEM_PROMPT
        if self.tenant:
            # Replace generic glossary line with tenant-specific one
            if self.tenant.glossary_line:
                system_content = system_content.replace(
                    "FACHBEGRIFFE: FAT=Werksabnahme, SAT=Standortabnahme, TFK=Tunnelfunk, GBT=Gotthard Basistunnel, RBT=Rhomberg Bahntechnik",
                    self.tenant.glossary_line
                )
            if self.tenant.system_prompt_extra:
                system_content += "\n\n" + self.tenant.system_prompt_extra.strip()
        if system_prompt_extra:
            system_content += "\n\n" + system_prompt_extra
        
        # Inject long-term memories into system prompt
        from .memory_store import get_memory_store
        tenant_id = self.tenant.short_name if self.tenant else "default"
        memory_text = get_memory_store().format_for_prompt(tenant_id)
        if memory_text:
            system_content += f"\n\nLANGZEIT-GEDÄCHTNIS (gespeicherte Notizen – berücksichtige diese bei deinen Antworten):\n{memory_text}"
        
        # Query analysis: inject tool hints for specific query types
        tool_hint = self._analyze_query(query)
        if tool_hint:
            system_content += f"\n\nHINWEIS ZUR AKTUELLEN ANFRAGE: {tool_hint}"
            print(f"💡 Tool hint: {tool_hint}")
        
        messages = [{"role": "system", "content": system_content}]
        
        # Add chat history (last 3 turns)
        if chat_history:
            messages.extend(chat_history[-6:])
        
        messages.append({"role": "user", "content": query})
        
        # Collect sources for linking
        all_sources = []
        forced_search_done = False  # Track if we did a forced search (to give LLM extra step for read_document)
        thinking_active = False
        thinking_start_time = None
        thinking_steps = 0
        
        # --- Shortcut for prompt-tool models on simple queries ---
        fs_code = self._auto_filesystem_code(query)
        if self._use_prompt_tools_strategy and not fs_code and not self._needs_search(query):
            print(f"⚡ Prompt-tool model shortcut: simple query, skipping tool-calling")
            async for evt in self._stream_with_thinking(messages):
                yield evt
            yield {"type": "done"}
            return
        
        # --- Forced first step for filesystem queries ---
        if fs_code:
            print(f"📂 Forced execute_python for filesystem query")
            thinking_active = True
            thinking_start_time = time.time()
            yield {"type": "thinking_start"}
            yield {"type": "phase", "content": "⚙️ Dateisystem-Analyse...\n\n"}
            yield {"type": "tool_call", "name": "execute_python", "args": {"code": fs_code}}
            
            result = await _execute_python({"code": fs_code, "description": "Dateisystem-Analyse"}, tenant=self.tenant)
            thinking_steps += 1
            yield {"type": "tool_result", "name": "execute_python", "summary": f"{len(result)} Zeichen"}
            
            # Inject result into conversation for the LLM to summarize
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "execute_python", "arguments": {"code": fs_code}}}]
            })
            messages.append({"role": "tool", "content": result})
        
        # --- ReAct Loop: non-streaming tool steps ---
        for step in range(max_steps):
            print(f"🤖 ReAct step {step+1}/{max_steps}")
            
            # Call LLM with tools (with keepalive to prevent timeouts)
            response = None
            async for evt in self._llm_with_tools_keepalive(messages):
                if evt["type"] == "keepalive":
                    if not thinking_active:
                        thinking_active = True
                        thinking_start_time = time.time()
                        yield {"type": "thinking_start"}
                    yield evt
                else:
                    response = evt["data"]
            
            tool_calls = response.get("message", {}).get("tool_calls")
            content = response.get("message", {}).get("content", "")
            
            if tool_calls:
                # LLM wants to use a tool
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    tool_args = func.get("arguments", {})
                    
                    # Parse arguments if string
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except:
                            tool_args = {"query": tool_args}
                    
                    # Fix malformed args from llama4: {"function": "name", "parameters": {...}}
                    if "parameters" in tool_args and isinstance(tool_args.get("parameters"), dict):
                        print(f"🔧 Fixing malformed tool args: unwrapping 'parameters'")
                        tool_args = tool_args["parameters"]
                    
                    print(f"🔧 Tool call: {tool_name}({tool_args})")
                    if not thinking_active:
                        thinking_active = True
                        thinking_start_time = time.time()
                        yield {"type": "thinking_start"}
                    yield {"type": "phase", "content": self._phase_label(tool_name, tool_args)}
                    yield {"type": "tool_call", "name": tool_name, "args": tool_args}
                    
                    # Execute tool
                    executor = TOOL_EXECUTORS.get(tool_name)
                    if executor:
                        try:
                            # Pass query hint for smart truncation in read_document
                            if tool_name == "read_document" and "query_hint" not in tool_args:
                                tool_args["query_hint"] = query
                            result = await executor(tool_args, tenant=self.tenant)
                            # Collect sources from search results
                            if tool_name == "search_documents":
                                all_sources.extend(self._extract_sources(result))
                        except Exception as e:
                            result = f"Fehler bei {tool_name}: {str(e)}"
                            print(f"❌ Tool error: {e}")
                    else:
                        result = f"Unbekanntes Tool: {tool_name}"
                    
                    summary = f"{len(result)} Zeichen" if len(result) > 100 else result[:100]
                    thinking_steps += 1
                    yield {"type": "tool_result", "name": tool_name, "summary": summary}
                    
                    # Add assistant tool_call + tool result to conversation
                    messages.append({
                        "role": "assistant",
                        "content": content or "",
                        "tool_calls": [tc]
                    })
                    messages.append({
                        "role": "tool",
                        "content": result
                    })
            else:
                # LLM wants to answer directly (no tool call)
                if step > 0 and not (step == 1 and forced_search_done):
                    # Already have tool context → stream final answer
                    if thinking_active:
                        elapsed = int(time.time() - thinking_start_time) if thinking_start_time else 0
                        yield {"type": "thinking_end", "steps": thinking_steps, "elapsed": elapsed}
                        thinking_active = False
                    async for evt in self._stream_with_thinking(messages):
                        yield evt
                elif step == 1 and forced_search_done:
                    # After forced search, LLM skipped read_document → force it
                    if all_sources:
                        top_src = all_sources[0]
                        top_path = top_src.get("path", "")
                        if top_path:
                            print(f"🔄 Forced read_document (LLM skipped): {top_path[-60:]}")
                            yield {"type": "phase", "content": f"📄 Lese: *{top_src.get('display_name', top_path)[:50]}*...\n\n"}
                            yield {"type": "tool_call", "name": "read_document", "args": {"path": top_path}}
                            doc_result = await _execute_read_document({"path": top_path, "query_hint": query}, tenant=self.tenant)
                            thinking_steps += 1
                            summary = f"{len(doc_result)} Zeichen"
                            yield {"type": "tool_result", "name": "read_document", "summary": summary}
                            messages.append({
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [{"function": {"name": "read_document", "arguments": {"path": top_path}}}]
                            })
                            messages.append({"role": "tool", "content": doc_result})
                    forced_search_done = False
                    continue
                elif step == 0 and not fs_code and self._needs_search(query):
                    # Step 0, no filesystem query, but looks like a document question
                    # → Force a search before answering
                    if not thinking_active:
                        thinking_active = True
                        thinking_start_time = time.time()
                        yield {"type": "thinking_start"}
                    search_query = query[:200]
                    print(f"🔄 Forced search_documents (LLM skipped tools): {search_query[:80]}")
                    yield {"type": "phase", "content": f"🔍 Suche: *{search_query[:60]}*...\n\n"}
                    
                    search_result = await _execute_search({"query": search_query}, tenant=self.tenant)
                    search_sources = self._extract_sources(search_result)
                    all_sources.extend(search_sources)
                    thinking_steps += 1
                    
                    # Inject search result into conversation
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": {"name": "search_documents", "arguments": {"query": search_query}}}]
                    })
                    messages.append({"role": "tool", "content": search_result})
                    # Hint: tell LLM to read the best document for details
                    messages.append({"role": "system", "content": 
                        "Die Suche hat Treffer gefunden. Die Snippets sind nur Ausschnitte. "
                        "Nutze read_document um das relevanteste Dokument VOLLSTÄNDIG zu lesen "
                        "und exakte Details/Zitate zu finden. Antworte NICHT nur mit Snippet-Infos."})
                    forced_search_done = True
                    # Continue loop – LLM will now read docs or answer with context
                    continue
                elif content:
                    # Simple question or model doesn't support tools
                    if thinking_active:
                        elapsed = int(time.time() - thinking_start_time) if thinking_start_time else 0
                        yield {"type": "thinking_end", "steps": thinking_steps, "elapsed": elapsed}
                        thinking_active = False
                    async for evt in self._stream_with_thinking(messages):
                        yield evt
                else:
                    # Empty response - fallback to streaming
                    if thinking_active:
                        elapsed = int(time.time() - thinking_start_time) if thinking_start_time else 0
                        yield {"type": "thinking_end", "steps": thinking_steps, "elapsed": elapsed}
                        thinking_active = False
                    async for evt in self._stream_with_thinking(messages):
                        yield evt
                
                # Yield sources
                if all_sources:
                    yield {"type": "sources", "sources": all_sources}
                
                yield {"type": "done"}
                return
        
        # Max steps reached - generate final answer with what we have
        if thinking_active:
            elapsed = int(time.time() - thinking_start_time) if thinking_start_time else 0
            yield {"type": "thinking_end", "steps": thinking_steps, "elapsed": elapsed}
            thinking_active = False
        async for evt in self._stream_with_thinking(messages):
            yield evt
        
        if all_sources:
            yield {"type": "sources", "sources": all_sources}
        yield {"type": "done"}
    
    # ------------------------------------------------------------------
    # LLM Calls
    # ------------------------------------------------------------------
    
    def _build_prompt_tools_instruction(self) -> str:
        """Build tool-calling instructions for models that don't support native tools."""
        tool_descs = []
        for t in TOOLS:
            func = t["function"]
            params = func.get("parameters", {}).get("properties", {})
            param_strs = ['"' + k + '": "<' + v.get("description", k) + '>"' for k, v in params.items()]
            param_block = "{" + ", ".join(param_strs) + "}"
            tool_descs.append("- **" + func["name"] + "**: " + func["description"] + "\n  Parameter: " + param_block)
        return (
            "\n\nWICHTIG – TOOL-AUFRUFE:\n"
            "Du hast folgende Tools zur Verfügung:\n" +
            "\n".join(tool_descs) +
            "\n\nWenn du ein Tool nutzen willst, antworte mit GENAU diesem Format:\n"
            '<tool_call>{"name": "tool_name", "arguments": {"param": "wert"}}</tool_call>\n'
            "Du kannst pro Antwort EIN Tool aufrufen. Nach dem Tool-Ergebnis kannst du weitere Tools aufrufen oder die finale Antwort geben.\n"
            "Wenn du KEIN Tool brauchst, antworte direkt mit deiner Antwort (OHNE <tool_call> Tags).\n"
        )
    
    def _parse_prompt_tool_calls(self, content: str) -> list:
        """Parse tool calls from text output of models using prompt-based tool calling."""
        import re
        tool_calls = []
        # Match <tool_call>...</tool_call> blocks
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                name = parsed.get("name", "")
                arguments = parsed.get("arguments", {})
                if name:
                    tool_calls.append({"function": {"name": name, "arguments": arguments}})
            except json.JSONDecodeError:
                print(f"⚠️ Could not parse tool call: {match[:100]}")
        return tool_calls
    
    def _sanitize_messages_for_online(self, messages: list) -> list:
        """Strip document content from messages for privacy-safe online routing.
        Only user queries, system prompt (without memories), and tool call summaries are sent."""
        sanitized = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            
            if role == "system":
                # Strip long-term memories and document-specific content from system prompt
                # Keep only the core instructions up to the LANGZEIT-GEDÄCHTNIS marker
                if "LANGZEIT-GEDÄCHTNIS" in content:
                    content = content[:content.index("LANGZEIT-GEDÄCHTNIS")].rstrip()
                sanitized.append({"role": "system", "content": content})
            elif role == "tool":
                # Replace tool results with a short summary – NO document content goes online
                lines = content.split("\n")
                summary_parts = []
                if "Treffer" in content:
                    for line in lines[:2]:
                        if "Treffer" in line or "Suche" in line:
                            summary_parts.append(line[:150])
                    summary = " ".join(summary_parts) if summary_parts else f"[Tool-Ergebnis: {len(content)} Zeichen]"
                else:
                    summary = f"[Tool-Ergebnis: {len(content)} Zeichen]"
                sanitized.append({"role": "user", "content": f"[Tool-Ergebnis]: {summary}"})
            elif "tool_calls" in m:
                # Keep tool call decisions (no sensitive data)
                tc = m.get("tool_calls", [{}])[0]
                func = tc.get("function", {})
                sanitized.append({"role": "assistant", "content": f"Tool aufgerufen: {func.get('name', '')}({json.dumps(func.get('arguments', {}))})"})
            elif role in ("user", "assistant"):
                # User queries and assistant responses are OK (user typed them)
                sanitized.append({"role": role, "content": content[:2000]})
        return sanitized
    
    async def _llm_with_tools_online(self, messages: list) -> dict:
        """Call an OpenAI-compatible API for fast tool routing. No document content is sent.
        Returns response in Ollama format for compatibility."""
        import httpx
        
        sanitized = self._sanitize_messages_for_online(messages)
        
        # Convert TOOLS to OpenAI format
        openai_tools = []
        for tool in TOOLS:
            openai_tools.append({
                "type": "function",
                "function": tool["function"]
            })
        
        payload = {
            "model": self._online_model,
            "messages": sanitized,
            "tools": openai_tools,
            "tool_choice": "auto",
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        
        total_chars = sum(len(m.get("content", "")) for m in sanitized)
        print(f"☁️ ReAct LLM (online): {total_chars} chars, model={self._online_model}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    f"{self._online_api_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._online_api_key}",
                        "Content-Type": "application/json",
                    }
                )
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            print(f"⚠️ Online strategy failed: {e} → falling back to local")
            return await self._llm_with_tools_local(messages)
        
        # Convert OpenAI response to Ollama format
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        
        ollama_msg = {
            "role": msg.get("role", "assistant"),
            "content": msg.get("content", "") or "",
        }
        
        # Convert tool_calls from OpenAI → Ollama format
        if msg.get("tool_calls"):
            ollama_tc = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"query": args}
                ollama_tc.append({"function": {"name": func.get("name", ""), "arguments": args}})
            ollama_msg["tool_calls"] = ollama_tc
        
        usage = data.get("usage", {})
        print(f"☁️ Online response: tool_calls={len(msg.get('tool_calls', []))}, "
              f"tokens={usage.get('total_tokens', '?')}")
        
        return {"message": ollama_msg, "done": True}
    
    async def _llm_with_tools_local(self, messages: list) -> dict:
        """Local Ollama strategy call (original implementation)."""
        import httpx
        
        model = self.model_strategy
        use_prompt = self._use_prompt_tools_strategy
        
        total_chars = sum(len(m.get("content", "")) for m in messages)
        est_tokens = total_chars // 3
        num_ctx = max(4096, est_tokens + 4096 + 512)
        num_ctx = min(num_ctx, min(65536, self.num_ctx_max))
        
        if use_prompt:
            return await self._llm_with_prompt_tools(messages, num_ctx, total_chars)
        
        payload = {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
            "options": {
                "num_ctx": num_ctx,
                "num_batch": self.num_batch,
                "num_predict": 2048,
                "temperature": 0.2,
            }
        }
        
        timeout = 120.0
        if total_chars > 20000:
            timeout = 300.0
        
        print(f"🔧 ReAct LLM (strategy/local): {total_chars} chars, num_ctx={num_ctx}, model={model}")
        
        last_err = None
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=15.0, read=timeout)) as client:
                    r = await client.post(f"{self.ollama_base}/api/chat", json=payload)
                    r.raise_for_status()
                    return r.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    print(f"⚠️ Model {model} returned 400 with tools → switching to prompt-based tool calling")
                    self._use_prompt_tools_strategy = True
                    _mark_prompt_tools(model)
                    return await self._llm_with_prompt_tools(messages, num_ctx, total_chars)
                raise LLMError(f"HTTP {e.response.status_code}: {e}")
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadError) as e:
                last_err = e
                if attempt == 0:
                    print(f"⚠️ LLM call failed (attempt 1/2): {type(e).__name__}: {e}")
                    print(f"🔄 Retrying with +60s timeout...")
                    timeout += 60.0
                    await asyncio.sleep(2)
                else:
                    print(f"❌ LLM call failed (attempt 2/2): {type(e).__name__}: {e}")
        raise LLMError(f"LLM nicht erreichbar nach 2 Versuchen: {type(last_err).__name__}")
    
    async def _llm_with_tools(self, messages: list) -> dict:
        """Non-streaming LLM call with tool definitions. Routes to online or local strategy model."""
        # Route to online API if enabled (privacy-safe: no doc content sent)
        if self._online_enabled:
            return await self._llm_with_tools_online(messages)
        return await self._llm_with_tools_local(messages)
    
    def _augment_messages_for_prompt_tools(self, messages: list) -> list:
        """Convert messages to prompt-based format (inject tool instructions, convert tool roles)."""
        tool_instruction = self._build_prompt_tools_instruction()
        augmented = []
        injected = False
        for m in messages:
            if m["role"] == "system" and not injected:
                augmented.append({"role": "system", "content": m["content"] + tool_instruction})
                injected = True
            elif m["role"] == "tool":
                augmented.append({"role": "user", "content": f"[Tool-Ergebnis]:\n{m['content']}"})
            elif "tool_calls" in m:
                tc = m.get("tool_calls", [{}])[0]
                func = tc.get("function", {})
                tc_name = func.get("name", "")
                tc_args = json.dumps(func.get("arguments", {}))
                tc_text = '<tool_call>{"name": "' + tc_name + '", "arguments": ' + tc_args + '}</tool_call>'
                augmented.append({"role": "assistant", "content": tc_text})
            else:
                augmented.append({"role": m["role"], "content": m.get("content", "")})
        return augmented
    
    async def _llm_with_prompt_tools(self, messages: list, num_ctx: int, total_chars: int) -> dict:
        """Prompt-based tool calling using STREAMING internally to avoid timeouts with reasoning models.
        Collects the full response, strips <think> blocks, then parses tool calls from text."""
        import httpx
        import re
        
        augmented_messages = self._augment_messages_for_prompt_tools(messages)
        
        # Use strategy model for prompt-based tool calling too
        model = self.model_strategy
        
        payload = {
            "model": model,
            "messages": augmented_messages,
            "stream": True,
            "options": {
                "num_ctx": num_ctx,
                "num_batch": self.num_batch,
                "num_predict": 2048,
                "temperature": 0.2,
            }
        }
        
        timeout = 600.0  # Long timeout for reasoning models
        
        print(f"🔧 ReAct LLM (prompt-tools/stream): {total_chars} chars, num_ctx={num_ctx}, model={model}")
        
        # Stream and collect full response
        full_content = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30.0, read=120.0)) as client:
            async with client.stream("POST", f"{self.ollama_base}/api/chat", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        token = obj.get("message", {}).get("content", "")
                        if token:
                            full_content.append(token)
                    except:
                        pass
        
        content = "".join(full_content)
        print(f"🔧 Prompt-tools response: {len(content)} chars collected")
        
        # Strip <think>...</think> blocks (DeepSeek-R1 reasoning)
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # Parse tool calls from text
        tool_calls = self._parse_prompt_tool_calls(clean_content)
        
        result = {"message": {"role": "assistant", "content": clean_content}}
        
        if tool_calls:
            # Also strip <tool_call> tags from display content
            display_content = re.sub(r'<tool_call>.*?</tool_call>', '', clean_content, flags=re.DOTALL).strip()
            result["message"]["tool_calls"] = tool_calls
            result["message"]["content"] = display_content
            print(f"🔧 Parsed {len(tool_calls)} tool call(s) from text: {[tc['function']['name'] for tc in tool_calls]}")
        
        return result
    
    async def _llm_stream_final(self, messages: list) -> AsyncGenerator[str, None]:
        """Streaming LLM call for final answer (no tools). Retries once on timeout/connection error."""
        import httpx
        
        total_chars = sum(len(m.get("content", "")) for m in messages)
        est_tokens = total_chars // 3
        num_ctx = max(4096, est_tokens + 8192 + 512)
        num_ctx = min(num_ctx, self.num_ctx_max)
        
        # Remove tool_calls from messages for clean streaming
        # For prompt-based models: convert tool/tool_calls messages to plain text
        clean_messages = []
        for m in messages:
            role = m["role"]
            content = m.get("content", "")
            if role == "tool":
                # Convert tool results to user message
                clean_messages.append({"role": "user", "content": f"[Tool-Ergebnis]:\n{content}"})
            elif "tool_calls" in m:
                # Convert tool_call to assistant text
                tc = m.get("tool_calls", [{}])[0]
                func = tc.get("function", {})
                tc_name = func.get("name", "")
                tc_args = json.dumps(func.get("arguments", {}))
                clean_messages.append({"role": "assistant", "content": f"Tool aufgerufen: {tc_name}({tc_args})"})
            else:
                clean_messages.append({"role": role, "content": content})
        
        # Final answer uses the answer model (big model for quality)
        answer_model = self.model_answer
        
        payload = {
            "model": answer_model,
            "messages": clean_messages,
            "stream": True,
            "options": {
                "num_ctx": num_ctx,
                "num_batch": self.num_batch,
                "temperature": 0.3,
                "num_predict": 8192,
            }
        }
        
        timeout = 300.0
        if total_chars > 20000:
            timeout = 600.0
        # Reasoning models need much longer for first token (internal <think> phase)
        if self._use_prompt_tools_answer:
            timeout = max(timeout, 600.0)
        
        print(f"🔧 ReAct stream (answer): {total_chars} chars, num_ctx={num_ctx}, timeout={timeout}s, model={answer_model}")
        
        last_err = None
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=15.0, read=timeout)) as client:
                    async with client.stream(
                        "POST",
                        f"{self.ollama_base}/api/chat",
                        json=payload
                    ) as r:
                        r.raise_for_status()
                        got_tokens = False
                        async for line in r.aiter_lines():
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                content = obj.get("message", {}).get("content", "")
                                if content:
                                    got_tokens = True
                                    yield content
                            except:
                                pass
                        if got_tokens:
                            return
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadError) as e:
                last_err = e
                if attempt == 0:
                    print(f"⚠️ LLM stream failed (attempt 1/2): {type(e).__name__}: {e}")
                    print(f"🔄 Retrying stream with +60s timeout...")
                    timeout += 60.0
                    import asyncio
                    await asyncio.sleep(2)
                else:
                    print(f"❌ LLM stream failed (attempt 2/2): {type(e).__name__}: {e}")
                    raise LLMError(f"LLM-Streaming fehlgeschlagen nach 2 Versuchen: {type(last_err).__name__}")
    
    async def _llm_with_tools_keepalive(self, messages: list, hint: str = "") -> AsyncGenerator[dict, None]:
        """Wrap _llm_with_tools: yields keepalive events every 5s while waiting for LLM."""
        task = asyncio.create_task(self._llm_with_tools(messages))
        # Determine context hint for keepalive messages
        if not hint:
            # Derive hint from last message context
            last_tool = None
            for m in reversed(messages):
                if m.get("role") == "tool":
                    last_tool = "Analysiere Suchergebnisse"
                    break
                if m.get("role") == "user":
                    hint = "Analysiere Anfrage"
                    break
            hint = last_tool or hint or "Plane nächsten Schritt"
        start = time.time()
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=5.0)
            if done:
                break
            elapsed = int(time.time() - start)
            yield {"type": "keepalive", "elapsed": elapsed, "hint": hint}
        
        if task.cancelled():
            raise LLMError("LLM-Aufruf abgebrochen")
        exc = task.exception()
        if exc:
            raise exc
        yield {"type": "result", "data": task.result()}
    
    async def _stream_with_thinking(self, messages: list) -> AsyncGenerator[dict, None]:
        """Stream final answer, separating <think> blocks into structured events for reasoning models."""
        buffer = ""
        in_think = False
        
        async for token in self._llm_stream_final(messages):
            buffer += token
            
            while True:
                if not in_think:
                    idx = buffer.find("<think>")
                    if idx != -1:
                        before = buffer[:idx]
                        if before:
                            yield {"type": "token", "content": before}
                        yield {"type": "reasoning_start"}
                        buffer = buffer[idx + 7:]
                        in_think = True
                        continue
                    else:
                        safe = max(0, len(buffer) - 7)
                        if safe > 0:
                            yield {"type": "token", "content": buffer[:safe]}
                            buffer = buffer[safe:]
                        break
                else:
                    idx = buffer.find("</think>")
                    if idx != -1:
                        before = buffer[:idx]
                        if before:
                            yield {"type": "reasoning_token", "content": before}
                        yield {"type": "reasoning_end"}
                        buffer = buffer[idx + 8:]
                        in_think = False
                        continue
                    else:
                        safe = max(0, len(buffer) - 8)
                        if safe > 0:
                            yield {"type": "reasoning_token", "content": buffer[:safe]}
                            buffer = buffer[safe:]
                        break
        
        # Flush remaining buffer
        if buffer:
            if in_think:
                yield {"type": "reasoning_token", "content": buffer}
                yield {"type": "reasoning_end"}
            else:
                yield {"type": "token", "content": buffer}
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    
    def _auto_filesystem_code(self, query: str) -> str:
        """Generate Python code for filesystem queries. Returns empty string if not a fs query."""
        import re
        q = query.lower()
        
        # Detect file extension from query
        ext_match = re.search(r'\.(pdf|eml|docx|doc|msg|xlsx|xls|pptx|txt|md)', q)
        ext = ext_match.group(1) if ext_match else None
        
        # Count files queries
        if re.search(r'(?:wie\s*viele|anzahl|zähl|count)\s+.*?(?:dateien|files|dokumente|mails|pdf|eml|docx|msg)', q):
            ext_filter = f"f.lower().endswith('.{ext}')" if ext else "True"
            ext_label = f".{ext}" if ext else ""
            return f"""import os
from collections import Counter

counts = Counter()
total = 0
for root, dirs, files in os.walk(DATA_ROOT):
    for f in files:
        if {ext_filter}:
            rel = os.path.relpath(root, DATA_ROOT)
            # Top-level folder only
            top = rel.split(os.sep)[0] if os.sep in rel else rel
            counts[top] += 1
            total += 1

print(f"Gesamt: {{total}} {ext_label}-Dateien im Projektarchiv")
print()
print("Pro Hauptordner:")
for folder, n in counts.most_common(20):
    print(f"  {{folder}}: {{n}}")
result = f"{{total}} {ext_label}-Dateien gefunden"
"""
        
        # List files queries
        if re.search(r'(?:liste|zeige|finde|suche)\s+(?:alle|sämtliche)\s+.*?(?:dateien|files)', q):
            ext_filter = f"f.lower().endswith('.{ext}')" if ext else "True"
            return f"""import os

files_found = []
for root, dirs, files in os.walk(DATA_ROOT):
    for f in files:
        if {ext_filter}:
            rel = os.path.relpath(os.path.join(root, f), DATA_ROOT)
            files_found.append(rel)

files_found.sort()
print(f"Gefunden: {{len(files_found)}} Dateien")
print()
for fp in files_found[:100]:
    print(f"  {{fp}}")
if len(files_found) > 100:
    print(f"  ... und {{len(files_found) - 100}} weitere")
result = f"{{len(files_found)}} Dateien gefunden"
"""
        
        # Directory structure queries
        if re.search(r'(?:ordnerstruktur|verzeichnisstruktur|dateistruktur|welche.*ordner|welche.*verzeichnisse)', q):
            return """import os

print("Ordnerstruktur (Ebene 1+2):")
print()
for item in sorted(os.listdir(DATA_ROOT)):
    path = os.path.join(DATA_ROOT, item)
    if os.path.isdir(path):
        sub_count = len(os.listdir(path))
        print(f"📁 {item}/ ({sub_count} Einträge)")
        for sub in sorted(os.listdir(path))[:10]:
            sub_path = os.path.join(path, sub)
            if os.path.isdir(sub_path):
                print(f"   📁 {sub}/")
            else:
                print(f"   📄 {sub}")
        if sub_count > 10:
            print(f"   ... und {sub_count - 10} weitere")
"""
        
        return ""
    
    def _analyze_query(self, query: str) -> str:
        """Analyze query and return tool hint if a specific tool is clearly needed"""
        import re
        q = query.lower()
        
        # Filesystem queries → execute_python
        fs_patterns = [
            r'(?:wie\s*viele|anzahl|zähl|count)\s+.*?(?:dateien|files|dokumente|mails|eml|pdf|docx|msg)',
            r'(?:liste|zeige|finde|suche)\s+(?:alle|sämtliche)\s+.*?(?:dateien|files)',
            r'(?:welche|was für)\s+(?:dateien|ordner|verzeichnisse)',
            r'(?:ordnerstruktur|verzeichnisstruktur|dateistruktur)',
            r'(?:gibt\s*es|existieren)\s+.*?(?:\.eml|\.pdf|\.docx|\.msg|\.xlsx)',
            r'(?:pro\s+ordner|pro\s+unterordner|pro\s+verzeichnis)',
        ]
        for p in fs_patterns:
            if re.search(p, q):
                return ("Diese Frage erfordert eine Dateisystem-Analyse. "
                        "Nutze execute_python mit os.walk(DATA_ROOT) wobei DATA_ROOT='/data'. "
                        "search_documents zeigt NUR indizierte Treffer, NICHT alle Dateien!")
        
        # Data analysis → execute_python
        data_patterns = [
            r'(?:berechn|statistik|durchschnitt|summe|mittelwert)',
            r'(?:csv|excel|xlsx)\s+.*?(?:analys|auswert|einles)',
            r'(?:analys|auswert)\s+.*?(?:csv|excel|xlsx|daten)',
        ]
        for p in data_patterns:
            if re.search(p, q):
                return ("Diese Frage erfordert Datenanalyse. "
                        "Nutze execute_python mit pandas. Dateien liegen unter DATA_ROOT='/data'.")
        
        # Transcript/Protocol → create_protocol
        # Only if long text is included (>500 chars after instruction)
        proto_patterns = [
            r'(?:erstell|schreib|mach|generier)\w*\s+.*?(?:protokoll|niederschrift)',
            r'transkript\w*\s+.*?(?:protokoll|aufbereite|verarbeit)',
        ]
        if len(query) > 500:
            for p in proto_patterns:
                if re.search(p, q):
                    return ("Der Benutzer möchte ein Protokoll aus einem Transkript erstellen. "
                            "Nutze create_protocol mit dem GESAMTEN Text als transcript-Parameter.")
        
        return ""
    
    def _needs_search(self, query: str) -> bool:
        """Check if query likely needs document search (vs. pure chat/greeting)"""
        import re
        q = query.lower().strip()
        # Skip search for greetings, simple chat, meta-questions
        skip_patterns = [
            r'^(hallo|hi|hey|guten\s*(tag|morgen|abend)|servus|grüezi)\b',
            r'^(danke|merci|vielen\s*dank)',
            r'^(wie\s*geht|was\s*kannst\s*du|wer\s*bist\s*du|hilfe|help)',
            r'^(ja|nein|ok|gut|genau|stimmt|richtig)$',
        ]
        for p in skip_patterns:
            if re.search(p, q):
                return False
        # Skip forced doc search for memory operations
        memory_patterns = [
            r'merk\s*dir', r'vergiss', r'erinner', r'notiz',
            r'was\s+(hast|weisst)\s+du\s+(dir\s+)?gemerkt',
            r'zeig.*notiz', r'lösch.*notiz',
        ]
        for p in memory_patterns:
            if re.search(p, q):
                print(f"⏭️ _needs_search=False: memory operation")
                return False
        # Skip forced doc search when user explicitly wants web search
        web_patterns = [
            r'(suche|such)\s*(im\s+)?internet',
            r'web.?such', r'online\s+such',
            r'google', r'im\s+netz\b',
        ]
        for p in web_patterns:
            if re.search(p, q):
                print(f"⏭️ _needs_search=False: explicit web search request")
                return False
        # Most queries benefit from search
        return len(q) > 10
    
    def _phase_label(self, tool_name: str, args: dict) -> str:
        """Human-readable phase label for UI"""
        if tool_name == "search_documents":
            q = args.get("query", "")[:60]
            return f"🔍 Suche: *{q}*...\n\n"
        elif tool_name == "read_document":
            p = args.get("path", "").split("/")[-1][:50]
            return f"📄 Lese: *{p}*...\n\n"
        elif tool_name == "execute_python":
            d = args.get("description", "Code")[:50]
            return f"⚙️ Code: *{d}*...\n\n"
        elif tool_name == "create_protocol":
            return "📝 Erstelle Protokoll...\n\n"
        elif tool_name == "list_files":
            p = args.get("path", "/")[:50]
            return f"📂 Ordner: *{p}*...\n\n"
        elif tool_name == "read_file":
            p = args.get("path", "").split("/")[-1][:50]
            return f"📄 Lese Datei: *{p}*...\n\n"
        elif tool_name == "web_search":
            q = args.get("query", "")[:60]
            return f"🌐 Web-Suche: *{q}*...\n\n"
        elif tool_name == "manage_memory":
            action = args.get("action", "")
            labels = {"save": "💾 Speichere Notiz", "list": "📋 Notizen abrufen", "search": "🔍 Notizen suchen", "delete": "🗑️ Notiz löschen"}
            return f"{labels.get(action, '🧠 Gedächtnis')}...\n\n"
        return f"🔧 {tool_name}...\n\n"
    
    def _extract_sources(self, search_result: str) -> list:
        """Extract source paths from search result text with snippet previews and dedup."""
        import re
        sources = []
        seen_filenames = set()
        file_base = self.tenant.document_root if self.tenant else os.getenv("FILE_BASE", "/media/felix/RAG/1")
        
        # Parse [N] path\nsnippet blocks
        blocks = re.split(r'(?=\[\d+\]\s)', search_result)
        for block in blocks:
            m = re.match(r'\[(\d+)\]\s+(.+?)(?:\n([\s\S]*?))?$', block.strip())
            if not m:
                continue
            n = int(m.group(1))
            path = m.group(2).strip()
            snippet = (m.group(3) or "").strip()[:150]
            
            if not path:
                continue
            # Dedup by filename (not full path) to catch duplicate entries with different prefixes
            filename = os.path.basename(path)
            if filename in seen_filenames:
                continue
            seen_filenames.add(filename)
            
            from urllib.parse import quote
            encoded = quote(f"{file_base}/{path}", safe="/:@")
            
            # Build display name: folder/filename
            parts = path.rsplit("/", 2)
            if len(parts) >= 2:
                display_name = f"{parts[-2]}/{parts[-1]}"
            else:
                display_name = parts[-1]
            
            sources.append({
                "n": n,
                "path": path,
                "display_name": display_name,
                "display_path": f"/{path}",
                "snippet_preview": snippet,
                "local_url": f"http://localhost:11436/open?path={encoded}"
            })
        
        return sources
