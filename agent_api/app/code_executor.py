"""
Code Executor - Client for the PyRunner sandbox service.

Sends Python code to the runner container for safe execution.
Returns stdout, result variables, and error information.
"""

import os
import re
import httpx
from typing import Optional, Dict, Any, Tuple

PYRUNNER_URL = os.getenv("PYRUNNER_URL", "http://runner:9000/run")
PYRUNNER_TIMEOUT = int(os.getenv("PYRUNNER_TIMEOUT", "30"))


async def execute_code(code: str, timeout: int = None, locals_dict: dict = None) -> Dict[str, Any]:
    """
    Execute Python code in the sandboxed runner.
    
    Returns:
        {
            "ok": bool,
            "stdout": str,       # print() output
            "stderr": str,       # warnings/errors
            "result": Any,       # value of 'result' variable if set
            "locals": dict,      # other variables
            "error": str | None  # error traceback if failed
        }
    """
    payload = {
        "code": code,
        "timeout": timeout or PYRUNNER_TIMEOUT,
    }
    if locals_dict:
        payload["locals"] = locals_dict
    
    try:
        async with httpx.AsyncClient(timeout=PYRUNNER_TIMEOUT + 5) as client:
            r = await client.post(PYRUNNER_URL, json=payload)
            return r.json()
    except httpx.ConnectError:
        return {"ok": False, "error": "PyRunner service not available", "stdout": "", "stderr": ""}
    except httpx.TimeoutException:
        return {"ok": False, "error": f"PyRunner timeout after {PYRUNNER_TIMEOUT}s", "stdout": "", "stderr": ""}
    except Exception as e:
        return {"ok": False, "error": str(e), "stdout": "", "stderr": ""}


def extract_code_blocks(text: str) -> list[Tuple[str, str]]:
    """
    Extract Python code blocks from LLM output.
    Returns list of (language, code) tuples.
    
    Matches: ```python ... ``` and ```py ... ```
    """
    pattern = r'```(python|py)\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [(lang, code.strip()) for lang, code in matches]


def format_execution_result(result: Dict[str, Any]) -> str:
    """
    Format execution result for display in chat.
    """
    parts = []
    
    if result.get("ok"):
        stdout = result.get("stdout", "").strip()
        stderr = result.get("stderr", "").strip()
        res = result.get("result")
        
        if stdout:
            parts.append(stdout)
        if res is not None:
            if isinstance(res, str):
                parts.append(res)
            else:
                import json
                parts.append(json.dumps(res, indent=2, ensure_ascii=False))
        if stderr:
            parts.append(f"⚠️ {stderr}")
        
        if not parts:
            parts.append("✅ Code ausgeführt (keine Ausgabe)")
    else:
        error = result.get("error", "Unknown error")
        stdout = result.get("stdout", "").strip()
        parts.append(f"❌ Fehler:\n```\n{error}\n```")
        if stdout:
            parts.append(f"Bisherige Ausgabe:\n{stdout}")
    
    return "\n".join(parts)


def detect_code_request(query: str) -> bool:
    """
    Detect if user query is asking for code execution / data analysis.
    """
    query_lower = query.lower()
    
    code_indicators = [
        "python", "script", "code", "berechne", "berechnung",
        "analysiere die daten", "datenanalyse", "statistik",
        "zähle", "zählung", "count",
        "liste alle dateien", "verzeichnis", "ordner",
        "csv", "excel", "xlsx",
        "führe aus", "execute", "run code",
    ]
    
    return any(indicator in query_lower for indicator in code_indicators)


def detect_filesystem_query(query: str) -> Optional[str]:
    """
    Detect if query requires filesystem operations (counting, listing, browsing files).
    These queries MUST use Python code execution, not search results.
    
    Returns a code-execution instruction string if detected, None otherwise.
    """
    query_lower = query.lower()
    
    # Patterns that indicate filesystem operations
    fs_patterns = [
        # Counting files
        (r'(?:wie\s*viele|anzahl|zähl|count)\s+.*?(?:dateien|files|dokumente|mails|eml|pdf|docx|msg)',
         "ZÄHLE die Dateien auf dem Dateisystem"),
        # Listing files by type
        (r'(?:liste|zeige|finde|suche)\s+(?:alle|sämtliche)\s+.*?(?:\.?\w{2,4})\s*(?:dateien|files)',
         "LISTE die Dateien vom Dateisystem auf"),
        # Directory browsing
        (r'(?:welche|was für)\s+(?:dateien|ordner|verzeichnisse)',
         "DURCHSUCHE das Dateisystem"),
        # File type queries  
        (r'(?:gibt\s*es|existieren|vorhanden)\s+.*?(?:\.eml|\.pdf|\.docx|\.msg|\.xlsx)\s*(?:dateien|files)?',
         "PRÜFE das Dateisystem"),
        # Explicit file counting with extension
        (r'(?:\.eml|\.pdf|\.docx|\.msg|\.xlsx|\.pptx)\s*(?:dateien|files)?\s*(?:im|in|unter|auf)',
         "ZÄHLE die Dateien auf dem Dateisystem"),
        # "how many files in folder X"
        (r'(?:wie\s*viele|anzahl)\s+.*?(?:im\s+archiv|im\s+system|im\s+projektarchiv|in\s+der\s+ablage|pro\s+ordner|pro\s+unterordner)',
         "ZÄHLE auf dem Dateisystem"),
    ]
    
    for pattern, hint in fs_patterns:
        if re.search(pattern, query_lower):
            return hint
    
    return None


FILESYSTEM_CODE_INSTRUCTION = """
WICHTIG: Diese Frage erfordert eine Dateisystem-Analyse. Die Suchresultate zeigen nur EINIGE Treffer, NICHT alle Dateien!
Du MUSST einen ```python Code-Block schreiben, der das Dateisystem unter DATA_ROOT='/data' durchsucht.

Beispiel für Dateien zählen:
```python
import os
from collections import Counter
counts = Counter()
total = 0
for root, dirs, files in os.walk(DATA_ROOT):
    for f in files:
        if f.lower().endswith('.eml'):  # Anpassen je nach Frage
            rel = os.path.relpath(root, DATA_ROOT)
            counts[rel] += 1
            total += 1
print(f"Gesamt: {total} Dateien")
for folder, n in counts.most_common(10):
    print(f"  {folder}: {n}")
result = f"{total} Dateien gefunden"
```

Antworte NICHT basierend auf den Suchresultaten - nutze IMMER Python-Code für Dateisystem-Fragen!
"""
