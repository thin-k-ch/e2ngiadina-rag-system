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
