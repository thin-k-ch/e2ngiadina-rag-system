import os
import io
import sys
import json
import threading
import traceback
from contextlib import redirect_stdout, redirect_stderr
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="PyRunner - Sandboxed Python Executor")

TIMEOUT = int(os.getenv("TIMEOUT_SECONDS", "25"))
DATA_ROOT = os.getenv("DATA_ROOT", "/data")


class Req(BaseModel):
    code: str
    locals: dict | None = None
    timeout: int | None = None


@app.get("/health")
def health():
    return {"status": "ok", "timeout": TIMEOUT, "data_root": DATA_ROOT}


@app.post("/run")
def run(req: Req):
    timeout = min(req.timeout or TIMEOUT, 60)  # Hard cap at 60s
    
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    
    # Pre-import useful libraries into the execution namespace
    import_ns = {}
    try:
        import pandas as pd
        import_ns["pd"] = pd
    except ImportError:
        pass
    try:
        from tabulate import tabulate
        import_ns["tabulate"] = tabulate
    except ImportError:
        pass
    try:
        import csv
        import_ns["csv"] = csv
    except ImportError:
        pass
    
    loc = dict(req.locals or {})
    loc.update(import_ns)
    loc["DATA_ROOT"] = DATA_ROOT
    
    glob = {"__builtins__": __builtins__}
    
    # Use threading for timeout (signal.alarm doesn't work in non-main threads)
    exec_result = {"done": False, "error": None}
    
    def _exec_code():
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(req.code, glob, loc)
            exec_result["done"] = True
        except Exception:
            exec_result["error"] = traceback.format_exc()
            exec_result["done"] = True
    
    thread = threading.Thread(target=_exec_code)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        # Timeout - thread is still running
        return {
            "ok": False,
            "error": f"Timeout: Code execution exceeded {timeout}s limit",
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        }
    
    if exec_result["error"]:
        return {
            "ok": False,
            "error": exec_result["error"],
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        }
    
    # Collect serializable locals (skip internal stuff)
    safe = {}
    skip_keys = set(import_ns.keys()) | {"DATA_ROOT"}
    for k, v in loc.items():
        if k.startswith("_") or k in skip_keys:
            continue
        try:
            json.dumps(v)
            safe[k] = v
        except Exception:
            safe[k] = str(v)
    
    stdout_text = stdout_buf.getvalue()
    stderr_text = stderr_buf.getvalue()
    
    # If there's a 'result' variable, use it as the primary output
    result = safe.pop("result", None)
    
    return {
        "ok": True,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "result": result,
        "locals": safe,
    }
