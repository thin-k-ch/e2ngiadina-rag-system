import traceback
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Req(BaseModel):
    code: str
    locals: dict | None = None

@app.post("/run")
def run(req: Req):
    try:
        loc = dict(req.locals or {})
        glob = {"__builtins__": __builtins__}
        exec(req.code, glob, loc)
        safe = {}
        for k, v in loc.items():
            if k == "__builtins__":
                continue
            try:
                import json
                json.dumps(v)
                safe[k] = v
            except Exception:
                safe[k] = str(v)
        return {"ok": True, "locals": safe}
    except Exception:
        return {"ok": False, "error": traceback.format_exc()}
