#!/usr/bin/env python3
"""
GPU Model Manager — tiny HTTP server to manage llama-server on the host.
Runs on port 8099, called by Agent API (from Docker) via host.docker.internal:8099.

Endpoints:
  GET  /models   — list available GPU models + current status
  POST /switch   — switch to a different model (restarts llama-server)
  GET  /health   — simple health check
"""

import json
import os
import signal
import subprocess
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLAMA_SERVER = os.getenv("LLAMA_SERVER_BIN", "/media/felix/RAG/llama.cpp/build/bin/llama-server")
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "8090"))
MANAGER_PORT = int(os.getenv("MANAGER_PORT", "8099"))

# Available models: name → llama-server args
MODELS = {
    "gpt-oss-120b": {
        "hf_repo": "ggml-org/gpt-oss-120b-GGUF",
        "description": "General Purpose (120B, 63 GB)",
        "size_gb": 63,
        "type": "general",
    },
    "deepseek-r1-70b": {
        "hf_repo": "unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF",
        "hf_file": "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf",
        "description": "Reasoning / Thinking (70B, 40 GB)",
        "size_gb": 40,
        "type": "reasoning",
    },
    "qwen3.5-35b-a3b": {
        "hf_repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
        "hf_file": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "description": "MoE Thinking (35B/3B active, 18 GB)",
        "size_gb": 18,
        "type": "reasoning",
    },
}

# Default llama-server arguments
DEFAULT_ARGS = [
    "-ngl", "999",
    "--no-mmap",
    "-fa", "on",
    "--ctx-size", "32768",
    "--port", str(LLAMA_PORT),
    "--host", "0.0.0.0",
    "-b", "2048",
    "-ub", "2048",
    "--jinja",
    "-np", "1",
]

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

current_model = None
current_process = None
switching = False


def find_current_llama_server():
    """Find an already-running llama-server process."""
    global current_model, current_process
    try:
        result = subprocess.run(
            ["pgrep", "-af", "llama-server"],
            capture_output=True, text=True
        )
        for line in result.stdout.strip().split("\n"):
            if "llama-server" in line and str(LLAMA_PORT) in line:
                # Extract HF repo from args
                for name, cfg in MODELS.items():
                    if cfg["hf_repo"] in line:
                        current_model = name
                        pid = int(line.split()[0])
                        print(f"✅ Found existing llama-server: {name} (PID {pid})")
                        return
                # Unknown model
                current_model = "unknown"
                print(f"⚠️ Found llama-server but unknown model: {line[:100]}")
                return
    except Exception as e:
        print(f"Error finding llama-server: {e}")


def stop_llama_server():
    """Stop the current llama-server process."""
    global current_process
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"llama-server.*{LLAMA_PORT}"],
            capture_output=True, text=True
        )
        pids = [int(p) for p in result.stdout.strip().split("\n") if p.strip()]
        for pid in pids:
            print(f"🛑 Stopping llama-server PID {pid}")
            os.kill(pid, signal.SIGTERM)
        # Wait for processes to die
        for _ in range(30):  # 30 seconds max
            result = subprocess.run(
                ["pgrep", "-f", f"llama-server.*{LLAMA_PORT}"],
                capture_output=True, text=True
            )
            if not result.stdout.strip():
                print("✅ llama-server stopped")
                return True
            time.sleep(1)
        # Force kill
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return True
    except Exception as e:
        print(f"Error stopping llama-server: {e}")
        return False


def start_llama_server(model_name):
    """Start llama-server with the given model."""
    global current_model, current_process
    cfg = MODELS.get(model_name)
    if not cfg:
        return False, f"Unknown model: {model_name}"

    args = [LLAMA_SERVER, "-hf", cfg["hf_repo"]]
    if "hf_file" in cfg:
        args.extend(["-hff", cfg["hf_file"]])
    args.extend(DEFAULT_ARGS)

    print(f"🚀 Starting llama-server: {model_name}")
    print(f"   Command: {' '.join(args)}")

    try:
        current_process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # Wait for server to be ready (check /health endpoint)
        import urllib.request
        for i in range(120):  # 2 minutes max
            time.sleep(2)
            if current_process.poll() is not None:
                return False, f"llama-server exited with code {current_process.returncode}"
            try:
                r = urllib.request.urlopen(f"http://localhost:{LLAMA_PORT}/health", timeout=2)
                if r.status == 200:
                    current_model = model_name
                    print(f"✅ llama-server ready: {model_name} (PID {current_process.pid})")
                    return True, f"Model {model_name} loaded successfully"
            except Exception:
                if i % 10 == 0:
                    print(f"   Waiting for llama-server... ({i*2}s)")
                continue

        return False, "Timeout waiting for llama-server to start"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == "/models":
            models = []
            for name, cfg in MODELS.items():
                models.append({
                    "name": name,
                    "hf_repo": cfg["hf_repo"],
                    "description": cfg["description"],
                    "size_gb": cfg["size_gb"],
                    "type": cfg["type"],
                    "active": name == current_model,
                })
            self._json({
                "models": models,
                "current": current_model,
                "switching": switching,
            })
        elif self.path == "/health":
            self._json({"status": "ok", "current_model": current_model})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/switch":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            model_name = body.get("model", "")

            if not model_name or model_name not in MODELS:
                self._json({"error": f"Unknown model: {model_name}"}, 400)
                return

            if model_name == current_model:
                self._json({"status": "ok", "message": "Already loaded", "model": model_name})
                return

            global switching
            if switching:
                self._json({"error": "Already switching model, please wait"}, 409)
                return

            # Start switch in background thread
            switching = True
            self._json({"status": "switching", "message": f"Switching to {model_name}...", "model": model_name})

            def do_switch():
                global switching
                try:
                    stop_llama_server()
                    ok, msg = start_llama_server(model_name)
                    if ok:
                        print(f"✅ Switch complete: {model_name}")
                    else:
                        print(f"❌ Switch failed: {msg}")
                finally:
                    switching = False

            Thread(target=do_switch, daemon=True).start()
        else:
            self._json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        print(f"[Manager] {args[0]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    find_current_llama_server()
    print(f"🎛️ GPU Model Manager starting on port {MANAGER_PORT}")
    print(f"   Current model: {current_model or 'none'}")
    print(f"   Available: {', '.join(MODELS.keys())}")

    server = HTTPServer(("0.0.0.0", MANAGER_PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Manager stopped")
        server.server_close()
