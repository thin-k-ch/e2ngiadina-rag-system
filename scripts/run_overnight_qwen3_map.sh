#!/usr/bin/env bash
# =================================================================
# Overnight Batch Run: Qwen3-235B MAP → gpt-oss-120b REDUCE+DRAFT
# =================================================================
#
# This script orchestrates a 2-model batch run:
#   Phase 1: MAP with Qwen3-235B-A22B (Q3_K_M, 112GB)
#   Phase 2: Switch llama-server to gpt-oss-120b
#   Phase 3: REDUCE + DRAFT with gpt-oss-120b (63GB)
#
# Prerequisites:
#   - Qwen3 GGUF downloaded to /media/felix/RAG/models/Qwen3-235B-A22B-GGUF/
#   - llama.cpp built at /media/felix/RAG/llama.cpp/build/bin/llama-server
#   - Elasticsearch running at localhost:9200
#   - Ollama models unloaded (GPU memory free)
#
# Usage:
#   nohup bash scripts/run_overnight_qwen3_map.sh > runs/overnight_qwen3.log 2>&1 &

set -euo pipefail

LLAMA_SERVER="/media/felix/RAG/llama.cpp/build/bin/llama-server"
BATCH_SCRIPT="/media/felix/RAG/AGENTIC/scripts/batch_report.py"
OUTLINE="/media/felix/RAG/AGENTIC/scripts/outline_schlussbericht.md"
RUN_DIR="/media/felix/RAG/AGENTIC/runs"

# Model paths
QWEN3_GGUF="/media/felix/RAG/models/Qwen3-235B-A22B-GGUF/Q3_K_M/Qwen3-235B-A22B-Q3_K_M-00001-of-00003.gguf"
GPT_OSS_HF="ggml-org/gpt-oss-120b-GGUF"

# Ports
PORT=8090

# Run ID for this overnight run
RUN_ID="$(date +%Y%m%d_%H%M%S)_qwen3map"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

kill_llama_server() {
    log "Stopping llama-server..."
    pkill -f "llama-server.*--port ${PORT}" 2>/dev/null || true
    sleep 5
    # Force kill if still running
    pkill -9 -f "llama-server.*--port ${PORT}" 2>/dev/null || true
    sleep 2
}

wait_for_server() {
    local max_wait=${1:-300}
    local elapsed=0
    log "Waiting for llama-server on port ${PORT} (max ${max_wait}s)..."
    while ! curl -s "http://localhost:${PORT}/health" | grep -q '"ok"' 2>/dev/null; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ "$elapsed" -ge "$max_wait" ]; then
            log "ERROR: llama-server did not start within ${max_wait}s"
            return 1
        fi
        if [ $((elapsed % 30)) -eq 0 ]; then
            log "  Still waiting... (${elapsed}s)"
        fi
    done
    log "llama-server ready (${elapsed}s)"
}

unload_ollama_models() {
    log "Unloading all Ollama models..."
    for model in $(curl -s http://localhost:11434/api/ps 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for m in d.get('models', []):
        print(m['name'])
except: pass
" 2>/dev/null); do
        curl -s http://localhost:11434/api/generate -d "{\"model\":\"${model}\",\"keep_alive\":0}" > /dev/null 2>&1
        log "  Unloaded: ${model}"
    done
    sleep 3
}

# =================================================================
log "=========================================="
log "OVERNIGHT RUN: ${RUN_ID}"
log "MAP:          Qwen3-235B-A22B Q3_K_M"
log "REDUCE/DRAFT: gpt-oss-120b"
log "=========================================="

# Verify Qwen3 GGUF exists
if [ ! -f "${QWEN3_GGUF}" ]; then
    log "ERROR: Qwen3 GGUF not found: ${QWEN3_GGUF}"
    log "Download it first with:"
    log "  python3 -c \"from huggingface_hub import hf_hub_download; ..."
    exit 1
fi

# Unload Ollama models to free GPU memory
unload_ollama_models

# =================================================================
# PHASE 1: MAP with Qwen3-235B-A22B
# =================================================================
log ""
log "========== PHASE 1: MAP (Qwen3-235B-A22B) =========="

kill_llama_server

log "Starting llama-server with Qwen3-235B-A22B Q3_K_M..."
${LLAMA_SERVER} \
    -m "${QWEN3_GGUF}" \
    -ngl 999 \
    --no-mmap \
    -fa on \
    --ctx-size 8192 \
    --port ${PORT} \
    --host 0.0.0.0 \
    -b 2048 -ub 2048 \
    --jinja \
    -np 1 \
    > "${RUN_DIR}/${RUN_ID}_qwen3_server.log" 2>&1 &

QWEN3_PID=$!
log "llama-server PID: ${QWEN3_PID}"

# Wait for Qwen3 to load (112GB model, may take 2-3 min)
wait_for_server 600

# Run MAP phase only (candidates + map)
log "Starting batch MAP phase..."
MAP_START=$(date +%s)

python3 "${BATCH_SCRIPT}" \
    --backend openai \
    --openai-base "http://localhost:${PORT}" \
    --model "Qwen3-235B-A22B-Q3_K_M" \
    --es-index rag_tfk18_v1 \
    --workers 1 \
    --max-candidates 3000 \
    --run-id "${RUN_ID}" \
    --out "${RUN_DIR}" \
    --start-at candidates \
    --stop-after map \
    --timeout 600 \
    2>&1 | tee -a "${RUN_DIR}/${RUN_ID}_map.log"

MAP_END=$(date +%s)
MAP_DURATION=$(( (MAP_END - MAP_START) / 60 ))
log "MAP phase completed in ${MAP_DURATION} minutes"

# Verify MAP output
CLAIMS_FILE="${RUN_DIR}/${RUN_ID}/claims.jsonl"
if [ ! -f "${CLAIMS_FILE}" ]; then
    log "ERROR: No claims.jsonl found after MAP phase!"
    exit 1
fi
CLAIM_COUNT=$(wc -l < "${CLAIMS_FILE}")
log "MAP produced ${CLAIM_COUNT} claims"

# =================================================================
# PHASE 2: Switch to gpt-oss-120b for REDUCE + DRAFT
# =================================================================
log ""
log "========== PHASE 2: REDUCE+DRAFT (gpt-oss-120b) =========="

kill_llama_server
sleep 5

log "Starting llama-server with gpt-oss-120b..."
${LLAMA_SERVER} \
    -hf "${GPT_OSS_HF}" \
    -ngl 999 \
    --no-mmap \
    -fa on \
    --ctx-size 32768 \
    --port ${PORT} \
    --host 0.0.0.0 \
    -b 2048 -ub 2048 \
    --jinja \
    -np 1 \
    > "${RUN_DIR}/${RUN_ID}_gptoss_server.log" 2>&1 &

GPTOSS_PID=$!
log "llama-server PID: ${GPTOSS_PID}"

# Wait for gpt-oss-120b to load
wait_for_server 300

# Delete findings.json and report_suggestions.md to force re-run from REDUCE
rm -f "${RUN_DIR}/${RUN_ID}/findings.json"
rm -f "${RUN_DIR}/${RUN_ID}/report_suggestions.md"

# Run REDUCE + DRAFT
log "Starting batch REDUCE+DRAFT phase..."
REDUCE_START=$(date +%s)

python3 "${BATCH_SCRIPT}" \
    --backend openai \
    --openai-base "http://localhost:${PORT}" \
    --model "gpt-oss-120b" \
    --es-index rag_tfk18_v1 \
    --workers 1 \
    --max-candidates 3000 \
    --max-findings 120 \
    --reduce-batch-size 20 \
    --run-id "${RUN_ID}" \
    --out "${RUN_DIR}" \
    --start-at reduce \
    --timeout 1800 \
    2>&1 | tee -a "${RUN_DIR}/${RUN_ID}_reduce_draft.log"

REDUCE_END=$(date +%s)
REDUCE_DURATION=$(( (REDUCE_END - REDUCE_START) / 60 ))
log "REDUCE+DRAFT completed in ${REDUCE_DURATION} minutes"

# =================================================================
# Summary
# =================================================================
log ""
log "=========================================="
log "OVERNIGHT RUN COMPLETE: ${RUN_ID}"
log "MAP:          Qwen3-235B-A22B Q3_K_M (${MAP_DURATION} min)"
log "REDUCE/DRAFT: gpt-oss-120b (${REDUCE_DURATION} min)"
log "Total:        $(( (REDUCE_END - MAP_START) / 60 )) min"
log ""
log "Artifacts:"
log "  Claims:     ${RUN_DIR}/${RUN_ID}/claims.jsonl"
log "  Findings:   ${RUN_DIR}/${RUN_ID}/findings.json"
log "  Report:     ${RUN_DIR}/${RUN_ID}/report_suggestions.md"
log "=========================================="
