#!/usr/bin/env bash
set -euo pipefail

echo "🧪 MVP Smoke Test - Streaming Contract"
echo "=========================================="

# Test 1: Health Check
echo "1. Health Check..."
HEALTH_RESPONSE=$(curl -s http://localhost:11436/health)
if echo "$HEALTH_RESPONSE" | grep -q '"ok":true'; then
    echo "✅ Health check passed"
else
    echo "❌ Health failed: $HEALTH_RESPONSE"
fi

# Test 2: Debug Endpoint (should be 404)
echo "2. Debug Endpoint (should be 404)..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:11436/debug/sse)
if [ "$HTTP_CODE" = "404" ]; then
    echo "✅ Debug endpoint correctly disabled"
else
    echo "❌ Debug endpoint should be 404, got $HTTP_CODE"
fi

# Test 3: Streaming Contract
echo "3. Streaming Contract..."
echo "Sending test request..."
curl -N http://localhost:11436/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"agentic-rag","stream":true,"messages":[{"role":"user","content":"smoke test: TRACE sofort, dann tokens"}]}' \
  2>/dev/null | head -5

echo ""
echo "✅ MVP Smoke Test completed!"
echo "Expected: role chunk, then [TRACE], then streaming tokens"
