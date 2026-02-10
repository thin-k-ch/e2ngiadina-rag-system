#!/bin/bash
# Test 01: Exact Phrase ES vs Agent - No hangs, no jq dependency

echo "=== TEST 01: Exact Phrase ES vs Agent ==="

# Source HTTP helper
source "$(dirname "$0")/../lib/http_helper.sh"

# Create temp directory
mkdir -p /tmp/windsurf_tests

# Test phrase
PHRASE="Projektleitung Konzepthase"
EXPECTED_FILE="Sockelkosten Konzeptphase.xlsx"

echo "Testing phrase: '$PHRASE'"
echo "Expected file: '$EXPECTED_FILE'"

# 1) ES Ground Truth Check
echo "1. Checking ES ground truth..."
ES_JSON="/tmp/windsurf_tests/es_exact.json"
ES_BODY='{"size":5,"_source":["file.filename","path.real"],"query":{"match_phrase":{"content":"'$PHRASE'"}}}'

es_code=$(http_post_json "http://localhost:9200/rag_files_v1/_search" "$ES_BODY" "$ES_JSON")

if [ "$es_code" != "200" ]; then
    echo "❌ FAIL: ES returned HTTP $es_code"
    exit 1
fi

# Check if expected file is in ES response
if grep -q "$EXPECTED_FILE" "$ES_JSON"; then
    echo "✅ ES ground truth: Found '$EXPECTED_FILE'"
else
    echo "❌ FAIL: ES did not find expected file '$EXPECTED_FILE'"
    echo "ES response:"
    cat "$ES_JSON"
    exit 1
fi

# 2) Agent Exact Phrase Check
echo "2. Checking Agent exact phrase..."
AGENT_JSON="/tmp/windsurf_tests/agent_exact.json"
AGENT_BODY='{"model":"llama4:latest","messages":[{"role":"user","content":"Suche exakt die Phrase: '$PHRASE'. Gib nur die Dateinamen der besten Treffer."}],"stream":false}'

agent_code=$(http_post_json "http://localhost:11436/v1/chat/completions" "$AGENT_BODY" "$AGENT_JSON")

if [ "$agent_code" != "200" ]; then
    echo "❌ FAIL: Agent returned HTTP $agent_code"
    exit 1
fi

# Check if expected file is in agent response
if grep -q "$EXPECTED_FILE" "$AGENT_JSON"; then
    echo "✅ Agent found expected file in content"
else
    echo "❌ FAIL: Agent did not find expected file in content"
    echo "Agent response:"
    cat "$AGENT_JSON"
    exit 1
fi

# Check if /open URL is in sources
if grep -q "/open?path=" "$AGENT_JSON"; then
    echo "✅ Agent provided /open URLs in sources"
else
    echo "❌ FAIL: Agent did not provide /open URLs"
    exit 1
fi

# Check if encoded filename is in sources
if grep -q "Sockelkosten%20Konzeptphase.xlsx" "$AGENT_JSON"; then
    echo "✅ Agent provided correctly encoded filename"
else
    echo "❌ FAIL: Agent did not provide correctly encoded filename"
    exit 1
fi

echo "✅ TEST 01 PASSED: Exact phrase working correctly"
exit 0
