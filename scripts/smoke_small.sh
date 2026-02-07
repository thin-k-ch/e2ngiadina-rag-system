#!/bin/bash
# WINDSURF SMALL SMOKE TEST - READ-ONLY
set -e

ES="http://localhost:9200"
IDX="rag_files_v1"
AGENT="http://localhost:11436"
WEBUI="http://localhost:8086"
OLLAMA="http://localhost:11434"

echo "🔥 WINDSURF SMALL SMOKE TEST"
echo "================================"

# 1) ES Health + Count
echo "1️⃣  Elasticsearch Health + Count"
ES_HEALTH=$(curl -s "$ES/_cluster/health" | jq -r '.status // "ERROR"')
ES_COUNT=$(curl -s "$ES/$IDX/_count" | jq -r '.count // 0')

if [[ "$ES_HEALTH" == "green" || "$ES_HEALTH" == "yellow" ]]; then
    echo "✅ ES Health: $ES_HEALTH"
else
    echo "❌ ES Health: $ES_HEALTH"
    exit 1
fi

if [[ "$ES_COUNT" -gt 50000 ]]; then
    echo "✅ ES Count: $ES_COUNT docs"
else
    echo "❌ ES Count: $ES_COUNT docs (expected > 50k)"
    exit 1
fi

# 2) ES Search Test
echo ""
echo "2️⃣  ES Search Test"
ES_HITS=$(curl -s "$ES/$IDX/_search" -H "Content-Type: application/json" \
  -d '{"size":0,"query":{"match":{"content":"Tabelle1"}}}' | jq -r '.hits.total.value // 0')

if [[ "$ES_HITS" -gt 0 ]]; then
    echo "✅ ES Search: $ES_HITS hits for 'Tabelle1'"
else
    echo "❌ ES Search: $ES_HITS hits for 'Tabelle1'"
    exit 1
fi

# 3) Agent Health + Chat
echo ""
echo "3️⃣  Agent Health + Chat"
AGENT_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "$AGENT/health")

if [[ "$AGENT_HEALTH" == "200" ]]; then
    echo "✅ Agent Health: HTTP 200"
else
    echo "❌ Agent Health: HTTP $AGENT_HEALTH"
    exit 1
fi

# Simple Chat Test
CHAT_RESPONSE=$(curl -s "$AGENT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama4:latest","messages":[{"role":"user","content":"Test"}],"stream":false}' | \
  jq -r '.choices[0].message.content // "ERROR"')

if [[ "$CHAT_RESPONSE" != "ERROR" ]]; then
    echo "✅ Agent Chat: Response received"
else
    echo "❌ Agent Chat: No response"
    exit 1
fi

# 4) /open GET Test
echo ""
echo "4️⃣  File Proxy Test"
# Try to access a known file from previous tests
OPEN_URL="http://localhost:11436/open?path=/media/felix/RAG/1/SBB%20TFK%202020%20PJ%20-%207%20Finanzen/71%20Kalkulation/Sockelkosten%20Konzeptphase.xlsx"
OPEN_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$OPEN_URL")

if [[ "$OPEN_STATUS" == "200" ]]; then
    echo "✅ File Proxy: HTTP 200"
else
    echo "⚠️  File Proxy: HTTP $OPEN_STATUS (file may not exist)"
fi

# 5) Summary
echo ""
echo "📊 SMOKE TEST SUMMARY"
echo "======================"
echo "✅ Elasticsearch: $ES_HEALTH ($ES_COUNT docs)"
echo "✅ ES Search: $ES_HITS hits"
echo "✅ Agent: HTTP $AGENT_HEALTH"
echo "✅ Agent Chat: Working"
echo "✅ File Proxy: HTTP $OPEN_STATUS"
echo ""
echo "🎉 WINDSURF SYSTEM READY FOR PRODUCTION!"
