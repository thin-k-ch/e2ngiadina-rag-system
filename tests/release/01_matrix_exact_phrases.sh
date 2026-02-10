#!/bin/bash
# Release Test: Matrix of Exact Phrases
# Test multiple phrases with expected hit/miss results

echo "=== RELEASE TEST 01: Matrix Exact Phrases ==="

# Test phrases with expected results
declare -a SHOULD_HIT=(
    "Projektleitung Konzepthase"
    "Engineering Konzeptphase"
    "Tabelle1"
    "SBB TFK"
    "Gotthard"
)

declare -a SHOULD_MISS=(
    "NonExistentPhrase12345"
    "RandomPhraseThatDoesNotExist"
    "NoSuchContentInDocuments"
    "FakePhraseTest123"
    "ImpossibleToFindPhrase"
)

# Function to test phrase
test_phrase() {
    local phrase="$1"
    local should_hit="$2"
    
    echo "Testing phrase: '$phrase' (should_hit=$should_hit)"
    
    # ES Check
    ES_RESULT=$(curl -s -X POST "http://localhost:9200/rag_files_v1/_search" \
      -H "Content-Type: application/json" \
      -d "{\"size\":1,\"query\":{\"match_phrase\":{\"content\":\"$phrase\"}}}")
    
    ES_HITS=$(echo "$ES_RESULT" | jq '.hits.hits | length')
    
    # Agent Check
    REQUEST_JSON="{\"model\":\"llama4:latest\",\"messages\":[{\"role\":\"user\",\"content\":\"Suche exakt die Phrase: $phrase\"}]}"
    AGENT_RESULT=$(curl -s -X POST "http://localhost:11436/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "$REQUEST_JSON")
    
    AGENT_CONTENT=$(echo "$AGENT_RESULT" | jq -r '.choices[0].message.content')
    
    # Check if agent found hits
    agent_has_hits=false
    if [ "$AGENT_CONTENT" != "0 exakte Treffer" ] && [ "$AGENT_CONTENT" != "null" ] && [ -n "$AGENT_CONTENT" ]; then
        agent_has_hits=true
    fi
    
    echo "  ES hits: $ES_HITS"
    echo "  Agent has hits: $agent_has_hits"
    echo "  Agent response: $AGENT_CONTENT"
    
    # Validate expectations
    if [ "$should_hit" = "true" ]; then
        # Should have hits
        if [ "$ES_HITS" -gt "0" ] && [ "$agent_has_hits" = "true" ]; then
            echo "  ✅ PASS: Both ES and Agent found hits"
            return 0
        elif [ "$ES_HITS" -eq "0" ]; then
            echo "  ❌ FAIL: ES should have hits but got 0"
            return 1
        elif [ "$agent_has_hits" = "false" ]; then
            echo "  ❌ FAIL: Agent should have hits but got none"
            return 1
        fi
    else
        # Should NOT have hits
        if [ "$ES_HITS" -eq "0" ] && [ "$agent_has_hits" = "false" ]; then
            echo "  ✅ PASS: Both ES and Agent correctly returned no hits"
            return 0
        elif [ "$ES_HITS" -gt "0" ]; then
            echo "  ❌ FAIL: ES should have 0 hits but got $ES_HITS"
            return 1
        elif [ "$agent_has_hits" = "true" ]; then
            echo "  ❌ FAIL: Agent should have 0 hits but got some"
            return 1
        fi
    fi
}

# Track results
total_tests=0
passed_tests=0
failed_tests=0

echo "Testing phrases that SHOULD HIT:"
for phrase in "${SHOULD_HIT[@]}"; do
    echo ""
    total_tests=$((total_tests + 1))
    if test_phrase "$phrase" "true"; then
        passed_tests=$((passed_tests + 1))
    else
        failed_tests=$((failed_tests + 1))
    fi
done

echo ""
echo "Testing phrases that SHOULD MISS:"
for phrase in "${SHOULD_MISS[@]}"; do
    echo ""
    total_tests=$((total_tests + 1))
    if test_phrase "$phrase" "false"; then
        passed_tests=$((passed_tests + 1))
    else
        failed_tests=$((failed_tests + 1))
    fi
done

# Summary
echo ""
echo "=========================================="
echo "MATRIX EXACT PHRASES SUMMARY"
echo "=========================================="
echo "Total tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $failed_tests"

if [ "$failed_tests" -eq "0" ]; then
    echo "✅ All phrase tests passed"
    exit 0
else
    echo "❌ Some phrase tests failed"
    exit 1
fi
