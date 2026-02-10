#!/bin/bash
# Test 02: Health Endpoints - No hangs, no jq dependency

echo "=== TEST 02: Health Endpoints ==="

# Source HTTP helper
source "$(dirname "$0")/../lib/http_helper.sh"

# Create temp directory
mkdir -p /tmp/windsurf_tests

# Track failures
failures=0

# Function to check endpoint
check_endpoint() {
    local url="$1"
    local expected_code="$2"
    local service_name="$3"
    
    echo "Checking $service_name: $url"
    
    # Get HTTP status code with timeout
    status_code=$(http_code "$url")
    
    if [ "$status_code" = "$expected_code" ]; then
        echo "✅ $service_name: HTTP $status_code (OK)"
        return 0
    else
        echo "❌ $service_name: HTTP $status_code (expected $expected_code)"
        return 1
    fi
}

# Check endpoints
if ! check_endpoint "http://localhost:11436/health" "200" "Agent API"; then
    failures=$((failures + 1))
fi

if ! check_endpoint "http://localhost:11436/v1/models" "200" "Agent Models"; then
    failures=$((failures + 1))
fi

if ! check_endpoint "http://localhost:11434/api/tags" "200" "Ollama"; then
    failures=$((failures + 1))
fi

if ! check_endpoint "http://localhost:9200" "200" "Elasticsearch"; then
    failures=$((failures + 1))
fi

if ! check_endpoint "http://localhost:8086" "200" "WebUI"; then
    failures=$((failures + 1))
fi

# Summary
echo ""
echo "=== HEALTH CHECK SUMMARY ==="
if [ "$failures" -eq "0" ]; then
    echo "✅ All endpoints healthy"
    echo "✅ TEST 02 PASSED"
    exit 0
else
    echo "❌ $failures endpoints failed"
    echo "❌ TEST 02 FAILED"
    exit 1
fi
