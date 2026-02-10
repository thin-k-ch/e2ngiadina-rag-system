#!/bin/bash
# Small Suite Runner - No hangs, hard timeout per test

echo "=== WINDSURF SMALL TEST SUITE ==="
echo "Running all tests in tests/small/"
echo ""

# Track results
total_tests=0
passed_tests=0
failed_tests=0

# Function to run test with timeout
run_test_with_timeout() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .sh)
    
    echo "=========================================="
    echo "Running: $test_name"
    echo "=========================================="
    
    total_tests=$((total_tests + 1))
    
    # Make test executable
    chmod +x "$test_file"
    
    # Run test with hard timeout (30 seconds)
    if command -v timeout >/dev/null 2>&1; then
        # Use timeout command if available
        if timeout 30s bash "$test_file"; then
            echo "✅ $test_name: PASSED"
            passed_tests=$((passed_tests + 1))
        else
            echo "❌ $test_name: FAILED or TIMEOUT"
            failed_tests=$((failed_tests + 1))
        fi
    else
        # Fallback: manual timeout implementation
        bash "$test_file" &
        local pid=$!
        local count=0
        while [ $count -lt 30 ]; do
            if ! kill -0 $pid 2>/dev/null; then
                wait $pid
                local exit_code=$?
                if [ $exit_code -eq 0 ]; then
                    echo "✅ $test_name: PASSED"
                    passed_tests=$((passed_tests + 1))
                else
                    echo "❌ $test_name: FAILED"
                    failed_tests=$((failed_tests + 1))
                fi
                break
            fi
            sleep 1
            count=$((count + 1))
        done
        
        # Kill if still running after 30 seconds
        if kill -0 $pid 2>/dev/null; then
            echo "❌ $test_name: TIMEOUT (30s)"
            kill -9 $pid 2>/dev/null
            wait $pid 2>/dev/null
            failed_tests=$((failed_tests + 1))
        fi
    fi
    
    echo ""
}

# Find and run all test scripts (exclude run.sh itself)
for test_file in tests/small/[0-9][0-9]_*.sh; do
    if [ -f "$test_file" ] && [ "$(basename "$test_file")" != "run.sh" ]; then
        run_test_with_timeout "$test_file"
    fi
done

# Final summary
echo "=========================================="
echo "SMALL TEST SUITE SUMMARY"
echo "=========================================="
echo "Total tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $failed_tests"
echo ""

if [ "$failed_tests" -eq "0" ]; then
    echo "🎉 ALL TESTS PASSED!"
    echo "✅ Small suite: SUCCESS"
    exit 0
else
    echo "💥 SOME TESTS FAILED!"
    echo "❌ Small suite: FAILURE"
    exit 1
fi
