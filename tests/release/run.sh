#!/bin/bash
# Release Test Suite Runner
# Run all release tests and generate report

echo "=== WINDSURF RELEASE TEST SUITE ==="
echo "Running comprehensive release tests"
echo ""

# Report file
REPORT_FILE="tests/reports/release_report.json"
REPORT_DIR=$(dirname "$REPORT_FILE")

# Ensure report directory exists
mkdir -p "$REPORT_DIR"

# Initialize report
cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "suite": "release",
  "tests": [],
  "summary": {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "status": "running"
  }
}
EOF

# Track results
total_tests=0
passed_tests=0
failed_tests=0

# Function to run test and update report
run_release_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .sh)
    local start_time=$(date -Iseconds)
    
    echo "=========================================="
    echo "Running Release Test: $test_name"
    echo "=========================================="
    
    total_tests=$((total_tests + 1))
    
    # Make test executable
    chmod +x "$test_file"
    
    # Run test and capture results
    local test_output=""
    local test_exit_code=0
    
    if test_output=$(bash "$test_file" 2>&1); then
        test_exit_code=0
        passed_tests=$((passed_tests + 1))
        echo "✅ $test_name: PASSED"
    else
        test_exit_code=1
        failed_tests=$((failed_tests + 1))
        echo "❌ $test_name: FAILED"
    fi
    
    local end_time=$(date -Iseconds)
    
    # Add test result to report (using temp file and jq for JSON manipulation)
    local temp_file="/tmp/test_result_$$"
    cat > "$temp_file" << EOF
{
  "name": "$test_name",
  "file": "$test_file",
  "start_time": "$start_time",
  "end_time": "$end_time",
  "exit_code": $test_exit_code,
  "status": "$([ $test_exit_code -eq 0 ] && echo "passed" || echo "failed")",
  "output": $(echo "$test_output" | jq -Rs .)
}
EOF
    
    # Update report (simple approach - recreate with new test added)
    local temp_report="/tmp/report_$$"
    jq --argjson new_test "$(cat "$temp_file")" '.tests += [$new_test]' "$REPORT_FILE" > "$temp_report"
    mv "$temp_report" "$REPORT_FILE"
    
    # Clean up temp files
    rm -f "$temp_file" "$temp_report"
    
    echo ""
}

# Find and run all numbered test files in tests/release/
for test_file in tests/release/[0-9][0-9]_*.sh; do
    if [ -f "$test_file" ]; then
        run_release_test "$test_file"
    fi
done

# Finalize report
jq \
  --argjson total "$total_tests" \
  --argjson passed "$passed_tests" \
  --argjson failed "$failed_tests" \
  --arg status "$([ $failed_tests -eq 0 ] && echo "passed" || echo "failed")" \
  '.summary.total = $total | .summary.passed = $passed | .summary.failed = $failed | .summary.status = $status' \
  "$REPORT_FILE" > "${REPORT_FILE}.tmp" && mv "${REPORT_FILE}.tmp" "$REPORT_FILE"

# Final summary
echo "=========================================="
echo "RELEASE TEST SUITE SUMMARY"
echo "=========================================="
echo "Total tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $failed_tests"
echo "Report saved to: $REPORT_FILE"
echo ""

if [ "$failed_tests" -eq "0" ]; then
    echo "🎉 ALL RELEASE TESTS PASSED!"
    echo "✅ Release suite: SUCCESS"
    exit 0
else
    echo "💥 SOME RELEASE TESTS FAILED!"
    echo "❌ Release suite: FAILURE"
    echo "Check report for details: $REPORT_FILE"
    exit 1
fi
