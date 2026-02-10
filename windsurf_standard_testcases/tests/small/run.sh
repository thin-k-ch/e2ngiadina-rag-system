#!/usr/bin/env bash
# tests/small/run.sh
set +e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="${OUTDIR:-/tmp/windsurf_tests}"
mkdir -p "$OUTDIR"

fail=0

echo "=========================================="
echo "WINDSURF SMALL TEST SUITE"
echo "OUTDIR=$OUTDIR"
echo "=========================================="

for t in "$ROOT/tests/small/"[0-9][0-9]_*.sh; do
  name="$(basename "$t" .sh)"
  echo
  echo "------------------------------------------"
  echo "Running Small Test: $name"
  echo "------------------------------------------"
  bash "$t"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "❌ $name: FAILED (rc=$rc)"
    fail=1
  else
    echo "✅ $name: PASSED"
  fi
done

exit $fail
