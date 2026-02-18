#!/bin/bash
# ============================================================================
# Overnight Batch-Run: Schlussbericht TFK18 Enrichment
# ============================================================================
#
# Startet die batch_report.py Pipeline im Hintergrund.
# Kann über Nacht laufen (geschätzt ~9h für 2000 candidates mit gpt-oss).
#
# Artefakte werden geschrieben nach: runs/<run_id>/
#   - candidates.jsonl    (ES-Treffer)
#   - claims.jsonl        (MAP-Extrakte)
#   - findings.json       (REDUCE-Verdichtung)
#   - report_suggestions.md (DRAFT-Ergänzungen zum Bericht)
#   - run.log             (Vollständiges Log)
#
# Resume bei Abbruch:
#   python3 scripts/batch_report.py --run-id <RUN_ID> --start-at map
#
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR="runs"
OUTLINE="docs/20260217ProjektabschlussberichtTFK18.pdf"
MODEL="gpt-oss:latest"
MAX_CANDIDATES=2000
ES_INDEX="rag_tfk18_v1"
WORKERS=4

echo "============================================================"
echo " Overnight Batch-Run: Schlussbericht TFK18"
echo " Modell:          $MODEL"
echo " Max Candidates:  $MAX_CANDIDATES"
echo " ES Index:        $ES_INDEX"
echo " Workers:         $WORKERS (concurrent)"
echo " Outline:         $OUTLINE"
echo " Output:          $RUN_DIR/"
echo "============================================================"
echo ""
echo "Starte in 5 Sekunden... (Ctrl+C zum Abbrechen)"
sleep 5

mkdir -p "$RUN_DIR"

# Prüfe ob bereits eine Instanz läuft
if pgrep -f "batch_report.py.*--es-index" > /dev/null 2>&1; then
    echo "⚠️  batch_report.py läuft bereits:"
    pgrep -af "batch_report.py.*--es-index"
    echo ""
    echo "Abbrechen? Ctrl+C. Oder bestehende Instanz stoppen mit:"
    echo "  pkill -f 'batch_report.py.*--es-index'"
    exit 1
fi

LOGFILE="$RUN_DIR/overnight_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Log wird geschrieben nach: $LOGFILE"
echo "Ctrl+C stoppt die Pipeline."
echo "============================================================"
echo ""

# Vordergrund: live Output + Log-Datei (tee schreibt beides)
python3 scripts/batch_report.py \
    --model "$MODEL" \
    --es-index "$ES_INDEX" \
    --max-candidates "$MAX_CANDIDATES" \
    --outline "$OUTLINE" \
    --out "$RUN_DIR" \
    --max-findings 150 \
    --reduce-batch-size 40 \
    --workers "$WORKERS" \
    --log-level INFO \
    2>&1 | tee "$LOGFILE"

echo ""
echo "============================================================"
echo " Pipeline beendet. Log: $LOGFILE"
echo "============================================================"
