#!/bin/bash
#
# COMPLETE CHUNK AUDIT - ORCHESTRATION SCRIPT
# ═════════════════════════════════════════════
# Runs all audit tests on a complete chunk and generates report
#
# Usage:
#   ./scripts/run_full_audit.sh --checkpoint <path> --chunk <1|2>
#

set -e

echo "════════════════════════════════════════════════════════════════"
echo "🔍 COMPREHENSIVE CHUNK AUDIT SUITE"
echo "════════════════════════════════════════════════════════════════"

# Parse arguments
CHECKPOINT=""
CHUNK_ID=""
OUTPUT_DIR="audit_results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --chunk)
            CHUNK_ID="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate inputs
if [ -z "$CHECKPOINT" ] || [ -z "$CHUNK_ID" ]; then
    echo "❌ Error: --checkpoint and --chunk are required"
    echo "Usage: ./scripts/run_full_audit.sh --checkpoint <path> --chunk <1|2>"
    exit 1
fi

if [ ! -d "$CHECKPOINT" ] && [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "📋 Audit Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Chunk ID: $CHUNK_ID"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# ─────────────────────────────────────────────────────────────────
# PHASE 1: Comprehensive Chunk Audit
# ─────────────────────────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 1: Comprehensive Chunk Audit"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

OUTPUT_FILE="$OUTPUT_DIR/chunk${CHUNK_ID}_audit.json"

python3 scripts/audit_chunk_comprehensive.py \
    --checkpoint "$CHECKPOINT" \
    --chunk "$CHUNK_ID" \
    --output "$OUTPUT_FILE"

echo ""
echo "✅ Phase 1 complete: Results saved to $OUTPUT_FILE"

# ─────────────────────────────────────────────────────────────────
# PHASE 2: Generalization Test (Walk-Forward)
# ─────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 2: Generalization Test (Walk-Forward)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

GEN_OUTPUT="$OUTPUT_DIR/generalization_test.json"

python3 scripts/test_generalization.py \
    --model "$CHECKPOINT" \
    --mode walk-forward \
    --output "$GEN_OUTPUT"

echo ""
echo "✅ Phase 2 complete: Results saved to $GEN_OUTPUT"

# ─────────────────────────────────────────────────────────────────
# PHASE 3: Generate Combined Report
# ─────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 3: Generate Combined Report"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

REPORT_FILE="$OUTPUT_DIR/AUDIT_REPORT_${CHUNK_ID}.md"

cat > "$REPORT_FILE" << 'EOF'
# COMPREHENSIVE CHUNK AUDIT REPORT

**Generated:** $(date)
**Chunk ID:** $CHUNK_ID

## Executive Summary

This audit validates a complete episode (chunk) of agent training across multiple dimensions:
- Trade-level PnL validation
- Position accounting consistency
- Value function effectiveness
- Lookahead bias detection
- Generalization capability

## Results

### Phase 1: Comprehensive Chunk Audit
- See: `chunk${CHUNK_ID}_audit.json`

### Phase 2: Generalization Test
- See: `generalization_test.json`

## Key Findings

[Results from JSON files will be integrated here]

## Recommendations

Based on the audit results, the following actions are recommended:

1. **If trade validation FAILED:**
   - Investigate PnL calculation logic
   - Extract manual trade sample to verify
   - Check position closing logic

2. **If value function R² < 0.1:**
   - Retrain value network with better hyperparameters
   - Consider network architecture changes
   - Increase training episodes

3. **If generalization test shows > 50% degradation:**
   - Agent is trend-dependent
   - NOT ready for production
   - Requires market-neutral strategy redesign

4. **If lookahead bias detected:**
   - CRITICAL: May invalidate all results
   - Review observation construction
   - Check reward calculation

## Next Steps

- [ ] Manual trade validation
- [ ] Value function retraining
- [ ] Walk-forward on out-of-sample data
- [ ] Deploy to paper trading (if all tests pass)

EOF

echo "✅ Phase 3 complete: Report generated at $REPORT_FILE"

# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ AUDIT COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📁 Output files:"
echo "  - $OUTPUT_FILE (comprehensive audit)"
echo "  - $GEN_OUTPUT (generalization test)"
echo "  - $REPORT_FILE (combined report)"
echo ""
echo "📊 Next: Review the JSON files and combined report"
echo ""
