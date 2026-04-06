#!/bin/bash
# ============================================================
# ADAN Quick Training Status Check
# Run this on your LOCAL machine:
#   bash scripts/quick_training_check.sh
# ============================================================

TRAIN_DIR="${1:-/mnt/new_data/t10_training}"
LOG_FILE="${TRAIN_DIR}/logs/FINAL_1M.log"

echo "================================================================"
echo "  ADAN Training Status Check"
echo "  Directory: ${TRAIN_DIR}"
echo "================================================================"

# 1. Is training still running?
echo ""
echo "--- Process Status ---"
if pgrep -f "train_parallel" > /dev/null 2>&1; then
    echo "  [RUNNING] train_parallel is active"
    ps aux | grep train_parallel | grep -v grep
else
    echo "  [STOPPED] train_parallel is NOT running"
fi

# 2. Directory structure
echo ""
echo "--- Training Directory ---"
if [ -d "$TRAIN_DIR" ]; then
    echo "  Size: $(du -sh "$TRAIN_DIR" 2>/dev/null | cut -f1)"
    echo "  Subdirectories:"
    ls -la "$TRAIN_DIR/" 2>/dev/null | head -20
else
    echo "  [ERROR] Directory not found: $TRAIN_DIR"
    exit 1
fi

# 3. Progress CSV summary
echo ""
echo "--- Progress CSV Files ---"
CSVS=$(find "$TRAIN_DIR" -name "progress.csv" -type f 2>/dev/null)
if [ -n "$CSVS" ]; then
    for csv in $CSVS; do
        LINES=$(wc -l < "$csv" 2>/dev/null)
        echo ""
        echo "  File: $csv ($LINES rows)"
        echo "  Columns: $(head -1 "$csv" 2>/dev/null | tr ',' '\n' | head -20 | tr '\n' ', ')"
        echo "  First data row: $(sed -n '2p' "$csv" 2>/dev/null | cut -c1-200)"
        echo "  Last data row:  $(tail -1 "$csv" 2>/dev/null | cut -c1-200)"
    done
else
    echo "  No progress.csv files found"
fi

# 4. Result JSON
echo ""
echo "--- Result JSON Files ---"
JSONS=$(find "$TRAIN_DIR" -name "result.json" -type f 2>/dev/null)
if [ -n "$JSONS" ]; then
    for jf in $JSONS; do
        SIZE=$(stat -c%s "$jf" 2>/dev/null || stat -f%z "$jf" 2>/dev/null)
        echo "  File: $jf (${SIZE} bytes)"
        # Try to get the last valid JSON line
        tail -5 "$jf" 2>/dev/null | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if line:
        try:
            d = json.loads(line)
            for k in ['training_iteration','timesteps_total','episode_reward_mean','mean_reward','done','status']:
                if k in d: print(f'    {k}: {d[k]}')
            break
        except: pass
" 2>/dev/null
    done
else
    echo "  No result.json files found"
fi

# 5. PBT Summary
echo ""
echo "--- PBT Summary ---"
PBT=$(find "$TRAIN_DIR" -name "pbt_summary.json" -type f 2>/dev/null)
if [ -n "$PBT" ]; then
    cat "$PBT" | python3 -m json.tool 2>/dev/null
else
    echo "  No pbt_summary.json (training may not have completed cleanly)"
fi

# 6. Log file analysis
echo ""
echo "--- Main Log File ---"
if [ -f "$LOG_FILE" ]; then
    SIZE_MB=$(du -m "$LOG_FILE" | cut -f1)
    echo "  File: $LOG_FILE (${SIZE_MB} MB)"
    echo "  Line count: $(wc -l < "$LOG_FILE")"

    echo ""
    echo "  Key events (first 200k lines):"
    head -200000 "$LOG_FILE" | grep -oP '(TRADE_OPEN|AGENT_CLOSE|EPISODE_END|HOLD_MIN|WAIT_BLOCK|training_iteration|CHECKPOINT|NameError|AttributeError|OOM|CUDA error)' | sort | uniq -c | sort -rn

    echo ""
    echo "  Last 10 lines:"
    tail -10 "$LOG_FILE"
else
    echo "  Log file not found: $LOG_FILE"
    echo "  Looking for other logs..."
    find "$TRAIN_DIR" -name "*.log" -type f -exec ls -lh {} \; 2>/dev/null | head -10
fi

# 7. Checkpoints
echo ""
echo "--- Model Checkpoints ---"
CKPTS=$(find "$TRAIN_DIR" -name "*.zip" -o -name "best_model*" -o -name "checkpoint_*" 2>/dev/null | head -20)
if [ -n "$CKPTS" ]; then
    echo "$CKPTS" | while read f; do
        echo "  $(ls -lh "$f" 2>/dev/null | awk '{print $5, $6, $7, $8, $9}')"
    done
else
    echo "  No checkpoint files found"
fi

echo ""
echo "================================================================"
echo "  For detailed analysis, run:"
echo "  python scripts/analyze_training_results.py --path $TRAIN_DIR"
echo "================================================================"
