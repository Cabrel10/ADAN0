#!/usr/bin/env bash
# Quick surveillance snapshot for the v14 anchor 500k run.
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
LOG=$(ls -t "$ROOT"/logs/training/train_v14_500k_*.log 2>/dev/null | head -1)
CSV="$ROOT/logs/training/diag_v14_500k.csv"
echo "=== $(date '+%H:%M:%S') ==="
PID=$(pgrep -f "train_parallel_agents.py.*--steps[= ]500000" | head -1)
if [ -n "$PID" ]; then echo "TRAIN: alive PID=$PID $(ps -p $PID -o etime= 2>/dev/null)"; else echo "TRAIN: NOT RUNNING"; fi
echo "STEP: $(grep -oE 'Starting step [0-9]+' "$LOG" 2>/dev/null | tail -1)"
ERR=$(grep -icE "error|traceback|exception" "$LOG" 2>/dev/null)
echo "ERRORS: $ERR"
echo "--- diag (header + last 4 rows) ---"
head -1 "$CSV" 2>/dev/null
tail -4 "$CSV" 2>/dev/null
DG=$(pgrep -f "disk_guard_v9.sh" | head -1)
echo "DISK_GUARD: ${DG:-DOWN}   FREE: $(df -h / | tail -1 | awk '{print $4}')"
