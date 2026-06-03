#!/bin/bash
# Clean training launcher - Minimal logging, no deadlocks

# Setup
mkdir -p /mnt/new_data/ray_tmp
mkdir -p /mnt/new_data/adan_logs/training

# Kill any existing Ray processes
pkill -f "ray::" 2>/dev/null || true
sleep 2

# Clean Ray temp
rm -rf /mnt/new_data/ray_tmp/* 2>/dev/null || true
rm -rf /tmp/ray_adan 2>/dev/null || true

# Environment - MINIMAL LOGGING
export RAY_TMPDIR="/mnt/new_data/ray_tmp"
export ADAN_TRAINING_SILENT=1
export ADAN_RICH_STEP_EVERY=999999
export PYTHONPATH=src

# Disable verbose Ray logging
export RAY_LOG_LEVEL=ERROR
export RAY_memory=10000000000  # 10GB
export RAY_object_store_memory=5000000000  # 5GB

# Log file
LOG="/mnt/new_data/adan_logs/training/train_v12_clean.log"

# Launch
echo "🚀 Launching Training V12 (Clean)..."
echo "   Log: $LOG"
echo "   Checkpoints: /mnt/new_data/adan_logs/checkpoints"
echo "   Ray Tmp: /mnt/new_data/ray_tmp"
echo ""

nohup python scripts/train_parallel_agents.py \
    --mode heavy \
    --steps 100000 \
    --num-cpus 4 \
    --num-samples 2 \
    --no-subproc \
    --checkpoint-dir /mnt/new_data/adan_logs/checkpoints \
    > "$LOG" 2>&1 &

PID=$!
echo "✅ Training started (PID: $PID)"
echo ""
echo "Monitor with: tail -f $LOG"
echo "Kill with: kill $PID"

