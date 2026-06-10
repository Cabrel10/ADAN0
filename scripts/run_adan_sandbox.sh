#!/bin/bash
# ADAN Sandbox/VPS Training Launcher
# Adapted from run_adan_pro.sh for sandbox environment
# Differences: local paths, reduced resources, no conda, no sudo

set -e

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "ADAN Training Launcher (Sandbox VPS)"
echo "═══════════════════════════════════════════════════════════════════════════════"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# STEP 1: Cleanup
echo ""
echo "STEP 1: System Cleanup..."
pkill -9 -f "ray" 2>/dev/null || true
pkill -9 -f "python.*train_parallel" 2>/dev/null || true
rm -rf /tmp/ray_* 2>/dev/null || true
rm -rf /tmp/ray_adan/* 2>/dev/null || true
echo "  Cleanup done"

# STEP 2: Create directories
echo ""
echo "STEP 2: Create directories..."
mkdir -p /tmp/ray_adan/spill
mkdir -p /tmp/ray_adan/tmp
mkdir -p "$PROJECT_DIR/logs/training"
mkdir -p "$PROJECT_DIR/checkpoints"
echo "  Directories ready"

# STEP 3: Environment
echo ""
echo "STEP 3: Environment Setup..."
export RAY_NODE_IP_ADDRESS="127.0.0.1"
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.85
export RAY_gcs_rpc_server_reconnect_timeout_s=600
export RAY_health_check_failure_threshold=10
export RAY_TMPDIR=/tmp/ray_adan/tmp
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
echo "  Environment configured"

# STEP 4: Checkpoint detection
echo ""
echo "STEP 4: Checkpoint Detection..."
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
RESUME_FLAG=""

# FRESH training by default (no resume from sandbox model — VecNormalize incompatible)
echo "  FRESH START mode (no resume from sandbox model)"

# STEP 5: Launch Training
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "Launching ADAN training (FRESH, ent_coef=0.10)..."
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$PROJECT_DIR/logs/training/train_${TIMESTAMP}.log"

python scripts/train_parallel_agents.py \
    --num-cpus 4 \
    --num-samples 1 \
    --no-subproc \
    $RESUME_FLAG \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "Training Complete (exit=$EXIT_CODE)"
echo "Log: $LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════════════════"

exit $EXIT_CODE
