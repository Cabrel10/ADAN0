#!/bin/bash
# 🚀 ADAN PRODUCTION LAUNCHER - SESSION 15 ULTIMATE CONFIG
# Hardware: 16GB RAM + 16GB Swap + 11GB SSD (M.2 NVMe)
# Purpose: Launch training with SSD spilling protection + hardened Ray config
# Output: Direct to terminal (tee to log file for analysis)

set -e  # Exit on error

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "🔥 ADAN Training Launcher (SESSION 15 - Ultimate Ray Config)"
echo "═══════════════════════════════════════════════════════════════════════════════"

# ============================================================================
# STEP 1: System Cleanup (Hard Reset)
# ============================================================================
echo ""
echo "📋 STEP 1: System Cleanup..."

# Kill any existing Ray processes
echo "   • Terminating existing Ray instances..."
pkill -9 -f "ray" 2>/dev/null || true
pkill -9 -f "python.*train_parallel" 2>/dev/null || true

# Clean Ray temp directories
echo "   • Cleaning Ray temporary files..."
rm -rf /tmp/ray_* 2>/dev/null || true
rm -rf /mnt/new_data/ray_tmp/* 2>/dev/null || true

# Clean spill directory but keep structure
echo "   • Cleaning spill directory..."
rm -rf /mnt/new_data/ray_spill/* 2>/dev/null || true

echo "   ✅ System cleanup complete"

# ============================================================================
# STEP 2: Filesystem Optimization (Linux Cache + Swap)
# ============================================================================
echo ""
echo "📋 STEP 2: Filesystem Optimization..."

echo "   • Syncing filesystem..."
sync

echo "   • Dropping Linux cache (frees ~2-3GB)..."
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true

echo "   • Resetting swap (clears old data)..."
sudo swapoff -a 2>/dev/null || true
sleep 1
sudo swapon -a 2>/dev/null || true

echo "   ✅ Filesystem optimization complete"

# ============================================================================
# STEP 3: Verify Directories & Disk Space
# ============================================================================
echo ""
echo "📋 STEP 3: Verify Directories & Disk Space..."

mkdir -p /mnt/new_data/ray_spill
mkdir -p /mnt/new_data/ray_tmp
mkdir -p /mnt/new_data/adan_logs/checkpoints
mkdir -p /mnt/new_data/adan_logs/training

echo "   • Ray spill directory: $(du -sh /mnt/new_data/ray_spill 2>/dev/null | awk '{print $1}')"
echo "   • Ray tmp directory: $(du -sh /mnt/new_data/ray_tmp 2>/dev/null | awk '{print $1}')"
echo "   • Disk space available: $(df /mnt/new_data | tail -1 | awk '{print $4}')"
echo "   ✅ Directories verified"

# ============================================================================
# STEP 4: Environment Setup
# ============================================================================
echo ""
echo "📋 STEP 4: Environment Setup..."

# Ray environment: Disable aggressive memory killer, use SSD spilling
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.88
export RAY_gcs_rpc_server_reconnect_timeout_s=600
export RAY_health_check_failure_threshold=10
export RAY_health_check_initial_delay_ms=1000
export RAY_TMPDIR=/mnt/new_data/ray_tmp

# Python environment
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

echo "   • RAY_TMPDIR=$RAY_TMPDIR"
echo "   • RAY_memory_usage_threshold=$RAY_memory_usage_threshold"
echo "   • RAY_gcs_rpc_server_reconnect_timeout_s=$RAY_gcs_rpc_server_reconnect_timeout_s"
echo "   ✅ Environment configured"

# ============================================================================
# STEP 5: Activate Conda & Launch Training
# ============================================================================
echo ""
echo "📋 STEP 5: Launching Training..."
echo ""

# Activate conda environment
source /home/morningstar/miniconda3/bin/activate trading_env 2>/dev/null || {
    echo "❌ Could not activate conda environment"
    exit 1
}

echo "✅ Conda environment activated"
echo ""
echo "🚀 Starting ADAN training..."
echo "   Command: python scripts/train_parallel_agents.py"
echo "   Log: /mnt/new_data/adan_logs/training/production_run.log"
echo "   Output: Direct to terminal + saved to log"
echo ""

# ============================================================================
# STEP 6: Launch Training (Foreground + Tee to Log)
# ============================================================================

cd /home/morningstar/Documents/trading/ADAN0-main

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "📊 TRAINING SESSION LOGS (Real-time display)"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Run in foreground - all output goes to terminal AND log file
python scripts/train_parallel_agents.py \
    --num-cpus 8 \
    --num-samples 2 \
    --no-subproc \
    --checkpoint-dir /mnt/new_data/adan_logs/checkpoints \
    2>&1 | tee /mnt/new_data/adan_logs/training/production_run.log

# Capture exit code
TRAIN_EXIT_CODE=$?

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "🏁 Training Session Complete"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "   Exit Code: $TRAIN_EXIT_CODE"
echo "   Full Log: /mnt/new_data/adan_logs/training/production_run.log"
echo "   Checkpoints: /mnt/new_data/adan_logs/checkpoints"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "✅ ADAN Configuration:"
echo "   • Object Store: 2GB (RAM) + SSD Spilling to /mnt/new_data/ray_spill"
echo "   • Memory Threshold: 88% (kill workers before GCS crash)"
echo "   • GCS Reconnect: 600s (10 min patience for network issues)"
echo "   • Task Retries: 3x with 5s delay between retries"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

exit $TRAIN_EXIT_CODE
