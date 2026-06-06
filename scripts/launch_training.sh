#!/bin/bash
# ============================================================================
# ADAN Trading Bot - Clean Training Launch Script
# ============================================================================
# Usage:
#   ./scripts/launch_training.sh [--resume] [--light] [--steps N]
#
# Options:
#   --resume     Resume from last checkpoint (default: new training)
#   --light      Use 2 workers (light mode, default: 4 workers heavy)
#   --steps N    Training steps (default: 500000)
#   --debug      Enable debug logging
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
RESUME=false
MODE="heavy"
STEPS=500000
LOG_LEVEL="INFO"
CHECKPOINT_DIR="/mnt/new_data/adan_logs/checkpoints"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --resume)
      RESUME=true
      shift
      ;;
    --light)
      MODE="light"
      shift
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --debug)
      LOG_LEVEL="DEBUG"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Map "light" mode to heavy with 2 samples, "heavy" to 4 samples
if [ "$MODE" = "light" ]; then
  NUM_SAMPLES=2
  ACTUAL_MODE="heavy"
else
  NUM_SAMPLES=4
  ACTUAL_MODE="heavy"
fi

# Create log directory
mkdir -p "$CHECKPOINT_DIR"

# Log training startup
echo "============================================================================"
echo "🔥 ADAN Trading Bot - PBT Training"
echo "============================================================================"
echo "Mode:           $MODE (num_samples=$NUM_SAMPLES)"
echo "Steps:          $STEPS"
echo "Resume:         $RESUME"
echo "Log level:      $LOG_LEVEL"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Time:           $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"
echo ""

# Check dependencies
echo "[1/3] Checking dependencies..."
python -c "import ray; print(f'  ✅ Ray {ray.__version__}')"
python -c "import stable_baselines3; print(f'  ✅ Stable-Baselines3 installed')"
python -c "import torch; print(f'  ✅ PyTorch {torch.__version__}')"
echo ""

# Show system resources
echo "[2/3] System resources:"
echo "  CPU cores: $(nproc)"
echo "  RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "  Available: $(free -h | grep Mem | awk '{print $7}')"
echo ""

# Clean Ray temp dir
echo "[3/3] Preparing environment..."
rm -rf /tmp/ray_adan 2>/dev/null || true
mkdir -p /tmp/ray_adan
echo "  ✅ Ray temp dir cleaned"
echo ""

# Build command
CMD="python scripts/train_parallel_agents.py"
CMD="$CMD --config config/config.yaml"
CMD="$CMD --mode $ACTUAL_MODE"
CMD="$CMD --steps $STEPS"
CMD="$CMD --num-cpus 4"
CMD="$CMD --num-samples $NUM_SAMPLES"
CMD="$CMD --no-subproc"
CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"
CMD="$CMD --log-level $LOG_LEVEL"

if [ "$RESUME" = true ]; then
  CMD="$CMD --resume"
fi

# Log file
LOG_FILE="$CHECKPOINT_DIR/training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "🚀 Launching training..."
echo "Command: $CMD"
echo "Logs: $LOG_FILE"
echo ""
echo "============================================================================"
echo ""

# Launch training (background so script returns immediately)
$CMD \
  > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "Process ID: $TRAIN_PID"
echo "To stop:    kill $TRAIN_PID"
echo "To follow:  tail -f $LOG_FILE"
echo ""

# Save PID
echo "$TRAIN_PID" > "$CHECKPOINT_DIR/.training_pid"

echo "✅ Training started successfully"
