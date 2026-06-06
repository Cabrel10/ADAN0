#!/bin/bash
# SESSION 12 FINAL: Clean launch with explicit Ray env vars

set -e

echo "============================================================================"
echo "🔥 ADAN Trading Bot - Session 12 Final Launch"
echo "============================================================================"

# ── EXPORT RAY TIMEOUT VARS GLOBALLY ──
export RAY_GCS_RPC_CLIENT_TIMEOUT_S=2400
export RAY_memory=8000000000
export RAY_DISABLE_TASK_RETRY=1

echo "✅ Ray timeouts set: GCS=${RAY_GCS_RPC_CLIENT_TIMEOUT_S}s RAM=${RAY_memory}"

# ── KILL EXISTING PROCESSES ──
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 2

# ── CLEAN TEMP FILES ──
rm -rf /tmp/ray_* /tmp/tmpsb_* 2>/dev/null || true

# ── CHECK RESOURCES ──
echo ""
echo "[1/3] Checking resources..."
CPU_COUNT=$(nproc)
RAM_GB=$(free -g | awk 'NR==2 {print $2}')
RAM_AVAIL=$(free -g | awk 'NR==2 {print $7}')
echo "  CPU cores: $CPU_COUNT"
echo "  RAM total: ${RAM_GB}G | Available: ${RAM_AVAIL}G"

# ── LAUNCH WITH LIGHT MODE (2 workers) ──
echo ""
echo "[2/3] Launching training..."
export PYTHONUNBUFFERED=1

LOG_DIR="/mnt/new_data/adan_logs/checkpoints"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_s12_final_${TIMESTAMP}.log"

echo "  Log: $LOG_FILE"

# ── EXECUTE TRAINING COMMAND ──
python scripts/train_parallel_agents.py \
    --config config/config.yaml \
    --mode heavy \
    --steps 500000 \
    --num-cpus 4 \
    --num-samples 2 \
    --no-subproc \
    --checkpoint-dir "$LOG_DIR" \
    --log-level INFO \
    --resume \
    2>&1 | tee "$LOG_FILE" &

PID=$!

echo ""
echo "============================================================================"
echo "✅ Training started (PID: $PID)"
echo "   Command: python scripts/train_parallel_agents.py --mode heavy --steps 500000 --num-samples 2 --resume"
echo ""
echo "📊 Monitor with:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🛑 Kill with:"
echo "   kill $PID"
echo "============================================================================"

# Wait for training to complete
wait $PID
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully"
else
    echo ""
    echo "❌ Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
