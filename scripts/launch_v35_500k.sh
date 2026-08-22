#!/usr/bin/env bash
# launch_v35_500k.sh — V35 full run (500k steps).
# ONE causal variable vs V34: ADAN_FREE_SLTP=1 (SL/TP liberation, sonde 4/4
# + smoke GO with drift=0). ALL V34 conditions kept IDENTICAL — critically the
# V34 critic fix (ADAN_NORM_REWARD=1) and the anchor, which launch_500k_v5.sh
# does NOT set. This launcher sets them explicitly to avoid a silent regression.
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
PY="$ROOT/../miniconda3/envs/trading_env/bin/python"
cd "$ROOT" || exit 1

STEPS="${STEPS:-500000}"
N_EPOCHS="${N_EPOCHS:-10}"
NTHREADS="${NTHREADS:-1}"
CKPT_FREQ="${CKPT_FREQ:-10000}"

pkill -9 -f train_parallel_agents 2>/dev/null
pkill -9 -f v32_monitor 2>/dev/null
sleep 2

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/v35/v35_train_${TS}.log"
mkdir -p logs/v35 checkpoints
echo "$LOG" > /tmp/v35_500k_log.txt

echo "=== LAUNCH V35 500k (SL/TP liberation) ==="
echo " steps=$STEPS  log=$LOG"

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS \
       OPENBLAS_NUM_THREADS=$NTHREADS NUMEXPR_NUM_THREADS=$NTHREADS \
       VECLIB_MAXIMUM_THREADS=$NTHREADS ADAN_NUM_THREADS=$NTHREADS
export OMP_DYNAMIC=FALSE KMP_BLOCKTIME=0
export ADAN_USE_SDE=0 ADAN_LOG_STD_INIT=-1.0
export ADAN_TRAINING_SILENT=1
export ADAN_N_EPOCHS=$N_EPOCHS ADAN_CKPT_FREQ=$CKPT_FREQ

# ---- V34 conditions (MUST stay identical — do NOT touch) ----
export ADAN_NORM_REWARD=1          # V34 critic fix (blind-critic root cause)
export ADAN_L2_ANCHOR_LAMBDA=0.20  # V34 anchor
export ADAN_MTM_REWARD=0           # MTM off (as V34)
# ---- V35 single new causal variable ----
export ADAN_FREE_SLTP=1

nohup "$PY" scripts/train_parallel_agents.py --mode sandbox --steps "$STEPS" \
    --checkpoint-out checkpoints/ppo_adan0_v35_500k.zip \
    > "$LOG" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > /tmp/v35_500k.pid
echo "TRAIN_PID=$TRAIN_PID"

# Read-only watchdog (never truncates the log).
sleep 5
nohup "$PY" scripts/diagnostics/watchdog_500k_v5.py "$LOG" 300 \
    > logs/v35/watchdog_${TS}.log 2>&1 &
echo "WATCHDOG_PID=$!"
echo "OK. Suivi: tail -f $LOG | grep -E 'SLCHAIN|explained_var|WARNING'"
