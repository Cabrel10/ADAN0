#!/usr/bin/env bash
# launch_v35_smoke.sh — V35 SL/TP-liberation smoke test (3k-5k steps).
# ONE causal variable vs V34: ADAN_FREE_SLTP=1. All V34 conditions kept
# IDENTICAL (norm_reward critic fix, L2 anchor, MTM off, SDE off, 1 thread).
# Goal: prove the SL/TP liberation reaches EXECUTION (TRADE_AUDIT_OPEN chain
#       sl_raw->sl_dbe->sl_final) AND that the V34 critic stays healthy.
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
PY="$ROOT/../miniconda3/envs/trading_env/bin/python"
cd "$ROOT" || exit 1

STEPS="${STEPS:-5000}"
N_EPOCHS="${N_EPOCHS:-10}"
NTHREADS="${NTHREADS:-1}"
CKPT_FREQ="${CKPT_FREQ:-5000}"

pkill -9 -f train_parallel_agents 2>/dev/null
sleep 1

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/v35/v35_smoke_${TS}.log"
mkdir -p logs/v35 checkpoints
echo "$LOG" > /tmp/v35_smoke_log.txt

echo "=== V35 SMOKE (SL/TP liberation) ==="
echo " steps=$STEPS  log=$LOG"

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS \
       OPENBLAS_NUM_THREADS=$NTHREADS NUMEXPR_NUM_THREADS=$NTHREADS \
       VECLIB_MAXIMUM_THREADS=$NTHREADS ADAN_NUM_THREADS=$NTHREADS
export OMP_DYNAMIC=FALSE KMP_BLOCKTIME=0
export ADAN_USE_SDE=0 ADAN_LOG_STD_INIT=-1.0
export ADAN_TRAINING_SILENT=1
export ADAN_N_EPOCHS=$N_EPOCHS ADAN_CKPT_FREQ=$CKPT_FREQ

# ---- V34 conditions (unchanged) ----
export ADAN_NORM_REWARD=1          # V34 critic fix (do NOT touch)
export ADAN_L2_ANCHOR_LAMBDA=0.20  # V34 anchor
export ADAN_MTM_REWARD=0           # MTM off (as V34)
# ---- V35 single new variable ----
export ADAN_FREE_SLTP=1            # THE causal change under test

nohup "$PY" scripts/train_parallel_agents.py --mode sandbox --steps "$STEPS" \
    --checkpoint-out checkpoints/ppo_adan0_v35_smoke.zip \
    > "$LOG" 2>&1 &
echo "TRAIN_PID=$!"
echo "OK. Suivi: tail -f $LOG | grep -E 'TRADE_AUDIT_OPEN|SLCHAIN|EV|WARNING'"
