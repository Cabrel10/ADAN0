#!/usr/bin/env bash
# _run_v35_bg.sh — helper: launch V35 500k fully detached.
# No pkill here (avoids self-match). Assumes caller already cleaned stale procs.
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
PY="$ROOT/../miniconda3/envs/trading_env/bin/python"
cd "$ROOT" || exit 1
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/v35/v35_train_${TS}.log"
mkdir -p logs/v35 checkpoints
echo "$LOG" > /tmp/v35_500k_log.txt
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ADAN_NUM_THREADS=1
export OMP_DYNAMIC=FALSE KMP_BLOCKTIME=0
export ADAN_USE_SDE=0 ADAN_LOG_STD_INIT=-1.0 ADAN_TRAINING_SILENT=1
export ADAN_N_EPOCHS=10 ADAN_CKPT_FREQ=5000
# V34 conditions (do NOT change)
export ADAN_NORM_REWARD=1 ADAN_L2_ANCHOR_LAMBDA=0.20 ADAN_MTM_REWARD=0
# V35 single new causal variable
export ADAN_FREE_SLTP=1
setsid nohup "$PY" scripts/train_parallel_agents.py --mode sandbox --steps 500000 \
    --checkpoint-out checkpoints/ppo_adan0_v35_500k.zip \
    < /dev/null > "$LOG" 2>&1 &
TPID=$!
echo "$TPID" > /tmp/v35_500k.pid
echo "SPAWNED TPID=$TPID LOG=$LOG"
