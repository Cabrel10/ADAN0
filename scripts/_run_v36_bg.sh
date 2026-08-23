#!/usr/bin/env bash
# _run_v36_bg.sh — launcher paramétrable pour l'ablation V36 (bras A/B/C).
# Usage: _run_v36_bg.sh <bras a|b|c> <steps> [smoke|full]
# NE TOUCHE PAS V35 : checkpoint-out dédié par bras. Conditions V34 conservées.
set -u
BRAS="${1:?usage: _run_v36_bg.sh <a|b|c> <steps> [tag]}"
STEPS="${2:?steps required}"
TAG="${3:-run}"

ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
PY="$ROOT/../miniconda3/envs/trading_env/bin/python"
cd "$ROOT" || exit 1

CFG="config/config_v36${BRAS}.yaml"
if [ ! -f "$CFG" ]; then echo "CONFIG ABSENTE: $CFG"; exit 1; fi

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/v36 checkpoints
LOG="logs/v36/v36${BRAS}_${TAG}_${TS}.log"
CKPT="checkpoints/ppo_adan0_v36${BRAS}_${TAG}.zip"
echo "$LOG" > /tmp/v36${BRAS}_log.txt

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ADAN_NUM_THREADS=1
export OMP_DYNAMIC=FALSE KMP_BLOCKTIME=0
export ADAN_USE_SDE=0 ADAN_LOG_STD_INIT=-1.0 ADAN_TRAINING_SILENT=1
export ADAN_N_EPOCHS=10 ADAN_CKPT_FREQ=5000
# Conditions V34/V35 conservées (isolation: seule la config reward change)
export ADAN_NORM_REWARD=1 ADAN_L2_ANCHOR_LAMBDA=0.20 ADAN_MTM_REWARD=0
export ADAN_FREE_SLTP=1

setsid nohup "$PY" scripts/train_parallel_agents.py --mode sandbox --steps "$STEPS" \
    --config "$CFG" \
    --checkpoint-out "$CKPT" \
    < /dev/null > "$LOG" 2>&1 &
TPID=$!
echo "$TPID" > /tmp/v36${BRAS}.pid
echo "SPAWNED bras=$BRAS TPID=$TPID STEPS=$STEPS CFG=$CFG LOG=$LOG CKPT=$CKPT"
