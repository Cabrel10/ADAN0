#!/usr/bin/env bash
# Lancement 500k PRODUCTION — preuve de reussite post-forensic (2026-06-27).
# Protection CPU: taskset coeurs 1-3 (reserve a ADAN0) + nice 10, vu que le VPS
# est partage avec gaintime (crash-loop) + whatsapp. Garde-fous: OMP=3, n_epochs=10,
# SILENT=1, checkpoints 10k. Mono-process SB3 (pas de Ray).
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
PY="$ROOT/../miniconda3/envs/trading_env/bin/python"
cd "$ROOT" || exit 1

STEPS="${STEPS:-500000}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/training/fa_500k_prod_${TS}.log"
mkdir -p logs/training checkpoints
echo "$TS" > /tmp/prod_ts.txt
echo "$LOG" > /tmp/prod_log.txt

# purge
pkill -9 -f train_parallel_agents 2>/dev/null
pkill -9 -f forensic_collector 2>/dev/null
pkill -9 -f "py-spy" 2>/dev/null
sleep 2

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 \
       NUMEXPR_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 ADAN_NUM_THREADS=3
export OMP_DYNAMIC=FALSE KMP_BLOCKTIME=0
export ADAN_USE_SDE=0 ADAN_LOG_STD_INIT=-1.0 ADAN_TRAINING_SILENT=1
export ADAN_N_EPOCHS=10 ADAN_CKPT_FREQ=10000

echo "=== LAUNCH 500k PROD (taskset 1-3, nice 10) log=$LOG ==="
nohup nice -n 10 taskset -c 1-3 "$PY" scripts/train_parallel_agents.py \
    --mode sandbox --steps "$STEPS" \
    --checkpoint-out checkpoints/ppo_adan0_FA_500k_prod.zip \
    > "$LOG" 2>&1 &
echo "TRAIN_PID=$!"
echo "$!" > /tmp/prod_pid.txt
sleep 3
echo "OK lance. tail -f $LOG"
