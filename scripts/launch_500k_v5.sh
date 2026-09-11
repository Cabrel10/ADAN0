#!/usr/bin/env bash
# Lanceur propre 500k v5 — applique TOUS les garde-fous systeme (2026-06-27).
# Hypotheses retenues (analyse utilisateur):
#  - Gel 8h = blocage SYSTEME (I/O + concurrence), pas un bug IA.
#  - Cause racine probable du gel: watcher Bash externe qui tronquait le log
#    pendant que Python ecrivait (desync FD -> flush deadlock). SUPPRIME.
#  - Deadlock OpenMP/PyTorch intermittent: bride a 1 thread (priorite #1).
# Variables surchargeables: STEPS, N_EPOCHS, NTHREADS, CKPT_FREQ.
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
PY="$ROOT/../miniconda3/envs/trading_env/bin/python"
cd "$ROOT" || exit 1

STEPS="${STEPS:-500000}"
N_EPOCHS="${N_EPOCHS:-10}"
NTHREADS="${NTHREADS:-1}"
CKPT_FREQ="${CKPT_FREQ:-10000}"

# 1) Purge des zombies (train + anciens watchers)
pkill -9 -f train_parallel_agents 2>/dev/null
pkill -9 -f watchdog_500k 2>/dev/null
pkill -9 -f surveil_fa_500k 2>/dev/null
sleep 2

TS=$(date +%Y%m%d_%H%M%S)
echo "$TS" > /tmp/run_500k_ts.txt
LOG="logs/training/fa_500k_v5_${TS}.log"
mkdir -p logs/training logs/surveillance checkpoints

echo "=== LAUNCH 500k v5 ==="
echo " steps=$STEPS n_epochs=$N_EPOCHS threads=$NTHREADS ckpt_freq=$CKPT_FREQ"
echo " log=$LOG"

# 2) Bride OpenMP/MKL/BLAS a NTHREADS (priorite #1) + DiagGaussian + SILENT.
#    SILENT=1 tue le flood per-step (DBE/RISK/STEP/REWARD via setLevel WARNING)
#    tout en gardant TRADE_AUDIT/STERILE/FA_WATCHDOG/ACTION_DIST (passes en WARNING)
#    et OHLC_INCOHER (.error). Pas de RotatingFileHandler externe agressif.
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS \
       OPENBLAS_NUM_THREADS=$NTHREADS NUMEXPR_NUM_THREADS=$NTHREADS \
       VECLIB_MAXIMUM_THREADS=$NTHREADS ADAN_NUM_THREADS=$NTHREADS
export OMP_DYNAMIC=FALSE KMP_BLOCKTIME=0
export ADAN_USE_SDE=0 ADAN_LOG_STD_INIT=-1.0
export ADAN_TRAINING_SILENT=1
export ADAN_N_EPOCHS=$N_EPOCHS ADAN_CKPT_FREQ=$CKPT_FREQ

nohup "$PY" scripts/train_parallel_agents.py --mode sandbox --steps "$STEPS" \
    --checkpoint-out checkpoints/ppo_adan0_FA_500k_v5.zip \
    > "$LOG" 2>&1 &
TRAIN_PID=$!
echo "TRAIN_PID=$TRAIN_PID"

# 3) Watchdog Python LECTURE SEULE (jamais de troncature) + criteres d'arret.
sleep 5
nohup "$PY" scripts/diagnostics/watchdog_500k_v5.py "$LOG" 300 \
    > logs/surveillance/watchdog_stdout.log 2>&1 &
echo "WATCHDOG_PID=$!"
echo "OK lance. Suivi: tail -f $LOG | grep -E 'WARNING|fps'"
