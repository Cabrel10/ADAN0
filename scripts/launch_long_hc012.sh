#!/usr/bin/env bash
# ============================================================================
# RUN LONG 500k — holding_cost=0.012 (valeur calibrée par bracket) — 2026-07-05
# ----------------------------------------------------------------------------
# JUSTIFICATION (mesurée, non hypothétique) — bracket isolé std=-2.0, intraday:
#   holding=0.006 -> dérive BUY  (pct_buy@10k=0.65, slope +1.8e-05)
#   holding=0.020 -> dérive SELL (pct_buy@15k=0.05, pct_sell=0.94 — sur-correction)
#   => point d'équilibre ~0.012. On lance DIRECTEMENT le run long à cette valeur.
#
# Objectif user: HORIZON LONG pour voir si le collapse (historiquement ~70k)
# survient ou non. Breaker OFF -> on capture le CRASH COMPLET + logs si crash.
# 1 worker intraday (directive: si souci 2-workers -> 1 intraday).
# std_init = défaut code (-2.0, sain). future/smart_flat/time_decay OFF (isolé).
# FRAIS 0.5% INTACTS. Dims 1-4 INTACTES.
# Débit observé ~13 steps/s (1 worker) -> 500k ≈ 10-11h.
# ============================================================================
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
PY="$ROOT/../miniconda3/envs/trading_env/bin/python"
cd "$ROOT" || exit 1

HC="${HC:-0.012}"
STEPS="${STEPS:-500000}"
PROFILES="${PROFILES:-intraday}"
N_EPOCHS="${N_EPOCHS:-10}"
NTHREADS="${NTHREADS:-3}"
CKPT_FREQ="${CKPT_FREQ:-50000}"
DIAG_EVERY="${DIAG_EVERY:-2000}"
ENT="${ENT:-0.04}"
TAG="${TAG:-long_hc012}"

pkill -9 -f train_parallel_agents 2>/dev/null
pkill -9 -f disk_guard 2>/dev/null
sleep 2

TS=$(date +%Y%m%d_%H%M%S)
echo "$TS" > /tmp/run_long_ts.txt
LOG="logs/training/train_${TAG}_${TS}.log"
DIAG="$ROOT/logs/training/diag_${TAG}.csv"
TELEM="$ROOT/logs/training/reward_components_${TAG}.csv"
CKDIR="$ROOT/checkpoints/${TAG}"
mkdir -p logs/training "$CKDIR"

echo "=== RUN LONG holding_cost=$HC steps=$STEPS profile=[$PROFILES] ==="
echo " std_init=default(-2.0) time_decay=0 smart_flat=0 diag_every=$DIAG_EVERY"
echo " breaker=OFF (capture crash complet) ckpt_freq=$CKPT_FREQ ckdir=$CKDIR"
echo " log=$LOG diag=$DIAG"

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS \
       OPENBLAS_NUM_THREADS=$NTHREADS NUMEXPR_NUM_THREADS=$NTHREADS \
       VECLIB_MAXIMUM_THREADS=$NTHREADS ADAN_NUM_THREADS=$NTHREADS
export OMP_DYNAMIC=FALSE KMP_BLOCKTIME=0
export ADAN_USE_SDE=0                      # log_std_init = défaut code (-2.0, sain)
export ADAN_TRAINING_SILENT=1
export ADAN_N_EPOCHS=$N_EPOCHS
export ADAN_CKPT_FREQ=$CKPT_FREQ
# NB: les checkpoints périodiques vont dans checkpoints/ nommés par step
# (ppo_adan0_sandbox_checkpoint_<N>_steps.zip) -> récupération possible après crash.

# Diagnostics longs
export ADAN_DIAG_COLLAPSE=1
export ADAN_DIAG_EVERY=$DIAG_EVERY
export ADAN_DIAG_CSV="$DIAG"
export ADAN_ENT_COEF=$ENT

# LEVIER isolé calibré + autres shaping OFF
export ADAN_HOLDING_COST=$HC
export ADAN_TIME_DECAY=0
export ADAN_SMART_FLAT=0

# Télémétrie reward (échantillonnée, léger sur 500k)
export ADAN_REWARD_TELEM=1
export ADAN_REWARD_TELEM_EVERY=2000
export ADAN_REWARD_TELEM_CSV="$TELEM"

# Breaker OFF -> le run va au bout, on veut le crash complet s'il arrive
export ADAN_COLLAPSE_BREAKER=0
export ADAN_COLLAPSE_PCT=0.99
export ADAN_COLLAPSE_A0=8.0
export ADAN_COLLAPSE_WINDOWS=6

setsid nohup "$PY" scripts/train_parallel_agents.py \
    --mode sandbox --steps "$STEPS" \
    --profiles $PROFILES \
    --config config/config.yaml \
    --checkpoint-out "checkpoints/ppo_adan0_${TAG}.zip" \
    > "$LOG" 2>&1 &
TRAIN_PID=$!
echo "TRAIN_PID=$TRAIN_PID"
echo "$TRAIN_PID" > /tmp/run_long_pid.txt
echo "$LOG" > /tmp/run_long_log.txt
echo "$DIAG" > /tmp/run_long_diag.txt

# Disk guard (lecture seule, arrête si disque bas). Pas de sudo.
setsid nohup bash scripts/disk_guard_v12.sh > logs/disk_guard_long.log 2>&1 &
echo "DISK_GUARD_PID=$!"
echo "OK lancé."
