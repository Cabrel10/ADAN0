#!/usr/bin/env bash
# ============================================================================
# LANCEUR 1M V13 (2026-07-04) — fixes reward + FSM déjà active + breaker OFF.
# ----------------------------------------------------------------------------
# Contexte (docs/HANDOFF_INVESTIGATION_COMPLETE.md §11) :
#   - Le routage FSM (FLAT: BUY/HOLD ; LONG: SELL/HOLD ; 1 position, pas de
#     pyramiding) est DÉJÀ appliqué à l'entraînement (action_routing.py, env L.7761).
#   - Le collapse ne vient pas de l'espace d'action mais du REWARD (asymétrie de
#     variance HOLD-flat=0 vs BUY-flat≠0). Fixes V13 : holding_cost recalibré par
#     mesure (§2) + SMART-FLAT (signal positif pour HOLD intelligent).
#   - Breaker OFF (télémétrie seule) + détecteur relâché : le run va au bout des 1M
#     pour donner une plage visuelle large à la prochaine session.
#   - FRAIS 0.5% INTACTS. Dims 1-4 (Size/TF/SL/TP = Future Arena) INTACTES.
#
# Workers : 2 (scalper w1 + intraday w2).
# Variables surchargeables : STEPS, PROFILES, K_SMART, H_HOLD, DIAG_EVERY, ENT.
# ============================================================================
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
PY="$ROOT/../miniconda3/envs/trading_env/bin/python"
cd "$ROOT" || exit 1

STEPS="${STEPS:-1000000}"
PROFILES="${PROFILES:-scalper intraday}"
N_EPOCHS="${N_EPOCHS:-10}"
NTHREADS="${NTHREADS:-2}"      # 2 workers -> 2 threads BLAS raisonnable
CKPT_FREQ="${CKPT_FREQ:-25000}"
DIAG_EVERY="${DIAG_EVERY:-500}"
K_SMART="${K_SMART:-0.05}"
H_HOLD="${H_HOLD:-0.006}"
ENT="${ENT:-0.04}"

# 1) Purge des zombies éventuels.
pkill -9 -f train_parallel_agents 2>/dev/null
pkill -9 -f watchdog_500k 2>/dev/null
sleep 2

TS=$(date +%Y%m%d_%H%M%S)
echo "$TS" > /tmp/run_1M_v13_ts.txt
LOG="logs/training/train_v13_1M_${TS}.log"
mkdir -p logs/training logs/surveillance checkpoints

echo "=== LAUNCH 1M V13 ==="
echo " steps=$STEPS profiles=[$PROFILES] threads=$NTHREADS ckpt=$CKPT_FREQ diag_every=$DIAG_EVERY"
echo " smart_flat=$K_SMART holding_cost=$H_HOLD ent_coef=$ENT breaker=OFF"
echo " log=$LOG  diag=logs/training/diag_v13_1M.csv"

# 2) Threads BLAS/OpenMP bridés (anti-deadlock intermittent).
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS \
       OPENBLAS_NUM_THREADS=$NTHREADS NUMEXPR_NUM_THREADS=$NTHREADS \
       VECLIB_MAXIMUM_THREADS=$NTHREADS ADAN_NUM_THREADS=$NTHREADS
export OMP_DYNAMIC=FALSE KMP_BLOCKTIME=0
export ADAN_USE_SDE=0 ADAN_LOG_STD_INIT=-1.0
export ADAN_TRAINING_SILENT=1
export ADAN_N_EPOCHS=$N_EPOCHS ADAN_CKPT_FREQ=$CKPT_FREQ

# 3) V13 : diagnostics + reward fixes + breaker OFF + détecteur relâché.
export ADAN_DIAG_COLLAPSE=1
export ADAN_DIAG_EVERY=$DIAG_EVERY
export ADAN_DIAG_CSV="$ROOT/logs/training/diag_v13_1M.csv"
export ADAN_ENT_COEF=$ENT
export ADAN_HOLDING_COST=$H_HOLD
export ADAN_SMART_FLAT=$K_SMART
export ADAN_SMART_FLAT_CAP=0.10
export ADAN_SMART_FLAT_HORIZON=12
export ADAN_COLLAPSE_BREAKER=0          # breaker OFF : le run va au bout
export ADAN_COLLAPSE_PCT=0.99           # détecteur relâché (télémétrie only)
export ADAN_COLLAPSE_A0=8.0
export ADAN_COLLAPSE_WINDOWS=6

# 4) Lancement détaché (survit à la fin du shell).
setsid nohup "$PY" scripts/train_parallel_agents.py \
    --mode sandbox --steps "$STEPS" \
    --profiles $PROFILES \
    --config config/config.yaml \
    --checkpoint-out checkpoints/ppo_adan0_v13_1M.zip \
    > "$LOG" 2>&1 &
TRAIN_PID=$!
echo "TRAIN_PID=$TRAIN_PID"
echo "$TRAIN_PID" > /tmp/run_1M_v13_pid.txt

# 5) Disk guard (lecture seule, arrête si disque < seuil).
setsid nohup bash scripts/disk_guard_v12.sh > logs/disk_guard_v13.log 2>&1 &
echo "DISK_GUARD_PID=$!"

echo "OK lancé. Suivi:"
echo "  tail -f $LOG | grep -E 'WARNING|fps|COLLAPSE'"
echo "  column -s, -t logs/training/diag_v13_1M.csv | less -S"
