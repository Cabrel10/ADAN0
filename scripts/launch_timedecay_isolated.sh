#!/usr/bin/env bash
# ============================================================================
# TEST ISOLÉ time_decay (protocole §3) — 2026-07-04
# ----------------------------------------------------------------------------
# UNE SEULE variable changée vs baseline: ADAN_TIME_DECAY=-0.001 (6-June value).
#   - smart_flat OFF (K_SMART=0), holding_cost OFF (H_HOLD=0).
#   - latent_pnl_shaping INTACT (§2: négligeable, ne pas y toucher).
#   - profil INTRADAY SEUL (1 worker) — directive user: si souci 2-workers, 1 intraday.
#   - diag EVERY=250, reward telemetry ON, breaker OFF (télémétrie seule).
# Objectif: vérifier que pct_buy ne franchit PAS 0.90 et a0_mean ne diverge PAS.
# FRAIS 0.5% INTACTS, dims 1-4 INTACTES.
# ============================================================================
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
PY="$ROOT/../miniconda3/envs/trading_env/bin/python"
cd "$ROOT" || exit 1

STEPS="${STEPS:-5000}"
PROFILES="${PROFILES:-intraday}"
N_EPOCHS="${N_EPOCHS:-10}"
NTHREADS="${NTHREADS:-2}"
CKPT_FREQ="${CKPT_FREQ:-2500}"
DIAG_EVERY="${DIAG_EVERY:-250}"
TD="${TD:--0.001}"        # <-- la variable testée
ENT="${ENT:-0.04}"

pkill -9 -f train_parallel_agents 2>/dev/null
sleep 2

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/training/train_td_iso_${TS}.log"
mkdir -p logs/training checkpoints

echo "=== TEST ISOLÉ time_decay=$TD (steps=$STEPS profile=[$PROFILES]) ==="
echo " smart_flat=OFF holding_cost=OFF diag_every=$DIAG_EVERY telem=ON breaker=OFF"
echo " log=$LOG diag=logs/training/diag_td_iso.csv"

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS \
       OPENBLAS_NUM_THREADS=$NTHREADS NUMEXPR_NUM_THREADS=$NTHREADS \
       VECLIB_MAXIMUM_THREADS=$NTHREADS ADAN_NUM_THREADS=$NTHREADS
export OMP_DYNAMIC=FALSE KMP_BLOCKTIME=0
# log_std_init: on laisse le DEFAUT du code (-2.0, std0≈0.135) pour comparer à UNE
# variable près contre les baselines v13_holdcost/nofuture (qui avaient std≈0.13).
# NE PAS forcer -1.0 ici (biais confondant identifié: std 2.7x plus large).
export ADAN_USE_SDE=0
export ADAN_TRAINING_SILENT=1
export ADAN_N_EPOCHS=$N_EPOCHS ADAN_CKPT_FREQ=$CKPT_FREQ

# Diagnostics
export ADAN_DIAG_COLLAPSE=1
export ADAN_DIAG_EVERY=$DIAG_EVERY
export ADAN_DIAG_CSV="$ROOT/logs/training/diag_td_iso.csv"
export ADAN_ENT_COEF=$ENT

# LA variable isolée + tous les autres shaping OFF
export ADAN_TIME_DECAY=$TD
export ADAN_HOLDING_COST=0
export ADAN_SMART_FLAT=0

# Reward telemetry ON (pour §2/vérif composantes)
export ADAN_REWARD_TELEM=1
export ADAN_REWARD_TELEM_EVERY=100
export ADAN_REWARD_TELEM_CSV="$ROOT/logs/training/reward_components_td_iso.csv"

# Breaker OFF (télémétrie seule)
export ADAN_COLLAPSE_BREAKER=0
export ADAN_COLLAPSE_PCT=0.99
export ADAN_COLLAPSE_A0=8.0
export ADAN_COLLAPSE_WINDOWS=6

setsid nohup "$PY" scripts/train_parallel_agents.py \
    --mode sandbox --steps "$STEPS" \
    --profiles $PROFILES \
    --config config/config.yaml \
    --checkpoint-out checkpoints/ppo_adan0_td_iso.zip \
    > "$LOG" 2>&1 &
TRAIN_PID=$!
echo "TRAIN_PID=$TRAIN_PID"
echo "$TRAIN_PID" > /tmp/run_td_iso_pid.txt
echo "$LOG" > /tmp/run_td_iso_log.txt
echo "OK lancé."
