#!/usr/bin/env bash
# ============================================================================
# RUN LONG 500k — FIX D (asymmetric SELL threshold) — 2026-07-05
# ----------------------------------------------------------------------------
# MESURE RACINE (diag archfix, PID 256463, sur logs REELS — pas raisonnement code):
#   a0_pct_sell @2k = 0.471 (l'agent VEUT vendre 47% du temps)
#   req_SELL_pct @2k = 0.081 (seuls 8% sont ROUTES en SELL)
#   borne haute si a0<0 en LONG -> SELL = 0.316 (31.6%)
#   => 23.5 pts de SELL manquants: l'intention de sortie (a0 negatif FAIBLE,
#      entre -0.10 et 0) tombe dans la ZONE MORTE |a0|<=threshold(0.10) et est
#      routee en HOLD. L'agent apprend "SELL ne se declenche presque jamais"
#      -> arrete d'essayer -> BUY runaway. Ce N'EST PAS les 3 gardes budget/gap/
#      barrier (mesure: req_SELL est capture AVANT les gardes).
#
# DISCIPLINE (critique methodologique acceptee — ne pas confondre):
#   FIX A (energie observable) = ON  -> INFORMATION pure, ne peut pas creer
#         l'artefact "desserrage". can_close reflete la vraie capacite.
#   FIX C (desserrage des 3 gardes) = OFF (defauts legacy) -> on RETIRE le
#         confondant "si desserré au point de ne jamais s'appliquer".
#   FIX D (seuil SELL asymetrique) = ON -> cible EXACTEMENT la zone morte
#         mesuree. ENTREE = engagement (thr 0.10) ; SORTIE = protection (0.02).
#   => Si pct_sell remonte, c'est la zone morte corrigee, PAS les gardes off.
#      Attribution propre (une cause de conception isolee).
#
# holding_cost=0 (prouve inefficace). std_init=-2.0. Frais 0.5% INTACTS.
# Dims 1-4 INTACTES. Breaker OFF (crash complet). 1 worker intraday.
# ============================================================================
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
PY="$ROOT/../miniconda3/envs/trading_env/bin/python"
cd "$ROOT" || exit 1

STEPS="${STEPS:-500000}"
PROFILES="${PROFILES:-intraday}"
N_EPOCHS="${N_EPOCHS:-10}"
NTHREADS="${NTHREADS:-3}"
CKPT_FREQ="${CKPT_FREQ:-50000}"
DIAG_EVERY="${DIAG_EVERY:-1000}"
ENT="${ENT:-0.04}"
SELL_THR="${SELL_THR:-0.02}"
TAG="${TAG:-selfix_500k}"

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

echo "=== RUN LONG SELFIX steps=$STEPS profile=[$PROFILES] ==="
echo " FIX A energy_obs=ON | FIX C exit gates=LEGACY(off) | FIX D sell_thr=$SELL_THR"
echo " holding=0 std_init=default(-2.0) breaker=OFF diag_every=$DIAG_EVERY"
echo " log=$LOG diag=$DIAG"

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS \
       OPENBLAS_NUM_THREADS=$NTHREADS NUMEXPR_NUM_THREADS=$NTHREADS \
       VECLIB_MAXIMUM_THREADS=$NTHREADS ADAN_NUM_THREADS=$NTHREADS
export OMP_DYNAMIC=FALSE KMP_BLOCKTIME=0
export ADAN_USE_SDE=0                      # std_init = defaut code (-2.0, sain)
export ADAN_TRAINING_SILENT=1
export ADAN_N_EPOCHS=$N_EPOCHS
export ADAN_CKPT_FREQ=$CKPT_FREQ

export ADAN_DIAG_COLLAPSE=1
export ADAN_DIAG_EVERY=$DIAG_EVERY
export ADAN_DIAG_CSV="$DIAG"
export ADAN_ENT_COEF=$ENT

# ==== CORRECTIONS ISOLEES ====
export ADAN_ENERGY_OBS=1          # FIX A: energie observable (information pure)
export ADAN_SELL_THRESHOLD=$SELL_THR   # FIX D: sortie facile (cible zone morte)
# FIX C exit gates: NON exportes -> defauts legacy (gap=12, cost=0.30, recharge=0.02)

# Shaping levers OFF
export ADAN_HOLDING_COST=0.0
export ADAN_TIME_DECAY=0
export ADAN_SMART_FLAT=0

export ADAN_REWARD_TELEM=1
export ADAN_REWARD_TELEM_EVERY=2000
export ADAN_REWARD_TELEM_CSV="$TELEM"

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

setsid nohup bash scripts/disk_guard_v12.sh > logs/disk_guard_long.log 2>&1 &
echo "DISK_GUARD_PID=$!"
echo "OK lance."
