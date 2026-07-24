#!/usr/bin/env bash
# ============================================================================
# RUN LONG 500k — ARCHITECTURE FIX (design root-cause) — 2026-07-05
# ----------------------------------------------------------------------------
# AUDIT D'ARCHITECTURE (docs/ARCHITECTURE_AUDIT_v14.md) — causes RACINES trouvees:
#   MALADIE 1: pas de concept de "trade" (reward per-step; PnL n'arrive qu'a la
#              fermeture).
#   MALADIE 2 (LA fuite inverse): la SORTIE etait etranglee par 3 gardes
#              simultanees (decision_budget<cost_close, gap<12, pnl<1.5x frais).
#              -> SELL quasi impossible -> l'agent apprend "SELL never works"
#              -> collapse BUY. On a tellement puni le sur-trading qu'on a rendu
#              la fermeture impossible.
#   MALADIE 3: modules pilotes invisibles/morts. decision_budget (l'ENERGIE)
#              bloquait CLOSE mais N'ETAIT PAS DANS L'OBS -> contrainte cachee
#              -> POMDP -> l'agent ne peut PAS apprendre a la gerer.
#
# CORRECTIONS APPLIQUEES (une famille de variables de conception a la fois):
#   FIX A (POMDP): ADAN_ENERGY_OBS=1 -> l'energie (decision_budget) devient
#                  OBSERVABLE via slot [21] can_close = has_pos * close_readiness.
#   FIX C (unblock exit): desserre les 3 gardes de sortie:
#                  ADAN_CLOSE_MIN_GAP=6 (12->6), ADAN_CLOSE_COST=0.20 (0.30->0.20),
#                  ADAN_CLOSE_RECHARGE=0.04 (0.02->0.04), ADAN_CLOSE_MAX_PER_DAY=12.
#   holding_cost = 0.0 (PROUVE inefficace: ne fait que retarder le collapse).
#
# std_init = defaut code (-2.0, sain, confondeur maitrise). Breaker OFF -> crash
# complet + logs capture si collapse. FRAIS 0.5% INTACTS. Dims 1-4 INTACTES.
# 1 worker intraday. ~13 steps/s -> 500k ~ 10-11h.
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
DIAG_EVERY="${DIAG_EVERY:-2000}"
ENT="${ENT:-0.04}"
TAG="${TAG:-archfix_500k}"

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

echo "=== RUN LONG ARCHFIX steps=$STEPS profile=[$PROFILES] ==="
echo " FIX A energy_obs=ON | FIX C exit: gap=6 cost=0.20 recharge=0.04 maxday=12"
echo " holding=0 std_init=default(-2.0) breaker=OFF diag_every=$DIAG_EVERY"
echo " log=$LOG diag=$DIAG"

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS \
       OPENBLAS_NUM_THREADS=$NTHREADS NUMEXPR_NUM_THREADS=$NTHREADS \
       VECLIB_MAXIMUM_THREADS=$NTHREADS ADAN_NUM_THREADS=$NTHREADS
export OMP_DYNAMIC=FALSE KMP_BLOCKTIME=0
export ADAN_USE_SDE=0                      # log_std_init = defaut code (-2.0, sain)
export ADAN_TRAINING_SILENT=1
export ADAN_N_EPOCHS=$N_EPOCHS
export ADAN_CKPT_FREQ=$CKPT_FREQ

# Diagnostics longs
export ADAN_DIAG_COLLAPSE=1
export ADAN_DIAG_EVERY=$DIAG_EVERY
export ADAN_DIAG_CSV="$DIAG"
export ADAN_ENT_COEF=$ENT

# ==== CORRECTIONS D'ARCHITECTURE ====
export ADAN_ENERGY_OBS=1          # FIX A: energie observable (corrige POMDP)
export ADAN_CLOSE_MIN_GAP=6       # FIX C: gap sortie 12->6
export ADAN_CLOSE_COST=0.20       # FIX C: cout close 0.30->0.20
export ADAN_CLOSE_RECHARGE=0.04   # FIX C: recharge 0.02->0.04 (budget 2x plus vite)
export ADAN_CLOSE_MAX_PER_DAY=12  # FIX C: quota jour 7->12

# Shaping levers OFF (prouves inefficaces / isolement)
export ADAN_HOLDING_COST=0.0
export ADAN_TIME_DECAY=0
export ADAN_SMART_FLAT=0

# Telemetrie reward
export ADAN_REWARD_TELEM=1
export ADAN_REWARD_TELEM_EVERY=2000
export ADAN_REWARD_TELEM_CSV="$TELEM"

# Breaker OFF -> run complet, capture crash si collapse
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
