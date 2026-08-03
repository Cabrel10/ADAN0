#!/usr/bin/env bash
# ============================================================================
# BRACKET holding_cost (protocole §2-C / §3) — 2026-07-05
# ----------------------------------------------------------------------------
# VERDICT mesuré (diag std=-2, intraday, [2000-10000]):
#   holding=0.006  -> pente pct_buy +1.8e-05, pct_buy@10k=0.65
#   time_decay=-0.001 seul -> +6.0e-05, pct_buy@10k=0.90 (INSUFFISANT, symétrique)
# => holding_cost (ASYMÉTRIQUE, position-only) est 3.3x meilleur levier anti-runaway.
# On teste des magnitudes plus fortes, ISOLÉES (time_decay=0, smart_flat=0).
#
# UNE variable: ADAN_HOLDING_COST (=$HC). std=-2.0 (défaut code), intraday seul.
# FRAIS 0.5% INTACTS, dims 1-4 INTACTES.
# ============================================================================
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
PY="$ROOT/../miniconda3/envs/trading_env/bin/python"
cd "$ROOT" || exit 1

HC="${HC:-0.02}"
STEPS="${STEPS:-15000}"
PROFILES="${PROFILES:-intraday}"
N_EPOCHS="${N_EPOCHS:-10}"
NTHREADS="${NTHREADS:-2}"
CKPT_FREQ="${CKPT_FREQ:-5000}"
DIAG_EVERY="${DIAG_EVERY:-250}"
ENT="${ENT:-0.04}"
TAG="${TAG:-hc${HC}}"

pkill -9 -f train_parallel_agents 2>/dev/null
sleep 2

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/training/train_${TAG}_${TS}.log"
DIAG="$ROOT/logs/training/diag_${TAG}.csv"
TELEM="$ROOT/logs/training/reward_components_${TAG}.csv"
mkdir -p logs/training checkpoints

echo "=== BRACKET holding_cost=$HC (steps=$STEPS profile=[$PROFILES]) ==="
echo " time_decay=0 smart_flat=0 std_init=default(-2.0) diag_every=$DIAG_EVERY tag=$TAG"
echo " log=$LOG diag=$DIAG"

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS \
       OPENBLAS_NUM_THREADS=$NTHREADS NUMEXPR_NUM_THREADS=$NTHREADS \
       VECLIB_MAXIMUM_THREADS=$NTHREADS ADAN_NUM_THREADS=$NTHREADS
export OMP_DYNAMIC=FALSE KMP_BLOCKTIME=0
export ADAN_USE_SDE=0          # log_std_init = défaut code (-2.0)
export ADAN_TRAINING_SILENT=1
export ADAN_N_EPOCHS=$N_EPOCHS ADAN_CKPT_FREQ=$CKPT_FREQ

export ADAN_DIAG_COLLAPSE=1
export ADAN_DIAG_EVERY=$DIAG_EVERY
export ADAN_DIAG_CSV="$DIAG"
export ADAN_ENT_COEF=$ENT

# LA variable isolée + autres shaping OFF
export ADAN_HOLDING_COST=$HC
export ADAN_TIME_DECAY=0
export ADAN_SMART_FLAT=0

export ADAN_REWARD_TELEM=1
export ADAN_REWARD_TELEM_EVERY=200
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
echo "$TRAIN_PID" > /tmp/run_bracket_pid.txt
echo "$LOG" > /tmp/run_bracket_log.txt
echo "$DIAG" > /tmp/run_bracket_diag.txt
echo "OK lancé."
