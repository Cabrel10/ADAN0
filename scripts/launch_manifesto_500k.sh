#!/usr/bin/env bash
# ============================================================================
# RUN MANIFESTO v1 — 500k — DELTA LATENT PnL LINEAIRE — 2026-07-06
# ----------------------------------------------------------------------------
# CAUSE RACINE MESUREE (reward_components_selfix, n=30 + simulation):
#   LONG+BUY (a0>0, no-op) raw=-0.0038 (GRATUIT) ; LONG+SELL raw=-0.3041 (PUNI 80x)
#   Le latent_pnl legacy (log1p) est ~200x trop faible: tenir -2%/20pas = -0.0164
#   vs -0.30 pour vendre => tenir 18x moins cher => disposition effect => a0->+inf.
#   AUCUN des 11 runs precedents n'a atteint le comportement sain (cf MANIFESTO §0).
#
# CORRECTIF (UNE seule variable): latent_pnl en mode LINEAIRE, chaque pas,
#   proportionnel au PnL non-realise, asymetrique (loss>gain), plafonne.
#   => tenir une position qui saigne n'est PLUS gratuit ; le no-op BUY-while-long
#      subit la meme douleur latente. Casse l'asymetrie 80x a sa source (le reward).
#
# CALIBRATION 'conservateur' (1er run de validation, prudence anti-exces-inverse):
#   lg=0.30 (gain doux, laisse courir les gagnantes = garde-fou hc020)
#   ll=0.60 (perte 2x le gain = asymetrie saine)
#   cap=0.02 (borne la douleur/pas => pas de panique, gradient stable)
#   every=1 (chaque pas de detention, le "battement de coeur")
#
# TEST DE VALIDATION (MANIFESTO §5, binaire, horizon >100k):
#   S1 a0_mean |slope|<5e-6 & |a0|<0.5  S2 pct_buy<0.90  S3 pct_sell>0.02
#   S4 count(BUY)/count(SELL) in [0.5,2]  S5 duree>3pas & AGENT/SLTP<5  S6 cap>0.9x
#
# CONTRAINTES INTACTES: frais 0.5%, dims 1-4 (Arena/Oracle), obs 28, capital 20.5,
#   pas de VecNormalize, pas de MaskablePPO. holding_cost=0. std_init=-2.0.
#   FIX A (energie obs) ON, FIX D (sell_thr) ON: on GARDE les acquis, on ISOLE la
#   NOUVELLE variable (latent lineaire) par rapport a selfix (meme profil sinon).
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
LAT_LG="${LAT_LG:-0.30}"
LAT_LL="${LAT_LL:-0.60}"
LAT_CAP="${LAT_CAP:-0.02}"
LAT_EVERY="${LAT_EVERY:-1}"
TAG="${TAG:-manifesto_500k}"

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

echo "=== RUN MANIFESTO v1 steps=$STEPS profile=[$PROFILES] ==="
echo " LATENT MODE=linear lg=$LAT_LG ll=$LAT_LL cap=$LAT_CAP every=$LAT_EVERY"
echo " FIX A energy_obs=ON | FIX D sell_thr=$SELL_THR | holding=0 std=-2.0 breaker=OFF"
echo " log=$LOG diag=$DIAG"

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS \
       OPENBLAS_NUM_THREADS=$NTHREADS NUMEXPR_NUM_THREADS=$NTHREADS \
       VECLIB_MAXIMUM_THREADS=$NTHREADS ADAN_NUM_THREADS=$NTHREADS
export OMP_DYNAMIC=FALSE KMP_BLOCKTIME=0
export ADAN_USE_SDE=0
export ADAN_TRAINING_SILENT=1
export ADAN_N_EPOCHS=$N_EPOCHS
export ADAN_CKPT_FREQ=$CKPT_FREQ

export ADAN_DIAG_COLLAPSE=1
export ADAN_DIAG_EVERY=$DIAG_EVERY
export ADAN_DIAG_CSV="$DIAG"
export ADAN_ENT_COEF=$ENT

# ==== NOUVELLE VARIABLE ISOLEE: DELTA LATENT PnL LINEAIRE ====
export ADAN_LATENT_MODE=linear
export ADAN_LATENT_LGAIN=$LAT_LG
export ADAN_LATENT_LLOSS=$LAT_LL
export ADAN_LATENT_CAP=$LAT_CAP
export ADAN_LATENT_EVERY=$LAT_EVERY

# ==== ACQUIS CONSERVES (identiques a selfix pour isolation) ====
export ADAN_ENERGY_OBS=1
export ADAN_SELL_THRESHOLD=$SELL_THR

# Shaping levers OFF (holding_cost prouve inefficace)
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
