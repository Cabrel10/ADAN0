#!/usr/bin/env bash
# Launcher FORENSIQUE — reproduire le freeze 12417 et capturer la preuve live.
# CONDITION = celle du run v4 qui a GELE: AUCUNE bride de thread (OMP libre).
# On lance le training SOUS py-spy record (seul moyen de tracer vu ptrace_scope=1
# + pas de root) -> flamegraph ou la fonction figee dominera au moment du freeze.
# En parallele: forensic_collector.py surveille wchan/state/io/iostat SANS ptrace.
#
# Variables surchargeables: STEPS (def 60000), FREEZE_S (def 90).
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
PY="$ROOT/../miniconda3/envs/trading_env/bin/python"
PYSPY="$ROOT/../miniconda3/envs/trading_env/bin/py-spy"
cd "$ROOT" || exit 1

STEPS="${STEPS:-60000}"
FREEZE_S="${FREEZE_S:-90}"

# Purge zombies
pkill -9 -f train_parallel_agents 2>/dev/null
pkill -9 -f forensic_collector 2>/dev/null
pkill -9 -f "py-spy record" 2>/dev/null
sleep 2

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/training/fa_forensic_${TS}.log"
SVG="logs/forensic/flamegraph_${TS}.svg"
RAW="logs/forensic/pyspy_raw_${TS}.txt"
mkdir -p logs/training logs/forensic checkpoints

echo "=== LANCEMENT FORENSIQUE (repro freeze v4) ==="
echo " steps=$STEPS  (AUCUNE bride thread = condition v4 gelee)"
echo " log=$LOG"
echo " flamegraph=$SVG"

# CONDITION v4: PAS de OMP_NUM_THREADS=1. On laisse OpenMP/MKL libres (4 CPU).
# On garde SILENT=1 pour limiter le flood (verbosite != cause; deja prouve).
# On NE met PAS ADAN_NUM_THREADS -> le code prendra max(1,ncpu-1)=3 par defaut.
# Pour reproduire EXACTEMENT v4 (libre), on neutralise la garde via ADAN_NUM_THREADS=4.
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export ADAN_NUM_THREADS=4
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS 2>/dev/null
export ADAN_USE_SDE=0 ADAN_LOG_STD_INIT=-1.0
export ADAN_TRAINING_SILENT=1
export ADAN_N_EPOCHS=20 ADAN_CKPT_FREQ=10000

# 1) Training SOUS py-spy record (py-spy = traceur autorise car parent direct).
#    --subprocesses pour suivre d'eventuels forks. --idle pour voir threads bloques.
#    -d = duree max d'echantillonnage (large pour couvrir le freeze).
nohup "$PYSPY" record --subprocesses --idle --nonblocking \
    --rate 5 --duration 14400 --format flamegraph -o "$SVG" -- \
    "$PY" scripts/train_parallel_agents.py --mode sandbox --steps "$STEPS" \
    --checkpoint-out checkpoints/ppo_adan0_forensic.zip \
    > "$LOG" 2>&1 &
PYSPY_PID=$!
echo "PYSPY_PID(wrapper)=$PYSPY_PID"

# 2) Retrouver le PID Python REEL (enfant de py-spy, comm=python, pas py-spy)
sleep 12
# enfant direct du wrapper py-spy dont la commande contient train_parallel_agents
TRAIN_PID=$(pgrep -P "$PYSPY_PID" -f "train_parallel_agents.py" | head -1)
# fallback: process python (pas py-spy) avec le script en args
if [ -z "$TRAIN_PID" ]; then
  TRAIN_PID=$(ps -eo pid,comm,args | awk '/train_parallel_agents.py/ && $2=="python" {print $1; exit}')
fi
echo "TRAIN_PID=$TRAIN_PID  (wrapper py-spy=$PYSPY_PID)"
echo "$TRAIN_PID" > /tmp/forensic_train_pid.txt
echo "$LOG" > /tmp/forensic_log.txt

# 3) Collecteur forensique en parallele (lecture /proc, pas de ptrace)
if [ -n "$TRAIN_PID" ]; then
  nohup "$PY" scripts/diagnostics/forensic_collector.py "$LOG" "$TRAIN_PID" "$FREEZE_S" 10 \
      > logs/forensic/collector_stdout_${TS}.log 2>&1 &
  echo "COLLECTOR_PID=$!"
else
  echo "ERREUR: PID training introuvable"
fi
echo "OK. Suivi: tail -f logs/forensic/collector_stdout_${TS}.log"
