#!/usr/bin/env bash
# Forensic scan across checkpoints. Pinned low-priority on cores 1-3 (shared
# with training) so it never starves the live run.
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
PY="/home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python"
STEPS="${2:-5000}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs/validation/forensic
for k in $1; do
  ck="checkpoints/ppo_adan0_sandbox_checkpoint_${k}_steps.zip"
  if [ ! -f "$ck" ]; then echo "SKIP $k (missing)"; continue; fi
  echo "=== FORENSIC $k @ $(date +%H:%M:%S) (steps=$STEPS) ==="
  nice -n 19 taskset -c 1-3 "$PY" scripts/backtest/forensic_trades.py \
    --ckpt "$ck" --split test --steps "$STEPS" \
    --out "logs/validation/forensic/forensic_${k}.json" 2>/dev/null
done
echo "FORENSIC_SCAN_DONE @ $(date +%H:%M:%S)"
