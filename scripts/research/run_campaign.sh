#!/usr/bin/env bash
# Carte-blanche parallel research campaign. Each task writes its own result so
# that even if some are slow, at least one full checkpoint result is usable.
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
PY="/home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs/validation/research

run() { nice -n 19 taskset -c 1-3 "$PY" "$@"; }

# (c) zone lookahead audit — fast, no model. Do first so we KNOW it's safe.
echo "[$(date +%H:%M:%S)] (c) zone lookahead audit"
run scripts/research/zone_lookahead_audit.py \
    --out logs/validation/research/zone_audit.json \
    > logs/validation/research/zone_audit.out 2>&1
echo "[$(date +%H:%M:%S)] (c) DONE"

# (a) confusion matrix action x state on 10k steps, key checkpoints.
for k in 430000 480000 500000; do
  ck="checkpoints/ppo_adan0_sandbox_checkpoint_${k}_steps.zip"
  [ -f "$ck" ] || { echo "skip $k"; continue; }
  echo "[$(date +%H:%M:%S)] (a) confusion $k (10000 steps)"
  run scripts/research/confusion_matrix.py --ckpt "$ck" --split test \
      --steps 10000 --out "logs/validation/research/confusion_${k}.json" \
      > "logs/validation/research/confusion_${k}.out" 2>&1
  echo "[$(date +%H:%M:%S)] (a) $k DONE"
done

# forensic for 480k & 500k (430k,200k etc already done) to feed (b)
for k in 480000 500000; do
  ck="checkpoints/ppo_adan0_sandbox_checkpoint_${k}_steps.zip"
  [ -f "$ck" ] || continue
  if [ ! -f "logs/validation/forensic/forensic_${k}.json" ]; then
    echo "[$(date +%H:%M:%S)] forensic $k for (b)"
    run scripts/backtest/forensic_trades.py --ckpt "$ck" --split test \
        --steps 5000 --out "logs/validation/forensic/forensic_${k}.json" \
        > "logs/validation/research/forensic_${k}.out" 2>&1
  fi
done

# (b) winner distribution across ALL forensic results
echo "[$(date +%H:%M:%S)] (b) winner distribution"
run scripts/research/winner_distribution.py \
    --glob 'logs/validation/forensic/forensic_*.json' \
    --out logs/validation/research/winner_dist.json \
    > logs/validation/research/winner_dist.out 2>&1
echo "[$(date +%H:%M:%S)] (b) DONE"

echo "CAMPAIGN_DONE @ $(date +%H:%M:%S)"
