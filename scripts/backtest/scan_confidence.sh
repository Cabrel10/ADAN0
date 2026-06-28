#!/bin/bash
# Confidence scan: backtest checkpoints on TEST split -> honest metrics
# (profit_factor, expectancy, total_return) to find the "intelligence peak".
# Pinned to cores 1-3 (shared w/ training) at nice 19 so training stays priority.
cd /home/ubuntu/webapp/MORNINGSTAR/ADAN0
PY=/home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python
export ADAN_TRAINING_SILENT=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs/validation/confidence_scan
CKPTS="${1:-40000 100000 160000 200000 240000}"
STEPS="${2:-3000}"
for STEPS_CKPT in $CKPTS; do
  CKPT="checkpoints/ppo_adan0_sandbox_checkpoint_${STEPS_CKPT}_steps.zip"
  OUT="logs/validation/confidence_scan/bt_${STEPS_CKPT}.json"
  echo "=== Backtesting ${STEPS_CKPT} @ $(date +%H:%M:%S) ==="
  nice -n 19 taskset -c 1-3 env PYTHONPATH=src "$PY" scripts/backtest/backtest_fixed_capital.py \
     --ckpt "$CKPT" --split test --steps "$STEPS" --out "$OUT" >/dev/null 2>&1
  "$PY" scripts/backtest/print_bt.py "$OUT" "$STEPS_CKPT"
done
echo "SCAN_DONE @ $(date +%H:%M:%S)"
