#!/bin/bash
# Clean training launcher - no escaping issues
cd "$(dirname "$(dirname "$0")")"

export RAY_TMPDIR="/mnt/new_data/ray_tmp"
export ADAN_TRAINING_SILENT=1
export ADAN_RICH_STEP_EVERY=999999
export PYTHONPATH="src"

mkdir -p /mnt/new_data/ray_tmp
mkdir -p /mnt/new_data/adan_logs/training

nohup python scripts/train_parallel_agents.py \
  --mode heavy \
  --steps 500000 \
  --num-cpus 4 \
  --num-samples 2 \
  --no-subproc \
  --checkpoint-dir /mnt/new_data/adan_logs/checkpoints/adan_pbt_training \
  --resume \
  > /mnt/new_data/adan_logs/training/train_v13_resume.log 2>&1 &

echo "✅ Training started (PID: $!)"
echo "$!" > /mnt/new_data/adan_logs/training/.train_pid
