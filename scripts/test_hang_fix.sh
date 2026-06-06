#!/bin/bash
# Test script to verify the hang fix

set -e

echo "=========================================="
echo "Testing Hang Fix - Step 915"
echo "=========================================="
echo ""

# Clean up old logs
echo "Cleaning up old logs..."
rm -f /mnt/new_data/adan_logs/training/train_v12_final.log

# Start training with timeout
echo "Starting training with 5-minute timeout..."
echo "If training reaches step 1000+, the hang is FIXED!"
echo ""

timeout 300 python scripts/train_parallel_agents.py \
    --config config/main_config.yaml \
    --num-workers 1 \
    --max-steps 2000 \
    --checkpoint-interval 500 \
    2>&1 | tee /tmp/hang_test.log

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 124 ]; then
    echo "❌ TIMEOUT - Training still hangs"
    echo "Last 50 lines of log:"
    tail -50 /mnt/new_data/adan_logs/training/train_v12_final.log
elif [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS - Training completed without hang!"
    echo "Last 50 lines of log:"
    tail -50 /mnt/new_data/adan_logs/training/train_v12_final.log
else
    echo "⚠️  Training exited with code $EXIT_CODE"
    echo "Last 50 lines of log:"
    tail -50 /mnt/new_data/adan_logs/training/train_v12_final.log
fi
echo "=========================================="
