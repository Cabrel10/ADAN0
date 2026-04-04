#!/bin/bash
# Real-time worker monitoring script

echo "🔍 MONITORING WORKERS IN REAL-TIME"
echo "=================================="

while true; do
    clear
    echo "📊 WORKER STATUS - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================="
    
    # Check if training is running
    if ! pgrep -f "train_parallel_agents" > /dev/null; then
        echo "❌ Training NOT running"
        exit 1
    fi
    
    # Get worker PIDs
    echo ""
    echo "🔴 Active Workers:"
    ps aux | grep "ray::ADAN_PBT_Worker.train" | grep -v grep | awk '{print "  PID: " $2 " | CPU: " $3 "% | MEM: " $6 "KB"}'
    
    # Get recent trading activity
    echo ""
    echo "📈 Recent Trading Activity (last 100 lines):"
    tail -100 /mnt/new_data/t10_training/logs/training.log 2>/dev/null | grep -E "TARGET_WEIGHT|POSITION FERMEE|STOP LOSS|TAKE PROFIT" | tail -10 | sed 's/^/  /'
    
    # Get progress metrics
    echo ""
    echo "📊 Progress Metrics:"
    python3 << 'PYEOF'
import csv
import re
from pathlib import Path

D = Path('/mnt/new_data/t10_training/ray_results/adan_pbt_training')
for f in sorted(D.glob('**/progress.csv')):
    p = re.search(r'profile=(\w+)', f.parent.name)
    prof = p.group(1) if p else '?'
    try:
        with open(f, errors='ignore') as file:
            lines = file.readlines()
            if len(lines) > 2:
                last_line = lines[-1].strip()
                values = last_line.split(',')
                if len(values) >= 10:
                    try:
                        bal = float(values[2])
                        sh = float(values[1])
                        itr = int(values[8])
                        stp = int(values[6])
                        pnl = (bal - 20.5) / 20.5 * 100 if bal > 0 else 0
                        status = "🟢" if pnl > 0 else "🔴"
                        print(f"  {prof:<12} | Iter: {itr:<3} | Steps: {stp:<8,} | Balance: ${bal:<7.2f} | PnL: {pnl:>+6.1f}% | Sharpe: {sh:>6.2f} {status}")
                    except:
                        pass
    except:
        pass
PYEOF
    
    echo ""
    echo "⏱️  Refreshing in 30 seconds... (Ctrl+C to exit)"
    sleep 30
done
