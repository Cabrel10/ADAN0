#!/bin/bash
# Monitor Polar Reward Training in Real-Time

LOGFILE="/mnt/new_data/adan_logs/checkpoints/training_20260605_162621.log"

if [ ! -f "$LOGFILE" ]; then
    echo "❌ Log file not found: $LOGFILE"
    exit 1
fi

echo "📊 POLAR REWARD TRAINING MONITOR"
echo "=================================="
echo ""

while true; do
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Get latest metrics
    echo "📈 Latest Metrics:"
    tail -500 "$LOGFILE" | grep "METRICS_SYNC" | tail -2 | awk '{
        for(i=1; i<=NF; i++) {
            if($i ~ /Step/) print "  Step: " $(i+1)
            if($i ~ /Worker/) print "  Worker: " $(i+1)
            if($i ~ /Sharpe=/) {
                match($i, /Sharpe=([^,]+)/, a)
                print "  Sharpe: " a[1]
            }
            if($i ~ /WinRate=/) {
                match($i, /WinRate=([^,]+)/, a)
                print "  Win Rate: " a[1]
            }
            if($i ~ /Trades=/) {
                match($i, /Trades=([^ ]+)/, a)
                print "  Trades: " a[1]
            }
        }
    }'
    
    echo ""
    echo "💰 Portfolio Status:"
    tail -100 "$LOGFILE" | grep "Portfolio Value:" | tail -1 | awk '{
        for(i=1; i<=NF; i++) {
            if($i ~ /Portfolio/) {
                gsub(/,/,"",$NF)
                print "  " $(i) " " $NF
            }
        }
    }'
    
    echo ""
    echo "🔄 Activity (last 30 sec):"
    tail -200 "$LOGFILE" | grep -c "STEP\|REWARD" | awk '{print "  Events: " $1}'
    
    echo ""
    echo "---"
    sleep 30
done
