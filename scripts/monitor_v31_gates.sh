#!/bin/bash
# V31-500k gate monitor — watches corrected run for collapse signatures.
# Gates (from RAPPORT_COLLAPSE_V31_500K.md):
#   a0_mean_raw must stay bounded ~[-0.5, 0.5]
#   a0_std_raw  must stay <= exp(2) ~= 7.4 under DiagGaussian
#   KL          must stay < 1.5 * target_kl = 0.0525
#   saturation  must not pin at 96-98% on all heads
# Alerts appended to forensics monitor log; never kills the run.

LOG_DIR="/home/ubuntu/webapp/MORNINGSTAR/ADAN0/logs/training"
OUT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0/forensics/v31_500k_corrected_monitor.log"
INTERVAL=300  # 5 min

echo "=== V31-500k CORRECTED RUN MONITOR started $(date) ===" >> "$OUT"

while true; do
    LOG=$(ls -t "$LOG_DIR"/v31_500k_*.log 2>/dev/null | head -1)
    if [ -z "$LOG" ]; then
        echo "$(date) | NO LOG FOUND" >> "$OUT"
        sleep "$INTERVAL"; continue
    fi

    # Is driver alive?
    DRIVER=$(ps aux | grep train_parallel_agents | grep -v grep | awk '{print $2}' | head -1)
    if [ -z "$DRIVER" ]; then
        echo "$(date) | !!! DRIVER DEAD — run terminated unexpectedly" >> "$OUT"
        sleep "$INTERVAL"; continue
    fi

    TS=$(date '+%H:%M:%S')
    STEP=$(grep -oE "Starting step [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+")

    # Extract latest PPO metrics if a table has been printed
    A0MEAN=$(grep -oE "a0_mean_raw[^0-9\-]*-?[0-9.]+" "$LOG" | tail -2 | tr '\n' ' ')
    A0STD=$(grep -oE "a0_std_raw[^0-9\-]*-?[0-9.]+" "$LOG" | tail -2 | tr '\n' ' ')
    KL=$(grep -oE "approx_kl[^0-9.]*[0-9.]+" "$LOG" | tail -2 | tr '\n' ' ')
    SATGUARD=$(grep -ci "satguard" "$LOG")
    PPO_TABLES=$(grep -c "a0_mean_raw" "$LOG")

    echo "$TS | pid=$DRIVER step=$STEP ppo_metrics_lines=$PPO_TABLES satguard_events=$SATGUARD | a0_mean: $A0MEAN | a0_std: $A0STD | kl: $KL" >> "$OUT"

    # Gate breach detection (only once PPO tables exist)
    if [ "$PPO_TABLES" -gt 0 ]; then
        LAST_STD=$(grep -oE "a0_std_raw[^0-9\-]*-?[0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+" | tail -1)
        if [ -n "$LAST_STD" ]; then
            BREACH=$(echo "$LAST_STD > 7.4" | bc -l 2>/dev/null)
            if [ "$BREACH" = "1" ]; then
                echo "$TS | !!! GATE BREACH: a0_std_raw=$LAST_STD > 7.4 (DiagGaussian bound) — COLLAPSE SIGNATURE" >> "$OUT"
            fi
        fi
        LAST_MEAN=$(grep -oE "a0_mean_raw[^0-9\-]*-?[0-9.]+" "$LOG" | tail -1 | grep -oE "\-?[0-9.]+" | tail -1)
        if [ -n "$LAST_MEAN" ]; then
            BREACH=$(echo "(${LAST_MEAN#-}) > 1.0" | bc -l 2>/dev/null)
            if [ "$BREACH" = "1" ]; then
                echo "$TS | !!! GATE WARN: a0_mean_raw=$LAST_MEAN beyond [-1,1] — watch for drift" >> "$OUT"
            fi
        fi
    fi

    sleep "$INTERVAL"
done
