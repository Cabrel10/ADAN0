#!/usr/bin/env bash
# disk_guard_v12.sh — non-invasive disk watcher for the v12 500k run.
# §4.5: survival criterion covers BOTH the training AND the paper-trading
# process (guard stays alive while EITHER runs). If free space on / drops
# below THRESHOLD_KB, truncate write-only reward jsonl debug files and cap
# verbose logs so neither process ever hits ENOSPC. Truncating an append-mode
# file is safe (writer keeps appending from offset 0, no crash).
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
THRESHOLD_KB=$((3 * 1024 * 1024))   # 3 GiB (disk started at 93%, act early)
LOGF="$ROOT/logs/disk_guard_v12.log"
mkdir -p "$ROOT/logs"

# Anchor patterns on real python invocations a wrapper shell cannot satisfy,
# and exclude our own PID/parent via -vx so the guard never self-matches.
TRAIN_PAT="python.*train_parallel_agents\.py.*--steps[= ][0-9]\{6,\}"
PAPER_PAT="python.*paper_trading_monitor\.py"

proc_alive() {
  pgrep -f "$1" 2>/dev/null | grep -vx -e "$$" -e "$PPID" | grep -q .
}

# §4.5: alive if EITHER training OR paper trading is running.
guard_target_alive() {
  proc_alive "$TRAIN_PAT" || proc_alive "$PAPER_PAT"
}

echo "[disk_guard_v12] started $(date) threshold=${THRESHOLD_KB}KB self_pid=$$" >> "$LOGF"
echo "[disk_guard_v12] watching TRAIN='$TRAIN_PAT' PAPER='$PAPER_PAT'" >> "$LOGF"

# Startup grace: wait for the target to appear before arming the exit.
STARTUP_GRACE=300
_waited=0
while ! guard_target_alive; do
  sleep 5; _waited=$((_waited + 5))
  if [ "$_waited" -ge "$STARTUP_GRACE" ]; then
    echo "[disk_guard_v12] no target after ${STARTUP_GRACE}s, exiting $(date)" >> "$LOGF"
    exit 0
  fi
done

while true; do
  if ! guard_target_alive; then
    echo "[disk_guard_v12] no training AND no paper trading, exiting $(date)" >> "$LOGF"
    break
  fi
  FREE_KB=$(df --output=avail / | tail -1 | tr -d ' ')
  if [ "$FREE_KB" -lt "$THRESHOLD_KB" ]; then
    echo "[disk_guard_v12] LOW DISK ${FREE_KB}KB < ${THRESHOLD_KB}KB @ $(date) -> truncating debug files" >> "$LOGF"
    # 1) reward debug jsonl (write-only, never read by code)
    for f in "$ROOT"/logs/rewards/*.jsonl; do
      [ -f "$f" ] && : > "$f"
    done
    # 2) cap the most-recent verbose training log (keep last ~50MB)
    TLOG=$(ls -t "$ROOT"/logs/training/train_v*_500k_*.log 2>/dev/null | head -1)
    if [ -n "${TLOG:-}" ] && [ -f "$TLOG" ]; then
      SZ=$(stat -c%s "$TLOG" 2>/dev/null || echo 0)
      if [ "$SZ" -gt $((300 * 1024 * 1024)) ]; then
        tail -c $((50 * 1024 * 1024)) "$TLOG" > "$TLOG.tmp" && mv "$TLOG.tmp" "$TLOG"
        echo "[disk_guard_v12] trimmed train log (was ${SZ} bytes)" >> "$LOGF"
      fi
    fi
    # 3) §4.5: cap verbose paper-trading logs too (keep last ~20MB)
    for PLOG in "$ROOT"/logs/paper_trading_v9/*.log "$ROOT"/logs/paper_trading_v12/*.log; do
      [ -f "$PLOG" ] || continue
      PSZ=$(stat -c%s "$PLOG" 2>/dev/null || echo 0)
      if [ "$PSZ" -gt $((100 * 1024 * 1024)) ]; then
        tail -c $((20 * 1024 * 1024)) "$PLOG" > "$PLOG.tmp" && mv "$PLOG.tmp" "$PLOG"
        echo "[disk_guard_v12] trimmed paper log $PLOG (was ${PSZ} bytes)" >> "$LOGF"
      fi
    done
    df -h / | tail -1 >> "$LOGF"
  fi
  sleep 120
done
