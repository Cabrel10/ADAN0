#!/usr/bin/env bash
# disk_guard_v9.sh — non-invasive disk watcher for the v9 500k run.
# If free space on / drops below THRESHOLD_KB, truncate the write-only
# reward jsonl debug files (never read back by the code) so the training
# process never hits ENOSPC. Truncation on an append-mode file is safe:
# the writer keeps appending from offset 0, no crash.
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
THRESHOLD_KB=$((2 * 1024 * 1024))   # 2 GiB
LOGF="$ROOT/logs/disk_guard_v9.log"
mkdir -p "$ROOT/logs"
echo "[disk_guard] started $(date) threshold=${THRESHOLD_KB}KB" >> "$LOGF"
while true; do
  # Stop if the 500k training is no longer running.
  if ! pgrep -f "train_parallel_agents.*500000" >/dev/null 2>&1; then
    echo "[disk_guard] training gone, exiting $(date)" >> "$LOGF"
    break
  fi
  FREE_KB=$(df --output=avail / | tail -1 | tr -d ' ')
  if [ "$FREE_KB" -lt "$THRESHOLD_KB" ]; then
    echo "[disk_guard] LOW DISK ${FREE_KB}KB < ${THRESHOLD_KB}KB @ $(date) -> truncating reward jsonl" >> "$LOGF"
    # Truncate reward debug jsonl (write-only, never read by code).
    for f in "$ROOT"/logs/rewards/*.jsonl; do
      [ -f "$f" ] && : > "$f"
    done
    # Also cap the verbose training log if it grows huge (keep last ~50MB).
    TLOG="$ROOT/logs/train_v9_500k.log"
    if [ -f "$TLOG" ]; then
      SZ=$(stat -c%s "$TLOG" 2>/dev/null || echo 0)
      if [ "$SZ" -gt $((300 * 1024 * 1024)) ]; then
        tail -c $((50 * 1024 * 1024)) "$TLOG" > "$TLOG.tmp" && mv "$TLOG.tmp" "$TLOG"
        echo "[disk_guard] trimmed train log (was ${SZ} bytes)" >> "$LOGF"
      fi
    fi
    df -h / | tail -1 >> "$LOGF"
  fi
  sleep 120
done
