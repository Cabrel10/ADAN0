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
# Match ONLY the real Python training process, and never our own PID/children.
# Rationale (§4): `pgrep -f "train_parallel_agents.*500000"` also matches any
# shell/launcher/echo whose cmdline merely CONTAINS that substring (verified: an
# interactive shell referencing the pattern was matched). We therefore (a) anchor
# on a `python ... train_parallel_agents.py ... --steps 500000` invocation that a
# wrapper shell cannot satisfy, and (b) exclude our own PID and parent via -v.
TRAIN_PAT="python.*train_parallel_agents\.py.*--steps[= ]500000"
train_alive() {
  # -v excludes our own PID ($$) and parent ($PPID) so the guard never self-matches.
  pgrep -f "$TRAIN_PAT" 2>/dev/null | grep -vx -e "$$" -e "$PPID" | grep -q .
}
echo "[disk_guard] started $(date) threshold=${THRESHOLD_KB}KB pat='$TRAIN_PAT' self_pid=$$" >> "$LOGF"
while true; do
  # Stop if the 500k training is no longer running.
  if ! train_alive; then
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
    # Pick the most-recently-modified train log (v9, v10, timestamped...).
    TLOG=$(ls -t "$ROOT"/logs/training/train_v*_500k_*.log "$ROOT"/logs/train_v*_500k.log 2>/dev/null | head -1)
    if [ -n "${TLOG:-}" ] && [ -f "$TLOG" ]; then
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
