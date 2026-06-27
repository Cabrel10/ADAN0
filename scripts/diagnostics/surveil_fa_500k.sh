#!/usr/bin/env bash
# Surveillance + log-rotation watcher for the FA 500k v4 run.
# - Extracts critical indicator lines into a compact, append-only audit log.
# - Truncates the raw verbose log when it exceeds MAX_RAW_MB to avoid disk pollution.
# - Never kills the training process; purely observational + rotation.
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
RAW="$ROOT/logs/training/fa_500k_v4.log"
AUDIT="$ROOT/logs/surveillance/fa_500k_v4_audit.log"
SNAP="$ROOT/logs/surveillance/fa_500k_v4_snapshot.txt"
MAX_RAW_MB=200          # truncate raw log above this size
INTERVAL=120           # seconds between sweeps
mkdir -p "$ROOT/logs/surveillance"

extract() {
  # Pull only the indicator lines we care about (dedup of doubled logging via the
  # "INFO [" plain variant is fine; we accept some dup, audit stays tiny).
  grep -hE "FA_WATCHDOG|ACTION_DIST|OHLC_INCOHER|Traceback|Exception|Error:|Training done" "$RAW" 2>/dev/null \
    | grep -vE "INFO \[" >> "$AUDIT" 2>/dev/null || true
}

snapshot() {
  {
    echo "===== SNAPSHOT $(date -u +%FT%TZ) ====="
    local step
    step=$(grep -oE "\[STEP [0-9]+" "$RAW" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
    echo "max_step=$step"
    echo -n "exceptions="; grep -icE "Traceback|Exception|Error:" "$RAW" 2>/dev/null
    echo -n "ohlc_incoher="; grep -ic "OHLC_INCOHER" "$RAW" 2>/dev/null
    echo "last_FA_WATCHDOG:"; grep "FA_WATCHDOG" "$RAW" 2>/dev/null | grep -v "INFO \[" | tail -1 | sed -E 's/.*FA_WATCHDOG/  FA_WATCHDOG/'
    echo "last_ACTION_DIST:"; grep "ACTION_DIST" "$RAW" 2>/dev/null | grep -v "INFO \[" | tail -1 | sed -E 's/.*ACTION_DIST/  ACTION_DIST/'
    echo "last_TRADE_AUDIT:"; grep "TRADE_AUDIT_OPEN" "$RAW" 2>/dev/null | grep -v "INFO \[" | tail -1 | sed -E 's/.*TRADE_AUDIT_OPEN/  TRADE_AUDIT_OPEN/'
    echo "raw_size_mb=$(du -m "$RAW" 2>/dev/null | cut -f1)"
    echo "training_alive=$(pgrep -f train_parallel_agents.py | wc -l)"
  } > "$SNAP" 2>/dev/null
}

rotate_if_big() {
  local mb
  mb=$(du -m "$RAW" 2>/dev/null | cut -f1)
  if [ "${mb:-0}" -ge "$MAX_RAW_MB" ]; then
    extract                       # save indicators first
    tail -n 5000 "$RAW" > "$RAW.tmp" 2>/dev/null && mv "$RAW.tmp" "$RAW"
    echo "[$(date -u +%FT%TZ)] ROTATED raw log (was ${mb}MB, kept last 5000 lines)" >> "$AUDIT"
  fi
}

echo "[$(date -u +%FT%TZ)] surveillance watcher started (pid $$)" >> "$AUDIT"
while pgrep -f train_parallel_agents.py >/dev/null 2>&1; do
  extract
  snapshot
  rotate_if_big
  sleep "$INTERVAL"
done
extract
snapshot
echo "[$(date -u +%FT%TZ)] training process ended; watcher stopping" >> "$AUDIT"
