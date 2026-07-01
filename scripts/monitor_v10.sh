#!/usr/bin/env bash
# monitor_v10.sh — EVENT-BASED watcher for the V10 500k run.
# Does NOT stream; writes a compact heartbeat every CHECK_EVERY seconds to
# logs/monitor_v10_status.txt (overwritten) and APPENDS alerts to
# logs/monitor_v10_alerts.txt ONLY on events worth an agent turn:
#   - milestone step crossed (every MILESTONE steps)
#   - collapse signal in diag CSV (pct_buy/pct_sell >= 0.97 or |a0_mean| >= 5)
#   - explained_variance snapshot at each milestone
#   - disk low
#   - run ended (process gone)
# The agent only needs to read the ALERTS file, not poll the training.
set -u
ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
CSV="$ROOT/logs/training/diag_v10_500k.csv"
STATUS="$ROOT/logs/monitor_v10_status.txt"
ALERTS="$ROOT/logs/monitor_v10_alerts.txt"
CHECK_EVERY=60
MILESTONE=25000
mkdir -p "$ROOT/logs/training"
: > "$ALERTS"
echo "[monitor] started $(date)" >> "$ALERTS"
last_milestone=0
while true; do
  if ! pgrep -f "train_parallel_agents.*500000" >/dev/null 2>&1; then
    echo "[monitor] RUN ENDED (process gone) @ $(date)" >> "$ALERTS"
    # capture final EV/step from log
    TLOG=$(ls -t "$ROOT"/logs/training/train_v10_500k_*.log 2>/dev/null | head -1)
    if [ -n "${TLOG:-}" ]; then
      echo "  last explained_variance lines:" >> "$ALERTS"
      grep -a "explained_variance" "$TLOG" | tail -3 >> "$ALERTS" 2>/dev/null
    fi
    break
  fi
  # latest step from diag CSV (col 1 assumed = step) — robust: last numeric first field
  STEP=0; PCTBUY=""; PCTSELL=""; A0MEAN=""; A0STD=""; EV=""; ILLEGAL=""
  if [ -f "$CSV" ]; then
    # Skip header: only consider data rows (first field numeric).
    LASTROW=$(awk -F, 'NR>1 && $1 ~ /^[0-9]+$/ {row=$0} END{print row}' "$CSV")
    STEP=$(echo "$LASTROW" | awk -F, '{print $1}' | tr -dc '0-9')
    [ -z "$STEP" ] && STEP=0
  fi
  FREE_KB=$(df --output=avail / | tail -1 | tr -d ' ')
  {
    echo "=== V10 monitor @ $(date) ==="
    echo "step=$STEP free_kb=$FREE_KB"
    [ -f "$CSV" ] && echo "diag header: $(head -1 "$CSV")" && echo "diag last  : $(tail -1 "$CSV")"
  } > "$STATUS"
  # Milestone alert
  if [ "${STEP:-0}" -ge $((last_milestone + MILESTONE)) ]; then
    last_milestone=$(( (STEP / MILESTONE) * MILESTONE ))
    TLOG=$(ls -t "$ROOT"/logs/training/train_v10_500k_*.log 2>/dev/null | head -1)
    EVLINE=$(grep -a "explained_variance" "$TLOG" 2>/dev/null | tail -1)
    echo "[milestone] step~$STEP @ $(date) | EV: ${EVLINE:-n/a}" >> "$ALERTS"
    [ -f "$CSV" ] && echo "  diag: $(tail -1 "$CSV")" >> "$ALERTS"
  fi
  # Collapse detector from CSV (parse pct_buy/pct_sell/a0_mean by header name)
  if [ -f "$CSV" ]; then
    /home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python - "$CSV" <<'PY' >> "$ALERTS" 2>/dev/null
import sys,csv
p=sys.argv[1]
try:
    rows=list(csv.DictReader(open(p)))
    if rows:
        r=rows[-1]
        def g(k):
            try: return float(r.get(k,"nan"))
            except: return float("nan")
        pb=g("a0_pct_buy") if "a0_pct_buy" in r else g("pct_buy")
        ps=g("a0_pct_sell") if "a0_pct_sell" in r else g("pct_sell")
        am=g("a0_mean")
        import math
        alert=[]
        if pb==pb and pb>=0.97: alert.append(f"pct_buy={pb:.2f}>=0.97")
        if ps==ps and ps>=0.97: alert.append(f"pct_sell={ps:.2f}>=0.97")
        if am==am and abs(am)>=5.0: alert.append(f"|a0_mean|={am:.2f}>=5")
        if alert:
            print(f"[COLLAPSE-SIGNAL] step={r.get('step','?')} "+", ".join(alert))
except Exception as e:
    pass
PY
  fi
  # Disk low alert
  if [ "$FREE_KB" -lt $((2*1024*1024)) ]; then
    echo "[disk-low] free_kb=$FREE_KB @ $(date)" >> "$ALERTS"
  fi
  sleep "$CHECK_EVERY"
done
echo "[monitor] exiting $(date)" >> "$ALERTS"
