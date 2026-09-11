#!/usr/bin/env bash
# Monitor the V30 500k BTCUSDT_BINANCE run (future-share cap active).
#
# Read-only: parses the run's own log, never touches the training process.
#
#   bash scripts/monitor_v30_500k.sh
#
# V29 failed by policy saturation (a0 pinned at +/-1) with the equity frozen.
# This monitor therefore reports an EARLY-WARNING block that would have
# caught V29 long before 480k steps.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

LOG="logs/v30_500k/run.log"
CKPT_DIR="checkpoints/v30_500k"
TARGET=500000

if ps aux | grep -q "[l]aunch_asset_run.*500000"; then STATE="RUNNING"
else STATE="STOPPED"; fi

echo "=========================================================="
echo " V30 500k  BTCUSDT_BINANCE  [$STATE]  $(date '+%F %T')"
echo "=========================================================="
[[ -f "$LOG" ]] || { echo "no log yet: $LOG"; exit 0; }

CLEAN=$(mktemp); trap 'rm -f "$CLEAN"' EXIT
tr -d '\000' < "$LOG" > "$CLEAN"

TS=$(grep -E '^\|[[:space:]]+total_timesteps' "$CLEAN" | tail -1 |
     grep -oE '[0-9]+' | tail -1); TS=${TS:-0}
echo " progress        : ${TS} / ${TARGET}  ($(awk -v a="$TS" -v b="$TARGET" \
  'BEGIN{printf "%.2f", (a/b)*100}')%)"

for k in explained_variance approx_kl clip_fraction value_loss \
         policy_gradient_loss entropy_loss std fps; do
  V=$(grep -E "^\|[[:space:]]+${k}[[:space:]]" "$CLEAN" | tail -1 |
      awk -F'|' '{gsub(/ /,"",$3); print $3}')
  [[ -n "${V:-}" ]] && printf " %-16s: %s\n" "$k" "$V"
done

echo " ---- explained_variance (last 12) ----"
grep -E '^\|[[:space:]]+explained_variance' "$CLEAN" | tail -12 |
  awk -F'|' '{gsub(/ /,"",$3); printf " %s", $3} END{print ""}'
grep -E '^\|[[:space:]]+explained_variance' "$CLEAN" |
  awk -F'|' '{gsub(/ /,"",$3); n++; if($3+0>0)p++}
             END{if(n)printf " ev positive     : %d / %d updates (%.0f%%)\n",
                 p,n,(p/n)*100}'

# ---- EARLY WARNING: the two failure modes V29 actually exhibited ----------
echo " ---- EARLY WARNING ----"

# 1. Action saturation. V29 ended at 100% |a0|=1.000. Anything above ~60%
#    sustained means the gaussian mean is running to the bounds.
SAT=$(grep -oE 'Raw=-?[0-9.]+' "$CLEAN" | tail -2000 |
      awk -F= '{v=$2<0?-$2:$2; n++; if(v>=0.999)s++}
               END{if(n)printf "%.1f", (s/n)*100; else printf "0"}')
NSAMP=$(grep -coE 'Raw=-?[0-9.]+' "$CLEAN")
printf " a0 saturation   : %s%% of last 2000 (n=%s)  " "$SAT" "$NSAMP"
awk -v s="$SAT" 'BEGIN{print (s+0>60)?"** COLLAPSE RISK **":(s+0>30)?"watch":"ok"}'

# 2. future_share. The V29 root cause: 65.9% from the FIRST window.
FS=$(grep -oE 'future_share=[0-9.]+%' "$CLEAN" | tail -1 | tr -d 'future_share=%')
printf " future_share    : %s%%  " "${FS:-n/a}"
awk -v s="${FS:-0}" 'BEGIN{print (s+0>40)?"** OVER TARGET **":"ok (<40)"}'

# 3. Equity freeze. V29 froze at 15.70 for 412,604 lines.
UNIQ=$(grep -oE 'Portfolio value: [0-9.]+' "$CLEAN" | tail -3000 |
       awk '{print $3}' | sort -u | wc -l)
printf " equity distinct : %s values in last 3000  " "$UNIQ"
awk -v u="$UNIQ" 'BEGIN{print (u+0<=2)?"** FROZEN **":"ok"}'

# 4. Tradeless streak — the env logs it itself.
STREAK=$(grep -oE 'Long period without trades: [0-9]+' "$CLEAN" | tail -1 |
         grep -oE '[0-9]+$')
printf " tradeless streak: %s  " "${STREAK:-0}"
awk -v s="${STREAK:-0}" 'BEGIN{print (s+0>5000)?"** STALLED **":"ok"}'

echo " ---- health ----"
echo " tracebacks      : $(grep -cE 'Traceback \(most recent call last\)|MemoryError' "$CLEAN")"
echo " economic deaths : $(grep -c 'DRAWDOWN_KILL' "$CLEAN")"
grep -oE 'Portfolio value: [0-9.]+' "$CLEAN" | tail -1 | sed 's/^/ /'
echo " ---- checkpoints ----"
ls -1t "$CKPT_DIR" 2>/dev/null | head -3 | sed 's/^/ /'
echo "=========================================================="
