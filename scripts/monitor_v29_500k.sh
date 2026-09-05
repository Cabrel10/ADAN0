#!/usr/bin/env bash
# Monitor the V29 500k BTCUSDT_BINANCE run.
#
# Reads ONLY the run's own log — never re-runs anything, never touches the
# training process. Safe to call at any time.
#
#   bash scripts/monitor_v29_500k.sh            # one snapshot
#   watch -n 60 bash scripts/monitor_v29_500k.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

LOG="logs/v29_500k/run.log"
CKPT_DIR="checkpoints/v29_500k"
TARGET=500000

if ! pgrep -f "steps 500000" >/dev/null 2>&1 && \
   ! pgrep -f "500000" >/dev/null 2>&1; then
  STATE="STOPPED"
else
  STATE="RUNNING"
fi

echo "=========================================================="
echo " V29 500k  BTCUSDT_BINANCE   [$STATE]   $(date '+%F %T')"
echo "=========================================================="

if [[ ! -f "$LOG" ]]; then
  echo "no log yet: $LOG"
  exit 0
fi

CLEAN=$(mktemp)
tr -d '\000' < "$LOG" > "$CLEAN"

TS=$(grep -E '^\|\s+total_timesteps' "$CLEAN" | tail -1 |
     grep -oE '[0-9]+' | tail -1)
TS=${TS:-0}
PCT=$(awk -v a="$TS" -v b="$TARGET" 'BEGIN{printf "%.2f", (a/b)*100}')
echo " progress        : ${TS} / ${TARGET}  (${PCT}%)"

for k in explained_variance approx_kl clip_fraction value_loss \
         policy_gradient_loss entropy_loss std n_updates fps; do
  V=$(grep -E "^\|\s+${k}\s" "$CLEAN" | tail -1 |
      awk -F'|' '{gsub(/ /,"",$3); print $3}')
  [[ -n "${V:-}" ]] && printf " %-16s: %s\n" "$k" "$V"
done

# explained_variance trajectory — the metric the terminated/truncated fix
# was supposed to unblock. Negative means the value head explains less than
# a constant predictor.
echo " ---- explained_variance trajectory (last 12) ----"
grep -E '^\|\s+explained_variance' "$CLEAN" | tail -12 |
  awk -F'|' '{gsub(/ /,"",$3); printf " %s", $3} END{print ""}'

EV_NEG=$(grep -E '^\|\s+explained_variance' "$CLEAN" |
         awk -F'|' '{gsub(/ /,"",$3); if ($3+0 < 0) c++} END{print c+0}')
EV_TOT=$(grep -cE '^\|\s+explained_variance' "$CLEAN")
echo " ev negative     : ${EV_NEG} / ${EV_TOT} updates"

echo " ---- health ----"
# NOTE: the env logs routine lifecycle events at CRITICAL level ("NOUVELLE
# INSTANCE ENV", "NOUVEAU DBE", "RESET appele"), so grepping CRITICAL gives
# false positives. Count only real Python tracebacks / fatal errors.
ERRS=$(grep -cE 'Traceback \(most recent call last\)|MemoryError|^[A-Za-z_.]*Error:' "$CLEAN")
echo " tracebacks      : $ERRS"
FA=$(grep -c 'FA_WATCHDOG CRITICAL' "$CLEAN")
FALAST=$(grep 'FA_WATCHDOG CRITICAL' "$CLEAN" | tail -1 |
         grep -oE 'future_share=[0-9.]+%')
echo " fa_watchdog     : ${FA} hits ${FALAST:-}   (target future_share<40%)"
KILL=$(grep -cE 'DRAWDOWN_KILL|BANKRUPT' "$CLEAN")
echo " economic deaths : $KILL"
TRUNC=$(grep -oE 'termination_kind.{0,14}' "$CLEAN" | grep -c truncated)
echo " truncated seen  : $TRUNC"

echo " ---- portfolio ----"
grep -oE 'Portfolio value: [0-9.]+' "$CLEAN" | tail -1
echo " ---- checkpoints ----"
ls -1t "$CKPT_DIR" 2>/dev/null | head -4 | sed 's/^/ /'

rm -f "$CLEAN"
echo "=========================================================="
