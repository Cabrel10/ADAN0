#!/bin/bash
# live.sh — Quick status check for all ADAN0 paper traders + AetherionOS
# Place at: ~/live.sh   Usage: bash ~/live.sh
# ─────────────────────────────────────────────────────────────────────────

ADAN_DIR=~/webapp/MORNINGSTAR/ADAN0
NOW=$(date '+%Y-%m-%d %H:%M:%S')

echo "══════════════════════════════════════════════════════════"
echo "  ADAN0 PAPER TRADERS — $NOW"
echo "══════════════════════════════════════════════════════════"

# ── Bot status (PID + uptime + CPU) ──────────────────────────────────────
echo ""
echo "▸ PROCESSUS:"
ps aux | grep run_bot | grep -v grep | \
  awk '{printf "  PID=%-8s  CPU=%-5s  MEM=%-5s  Started=%s  %s\n", $2, $3"%", $4"%", $9, $NF}'

if ! ps aux | grep run_bot | grep -v grep | grep -q .; then
  echo "  ⚠️  AUCUN BOT EN COURS !"
fi

# ── Latest tick per bot ───────────────────────────────────────────────────
echo ""
echo "▸ DERNIER TICK PAR BOT:"
for d in $(ls -dt "$ADAN_DIR"/logs/paper/*/ 2>/dev/null | head -5); do
  name=$(basename "$d")
  log="${d%/}/nohup.out"
  if [ ! -f "$log" ]; then continue; fi

  # DEBUG_ACTION format: <date> <time> [INFO] [DEBUG_ACTION] tick=N dir=X size=X tf=X sl=X tp=X
  last_tick=$(grep -E "DEBUG_ACTION" "$log" 2>/dev/null | tail -1 | \
    sed -E 's/.*(tick=[0-9]+).*(dir=[-0-9.]+).*(sl=[-0-9.]+).*(tp=[-0-9.]+).*/\1 \2 \3 \4/')
  last_trade=$(grep "PAPER_TRADE" "$log" 2>/dev/null | tail -1 | \
    sed 's/.*\[PAPER_TRADE\] //' | cut -c1-90)
  stoch=$(grep -c "STOCHASTIC SL/TP calibrator ENABLED" "$log" 2>/dev/null)

  tag=""
  [[ "$stoch" -gt 0 ]] && tag=" [STOCH]"
  echo "  ── $name$tag"
  echo "     tick  : $last_tick"
  echo "     trade : $last_trade"
done

# ── Equity snapshot (CSV) ─────────────────────────────────────────────────
echo ""
echo "▸ TRADES (derniers PnL):"
for d in $(ls -dt "$ADAN_DIR"/logs/paper/*/ 2>/dev/null | head -5); do
  name=$(basename "$d")
  csv=$(ls -t "${d%/}"/trades_*.csv 2>/dev/null | head -1)
  if [ -z "$csv" ]; then continue; fi
  count=$(tail -n +2 "$csv" 2>/dev/null | grep -c .)
  # CSV cols: timestamp,side,symbol,price,size_usd,size_asset,sl_pct,tp_pct,fee_usd,pnl_usd,reason,...
  last_pnl=$(tail -1 "$csv" 2>/dev/null | tr -d '\r' | \
    awk -F',' 'NF>=11{printf "side=%-4s price=%.2f pnl=%+.4f reason=%s", $2, $4, $10, $11}')
  cum_pnl=$(tail -n +2 "$csv" 2>/dev/null | tr -d '\r' | \
    awk -F',' 'NF>=10{s+=$10} END{printf "%+.4f", s}')
  echo "  $name: $count trades | cumPnL=$cum_pnl | $last_pnl"
done

# ── AetherionOS build status ──────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  AETHERION OS"
echo "══════════════════════════════════════════════════════════"
AETH_DIR=~/webapp/MORNINGSTAR/AetherionOS

KERN_ELF="$AETH_DIR/target/x86_64-unknown-none/release/aetherion-kernel"
if pgrep -f "cargo.*aetherion-kernel" &>/dev/null || pgrep -x rustc &>/dev/null; then
  echo "  🔨 Build en cours (cargo release)..."
  tail -1 /tmp/kernel_build.log 2>/dev/null | cut -c1-80
elif [ -f "$KERN_ELF" ]; then
  echo "  ✅ kernel ELF présent: $(du -sh "$KERN_ELF" 2>/dev/null | cut -f1)  $(date -r "$KERN_ELF" '+%Y-%m-%d %H:%M:%S')"
  if grep -q "error\[\|error:" /tmp/kernel_build.log 2>/dev/null; then
    echo "  ⚠️  des erreurs figurent dans /tmp/kernel_build.log"
  fi
else
  echo "  ❌ kernel ELF absent — build non terminé"
  tail -3 /tmp/kernel_build.log 2>/dev/null | sed 's/^/     /'
fi
ISO="$AETH_DIR/target/aetherion-limine.iso"
[ -f "$ISO" ] && echo "  ISO: $(du -sh $ISO | cut -f1)  $(date -r $ISO '+%Y-%m-%d %H:%M')"

cd "$AETH_DIR" 2>/dev/null
echo "  Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)  $(git log --oneline -1 2>/dev/null)"

echo ""
echo "══════════════════════════════════════════════════════════"
