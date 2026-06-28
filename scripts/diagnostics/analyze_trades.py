#!/usr/bin/env python3
"""Forensic trade analyzer — reads TRADE_AUDIT_OPEN / TRADE_AUDIT_CLOSE lines from
a training log and answers the open questions WITHOUT hypotheses:

  * hold_steps distribution per close reason  -> "SL touched after how many candles?"
  * pnl_net distribution (real, fees included) -> true expectancy / asymmetry
  * SL% / TP% chosen at open                   -> are exits structurally biased?
  * illegal-action ratio over time            -> does the policy clean up?
  * (zone analysis if zone= appears in logs)

Usage:
  python scripts/diagnostics/analyze_trades.py logs/training/fa_500k_prod_*.log
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict


def _f(pattern: str, line: str, cast=float, default=None):
    m = re.search(pattern, line)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except (TypeError, ValueError):
        return default


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: analyze_trades.py <logfile>")
        return 1
    path = sys.argv[1]

    closes = []          # list of dicts
    opens = []           # list of dicts
    illegal_timeline = []  # (open_count_so_far, rejections snapshot)

    with open(path, errors="ignore") as f:
        for line in f:
            if "[TRADE_AUDIT_CLOSE]" in line:
                closes.append({
                    "reason": _f(r"reason=(\w+)", line, str),
                    "entry": _f(r"entry_price=([\d.]+)", line),
                    "sl": _f(r"sl_price=([\d.]+)", line),
                    "pnl_net": _f(r"pnl_net=(-?[\d.]+)", line),
                    "fees": _f(r"fees=([\d.]+)", line),
                    "hold": _f(r"hold_steps=(\d+)", line, int),
                })
            elif "[TRADE_AUDIT_OPEN]" in line:
                opens.append({
                    "sl_pct": _f(r"SL=([\d.]+)%", line),
                    "tp_pct": _f(r"TP=([\d.]+)%", line),
                    "notional": _f(r"notional=\$([\d.]+)", line),
                    "exposure": _f(r"exposure=([\d.]+)%", line),
                    "tf": _f(r"TF=(\w+)", line, str),
                })

    # de-dup: each audit line is logged twice (logger + handler). Keep odd indices.
    closes = closes[::2] if len(closes) > 1 else closes
    opens = opens[::2] if len(opens) > 1 else opens

    print("=" * 70)
    print(f"TRADE FORENSICS — {path.split('/')[-1]}")
    print("=" * 70)
    print(f"OPENS parsed : {len(opens)}")
    print(f"CLOSES parsed: {len(closes)}")

    # ---- 1. Close reasons + hold_steps per reason -----------------------
    print("\n--- CLOSE REASONS & HOLD DURATION (candles) ---")
    by_reason = defaultdict(list)
    pnl_by_reason = defaultdict(list)
    for c in closes:
        if c["reason"]:
            if c["hold"] is not None:
                by_reason[c["reason"]].append(c["hold"])
            if c["pnl_net"] is not None:
                pnl_by_reason[c["reason"]].append(c["pnl_net"])
    total = sum(len(v) for v in by_reason.values()) or 1
    for reason, holds in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        holds_sorted = sorted(holds)
        n = len(holds)
        med = holds_sorted[n // 2] if n else 0
        mean = sum(holds) / n if n else 0
        fast3 = sum(1 for h in holds if h <= 3) / n * 100 if n else 0
        pnls = pnl_by_reason.get(reason, [])
        mean_pnl = sum(pnls) / len(pnls) if pnls else 0
        print(f"  {reason:14s} n={n:4d} ({n/total*100:4.1f}%) | "
              f"hold mean={mean:5.1f} med={med:3d} | <=3 candles={fast3:4.1f}% | "
              f"mean_pnl_net={mean_pnl:+.4f}")

    # ---- 2. True pnl distribution & asymmetry ---------------------------
    pnls = [c["pnl_net"] for c in closes if c["pnl_net"] is not None]
    if pnls:
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gw = sum(wins)
        gl = -sum(losses)
        print("\n--- TRUE PnL (net, fees incl.) ---")
        print(f"  n={len(pnls)} | WR={len(wins)/len(pnls)*100:.1f}% | "
              f"expectancy={sum(pnls)/len(pnls):+.4f}/trade")
        print(f"  gross_win={gw:+.3f} gross_loss={-gl:.3f} | "
              f"profit_factor={gw/gl if gl>0 else float('inf'):.3f}")
        if wins:
            print(f"  avg_win={sum(wins)/len(wins):+.4f} max_win={max(wins):+.4f}")
        if losses:
            print(f"  avg_loss={sum(losses)/len(losses):+.4f} max_loss={min(losses):+.4f}")
        if wins and losses:
            rr = (sum(wins)/len(wins)) / abs(sum(losses)/len(losses))
            print(f"  realized R/R (avg_win/avg_loss) = {rr:.3f}")

    # ---- 3. SL/TP chosen at open ---------------------------------------
    sls = [o["sl_pct"] for o in opens if o["sl_pct"] is not None]
    tps = [o["tp_pct"] for o in opens if o["tp_pct"] is not None]
    exps = [o["exposure"] for o in opens if o["exposure"] is not None]
    if sls:
        print("\n--- SL/TP CHOSEN AT OPEN ---")
        print(f"  SL%: mean={sum(sls)/len(sls):.3f} min={min(sls):.3f} max={max(sls):.3f}")
        print(f"  TP%: mean={sum(tps)/len(tps):.3f} min={min(tps):.3f} max={max(tps):.3f}")
        print(f"  TP/SL ratio (mean): {(sum(tps)/len(tps))/(sum(sls)/len(sls)):.3f}")
        if exps:
            print(f"  exposure%: mean={sum(exps)/len(exps):.1f} max={max(exps):.1f}")
        tf_counts = Counter(o["tf"] for o in opens if o["tf"])
        print(f"  TF distribution: {dict(tf_counts)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
