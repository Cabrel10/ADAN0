#!/usr/bin/env python3
"""
run_metrics.py — READ-ONLY trading KPIs for the live 500k runs.

Parses logs/exp5y/run500k_{BTC,DOGE}.log (never touches the processes) and
reports the metrics requested:
  - EV radar : PPO explained_variance trajectory (last N iterations)
  - capital  : portfolio value trajectory (start / min / max / last / Δ%)
  - #trades  : total closes, and per close-reason breakdown
  - %TP / %SL: share of closes that were take_profit vs stop_loss
  - winrate  : share of closed trades with net PnL > 0
  - PF       : profit factor = sum(win PnL) / |sum(loss PnL)|
  - RR       : reward/risk = avg win / avg loss (absolute)

Pure log parsing => zero CPU competition with the training runs.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs" / "exp5y"
RUNS = {"BTC": LOGS / "run500k_BTC.log", "DOGE": LOGS / "run500k_DOGE.log"}

# [POSITION FERMÉE] ASSET: qty @ entry -> exit | PnL: $-0.06 (brut $-0.01, frais $0.0539) | Raison: MaxDuration
CLOSE_RE = re.compile(
    r"POSITION FERMÉE\]\s+(?P<asset>\S+):\s+[\d.]+\s+@\s+(?P<entry>[\d.]+)\s+->\s+(?P<exit>[\d.]+)\s+"
    r"\|\s+PnL:\s+\$(?P<net>[-+]?[\d.]+)\s+\(brut\s+\$(?P<gross>[-+]?[\d.]+),\s+frais\s+\$(?P<fee>[\d.]+)\)\s+"
    r"\|\s+Raison:\s+(?P<reason>\S+)"
)
PV_RE = re.compile(r"Portfolio value:\s*([\d.]+)")
EV_RE = re.compile(r"explained_variance\s*\|\s*([-\d.eE+]+)")
KL_RE = re.compile(r"approx_kl\s*\|\s*([-\d.eE+]+)")
TS_RE = re.compile(r"total_timesteps\s*\|\s*(\d+)")


def _pct(n, d):
    return round(100.0 * n / d, 2) if d else 0.0


def analyse(tag, path):
    if not path.exists():
        return {"tag": tag, "error": "log missing"}
    closes, pvs, evs, kls, tss = [], [], [], [], []
    with path.open(errors="replace") as fh:
        for line in fh:
            m = CLOSE_RE.search(line)
            if m:
                closes.append((float(m["net"]), m["reason"]))
                continue
            pm = PV_RE.search(line)
            if pm:
                pvs.append(float(pm.group(1)))
            em = EV_RE.search(line)
            if em:
                try:
                    evs.append(float(em.group(1)))
                except ValueError:
                    pass
            km = KL_RE.search(line)
            if km:
                try:
                    kls.append(float(km.group(1)))
                except ValueError:
                    pass
            tm = TS_RE.search(line)
            if tm:
                tss.append(int(tm.group(1)))

    n = len(closes)
    reasons = {}
    wins = [p for p, _ in closes if p > 0]
    losses = [p for p, _ in closes if p <= 0]
    for _, r in closes:
        reasons[r] = reasons.get(r, 0) + 1
    sum_win = sum(wins)
    sum_loss = abs(sum(losses))
    pf = round(sum_win / sum_loss, 3) if sum_loss > 0 else (float("inf") if sum_win > 0 else 0.0)
    avg_win = (sum_win / len(wins)) if wins else 0.0
    avg_loss = (sum_loss / len(losses)) if losses else 0.0
    rr = round(avg_win / avg_loss, 3) if avg_loss > 0 else (float("inf") if avg_win > 0 else 0.0)

    return {
        "tag": tag,
        "timesteps": tss[-1] if tss else None,
        "n_trades": n,
        "reasons": reasons,
        "pct_tp": _pct(reasons.get("take_profit", 0), n),
        "pct_sl": _pct(reasons.get("stop_loss", 0), n),
        "pct_maxdur": _pct(reasons.get("MaxDuration", 0), n),
        "winrate": _pct(len(wins), n),
        "profit_factor": pf,
        "rr": rr,
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "net_sum": round(sum(p for p, _ in closes), 4),
        "capital": {
            "start": round(pvs[0], 3) if pvs else None,
            "min": round(min(pvs), 3) if pvs else None,
            "max": round(max(pvs), 3) if pvs else None,
            "last": round(pvs[-1], 3) if pvs else None,
            "delta_pct": round(100.0 * (pvs[-1] - pvs[0]) / pvs[0], 2) if pvs and pvs[0] else None,
        },
        "ev_radar": [round(x, 3) for x in evs[-10:]],
        "kl_radar": [round(x, 4) for x in kls[-10:]],
    }


def render(r):
    if r.get("error"):
        return f"[{r['tag']}] {r['error']}"
    c = r["capital"]
    lines = [
        f"===== {r['tag']}  (timesteps={r['timesteps']}) =====",
        f"  trades       : {r['n_trades']}   reasons={r['reasons']}",
        f"  %TP / %SL    : {r['pct_tp']}%  / {r['pct_sl']}%   (MaxDuration {r['pct_maxdur']}%)",
        f"  winrate      : {r['winrate']}%",
        f"  profit factor: {r['profit_factor']}    RR (avgWin/avgLoss): {r['rr']}",
        f"  avg win/loss : +{r['avg_win']} / -{r['avg_loss']}   net PnL sum: {r['net_sum']}",
        f"  capital      : start={c['start']} min={c['min']} max={c['max']} last={c['last']} Δ={c['delta_pct']}%",
        f"  EV radar     : {r['ev_radar']}",
        f"  KL radar     : {r['kl_radar']}",
    ]
    return "\n".join(lines)


def main():
    reports = {tag: analyse(tag, p) for tag, p in RUNS.items()}
    print("=" * 74)
    print("ADAN 500k — TRADING METRICS (read-only, from logs)")
    print("=" * 74)
    for tag in ("BTC", "DOGE"):
        print(render(reports[tag]))
    # BTC vs DOGE compact comparison
    b, d = reports["BTC"], reports["DOGE"]
    if not b.get("error") and not d.get("error"):
        print("-" * 74)
        print("COMPARISON        BTC        DOGE")
        rows = [
            ("trades", b["n_trades"], d["n_trades"]),
            ("winrate%", b["winrate"], d["winrate"]),
            ("PF", b["profit_factor"], d["profit_factor"]),
            ("RR", b["rr"], d["rr"]),
            ("%TP", b["pct_tp"], d["pct_tp"]),
            ("%SL", b["pct_sl"], d["pct_sl"]),
            ("capitalΔ%", b["capital"]["delta_pct"], d["capital"]["delta_pct"]),
        ]
        for name, bv, dv in rows:
            print(f"  {name:<14} {str(bv):<10} {str(dv)}")
    print("=" * 74)
    import json
    (LOGS / "run_metrics_snapshot.json").write_text(json.dumps(reports, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
