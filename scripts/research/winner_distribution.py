"""(b) Winner/loser distribution + capture ratio across checkpoints.

Consumes the per-trade JSONs produced by scripts/backtest/forensic_trades.py
(logs/validation/forensic/forensic_<k>.json), so it costs NO extra training
compute — it just re-reads the trade samples already captured.

For each checkpoint it reports:
  - winner distribution: count, mean, median, p90, max of pnl_pct>0
  - loser distribution:  count, mean, median, p10, min of pnl_pct<0
  - realized R/R = |avg_win / avg_loss|
  - capture_ratio = mean(pnl_pct / (100*MFE)) over winners  (how much of the
    favourable excursion the agent actually banked; 1.0 = sold the top)
  - giveback_ratio = mean(|pnl| relative to MFE) on losers that had MFE>0
    (trades that were in profit then turned losing)
  - "glass ceiling" check: is best_win clustered (low std) => capped TP?

Answers the user's correct objection that a fixed "TP=2.5%" is arbitrary:
the right TP follows from the winner distribution + capture ratio, not a guess.

Usage:
  PYTHONPATH=src python3 scripts/research/winner_distribution.py \
      --glob 'logs/validation/forensic/forensic_*.json' \
      --out logs/validation/research/winner_dist.json
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ck_steps(name: str) -> int:
    m = re.search(r"(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else -1


def analyze_file(path: str) -> dict:
    d = json.load(open(path))
    trades = d.get("trades_sample", [])
    pnls = np.array([t["pnl_pct"] for t in trades if t.get("pnl_pct") is not None],
                    dtype=float)
    if pnls.size == 0:
        return {"checkpoint": d.get("checkpoint"), "n_trades": 0}

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else None

    # capture ratio on winners: pnl_pct vs 100*MFE (MFE is a ratio)
    cap = []
    giveback = []
    for t in trades:
        p = t.get("pnl_pct"); mfe = t.get("mfe")
        if p is None or mfe is None or mfe <= 0:
            continue
        avail = 100.0 * mfe
        if p > 0:
            cap.append(p / avail)
        elif p < 0:
            # had upside (MFE>0) but ended negative => gave it all back
            giveback.append(avail)  # the profit that was available and lost

    return {
        "checkpoint": d.get("checkpoint"),
        "steps_k": _ck_steps(path) // 1000,
        "n_trades": int(pnls.size),
        "win_rate": round(float((pnls > 0).mean()), 4),
        "winners": {
            "n": int(wins.size),
            "mean": round(avg_win, 4),
            "median": round(float(np.median(wins)), 4) if wins.size else None,
            "p90": round(float(np.percentile(wins, 90)), 4) if wins.size else None,
            "max": round(float(wins.max()), 4) if wins.size else None,
            "std": round(float(wins.std()), 4) if wins.size else None,
        },
        "losers": {
            "n": int(losses.size),
            "mean": round(avg_loss, 4),
            "median": round(float(np.median(losses)), 4) if losses.size else None,
            "p10": round(float(np.percentile(losses, 10)), 4) if losses.size else None,
            "min": round(float(losses.min()), 4) if losses.size else None,
            "std": round(float(losses.std()), 4) if losses.size else None,
        },
        "realized_RR": round(rr, 3) if rr is not None else None,
        "expectancy": round(float(pnls.mean()), 4),
        "capture_ratio_mean": round(float(np.mean(cap)), 3) if cap else None,
        "capture_ratio_median": round(float(np.median(cap)), 3) if cap else None,
        "avg_profit_available_on_losers": round(float(np.mean(giveback)), 4) if giveback else None,
        "glass_ceiling_winners_std": round(float(wins.std()), 4) if wins.size else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="logs/validation/forensic/forensic_*.json")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    files = sorted(glob.glob(str(REPO_ROOT / a.glob)), key=_ck_steps)
    if not files:
        files = sorted(glob.glob(a.glob), key=_ck_steps)
    rows = [analyze_file(f) for f in files]
    rows = [r for r in rows if r.get("n_trades", 0) > 0]

    out = Path(a.out) if a.out else (REPO_ROOT / "logs/validation/research/winner_dist.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    print(f"{'ckpt':>6} {'n':>5} {'WR':>6} {'avgW':>7} {'avgL':>7} {'R/R':>6} "
          f"{'exp':>7} {'capt':>6} {'maxW':>7}")
    for r in rows:
        w, l = r["winners"], r["losers"]
        print(f"{r['steps_k']:>5}k {r['n_trades']:>5} {r['win_rate']:>6.3f} "
              f"{w['mean']:>7.3f} {l['mean']:>7.3f} "
              f"{(r['realized_RR'] or 0):>6.3f} {r['expectancy']:>7.3f} "
              f"{(r['capture_ratio_mean'] or 0):>6.3f} {(w['max'] or 0):>7.3f}")
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
