#!/usr/bin/env python3
"""Rigorous trajectory monitor for a diag CSV (v13 discipline).

<30 points => trajectory table + simple linear regression with 95% CI on the slope.
NO PCA/LDA/SVD (statistical theater at this scale — see HANDOFF §11.3).

Usage: python monitor_slope.py logs/training/diag_v13_1M.csv
"""
import sys, csv
import numpy as np


def slope_ci(x, y):
    """OLS slope + 95% CI. Returns (slope, lo, hi, n)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 3:
        return (float("nan"),) * 3 + (n,)
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    resid = y - yhat
    s2 = (resid @ resid) / (n - 2)
    sxx = ((x - x.mean()) ** 2).sum()
    se = np.sqrt(s2 / sxx) if sxx > 0 else float("nan")
    # t_0.975 ~ 2.0 for modest n (honest approximation; underpowered <10 pts)
    t = 2.0 if n >= 10 else 2.78
    return b, b - t * se, b + t * se, n


def main(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("no data yet"); return
    ts = [int(r["timesteps"]) for r in rows]
    print(f"points={len(rows)}  range={ts[0]}..{ts[-1]}\n")
    print(f"{'step':>7} {'a0_mean':>8} {'pct_buy':>8} {'pct_sell':>8} {'a0_std':>7} {'entropy':>8} {'st_open':>7}")
    for r in rows:
        print(f"{int(r['timesteps']):>7} {float(r['a0_mean']):>+8.3f} "
              f"{float(r['a0_pct_buy']):>8.3f} {float(r['a0_pct_sell']):>8.3f} "
              f"{float(r['a0_std']):>7.3f} {float(r['policy_entropy']):>+8.3f} "
              f"{float(r['steps_open_pct']):>7.3f}")
    print()
    for col, label in [("a0_mean", "a0_mean"), ("a0_pct_buy", "pct_buy"),
                       ("a0_pct_sell", "pct_sell")]:
        y = [float(r[col]) for r in rows]
        b, lo, hi, n = slope_ci(ts, y)
        # per-1000-steps for readability
        sig = "" if (lo <= 0 <= hi) else "  <-- CI excludes 0 (drift real)"
        print(f"{label:>8} slope = {b*1000:+.5f}/1k  95%CI=[{lo*1000:+.5f},{hi*1000:+.5f}]/1k (n={n}){sig}")
    # verdict
    y = [float(r["a0_pct_buy"]) for r in rows]
    last = y[-1]
    print()
    if last > 0.90:
        print(f"VERDICT: pct_buy={last:.3f} > 0.90 -> smart_flat OVER-corrects OR under-powered; consider k adjust")
    elif last < 0.35:
        print(f"VERDICT: pct_buy={last:.3f} < 0.35 -> inverse collapse (flat-forever) risk")
    else:
        print(f"VERDICT: pct_buy={last:.3f} in balanced band [0.35,0.90] -> mechanism holding so far")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/training/diag_v13_1M.csv")
