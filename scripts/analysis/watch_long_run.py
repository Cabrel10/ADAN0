#!/usr/bin/env python3
"""
watch_long_run.py — verdict chiffré sur un run long (diag CSV), sans PCA/LDA/SVD.
Mesure: pente OLS + CI95 sur a0_mean, pct_buy, pct_sell, et Δ=pct_buy-pct_sell.
Détecte franchissement d'horizon de collapse (pct_buy>=0.90 ou pct_sell>=0.90).

Usage:
  python scripts/analysis/watch_long_run.py [diag_csv]
  (défaut: logs/training/diag_long_hc012.csv)
"""
import sys
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "logs/training/diag_long_hc012.csv"


def slope_ci(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 3:
        return None
    b1, b0 = np.polyfit(x, y, 1)
    yh = b0 + b1 * x
    resid = y - yh
    s2 = (resid @ resid) / (n - 2)
    sxx = ((x - x.mean()) ** 2).sum()
    if sxx <= 0:
        return None
    se = np.sqrt(s2 / sxx)
    from scipy import stats
    t = stats.t.ppf(0.975, n - 2)
    return b1, b1 - t * se, b1 + t * se, n


def main():
    df = pd.read_csv(CSV).sort_values("timesteps").reset_index(drop=True)
    n = len(df)
    last = df.iloc[-1]
    print(f"=== {CSV} — n={n} points, last timestep={int(last['timesteps'])} ===")
    print(df[["timesteps", "a0_mean", "a0_std", "a0_pct_buy",
              "a0_pct_sell", "policy_entropy"]].to_string(index=False))

    if "a0_pct_buy" in df and "a0_pct_sell" in df:
        df["_delta"] = df["a0_pct_buy"] - df["a0_pct_sell"]

    print("\n--- pentes OLS (95% CI) ---")
    for col in ["a0_mean", "a0_pct_buy", "a0_pct_sell", "_delta"]:
        if col in df:
            r = slope_ci(df["timesteps"], df[col])
            if r:
                b1, lo, hi, nn = r
                print(f"  {col:12} slope={b1:+.3e}/step  CI95=[{lo:+.3e},{hi:+.3e}]")

    # verdict horizon collapse
    print("\n--- verdict ---")
    pb = df["a0_pct_buy"]
    ps = df["a0_pct_sell"]
    crossed_buy = df[pb >= 0.90]
    crossed_sell = df[ps >= 0.90]
    if len(crossed_buy):
        print(f"  ⚠ BUY-runaway: pct_buy>=0.90 dès step {int(crossed_buy.iloc[0]['timesteps'])}")
    if len(crossed_sell):
        print(f"  ⚠ SELL-runaway: pct_sell>=0.90 dès step {int(crossed_sell.iloc[0]['timesteps'])}")
    if not len(crossed_buy) and not len(crossed_sell):
        print(f"  ✓ SAIN: ni BUY ni SELL runaway. |Δ| max={df['_delta'].abs().max():.3f} "
              f"(critère équilibre <0.10)")
    print(f"  dernier: pct_buy={last['a0_pct_buy']:.3f} pct_sell={last['a0_pct_sell']:.3f} "
          f"a0_mean={last['a0_mean']:+.3f} a0_std={last['a0_std']:.3f}")


if __name__ == "__main__":
    main()
