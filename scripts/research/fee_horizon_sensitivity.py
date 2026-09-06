"""(decision) Fee / horizon sensitivity — when is this market winnable?

Pure ex-post data computation (no model). For the test 5m split we ask:
"If a PERFECT entry filter only took GREEN-zone bars, and exited at the
MFE-optimal point, what net expectancy results at various fee levels and
horizons?" This bounds what ANY policy could achieve and tells us how to
recalibrate the environment (fees, horizon, TP) so an edge is even possible.

For each (horizon, fee):
  - oracle_green_exp = mean over GREEN bars of (min(MFE, TP_cap) - fee)
    using TP_cap = the realistic exit (we report both raw MFE capture and a
    capped 2.0% TP).
  - breakeven_fee = the fee at which median GREEN MFE == fee.

This is an UPPER BOUND (oracle entry + oracle exit). If even the oracle is
negative at 0.5% fee, no learnable policy can win.

Usage:
  PYTHONPATH=src python3 scripts/research/fee_horizon_sensitivity.py \
      --out logs/validation/research/fee_sensitivity.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    import pandas as pd
    from adan_trading_bot.future_arena.future_zones import (
        ZoneConfig, classify_zone, compute_mfe_mae, PivotDirection,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/indicators/test/BTCUSDT/5m.parquet")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    df = pd.read_parquet(REPO_ROOT / a.data)
    df.columns = [c.lower() for c in df.columns]
    zcfg = ZoneConfig()
    mae_floor = float(getattr(zcfg, "mae_floor", 0.0015))

    horizons = [36, 72, 144, 288]   # 3h, 6h, 12h, 24h at 5m
    fees = [0.1, 0.2, 0.3, 0.5]     # round-trip %
    tp_cap = 2.0                    # current scalper TP target %

    results = {}
    n = len(df)
    for H in horizons:
        mfes, zones = [], []
        for i in range(20, n - H - 1):
            mfe, mae = compute_mfe_mae(df, i, PivotDirection.LOW, H, mae_floor=mae_floor)
            z, _ = classify_zone(mfe, mae, zcfg)
            mfes.append(mfe * 100); zones.append(z.value)
        mfes = np.array(mfes); zones = np.array(zones)
        green = mfes[zones == "green"]
        allm = mfes
        block = {
            "n_bars": int(len(mfes)),
            "green_pct": round(float((zones == "green").mean() * 100), 1),
            "median_mfe_all": round(float(np.median(allm)), 4),
            "median_mfe_green": round(float(np.median(green)), 4) if green.size else None,
            "mean_mfe_green": round(float(np.mean(green)), 4) if green.size else None,
            "per_fee": {},
        }
        for fee in fees:
            # oracle: enter only GREEN, exit at min(MFE, tp_cap), pay fee
            capt_green = np.minimum(green, tp_cap) - fee if green.size else np.array([])
            capt_all = np.minimum(allm, tp_cap) - fee
            block["per_fee"][f"{fee}%"] = {
                "oracle_green_exp": round(float(capt_green.mean()), 4) if capt_green.size else None,
                "oracle_green_winnable": bool(capt_green.mean() > 0) if capt_green.size else None,
                "oracle_all_exp": round(float(capt_all.mean()), 4),
            }
        results[f"H{H}"] = block

    # find the (horizon, fee) cells where oracle GREEN is positive
    winnable = []
    for h, b in results.items():
        for fee, v in b["per_fee"].items():
            if v.get("oracle_green_winnable"):
                winnable.append(f"{h}@fee{fee} (exp={v['oracle_green_exp']:+.3f}%)")

    res = {
        "tp_cap_pct": tp_cap,
        "note": ("oracle = PERFECT entry (GREEN only) + PERFECT exit (sell at "
                 "min(MFE,TP)) - fee. This is an UPPER BOUND on any policy."),
        "by_horizon": results,
        "winnable_cells": winnable,
        "VERDICT": (
            "RECALIBRATION_PATH: " + ("; ".join(winnable) if winnable else
            "NO cell winnable even with oracle entry+exit — fees too high vs MFE.")
        ),
    }
    out = Path(a.out) if a.out else (REPO_ROOT / "logs/validation/research/fee_sensitivity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
