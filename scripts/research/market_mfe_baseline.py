"""(b-extension) Market MFE baseline vs entered-trade MFE.

Question: does the market OFFER better entries than the agent takes, or is the
whole test market low-MFE? We compute the ex-post MFE/MAE/zone for EVERY 5m bar
on the test split (long direction, same horizon/config as training) and compare
its distribution to the agent's ENTERED trades (from a forensic JSON).

If market-wide GREEN% >> agent's GREEN%  -> the agent SELECTS bad entries.
If market-wide MFE is also ~0.27%        -> fees make the market unwinnable.

Usage:
  PYTHONPATH=src python3 scripts/research/market_mfe_baseline.py \
      --forensic logs/validation/forensic/forensic_430000.json \
      --out logs/validation/research/market_baseline.json
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
    p.add_argument("--forensic", default="logs/validation/forensic/forensic_430000.json")
    p.add_argument("--data", default="data/processed/indicators/test/BTCUSDT/5m.parquet")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    df = pd.read_parquet(REPO_ROOT / a.data)
    df.columns = [c.lower() for c in df.columns]
    zcfg = ZoneConfig()
    horizon = int(getattr(zcfg, "horizon", 36))
    mae_floor = float(getattr(zcfg, "mae_floor", 0.0015))

    n = len(df)
    mfes, maes, zones = [], [], []
    # sample every bar that has a full forward horizon
    for i in range(max(20, 0), n - horizon - 1):
        mfe, mae = compute_mfe_mae(df, i, PivotDirection.LOW, horizon, mae_floor=mae_floor)
        z, _ = classify_zone(mfe, mae, zcfg)
        mfes.append(mfe * 100); maes.append(mae * 100); zones.append(z.value)
    mfes = np.array(mfes); maes = np.array(maes)
    zc = {z: zones.count(z) for z in ("green", "orange", "red")}
    tot = max(1, len(zones))

    market = {
        "n_bars": len(zones),
        "mfe_pct_median": round(float(np.median(mfes)), 4),
        "mfe_pct_mean": round(float(np.mean(mfes)), 4),
        "mfe_pct_p90": round(float(np.percentile(mfes, 90)), 4),
        "mae_pct_median": round(float(np.median(maes)), 4),
        "pct_bars_mfe_ge_0p6": round(float((mfes >= 0.6).mean() * 100), 1),
        "pct_bars_mfe_ge_2p0": round(float((mfes >= 2.0).mean() * 100), 1),
        "zone_distribution_pct": {z: round(zc[z] / tot * 100, 1) for z in zc},
    }

    # agent's entered trades from forensic
    agent = {}
    fp = REPO_ROOT / a.forensic
    if fp.exists():
        d = json.load(open(fp))
        tr = [t for t in d.get("trades_sample", []) if t.get("mfe") is not None]
        if tr:
            amfe = np.array([t["mfe"] * 100 for t in tr])
            azones = [t["zone"] for t in tr if t.get("zone")]
            azc = {z: azones.count(z) for z in ("green", "orange", "red")}
            atot = max(1, len(azones))
            agent = {
                "checkpoint": d.get("checkpoint"),
                "n_trades": len(tr),
                "mfe_pct_median": round(float(np.median(amfe)), 4),
                "mfe_pct_mean": round(float(np.mean(amfe)), 4),
                "zone_distribution_pct": {z: round(azc[z] / atot * 100, 1) for z in azc},
            }

    # verdict
    mg = market["zone_distribution_pct"].get("green", 0)
    ag = agent.get("zone_distribution_pct", {}).get("green", 0)
    if market["mfe_pct_median"] < 0.4 and market["pct_bars_mfe_ge_0p6"] < 30:
        market_verdict = ("MARKET_IS_LOW_MFE — even the BEST possible entries rarely "
                          "reach the fee-positive 0.6% TP. With 0.5% fees this market "
                          "is near-unwinnable at 5m scalp horizon regardless of policy.")
    elif ag < mg - 5:
        market_verdict = ("ENTRY_SELECTION_FAILURE — market offers more GREEN bars "
                          "than the agent enters; the policy picks worse-than-random bars.")
    else:
        market_verdict = "MIXED — see numbers."

    res = {"horizon": horizon, "fees_round_trip_pct": 0.5,
           "market_baseline": market, "agent_entries": agent,
           "VERDICT": market_verdict}
    out = Path(a.out) if a.out else (REPO_ROOT / "logs/validation/research/market_baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
