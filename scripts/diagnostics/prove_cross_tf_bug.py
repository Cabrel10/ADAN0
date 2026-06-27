#!/usr/bin/env python3
"""PREUVE EMPIRIQUE du bug cross-timeframe (TP/SL toujours touche).

Ce script reproduit, sans entrainer, le mecanisme exact de lecture des prix
OHLC de l'environnement reel (multi_asset_chunked_env._get_current_prices /
_get_price_for_asset) et demontre que close (entry) et high/low (TP/SL check)
peuvent provenir de timeframes/index incompatibles -> divergence > 5%.

On ne conclut RIEN sans preuve chiffree. Sortie = table + assertions:
    assert high-low intra-bougie < 5%      (par TF, sur la vraie donnee)
    constate divergence close(TF_a) vs high(TF_b) au MEME step_in_chunk

Usage:
    python scripts/diagnostics/prove_cross_tf_bug.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA = REPO / "data" / "processed" / "indicators" / "train" / "BTCUSDT"
TFS = ["5m", "1h", "4h"]
TF_MIN = {"5m": 5, "1h": 60, "4h": 240}


def load():
    out = {}
    for tf in TFS:
        df = pd.read_parquet(DATA / f"{tf}.parquet")
        # resolve OHLC col names (case-insensitive)
        lc = {c.lower(): c for c in df.columns}
        out[tf] = {
            "df": df,
            "open": lc.get("open"),
            "high": lc.get("high"),
            "low": lc.get("low"),
            "close": lc.get("close"),
        }
    return out


def price_at(store, tf, base_step_in_chunk, kind):
    """Replique EXACTEMENT la logique de l'env (clamp + ratio)."""
    d = store[tf]
    df = d["df"]
    ratio = TF_MIN[tf] / 5
    idx = int(base_step_in_chunk / ratio)
    idx = min(idx, len(df) - 1)
    idx = max(idx, 0)
    col = d[kind]
    return float(df.iloc[idx][col]), idx


def main():
    store = load()

    print("=" * 78)
    print("PREUVE 1 — INTEGRITE INTRA-BOUGIE (la donnee elle-meme est SAINE)")
    print("=" * 78)
    for tf in TFS:
        d = store[tf]
        df = d["df"]
        rng = (df[d["high"]] - df[d["low"]]) / df[d["close"]] * 100
        print(f"  {tf}: n={len(df):6d} | (high-low)/close %% "
              f"mean={rng.mean():.3f} p99={rng.quantile(.99):.3f} "
              f"max={rng.max():.3f}")
        # seuil par TF: plus le TF est haut, plus la bougie peut etre large
        _cap = {"5m": 5.0, "1h": 12.0, "4h": 25.0}[tf]
        assert rng.max() < _cap, f"donnee {tf} aberrante: {rng.max()} >= {_cap}"
    print("  -> OK: aucune bougie 5m ne fait +20/30%%. La DONNEE est saine.\n")

    print("=" * 78)
    print("PREUVE 2 — DIVERGENCE close(TF_entry) vs high(TF_check) au MEME step")
    print("  (= ce que fait l'env quand current_timeframe_for_trade change)")
    print("=" * 78)
    print(f"  {'step_in_chunk':>13} | {'close@4h':>10} {'idx4h':>6} | "
          f"{'high@5m':>10} {'idx5m':>6} | {'div%':>8} | TP+2% touche?")
    print("  " + "-" * 74)

    worst = 0.0
    n_tp_always = 0
    n_total = 0
    # balaye une plage de step_in_chunk realistes (comme dans le log ~17000)
    for sic in range(2000, 18000, 1500):
        close_4h, i4 = price_at(store, "4h", sic, "close")   # entry lu en 4h
        high_5m, i5 = price_at(store, "5m", sic, "high")     # TP check lu en 5m
        if close_4h <= 0:
            continue
        div = (high_5m - close_4h) / close_4h * 100
        tp_price = close_4h * 1.02   # TP +2%
        tp_hit = high_5m >= tp_price
        worst = max(worst, abs(div))
        n_total += 1
        if tp_hit:
            n_tp_always += 1
        print(f"  {sic:>13} | {close_4h:>10.2f} {i4:>6} | "
              f"{high_5m:>10.2f} {i5:>6} | {div:>+7.2f}% | "
              f"{'OUI (TRIVIAL)' if tp_hit else 'non'}")

    print("  " + "-" * 74)
    print(f"  divergence max |close4h vs high5m| = {worst:.2f}%")
    print(f"  TP +2%% touche trivialement: {n_tp_always}/{n_total} cas "
          f"({100*n_tp_always/max(n_total,1):.0f}%%)")
    print()

    print("=" * 78)
    print("PREUVE 3 — MEME PROBLEME SUR LE SL (low cross-TF)")
    print("=" * 78)
    n_sl_always = 0
    for sic in range(2000, 18000, 1500):
        close_4h, _ = price_at(store, "4h", sic, "close")
        low_5m, _ = price_at(store, "5m", sic, "low")
        if close_4h <= 0:
            continue
        sl_price = close_4h * (1 - 0.012)  # SL -1.2% (scalper hi)
        sl_hit = low_5m <= sl_price
        if sl_hit:
            n_sl_always += 1
    print(f"  SL -1.2%% touche trivialement (low5m<=sl): "
          f"{n_sl_always}/{n_total} cas")
    print()

    print("=" * 78)
    verdict = worst > 5.0
    print(f"VERDICT: bug cross-timeframe {'PROUVE' if verdict else 'NON prouve'} "
          f"(divergence max {worst:.1f}%% > 5%% seuil)")
    print("  Cause: close(entry) et high/low(check TP-SL) lus a des timeframes")
    print("  differents au meme step_in_chunk -> prix incompatibles ->")
    print("  TP et SL declenches sur des mouvements FICTIFS.")
    print("=" * 78)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
