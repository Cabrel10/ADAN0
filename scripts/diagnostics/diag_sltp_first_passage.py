#!/usr/bin/env python3
"""diag_sltp_first_passage.py — mesure la géométrie SL/TP sur les données réelles.

Pour chaque paire (sl, tp) d'une grille, et chaque barre d'entrée échantillonnée
du parquet 5m TRAIN de l'actif, on simule le premier passage :
    TP touché en premier (high >= entry*(1+tp))  -> gain +tp
    SL touché en premier (low  <= entry*(1-sl))  -> perte -sl
    ni l'un ni l'autre avant H barres            -> MaxDuration, close au marché
Si high et low touchent tous deux la même barre, on tranche conservativement
en faveur du SL (hypothèse pessimiste standard quand l'ordre intra-barre
est inconnu).

EV net par trade = p_tp*tp - p_sl*sl + p_md*E[ret_close_H | MD] - round_trip_fees.

Sortie : JSON dans logs/validation/ + tableau Markdown sur stdout.
Lecture seule : ne modifie aucun fichier de production.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = ROOT / "data/processed/indicators/train/BTCUSDT_BINANCE/5m.parquet"
OUT_DIR = ROOT / "logs/validation"

# Frais : le run v30_500k a mesuré 0.40% round-trip effectif (config 0.004).
ROUND_TRIP_FEES = 0.004

# Grille : couvre la boîte actuelle BTC (sl 0.003-0.0235, tp 0.0135-0.0222)
# et des voisinages plus serrés / plus larges.
SL_GRID = [0.003, 0.005, 0.008, 0.012, 0.016, 0.020, 0.0235]
TP_GRID = [0.006, 0.010, 0.0135, 0.018, 0.0222, 0.030]


def first_passage(high, low, close, sl, tp, horizon, stride):
    """Retourne (p_tp, p_sl, p_md, mean_md_ret, n) par échantillonnage stride."""
    n = len(close)
    idx = np.arange(0, n - horizon - 1, stride)
    entry = close[idx]
    hit_tp = np.zeros(len(idx), dtype=bool)
    hit_sl = np.zeros(len(idx), dtype=bool)
    run_max = np.full(len(idx), -np.inf)
    run_min = np.full(len(idx), np.inf)
    for h in range(1, horizon + 1):
        hh = high[idx + h] / entry - 1.0
        ll = low[idx + h] / entry - 1.0
        run_max = np.maximum(run_max, hh)
        run_min = np.minimum(run_min, ll)
        # même barre touche les deux -> SL (pessimiste) : on marque SL d'abord
        both = (~hit_tp) & (~hit_sl) & (hh >= tp) & (ll <= -sl)
        hit_sl |= both
        new_tp = (~hit_tp) & (~hit_sl) & (hh >= tp)
        hit_tp |= new_tp
        new_sl = (~hit_tp) & (~hit_sl) & (ll <= -sl)
        hit_sl |= new_sl
    md = (~hit_tp) & (~hit_sl)
    md_ret = (close[idx + horizon] / entry - 1.0)
    mean_md = float(md_ret[md].mean()) if md.any() else 0.0
    p_tp = float(hit_tp.mean())
    p_sl = float(hit_sl.mean())
    p_md = float(md.mean())
    ev = p_tp * tp - p_sl * sl + p_md * mean_md - ROUND_TRIP_FEES
    return p_tp, p_sl, p_md, mean_md, len(idx), ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    ap.add_argument("--horizons", default="40,100",
                    help="horizons MaxDuration en barres 5m (scalper=40, 5m=100)")
    ap.add_argument("--stride", type=int, default=3,
                    help="échantillonnage des barres d'entrée (1 = toutes)")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet, columns=["open", "high", "low", "close"])
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    horizons = [int(x) for x in args.horizons.split(",")]

    print(f"# Premier passage SL/TP — {Path(args.parquet).name}")
    print(f"rows={len(close)} stride={args.stride} fees={ROUND_TRIP_FEES:.3%}\n")

    report = {"parquet": str(args.parquet), "rows": len(close),
              "fees": ROUND_TRIP_FEES, "stride": args.stride,
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "grids": {}}

    for H in horizons:
        rows = []
        for sl in SL_GRID:
            for tp in TP_GRID:
                p_tp, p_sl, p_md, m_md, n, ev = first_passage(
                    high, low, close, sl, tp, H, args.stride)
                rr = tp / sl
                rows.append(dict(sl=sl, tp=tp, rr=round(rr, 2),
                                 p_tp=round(p_tp, 4), p_sl=round(p_sl, 4),
                                 p_md=round(p_md, 4), md_ret=round(m_md, 5),
                                 ev_net=round(ev, 5), n=n))
        report["grids"][f"H{H}"] = rows
        print(f"## Horizon H={H} barres 5m")
        print("| sl | tp | RR | P(TP) | P(SL) | P(MD) | E[ret\\|MD] | EV net |")
        print("|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: -r["ev_net"]):
            flag = " **+EV**" if r["ev_net"] > 0 else ""
            print(f"| {r['sl']:.4f} | {r['tp']:.4f} | {r['rr']:.2f} "
                  f"| {r['p_tp']:.3f} | {r['p_sl']:.3f} | {r['p_md']:.3f} "
                  f"| {r['md_ret']:+.4f} | {r['ev_net']:+.4f}{flag} |")
        pos = [r for r in rows if r["ev_net"] > 0]
        print(f"\nPaires +EV : {len(pos)}/{len(rows)}")
        cur = [r for r in rows if abs(r["sl"] - 0.0235) < 1e-9 and abs(r["tp"] - 0.0135) < 1e-9]
        if cur:
            print(f"Paire actuelle (sl_hi/tp_lo) 0.0235/0.0135 : EV={cur[0]['ev_net']:+.4f}")
        print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"sltp_first_passage_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Rapport JSON : {out}")


if __name__ == "__main__":
    main()
