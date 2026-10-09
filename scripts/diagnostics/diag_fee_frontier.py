#!/usr/bin/env python3
"""diag_fee_frontier.py — frontière de frais par timeframe.

Les deux sondes précédentes ont mesuré, sur le 5m TRAIN BTC :
  - inconditionnel : 42/42 paires à EV ≈ -frais (jeu à somme nulle avant frais)
  - conditionné    : meilleur filtre simple edge brut +0.107% << frais 0.400%

Question résiduelle décisive : la contrainte liante est-elle le NIVEAU de frais
ou le TIMEFRAME ? Un mouvement de ±2-3% est 12x plus fréquent sur 1h que sur
5m, alors que les frais sont constants par trade. Cette sonde mesure, pour
chaque timeframe réel (5m/1h/4h) aux horizons MaxDuration de la config
(5m: H=40/100, 1h: H=40, 4h: H=30) :

  1. le meilleur edge brut sur une grille (sl, tp) élargie,
  2. la FRONTIÈRE DE FRAIS : frais_round_trip_max tels que EV_net > 0
     (= edge brut, puisque EV_net = EV_brut - frais),
  3. la meilleure paire à chaque niveau de frais simulé
     {0.10%, 0.20%, 0.30%, 0.40%}.

Si même en 4h la frontière reste < 0.40%, la conclusion est définitive :
aucun run ne peut être économiquement viable à ces frais, quelle que soit la
policy. Si la frontière dépasse 0.40% sur un timeframe, c'est LA configuration
à tester en priorité.

Lecture seule. Sortie JSON dans logs/validation/ + tableaux stdout.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IND = ROOT / "data/processed/indicators/train/BTCUSDT_BINANCE"
OUT_DIR = ROOT / "logs/validation"

FEE_LEVELS = [0.001, 0.002, 0.003, 0.004]  # round-trip
FEES_ACTUAL = 0.004

# Grille élargie : les mouvements 1h/4h sont plus amples.
SL_GRID = [0.003, 0.005, 0.008, 0.012, 0.016, 0.020, 0.030]
TP_GRID = [0.006, 0.010, 0.0135, 0.020, 0.030, 0.045, 0.060]

# (fichier, horizons MaxDuration en barres du TF, stride)
TF_SPECS = {
    "5m": ("5m.parquet", [40, 100], 3),
    "1h": ("1h.parquet", [40], 1),
    "4h": ("4h.parquet", [30], 1),
}


def first_passage(high, low, close, sl, tp, horizon, stride):
    n = len(close)
    idx = np.arange(0, n - horizon - 1, stride)
    entry = close[idx]
    hit_tp = np.zeros(len(idx), dtype=bool)
    hit_sl = np.zeros(len(idx), dtype=bool)
    for h in range(1, horizon + 1):
        hh = high[idx + h] / entry - 1.0
        ll = low[idx + h] / entry - 1.0
        both = (~hit_tp) & (~hit_sl) & (hh >= tp) & (ll <= -sl)
        hit_sl |= both
        hit_tp |= (~hit_tp) & (~hit_sl) & (hh >= tp)
        hit_sl |= (~hit_tp) & (~hit_sl) & (ll <= -sl)
    md = (~hit_tp) & (~hit_sl)
    md_ret = close[idx + horizon] / entry - 1.0
    mean_md = float(md_ret[md].mean()) if md.any() else 0.0
    # EV BRUT (avant frais) — la frontière de frais est exactement cette valeur
    ev_gross = float(hit_tp.mean()) * tp - float(hit_sl.mean()) * sl \
        + float(md.mean()) * mean_md
    return ev_gross, float(hit_tp.mean()), float(hit_sl.mean()), len(idx)


def main():
    report = {"fees_actual": FEES_ACTUAL, "fee_levels": FEE_LEVELS,
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "timeframes": {}}
    print(f"# Frontière de frais par timeframe — BTC TRAIN, fees actuels {FEES_ACTUAL:.3%}\n")
    print("| TF | H | meilleure (sl,tp) | edge BRUT | frontière frais max | viable à 0.40% ? |")
    print("|---|---|---|---|---|---|")
    for tf, (fname, horizons, stride) in TF_SPECS.items():
        path = IND / fname
        df = pd.read_parquet(path, columns=["high", "low", "close"])
        high = df["high"].to_numpy(np.float64)
        low = df["low"].to_numpy(np.float64)
        close = df["close"].to_numpy(np.float64)
        tf_res = {"rows": len(close), "horizons": {}}
        for H in horizons:
            best = None
            for sl in SL_GRID:
                for tp in TP_GRID:
                    ev_g, p_tp, p_sl, n = first_passage(high, low, close, sl, tp, H, stride)
                    if best is None or ev_g > best[0]:
                        best = (ev_g, sl, tp, p_tp, p_sl, n)
            ev_g, sl, tp, p_tp, p_sl, n = best
            frontier = ev_g  # frais max pour EV net > 0
            viable = "OUI" if frontier > FEES_ACTUAL else "NON"
            tf_res["horizons"][f"H{H}"] = dict(
                best_sl=sl, best_tp=tp, edge_gross=round(ev_g, 5),
                fee_frontier=round(frontier, 5), viable_at_actual=viable,
                p_tp=round(p_tp, 4), p_sl=round(p_sl, 4), n=n,
                # EV net à chaque niveau de frais simulé
                ev_net_by_fee={f"{f:.3%}": round(ev_g - f, 5) for f in FEE_LEVELS})
            print(f"| {tf} | {H} | ({sl:.4f},{tp:.4f}) | {ev_g:+.4%} "
                  f"| {frontier:+.4%} | {viable} |")
        report["timeframes"][tf] = tf_res

    print("\n## Lecture")
    print("- edge BRUT = EV avant frais de la meilleure paire de la grille.")
    print("- frontière frais max = edge brut : en dessous, le trade est net +EV ;")
    print(f"  au-dessus, aucune policy ne peut être profitable. Actuel : {FEES_ACTUAL:.3%}.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"fee_frontier_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nRapport JSON : {out}")


if __name__ == "__main__":
    main()
