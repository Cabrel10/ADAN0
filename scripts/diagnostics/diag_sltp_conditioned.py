#!/usr/bin/env python3
"""diag_sltp_conditioned.py — un filtre d'ENTRÉE peut-il rendre une paire
SL/TP nettement +EV après frais ?

La sonde diag_sltp_first_passage.py a mesuré EV ≈ -frais sur TOUTE la grille
inconditionnelle (42/42 paires, H=40 et H=100) : le 5m BTC est un jeu à somme
nulle avant frais. Aucune géométrie statique ne peut sauver le run.

Reste la question décisive pour le RL : existe-t-il un CONDITIONNEMENT simple
(= ce qu'une policy peut apprendre) qui crée un edge net ? On teste ici des
filtres d'entrée canoniques, avec la même simulation de premier passage :

  momentum_k   : entrer LONG seulement si ret sur k barres > seuil
  meanrev_k    : entrer LONG seulement si ret sur k barres < -seuil
  rsi_low      : entrer seulement si RSI14 < 30 (survendu)
  rsi_high     : entrer seulement si RSI14 > 70 (suracheté)
  atr_low      : entrer seulement si atr_pct < p25 (calme)
  atr_high     : entrer seulement si atr_pct > p75 (volatil)
  trend_up     : ema_20_ratio > 1 (prix au-dessus EMA20)
  trend_down   : ema_20_ratio < 1

Pour chaque filtre actif, on rapporte la meilleure paire (sl, tp) de la grille
et son EV net. Verdict : si même le MEILLEUR (filtre, paire) reste < 0,
l'edge n'est pas dans ces features simples à cet horizon — la conclusion
"priorité = baisser les frais / changer d'horizon" tient. Si un filtre passe
nettement au-dessus de 0, c'est une piste de signal exploitable.

Lecture seule. Sortie JSON dans logs/validation/ + tableau stdout.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data/processed/indicators/train/BTCUSDT_BINANCE/5m.parquet"
OUT_DIR = ROOT / "logs/validation"
ROUND_TRIP_FEES = 0.004

SL_GRID = [0.003, 0.005, 0.008, 0.012, 0.016, 0.020, 0.0235]
TP_GRID = [0.006, 0.010, 0.0135, 0.018, 0.0222, 0.030]
HORIZONS = [40, 100]
STRIDE = 3


def first_passage_idx(idx, high, low, close, sl, tp, horizon):
    """Premier passage pour un ensemble d'indices d'entrée pré-filtrés."""
    n = len(close)
    idx = idx[idx + horizon < n]
    if len(idx) < 200:
        return None
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
    p_tp = float(hit_tp.mean())
    p_sl = float(hit_sl.mean())
    ev = p_tp * tp - p_sl * sl + float(md.mean()) * mean_md - ROUND_TRIP_FEES
    return ev, p_tp, p_sl, len(idx)


def main():
    df = pd.read_parquet(PARQUET)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    atr = df["atr_pct"].to_numpy(np.float64)
    rsi = df["rsi_14"].to_numpy(np.float64)
    ema = df["ema_20_ratio"].to_numpy(np.float64)
    n = len(close)

    ret6 = np.full(n, np.nan); ret6[6:] = close[6:] / close[:-6] - 1.0
    ret24 = np.full(n, np.nan); ret24[24:] = close[24:] / close[:-24] - 1.0
    atr_p25 = np.nanpercentile(atr, 25)
    atr_p75 = np.nanpercentile(atr, 75)

    base = np.arange(0, n, STRIDE)
    filters = {
        "ALL (inconditionnel)": base,
        "momentum_6b>+0.1%": base[ret6[base] > 0.001],
        "momentum_24b>+0.3%": base[ret24[base] > 0.003],
        "meanrev_6b<-0.1%": base[ret6[base] < -0.001],
        "meanrev_24b<-0.3%": base[ret24[base] < -0.003],
        "rsi<30": base[rsi[base] < 30],
        "rsi>70": base[rsi[base] > 70],
        f"atr<p25({atr_p25:.4f})": base[atr[base] < atr_p25],
        f"atr>p75({atr_p75:.4f})": base[atr[base] > atr_p75],
        "trend_up(ema20>1)": base[ema[base] > 1.0],
        "trend_down(ema20<1)": base[ema[base] < 1.0],
    }

    report = {"parquet": str(PARQUET), "rows": n, "fees": ROUND_TRIP_FEES,
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "results": []}

    print(f"# Premier passage CONDITIONNÉ — {n} barres 5m, fees={ROUND_TRIP_FEES:.3%}\n")
    print("| filtre | n_entrées | H | meilleure (sl,tp) | EV net | P(TP) | P(SL) |")
    print("|---|---|---|---|---|---|---|")
    for fname, idx in filters.items():
        for H in HORIZONS:
            best = None
            for sl in SL_GRID:
                for tp in TP_GRID:
                    r = first_passage_idx(idx, high, low, close, sl, tp, H)
                    if r is None:
                        continue
                    ev, p_tp, p_sl, ni = r
                    if best is None or ev > best[0]:
                        best = (ev, sl, tp, p_tp, p_sl, ni)
            if best is None:
                print(f"| {fname} | <200 | {H} | — | — | — | — |")
                continue
            ev, sl, tp, p_tp, p_sl, ni = best
            report["results"].append(dict(filter=fname, H=H, n=ni, sl=sl,
                                          tp=tp, ev=round(ev, 5),
                                          p_tp=round(p_tp, 4), p_sl=round(p_sl, 4)))
            mark = " **+EV**" if ev > 0 else ""
            print(f"| {fname} | {ni} | {H} | ({sl:.4f},{tp:.4f}) "
                  f"| {ev:+.4f}{mark} | {p_tp:.3f} | {p_sl:.3f} |")

    # sensibilité aux frais sur la meilleure config trouvée
    if report["results"]:
        top = max(report["results"], key=lambda r: r["ev"])
        print(f"\nMeilleure config : {top['filter']} H={top['H']} "
              f"(sl={top['sl']}, tp={top['tp']}) EV={top['ev']:+.4f}")
        print(f"Point mort de frais pour cette config : EV_brut = "
              f"{top['ev'] + ROUND_TRIP_FEES:+.4f} → frais max pour EV net > 0 : "
              f"{(top['ev'] + ROUND_TRIP_FEES):.4%} (actuel {ROUND_TRIP_FEES:.4%})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"sltp_conditioned_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nRapport JSON : {out}")


if __name__ == "__main__":
    main()
