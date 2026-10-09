#!/usr/bin/env python3
"""diag_mtf_atr_geometry.py — round 2 : géométrie SL/TP adaptative ATR + trigger sélectif.

Round 1 (diag_mtf_confluence) a établi :
  - la confluence MTF crée un edge CONDITIONNEL réel (filtré > aveugle sur
    train/val/test, non superposé) -> la critique mono-TF était fondée ;
  - mais l'amplitude reste << frontière de frais 0.40%.

Causes identifiées (négatif -> pourquoi -> correction) :
  C1. trigger en ÉTAT (macdh>0) présent ~25-50% du temps -> sélectivité diluée ;
      correction : trigger en ÉVÉNEMENT (croisement haussier macdh) = rare.
  C2. grille SL/TP FIXE indépendante de la volatilité -> "SL = f(ATR, structure
      MTF)" ; correction : SL = k_sl x ATR%_5m, TP = k_tp x ATR%_5m à l'entrée.
  C3. H=40 trop court : MFE médian 0.5-1.2% n'a pas le temps de se convertir en
      TP ; correction : H in {40, 100, 288} (288 barres 5m = 24h).
  C4. schéma non-superposé événementiel réaliste : on prend un signal, on saute
      H barres, on continue (au lieu d'échantillonner le masque à pas fixe).

Sorties : P(TP)/P(SL)/P(MD), EV net par (k_sl,k_tp) = surface EV(SL,TP | état
MTF), MFE/MAE, efficacité du chemin, comparaison filtré vs aveugle, train/val/test.
Lecture seule. JSON dans logs/validation/.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_mtf_confluence import load_aligned, DATA, OUT_DIR, ASSETS, SPLITS

FEES_RT = 0.004
MIN_TRADES = 30

K_SL = [0.5, 0.75, 1.0, 1.5, 2.0]      # SL = k_sl x ATR%
K_TP = [1.0, 1.5, 2.0, 3.0, 4.0]       # TP = k_tp x ATR%
HORIZONS = [40, 100, 288]              # barres 5m


def build_masks(df):
    """Masques booléens (numpy) par règle, trigger en ÉVÉNEMENT sur le 5m."""
    f = {c: df[c].to_numpy(np.float64) for c in df.columns}
    macdh = f["macdh_12_26_9"]
    cross_up = np.zeros(len(df), dtype=bool)
    cross_up[1:] = (macdh[1:] > 0) & (macdh[:-1] <= 0)

    up4 = f["ema_100_ratio_4h"] > 1.0
    up1 = f["ema_50_ratio_1h"] > 1.0
    rsi1 = f["rsi_21_1h"]
    rsi5 = f["rsi_14"]
    adx5 = f["adx_14"]
    ema20 = f["ema_20_ratio"]

    return {
        # A2 : tendance 4h+1h + croisement haussier 5m (événement)
        "A2_trend_cross": up4 & up1 & cross_up,
        # B2 : pullback en tendance 4h, RSI1h replié, prix sous/près EMA20_5m, reprise
        "B2_pullback_cross": up4 & (rsi1 >= 35) & (rsi1 <= 50) & cross_up & (ema20 <= 1.002),
        # D2 : confluence stricte + événement
        "D2_strict_cross": up4 & up1 & (rsi1 >= 40) & (rsi1 <= 55)
                           & (rsi5 > 50) & (adx5 > 20) & cross_up,
        # E2 : contre-tendance (contrôle négatif attendu)
        "E2_counter_cross": (~up4) & cross_up,
    }


def select_nonoverlapping(mask, n, horizon):
    """Marche événementielle : signal pris -> on saute `horizon` barres."""
    entries = []
    next_free = 0
    for i in range(0, n - horizon - 1):
        if i >= next_free and mask[i]:
            entries.append(i)
            next_free = i + horizon
    return np.array(entries, dtype=np.int64)


def simulate_atr(high, low, close, entries, horizon, atr_entry):
    """Premier passage avec SL/TP par-trade = k x ATR% entrée. Retourne la surface EV."""
    entry = close[entries]
    mfe = np.full(len(entries), -np.inf)
    mae = np.full(len(entries), np.inf)
    path_len = np.zeros(len(entries))
    prev = entry.copy()
    for h in range(1, horizon + 1):
        mfe = np.maximum(mfe, high[entries + h] / entry - 1.0)
        mae = np.minimum(mae, low[entries + h] / entry - 1.0)
        cur = close[entries + h]
        path_len += np.abs(cur / prev - 1.0)
        prev = cur
    final = close[entries + horizon] / entry - 1.0
    eff = np.where(path_len > 1e-12, np.abs(final) / path_len, 0.0)

    surface = {}
    best = None
    for k_sl in K_SL:
        for k_tp in K_TP:
            sl = k_sl * atr_entry
            tp = k_tp * atr_entry
            hit_tp = np.zeros(len(entries), dtype=bool)
            hit_sl = np.zeros(len(entries), dtype=bool)
            for h in range(1, horizon + 1):
                hh = high[entries + h] / entry - 1.0
                ll = low[entries + h] / entry - 1.0
                both = (~hit_tp) & (~hit_sl) & (hh >= tp) & (ll <= -sl)
                hit_sl |= both
                hit_tp |= (~hit_tp) & (~hit_sl) & (hh >= tp)
                hit_sl |= (~hit_tp) & (~hit_sl) & (ll <= -sl)
            md = (~hit_tp) & (~hit_sl)
            md_ret = final[md]
            mean_md = float(md_ret.mean()) if md.any() else 0.0
            ev_gross = float(hit_tp.mean() * tp.mean()) if False else float(
                (hit_tp * tp).mean() - (hit_sl * sl).mean() + md.mean() * mean_md)
            ev_net = ev_gross - FEES_RT
            cell = {"k_sl": k_sl, "k_tp": k_tp,
                    "sl_mean": round(float(sl.mean()), 5),
                    "tp_mean": round(float(tp.mean()), 5),
                    "ev_gross": round(ev_gross, 5), "ev_net": round(ev_net, 5),
                    "p_tp": round(float(hit_tp.mean()), 4),
                    "p_sl": round(float(hit_sl.mean()), 4),
                    "p_md": round(float(md.mean()), 4)}
            surface[f"{k_sl}x{k_tp}"] = cell
            if best is None or ev_net > best["ev_net"]:
                best = cell
    return {
        "n": len(entries),
        "mfe_median": round(float(np.median(mfe)), 5),
        "mae_median": round(float(np.median(mae)), 5),
        "atr_entry_median": round(float(np.median(atr_entry)), 5),
        "path_efficiency_mean": round(float(eff.mean()), 4),
        "best_net": best,
        "n_positive_cells": sum(1 for c in surface.values() if c["ev_net"] > 0),
        "surface": surface,
    }


def run():
    report = []
    for split in SPLITS:
        for asset in ASSETS:
            df = load_aligned(asset, split)
            if df is None or len(df) < 400:
                continue
            high = df["high"].to_numpy(np.float64)
            low = df["low"].to_numpy(np.float64)
            close = df["close"].to_numpy(np.float64)
            atr = df["atr_pct"].to_numpy(np.float64)
            masks = build_masks(df)
            masks["BLIND"] = np.ones(len(df), dtype=bool)
            n = len(df)

            for H in HORIZONS:
                if n < H + 400:
                    continue
                print(f"[run] {split}/{asset} H={H}", flush=True)
                row = {"asset": asset, "split": split, "H": H, "rules": {}}
                for name, mask in masks.items():
                    entries = select_nonoverlapping(mask, n, H)
                    if len(entries) < MIN_TRADES:
                        row["rules"][name] = {"n": int(len(entries)), "skipped": True}
                        continue
                    row["rules"][name] = simulate_atr(
                        high, low, close, entries, H, atr[entries])
                report.append(row)
    return report


def print_summary(report):
    for row in report:
        print(f"\n## {row['asset']} [{row['split']}] H={row['H']}")
        print(f"{'règle':<20} {'n':>6} {'ATR_med':>8} {'MFE_med':>8} {'MAE_med':>8} {'eff':>6} {'bestEVnet':>10} {'(ksl,ktp)':>10} {'P(TP)':>6} {'P(SL)':>6} {'#cell>0':>7}")
        for name, r in row["rules"].items():
            if r.get("skipped"):
                print(f"{name:<20} n={r['n']:>5}  (ignoré)")
                continue
            b = r["best_net"]
            flag = " *** EDGE NET POSITIF ***" if b["ev_net"] > 0 else ""
            print(f"{name:<20} {r['n']:>6} {r['atr_entry_median']:>8.3%} {r['mfe_median']:>+8.3%} "
                  f"{r['mae_median']:>+8.3%} {r['path_efficiency_mean']:>6.3f} "
                  f"{b['ev_net']:>+10.4%} ({b['k_sl']},{b['k_tp']}) "
                  f"{b['p_tp']:>6.2%} {b['p_sl']:>6.2%} {r['n_positive_cells']:>7}{flag}")


def main():
    report = run()
    print_summary(report)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"mtf_atr_geometry_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"generated_utc": datetime.now(timezone.utc).isoformat(),
                               "params": {"fees_rt": FEES_RT, "min_trades": MIN_TRADES,
                                          "k_sl": K_SL, "k_tp": K_TP, "horizons": HORIZONS},
                               "report": report}, indent=2))
    print(f"\nRapport JSON : {out}")

    print("\n## VERDICT round 2 (delta EV net filtré - aveugle, par split, meilleur H)")
    index = {}
    for row in report:
        index.setdefault((row["asset"], row["split"]), {})[row["H"]] = row["rules"]
    for (asset, split), by_h in sorted(index.items()):
        best_h = max(by_h, key=lambda h: by_h[h].get("BLIND", {}).get("n", 0))
        rules = by_h[best_h]
        blind = rules.get("BLIND", {})
        if "best_net" not in blind:
            continue
        for name in ("A2_trend_cross", "B2_pullback_cross", "D2_strict_cross", "E2_counter_cross"):
            r = rules.get(name, {})
            if "best_net" in r:
                d = r["best_net"]["ev_net"] - blind["best_net"]["ev_net"]
                print(f"  {asset:<20} {split:<6} H={best_h:<4} {name:<20} "
                      f"EV={r['best_net']['ev_net']:+.4%}  delta_vs_blind={d:+.4%}")


if __name__ == "__main__":
    main()
