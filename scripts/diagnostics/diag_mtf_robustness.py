#!/usr/bin/env python3
"""diag_mtf_robustness.py — round 3 : la edge MTF survit-elle au test de robustesse ?

Rounds 1-2 : confluence MTF > aveugle (delta positif fréquent) mais aucune
cellule (règle, k_sl, k_tp, H) ne dépasse la frontière de frais 0.40%, et la
cellule gagnante change d'un split à l'autre -> soupçon de bruit.

Round 3 tranche statistiquement :
  1. FILTRE DE CONSISTANCE : une cellule n'est retenue que si EV_net > 0 sur
     train ET val ET test (même asset), n >= 30 partout.
  2. BLOCK BOOTSTRAP : sur la meilleure cellule de chaque (asset, règle),
     IC 95% de EV_net par rééchantillonnage en blocs de 5 trades (préserve
     l'autocorrélation résiduelle). 2000 réplications.
  3. RATIO DE SÉLECTIVITÉ : EV_net par trade x fréquence = P&L quotidien
     espéré (à ~288 barres 5m/jour) -> est-ce économiquement pertinent même
     si > 0 ?

Si aucune cellule ne passe (1)+(2), conclusion honnête : la confluence MTF est
un biais prédictif RÉEL mais faible (quelques bps) -> à intégrer comme PRIOR
de policy / feature pour l'agent RL (réduction de l'espace d'action, reward
shaping par qualité de chemin), PAS comme stratégie autonome à 0.40% de frais.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_mtf_confluence import load_aligned, ASSETS, SPLITS, OUT_DIR
from diag_mtf_atr_geometry import build_masks, select_nonoverlapping, K_SL, K_TP

FEES_RT = 0.004
MIN_TRADES = 30
H_LIST = [40, 100, 288]
N_BOOT = 2000
BLOCK = 5
RNG = np.random.default_rng(42)

RULES = ["A2_trend_cross", "B2_pullback_cross", "D2_strict_cross"]


def trade_returns(high, low, close, entries, H, atr_entry, k_sl, k_tp):
    """Retourne le vecteur des P&L par trade pour un (k_sl, k_tp) donné."""
    entry = close[entries]
    sl = k_sl * atr_entry
    tp = k_tp * atr_entry
    res = np.zeros(len(entries))
    done = np.zeros(len(entries), dtype=bool)
    for h in range(1, H + 1):
        hh = high[entries + h] / entry - 1.0
        ll = low[entries + h] / entry - 1.0
        both = (~done) & (hh >= tp) & (ll <= -sl)
        res[both] = -sl[both]
        done |= both
        tpo = (~done) & (hh >= tp)
        res[tpo] = tp[tpo]
        done |= tpo
        slo = (~done) & (ll <= -sl)
        res[slo] = -sl[slo]
        done |= slo
    md = ~done
    res[md] = close[entries + H][md] / entry[md] - 1.0
    return res - FEES_RT


def block_bootstrap_ci(pnl, n_boot=N_BOOT, block=BLOCK):
    n = len(pnl)
    if n < block * 3:
        return None
    idx0 = np.arange(0, n - block + 1)
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = RNG.choice(idx0, size=int(np.ceil(n / block)), replace=True)
        sample = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        means[b] = pnl[sample].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float(means.mean())


def main():
    report = {}
    for asset in ASSETS:
        data = {}
        for split in SPLITS:
            df = load_aligned(asset, split)
            if df is None or len(df) < 400:
                continue
            data[split] = df
        if len(data) < 3:
            continue
        masks = {sp: build_masks(df) for sp, df in data.items()}
        asset_res = {}
        for rule in RULES:
            cells = {}
            for H in H_LIST:
                for k_sl in K_SL:
                    for k_tp in K_TP:
                        per_split = {}
                        ok = True
                        for sp in SPLITS:
                            df = data[sp]
                            entries = select_nonoverlapping(masks[sp][rule], len(df), H)
                            if len(entries) < MIN_TRADES:
                                ok = False
                                break
                            pnl = trade_returns(
                                df["high"].to_numpy(np.float64),
                                df["low"].to_numpy(np.float64),
                                df["close"].to_numpy(np.float64),
                                entries, H,
                                df["atr_pct"].to_numpy(np.float64)[entries],
                                k_sl, k_tp)
                            per_split[sp] = pnl
                        if not ok:
                            continue
                        evs = {sp: float(p.mean()) for sp, p in per_split.items()}
                        if all(v > 0 for v in evs.values()):
                            cells[f"H{H}|{k_sl}x{k_tp}"] = {
                                "ev_net": {k: round(v, 5) for k, v in evs.items()},
                                "n": {sp: len(p) for sp, p in per_split.items()},
                                "min_ev": round(min(evs.values()), 5),
                                "pnl_test": per_split["test"],
                                "freq_per_day_test": len(per_split["test"]) / (len(data["test"]) / 288.0),
                            }
            if cells:
                # meilleure cellule = max min_ev (robuste au pire split)
                best_key = max(cells, key=lambda k: cells[k]["min_ev"])
                best = cells[best_key]
                ci = block_bootstrap_ci(best.pop("pnl_test"))
                best["bootstrap_ci95_test"] = [round(x, 5) for x in ci] if ci else None
                best["ci_excludes_zero"] = bool(ci and ci[0] > 0)
                best["daily_pnl_expectancy"] = round(best["min_ev"] * best["freq_per_day_test"], 6)
                asset_res[rule] = {"best_cell": best_key, **best,
                                   "n_consistent_cells": len(cells)}
            else:
                asset_res[rule] = {"n_consistent_cells": 0}
        report[asset] = asset_res
        print(f"[done] {asset}", flush=True)

    print("\n## ROBUSTESSE — cellules EV_net>0 sur train ET val ET test (n>=30)")
    verdict_any = False
    for asset, rules in report.items():
        print(f"\n### {asset}")
        for rule, r in rules.items():
            if r["n_consistent_cells"] == 0:
                print(f"  {rule:<20} : aucune cellule cohérente")
                continue
            ci = r["bootstrap_ci95_test"]
            sig = "SIGNIFICATIF (IC95% exclut 0)" if r["ci_excludes_zero"] else "non significatif (IC95% inclut 0)"
            print(f"  {rule:<20} : {r['n_consistent_cells']} cellule(s), meilleure {r['best_cell']} "
                  f"minEV={r['min_ev']:+.4%} IC95_test={ci} -> {sig}, "
                  f"~{r['freq_per_day_test']:.2f} trades/j, P&L/j espéré={r['daily_pnl_expectancy']:+.4%}")
            verdict_any = verdict_any or r["ci_excludes_zero"]

    print("\n## VERDICT FINAL")
    if verdict_any:
        print("  Au moins une cellule MTF est EV>0 cohérente ET significative -> à intégrer comme stratégie.")
    else:
        print("  Aucune cellule n'est à la fois cohérente (3 splits) et significative (bootstrap).")
        print("  -> La confluence MTF est un biais prédictif réel mais trop faible pour 0.40% de frais.")
        print("  -> Intégration recommandée : PRIOR de policy RL (masque d'actions + reward shaping")
        print("     par qualité de chemin MFE/MAE/efficacité), PAS stratégie autonome à ces frais.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"mtf_robustness_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"generated_utc": datetime.now(timezone.utc).isoformat(),
                               "params": {"fees_rt": FEES_RT, "min_trades": MIN_TRADES,
                                          "n_boot": N_BOOT, "block": BLOCK},
                               "report": report}, indent=2))
    print(f"\nRapport JSON : {out}")


if __name__ == "__main__":
    main()
