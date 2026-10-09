#!/usr/bin/env python3
"""diag_vol_regime_filter.py — voie #2 du rapport MFE : filtre de regime de volatilite.

Pourquoi cette sonde existe
---------------------------
RAPPORT_MFE_EXCURSION.md, voies restantes :

  2. "Horizon/conditionnement : l'ecart MTF_ALIGNE vs CONTRADICTOIRE existe
      (+3-10 pts de P(>3x couts)) mais est trop faible seul ; a combiner avec
      un FILTRE DE REGIME (VOLATILITE) plutot qu'avec la direction seule."

La voie #1 (couts) est close (RAPPORT_FEE_SENSITIVITY) : sur test, l'EV BRUTE
de B1/B2/B3 est <= +0,04 %, donc meme a frais nuls rien ne gagne. Il reste a
tester si un CONDITIONNEMENT des entrees par regime de volatilite retourne le
signe sur val ET test. C'est la derniere sous-hypothese pre-listee.

Discipline anti-dredging (pre-enregistre)
-----------------------------------------
- Seuils de terciles d'ATR (atr_pct a l'entree) calcules sur le split TRAIN
  UNIQUEMENT, par actif, puis GELES. Val et test sont classes avec les seuils
  du train — aucune fuite, aucun ajustement sur les splits d'evaluation.
- Regles de sortie : B2 (SL 0,5xATR + invalidation 4h) et B3 (SL 3xATR +
  invalidation 4h), H_max 7j — les seules qui avaient un edge sur train.
  B1 est exclue (deja negative partout, y compris a cout nul).
- Entrees identiques aux probes precedents : cross-up MACD 5m + MTF_ALIGNE,
  stride 40, non superposees.
- Mesure : EV brute (cout 0) ET EV a 0,10 % RT, block bootstrap (n_boot=2000,
  block=5, seed fixe), par tercile {LOW, MID, HIGH}.
- Criteres de verdict : DEBLOQUE si un couple (regle x tercile) a, sur val ET
  test, a 0,10 % RT : EV > 0 ET IC95 bas > 0. Exige sur les DEUX actifs pour
  etre considere robuste (cross-asset), sinon signale comme fragile.
- Multiple-comparaison : 2 regles x 3 terciles = 6 cellules par actif/split ;
  un PASS isole a p~0,05 sur 12 cellules d'evaluation est attendu ~1 fois par
  chance. Le verdict exige la replication val ET test (et idealement
  cross-asset) precisement pour ce piege.

Lecture seule. Univers launcher garanti par _asset_guard.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_mtf_confluence import load_aligned, OUT_DIR  # noqa: E402
from diag_mtf_confluence_v2 import (  # noqa: E402
    cross_up_events, select_nonoverlapping, N_BOOT, BLOCK)
from diag_mfe_excursion import (  # noqa: E402
    sortie_invalidation, get_masks, K_SL_B2, K_SL_B3, STRIDE, H_MAX)
from _asset_guard import get_launcher_assets  # noqa: E402

ASSETS = list(get_launcher_assets())
SPLITS = ["train", "val", "test"]
COST_EVAL = 0.001          # 0,10 % RT — borne de la condition de deblocage
RNG = np.random.default_rng(20260911)
TERCILES = ("LOW", "MID", "HIGH")


def boot_ci(pnl, n_boot=N_BOOT, block=BLOCK):
    n = len(pnl)
    if n < 5:
        return (float(np.mean(pnl)) if n else float("nan"),) + (float("nan"),) * 3
    n_blocks = int(np.ceil(n / block))
    starts = RNG.integers(0, max(1, n - block + 1), size=(n_boot, n_blocks))
    ar = np.arange(block)[None, :]
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.clip((starts[b][:, None] + ar).ravel()[:n], 0, n - 1)
        means[b] = pnl[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    p_two = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(pnl.mean()), float(lo), float(hi), float(min(1.0, p_two))


def entries_and_arrays(df):
    n = len(df)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    regime4h = df["ema_100_ratio_4h"].to_numpy(np.float64)
    atr = df["atr_pct"].to_numpy(np.float64)
    events = cross_up_events(df)
    ent = select_nonoverlapping(events, n, STRIDE)
    ent = ent[ent + H_MAX < n]
    aligne, _ = get_masks(df)
    ent = ent[aligne[ent]]
    return ent, (high, low, close, regime4h, atr)


def eval_rule(high, low, close, regime4h, atr, ent, k_sl):
    if len(ent) < 30:
        return None
    pnl, ht, reason = sortie_invalidation(
        high, low, close, regime4h, ent, H_MAX, atr[ent], k_sl)
    return pnl


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[params] cost_eval={COST_EVAL:.4%}  n_boot={N_BOOT}  block={BLOCK}"
          f"  stride={STRIDE}  H_max={H_MAX}", flush=True)

    results = []
    for asset in ASSETS:
        # ---- 1) charger les 3 splits --------------------------------------
        data = {}
        for split in SPLITS:
            df = load_aligned(asset, split)
            if df is None:
                continue
            df = df.dropna(subset=["ema_100_ratio_4h", "ema_50_ratio_1h",
                                   "macdh_12_26_9", "atr_pct"])
            data[split] = df
            print(f"[run] {asset}/{split}: {len(df)} barres 5m", flush=True)
        if "train" not in data:
            continue

        # ---- 2) seuils de terciles ATR geles sur TRAIN ---------------------
        ent_tr, arr_tr = entries_and_arrays(data["train"])
        atr_tr = arr_tr[4][ent_tr]
        q33, q66 = np.quantile(atr_tr, [1 / 3, 2 / 3])
        print(f"[frozen] {asset} terciles ATR (train) : "
              f"LOW<{q33:.5f}  MID  HIGH>{q66:.5f}", flush=True)

        def tercile_of(atr_vals):
            return np.where(atr_vals < q33, 0,
                            np.where(atr_vals > q66, 2, 1))

        # ---- 3) evaluation par split / regle / tercile ---------------------
        for split, df in data.items():
            ent, (high, low, close, regime4h, atr) = entries_and_arrays(df)
            cell = {"asset": asset, "split": split,
                    "rows_5m": int(len(df)), "n_entries": int(len(ent)),
                    "atr_tercile_edges_train": [float(q33), float(q66)],
                    "cells": {}}
            if len(ent) < 30:
                cell["skipped"] = "n_entries<30"
                results.append(cell)
                continue
            t_of_entry = tercile_of(atr[ent])
            for k_sl, rname in ((K_SL_B2, "B2"), (K_SL_B3, "B3")):
                pnl_all = eval_rule(high, low, close, regime4h, atr, ent, k_sl)
                if pnl_all is None:
                    continue
                for t_idx, t_name in enumerate(TERCILES):
                    mask = t_of_entry == t_idx
                    pnl = pnl_all[mask]
                    if len(pnl) < 30:
                        cell["cells"][f"{rname}_{t_name}"] = {
                            "n": int(len(pnl)), "skipped": "n<30"}
                        continue
                    ev0, lo0, hi0, p0 = boot_ci(pnl)
                    ev_c = ev0 - COST_EVAL
                    lo_c = lo0 - COST_EVAL
                    hi_c = hi0 - COST_EVAL
                    cell["cells"][f"{rname}_{t_name}"] = {
                        "n": int(len(pnl)),
                        "ev_brut": ev0, "ci95_brut": [lo0, hi0],
                        "ev_at_unlock_cost": ev_c,
                        "ci95_at_unlock_cost": [lo_c, hi_c],
                        "p_boot_at_unlock_cost": p0,
                        "wr_brut": float((pnl > 0).mean())}
                    print(f"  {split:<5} {rname}_{t_name:<5} n={len(pnl):>5}  "
                          f"EVbrut={ev0:+.4%}  "
                          f"EV@0.10%RT={ev_c:+.4%} [{lo_c:+.4%},{hi_c:+.4%}]",
                          flush=True)
            results.append(cell)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = {"generated_utc": stamp,
           "params": {"cost_eval_rt": COST_EVAL, "n_boot": N_BOOT,
                      "block": BLOCK, "stride": STRIDE, "H_max": H_MAX,
                      "tercile_edges": "frozen on train per asset",
                      "entries": "cross_up MACD 5m + MTF_ALIGNE, stride 40",
                      "anti_dredging": "verdict exige val ET test (+ cross-asset)"},
           "results": results}
    path = OUT_DIR / f"vol_regime_filter_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nRapport JSON : {path}")

    # ------------------------------------------------- verdict pre-enregistre
    print("\n===== VERDICT — filtre volatilite (EV>0 ET IC95bas>0 a 0,10% RT, "
          "val ET test) =====")
    passes = []
    for cellkey in ("B2_LOW", "B2_MID", "B2_HIGH", "B3_LOW", "B3_MID",
                    "B3_HIGH"):
        per_asset = {}
        for asset in ASSETS:
            row = {}
            for split in ("val", "test"):
                c = next((r for r in results
                          if r.get("asset") == asset
                          and r.get("split") == split), None)
                cc = (c or {}).get("cells", {}).get(cellkey)
                if cc is None or "ev_at_unlock_cost" not in cc:
                    row[split] = None
                    continue
                ok = (cc["ev_at_unlock_cost"] > 0
                      and cc["ci95_at_unlock_cost"][0] > 0)
                row[split] = {"ev": cc["ev_at_unlock_cost"],
                              "lo": cc["ci95_at_unlock_cost"][0],
                              "pass": bool(ok)}
            per_asset[asset] = row
        both_assets = all(
            r.get("val") and r.get("test")
            and r["val"]["pass"] and r["test"]["pass"]
            for r in per_asset.values())
        one_asset = any(
            r.get("val") and r.get("test")
            and r["val"]["pass"] and r["test"]["pass"]
            for r in per_asset.values())
        tag = "ROBUSTE" if both_assets else ("fragile" if one_asset else "non")
        print(f"  {cellkey:<9} {tag:<8} "
              + "  ".join(
                  f"{a.split('U')[0]}: "
                  + "/".join(
                      "n/a" if r[s] is None
                      else f"{r[s]['ev']:+.3%}{'*' if r[s]['pass'] else ''}"
                      for s in ("val", "test"))
                  for a, r in per_asset.items()))
        if both_assets:
            passes.append(cellkey)
        elif one_asset:
            passes.append(cellkey + " (fragile, 1 actif)")
    print("\n==> " + (f"DEBLOQUAGE : {passes}" if passes else
                       "NO-GO MAINTENU : aucun couple regle x tercile ne "
                       "replique val ET test. Voie #2 (conditionnement "
                       "volatilite) close."))


if __name__ == "__main__":
    main()
