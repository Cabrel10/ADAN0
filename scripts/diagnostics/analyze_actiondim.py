#!/usr/bin/env python3
"""ANALYZE-ACTIONDIM — verdict Cas A vs Cas B depuis le CSV de l'ActionDimMonitor.

Lit le CSV produit par ActionDimMonitor pendant le run V2 et tranche la question
centrale (analyse externe utilisateur) : μ(size) remonte-t-il ?

  Cas A (RÉVEIL)   : μ(size) remonte significativement (ex. -7.2 -> > -3.0)
                     ET std post-tanh de size augmente -> PPO réapprend la tête.
                     => PAS besoin d'ActionSaturationGuard.
  Cas B (BLOCAGE)  : μ(size) reste ≈ inchangé (|Δμ| faible) malgré σ élevé
                     -> le problème n'est plus l'exploration mais le reward /
                        credit-assignment. => envisager le guard.

Usage : python scripts/analyze_actiondim.py logs/training/actiondim_v2_*.csv
"""
from __future__ import annotations

import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

HEADS = ["direction", "size", "tf", "sl", "tp"]
# seuils de décision
MU_WAKE_DELTA = 2.0     # remontée de μ(size) jugée significative
MU_WAKE_TARGET = -3.0   # μ(size) au-dessus = sorti de la crevasse profonde
STD_WAKE = 0.05         # std post-tanh de size au-dessus = action devient variable


def _resolve(path_arg: str) -> str:
    if "*" in path_arg:
        cands = sorted(glob.glob(path_arg))
        if not cands:
            print(f"Aucun CSV ne correspond à {path_arg}")
            sys.exit(1)
        return cands[-1]  # le plus récent
    return path_arg


def main():
    if len(sys.argv) < 2:
        # par défaut : dernier CSV V2
        default = "logs/training/actiondim_v2_*.csv"
        cands = sorted(glob.glob(default))
        if not cands:
            print(f"Usage: {sys.argv[0]} <actiondim.csv>")
            sys.exit(1)
        path = cands[-1]
    else:
        path = _resolve(sys.argv[1])

    df = pd.read_csv(path)
    if df.empty:
        print(f"CSV vide: {path}")
        sys.exit(1)
    print("=" * 78)
    print(f"  ANALYSE ACTIONDIM — {Path(path).name}")
    print("=" * 78)
    print(f"Fenêtres journalisées : {len(df)}  "
          f"(step {int(df['step'].iloc[0])} → {int(df['step'].iloc[-1])})")

    first, last = df.iloc[0], df.iloc[-1]
    print(f"\n{'tête':<10}{'μ début':>10}{'μ fin':>10}{'Δμ':>9}"
          f"{'σ̂ début':>10}{'σ̂ fin':>10}{'sat fin':>9}")
    for h in HEADS:
        mu_c = f"{h}_mu_mean"
        std_c = f"{h}_post_std"
        sat_c = f"{h}_sat_frac"
        if mu_c not in df.columns:
            continue
        mu0 = float(first[mu_c]) if pd.notna(first[mu_c]) else float("nan")
        mu1 = float(last[mu_c]) if pd.notna(last[mu_c]) else float("nan")
        s0 = float(first[std_c]); s1 = float(last[std_c])
        sat1 = float(last[sat_c]) if sat_c in df.columns else float("nan")
        print(f"{h:<10}{mu0:>10.3f}{mu1:>10.3f}{mu1-mu0:>9.3f}"
              f"{s0:>10.4f}{s1:>10.4f}{sat1:>9.3f}")

    # verdict SIZE
    print("\n" + "=" * 78)
    print("  VERDICT — TÊTE SIZE")
    print("=" * 78)
    if "size_mu_mean" not in df.columns or df["size_mu_mean"].isna().all():
        print("  (μ pré-tanh non disponible dans ce CSV — relancer avec "
              "ADAN_ACTIONDIM_BATCH>0)")
        return
    mu_series = df["size_mu_mean"].astype(float)
    mu0 = float(mu_series.iloc[0]); mu1 = float(mu_series.iloc[-1])
    dmu = mu1 - mu0
    std1 = float(df["size_post_std"].iloc[-1])
    mu_max = float(mu_series.max())
    print(f"  μ(size) : début={mu0:+.3f}  fin={mu1:+.3f}  max={mu_max:+.3f}  "
          f"Δ={dmu:+.3f}")
    print(f"  std post-tanh(size) fin = {std1:.4f}")

    woke = (dmu >= MU_WAKE_DELTA and mu1 >= MU_WAKE_TARGET) or std1 >= STD_WAKE
    if woke:
        print("\n  ✅ CAS A — RÉVEIL : μ(size) remonte / l'action redevient variable.")
        print("     PPO réapprend la tête avec log_std_init+ent_coef seuls.")
        print("     => PAS besoin d'ActionSaturationGuard. Étape suivante :")
        print("        refaire C4/H4 avec MFE/MAE réels -> recalibrer A5 -> bridge.")
    else:
        print("\n  ⛔ CAS B — BLOCAGE : μ(size) stagne malgré l'exploration.")
        print(f"     Δμ={dmu:+.2f} (<{MU_WAKE_DELTA}) et std={std1:.3f} (<{STD_WAKE}).")
        print("     Le problème n'est PLUS l'exploration mais reward/credit-assignment.")
        print("     => intégrer ActionSaturationGuard (déjà prêt, en réserve) au")
        print("        PROCHAIN run, puis ré-observer cette même courbe.")
    print(f"\nNote : seuils μ_wake_delta={MU_WAKE_DELTA}, μ_target={MU_WAKE_TARGET}, "
          f"std_wake={STD_WAKE}.")


if __name__ == "__main__":
    main()
