#!/usr/bin/env python3
"""PRE-TANH AUDIT — μ et σ AVANT tanh, par tête d'action (point de prudence 1).

MOTIVATION (revue utilisateur juin 2026)
-----------------------------------------
inspect_action_heads.py a prouvé : tête SIZE GELÉE en SORTIE (post-tanh : 100 %
à -1.0, std=0). Mais cela ne dit PAS d'où vient le blocage. Deux causes très
différentes mènent au même symptôme post-tanh :

  (μ) moyenne pré-tanh très négative  : μ(size) = -8  → tanh(-8) ≈ -1 même avec
       une exploration σ correcte. Le problème est le BIAIS/POIDS de la tête.
       → un simple reset de log_std NE SUFFIRA PAS ; il faut ré-apprendre μ
         (ent_coef + gradient de reward sur des positions enfin variées).

  (σ) exploration pré-tanh trop faible : μ(size) ≈ -0.5 mais σ ≈ 0.01 → l'agent
       ne sort jamais explorer. Le problème est l'EXPLORATION.
       → log_std_init plus haut / ent_coef plus fort SUFFIT a priori.

Ce script lit, depuis la distribution gSDE de la politique (squash_output=False),
les vraies grandeurs PRÉ-tanh :
  * pre_tanh_mean[j] = action_net(latent_pi)[j]            (= dist.distribution.mean)
  * pre_tanh_std[j]  = sqrt(latent_sde² @ std² + eps)[j]   (= dist.distribution.scale)
puis applique tanh pour relier au post-tanh observé. Aucune écriture, read-only.

DIAGNOSTIC PAR TÊTE
-------------------
  |μ| > ~2.5 (tanh saturé) ET σ petit            → cause = MOYENNE (μ)
  |μ| modéré MAIS σ < ~0.05                       → cause = EXPLORATION (σ)
  |μ| > ~2.5 ET σ raisonnable                     → MIXTE (μ domine quand même)
  sinon                                           → tête saine
"""
from __future__ import annotations

import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(str(ROOT))

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

NAMES = ["direction", "size", "tf", "sl", "tp"]
# seuils heuristiques (tanh' < 0.1 dès |μ|>~1.8 ; tanh(2.5)=0.987)
MU_SAT = 2.5      # au-delà, tanh(μ) est déjà collé à ±1
SIG_LOW = 0.05    # en dessous, l'exploration pré-tanh est négligeable


def find_checkpoint() -> str:
    cands = sorted(glob.glob("checkpoints/*FIXED*.zip")) or \
        sorted(glob.glob("checkpoints/*.zip"))
    for c in cands:
        if "500k_FIXED" in c:
            return c
    return cands[-1]


def build_real_observations(n: int, seed: int = 0) -> list[dict]:
    """Identique à inspect_action_heads : obs réelles depuis Parquet val."""
    try:
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        from adan_trading_bot.trading.live_state_builder import (
            TRAIN_COLUMNS, OBS_WINDOW,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [obs réelles] import échoué: {e}")
        return []
    val_dir = ROOT / "data" / "processed" / "indicators" / "val" / "BTCUSDT"
    data_dict = {}
    for tf in ["5m", "1h", "4h"]:
        p = val_dir / f"{tf}.parquet"
        if p.exists():
            data_dict[tf] = pd.read_parquet(p)
    if len(data_dict) < 3:
        print(f"  [obs réelles] Parquet manquant dans {val_dir}")
        return []
    try:
        sb = StateBuilder(features_config=TRAIN_COLUMNS,
                          window_sizes={tf: OBS_WINDOW for tf in ["5m", "1h", "4h"]},
                          include_portfolio_state=True, normalize=True)
        sb.fit_scalers({"BTCUSDT": data_dict})
        sb.scalers_loaded_from_training = True
    except Exception as e:  # noqa: BLE001
        print(f"  [obs réelles] init échoué: {e}")
        return []
    nested = {"BTCUSDT": data_dict}
    n5 = len(data_dict["5m"]); lo, hi = 300, n5 - 2
    if hi <= lo:
        return []
    rng = np.random.default_rng(seed)
    idxs = rng.integers(lo, hi, size=min(n, hi - lo))
    out = []
    for ci in idxs:
        try:
            o = sb.build_observation(current_idx=int(ci), data=nested)
        except Exception:
            continue
        if "portfolio_state" not in o:
            ps = np.zeros(20, dtype=np.float32); ps[0] = 20.5; ps[1] = 20.5
            o["portfolio_state"] = ps
        if "context_vector" not in o:
            o["context_vector"] = np.full(17, 1.0 / 17.0, dtype=np.float32)
        out.append({k: np.asarray(v, dtype=np.float32) for k, v in o.items()})
    print(f"  [obs réelles] {len(out)} observations construites")
    return out


def obs_to_tensor(model, obs: dict):
    """Convertit une obs dict en batch tensor (1, ...) sur le device du modèle."""
    t = {}
    for k, v in obs.items():
        arr = np.asarray(v, dtype=np.float32)[None, ...]  # ajoute la dim batch
        t[k] = torch.as_tensor(arr, device=model.device)
    return t


def collect_pre_tanh(model, obs_batch):
    """Renvoie (MU, SIG) pré-tanh, shapes (N, A), depuis la distribution gSDE."""
    policy = model.policy
    policy.set_training_mode(False)
    mus, sigs = [], []
    with torch.no_grad():
        for obs in obs_batch:
            t = obs_to_tensor(model, obs)
            dist = policy.get_distribution(t)
            inner = dist.distribution  # torch.distributions.Normal (pré-squash)
            mu = inner.mean.detach().cpu().numpy().reshape(-1)
            sig = inner.scale.detach().cpu().numpy().reshape(-1)
            mus.append(mu)
            sigs.append(sig)
    return np.array(mus), np.array(sigs)


def diagnose(mu_abs_mean: float, sig_mean: float) -> str:
    sat_mu = mu_abs_mean > MU_SAT
    low_sig = sig_mean < SIG_LOW
    if sat_mu and low_sig:
        return "CAUSE=μ (+σ bas)"
    if sat_mu and not low_sig:
        return "CAUSE=μ (σ ok mais μ noie)"
    if not sat_mu and low_sig:
        return "CAUSE=σ (exploration)"
    return "SAINE"


def main():
    n = int(os.environ.get("AUDIT_N", "2000"))
    cp = find_checkpoint()
    print("=" * 78)
    print("  PRE-TANH AUDIT — μ et σ AVANT tanh par tête (cause μ vs σ)")
    print("=" * 78)
    print(f"Checkpoint : {cp}")
    model = PPO.load(cp, device="cpu")
    A = model.action_space.shape[0]
    print(f"use_sde={getattr(model, 'use_sde', '?')}  "
          f"squash_output={getattr(model.policy, 'squash_output', '?')}  "
          f"action_dist={type(model.policy.action_dist).__name__}")

    obs = build_real_observations(n)
    src = "RÉELLES (Parquet val)"
    if not obs:
        rng = np.random.default_rng(0)
        obs = [{k: rng.normal(0, 1, sp.shape).astype(np.float32)
                for k, sp in model.observation_space.spaces.items()}
               for _ in range(min(n, 500))]
        src = "BRUIT (secours)"
    print(f"Source obs : {src}  |  N = {len(obs)}")

    MU, SIG = collect_pre_tanh(model, obs)
    # action déterministe attendue = tanh(μ) (squash_output=False => predict squash)
    TANH_MU = np.tanh(MU)

    print("\n── DISTRIBUTION PRÉ-TANH (μ, σ) ET POST-TANH (tanh μ) PAR TÊTE ──")
    print(f"{'dim':<10}{'μ_mean':>10}{'μ_std':>9}{'|μ|_mn':>9}"
          f"{'σ_mean':>9}{'σ_std':>9}{'tanhμ_mn':>10}{'tanhμ_sd':>10}"
          f"{'  diagnostic':<22}")
    diags = {}
    for j in range(A):
        mu = MU[:, j]; sig = SIG[:, j]; tm = TANH_MU[:, j]
        mu_abs = float(np.abs(mu).mean())
        sig_mean = float(sig.mean())
        d = diagnose(mu_abs, sig_mean)
        diags[j] = d
        nm = NAMES[j] if j < len(NAMES) else f"a{j}"
        print(f"{nm:<10}{mu.mean():>10.4f}{mu.std():>9.4f}{mu_abs:>9.4f}"
              f"{sig_mean:>9.4f}{sig.std():>9.4f}{tm.mean():>10.4f}{tm.std():>10.4f}"
              f"  {d:<22}")

    # focus SIZE (dim 1)
    si = 1
    mu_s = MU[:, si]; sig_s = SIG[:, si]
    print(f"\n── FOCUS SIZE (dim {si}) ──")
    print(f"  μ pré-tanh : mean={mu_s.mean():+.4f} std={mu_s.std():.4f} "
          f"min={mu_s.min():+.4f} max={mu_s.max():+.4f}")
    print(f"  σ pré-tanh : mean={sig_s.mean():.4f} std={sig_s.std():.4f} "
          f"min={sig_s.min():.4f} max={sig_s.max():.4f}")
    print(f"  tanh(μ)    : mean={np.tanh(mu_s).mean():+.4f} "
          f"std={np.tanh(mu_s).std():.4f}")
    # quelle σ faudrait-il pour que μ+σ remonte au-dessus de -0.9 post-tanh ?
    # atanh(-0.9) = -1.4722 ; il faut μ + k·σ > -1.4722 ; combien de σ d'écart ?
    target = np.arctanh(-0.9)
    gap = target - mu_s.mean()  # distance à parcourir (négatif si μ déjà au-dessus)
    n_sigmas = gap / max(sig_s.mean(), 1e-6)
    print(f"  Pour atteindre tanh=-0.9 (μ_cible={target:+.3f}) depuis μ_mean : "
          f"écart={gap:+.3f} = {n_sigmas:+.1f} σ")
    if mu_s.mean() < -MU_SAT:
        print(f"  -> μ_mean={mu_s.mean():+.2f} < -{MU_SAT} : tanh DÉJÀ saturé par la "
              f"MOYENNE. Augmenter σ seul est INSUFFISANT ; il faut RÉAPPRENDRE μ.")
    elif sig_s.mean() < SIG_LOW:
        print(f"  -> σ_mean={sig_s.mean():.3f} < {SIG_LOW} : μ pas catastrophique mais "
              f"EXPLORATION quasi nulle. Relever log_std_init/ent_coef peut suffire.")

    print("\n" + "=" * 78)
    print("  CONCLUSION (cause racine SIZE)")
    print("=" * 78)
    ds = diags.get(si, "?")
    if "CAUSE=μ" in ds:
        print("  • SIZE : la MOYENNE pré-tanh est le problème dominant.")
        print("    Un simple reset de log_std NE SUFFIRA PAS. Stratégie :")
        print("    1) log_std_init plus haut (0.0/0.25) POUR commencer à explorer ;")
        print("    2) ent_coef plus fort (0.02) pour pousser hors du plateau ;")
        print("    3) MAIS surtout laisser le gradient de reward ré-apprendre μ sur")
        print("       des positions enfin variées (donc entraînement, pas patch).")
    elif "CAUSE=σ" in ds:
        print("  • SIZE : l'EXPLORATION (σ) est le problème dominant.")
        print("    Relever log_std_init (0.0/0.25) + ent_coef (0.02) devrait suffire")
        print("    à réveiller la tête en 50-100k steps, sans nouveau mécanisme.")
    else:
        print(f"  • SIZE : diagnostic = {ds}. Voir tableau ci-dessus.")
    print(f"\nNote : obs = {src}. μ/σ lus depuis la distribution gSDE pré-squash "
          f"(squash_output=False).")
    print("Rappel : ce script est read-only — aucun entraînement, aucun trade.")


if __name__ == "__main__":
    main()
