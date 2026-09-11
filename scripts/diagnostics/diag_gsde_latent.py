#!/usr/bin/env python3
"""diag_gsde_latent.py — MESURER (pas supposer) la cause de l'explosion gSDE.

Contexte : avec la vraie architecture (ContextualTemporalFusionExtractor), la
variance gSDE σ explose (3.4 -> 13 -> 41 -> 110) en quelques rollouts. Plusieurs
hypothèses circulent (obs non normalisées, features CNN trop grandes, FiLM qui
amplifie, gSDE mal adapté). AUCUNE n'est prouvée tant qu'on n'a pas mesuré.

Ce script construit la policy gSDE EXACTE du sandbox (même extracteur, même
log_std_init), passe de VRAIES observations normalisées (StateBuilder, comme en
training) et mesure, SANS rien entraîner :

  1. magnitude du vecteur de features (sortie de l'extracteur) :
        mean, std, min, max, et |features| moyen — c'est latent_sde.
  2. la statistique latent_sde utilisée par gSDE (policy.policy.latent_sde si
     dispo, sinon les features partagées).
  3. la σ pré-tanh effective par tête = sqrt(latent_sde² @ exp(log_std)² + eps),
     pour log_std_init ∈ {-0.5, 0.0} et avec/sans une LayerNorm de features.

But : déterminer EMPIRIQUEMENT si l'explosion vient de la magnitude des features
(donc une LayerNorm sur la sortie de l'extracteur la corrige) ou d'autre chose.

Usage :
    PYTHONPATH=src python scripts/diag_gsde_latent.py [--n 256]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _build_env_and_obs(n_obs):
    """Construit le MultiAssetChunkedEnv comme le sandbox et collecte n_obs
    observations réelles (normalisées par StateBuilder)."""
    from adan_trading_bot.common.config_loader import ConfigLoader
    import copy as _copy
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

    config = ConfigLoader.load_config("config/config.yaml")
    worker_config = _copy.deepcopy(config.get("workers", {}).get("w1", {}))
    env = MultiAssetChunkedEnv(
        config=config,
        worker_config=worker_config,
        worker_id=0,
        live_mode=False,
    )
    obs, _ = env.reset()
    obs_list = [obs]
    for _ in range(n_obs - 1):
        a = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(a)
        obs_list.append(obs)
        if term or trunc:
            obs, _ = env.reset()
    return env, obs_list


def _stack_obs(obs_list, device):
    keys = obs_list[0].keys()
    out = {}
    for k in keys:
        arr = np.stack([np.asarray(o[k], dtype=np.float32) for o in obs_list], axis=0)
        out[k] = torch.as_tensor(arr, device=device)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from adan_trading_bot.agent.feature_extractors import ContextualTemporalFusionExtractor

    print("=" * 76)
    print("DIAG gSDE — mesure de la magnitude des features et de σ effective")
    print("=" * 76)

    print(f"\n[1] Construction env + collecte de {args.n} observations réelles...")
    env, obs_list = _build_env_and_obs(args.n)
    obs_space = env.observation_space
    act_space = env.action_space

    # raw obs magnitude (preuve que StateBuilder normalise/clippe bien)
    print("\n[2] Magnitude des OBSERVATIONS BRUTES (par clé) — attendu: ~[-10,10]")
    for k in obs_list[0].keys():
        arr = np.stack([np.asarray(o[k], dtype=np.float32).ravel() for o in obs_list])
        print(f"    {k:18} mean={arr.mean():+.3f} std={arr.std():.3f} "
              f"min={arr.min():+.2f} max={arr.max():+.2f}")

    # policy gSDE comme le sandbox
    print("\n[3] Construction policy gSDE (ContextualTemporalFusionExtractor)...")

    import gymnasium as gym

    class _Stub(gym.Env):
        observation_space = obs_space
        action_space = act_space

        def reset(self, *a, **k):
            return obs_space.sample(), {}

        def step(self, action):
            return obs_space.sample(), 0.0, True, False, {}

    venv = DummyVecEnv([lambda: _Stub()])
    policy_kwargs = dict(
        features_extractor_class=ContextualTemporalFusionExtractor,
        features_extractor_kwargs={"context_dim": 14},
        share_features_extractor=True,
        log_std_init=-0.5,
    )
    model = PPO("MultiInputPolicy", venv, use_sde=True, sde_sample_freq=4,
                policy_kwargs=policy_kwargs, device="cpu", verbose=0)
    policy = model.policy
    device = model.device
    obs_t = _stack_obs(obs_list, device)

    # ---- mesurer la sortie de l'extracteur (= latent_sde / features) ----
    print("\n[4] Magnitude des FEATURES (sortie extracteur) = ce que gSDE consomme")
    with torch.no_grad():
        feats = policy.extract_features(obs_t)
        if isinstance(feats, tuple):
            feats = feats[0]
        f = feats.detach().cpu().numpy()
    fabs = np.abs(f)
    print(f"    features shape={f.shape}")
    print(f"    mean={f.mean():+.4f}  std={f.std():.4f}  "
          f"min={f.min():+.3f}  max={f.max():+.3f}")
    print(f"    |features| mean={fabs.mean():.4f}  p99={np.percentile(fabs,99):.3f}  "
          f"max={fabs.max():.3f}")
    feat_l2_per_sample = np.sqrt((f * f).sum(axis=1))
    print(f"    ||features||_2 par échantillon: mean={feat_l2_per_sample.mean():.3f} "
          f"max={feat_l2_per_sample.max():.3f}")

    # ---- σ effective gSDE = sqrt(latent_sde² @ std² + eps) ----
    # SB3 StateDependentNoiseDistribution: variance = latent_sde**2 @ self.get_std(log_std)**2
    print("\n[5] σ gSDE EFFECTIVE = sqrt(features² @ exp(log_std)² + eps)")
    feats_t = torch.as_tensor(f)
    action_dim = act_space.shape[0]
    for lsi in (-0.5, 0.0):
        std = float(np.exp(lsi))  # get_std ~ exp(log_std) (sans use_expln)
        # variance par dim d'action = sum_j features_j² * std²  (latent_sde @ std²)
        # approximation: σ_dim ≈ sqrt( mean_j features² * dim_features ) * std
        var = (feats_t.pow(2).sum(dim=1, keepdim=True) * (std ** 2))  # [N,1]
        sigma = torch.sqrt(var + 1e-6)
        print(f"    log_std_init={lsi:+.1f} (std={std:.3f}): "
              f"σ_eff mean={sigma.mean().item():.2f}  max={sigma.max().item():.2f}")

    # ---- effet d'une LayerNorm sur les features (correctif candidat) ----
    print("\n[6] EFFET d'une LayerNorm sur les features (correctif candidat)")
    ln = torch.nn.LayerNorm(f.shape[1])
    with torch.no_grad():
        f_ln = ln(torch.as_tensor(f)).numpy()
    print(f"    après LayerNorm: |features| mean={np.abs(f_ln).mean():.4f} "
          f"||.||_2 mean={np.sqrt((f_ln**2).sum(1)).mean():.3f}")
    f_ln_t = torch.as_tensor(f_ln)
    for lsi in (-0.5, 0.0):
        std = float(np.exp(lsi))
        var = (f_ln_t.pow(2).sum(dim=1, keepdim=True) * (std ** 2))
        sigma = torch.sqrt(var + 1e-6)
        print(f"    [LN] log_std_init={lsi:+.1f}: σ_eff mean={sigma.mean().item():.2f} "
              f"max={sigma.max().item():.2f}")

    print("\n" + "=" * 76)
    print("LECTURE :")
    print("  - Si |features| et ||features||_2 sont GRANDS (≫ qq unités) et que")
    print("    σ_eff dépasse déjà ~10 sans entraînement, alors la magnitude des")
    print("    features est la cause directe (gSDE consomme features²).")
    print("  - Si la LayerNorm ramène σ_eff sous ~2-3, alors normaliser la sortie")
    print("    de l'extracteur AVANT gSDE est le correctif propre (et n'affecte ni")
    print("    les obs déjà normalisées, ni le reward).")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
