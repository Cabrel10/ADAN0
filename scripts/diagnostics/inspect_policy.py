#!/usr/bin/env python3
"""A7-PRE — Introspection forensique de la politique PPO.

NE CALCULE AUCUNE STATISTIQUE D'ACTION. Se contente de répondre aux questions
soulevées par la revue utilisateur AVANT de rejouer A7 :

  Q1. Le checkpoint chargé est-il VRAIMENT 500k_FIXED ? (hash + chemin résolu)
  Q2. La politique utilise-t-elle squash_output (tanh) et/ou gSDE (use_sde) ?
       -> détermine si dist.distribution.mean est PRÉ-squash (non borné).
  Q3. Quel est le TYPE exact de la distribution d'action ?
  Q4. Le paramètre brut policy.log_std : shape + valeurs PAR DIMENSION.
       -> réconcilie log_std = -2.04 (vu plus tôt) vs +0.14 (A7).
  Q5. action_space : bornes réelles (low/high).

Aucune écriture, aucun trade, aucun entraînement.
"""
from __future__ import annotations

import os
import sys
import glob
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(str(ROOT))

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402


def sha256(path: str, limit: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def find_checkpoint() -> str:
    cands = sorted(glob.glob("checkpoints/*FIXED*.zip")) or \
        sorted(glob.glob("checkpoints/*.zip"))
    for c in cands:
        if "500k_FIXED" in c:
            return c
    return cands[-1]


def main():
    cp = find_checkpoint()
    cp_abs = str(Path(cp).resolve())
    print("=" * 72)
    print("  A7-PRE — INTROSPECTION POLITIQUE (read-only)")
    print("=" * 72)
    print(f"Q1. Checkpoint résolu : {cp_abs}")
    print(f"    sha256[:16] (4MB) : {sha256(cp)}")
    print(f"    contient '500k_FIXED' : {'500k_FIXED' in cp}")

    model = PPO.load(cp, device="cpu")
    policy = model.policy

    print("\nQ2. Flags de la politique")
    use_sde = getattr(model, "use_sde", None)
    if use_sde is None:
        use_sde = getattr(policy, "use_sde", None)
    squash = getattr(policy, "squash_output", None)
    print(f"    use_sde        : {use_sde}")
    print(f"    squash_output  : {squash}")
    print(f"    policy class   : {type(policy).__name__}")

    print("\nQ3. Type de distribution d'action")
    ad = getattr(policy, "action_dist", None)
    print(f"    action_dist    : {type(ad).__name__ if ad is not None else None}")

    print("\nQ4. Paramètre BRUT policy.log_std")
    ls_param = None
    for name, p in policy.named_parameters():
        if "log_std" in name:
            ls_param = (name, p.detach().cpu().numpy())
            break
    if ls_param is not None:
        name, arr = ls_param
        print(f"    nom            : {name}")
        print(f"    shape          : {arr.shape}")
        flat = arr.flatten()
        print(f"    valeurs        : {np.round(flat, 4).tolist()}")
        print(f"    -> std=exp(ls) : {np.round(np.exp(flat), 4).tolist()}")
        # dim 4 = tp (si 5 dims aplaties)
        if flat.size >= 5:
            print(f"    log_std[tp=4]  : {flat[4]:.4f}  (std={np.exp(flat[4]):.4f})")
            print(f"    log_std[sl=3]  : {flat[3]:.4f}  (std={np.exp(flat[3]):.4f})")
    else:
        print("    AUCUN paramètre 'log_std' trouvé (gSDE ? scale state-dependent ?)")
        # lister les params pour comprendre
        print("    params de la politique contenant 'std'/'sde'/'noise' :")
        for name, p in policy.named_parameters():
            low = name.lower()
            if any(k in low for k in ("std", "sde", "noise", "log")):
                print(f"      - {name}  shape={tuple(p.shape)}")

    print("\nQ5. action_space")
    sp = model.action_space
    print(f"    type           : {type(sp).__name__}")
    print(f"    shape          : {sp.shape}")
    if hasattr(sp, "low"):
        print(f"    low            : {np.round(sp.low, 3).tolist()}")
        print(f"    high           : {np.round(sp.high, 3).tolist()}")

    # Q6 bonus : sur 1 obs, comparer mean PRÉ-squash vs action POST-predict
    print("\nQ6. PRÉ-squash (dist.mean) vs POST (predict déterministe) — 1 obs nulle")
    obs = {}
    for k, space in model.observation_space.spaces.items():
        obs[k] = np.zeros(space.shape, dtype=np.float32)
    policy.set_training_mode(False)
    with torch.no_grad():
        obs_t, _ = policy.obs_to_tensor(obs)
        dist = policy.get_distribution(obs_t)
        mu_pre = dist.distribution.mean.cpu().numpy().flatten()
    act_det, _ = model.predict(obs, deterministic=True)
    act_det = np.asarray(act_det).flatten()
    print(f"    mu PRÉ-squash  : {np.round(mu_pre, 4).tolist()}")
    print(f"    action POST    : {np.round(act_det, 4).tolist()}")
    print(f"    tanh(mu_pre)   : {np.round(np.tanh(mu_pre), 4).tolist()}")
    print(f"    -> POST == tanh(PRÉ) ? {np.allclose(act_det, np.tanh(mu_pre), atol=1e-3)}")


if __name__ == "__main__":
    main()
