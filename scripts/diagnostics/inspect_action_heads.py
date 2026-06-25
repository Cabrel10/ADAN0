#!/usr/bin/env python3
"""SIZE-CHECK — Vérification DÉFINITIVE de l'état des 5 têtes d'action.

Motivation (revue utilisateur juin 2026) :
  L'audit A7 v2 a révélé incidemment : size mean=-1.000, std=0.000.
  Une tête size GELÉE à -1.0 (= toujours taille minimale) est PLUS GRAVE que
  la saturation TP : elle compresse le PnL réel et fausse l'interprétation.
  À vérifier AVANT de toucher au reward.

Méthode (read-only, obs RÉELLES, action POST-clip) :
  1. Obs réelles depuis Parquet val (StateBuilder, comme audit_tp_head v2).
  2. Action déterministe (DÉCISION) ET stochastique (EXPLORATION) via model.predict.
  3. Par dimension : min/max/mean/std + histogramme + fraction au plancher/plafond.
  4. DISTINCTION CLÉ :
       - GELÉE (poids morts) : std_det≈0 ET std_sto≈0 → le reward seul ne peut RIEN.
       - SATURÉE (explorable): std_det petit MAIS std_sto > 0 → reward + entropie
                                peuvent désaturer.
  5. Inspecter le réseau : poids de la dernière couche (action_net) par dimension,
     pour voir si une tête a des poids ~0 (vraiment morte) ou des poids vivants
     mais un biais qui sature.

Aucune écriture, aucun trade, aucun entraînement.
"""
from __future__ import annotations

import os
import sys
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(str(ROOT))

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

NAMES = ["direction", "size", "tf", "sl", "tp"]


def find_checkpoint() -> str:
    cands = sorted(glob.glob("checkpoints/*FIXED*.zip")) or \
        sorted(glob.glob("checkpoints/*.zip"))
    for c in cands:
        if "500k_FIXED" in c:
            return c
    return cands[-1]


def build_real_observations(n: int, seed: int = 0) -> list[dict]:
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


def collect(model, obs_batch):
    model.policy.set_training_mode(False)
    dets, stos = [], []
    for obs in obs_batch:
        a_d, _ = model.predict(obs, deterministic=True)
        a_s, _ = model.predict(obs, deterministic=False)
        dets.append(np.asarray(a_d, float).flatten())
        stos.append(np.asarray(a_s, float).flatten())
    return np.array(dets), np.array(stos)


def hist_str(x, bins=10, lo=-1, hi=1):
    h, e = np.histogram(x, bins=bins, range=(lo, hi))
    t = max(h.sum(), 1)
    return "\n".join(f"    [{e[i]:+.2f},{e[i+1]:+.2f}) {h[i]/t*100:5.1f}% "
                     f"{'█'*int(h[i]/t*30)}" for i in range(bins))


def inspect_network(model):
    """Poids de la couche de sortie action (action_net) par dimension."""
    print("\n── POIDS RÉSEAU : couche de sortie action (action_net) ──")
    policy = model.policy
    anet = getattr(policy, "action_net", None)
    if anet is None:
        print("  (action_net introuvable)")
        return
    # action_net : Linear(latent -> A). weight shape [A, latent], bias [A].
    W = b = None
    for name, p in anet.named_parameters():
        arr = p.detach().cpu().numpy()
        if "weight" in name:
            W = arr
        elif "bias" in name:
            b = arr
    if W is None:
        print("  (poids non standard)")
        return
    A = W.shape[0]
    print(f"  action_net.weight shape={W.shape}  bias shape="
          f"{None if b is None else b.shape}")
    print(f"  {'dim':<10}{'||w||_2':>10}{'w_mean':>10}{'w_max|':>10}{'bias':>10}")
    for j in range(A):
        wn = float(np.linalg.norm(W[j]))
        wm = float(W[j].mean())
        wmx = float(np.abs(W[j]).max())
        bj = float(b[j]) if b is not None else float("nan")
        nm = NAMES[j] if j < len(NAMES) else f"a{j}"
        print(f"  {nm:<10}{wn:>10.4f}{wm:>10.4f}{wmx:>10.4f}{bj:>10.4f}")
    print("  -> poids ~0 + ||w||≈0 = tête VRAIMENT morte (gradient ne passera pas).")
    print("  -> poids vivants + biais extrême = tête SATURÉE (réveillable).")


def main():
    n = int(os.environ.get("AUDIT_N", "2000"))
    cp = find_checkpoint()
    print("=" * 72)
    print("  SIZE-CHECK — état des 5 têtes d'action (read-only, obs réelles)")
    print("=" * 72)
    print(f"Checkpoint : {cp}")
    model = PPO.load(cp, device="cpu")
    A = model.action_space.shape[0]

    obs = build_real_observations(n)
    src = "RÉELLES (Parquet val)"
    if not obs:
        rng = np.random.default_rng(0)
        obs = [{k: rng.normal(0, 1, sp.shape).astype(np.float32)
                for k, sp in model.observation_space.spaces.items()}
               for _ in range(n)]
        src = "BRUIT (secours)"
    print(f"Source obs : {src}  |  N = {len(obs)}")

    DET, STO = collect(model, obs)

    print("\n── ÉTAT DES TÊTES (action post-clip) ──")
    print(f"{'dim':<10}{'min':>9}{'max':>9}{'mean':>9}{'std_det':>9}"
          f"{'std_sto':>9}{'verdict':>16}")
    for j in range(A):
        d = DET[:, j]
        sd, ss = d.std(), STO[:, j].std()
        if sd < 0.02 and ss < 0.02:
            v = "GELÉE"
        elif sd < 0.10 and ss >= 0.02:
            v = "SATURÉE(explor.)"
        else:
            v = "VIVANTE"
        nm = NAMES[j] if j < len(NAMES) else f"a{j}"
        print(f"{nm:<10}{d.min():>9.4f}{d.max():>9.4f}{d.mean():>9.4f}"
              f"{sd:>9.4f}{ss:>9.4f}{v:>16}")

    # focus SIZE (dim 1)
    si = 1
    print(f"\n── FOCUS TÊTE SIZE (dim {si}) ──")
    s = DET[:, si]
    print(f"  déterministe : min={s.min():.5f} max={s.max():.5f} "
          f"mean={s.mean():.5f} std={s.std():.6f}")
    print(f"  stochastique : min={STO[:,si].min():.5f} max={STO[:,si].max():.5f} "
          f"mean={STO[:,si].mean():.5f} std={STO[:,si].std():.6f}")
    print(f"  % au plancher (-1.0, atol 1e-3) : "
          f"{float(np.mean(np.isclose(s,-1.0,atol=1e-3)))*100:.1f}% (det) / "
          f"{float(np.mean(np.isclose(STO[:,si],-1.0,atol=1e-3)))*100:.1f}% (sto)")
    print("  histogramme déterministe:")
    print(hist_str(s))
    print("  histogramme stochastique:")
    print(hist_str(STO[:, si]))

    inspect_network(model)

    print("\n" + "=" * 72)
    print("  CONCLUSION SIZE")
    print("=" * 72)
    s_sd, s_ss = DET[:, si].std(), STO[:, si].std()
    if s_sd < 0.02 and s_ss < 0.02:
        print("  • Tête SIZE GELÉE : déterministe ET stochastique quasi constants à -1.")
        print("    Le reward seul NE LA RÉVEILLERA PAS. Au prochain entraînement :")
        print("    augmenter ent_coef (0 -> ~0.01-0.02) + reset/ré-init log_std de la")
        print("    tête, sinon le gradient guidé par le futur ne pénètre pas.")
    elif s_sd < 0.10 and s_ss >= 0.02:
        print("  • Tête SIZE SATURÉE mais EXPLORABLE (std_sto>0). Le reward futur +")
        print("    un ent_coef modéré PEUVENT la désaturer progressivement.")
    else:
        print("  • Tête SIZE VIVANTE : varie déjà significativement.")
    print("\nNote : obs = " + src + ". Action mesurée = post-clip (model.predict).")


if __name__ == "__main__":
    main()
