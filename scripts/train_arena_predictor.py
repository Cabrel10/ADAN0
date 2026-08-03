#!/usr/bin/env python3
"""V18 — Entraînement HORS-LIGNE du prédicteur Arena.

Lit les échantillons JSONL produits par le Collector pendant l'entraînement RL
(present_state -> params optimaux ex-post) et apprend un MLP hétéroscédastique
qui prédit, pour chaque état présent, la DISTRIBUTION des paramètres optimaux.

Usage :
    python scripts/train_arena_predictor.py \
        --samples logs/arena/arena_samples.jsonl \
        --out models/arena_predictor/arena_predictor.pt \
        --epochs 200 --batch 256 --lr 1e-3

Live-safe : l'entrée du modèle est UNIQUEMENT le vecteur d'état présent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# --- Path bootstrap (scripts/ est un niveau sous la racine du repo) --------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from adan_trading_bot.arena_predictor.state_schema import (  # noqa: E402
    STATE_DIM, TARGET_DIM, TARGET_PARAMS,
)
from adan_trading_bot.arena_predictor.predictor import (  # noqa: E402
    ArenaPredictor, TargetScaler, gaussian_nll, save_predictor,
)


def load_samples(path: str):
    """Charge les JSONL -> (states[N,13], targets[N,5])."""
    states, targets = [], []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier d'échantillons introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            st = d.get("state")
            if not st or len(st) != STATE_DIM:
                continue
            tgt = [float(d.get(p, 0.0)) for p in TARGET_PARAMS]
            states.append([float(x) for x in st])
            targets.append(tgt)
    if not states:
        raise ValueError("Aucun échantillon valide trouvé.")
    return torch.tensor(states, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="logs/arena/arena_samples.jsonl")
    ap.add_argument("--out", default="models/arena_predictor/arena_predictor.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=25)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    X, Y = load_samples(args.samples)
    n = X.shape[0]
    print(f"[ARENA_TRAIN] {n} échantillons chargés | state_dim={X.shape[1]} target_dim={Y.shape[1]}")

    # Split train/val chronologique (les JSONL sont append-only, donc ordonnés).
    n_val = max(1, int(n * args.val_frac))
    n_tr = n - n_val
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]
    X_va, Y_va = X[n_tr:], Y[n_tr:]

    scaler = TargetScaler().fit(Y_tr)
    Ytr_z = scaler.transform(Y_tr)
    Yva_z = scaler.transform(Y_va)

    ds = TensorDataset(X_tr, Ytr_z)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False)

    model = ArenaPredictor(state_dim=STATE_DIM, target_dim=TARGET_DIM, dropout=args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val = float("inf")
    best_state = None
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            mean, log_std = model(xb)
            loss = gaussian_nll(mean, log_std, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item() * xb.shape[0]
        tr_loss /= max(1, n_tr)

        model.eval()
        with torch.no_grad():
            mean_v, log_std_v = model(X_va)
            val_loss = gaussian_nll(mean_v, log_std_v, Yva_z).item()

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"[ARENA_TRAIN] epoch {epoch:4d} | train_nll={tr_loss:.4f} | val_nll={val_loss:.4f} | best={best_val:.4f}")

        if bad >= args.patience:
            print(f"[ARENA_TRAIN] early stop @ epoch {epoch} (patience={args.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Diagnostic final : erreur moyenne par cible (échelle réelle).
    model.eval()
    with torch.no_grad():
        mean_v, log_std_v = model(X_va)
        mean_real = scaler.inverse_mean(mean_v)
        mae = (mean_real - Y_va).abs().mean(dim=0)
    print("[ARENA_TRAIN] MAE par cible (échelle réelle) :")
    for i, p in enumerate(TARGET_PARAMS):
        print(f"    {p:12s} MAE={mae[i].item():.5f}")

    save_predictor(model, scaler, args.out)
    print(f"[ARENA_TRAIN] Modèle sauvegardé -> {args.out}")


if __name__ == "__main__":
    main()
