"""
train_calibrated_judgments.py — LEGACY EXPERIMENTAL DRAFT, TRAINING BLOCKED

The description below is historical, not an authorized pipeline. CPU 500K
is prohibited; labels/availability/plan-targets/geometry are not locked.
Calibration temperatures are not proof of reliability; TEST is not a tuning
split. The flattened MLP shortcut is disabled. Rewrite at GATE 7/8.
================================================================================

Remplace définitivement l'apprentissage par PnL (PPO) par un entraînement
SUPERVISÉ CALIBRÉ des jugements (workflow étape 12).

Pipeline :
  1. Encodeur MLP provisoire (features 88 → Z 128). L'étape 3 (perception
     CNN + attention + FiLM) le remplacera plus tard SANS toucher au contrat
     Z → jugements (SystemOneCore inchangé).
  2. Entraînement multi-tâches sur les parquets labellisés (étape 12a) :
       NOUL   : BCE + Brier (équivalent à MSE sur sigmoïde), pondéré par classe
       CHOICE : Cross-Entropy (régime, direction), pondérée par classe
       SCORE  : NLL ordinale CORAL (BCE sur seuils cumulatifs P(score>k))
  3. Temperature Scaling sur le split VALIDATION (2022-2023) : ajuste les
     buffers temp_* pour que P=0.70 signifie réellement 70 % de réussite.
  4. Évaluation OOS sur le split TEST (≥2024, jamais vu) : NLL, Brier, ECE,
     diagramme de fiabilité par question NOUL.

Contrainte CPU : modèle ~85K params, batches de 512. 500K pas ≈ réaliste en
background avec checkpoints périodiques + early stopping sur NLL validation.

Sorties :
  models/system_one_judgments.pt   (poids encodeur + SystemOneCore + températures)
  logs/system_one_training.json    (métriques par époque + calibration + test)

Usage :
  PYTHONPATH=src python -m adan_trading_bot.offline.train_calibrated_judgments \
      [--steps 500000] [--batch 512] [--lr 3e-4] [--eval-every 5000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from adan_trading_bot.models.system_one_core import (
    SystemOneCore, Z_FEATURES_DIM, NOUL_QUESTIONS, CHOICE_QUESTIONS,
    SCORE_QUESTIONS, N_SCORE_LEVELS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_MODEL = "models/system_one_judgments.pt"
OUT_LOG = "logs/system_one_training.json"

NOUL_COLS = ["y_mfe_atteint_tp", "y_risque_adverse_faible", "y_expansion_imminente",
             "y_anomalie_donnees", "y_sweep_confirme", "y_reintegration_valide"]
FEATURE_COLS = [f"f{k}" for k in range(Z_FEATURES_DIM)]


# ─── Encodeur provisoire (étape 3 simplifiée — remplaçable sans casser Z) ────
class MlpEncoder(nn.Module):
    """features 88 → Z 128. Perception minimale en attendant le CNN/attention."""

    def __init__(self, in_dim: int = Z_FEATURES_DIM, z_dim: int = 128):
        super().__init__()
        raise RuntimeError("Flattened MLP shortcut BLOCKED; require lineage-preserving relational and temporal perception")
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, z_dim),
        )

    def forward(self, x):
        return self.net(x)


# ─── Perte ordinale CORAL (NLL sur seuils cumulatifs monotones) ──────────────
def ordinal_coral_loss(cum_logits, y):
    """cum_logits : (b, 10) logits de P(score > k) pour k=0..9.
    y : (b,) entier 0..10. Cible binaire par seuil : P(>k) = 1 si y > k."""
    k = torch.arange(cum_logits.shape[1], device=cum_logits.device)
    target = (y.unsqueeze(1) > k.unsqueeze(0)).float()   # (b, 10)
    return F.binary_cross_entropy_with_logits(cum_logits, target)


def expected_calibration_error(preds, targets, n_bins=10):
    """ECE : écart moyen entre confiance prédite et fréquence observée."""
    preds = np.asarray(preds); targets = np.asarray(targets)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        m = (preds >= bins[b]) & (preds < bins[b + 1])
        if m.sum() > 0:
            ece += (m.mean()) * abs(targets[m].mean() - preds[m].mean())
    return float(ece)


def load_split(name):
    df = pd.read_parquet(f"data/labeled/{name}.parquet")
    X = torch.tensor(df[FEATURE_COLS].to_numpy(np.float32))
    y = {c: torch.tensor(df[c].to_numpy(np.int64)) for c in
         NOUL_COLS + ["y_regime", "y_direction", "y_qualite_setup", "y_conviction"]}
    return X, y


def class_weights_binary(y):
    """Pondération pos/neg pour BCE déséquilibrée : pos_weight = neg/pos."""
    pos = float((y == 1).sum()); neg = float((y == 0).sum())
    return torch.tensor(neg / max(pos, 1.0), dtype=torch.float32)


def class_weights_choice(y, n_classes):
    counts = torch.bincount(y, minlength=n_classes).float()
    w = 1.0 / torch.clamp(counts, min=1.0)
    return (w / w.sum() * n_classes)


def forward_all(encoder, core, X):
    """Encode → jugements. Retourne le dict de logits du SystemOneCore."""
    z = encoder(X)
    return core(z)


def compute_losses(encoder, core, X, y, w_noul, w_choice):
    """Perte multi-tâches pondérée. Retourne (loss_totale, dict des composantes)."""
    out = forward_all(encoder, core, X)
    parts = {}
    # NOUL : BCE pondérée par classe (pos_weight)
    ln = 0.0
    for qi, q in enumerate(NOUL_QUESTIONS):
        logit = out["noul"][q]
        target = y[NOUL_COLS[qi]].float()
        ln = ln + F.binary_cross_entropy_with_logits(
            logit, target, pos_weight=w_noul[qi].to(X.device))
    parts["noul"] = ln / len(NOUL_QUESTIONS)

    # CHOICE : Cross-Entropy pondérée
    lc = 0.0
    for ci, q in enumerate(CHOICE_QUESTIONS.keys()):
        logits = out["choice"][q]
        target = y[f"y_{q}"]
        lc = lc + F.cross_entropy(logits, target, weight=w_choice[ci].to(X.device))
    parts["choice"] = lc / len(CHOICE_QUESTIONS)

    # SCORE : NLL ordinale CORAL
    ls = 0.0
    for qi, q in enumerate(SCORE_QUESTIONS):
        cum = out["score"][q]
        target = y["y_qualite_setup" if q == "qualite_setup" else "y_conviction"]
        ls = ls + ordinal_coral_loss(cum, target)
    parts["score"] = ls / len(SCORE_QUESTIONS)

    total = parts["noul"] + parts["choice"] + parts["score"]
    return total, parts


@torch.no_grad()
def eval_nll(encoder, core, X, y, w_noul, w_choice, n=20000):
    """NLL de validation sur un sous-échantillon (rapidité)."""
    encoder.eval(); core.eval()
    m = min(n, len(X))
    idx = torch.randperm(len(X))[:m].to(X.device)
    Xs, ys = X[idx], {k: v[idx] for k, v in y.items()}
    loss, parts = compute_losses(encoder, core, Xs, ys, w_noul, w_choice)
    encoder.train(); core.train()
    return float(loss.item()), {k: float(v.item()) for k, v in parts.items()}


# ─── Temperature Scaling sur VALIDATION (étape 12 : calibration) ─────────────
def _fit_scalar_temperature(loss_fn, iters=300, lr=0.01):
    """Ajuste UNE température scalaire par Adam + clamp [0.05, 10].
    Plus robuste que LBFGS sur ces petites NLL (évite les divergences
    T→2761 / T→0 observées au smoke test). Retourne la température."""
    log_t = torch.zeros(1, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        t = log_t.exp().clamp(0.05, 10.0)
        loss = loss_fn(t)
        loss.backward()
        opt.step()
    return float(log_t.exp().clamp(0.05, 10.0).item())


def fit_temperatures(encoder, core, X_val, y_val, iters=300):
    """Ajuste les températures par question sur le split validation (NLL)."""
    encoder.eval(); core.eval()
    m = min(50000, len(X_val))
    idx = torch.randperm(len(X_val))[:m].to(DEVICE)
    X = X_val[idx]; yv = {k: v[idx] for k, v in y_val.items()}

    temps = {"noul": [], "choice": [], "score": []}

    with torch.no_grad():
        out = forward_all(encoder, core, X)

    # NOUL : une température scalaire par question (BCE)
    for qi, q in enumerate(NOUL_QUESTIONS):
        logit = out["noul"][q].detach()
        target = yv[NOUL_COLS[qi]].float()
        temps["noul"].append(_fit_scalar_temperature(
            lambda t: F.binary_cross_entropy_with_logits(logit / t, target), iters))

    # CHOICE : température scalaire par question (CE)
    for q in CHOICE_QUESTIONS.keys():
        logits = out["choice"][q].detach()
        target = yv[f"y_{q}"]
        temps["choice"].append(_fit_scalar_temperature(
            lambda t: F.cross_entropy(logits / t, target), iters))

    # SCORE : température scalaire sur les logits cumulatifs (NLL ordinale)
    for q in SCORE_QUESTIONS:
        cum = out["score"][q].detach()
        target = yv["y_qualite_setup" if q == "qualite_setup" else "y_conviction"]
        temps["score"].append(_fit_scalar_temperature(
            lambda t: ordinal_coral_loss(cum / t, target), iters))

    return temps


def reliability_report(encoder, core, X_test, y_test):
    """Diagramme de fiabilité + ECE par question NOUL sur le TEST (OOS)."""
    encoder.eval(); core.eval()
    report = {}
    B = 20000
    with torch.no_grad():
        for s in range(0, min(len(X_test), 120000), B):
            X = X_test[s:s + B]
            out = forward_all(encoder, core, X)
            for qi, q in enumerate(NOUL_QUESTIONS):
                p = torch.sigmoid(out["noul"][q] / core.temp_noul[qi]).cpu().numpy()
                t = y_test[NOUL_COLS[qi]][s:s + B].cpu().numpy()
                report.setdefault(q, {"p": [], "t": []})
                report[q]["p"].append(p); report[q]["t"].append(t)
    summary = {}
    for q, d in report.items():
        p = np.concatenate(d["p"]); t = np.concatenate(d["t"])
        summary[q] = {
            "brier": float(np.mean((p - t) ** 2)),
            "ece": expected_calibration_error(p, t),
            "prevalence": float(t.mean()),
            "pred_moyenne": float(p.mean()),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--z-dim", type=int, default=128)
    ap.add_argument("--out", type=str, default=OUT_MODEL)
    args = ap.parse_args()

    from adan_trading_bot.features.feature_registry import get_feature_registry
    from adan_trading_bot.features.feature_availability_contract import FeatureAvailabilityContract
    # BEFORE any dataset load, optimizer, checkpoint or directory creation.
    FeatureAvailabilityContract(get_feature_registry()).require_training()

    t0 = time.time()
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print("=" * 74)
    print(f"ADAN-SYSTEM-ONE · ÉTAPE 12b — Entraînement {args.steps:,} pas "
          f"(batch {args.batch}, device {DEVICE})")
    print("=" * 74, flush=True)

    # ── Données ──
    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")
    Xte, yte = load_split("test")
    print(f"train {len(Xtr):,} · val {len(Xva):,} · test {len(Xte):,}", flush=True)
    Xtr, Xva, Xte = Xtr.to(DEVICE), Xva.to(DEVICE), Xte.to(DEVICE)
    ytr = {k: v.to(DEVICE) for k, v in ytr.items()}
    yva = {k: v.to(DEVICE) for k, v in yva.items()}
    yte = {k: v.to(DEVICE) for k, v in yte.items()}

    # Pondérations de classe (déséquilibre : sweep 1.2%, TRAP 94%)
    w_noul = [class_weights_binary(ytr[c]) for c in NOUL_COLS]
    w_choice = [class_weights_choice(ytr["y_regime"], 4),
                class_weights_choice(ytr["y_direction"], 3)]

    # ── Modèle ──
    encoder = MlpEncoder(Z_FEATURES_DIM, args.z_dim).to(DEVICE)
    core = SystemOneCore(dim_z=args.z_dim).to(DEVICE)
    n_params = sum(p.numel() for p in list(encoder.parameters()) + list(core.parameters()))
    print(f"Paramètres : {n_params:,} (encodeur + SystemOneCore)", flush=True)

    params = list(encoder.parameters()) + list(core.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # ── Entraînement ──
    history = []
    best_val = float("inf"); best_state = None; since_best = 0
    n = len(Xtr)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch,), device=DEVICE)
        X = Xtr[idx]; y = {k: v[idx] for k, v in ytr.items()}
        loss, parts = compute_losses(encoder, core, X, y, w_noul, w_choice)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            val_loss, val_parts = eval_nll(encoder, core, Xva, yva, w_noul, w_choice)
            history.append({"step": step, "train_loss": float(loss.item()),
                            "val_loss": val_loss, **{f"val_{k}": v for k, v in val_parts.items()}})
            flag = ""
            if val_loss < best_val - 1e-4:
                best_val = val_loss; since_best = 0
                best_state = {"encoder": encoder.state_dict(), "core": core.state_dict()}
                flag = "  ← best"
            else:
                since_best += 1
            print(f"  step {step:>7,}  train {loss.item():.4f}  val {val_loss:.4f} "
                  f"(noul {val_parts['noul']:.4f} choice {val_parts['choice']:.4f} "
                  f"score {val_parts['score']:.4f}){flag}  [{time.time()-t0:.0f}s]", flush=True)
            if since_best >= args.patience:
                print(f"  Early stopping (patience {args.patience}) au step {step:,}", flush=True)
                break

    # ── Restaurer le meilleur état ──
    if best_state:
        encoder.load_state_dict(best_state["encoder"])
        core.load_state_dict(best_state["core"])
        print(f"  Meilleur état restauré (val_loss {best_val:.4f})", flush=True)

    # ── Temperature Scaling sur VALIDATION ──
    print("\n── Temperature Scaling (split validation 2022-2023) ──", flush=True)
    temps = fit_temperatures(encoder, core, Xva, yva)
    core.calibrate(temp_noul=temps["noul"], temp_choice=temps["choice"],
                   temp_score=temps["score"])
    print("  temp NOUL   :", [round(t, 3) for t in temps["noul"]])
    print("  temp CHOICE :", [round(t, 3) for t in temps["choice"]])
    print("  temp SCORE  :", [round(t, 3) for t in temps["score"]])

    # ── Évaluation OOS sur TEST (≥2024, jamais vu) ──
    print("\n── Fiabilité OOS (split test ≥2024) ──", flush=True)
    rel = reliability_report(encoder, core, Xte, yte)
    for q, m in rel.items():
        print(f"  {q:28s} Brier {m['brier']:.4f}  ECE {m['ece']:.4f}  "
              f"prev {m['prevalence']:.3f}  pred {m['pred_moyenne']:.3f}", flush=True)

    # ── Sauvegarde ──
    torch.save({
        "encoder": encoder.state_dict(),
        "core": core.state_dict(),
        "z_dim": args.z_dim,
        "temperatures": temps,
        "best_val_loss": best_val,
        "steps_ran": step,
        "config": vars(args),
    }, args.out)
    log = {"history": history, "temperatures": temps, "reliability_test": rel,
           "best_val_loss": best_val, "steps_ran": step, "params": n_params,
           "device": str(DEVICE), "duration_s": round(time.time() - t0, 1)}
    with open(OUT_LOG, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n✅ Modèle sauvegardé → {args.out}")
    print(f"✅ Journal → {OUT_LOG}")
    print(f"✅ Terminé en {time.time()-t0:.0f}s ({step:,} pas, best val {best_val:.4f})")


if __name__ == "__main__":
    main()
