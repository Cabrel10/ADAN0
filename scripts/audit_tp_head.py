#!/usr/bin/env python3
"""A7 — Audit READ-ONLY de la tête d'action TP (et SL) du PPO.

Objectif (cahier §13.7 / revue utilisateur) : PROUVER lequel des 3 cas est vrai,
SANS réentraîner, sur un checkpoint existant.

  Cas A — tête TP morte      : variance(mu_tp) ≈ 0 ET log_std_tp très négatif.
  Cas B — TP écrasé en aval  : la tête varie, mais le mapping env [tp_lo,tp_hi] +
                               la règle R/R écrasent la sortie (impact effectif ~0).
  Cas C — signal faible      : la tête varie, le TP varie un peu, mais les sorties
                               AGENT_CLOSE dominent / TP rarement atteint.

Méthode :
  1. Charger le checkpoint PPO (CPU, read-only).
  2. Générer N observations diverses (bruit gaussien sur l'espace d'obs réel, +
     option : observations dérivées des features réelles si disponibles).
  3. Extraire de la POLITIQUE : mu (moyenne) et log_std PAR DIMENSION d'action,
     en particulier dim 3 (sl) et dim 4 (tp).
  4. Statistiques : moyenne, écart-type de mu, log_std→std, histogramme, entropie.
  5. Mapping env : simuler tp_raw -> tp_pct via les bornes profil pour mesurer
     l'AMPLITUDE EFFECTIVE du TP après post-processing (détection cas B).
  6. (si CSV paper dispo) corrélation tp_pct ↔ PnL, taux d'atteinte TP.
  7. Verdict A/B/C automatique.

Aucune écriture dans l'environnement, aucun trade, aucun entraînement.
"""

from __future__ import annotations

import os
import sys
import glob
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(str(ROOT))

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

# Bornes env (single source of truth, multi_asset_chunked_env l.7031-7036).
PROFILE_BOUNDS = {
    "scalper":  {"sl": (0.020, 0.030), "tp": (0.040, 0.060)},
    "intraday": {"sl": (0.040, 0.060), "tp": (0.080, 0.120)},
    "swing":    {"sl": (0.070, 0.100), "tp": (0.140, 0.200)},
    "position": {"sl": (0.150, 0.200), "tp": (0.300, 0.400)},
}
TP_FEE_FLOOR = 0.006  # l.7040
RR_MIN = 1.5          # l.7049


def find_checkpoint() -> str:
    cands = sorted(glob.glob("checkpoints/*FIXED*.zip")) or \
        sorted(glob.glob("checkpoints/*.zip"))
    if not cands:
        raise SystemExit("Aucun checkpoint trouvé dans checkpoints/")
    # préférer le 500k FIXED si présent
    for c in cands:
        if "500k_FIXED" in c:
            return c
    return cands[-1]


def sample_observations(model, n: int, seed: int = 0) -> list[dict]:
    """N observations variées (bruit gaussien réaliste mean≈0, std≈1)."""
    rng = np.random.default_rng(seed)
    obs_list = []
    for _ in range(n):
        o = {}
        for k, space in model.observation_space.spaces.items():
            # std=1 simule des observations normalisées (comme à l'entraînement).
            o[k] = rng.normal(0.0, 1.0, size=space.shape).astype(np.float32)
        obs_list.append(o)
    return obs_list


def policy_mu_logstd(model, obs_batch: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extrait mu (moyenne d'action) et log_std PAR observation, PAR dimension.

    Renvoie (MU [N, A], LOGSTD [N, A]). On lit directement la distribution de la
    politique (avant échantillonnage) pour mesurer la DÉCISION, pas le tirage.
    """
    policy = model.policy
    policy.set_training_mode(False)
    mus, logstds = [], []
    with torch.no_grad():
        for obs in obs_batch:
            obs_t, _ = policy.obs_to_tensor(obs)
            dist = policy.get_distribution(obs_t)
            d = dist.distribution  # torch Normal (Diagonal Gaussian)
            mu = d.mean.cpu().numpy().flatten()
            if hasattr(d, "scale"):
                std = d.scale.cpu().numpy().flatten()
                ls = np.log(np.clip(std, 1e-9, None))
            else:
                ls = np.full_like(mu, np.nan)
            mus.append(mu)
            logstds.append(ls)
    return np.array(mus), np.array(logstds)


def tp_raw_to_pct(tp_raw: float, sl_raw: float, profile: str) -> float:
    """Reproduit le mapping env tp_raw -> tp_pct (détection cas B)."""
    b = PROFILE_BOUNDS.get(profile, PROFILE_BOUNDS["intraday"])
    sl_lo, sl_hi = b["sl"]
    tp_lo, tp_hi = b["tp"]
    tp_lo = max(tp_lo, TP_FEE_FLOOR)
    sl_pct = float(np.clip(sl_lo + (sl_raw + 1) / 2 * (sl_hi - sl_lo), sl_lo, sl_hi))
    tp_pct = float(np.clip(tp_lo + (tp_raw + 1) / 2 * (tp_hi - tp_lo), tp_lo, tp_hi))
    if tp_pct < sl_pct * RR_MIN:
        tp_pct = float(min(sl_pct * RR_MIN, tp_hi))
    return tp_pct


def hist_str(x: np.ndarray, bins: int = 10, lo: float = -1, hi: float = 1) -> str:
    h, edges = np.histogram(x, bins=bins, range=(lo, hi))
    total = max(h.sum(), 1)
    out = []
    for i in range(bins):
        frac = h[i] / total
        bar = "█" * int(frac * 40)
        out.append(f"  [{edges[i]:+.2f},{edges[i+1]:+.2f}) {frac*100:5.1f}% {bar}")
    return "\n".join(out)


def gaussian_entropy(log_std: float) -> float:
    """Entropie d'une gaussienne 1D = 0.5*log(2*pi*e) + log_std."""
    return 0.5 * math.log(2 * math.pi * math.e) + log_std


def parse_paper_csvs() -> dict:
    """TP-hit rate + corrélation tp↔pnl depuis les CSV paper (si dispo)."""
    import csv
    rows = []
    for f in glob.glob("logs/paper/**/trades_paper_*.csv", recursive=True):
        try:
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    rows.append(r)
        except Exception:
            continue
    if not rows:
        return {"n": 0}
    reasons = {}
    pnls, tps = [], []
    for r in rows:
        reason = (r.get("reason") or r.get("close_reason") or "").upper()
        reasons[reason] = reasons.get(reason, 0) + 1
        try:
            pnls.append(float(r.get("pnl", r.get("pnl_pct", "nan"))))
        except Exception:
            pass
        try:
            tps.append(float(r.get("tp", r.get("take_profit_pct", "nan"))))
        except Exception:
            pass
    out = {"n": len(rows), "reasons": reasons}
    n_tp = reasons.get("TP", 0) + reasons.get("TAKE_PROFIT", 0)
    out["tp_hit_rate"] = n_tp / max(len(rows), 1)
    pa, ta = np.array(pnls, dtype=float), np.array(tps, dtype=float)
    mask = ~(np.isnan(pa) if pa.size else np.array([])) 
    if pa.size and ta.size and pa.size == ta.size:
        m = ~np.isnan(pa) & ~np.isnan(ta)
        if m.sum() > 2 and np.std(ta[m]) > 1e-9:
            out["corr_tp_pnl"] = float(np.corrcoef(tp := ta[m], pa[m])[0, 1])
    return out


def main():
    n = int(os.environ.get("AUDIT_N", "5000"))
    profile = os.environ.get("AUDIT_PROFILE", "intraday")
    cp = find_checkpoint()
    print("=" * 72)
    print("  A7 — AUDIT TÊTE TP (read-only)")
    print("=" * 72)
    print(f"Checkpoint : {cp}")
    print(f"Profil     : {profile}   |  N obs : {n}")

    model = PPO.load(cp, device="cpu")
    A = model.action_space.shape[0]
    print(f"Action dim : {A}  (attendu 5 = [direction, size, tf, sl, tp])")

    obs_batch = sample_observations(model, n)
    MU, LOGSTD = policy_mu_logstd(model, obs_batch)

    names = ["direction", "size", "tf", "sl", "tp"][:A]
    print("\n── STATISTIQUES PAR DIMENSION D'ACTION (mu = décision, avant tirage) ──")
    print(f"{'dim':<10}{'mu_mean':>10}{'mu_std':>10}{'logstd':>10}{'std':>10}{'entropy':>10}")
    stats = {}
    for j in range(A):
        mu_mean = MU[:, j].mean()
        mu_std = MU[:, j].std()           # VARIANCE DE LA DÉCISION (clé cas A)
        ls = np.nanmean(LOGSTD[:, j])
        std = math.exp(ls) if not math.isnan(ls) else float("nan")
        ent = gaussian_entropy(ls) if not math.isnan(ls) else float("nan")
        stats[names[j]] = dict(mu_mean=mu_mean, mu_std=mu_std, logstd=ls,
                               std=std, entropy=ent)
        print(f"{names[j]:<10}{mu_mean:>10.4f}{mu_std:>10.4f}{ls:>10.4f}"
              f"{std:>10.4f}{ent:>10.4f}")

    tp_i, sl_i = (4, 3) if A >= 5 else (A - 1, A - 2)
    print(f"\n── HISTOGRAMME mu_tp (dim {tp_i}) sur {n} obs ──")
    print(hist_str(MU[:, tp_i]))

    # Mapping env : amplitude EFFECTIVE du TP (détection cas B).
    print(f"\n── IMPACT EFFECTIF APRÈS MAPPING ENV (profil={profile}) ──")
    tp_pcts = np.array([
        tp_raw_to_pct(MU[k, tp_i], MU[k, sl_i], profile) for k in range(n)
    ])
    b = PROFILE_BOUNDS.get(profile, PROFILE_BOUNDS["intraday"])
    tp_lo_eff = max(b["tp"][0], TP_FEE_FLOOR)
    tp_hi_eff = b["tp"][1]
    print(f"  Bande TP autorisée : [{tp_lo_eff:.3f}, {tp_hi_eff:.3f}]  "
          f"(spread = {tp_hi_eff - tp_lo_eff:.3f})")
    print(f"  tp_pct effectif    : min={tp_pcts.min():.4f}  "
          f"max={tp_pcts.max():.4f}  mean={tp_pcts.mean():.4f}  std={tp_pcts.std():.4f}")
    span_used = (tp_pcts.max() - tp_pcts.min())
    span_frac = span_used / max(tp_hi_eff - tp_lo_eff, 1e-9)
    print(f"  Fraction de bande RÉELLEMENT utilisée : {span_frac*100:.1f}%")
    # combien clampés à la borne basse / écrasés par la règle R/R
    at_floor = float(np.mean(np.isclose(tp_pcts, tp_lo_eff, atol=1e-4)))
    print(f"  % au plancher (tp=tp_lo)              : {at_floor*100:.1f}%")

    # CSV paper
    print("\n── DONNÉES PAPER TRADING (read-only) ──")
    paper = parse_paper_csvs()
    if paper.get("n", 0) == 0:
        print("  (aucun CSV de trades exploitable)")
    else:
        print(f"  trades loggés      : {paper['n']}")
        print(f"  raisons de clôture : {paper.get('reasons')}")
        print(f"  taux atteinte TP   : {paper.get('tp_hit_rate', 0)*100:.1f}%")
        if "corr_tp_pnl" in paper:
            print(f"  corr(tp, pnl)      : {paper['corr_tp_pnl']:+.3f}")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  VERDICT A7")
    print("=" * 72)
    tp_mu_std = stats[names[tp_i]]["mu_std"]
    tp_std = stats[names[tp_i]]["std"]
    verdict = []
    # Cas A : décision quasi constante.
    dead = tp_mu_std < 0.02 and (math.isnan(tp_std) or tp_std < 0.05)
    # Cas B : la tête varie mais l'env écrase (bande étroite + peu utilisée).
    crushed = (span_frac < 0.40) or (at_floor > 0.5)
    if dead:
        verdict.append("CAS A (tête TP MORTE) : mu_tp quasi constant + std faible. "
                       "Aucun reward ne corrigera → problème d'architecture/tête.")
    if crushed and not dead:
        verdict.append("CAS B (TP ÉCRASÉ EN AVAL) : la tête varie mais le mapping env "
                       f"n'utilise que {span_frac*100:.0f}% de la bande "
                       f"(/{at_floor*100:.0f}% au plancher). Le reward est innocent ; "
                       "élargir la bande / revoir la règle R/R.")
    if not dead and not crushed:
        verdict.append("CAS C (SIGNAL FAIBLE) : la tête TP varie et l'amplitude passe. "
                       "Le reward-service guidé par le futur DEVRAIT aider "
                       "(récompenser TP cohérent avec le MFE futur).")
    for v in verdict:
        print("  • " + v)
    print("\nNote : verdict basé sur bruit gaussien d'obs. À confirmer sur obs réelles\n"
          "       si le pipeline LiveStateBuilder est disponible (AUDIT_REAL=1).")


if __name__ == "__main__":
    main()
