#!/usr/bin/env python3
"""A7 (v2, CORRIGÉ) — Audit READ-ONLY de la tête d'action TP (et SL) du PPO.

CORRECTIONS suite à la revue utilisateur (juin 2026) :

  BUG-1 (mu pré-squash) : la v1 lisait `dist.distribution.mean`. Or la politique
        utilise gSDE (use_sde=True, squash_output=False). `dist.mean` est la moyenne
        latente de la gaussienne, NON clippée dans [-1,1]. La v2 mesure désormais
        l'ACTION RÉELLE via `model.predict(deterministic=True)` (= action post-clip
        appliquée à l'env) ET via tirages stochastiques.
  BUG-2 (obs bruit) : la v1 testait sur du bruit gaussien std=1 → obs irréalistes,
        gSDE state-dependent produisait des mu extrêmes (mu_std=4.36 = artefact).
        La v2 utilise des OBSERVATIONS RÉELLES issues du Parquet `val` via StateBuilder
        (scalers ajustés + verrouillés, exactement comme diagnose_obs.py).
  CLARIF log_std : le paramètre brut `policy.log_std` a shape (64,5) — c'est la
        matrice de bruit gSDE (64 features latentes × 5 actions), PAS un log_std par
        dimension. log_std[*,4]≈-2.01 (vu plus tôt comme "-2.04") ≠ écart-type EFFECTIF
        de l'action (state-dependent, mesuré ici directement sur les tirages).

Verdict A/B/C basé sur l'action RÉELLE, sur OBS RÉELLES.
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

# Bornes env (single source of truth, multi_asset_chunked_env l.7031-7036).
PROFILE_BOUNDS = {
    "scalper":  {"sl": (0.020, 0.030), "tp": (0.040, 0.060)},
    "intraday": {"sl": (0.040, 0.060), "tp": (0.080, 0.120)},
    "swing":    {"sl": (0.070, 0.100), "tp": (0.140, 0.200)},
    "position": {"sl": (0.150, 0.200), "tp": (0.300, 0.400)},
}
TP_FEE_FLOOR = 0.006
RR_MIN = 1.5


def find_checkpoint() -> str:
    cands = sorted(glob.glob("checkpoints/*FIXED*.zip")) or \
        sorted(glob.glob("checkpoints/*.zip"))
    for c in cands:
        if "500k_FIXED" in c:
            return c
    return cands[-1]


# ── OBSERVATIONS RÉELLES via StateBuilder + Parquet val ──────────────────────
def build_real_observations(n: int, seed: int = 0) -> list[dict]:
    """Génère N observations RÉELLES en balayant des fenêtres du Parquet val.

    Réplique diagnose_obs.py : ajuste les scalers sur le Parquet puis les verrouille,
    et appelle StateBuilder.build_observation(current_idx) pour des indices variés.
    Retourne [] si les données réelles ne sont pas disponibles (fallback bruit).
    """
    try:
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        from adan_trading_bot.trading.live_state_builder import (
            TRAIN_COLUMNS, OBS_WINDOW,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [obs réelles] import StateBuilder échoué: {e}")
        return []

    val_dir = ROOT / "data" / "processed" / "indicators" / "val" / "BTCUSDT"
    data_dict = {}
    for tf in ["5m", "1h", "4h"]:
        p = val_dir / f"{tf}.parquet"
        if p.exists():
            data_dict[tf] = pd.read_parquet(p)
    if len(data_dict) < 3:
        print(f"  [obs réelles] Parquet val incomplet ({list(data_dict)}) → fallback bruit")
        return []

    try:
        sb = StateBuilder(
            features_config=TRAIN_COLUMNS,
            window_sizes={tf: OBS_WINDOW for tf in ["5m", "1h", "4h"]},
            include_portfolio_state=True,
            normalize=True,
        )
        sb.fit_scalers({"BTCUSDT": data_dict})
        sb.scalers_loaded_from_training = True
    except Exception as e:  # noqa: BLE001
        print(f"  [obs réelles] init/fit StateBuilder échoué: {e} → fallback bruit")
        return []

    nested = {"BTCUSDT": data_dict}
    n5 = len(data_dict["5m"])
    lo, hi = 300, n5 - 2  # warmup 300 bars, garder une marge
    if hi <= lo:
        print("  [obs réelles] pas assez de barres → fallback bruit")
        return []

    rng = np.random.default_rng(seed)
    idxs = rng.integers(lo, hi, size=min(n, hi - lo))
    obs_list = []
    for ci in idxs:
        try:
            o = sb.build_observation(current_idx=int(ci), data=nested)
        except Exception:
            continue
        # portfolio_state / context_vector réalistes (flat pos initiale)
        if "portfolio_state" not in o:
            ps = np.zeros(20, dtype=np.float32); ps[0] = 20.5; ps[1] = 20.5
            o["portfolio_state"] = ps
        if "context_vector" not in o:
            o["context_vector"] = np.full(17, 1.0 / 17.0, dtype=np.float32)
        obs_list.append({k: np.asarray(v, dtype=np.float32) for k, v in o.items()})
    print(f"  [obs réelles] {len(obs_list)} observations construites depuis Parquet val")
    return obs_list


def sample_noise_observations(model, n: int, seed: int = 0) -> list[dict]:
    """Fallback : bruit gaussien std=1 (IRRÉALISTE — à n'utiliser qu'en secours)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        o = {k: rng.normal(0, 1, size=sp.shape).astype(np.float32)
             for k, sp in model.observation_space.spaces.items()}
        out.append(o)
    return out


# ── ACTION RÉELLE (post-clip) ────────────────────────────────────────────────
def collect_actions(model, obs_batch: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (ACT_DET [N,A], ACT_STO [N,A]).

    ACT_DET = action déterministe (mu post-clip) = LA DÉCISION appliquée à l'env.
    ACT_STO = action stochastique (un tirage) — pour mesurer l'écart-type EFFECTIF
              (state-dependent gSDE), borné dans [-1,1].
    """
    model.policy.set_training_mode(False)
    dets, stos = [], []
    for obs in obs_batch:
        a_det, _ = model.predict(obs, deterministic=True)
        a_sto, _ = model.predict(obs, deterministic=False)
        dets.append(np.asarray(a_det, dtype=float).flatten())
        stos.append(np.asarray(a_sto, dtype=float).flatten())
    return np.array(dets), np.array(stos)


def tp_raw_to_pct(tp_raw: float, sl_raw: float, profile: str) -> float:
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
    return "\n".join(
        f"  [{edges[i]:+.2f},{edges[i+1]:+.2f}) {h[i]/total*100:5.1f}% "
        f"{'█' * int(h[i]/total*40)}"
        for i in range(bins)
    )


def parse_paper_csvs() -> dict:
    import csv
    rows = []
    for f in glob.glob("logs/paper/**/trades_paper_*.csv", recursive=True):
        try:
            with open(f) as fh:
                rows.extend(list(csv.DictReader(fh)))
        except Exception:
            continue
    if not rows:
        return {"n": 0}
    reasons, pnls, tps = {}, [], []
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
    pa, ta = np.array(pnls, float), np.array(tps, float)
    if pa.size and ta.size and pa.size == ta.size:
        m = ~np.isnan(pa) & ~np.isnan(ta)
        if m.sum() > 2 and np.std(ta[m]) > 1e-9:
            out["corr_tp_pnl"] = float(np.corrcoef(ta[m], pa[m])[0, 1])
    return out


def main():
    n = int(os.environ.get("AUDIT_N", "3000"))
    profile = os.environ.get("AUDIT_PROFILE", "intraday")
    cp = find_checkpoint()
    print("=" * 72)
    print("  A7 v2 — AUDIT TÊTE TP (read-only, action POST-clip, obs RÉELLES)")
    print("=" * 72)
    print(f"Checkpoint : {cp}")
    print(f"Profil     : {profile}   |  N demandé : {n}")

    model = PPO.load(cp, device="cpu")
    A = model.action_space.shape[0]
    print(f"Action dim : {A}  (attendu 5 = [direction, size, tf, sl, tp])")
    print(f"use_sde    : {getattr(model, 'use_sde', '?')}  |  "
          f"squash_output : {getattr(model.policy, 'squash_output', '?')}")

    # 1) Observations RÉELLES (priorité), sinon bruit (avec avertissement explicite)
    obs_batch = build_real_observations(n)
    obs_source = "RÉELLES (Parquet val)"
    if not obs_batch:
        obs_batch = sample_noise_observations(model, n)
        obs_source = "BRUIT gaussien (IRRÉALISTE — secours)"
    print(f"Source obs : {obs_source}   |  N effectif : {len(obs_batch)}")

    # 2) Actions RÉELLES (post-clip)
    ACT_DET, ACT_STO = collect_actions(model, obs_batch)
    names = ["direction", "size", "tf", "sl", "tp"][:A]

    print("\n── ACTION DÉTERMINISTE (la DÉCISION post-clip appliquée à l'env) ──")
    print(f"{'dim':<10}{'min':>9}{'max':>9}{'mean':>9}{'std':>9}"
          f"{'std_sto':>9}{'entropy*':>10}")
    stats = {}
    for j in range(A):
        col = ACT_DET[:, j]
        std_sto = ACT_STO[:, j].std()  # écart-type EFFECTIF state-dependent (post-clip)
        # entropie ~ d'une gaussienne tronquée approx via std effectif
        ent = (0.5 * math.log(2 * math.pi * math.e) + math.log(max(std_sto, 1e-9)))
        stats[names[j]] = dict(min=col.min(), max=col.max(), mean=col.mean(),
                               std=col.std(), std_sto=std_sto, entropy=ent)
        print(f"{names[j]:<10}{col.min():>9.4f}{col.max():>9.4f}{col.mean():>9.4f}"
              f"{col.std():>9.4f}{std_sto:>9.4f}{ent:>10.4f}")
    print("  (*entropy approx via std stochastique effectif, post-clip)")

    tp_i, sl_i = (4, 3) if A >= 5 else (A - 1, A - 2)
    print(f"\n── DEMANDE EXPLICITE UTILISATEUR : action_det[:, {tp_i}] (=tp) ──")
    tp = ACT_DET[:, tp_i]
    print(f"  min  = {tp.min():.6f}")
    print(f"  max  = {tp.max():.6f}")
    print(f"  mean = {tp.mean():.6f}")
    print(f"  std  = {tp.std():.6f}")
    print(f"  borné dans [-1,1] ? {tp.min() >= -1.0001 and tp.max() <= 1.0001}")

    print(f"\n── HISTOGRAMME action_det tp (dim {tp_i}) sur {len(obs_batch)} obs ──")
    print(hist_str(tp))

    # 3) Mapping env : amplitude EFFECTIVE du TP (détection cas B)
    print(f"\n── IMPACT EFFECTIF APRÈS MAPPING ENV (profil={profile}) ──")
    tp_pcts = np.array([tp_raw_to_pct(ACT_DET[k, tp_i], ACT_DET[k, sl_i], profile)
                        for k in range(len(obs_batch))])
    b = PROFILE_BOUNDS.get(profile, PROFILE_BOUNDS["intraday"])
    tp_lo_eff = max(b["tp"][0], TP_FEE_FLOOR); tp_hi_eff = b["tp"][1]
    print(f"  Bande TP autorisée : [{tp_lo_eff:.3f}, {tp_hi_eff:.3f}]  "
          f"(spread={tp_hi_eff - tp_lo_eff:.3f})")
    print(f"  tp_pct effectif    : min={tp_pcts.min():.4f}  max={tp_pcts.max():.4f}  "
          f"mean={tp_pcts.mean():.4f}  std={tp_pcts.std():.4f}")
    span_frac = (tp_pcts.max() - tp_pcts.min()) / max(tp_hi_eff - tp_lo_eff, 1e-9)
    at_floor = float(np.mean(np.isclose(tp_pcts, tp_lo_eff, atol=1e-4)))
    print(f"  Fraction de bande utilisée : {span_frac*100:.1f}%")
    print(f"  % au plancher (tp=tp_lo)   : {at_floor*100:.1f}%")

    # 4) CSV paper
    print("\n── DONNÉES PAPER TRADING (read-only) ──")
    paper = parse_paper_csvs()
    if paper.get("n", 0) == 0:
        print("  (aucun CSV de trades exploitable — voir _archive)")
    else:
        print(f"  trades loggés      : {paper['n']}")
        print(f"  raisons de clôture : {paper.get('reasons')}")
        print(f"  taux atteinte TP   : {paper.get('tp_hit_rate', 0)*100:.1f}%")
        if "corr_tp_pnl" in paper:
            print(f"  corr(tp, pnl)      : {paper['corr_tp_pnl']:+.3f}")

    # 5) VERDICT (sur action RÉELLE post-clip)
    print("\n" + "=" * 72)
    print("  VERDICT A7 v2  (base : action déterministe post-clip, obs réelles)")
    print("=" * 72)
    tp_std_det = stats[names[tp_i]]["std"]       # variance de la DÉCISION
    tp_std_sto = stats[names[tp_i]]["std_sto"]   # écart-type effectif (exploration)
    # Concentration au plafond : fraction des décisions dans le bin haut [+0.80,+1.0].
    ceil_frac = float(np.mean(tp >= 0.80))
    # tp_pct effectif au plafond de bande ?
    tp_at_band_ceil = float(np.mean(np.isclose(tp_pcts, tp_hi_eff, atol=2e-3)))
    print(f"  Concentration décision TP dans [+0.80,+1.00] : {ceil_frac*100:.1f}%")
    print(f"  tp_pct effectif AU PLAFOND de bande          : {tp_at_band_ceil*100:.1f}%")

    dead = (tp_std_det < 0.02) and (tp_std_sto < 0.05)
    saturated = (ceil_frac > 0.90) or (tp_at_band_ceil > 0.90)
    crushed = (span_frac < 0.40) or (at_floor > 0.5)
    verdict = []
    if dead:
        verdict.append("CAS A (tête TP MORTE) : décision TP quasi constante "
                       f"(std_det={tp_std_det:.4f}) ET exploration quasi nulle "
                       f"(std_sto={tp_std_sto:.4f}). Aucun reward ne corrigera.")
    elif saturated:
        verdict.append("CAS B' (TP SATURÉ AU PLAFOND) : "
                       f"{ceil_frac*100:.0f}% des décisions TP dans [+0.80,+1.00], "
                       f"tp_pct EFFECTIF au plafond dans {tp_at_band_ceil*100:.0f}% "
                       "des cas. La queue (excursions basses) gonfle artificiellement "
                       f"le span ({span_frac*100:.0f}%) mais NE reflète PAS la décision "
                       "typique. Le TP réel est FIXE au maximum de la bande → jamais "
                       "atteignable sur l'horizon → AGENT_CLOSE devient seule sortie. "
                       "Le reward guidé par le futur PEUT aider à DÉSATURER (récompenser "
                       "un TP cohérent avec le MFE futur < plafond), mais il faudra AUSSI "
                       "réveiller l'exploration (entropie/log_std init).")
    elif crushed:
        verdict.append("CAS B (TP ÉCRASÉ EN AVAL) : la décision varie mais le mapping "
                       f"env n'utilise que {span_frac*100:.0f}% de la bande "
                       f"({at_floor*100:.0f}% au plancher). Reward innocent.")
    else:
        verdict.append("CAS C (SIGNAL FAIBLE) : décision TP variée "
                       f"(std_det={tp_std_det:.4f}), amplitude passe le mapping. "
                       "Le reward guidé par le futur DEVRAIT aider — À CONDITION que "
                       "les CSV paper confirment AGENT_CLOSE >> TP (TP jamais atteint).")
    for v in verdict:
        print("  • " + v)
    print("\nNote méthodo : obs = " + obs_source + ". Action mesurée = post-clip "
          "(model.predict), PAS dist.mean pré-squash. log_std brut (64,5)=matrice gSDE, "
          "≠ std effectif (lu ici sur tirages stochastiques).")


if __name__ == "__main__":
    main()
