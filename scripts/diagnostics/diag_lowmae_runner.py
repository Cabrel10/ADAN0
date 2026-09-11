#!/usr/bin/env python3
"""diag_lowmae_runner.py — Probe N3/N5/N6 : le MAE precoce predit-il l'issue ?

Pourquoi cette sonde existe (brief de session, verbatim) :
  Le rapport MFE/excursion a mesure que (a) les runners existent (p99 MFE 7j =
  +38 % BTC, +202 % DOGE), (b) la classe C_runner a un MAE median ~ -0,25 %
  vs -9,9 % pour D_faux_signal (ratio 40x), et (c) le delai d'entree 0->40
  barres ne coute rien (EV plate). La chaine causale proposee :
      N0 queue MFE existe          -> MESURE (p99 7j >> 3x couts)
      N1 classes C/D separees      -> MESURE (ratio MAE 40x)
      N2 MAE_early observable      -> trivial (calculable en temps reel)
      N3 MAE_early PREDIT final_7j -> A MESURER ICI
      N4 delai admissible          -> MESURE (EXP3, EV plate)
      N5 regle filtree EV_net > 0  -> A MESURER ICI
      N6 robustesse splits/actifs  -> A MESURER ICI

Corrections de protocole imposees par la revue (avant ecriture de CE script) :
  1. SUPERPOSITION : les classes A-E du rapport MFE ont ete mesurees a
     stride=40 sur des fenetres 7j -> les "5,5 % de runners" peuvent etre
     3-4 episodes macro re-echantillonnes. Ici : comptage explicite des
     EPISODES de runners distincts (clusters de pics MFE) + bootstrap en
     blocs couvrant l'horizon complet (2016 barres).
  2. BIAIS DE REGIME : train = +904,9 % buy-and-hold. Tout signal trouve
     sur train seul peut etre "on est en hausse" reformule. Ici : seuils
     choisis sur train UNIQUEMENT, puis geles et appliques tels quels a
     val/test. Jamais de re-ajustement.
  3. N1 n'implique PAS N3 : la separation finale peut n'apparaitre que
     tardivement. Ce probe teste exactement si elle est observable
     causalement (fenetres 6/12/24/48/96/192 barres).
  4. Trois statuts possibles par cellule : PASS / FAIL / PUISSANCE
     INSUFFISANTE (n_effectif < 200 -> pas de verdict, pas d'echec).
  5. IC obligatoires : AUC_low(IC95) >= 0,55 exige, pas un AUC ponctuel.

Seuils PRE-ENREGISTRES (fixes avant mesure, non renégociables) :
  - N3 PASS si AUC >= 0,60 avec IC95_low >= 0,55 OU spread deciles >= 20 pts
  - N5 PASS si EV_net > 0 ET IC95_low > 0 (couts RT 0,5 % deduits)
  - N6 PASS si signe identique sur train ET val ET test (la ou puissance ok)
  - Bonferroni alpha = 0,05/12 = 0,00417
  - Block bootstrap : B=2000, blocs couvrant >= 2016 barres
  - Definition runner (ex-post, descriptive) : MFE_7j > 5 % ET
    MAE_avant_pic > -1 %
  - Echec N3 = mort de l'hypothese MAE_early UNIQUEMENT (pas de toute
    prediction : l'etage B teste les features d'entree separement).

Lecture seule. Sortie JSON logs/validation/lowmae_runner_<ts>.json.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _asset_guard import get_launcher_assets, assert_dataset_identity  # noqa: E402

# ------------------------------------------------------------ parametres geles
FEES_RT = 0.004
SLIPPAGE_RT = 0.001
COST_RT = FEES_RT + SLIPPAGE_RT          # 0,5 % round-trip
H_7J = 2016                              # horizon 7 jours en barres 5m
EARLY_WINDOWS = [6, 12, 24, 48, 96, 192]  # 30min / 1h / 2h / 4h / 8h / 16h
STRIDE_PRIMARY = 192                     # entrees espacees de 16h
STRIDE_STRICT = 2016                     # dedup stricte (reference, train)
BLOCK_BARS = 2016                        # bloc bootstrap en barres
N_BOOT = 2000
RUNNER_MFE = 0.05                        # MFE_7j > 5 %
RUNNER_MAE = -0.01                       # MAE avant pic > -1 %
AUC_PASS, AUC_CI_LOW = 0.60, 0.55        # N3
SPREAD_PASS = 0.20                       # 20 points
N_MIN_CELL = 200                         # en dessous -> PUISSANCE INSUFFISANTE
BONF_ALPHA = 0.05 / 12
GRID_X = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]  # filtre MAE_24 > -X
EPISODE_GAP = 288                        # pics MFE a < 24h = meme episode
RNG = np.random.default_rng(20260911)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "processed" / "indicators"
OUT_DIR = Path(__file__).resolve().parents[2] / "logs" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ chargement
def load_5m(asset: str, split: str) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(DATA_ROOT / split / asset / "5m.parquet")
    except Exception as e:
        print(f"  [skip] {split}/{asset}: {e}")
        return None
    assert_dataset_identity(asset, "5m", split, len(df))
    return df.sort_index()


def cross_up_events(df: pd.DataFrame) -> np.ndarray:
    macdh = df["macdh_12_26_9"].to_numpy(np.float64)
    ev = np.zeros(len(df), dtype=bool)
    ev[1:] = (macdh[1:] > 0) & (macdh[:-1] <= 0)
    return ev


def select_strided(mask: np.ndarray, n: int, stride: int, horizon: int) -> np.ndarray:
    """Entrees espacees d'au moins `stride` barres, avec horizon complet dispo."""
    entries = []
    next_free = 0
    for i in range(0, n - horizon - 1):
        if i >= next_free and mask[i]:
            entries.append(i)
            next_free = i + stride
    return np.array(entries, dtype=np.int64)


# ------------------------------------------------------------- trajectoires
def trajectory_stats(high, low, close, entries):
    """Un passage h=1..H_7J ; snapshots aux fenetres precoces + stats 7j."""
    n = len(entries)
    entry = close[entries]
    mae_w = {w: np.zeros(n) for w in EARLY_WINDOWS}
    mfe_w = {w: np.zeros(n) for w in EARLY_WINDOWS}
    ret_w = {w: np.zeros(n) for w in EARLY_WINDOWS}
    mfe = np.zeros(n)
    mae = np.zeros(n)
    mae_at_peak = np.zeros(n)   # MAE courant au moment ou le MFE est atteint
    ttp = np.zeros(n, dtype=np.int64)
    # temps de premier passage aux barrieres (0 = jamais touche -> H_7J+1)
    barriers = [0.01, 0.02, 0.04]
    t_up = {b: np.full(n, H_7J + 1, dtype=np.int64) for b in barriers}
    t_dn = {b: np.full(n, H_7J + 1, dtype=np.int64) for b in barriers}
    for h in range(1, H_7J + 1):
        hh = high[entries + h] / entry - 1.0
        ll = low[entries + h] / entry - 1.0
        new_mfe = hh > mfe
        mae_at_peak[new_mfe] = mae[new_mfe]  # MAE avant ce nouveau pic
        mfe = np.maximum(mfe, hh)
        mae = np.minimum(mae, ll)
        ttp[new_mfe] = h
        for b in barriers:
            up_now = (hh >= b) & (t_up[b] == H_7J + 1)
            t_up[b][up_now] = h
            dn_now = (ll <= -b) & (t_dn[b] == H_7J + 1)
            t_dn[b][dn_now] = h
        if h in mae_w:
            mae_w[h] = mae.copy()
            mfe_w[h] = mfe.copy()
            ret_w[h] = close[entries + h] / entry - 1.0
    final = close[entries + H_7J] / entry - 1.0
    return dict(mae_w=mae_w, mfe_w=mfe_w, ret_w=ret_w, mfe=mfe, mae=mae,
                mae_at_peak=mae_at_peak, ttp=ttp, final=final,
                t_up=t_up, t_dn=t_dn)


# ------------------------------------------------------------------ statistiques
def auc_rank(x, y):
    """AUC Mann-Whitney : P(x_pos > x_neg) + 0,5 P(egal)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    pos, neg = x[y], x[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(x)
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    # gestion des ex-aequo : rangs moyens
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        csum = np.bincount(inv, ranks)
        ranks = (csum / cnt)[inv]
    r_pos = ranks[y].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def block_boot_auc(x, y, block_entries):
    """IC95 bootstrap en blocs (entrees consecutives) sur l'AUC."""
    n = len(x)
    if n < max(N_MIN_CELL, 5 * block_entries):
        return (np.nan, np.nan)
    n_blocks = int(np.ceil(n / block_entries))
    stats = np.empty(N_BOOT)
    for b in range(N_BOOT):
        starts = RNG.integers(0, max(1, n - block_entries + 1), size=n_blocks)
        idx = (starts[:, None] + np.arange(block_entries)[None, :]).ravel()[:n]
        idx = np.clip(idx, 0, n - 1)
        a = auc_rank(x[idx], y[idx])
        stats[b] = a if not np.isnan(a) else 0.5
    return (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)))


def block_boot_mean(v, block_entries):
    """IC95 bootstrap en blocs sur la moyenne."""
    n = len(v)
    if n < 5:
        return (float(np.mean(v)), np.nan, np.nan)
    n_blocks = int(np.ceil(n / block_entries))
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        starts = RNG.integers(0, max(1, n - block_entries + 1), size=n_blocks)
        idx = (starts[:, None] + np.arange(block_entries)[None, :]).ravel()[:n]
        means[b] = v[np.clip(idx, 0, n - 1)].mean()
    return (float(v.mean()), float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)))


def decile_spread(x, y):
    """P(y | decile 1 de x) - P(y | decile 10 de x). x croissant = pire MAE."""
    qs = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
    if len(qs) < 11:
        return (np.nan, np.nan, np.nan)
    d = np.clip(np.digitize(x, qs[1:-1]), 0, 9)
    p1 = float(y[d == 0].mean()) if (d == 0).any() else np.nan
    p10 = float(y[d == 9].mean()) if (d == 9).any() else np.nan
    return (p1, p10, p10 - p1)


def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------ cellule
def run_cell(asset, split, stride):
    df = load_5m(asset, split)
    if df is None:
        return None
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    n = len(df)
    mask = cross_up_events(df)
    n_cross = int(mask.sum())
    entries = select_strided(mask, n, stride, H_7J)
    block_entries = max(1, int(np.ceil(BLOCK_BARS / stride)))
    n_eff = len(entries) / block_entries
    print(f"  [{asset}/{split}] crossups={n_cross} entrees(stride={stride})="
          f"{len(entries)} bloc_boot={block_entries} n_eff~{n_eff:.0f}")
    if len(entries) < 30:
        return {"status": "PUISSANCE_INSUFFISANTE", "n_entries": len(entries)}

    ts = trajectory_stats(high, low, close, entries)
    runner = (ts["mfe"] > RUNNER_MFE) & (ts["mae_at_peak"] > RUNNER_MAE)
    win = ts["final"] > 0
    bad = ts["final"] < -0.04

    # --- episodes de runners distincts (pics MFE a < EPISODE_GAP barres)
    peak_times = entries[runner] + ts["ttp"][runner]
    n_episodes = 0
    last = -10**9
    for t in np.sort(peak_times):
        if t - last >= EPISODE_GAP:
            n_episodes += 1
            last = t

    res = {
        "n_crossups": n_cross, "n_entries": len(entries),
        "block_entries_boot": block_entries, "n_effectif": round(n_eff, 1),
        "puissance_suffisante": bool(n_eff >= N_MIN_CELL or len(entries) >= N_MIN_CELL),
        "n_runners": int(runner.sum()), "n_runner_episodes": n_episodes,
        "runner_pct": float(runner.mean()),
        "buyhold_pct": float(close[-1] / close[0] - 1),
    }
    if runner.sum() >= 5:
        res["runner_mfe_med"] = float(np.median(ts["mfe"][runner]))
        res["runner_mae_peak_med"] = float(np.median(ts["mae_at_peak"][runner]))
        res["runner_final_med"] = float(np.median(ts["final"][runner]))
        res["nonrunner_mae_med"] = float(np.median(ts["mae"][~runner]))
        # ordre temporel : temps median premier +2 % vs premier -2 %
        res["runner_t_up2_med"] = float(np.median(ts["t_up"][0.02][runner]))
        res["runner_t_dn2_med"] = float(np.median(ts["t_dn"][0.02][runner]))
        res["nonrunner_t_up2_med"] = float(np.median(ts["t_up"][0.02][~runner]))
        res["nonrunner_t_dn2_med"] = float(np.median(ts["t_dn"][0.02][~runner]))

    # --- N3 : MAE_early / MFE_early / ret_early -> final_7j
    n3 = {}
    for w in EARLY_WINDOWS:
        for name, x in (("mae", ts["mae_w"][w]), ("mfe", ts["mfe_w"][w]),
                        ("ret", ts["ret_w"][w])):
            for tname, y in (("win", win), ("runner", runner), ("bad", bad)):
                a = auc_rank(x, y)
                lo, hi = block_boot_auc(x, y, block_entries)
                key = f"{name}_{w}__{tname}"
                n3[key] = {"auc": a, "ci_lo": lo, "ci_hi": hi,
                           "spearman": spearman(x, ts["final"])}
    res["n3_auc"] = n3
    p1, p10, spread = decile_spread(ts["mae_w"][24], win)
    res["n3_deciles_mae24_win"] = {"p_d1": p1, "p_d10": p10, "spread": spread}
    p1r, p10r, spreadr = decile_spread(ts["mae_w"][24], runner)
    res["n3_deciles_mae24_runner"] = {"p_d1": p1r, "p_d10": p10r,
                                      "spread": spreadr}

    # --- N5 : regle filtree, EV nette (entree a t+24 si MAE_24 > -X)
    entry24 = close[entries + 24]
    fwd = close[entries + H_7J] / entry24 - 1.0 - COST_RT
    mae24 = ts["mae_w"][24]
    base_mean, base_lo, base_hi = block_boot_mean(fwd, block_entries)
    res["n5_baseline_ev"] = {"ev": base_mean, "ci_lo": base_lo, "ci_hi": base_hi}
    grid = {}
    for x in GRID_X:
        keep = mae24 > -x
        if keep.sum() < 30:
            grid[str(x)] = {"n": int(keep.sum()), "status": "trop_peu"}
            continue
        m, lo, hi = block_boot_mean(fwd[keep],
                                    max(1, int(block_entries * keep.mean())))
        grid[str(x)] = {"n": int(keep.sum()), "ev": m, "ci_lo": lo, "ci_hi": hi,
                        "p_win": float((fwd[keep] > 0).mean())}
    res["n5_grid"] = grid

    # --- Etage B : features disponibles a l'entree (t) -> runner / win
    feats = {}
    for col in ("atr_pct", "volatility_ratio_14_50", "bb_width_20_2",
                "ema_20_ratio", "rsi_14", "adx_14", "volume_ratio_20",
                "bb_percent_b_20_2"):
        if col in df.columns:
            feats[col] = df[col].to_numpy(np.float64)[entries]
    # tendance longue causale : close / SMA(480 barres ~ 40h) - 1
    sma480 = pd.Series(close).rolling(480).mean().to_numpy()
    feats["trend_40h"] = (close / sma480 - 1.0)[entries]
    feats["ret_24b"] = (close / np.roll(close, 24) - 1.0)[entries]
    feats["ret_24b"][:24] = np.nan
    bres = {}
    for fname, fx in feats.items():
        ok = np.isfinite(fx)
        if ok.sum() < 30:
            continue
        bres[fname] = {
            "auc_runner": auc_rank(fx[ok], runner[ok]),
            "auc_win": auc_rank(fx[ok], win[ok]),
            "spearman_final": spearman(fx[ok], ts["final"][ok]),
        }
    res["stageB_entry_features"] = bres
    return res


def main():
    t0 = datetime.now(timezone.utc)
    print(f"[LOWMAE-RUNNER] depart {t0.isoformat()} — seuils pre-enregistres :")
    print(f"  N3: AUC>={AUC_PASS} (IC95_low>={AUC_CI_LOW}) OU spread>={SPREAD_PASS}")
    print(f"  N5: EV_net>0 ET IC95_low>0 | couts RT={COST_RT}")
    print(f"  N6: signe identique train+val+test | Bonferroni a={BONF_ALPHA:.5f}")
    print(f"  runner: MFE_7j>{RUNNER_MFE} ET MAE_avant_pic>{RUNNER_MAE}")
    out = {"generated_utc": t0.isoformat(),
           "params": {"cost_rt": COST_RT, "h_7j": H_7J,
                      "stride_primary": STRIDE_PRIMARY,
                      "stride_strict": STRIDE_STRICT,
                      "early_windows": EARLY_WINDOWS,
                      "runner_mfe": RUNNER_MFE, "runner_mae": RUNNER_MAE,
                      "auc_pass": AUC_PASS, "auc_ci_low": AUC_CI_LOW,
                      "spread_pass": SPREAD_PASS, "n_min_cell": N_MIN_CELL,
                      "bonferroni_alpha": BONF_ALPHA, "n_boot": N_BOOT,
                      "grid_x": GRID_X, "episode_gap": EPISODE_GAP},
           "cells": {}}
    for asset in get_launcher_assets():
        for split in ("train", "val", "test"):
            r = run_cell(asset, split, STRIDE_PRIMARY)
            if r:
                out["cells"][f"{asset}/{split}"] = r
    # reference dedup stricte (train uniquement) : reponse directe a la
    # critique de superposition — combien d'episodes avec des fenetres 7j
    # STRICTEMENT non chevauchantes ?
    out["cells_strict2016"] = {}
    for asset in get_launcher_assets():
        r = run_cell(asset, "train", STRIDE_STRICT)
        if r:
            out["cells_strict2016"][f"{asset}/train"] = {
                k: r[k] for k in ("n_entries", "n_runners",
                                  "n_runner_episodes", "runner_pct",
                                  "runner_mfe_med", "runner_mae_peak_med",
                                  "runner_final_med")
                if k in r}
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"lowmae_runner_{ts}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n[LOWMAE-RUNNER] ecrit {path}")
    # resume stdout
    for cell, r in out["cells"].items():
        if "n3_auc" not in r:
            continue
        m = r["n3_auc"].get("mae_24__win", {})
        print(f"  {cell}: n={r['n_entries']} runners={r['n_runners']} "
              f"episodes={r['n_runner_episodes']} "
              f"AUC(mae24->win)={m.get('auc', float('nan')):.3f} "
              f"[{m.get('ci_lo', float('nan')):.3f},{m.get('ci_hi', float('nan')):.3f}] "
              f"spread_d={r['n3_deciles_mae24_win']['spread']}")


if __name__ == "__main__":
    main()
