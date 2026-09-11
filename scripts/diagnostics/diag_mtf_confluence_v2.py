#!/usr/bin/env python3
"""diag_mtf_confluence_v2.py — methodologie durcie (round 4).

Succede aux rounds 1-3 :
  R1 : confluence MTF > aveugle (delta reproductible) mais amplitude << frais.
  R2 : SL/TP ATR-adaptatifs + trigger evenementiel -> meilleure cellule instable
       d'un split a l'autre -> bruit.
  R3 : aucune cellule EV_net>0 coherente sur train ET val ET test -> verdict
       NEGATIF documente (RAPPORT_MTF_ROBUSTNESS_R3.md).

Corrections methodologiques appliquees ici (cahier des charges utilisateur) :
  1. Confluence CONTINUE  c(t) = tanh( sum_k w_k * z_k(t) ), z_k standardises
     sur train ; poids w_k proportionnels a l'information mutuelle I(z_k ; MFE_H)
     estimee sur TRAIN UNIQUEMENT (signe = signe de la correlation).
  2. Premier passage bivarie G_H(x,y|c) avec censure a H ; temps de detention
     mesure -> EV normalisee par temps de capital immobilise.
  3. Edge Ratio (Sweeney) : separation MAE(gagnants) vs MAE(perdants) ->
     SL* = argmax_s [ F_loser(s) - F_winner(s) ] (Youden) ;
     TP* = quantile du MFE CONDITIONNE a la survie a SL*.
  4. Stabilite spatiale : une cellule n'est retenue que si le MIN de son
     voisinage 3x3 est > frais+slippage (plateau, pas crete isolee).
  5. Block bootstrap (blocs de trades consecutifs, longueur 5 ~ 2H en temps,
     B=2000) : edge valide seulement si CI95_low(EV_net) > 0.
  6. IC de Wilson sur P(TP)/P(SL) ; puissance pre-enregistree n>=200/cellule
     (les cellules sous-dimensionnees sont marquees UNDERPOWERED, pas "edge").
  7. Correction FDR Benjamini-Hochberg (q=0.05) sur TOUTES les cellules testees.
  8. Selection des parametres sur TRAIN, val = confirmation, test = gele.
  9. Validation croisee inter-actifs BTC <-> DOGE (seuils geles).
 10. Test de monotonie : E[MFE_H | decile(c)] croissant en c (regression +
     terme quadratique, p-value par permutation).
 11. Walk-forward multi-regimes : 3 tiers chronologiques du train, fit sur le
     passe, evaluation OOS sur le tiers suivant (2 fenetres) + val + test.

Garde-fou : tant que la chaine confluence -> G_H biaisée -> EV>0 stable n'est
pas demontree, AUCUNE modification du reward PPO.

Lecture seule. Sortie JSON dans logs/validation/ + resume stdout.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_mtf_confluence import load_aligned, DATA, OUT_DIR  # noqa: E402
from _asset_guard import get_launcher_assets  # noqa: E402

# ---------------------------------------------------------------- parametres
FEES_RT = 0.004            # round-trip 0.40 %
SLIPPAGE_RT = 0.001        # hypothese slippage round-trip 0.10 %
COST_RT = FEES_RT + SLIPPAGE_RT
H_LIST = [40, 80, 160, 288]          # barres 5m
K_SL = [0.5, 0.75, 1.0, 1.5, 2.0]    # SL = k_sl x ATR% entree
K_TP = [1.0, 1.5, 2.0, 3.0, 4.0]     # TP = k_tp x ATR% entree
N_BOOT = 2000
BLOCK = 5                            # blocs de trades consecutifs (~2H en temps)
N_MIN_POWER = 200                    # puissance pre-enregistree par cellule
N_MIN_ABS = 30                       # plancher absolu pour rapporter un chiffre
FDR_Q = 0.05
N_DECILES = 10
RNG = np.random.default_rng(42)

# Univers launcher UNIQUEMENT (correction contamination dataset : le run original
# apprenait les poids MI sur BTCUSDT = 7 991 barres ~28 j au lieu de
# BTCUSDT_BINANCE = 662 643 barres). Source unique = _asset_guard.
ASSET_CORE = list(get_launcher_assets())
SPLITS = ["train", "val", "test"]

# Signaux directionnels par TF (positif = haussier)
SIG_COL = {"5m": "macdh_12_26_9", "1h": "ema_50_ratio_1h", "4h": "ema_100_ratio_4h"}
SIG_OFFSET = {"5m": 0.0, "1h": 1.0, "4h": 1.0}


# ---------------------------------------------------------------- utilitaires
def wilson_ci(k, n, z=1.959964):
    """IC de Wilson sur une proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z / den) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def mutual_information(x, y, bins=10):
    """I(X;Y) en nats, discretisation par quantiles (appelée sur train only)."""
    qs = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    xb = np.clip(np.digitize(x, qs[1:-1]), 0, bins - 1)
    qs = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    yb = np.clip(np.digitize(y, qs[1:-1]), 0, bins - 1)
    joint = np.zeros((bins, bins))
    np.add.at(joint, (xb, yb), 1)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = joint * np.log(joint / (px @ py))
    return float(np.nansum(t))


def block_bootstrap_ci(pnl, n_boot=N_BOOT, block=BLOCK):
    """IC95 bootstrap en blocs sur la moyenne de P&L + p-value bilaterale vs 0."""
    n = len(pnl)
    if n < 5:
        return (float(np.mean(pnl)), np.nan, np.nan, np.nan)
    n_blocks = int(np.ceil(n / block))
    starts = RNG.integers(0, max(1, n - block + 1), size=(n_boot, n_blocks))
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = (starts[b][:, None] + np.arange(block)[None, :]).ravel()[:n]
        idx = np.clip(idx, 0, n - 1)
        means[b] = pnl[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    p_two = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return (float(pnl.mean()), float(lo), float(hi), float(min(1.0, p_two)))


def bh_fdr(pvals, q=FDR_Q):
    """Benjamini-Hochberg : retourne masque des rejets."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    rej = np.zeros(len(p), dtype=bool)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return rej
    order = idx[np.argsort(p[idx])]
    m = len(order)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        rej[order[: kmax + 1]] = True
    return rej


# ------------------------------------------------------------- coeur de calcul
def cross_up_events(df):
    """Trigger evenementiel 5m : croisement haussier du histogramme MACD."""
    macdh = df["macdh_12_26_9"].to_numpy(np.float64)
    ev = np.zeros(len(df), dtype=bool)
    ev[1:] = (macdh[1:] > 0) & (macdh[:-1] <= 0)
    return ev


def select_nonoverlapping(mask, n, horizon):
    entries = []
    next_free = 0
    for i in range(0, n - horizon - 1):
        if i >= next_free and mask[i]:
            entries.append(i)
            next_free = i + horizon
    return np.array(entries, dtype=np.int64)


def path_stats(high, low, close, entries, H):
    """MFE, MAE, rendement final, longueur de chemin, par trade."""
    entry = close[entries]
    mfe = np.full(len(entries), -np.inf)
    mae = np.full(len(entries), np.inf)
    path_len = np.zeros(len(entries))
    prev = entry.copy()
    for h in range(1, H + 1):
        mfe = np.maximum(mfe, high[entries + h] / entry - 1.0)
        mae = np.minimum(mae, low[entries + h] / entry - 1.0)
        cur = close[entries + h]
        path_len += np.abs(cur / prev - 1.0)
        prev = cur
    final = close[entries + H] / entry - 1.0
    return mfe, mae, final, path_len


def first_passage_trades(high, low, close, entries, H, sl, tp):
    """Premier passage pessimiste avec SL/TP par trade.

    Retourne (pnl, holding_time, hit_tp, hit_sl)."""
    n = len(entries)
    entry = close[entries]
    pnl = np.zeros(n)
    ht = np.full(n, H, dtype=np.int64)
    hit_tp = np.zeros(n, dtype=bool)
    hit_sl = np.zeros(n, dtype=bool)
    done = np.zeros(n, dtype=bool)
    for h in range(1, H + 1):
        hh = high[entries + h] / entry - 1.0
        ll = low[entries + h] / entry - 1.0
        both = (~done) & (hh >= tp) & (ll <= -sl)
        pnl[both] = -sl[both]
        ht[both] = h
        hit_sl |= both
        done |= both
        only_tp = (~done) & (hh >= tp)
        pnl[only_tp] = tp[only_tp]
        ht[only_tp] = h
        hit_tp |= only_tp
        done |= only_tp
        only_sl = (~done) & (ll <= -sl)
        pnl[only_sl] = -sl[only_sl]
        ht[only_sl] = h
        hit_sl |= only_sl
        done |= only_sl
    rest = ~done
    pnl[rest] = close[entries[rest] + H] / entry[rest] - 1.0
    return pnl, ht, hit_tp, hit_sl


def edge_ratio_geometry(mfe, mae, atr_entry, final):
    """Edge Ratio (Sweeney) + SL* par separation Youden + TP* conditionnel.

    Retourne dict avec ER, k_sl_star, k_tp_star, separation."""
    er = float(mfe.sum() / np.abs(mae[mae < 0]).sum()) if (mae < 0).any() else np.nan
    winners = final > 0
    # |MAE| en multiples d'ATR par trade
    mae_atr = np.where(atr_entry > 1e-12, -mae / atr_entry, np.inf)
    best = None
    for s in K_SL:
        stopped_w = np.mean(mae_atr[winners] <= s) if winners.any() else 0.0
        stopped_l = np.mean(mae_atr[~winners] <= s) if (~winners).any() else 0.0
        sep = stopped_l - stopped_w
        if best is None or sep > best["sep"]:
            best = {"k_sl": s, "sep": float(sep),
                    "p_stop_winner": float(stopped_w), "p_stop_loser": float(stopped_l)}
    k_sl_star = best["k_sl"]
    survived = mae_atr > k_sl_star
    if survived.sum() >= 5:
        mfe_atr = mfe[survived] / atr_entry[survived]
        k_tp_star = float(np.clip(np.median(mfe_atr), K_TP[0], K_TP[-1]))
    else:
        k_tp_star = np.nan
    return {"edge_ratio": er, "k_sl_star": float(k_sl_star),
            "k_tp_star": k_tp_star, "separation": best,
            "n_survivors": int(survived.sum())}


def ev_grid(high, low, close, entries, H, atr_entry):
    """Surface EV_net(k_sl, k_tp) + stabilite spatiale (min voisinage 3x3)."""
    grid = np.full((len(K_SL), len(K_TP)), np.nan)
    ht_mean = np.full_like(grid, np.nan)
    for i, ks in enumerate(K_SL):
        for j, kt in enumerate(K_TP):
            pnl, ht, _, _ = first_passage_trades(
                high, low, close, entries, H, ks * atr_entry, kt * atr_entry)
            grid[i, j] = pnl.mean() - COST_RT
            ht_mean[i, j] = ht.mean()
    stab = np.full_like(grid, np.nan)
    for i in range(len(K_SL)):
        for j in range(len(K_TP)):
            i0, i1 = max(0, i - 1), min(len(K_SL), i + 2)
            j0, j1 = max(0, j - 1), min(len(K_TP), j + 2)
            stab[i, j] = np.nanmin(grid[i0:i1, j0:j1])
    return grid, stab, ht_mean


# ------------------------------------------------------------------ pipeline
def fit_confluence(df_train, H_ref=80):
    """Poids w_k par information mutuelle (train only) + stats de standardisation."""
    stats, weights = {}, {}
    n = len(df_train)
    events = cross_up_events(df_train)
    entries = select_nonoverlapping(events, n, H_ref)
    high = df_train["high"].to_numpy(np.float64)
    low = df_train["low"].to_numpy(np.float64)
    close = df_train["close"].to_numpy(np.float64)
    mfe, _, _, _ = path_stats(high, low, close, entries, H_ref)
    for tf, col in SIG_COL.items():
        s = df_train[col].to_numpy(np.float64) - SIG_OFFSET[tf]
        mu, sd = float(s.mean()), float(s.std() + 1e-12)
        stats[tf] = (mu, sd)
        z = (s - mu) / sd
        z_ev = z[entries]
        mi = mutual_information(z_ev, mfe)
        sign = 1.0 if np.corrcoef(z_ev, mfe)[0, 1] >= 0 else -1.0
        weights[tf] = sign * mi
    tot = sum(abs(w) for w in weights.values()) or 1.0
    weights = {k: v / tot for k, v in weights.items()}
    return stats, weights


def confluence_score(df, stats, weights):
    c = np.zeros(len(df))
    for tf, col in SIG_COL.items():
        mu, sd = stats[tf]
        c += weights[tf] * ((df[col].to_numpy(np.float64) - SIG_OFFSET[tf] - mu) / sd)
    return np.tanh(c)


def eval_bucket(df, entries, H, label):
    """Evaluation complete d'un bucket d'entrees pour un horizon H."""
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    # atr_pct est deja une FRACTION (mediane ~0.0014 = 0.14%) : ne PAS diviser par 100
    atr_entry = df["atr_pct"].to_numpy(np.float64)[entries]
    mfe, mae, final, path_len = path_stats(high, low, close, entries, H)
    n = len(entries)
    out = {"label": label, "H": H, "n": n,
           "power_ok": bool(n >= N_MIN_POWER),
           "underpowered": bool(n < N_MIN_POWER)}
    if n < N_MIN_ABS:
        out["skipped"] = f"n<{N_MIN_ABS}"
        return out, None
    geo = edge_ratio_geometry(mfe, mae, atr_entry, final)
    out["geometry"] = geo
    out["mfe_median"] = float(np.median(mfe))
    out["mae_median"] = float(np.median(mae))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["path_efficiency"] = float(np.nanmean(
            np.where(path_len > 1e-12, np.abs(final) / path_len, 0.0)))
    # EV a la geometrie Edge-Ratio (SL*, TP*) si definie
    if not np.isnan(geo["k_tp_star"]):
        pnl, ht, hit_tp, hit_sl = first_passage_trades(
            high, low, close, entries, H,
            geo["k_sl_star"] * atr_entry, geo["k_tp_star"] * atr_entry)
        pnl_net = pnl - COST_RT
        mean, lo, hi, p = block_bootstrap_ci(pnl_net)
        p_tp, p_lo, p_hi = wilson_ci(int(hit_tp.sum()), n)
        out["ev_star"] = {
            "ev_net": mean, "ci95_low": lo, "ci95_high": hi, "p_boot": p,
            "ev_per_bar": float(mean / ht.mean()),
            "holding_mean": float(ht.mean()),
            "p_tp_wilson": [p_tp, p_lo, p_hi],
            "p_sl_wilson": list(wilson_ci(int(hit_sl.sum()), n)),
            "ci95_low_gt_cost": bool(lo > 0),  # pnl_net deja net de couts
        }
    # Surface EV + stabilite spatiale
    grid, stab, ht_m = ev_grid(high, low, close, entries, H, atr_entry)
    j_valid = np.argwhere(stab > 0)  # min voisinage > 0 (EV nette de couts)
    out["n_plateau_cells"] = int(len(j_valid))
    if len(j_valid):
        i, j = max(j_valid, key=lambda ij: stab[ij[0], ij[1]])
        out["best_plateau"] = {"k_sl": K_SL[i], "k_tp": K_TP[j],
                               "ev_net_center": float(grid[i, j]),
                               "ev_net_min_neighborhood": float(stab[i, j]),
                               "holding_mean": float(ht_m[i, j])}
    return out, (grid, stab)


def monotonicity_test(c_bucket_idx, values, n_perm=2000):
    """Regression value ~ a + b1*c + b2*c^2 ; p-value de b1 par permutation."""
    x = np.asarray(c_bucket_idx, float)
    y = np.asarray(values, float)
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.ptp(x) == 0:
        return {"b1": np.nan, "p_perm": np.nan}
    X = np.column_stack([np.ones_like(x), x, x * x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    b1 = beta[1]
    cnt = 0
    for _ in range(n_perm):
        yp = RNG.permutation(y)
        bp, *_ = np.linalg.lstsq(X, yp, rcond=None)
        if abs(bp[1]) >= abs(b1):
            cnt += 1
    return {"b1": float(b1), "p_perm": float((cnt + 1) / (n_perm + 1))}


def run_asset(asset, splits_data, H_ref=80):
    """Pipeline complet pour un actif. splits_data: {split: df_aligne}."""
    report = {"asset": asset}
    df_train = splits_data.get("train")
    if df_train is None:
        return None
    stats, weights = fit_confluence(df_train, H_ref)
    report["weights_mi"] = {k: round(v, 4) for k, v in weights.items()}
    c_train = confluence_score(df_train, stats, weights)
    dec_edges = np.quantile(c_train, np.linspace(0, 1, N_DECILES + 1))
    dec_edges[0], dec_edges[-1] = -np.inf, np.inf

    all_pvals, all_cells = [], []
    per_split = {}
    for split, df in splits_data.items():
        n = len(df)
        c = confluence_score(df, stats, weights)
        events = cross_up_events(df)
        buckets = {"ALL": events}
        dec = np.digitize(c, dec_edges[1:-1])
        for d in range(N_DECILES):
            buckets[f"D{d}"] = events & (dec == d)
        split_res = {}
        for H in H_LIST:
            if n < H + 50:
                continue
            for label, mask in buckets.items():
                entries = select_nonoverlapping(mask, n, H)
                res, _ = eval_bucket(df, entries, H, label)
                split_res[f"{label}|H{H}"] = res
                if split == "train" and "ev_star" in res:
                    all_pvals.append(res["ev_star"]["p_boot"])
                    all_cells.append(f"{label}|H{H}")
        per_split[split] = split_res
    report["splits"] = per_split

    # FDR sur les cellules train candidates
    rej = bh_fdr(all_pvals)
    report["fdr_train"] = {"n_cells": len(all_pvals),
                           "rejected": [c for c, r in zip(all_cells, rej) if r]}

    # Monotonie : MFE median par decile (train, H_ref)
    df = df_train
    c = confluence_score(df, stats, weights)
    dec = np.digitize(c, dec_edges[1:-1])
    events = cross_up_events(df)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    xs, ys = [], []
    for d in range(N_DECILES):
        entries = select_nonoverlapping(events & (dec == d), len(df), H_ref)
        if len(entries) >= N_MIN_ABS:
            mfe, _, _, _ = path_stats(high, low, close, entries, H_ref)
            xs.append(d)
            ys.append(float(np.median(mfe)))
    report["monotonicity"] = monotonicity_test(xs, ys)
    report["monotonicity_points"] = {"decile": xs, "mfe_median": ys}
    return report, stats, weights, dec_edges


def walk_forward(df, stats, weights, dec_edges, H_ref=80, n_folds=3):
    """Walk-forward : tiers chronologiques du train ; geometrie Edge-Ratio
    ajustee sur le passe, evaluee OOS sur le tiers suivant."""
    n = len(df)
    bounds = [int(i * n / n_folds) for i in range(n_folds + 1)]
    results = []
    for f in range(n_folds - 1):
        fit_slice = slice(0, bounds[f + 1])
        oos_slice = slice(bounds[f + 1], bounds[f + 2])
        df_fit, df_oos = df.iloc[fit_slice], df.iloc[oos_slice]
        ev_fit = cross_up_events(df_fit)
        entries_fit = select_nonoverlapping(ev_fit, len(df_fit), H_ref)
        if len(entries_fit) < N_MIN_ABS:
            results.append({"fold": f, "skipped": "fit trop petit"})
            continue
        high = df_fit["high"].to_numpy(np.float64)
        low = df_fit["low"].to_numpy(np.float64)
        close = df_fit["close"].to_numpy(np.float64)
        atr = df_fit["atr_pct"].to_numpy(np.float64)[entries_fit]
        mfe, mae, final, _ = path_stats(high, low, close, entries_fit, H_ref)
        geo = edge_ratio_geometry(mfe, mae, atr, final)
        if np.isnan(geo["k_tp_star"]):
            results.append({"fold": f, "skipped": "geometrie indefinie"})
            continue
        # OOS gele
        ev_oos = cross_up_events(df_oos)
        entries_oos = select_nonoverlapping(ev_oos, len(df_oos), H_ref)
        if len(entries_oos) < N_MIN_ABS:
            results.append({"fold": f, "skipped": "oos trop petit",
                            "n_oos": int(len(entries_oos))})
            continue
        high = df_oos["high"].to_numpy(np.float64)
        low = df_oos["low"].to_numpy(np.float64)
        close = df_oos["close"].to_numpy(np.float64)
        atr_o = df_oos["atr_pct"].to_numpy(np.float64)[entries_oos]
        pnl, ht, hit_tp, _ = first_passage_trades(
            high, low, close, entries_oos, H_ref,
            geo["k_sl_star"] * atr_o, geo["k_tp_star"] * atr_o)
        pnl_net = pnl - COST_RT
        mean, lo, hi, p = block_bootstrap_ci(pnl_net)
        results.append({"fold": f, "n_oos": int(len(entries_oos)),
                        "geometry_fit": geo, "ev_net_oos": mean,
                        "ci95_low": lo, "ci95_high": hi,
                        "ev_per_bar": float(mean / ht.mean())})
    return results


def cross_asset_eval(df_target, stats, weights, dec_edges, geo_frozen, H_ref=80):
    """Evalue une geometrie gelee (ajustee sur un AUTRE actif) sur df_target."""
    events = cross_up_events(df_target)
    entries = select_nonoverlapping(events, len(df_target), H_ref)
    out = {"n": int(len(entries))}
    if len(entries) < N_MIN_ABS or np.isnan(geo_frozen.get("k_tp_star", np.nan)):
        out["skipped"] = "n ou geometrie insuffisante"
        return out
    high = df_target["high"].to_numpy(np.float64)
    low = df_target["low"].to_numpy(np.float64)
    close = df_target["close"].to_numpy(np.float64)
    atr = df_target["atr_pct"].to_numpy(np.float64)[entries]
    pnl, ht, hit_tp, _ = first_passage_trades(
        high, low, close, entries, H_ref,
        geo_frozen["k_sl_star"] * atr, geo_frozen["k_tp_star"] * atr)
    pnl_net = pnl - COST_RT
    mean, lo, hi, p = block_bootstrap_ci(pnl_net)
    out.update({"ev_net": mean, "ci95_low": lo, "ci95_high": hi,
                "p_tp_wilson": list(wilson_ci(int(hit_tp.sum()), len(entries))),
                "ev_per_bar": float(mean / ht.mean())})
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for asset in ASSET_CORE:
        splits_data = {}
        for split in SPLITS:
            df = load_aligned(asset, split)
            if df is not None:
                df = df.dropna(subset=list(SIG_COL.values()) + ["atr_pct"])
                splits_data[split] = df
                print(f"[load] {asset}/{split}: {len(df)} barres 5m")
            else:
                print(f"[skip] {asset}/{split}")
        data[asset] = splits_data

    reports = {}
    fitted = {}
    for asset in ASSET_CORE:
        if "train" not in data[asset]:
            continue
        print(f"\n===== {asset} : fit confluence (MI, train only) =====")
        rep, stats, weights, dec_edges = run_asset(asset, data[asset])
        reports[asset] = rep
        fitted[asset] = (stats, weights, dec_edges)
        print("  poids MI:", rep["weights_mi"])
        print("  monotonie:", rep["monotonicity"])
        print("  FDR rejetes:", rep["fdr_train"]["rejected"])
        # Walk-forward sur le train
        rep["walk_forward"] = walk_forward(
            data[asset]["train"], stats, weights, dec_edges)
        for w in rep["walk_forward"]:
            print("  WF:", w)

    # Validation croisee inter-actifs (geometrie gelee sur train source)
    print("\n===== Validation croisee inter-actifs (H=80, geometrie gelee) =====")
    xasset = {}
    for src in ASSET_CORE:
        if src not in fitted or "train" not in data[src]:
            continue
        stats, weights, dec_edges = fitted[src]
        df_src = data[src]["train"]
        ev = cross_up_events(df_src)
        entries = select_nonoverlapping(ev, len(df_src), 80)
        high = df_src["high"].to_numpy(np.float64)
        low = df_src["low"].to_numpy(np.float64)
        close = df_src["close"].to_numpy(np.float64)
        atr = df_src["atr_pct"].to_numpy(np.float64)[entries]
        mfe, mae, final, _ = path_stats(high, low, close, entries, 80)
        geo = edge_ratio_geometry(mfe, mae, atr, final)
        for dst in ASSET_CORE:
            if dst == src:
                continue
            for split in ("val", "test"):
                if split not in data[dst]:
                    continue
                r = cross_asset_eval(data[dst][split], *fitted[src], geo)
                xasset[f"{src}->{dst}/{split}"] = r
                print(f"  {src} -> {dst}/{split}: {r}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = {"generated_utc": stamp,
           "params": {"fees_rt": FEES_RT, "slippage_rt": SLIPPAGE_RT,
                      "H_list": H_LIST, "K_SL": K_SL, "K_TP": K_TP,
                      "n_boot": N_BOOT, "block": BLOCK,
                      "n_min_power": N_MIN_POWER, "fdr_q": FDR_Q},
           "reports": reports, "cross_asset": xasset}
    path = OUT_DIR / f"mtf_confluence_v2_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nRapport JSON : {path}")

    # Verdict synthetique
    print("\n===== VERDICT V2 =====")
    any_valid = False
    for asset, rep in reports.items():
        for split, cells in rep["splits"].items():
            for name, cell in cells.items():
                ev = cell.get("ev_star")
                if ev and ev.get("ci95_low_gt_cost") and cell.get("power_ok"):
                    print(f"  [CANDIDAT] {asset}/{split}/{name}: "
                          f"EV_net={ev['ev_net']:.4%} IC95=[{ev['ci95_low']:.4%},"
                          f"{ev['ci95_high']:.4%}] n={cell['n']}")
                    if split != "train":
                        any_valid = True
    if not any_valid:
        print("  Aucune cellule ne passe (CI95_low > 0 ET n>=200 hors train).")
        print("  -> Chaine NON demontree : pas de modification du reward PPO.")
        print("  -> Voie recommandee : prior de policy / reduction des frais / horizon long.")


if __name__ == "__main__":
    main()
