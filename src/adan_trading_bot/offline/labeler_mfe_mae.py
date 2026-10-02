"""
labeler_mfe_mae.py — ADAN-System-One · Étape 12a (Labellisation hors-ligne)
============================================================================

Génère les vérités terrain Y (labels) pour l'entraînement supervisé du
SystemOneCore — le remplacement définitif de l'apprentissage par PnL (PPO).

Version VECTORISÉE (CPU-only, ~946k barres) :
  · features 88 dims reconstruites en numpy depuis les tableaux du
    NestedStateBuilder (même ordre EXACT que snapshot_to_z_features) ;
  · vérités MFE/MAE par scan d'événements vectorisé : 288 passes numpy
    (une par pas futur) au lieu d'une boucle Python par barre ;
  · aucune fuite : les features n'utilisent que les barres ≤ i, les labels
    ne lisent que les barres > i (vérité ex-post).

Labels produits (ordre aligné sur NOUL_QUESTIONS / CHOICE_QUESTIONS /
SCORE_QUESTIONS de system_one_core.py) :

  NOUL : y_mfe_atteint_tp (TP +3.5R avant SL −1.2%), y_risque_adverse_faible
         (SL jamais touché), y_expansion_imminente (range 12 barres futures
         > 1.5×ATR), y_anomalie_donnees (intégrité False), y_sweep_confirme
         (sweep Phase 0 : k∈{11,12}, mèche≥40%, réintégration),
         y_reintegration_valide (clôture suivante de retour dans le range).
  CHOICE : y_regime ∈ {0 BULL, 1 BEAR, 2 RANGE, 3 TRAP} (règle déterministe
           pente 4h + ratio range/ATR) ; y_direction ∈ {0 LONG, 1 SHORT,
           2 AUCUNE} (sens du sweep dominant).
  SCORE : y_qualite_setup, y_conviction ∈ 0..10 (ordinaux déterministes).

Splits temporels STRICTS : train < 2022 · val 2022-2023 · test ≥ 2024.
Sortie : data/labeled/{train,val,test}.parquet (f0..f87 float32 + labels + ts).

Usage :
  python -m adan_trading_bot.offline.labeler_mfe_mae [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from adan_trading_bot.data.nested_state_builder import (
    NestedStateBuilder, BARS_PER_1H, BARS_PER_4H,
)

PARQUET = "data/processed/BTCUSDT_binance/BTCUSDT_5m_featured.parquet"
OUT_DIR = "data/labeled"
LABEL_VERSION = "gate1-baseline-v2"
SL_PCT = 0.012            # exploratory baseline; not permanently optimal
TP_R = 3.5                # cible 3.5R
MAX_HOLD = 288            # fenêtre de vérité : 24 h
WICK_MIN = 0.40
LOOKBACK_SWEEP = 10
SLOPE_TREND = 0.004       # pente 4h > 0.4 % → tendance
RANGE_ATR_TRAP = 1.8      # range 4h / ATR élevé → chop/piège

FEATURE_COLS = [f"f{k}" for k in range(88)]
NOUL_COLS = ["y_mfe_atteint_tp", "y_risque_adverse_faible", "y_expansion_imminente",
             "y_anomalie_donnees", "y_sweep_confirme", "y_reintegration_valide"]
PLAN_COLS = ["y_win", "y_tp_first", "y_sl_first", "mfe", "mae", "time_to_tp", "time_to_sl", "net_return"]
LABEL_COLS = NOUL_COLS + ["y_regime", "y_direction", "y_qualite_setup", "y_conviction"] + PLAN_COLS


def atr14(o, h, l, c):
    tr = np.maximum(h - l,
                    np.maximum(np.abs(h - np.roll(c, 1)),
                               np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return pd.Series(tr).rolling(14).mean().to_numpy()

def compute_true_atr_1h(b: NestedStateBuilder) -> np.ndarray:
    """ATR14 of COMPLETE consecutive hours, available in the NEXT hour.

    A partial starting hour is excluded. Missing hours reset the rolling
    window. Warmup stays NaN: never backfill from future observations.
    Convention is deliberately one-hour lag even at the xx:55 decision.
    """
    ts_ns = b.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    ids = ts_ns // (3600 * 10**9)
    bars = pd.DataFrame({"id": ids, "h": b.h, "l": b.l, "c": b.c})
    candles = bars.groupby("id").agg(h=("h", "max"), l=("l", "min"),
                                      c=("c", "last"), count=("c", "count"))
    candles = candles.reindex(np.arange(ids.min(), ids.max() + 1))
    complete = candles["count"].eq(12)
    candles.loc[~complete, ["h", "l", "c"]] = np.nan
    prev_c = candles.c.shift(1)
    tr = pd.concat([candles.h - candles.l, (candles.h - prev_c).abs(),
                    (candles.l - prev_c).abs()], axis=1).max(axis=1)
    tr[~complete] = np.nan
    atr = tr.rolling(14, min_periods=14).mean().shift(1)
    return atr.reindex(ids).to_numpy()

def build_features(b: NestedStateBuilder) -> np.ndarray:
    """Assemble le vecteur 88 dims pour TOUTES les barres (ordre strict =
    snapshot_to_z_features). Sortie float32 (n, 88)."""
    n = b.n
    o, h, l, c, v = b.o, b.h, b.l, b.c, b.v

    # 1. bar_5m (8)
    bar = b.bars_5m                                            # (n, 8)

    # 2. seq_5m[-6:] flatten (48) — barres i-5..i, row-major
    seq_parts = []
    for d in range(5, -1, -1):
        shifted = np.vstack([np.zeros((d, 8)), b.bars_5m[: n - d]]) if d else b.bars_5m
        seq_parts.append(shifted)
    seq6 = np.concatenate(seq_parts, axis=1)                   # (n, 48)

    # 3. running_1h (4) + phase/pos/sweeps (4)
    run1 = np.stack(b.run_1h, axis=1)                          # (n, 4)
    rng1 = np.maximum(run1[:, 1] - run1[:, 2], 1e-12)
    pos1 = np.clip((c - run1[:, 2]) / rng1, 0.0, 1.0)
    sw_h1 = (np.isfinite(b.prev_1h[:, 1]) & (h > b.prev_1h[:, 1]) & (c < b.prev_1h[:, 1])).astype(float)
    sw_l1 = (np.isfinite(b.prev_1h[:, 2]) & (l < b.prev_1h[:, 2]) & (c > b.prev_1h[:, 2])).astype(float)
    ctx1 = np.column_stack([b.k_1h / BARS_PER_1H, pos1, sw_h1, sw_l1])
    prev1 = np.nan_to_num(b.prev_1h, nan=0.0)                  # (n, 5)

    # 4. running_4h (4) + phase/pos/sweeps (4)
    run4 = np.stack(b.run_4h, axis=1)
    rng4 = np.maximum(run4[:, 1] - run4[:, 2], 1e-12)
    pos4 = np.clip((c - run4[:, 2]) / rng4, 0.0, 1.0)
    sw_h4 = (np.isfinite(b.prev_4h[:, 1]) & (h > b.prev_4h[:, 1]) & (c < b.prev_4h[:, 1])).astype(float)
    sw_l4 = (np.isfinite(b.prev_4h[:, 2]) & (l < b.prev_4h[:, 2]) & (c > b.prev_4h[:, 2])).astype(float)
    ctx4 = np.column_stack([b.m_4h / BARS_PER_4H, pos4, sw_h4, sw_l4])
    prev4 = np.nan_to_num(b.prev_4h, nan=0.0)                  # (n, 5)

    # 5. portefeuille (5 zéros — état neutre hors-ligne) + intégrité (1)
    portfolio = np.zeros((n, 5))
    finite_win = (pd.Series(np.isfinite(b.bars_5m).all(axis=1))
                  .rolling(BARS_PER_4H + 1, min_periods=1).min().to_numpy() > 0)
    ohlc_ok = (h >= np.maximum(o, c)) & (l <= np.minimum(o, c)) & (l <= h) & (v >= 0)
    prev_ok = np.isfinite(b.prev_1h).all(axis=1) & np.isfinite(b.prev_4h).all(axis=1)
    integrity = (finite_win & ohlc_ok & prev_ok
                 & (np.arange(n) >= NestedStateBuilder.MIN_BARS)).astype(float)

    feats = np.concatenate([bar, seq6, run1, ctx1, prev1, run4, ctx4, prev4,
                            portfolio, integrity[:, None]], axis=1)
    return feats.astype(np.float32)


def validate_label_inputs(b, feats):
    """Fail closed: callers must split gaps into contiguous segments, not drop bars."""
    if b.n < 12 or feats.shape != (b.n, 88):
        raise ValueError("Need >=12 bars and aligned (n,88) features")
    ts = b.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if b.ts.hasnans or (ts % (300 * 10**9)).any() or (np.diff(ts) != 300 * 10**9).any():
        raise ValueError("Timeline must be unique, ordered and continuous on the 5m grid")
    bars = np.column_stack([b.o, b.h, b.l, b.c, b.v])
    if (not np.isfinite(bars).all() or (bars[:, :4] <= 0).any()
            or (b.v < 0).any() or (b.h < np.maximum(b.o, b.c)).any()
            or (b.l > np.minimum(b.o, b.c)).any()):
        raise ValueError("Nonfinite, nonpositive or incoherent OHLCV")


def compute_labels(b: NestedStateBuilder, feats: np.ndarray):
    """Vérités ex-post vectorisées rigoureuses.
    Refonte ORDRE 2B (y_expansion 1.8xATR_1h), ORDRE 2C (y_regime structurel),
    ORDRE 2D (métriques plan-conditionnées).
    """
    validate_label_inputs(b, feats)
    o, h, l, c, v = b.o, b.h, b.l, b.c, b.v
    n = b.n
    atr = atr14(o, h, l, c)
    atr_pct = atr / np.maximum(c, 1e-9)

    # 1h running & ATR_1h (true ATR from completed 1h candles)
    run1 = np.stack(b.run_1h, axis=1)
    atr_1h_old = pd.Series(np.maximum(run1[:, 1] - run1[:, 2], 1e-12)).rolling(12 * 14).mean().bfill().to_numpy()
    atr_1h = compute_true_atr_1h(b)

    # ── Sweeps (référence Phase 0 : rolling-10 + mèche ≥ 40 % + réintégration) ──
    prev_high10 = pd.Series(h).rolling(LOOKBACK_SWEEP).max().shift(1).to_numpy()
    prev_low10 = pd.Series(l).rolling(LOOKBACK_SWEEP).min().shift(1).to_numpy()
    rng = np.maximum(h - l, 1e-12)
    wick_up = h - np.maximum(o, c)
    wick_dn = np.minimum(o, c) - l
    is_sweep_high = (np.isfinite(prev_high10) & (rng > 0) & (h > prev_high10)
                     & (c < prev_high10) & (wick_up >= WICK_MIN * rng))
    is_sweep_low = (np.isfinite(prev_low10) & (rng > 0) & (l < prev_low10)
                    & (c > prev_low10) & (wick_dn >= WICK_MIN * rng))
    in_phase = np.isin(b.k_1h, [11, 12])
    y_sweep = (in_phase & (is_sweep_high | is_sweep_low)).astype(np.int64)

    # Direction du sweep dominant (0 LONG / 1 SHORT / 2 AUCUNE)
    y_direction = np.full(n, 2, dtype=np.int64)
    y_direction[is_sweep_low & ~is_sweep_high] = 0
    y_direction[is_sweep_high & ~is_sweep_low] = 1

    # ── ORDRE 2B : Refonte y_expansion (seuil 1.8 × ATR_1h) ──
    # rolling max/min sur fenêtre future (12 barres 5m = 1h future)
    fh = pd.Series(h[::-1]).rolling(12).max().to_numpy()[::-1]
    fl = pd.Series(l[::-1]).rolling(12).min().to_numpy()[::-1]
    fut_range = np.roll(fh, -1) - np.roll(fl, -1)
    y_expansion = ((fut_range >= 1.8 * atr_1h) & np.isfinite(atr_1h)
                   & np.isfinite(fut_range)).astype(np.int64)

    # ── ORDRE 2C : Refonte structurelle de y_regime (priorité TRAP explicite) ──
    ema_1h = pd.Series(c).ewm(span=12 * 12).mean().to_numpy()
    ema_slope_1h = (ema_1h - np.roll(ema_1h, 12)) / np.maximum(np.roll(ema_1h, 12), 1e-9)
    o_1h = run1[:, 0]
    rng_1h = run1[:, 1] - run1[:, 2]
    rho_1h = np.clip((c - run1[:, 2]) / np.maximum(rng_1h, 1e-12), 0.0, 1.0)
    vol_sma_1h = pd.Series(v).rolling(12 * 20).mean().bfill().to_numpy()
    abnormal_vol = (v > 2.0 * vol_sma_1h) & (np.abs(c - o) < 0.3 * rng)

    trap_sweep = (
        ((h > prev_high10) & (c < prev_high10) & (wick_up >= WICK_MIN * rng)) |
        ((l < prev_low10) & (c > prev_low10) & (wick_dn >= WICK_MIN * rng))
    )
    is_trap = trap_sweep | abnormal_vol
    is_bull = (ema_slope_1h > 0) & (c > o_1h) & (rho_1h > 0.60) & ~is_trap
    is_bear = (ema_slope_1h < 0) & (c < o_1h) & (rho_1h < 0.40) & ~is_trap
    is_range = (rng_1h < 1.0 * atr_1h) & (rho_1h >= 0.30) & (rho_1h <= 0.70) & ~is_trap & ~is_bull & ~is_bear

    y_regime = np.full(n, 2, dtype=np.int64)  # 2: RANGE par défaut
    y_regime[is_bull] = 0                     # 0: BULL
    y_regime[is_bear] = 1                     # 1: BEAR
    y_regime[is_trap] = 3                     # 3: TRAP

    # ── ORDRE 2D : MFE/MAE et résultats Plan-Conditionnés ──
    # Baseline Plan Candidate : SL = 1.2%, TP = 3.5R, horizon = 288 barres (24h)
    # Entry bar i+1 is INCLUDED; final evaluated bar is i+MAX_HOLD.
    valid = np.arange(n) < (n - MAX_HOLD)
    entry = np.roll(o, -1)
    is_short = (y_direction == 1)
    risk = entry * SL_PCT
    stop = np.where(is_short, entry + risk, entry - risk)
    tp = np.where(is_short, entry - TP_R * risk, entry + TP_R * risk)

    y_tp_first = np.zeros(n, dtype=np.int64)
    y_sl_first = np.zeros(n, dtype=np.int64)
    time_to_tp = np.zeros(n, dtype=np.int64)
    time_to_sl = np.zeros(n, dtype=np.int64)
    mfe = np.zeros(n, dtype=np.float32)
    mae = np.zeros(n, dtype=np.float32)

    active = valid.copy()
    idx = np.arange(n)

    for k in range(1, MAX_HOLD + 1):
        j = idx + k
        ok = active & (j < n)
        if not ok.any():
            break

        hj = h[np.minimum(j, n - 1)]
        lj = l[np.minimum(j, n - 1)]

        fav = np.where(is_short, (entry - lj) / entry, (hj - entry) / entry)
        adv = np.where(is_short, (hj - entry) / entry, (entry - lj) / entry)
        mfe = np.where(ok, np.maximum(mfe, fav), mfe)
        mae = np.where(ok, np.maximum(mae, adv), mae)

        hit_sl = ok & np.where(is_short, hj >= stop, lj <= stop)
        hit_tp = ok & np.where(is_short, lj <= tp, hj >= tp)

        both = hit_sl & hit_tp
        sl_only = hit_sl | both  # conservateur
        tp_only = hit_tp & ~hit_sl

        # Enregistrement du premier événement
        new_tp = tp_only & (y_tp_first == 0) & (y_sl_first == 0)
        new_sl = sl_only & (y_tp_first == 0) & (y_sl_first == 0)
        y_tp_first[new_tp] = 1
        y_sl_first[new_sl] = 1
        time_to_tp[new_tp] = k
        time_to_sl[new_sl] = k

        active[sl_only | tp_only] = False

    # Net Return en R (frais Maker estimés 0.08% RT -> 0.08/1.2 = 0.0667 R)
    fees_r = 0.0008 / SL_PCT
    net_return = np.where(y_tp_first == 1, TP_R - fees_r,
                 np.where(y_sl_first == 1, -1.0 - fees_r, 0.0)).astype(np.float32)
    y_win = (y_tp_first == 1).astype(np.int64)
    y_lowmae = (valid & (y_sl_first == 0)).astype(np.int64)

    # ── Réintégration : clôture suivante de retour dans le range ──
    next_c = np.roll(c, -1)
    y_reint = np.zeros(n, dtype=np.int64)
    y_reint[is_sweep_high] = (next_c[is_sweep_high] < prev_high10[is_sweep_high]).astype(np.int64)
    y_reint[is_sweep_low] = (next_c[is_sweep_low] > prev_low10[is_sweep_low]).astype(np.int64)
    y_reint[-1] = 0  # No next bar: np.roll must not wrap to the first bar.

    # ── Anomalie : intégrité False ──
    y_anomalie = (feats[:, 87] < 0.5).astype(np.int64)

    # ── Scores ordinaux 0..10 ──
    pos1 = feats[:, 61]
    qualite = (3.0 * y_sweep + 2.0 * y_expansion
               + 3.0 * y_tp_first * (y_direction != 2)
               + 2.0 * np.where(y_direction == 1, pos1,
                                np.where(y_direction == 0, 1 - pos1, 0.5)))
    y_qualite = np.clip(np.round(qualite), 0, 10).astype(np.int64)
    conviction = 3.0 * y_sweep + 3.0 * y_tp_first + 2.0 * y_reint + 2.0 * y_lowmae
    y_conviction = np.clip(np.round(conviction), 0, 10).astype(np.int64)

    labels = {
        "y_mfe_atteint_tp": y_tp_first,
        "y_risque_adverse_faible": y_lowmae,
        "y_expansion_imminente": y_expansion,
        "y_anomalie_donnees": y_anomalie,
        "y_sweep_confirme": y_sweep,
        "y_reintegration_valide": y_reint,
        "y_regime": y_regime,
        "y_direction": y_direction,
        "y_qualite_setup": y_qualite,
        "y_conviction": y_conviction,
        "y_win": y_win,
        "y_tp_first": y_tp_first,
        "y_sl_first": y_sl_first,
        "mfe": mfe,
        "mae": mae,
        "time_to_tp": time_to_tp,
        "time_to_sl": time_to_sl,
        "net_return": net_return,
        "atr_1h_old": atr_1h_old,
        "atr_1h_new": atr_1h,
        "plan_valid": valid.astype(np.int64)
    }
    
    # --- Diagnostics ---
    print("\n── ATR_1h Audit (old vs new) ──")
    valid_mask = np.isfinite(atr_1h_old) & np.isfinite(atr_1h)
    old_v = atr_1h_old[valid_mask]
    new_v = atr_1h[valid_mask]
    if len(old_v) > 0:
        corr = np.corrcoef(old_v, new_v)[0, 1]
        print(f"  Correlation: {corr:.6f}")
        print(f"  Old — mean: {old_v.mean():.4f}, median: {np.median(old_v):.4f}, p95: {np.percentile(old_v, 95):.4f}, max: {old_v.max():.4f}")
        print(f"  New — mean: {new_v.mean():.4f}, median: {np.median(new_v):.4f}, p95: {np.percentile(new_v, 95):.4f}, max: {new_v.max():.4f}")
    
    n_trap = is_trap.sum()
    n_bull = is_bull.sum()
    n_bear = is_bear.sum()
    n_range = is_range.sum()
    n_default = n - (n_trap + n_bull + n_bear + n_range)
    print(f"\n── Regime Overlap Diagnostics ──")
    print(f"  Regime candidates: BULL={n_bull:,} BEAR={n_bear:,} RANGE(explicit)={n_range:,} TRAP={n_trap:,} default_RANGE={n_default:,}")
    bull_raw = (ema_slope_1h > 0) & (c > o_1h) & (rho_1h > 0.60)
    bear_raw = (ema_slope_1h < 0) & (c < o_1h) & (rho_1h < 0.40)
    trap_and_bull = (is_trap & bull_raw).sum()
    trap_and_bear = (is_trap & bear_raw).sum()
    print(f"  Multi-match: TRAP∩BULL_raw={trap_and_bull:,} TRAP∩BEAR_raw={trap_and_bear:,}")
    print(f"  Priority: TRAP takes precedence over BULL/BEAR/RANGE")

    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="limite de barres (smoke test)")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 74)
    print("ADAN-SYSTEM-ONE · ÉTAPE 12a — Labellisation MFE/MAE (vectorisée)")
    print("=" * 74)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET, columns=["open", "high", "low", "close", "volume"])
    # Never silently drop malformed bars: that would change horizon/timestamps.
    if args.limit:
        df = df.iloc[: args.limit]
    print(f"Données : {len(df):,} bougies ({df.index[0]} → {df.index[-1]})")

    print("Construction du builder (contenants vivants)…", flush=True)
    b = NestedStateBuilder(df)
    print(f"  OK ({time.time()-t0:.0f}s)", flush=True)

    print("Features 88 dims (vectorisé)…", flush=True)
    feats = build_features(b)
    print(f"  OK {feats.shape} ({time.time()-t0:.0f}s)", flush=True)

    print("Labels (scan d'événements 288 passes)…", flush=True)
    labels = compute_labels(b, feats)
    print(f"  OK ({time.time()-t0:.0f}s)", flush=True)

    data = pd.DataFrame(feats, columns=FEATURE_COLS)
    for k, v in labels.items():
        data[k] = v
    data["ts"] = df.index.to_numpy()

    # Fenêtre valide : MIN_BARS ≤ i < n − MAX_HOLD
    n = len(df)
    keep = np.zeros(n, dtype=bool)
    keep[NestedStateBuilder.MIN_BARS: n - MAX_HOLD] = True
    data = data[keep].reset_index(drop=True)
    print(f"\nExemples labellisés : {len(data):,}")

    train = data[data.ts < "2022-01-01"]
    val = data[(data.ts >= "2022-01-01") & (data.ts < "2024-01-01")]
    test = data[data.ts >= "2024-01-01"]
    for name, part in [("train", train), ("val", val), ("test", test)]:
        out = os.path.join(OUT_DIR, f"{name}.parquet")
        part.to_parquet(out, index=False)
        print(f"  {name:5s} : {len(part):>8,} exemples → {out}")

    print("\n── Équilibre des labels (train) ──")
    for col in NOUL_COLS:
        print(f"  {col:28s} : {100*float(train[col].mean()):5.2f}% positifs")
    print("  y_regime    :", train.y_regime.value_counts(normalize=True).sort_index().round(4).to_dict())
    print("  y_direction :", train.y_direction.value_counts(normalize=True).sort_index().round(4).to_dict())
    print("  qualité moy.:", round(float(train.y_qualite_setup.mean()), 2),
          "| conviction moy.:", round(float(train.y_conviction.mean()), 2))
    print(f"\nDataset baseline labellisé en {time.time()-t0:.0f}s; "
          "NOT training-authorized until gates 2–7, split purging and economics pass.")


if __name__ == "__main__":
    main()
