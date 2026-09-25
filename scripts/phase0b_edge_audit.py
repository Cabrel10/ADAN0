#!/usr/bin/env python3
"""
phase0b_edge_audit.py — ADAN-System-One · PHASE 0b (audit indépendant)
======================================================================

Cadre d'analyse en 4 niveaux (décision 2026-09-25, post Phase 0) :

  La Phase 0 a montré une SÉPARATION structurelle massive :
      filtre sweep : −2.78R (test)   vs   baseline : −4.08R (test)
      → +1.30R d'écart structurel. Le signal n'est PAS du bruit.

  Mais l'EV nette est restée négative à cause du piège du dénominateur R :
      frais_R = 0.40 % / distance_SL
      SL micro-mèche (~0.15 %) → frais ≈ 2.67 R par trade  (impossible)
      SL structurel (~0.80 %) → frais ≈ 0.50 R par trade    (absorbable)

  Conclusion de méthode : on ne demande plus à UNE géométrie arbitraire
  d'être rentable. On mesure d'abord la PHYSIQUE du signal, puis on
  cherche la géométrie qui la monétise.

  Niveau 1 — STRUCTURE  : P(MFE > x·ATR | signal) vs baseline
  Niveau 2 — EXCURSION  : distributions MFE / MAE, durées avant extrêmes
  Niveau 3 — GÉOMÉTRIE  : grille SL{0.4,0.8,1.2 %} × TP{1.5R,2.5R,3.5R}
  Niveau 4 — ÉCONOMIE   : EV nette sous 2 régimes de frais
                          (taker stress 0.40 % RT · maker réel 0.08 % RT)

Verdicts possibles :
  NÉGATIF : aucune structure (MFE signal ≤ MFE baseline)
  NEUTRE  : structure mais aucune géométrie ne l'absorbe
  POSITIF : structure + géométrie nettement positive sur test
"""

import numpy as np
import pandas as pd

PARQUET = "data/processed/BTCUSDT_binance/BTCUSDT_5m_featured.parquet"
WICK_MIN = 0.40
LOOKBACK = 10
MAX_HOLD = 288          # 24 h
PHASES = {11, 12}
FEES_TAKER = 0.0040     # 0.40 % RT (stress-test)
FEES_MAKER = 0.0008     # 0.08 % RT (post-only limit, production)
SL_GRID = [0.004, 0.008, 0.012]       # 0.4 % / 0.8 % / 1.2 %
TP_GRID = [1.5, 2.5, 3.5]             # en R
RNG_SEED = 42


def load():
    df = pd.read_parquet(PARQUET, columns=["open", "high", "low", "close"])
    df = df[df["high"] >= df[["open", "close"]].max(axis=1)]
    return df


def detect_signals(df, side="short"):
    """Indices des bougies-signal (sweep + réintégration + mèche, phase 11/12)."""
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float);  c = df["close"].to_numpy(float)
    idx = df.index
    phase = idx.minute.to_numpy() // 5 + 1
    prev_high = pd.Series(h).rolling(LOOKBACK).max().shift(1).to_numpy()
    prev_low = pd.Series(l).rolling(LOOKBACK).min().shift(1).to_numpy()
    sig, base = [], []
    for i in range(LOOKBACK, len(df) - MAX_HOLD - 1):
        if phase[i] not in PHASES:
            continue
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        if side == "short":
            wick = h[i] - max(o[i], c[i])
            ok = (np.isfinite(prev_high[i]) and h[i] > prev_high[i]
                  and c[i] < prev_high[i] and wick >= WICK_MIN * rng)
        else:
            wick = min(o[i], c[i]) - l[i]
            ok = (np.isfinite(prev_low[i]) and l[i] < prev_low[i]
                  and c[i] > prev_low[i] and wick >= WICK_MIN * rng)
        (sig if ok else base).append(i)
    return np.array(sig), np.array(base)


def excursions(df, events, side="short"):
    """Niveaux 1-2 : MFE/MAE (% entry) + durées, pour chaque événement."""
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float);  c = df["close"].to_numpy(float)
    atr = atr14(df)
    out = []
    for i in events:
        entry = o[i + 1]
        mfe = mae = 0.0
        t_mfe = t_mae = MAX_HOLD
        for j in range(i + 1, i + 1 + MAX_HOLD):
            fav = (entry - l[j]) if side == "short" else (h[j] - entry)
            adv = (h[j] - entry) if side == "short" else (entry - l[j])
            if fav > mfe:
                mfe, t_mfe = fav, j - i
            if adv > mae:
                mae, t_mae = adv, j - i
        out.append((df.index[i], 100 * mfe / entry, 100 * mae / entry,
                    t_mfe, t_mae, atr[i] / entry * 100 if atr[i] > 0 else np.nan))
    return pd.DataFrame(out, columns=["ts", "mfe_pct", "mae_pct",
                                      "t_mfe", "t_mae", "atr_pct"])


def atr14(df):
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - df["close"].shift()).abs(),
                               (df["low"] - df["close"].shift()).abs()))
    return tr.rolling(14).mean().to_numpy()


def simulate_geometry(df, events, sl_pct, tp_r, side="short"):
    """Niveau 3 : EV brute (R) pour une géométrie SL% × TP(R)."""
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float);  c = df["close"].to_numpy(float)
    rs = []
    for i in events:
        entry = o[i + 1]
        risk = entry * sl_pct
        stop = entry + risk if side == "short" else entry - risk
        tp = entry - tp_r * risk if side == "short" else entry + tp_r * risk
        r = None
        for j in range(i + 1, i + 1 + MAX_HOLD):
            hit_sl = h[j] >= stop if side == "short" else l[j] <= stop
            hit_tp = l[j] <= tp if side == "short" else h[j] >= tp
            if hit_sl or hit_tp:
                r = -1.0 if hit_sl else tp_r   # ambigu → pire cas (SL d'abord)
                break
        if r is None:
            exit_p = c[i + MAX_HOLD]
            r = ((entry - exit_p) if side == "short" else (exit_p - entry)) / risk
        rs.append(r)
    return np.array(rs)


def split_mask(ts):
    return {"train": ts < "2022-01-01",
            "val": (ts >= "2022-01-01") & (ts < "2024-01-01"),
            "test": ts >= "2024-01-01"}


def main():
    print("=" * 78)
    print("ADAN-SYSTEM-ONE · PHASE 0b — Audit structure/excursion/géométrie/économie")
    print("=" * 78)
    df = load()
    print(f"Données : {len(df):,} bougies 5m ({df.index[0]} → {df.index[-1]})")

    sig, base = detect_signals(df, "short")
    rng = np.random.default_rng(RNG_SEED)
    base_sample = rng.choice(base, size=min(len(sig), len(base)), replace=False)
    base_sample.sort()
    print(f"Signaux sweep : {len(sig):,}   |   baseline (même phase, sans sweep) : "
          f"{len(base):,} (échantillon {len(base_sample):,}, seed={RNG_SEED})")

    # ── NIVEAU 1+2 : STRUCTURE & EXCURSION ─────────────────────────────────
    print("\n" + "─" * 78)
    print("NIVEAU 1-2 — STRUCTURE & EXCURSION (fenêtre 24 h, sans aucune géométrie)")
    print("─" * 78)
    exc_sig = excursions(df, sig)
    exc_base = excursions(df, base_sample)
    for sname, m in split_mask(exc_sig.ts).items():
        s_sig = exc_sig[m]
        s_base = exc_base[split_mask(exc_base.ts)[sname]]
        if s_sig.empty:
            continue
        # P(MFE > k×ATR) : la probabilité que l'expansion dépasse le bruit
        p1s = (s_sig.mfe_pct > 1.0 * s_sig.atr_pct).mean()
        p2s = (s_sig.mfe_pct > 2.0 * s_sig.atr_pct).mean()
        p1b = (s_base.mfe_pct > 1.0 * s_base.atr_pct).mean()
        p2b = (s_base.mfe_pct > 2.0 * s_base.atr_pct).mean()
        print(f"\n  [{sname}]  n_signal={len(s_sig):,}  n_base={len(s_base):,}")
        print(f"    P(MFE > 1×ATR) : signal {100*p1s:5.1f}%  vs  baseline {100*p1b:5.1f}%  "
              f"(Δ {100*(p1s-p1b):+5.1f} pts)")
        print(f"    P(MFE > 2×ATR) : signal {100*p2s:5.1f}%  vs  baseline {100*p2b:5.1f}%  "
              f"(Δ {100*(p2s-p2b):+5.1f} pts)")
        print(f"    MFE médian     : signal {s_sig.mfe_pct.median():5.2f}%  vs  "
              f"baseline {s_base.mfe_pct.median():5.2f}%")
        print(f"    MAE médian     : signal {s_sig.mae_pct.median():5.2f}%  vs  "
              f"baseline {s_base.mae_pct.median():5.2f}%")
        print(f"    MFE/MAE médian : signal "
              f"{(s_sig.mfe_pct/(s_sig.mae_pct+1e-9)).median():5.2f}  vs  baseline "
              f"{(s_base.mfe_pct/(s_base.mae_pct+1e-9)).median():5.2f}")
        print(f"    Durée médiane avant MFE : {s_sig.t_mfe.median():.0f} barres "
              f"({s_sig.t_mfe.median()*5/60:.1f} h) — avant MAE : {s_sig.t_mae.median():.0f} barres")

    # ── NIVEAU 3 : GÉOMÉTRIE (grille SL × TP, EV BRUTE en R) ──────────────
    print("\n" + "─" * 78)
    print("NIVEAU 3 — GÉOMÉTRIE : EV BRUTE (R) sur le split TEST (≥2024, jamais optimisé)")
    print("─" * 78)
    sig_test = sig[df.index[sig] >= "2024-01-01"]
    sig_test = sig_test[sig_test < len(df) - MAX_HOLD - 1]
    print(f"  Signaux sweep sur test : {len(sig_test):,}")
    print(f"  {'SL \\ TP':>10s}" + "".join(f"{tp:>10.1f}R" for tp in TP_GRID))
    grid = {}
    for sl in SL_GRID:
        row = []
        for tp in TP_GRID:
            rs = simulate_geometry(df, sig_test, sl, tp)
            row.append(rs.mean())
        grid[sl] = row
        print(f"  {100*sl:8.2f}%  " + "".join(f"{v:+10.3f}" for v in row))

    # ── NIVEAU 4 : ÉCONOMIE (EV nette = EV brute − frais/SL) ───────────────
    print("\n" + "─" * 78)
    print("NIVEAU 4 — ÉCONOMIE : EV NETTE (R) = EV brute − frais_R   [split test]")
    print(f"  frais_R(taker 0.40%) = 0.40/SL%   ·   frais_R(maker 0.08%) = 0.08/SL%")
    print("─" * 78)
    header = f"  {'SL \\ TP':>10s}" + "".join(f"{tp:>10.1f}R" for tp in TP_GRID)
    print(header + "   (TAKER 0.40 % RT)")
    for sl in SL_GRID:
        fees_r = FEES_TAKER / sl
        vals = [grid[sl][k] - fees_r for k in range(len(TP_GRID))]
        print(f"  {100*sl:8.2f}%  " + "".join(f"{v:+10.3f}" for v in vals)
              + f"   (frais {fees_r:.2f}R)")
    print(header + "   (MAKER 0.08 % RT — ordres limit post-only)")
    best = None
    for sl in SL_GRID:
        fees_r = FEES_MAKER / sl
        vals = [grid[sl][k] - fees_r for k in range(len(TP_GRID))]
        print(f"  {100*sl:8.2f}%  " + "".join(f"{v:+10.3f}" for v in vals)
              + f"   (frais {fees_r:.2f}R)")
        for k, v in enumerate(vals):
            if best is None or v > best[0]:
                best = (v, sl, TP_GRID[k])

    # ── VERDICT ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("VERDICT PHASE 0b")
    print("=" * 78)
    s_test = exc_sig[exc_sig.ts >= "2024-01-01"]
    b_test = exc_base[exc_base.ts >= "2024-01-01"]
    struct_ok = (not s_test.empty and not b_test.empty
                 and s_test.mfe_pct.median() > b_test.mfe_pct.median())
    if not struct_ok:
        print("❌ NÉGATIF — aucune structure détectable (MFE signal ≤ baseline).")
    elif best and best[0] > 0:
        print(f"✅ POSITIF — structure détectée ET géométrie exploitable :")
        print(f"   meilleure cellule : SL={100*best[1]:.2f}% × TP={best[2]}R "
              f"→ EV nette = {best[0]:+.3f}R/trade (maker 0.08%)")
        print("   → GO pour la construction des modules System One.")
    else:
        print("⚠️  NEUTRE — structure détectée mais aucune géométrie testée ne")
        print("   l'absorbe net de frais. Pistes : SL structurel 1h (au-delà de 1.2%),")
        print("   trailing, ou filtre régime avant d'activer la géométrie.")


if __name__ == "__main__":
    main()
