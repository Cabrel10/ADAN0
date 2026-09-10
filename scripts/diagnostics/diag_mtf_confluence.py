#!/usr/bin/env python3
"""diag_mtf_confluence.py — sonde de confluence multi-timeframe (5m/1h/4h).

Répond à la critique méthodologique : les sondes précédentes testaient chaque
timeframe ISOLÉMENT. ADAN décide sur 3 TF simultanés (~7 trades/jour). La vraie
hypothèse : le 4h donne le régime, le 1h la structure/pullback, le 5m le trigger.

Cette sonde :
  1. ALIGNE les indicateurs 1h/4h sur la timeline 5m SANS lookahead
     (une barre 1h [T,T+1h) n'est disponible qu'à T+1h — merge_asof backward
     sur le timestamp de clôture de la barre haute TF).
  2. CROISE des conditions multi-TF (tendance 4h ET structure 1h ET trigger 5m).
  3. MESURE sur les entrées retenues :
       - MFE/MAE (TP/SL utile — ce que le trade aurait pu capturer),
       - probabilités de premier passage P(TP)/P(SL)/P(MD) sur grille SL/TP,
       - efficacité du chemin (gain_final / Σ|ΔP|) pour le reward géométrique.
  4. COMPARE entrées filtrées vs aveugles, trades NON SUPERPOSÉS (stride = H).
  5. VALIDE sur train / val / test séparément (pas de mélange).

Règle d'honnêteté statistique : EV rapporté seulement si n_trades >= MIN_TRADES.
Lecture seule. Sortie JSON dans logs/validation/ + tableaux stdout.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/indicators"
OUT_DIR = ROOT / "logs/validation"

FEES_RT = 0.004          # round-trip 0.40%
MIN_TRADES = 30          # seuil de signifiance minimale
H = 40                   # horizon MaxDuration en barres 5m (~3h20)

SL_GRID = [0.003, 0.005, 0.008, 0.012, 0.016, 0.020, 0.030]
TP_GRID = [0.006, 0.010, 0.0135, 0.020, 0.030, 0.045, 0.060]

ASSETS = ["BTCUSDT", "BTCUSDT_BINANCE", "DOGEUSDT", "DOGEUSDT_BINANCE"]
SPLITS = ["train", "val", "test"]

TF_DELTA = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4)}

# ---------------------------------------------------------------------------
# Conditions multi-TF candidates (LONG). Chaque règle = dict nom -> lambda(row_arrays)
# Les features alignées portent le suffixe du TF : _5m, _1h, _4h.
# ---------------------------------------------------------------------------
RULES = {
    # A. Tendance pleine : 4h au-dessus EMA100, 1h au-dessus EMA50, momentum 5m positif
    "A_trend_full": lambda f: (f["ema_100_ratio_4h"] > 1.0)
                              & (f["ema_50_ratio_1h"] > 1.0)
                              & (f["macdh_12_26_9_5m"] > 0),
    # B. Pullback en tendance : 4h up, RSI 1h replié (35-50), reprise momentum 5m
    "B_pullback": lambda f: (f["ema_100_ratio_4h"] > 1.0)
                            & (f["rsi_21_1h"] >= 35) & (f["rsi_21_1h"] <= 50)
                            & (f["macdh_12_26_9_5m"] > 0),
    # C. Régime 4h seul + trigger 5m (1h libre) — teste si le 1h ajoute qqch
    "C_4h_5m_only": lambda f: (f["ema_100_ratio_4h"] > 1.0)
                              & (f["macdh_12_26_9_5m"] > 0),
    # D. Confluence stricte : 4h up + 1h up + RSI1h zone [40,55] + RSI5m > 50 + ADX5m > 20
    "D_strict": lambda f: (f["ema_100_ratio_4h"] > 1.0)
                          & (f["ema_50_ratio_1h"] > 1.0)
                          & (f["rsi_21_1h"] >= 40) & (f["rsi_21_1h"] <= 55)
                          & (f["rsi_14_5m"] > 50)
                          & (f["adx_14_5m"] > 20),
    # E. Contre-tendance (négatif attendu — contrôle) : 4h down + trigger 5m long
    "E_countertrend": lambda f: (f["ema_100_ratio_4h"] < 1.0)
                                & (f["macdh_12_26_9_5m"] > 0),
}


def load_aligned(asset: str, split: str) -> pd.DataFrame | None:
    """Charge 5m et aligne 1h/4h dessus sans lookahead (asof backward sur clôture)."""
    try:
        df5 = pd.read_parquet(DATA / split / asset / "5m.parquet")
        df1 = pd.read_parquet(DATA / split / asset / "1h.parquet")
        df4 = pd.read_parquet(DATA / split / asset / "4h.parquet")
    except Exception as e:
        print(f"  [skip] {split}/{asset}: {e}")
        return None

    df5 = df5.sort_index()
    out = df5.copy()
    out.index = out.index.astype("datetime64[ms]")

    for tf, dsrc in (("1h", df1), ("4h", df4)):
        d = dsrc.sort_index()
        # Colonnes indicateurs utiles (on ignore OHLCV du haut TF sauf close)
        keep = [c for c in d.columns if c not in ("open", "high", "low", "volume")]
        d = d[keep].copy()
        # DISPONIBILITÉ : la barre haute-TF ouverte à T est close (donc observable)
        # à T + durée(TF). On décale l'index : pas de lookahead.
        # (astype : le décalage Timedelta change l'unité ms->us, merge_asof exige
        #  le même dtype des deux côtés)
        d.index = (d.index + TF_DELTA[tf]).astype("datetime64[ms]")
        d = d.add_suffix(f"_{tf}")
        out = pd.merge_asof(out, d, left_index=True, right_index=True,
                            direction="backward")
    return out


def first_passage_batch(high, low, close, entries, horizon, sl, tp):
    """Premier passage pessimiste (même barre high+low -> SL) pour un lot d'entrées."""
    n = len(entries)
    hit_tp = np.zeros(n, dtype=bool)
    hit_sl = np.zeros(n, dtype=bool)
    entry = close[entries]
    for h in range(1, horizon + 1):
        hh = high[entries + h] / entry - 1.0
        ll = low[entries + h] / entry - 1.0
        both = (~hit_tp) & (~hit_sl) & (hh >= tp) & (ll <= -sl)
        hit_sl |= both
        hit_tp |= (~hit_tp) & (~hit_sl) & (hh >= tp)
        hit_sl |= (~hit_tp) & (~hit_sl) & (ll <= -sl)
    md = (~hit_tp) & (~hit_sl)
    md_ret = close[entries + horizon] / entry - 1.0
    mean_md = float(md_ret[md].mean()) if md.any() else 0.0
    ev_gross = hit_tp.mean() * tp - hit_sl.mean() * sl + md.mean() * mean_md
    return float(ev_gross), float(hit_tp.mean()), float(hit_sl.mean()), float(md.mean())


def mfe_mae_batch(high, low, close, entries, horizon):
    """MFE / MAE / efficacité du chemin — indépendant du couple (sl,tp)."""
    entry = close[entries]
    mfe = np.full(len(entries), -np.inf)
    mae = np.full(len(entries), np.inf)
    path_len = np.zeros(len(entries))
    prev = entry.copy()
    for h in range(1, horizon + 1):
        mfe = np.maximum(mfe, high[entries + h] / entry - 1.0)
        mae = np.minimum(mae, low[entries + h] / entry - 1.0)
        cur = close[entries + h]
        path_len += np.abs(cur / prev - 1.0)
        prev = cur
    final = close[entries + horizon] / entry - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = np.where(path_len > 1e-12, np.abs(final) / path_len, 0.0)
    return mfe, mae, final, efficiency


def eval_entries(df, entries, label):
    """Évalue un ensemble d'entrées (non superposées) sur la grille SL/TP + MFE/MAE."""
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    n = len(entries)
    if n < MIN_TRADES:
        return {"label": label, "n": n, "skipped": f"n<{MIN_TRADES}"}

    mfe, mae, final, eff = mfe_mae_batch(high, low, close, entries, H)
    grid = []
    best = None
    for sl in SL_GRID:
        for tp in TP_GRID:
            ev_g, p_tp, p_sl, p_md = first_passage_batch(high, low, close, entries, H, sl, tp)
            ev_net = ev_g - FEES_RT
            grid.append({"sl": sl, "tp": tp, "ev_gross": round(ev_g, 5),
                         "ev_net": round(ev_net, 5), "p_tp": round(p_tp, 4),
                         "p_sl": round(p_sl, 4), "p_md": round(p_md, 4)})
            if best is None or ev_net > best["ev_net"]:
                best = grid[-1]
    return {
        "label": label, "n": n, "H": H,
        "mfe_mean": round(float(mfe.mean()), 5),
        "mfe_median": round(float(np.median(mfe)), 5),
        "mae_mean": round(float(mae.mean()), 5),
        "mae_median": round(float(np.median(mae)), 5),
        "final_mean": round(float(final.mean()), 5),
        "path_efficiency_mean": round(float(eff.mean()), 4),
        "best_net": best,
        "n_positive_net_cells": sum(1 for g in grid if g["ev_net"] > 0),
        "grid": grid,
    }


def run_one(asset, split):
    df = load_aligned(asset, split)
    if df is None or len(df) < H + 2:
        return None
    n = len(df)
    # Entrées non superposées : stride = H, on garde de la marge pour l'horizon
    stride_idx = np.arange(0, n - H - 1, H)

    feats = {c: df[c].to_numpy(np.float64) for c in df.columns
             if c.endswith(("_5m", "_1h", "_4h")) or c in ("rsi_14",)}
    # Les colonnes 5m natives n'ont pas de suffixe -> on les renomme logiquement
    native = {"rsi_14_5m": "rsi_14", "macdh_12_26_9_5m": "macdh_12_26_9",
              "adx_14_5m": "adx_14", "ema_20_ratio_5m": "ema_20_ratio"}
    for alias, col in native.items():
        if col in df.columns:
            feats[alias] = df[col].to_numpy(np.float64)

    res = {"asset": asset, "split": split, "rows_5m": n,
           "entries": {}}

    # Baseline aveugle : toutes les positions stride
    res["entries"]["BLIND"] = eval_entries(df, stride_idx, "BLIND")

    for name, rule in RULES.items():
        try:
            mask_full = rule(feats)
        except KeyError as e:
            res["entries"][name] = {"error": f"missing feature {e}"}
            continue
        sel = stride_idx[mask_full[stride_idx]]
        res["entries"][name] = eval_entries(df, sel, name)
    return res


def print_summary(all_res):
    print("\n" + "=" * 100)
    print(f"# CONFLUENCE MTF — H={H} barres 5m, stride={H} (non superposé), frais RT={FEES_RT:.2%}, min_trades={MIN_TRADES}")
    print("=" * 100)
    hdr = f"{'asset/split':<28} {'règle':<16} {'n':>5} {'MFE_med':>8} {'MAE_med':>8} {'eff':>6} {'bestEVnet':>10} {'(sl,tp)':>14} {'P(TP)':>6} {'P(SL)':>6}"
    for res in all_res:
        if res is None:
            continue
        print(f"\n## {res['asset']} [{res['split']}] — {res['rows_5m']} barres 5m")
        print(hdr)
        print("-" * len(hdr))
        for name, e in res["entries"].items():
            if "error" in e:
                print(f"{'':<28} {name:<16} ERREUR {e['error']}")
                continue
            if "skipped" in e:
                print(f"{'':<28} {name:<16} n={e['n']:>4}  (ignoré: {e['skipped']})")
                continue
            b = e["best_net"]
            flag = " ***" if b["ev_net"] > 0 else ""
            print(f"{'':<28} {name:<16} {e['n']:>5} {e['mfe_median']:>+8.3%} {e['mae_median']:>+8.3%} "
                  f"{e['path_efficiency_mean']:>6.3f} {b['ev_net']:>+10.4%} "
                  f"({b['sl']:.3f},{b['tp']:.4f}) {b['p_tp']:>6.2%} {b['p_sl']:>6.2%}{flag}")


def main():
    all_res = []
    for split in SPLITS:
        for asset in ASSETS:
            print(f"[run] {split}/{asset} ...", flush=True)
            all_res.append(run_one(asset, split))

    print_summary(all_res)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"mtf_confluence_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "params": {"fees_rt": FEES_RT, "H": H, "stride": H,
                          "min_trades": MIN_TRADES,
                          "sl_grid": SL_GRID, "tp_grid": TP_GRID},
               "results": [r for r in all_res if r is not None]}
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nRapport JSON : {out}")

    # Verdict synthétique : une règle filtrée bat-elle l'aveugle en NET sur train ET val ET test ?
    print("\n## VERDICT (edge net filtré > edge net aveugle, sur les 3 splits)")
    by_asset = {}
    for r in payload["results"]:
        by_asset.setdefault(r["asset"], {})[r["split"]] = r["entries"]
    for asset, splits in by_asset.items():
        for rule in RULES:
            row = []
            for sp in SPLITS:
                e = splits.get(sp, {}).get(rule, {})
                b = splits.get(sp, {}).get("BLIND", {})
                if "best_net" in e and "best_net" in b:
                    delta = e["best_net"]["ev_net"] - b["best_net"]["ev_net"]
                    row.append(f"{sp}:{delta:+.4%}")
                else:
                    row.append(f"{sp}:n/a")
            print(f"  {asset:<22} {rule:<16} {'  '.join(row)}")


if __name__ == "__main__":
    main()
