#!/usr/bin/env python3
"""diag_conditional_edge.py — test du QUADRANT CACHE (ETAPE 1 du plan).

Pourquoi cette sonde existe
---------------------------
Toutes les sondes R1->spot_mirror mesurent la moyenne INCONDITIONNELLE d'une
regle fixe appliquee partout ou elle se declenche : 9840 allers-retours,
EV brut +0,01 %/trade. Mais une distribution qui contient 400 trades a
+0,60 % et 9440 trades a -0,02 % donne EXACTEMENT cette moyenne : la sonde
inconditionnelle ecrase le quadrant favorable.

Or le marche offre l'amplitude (mfe_excursion : MFE p90 4h = +2,43 %, p95
24h = +8,68 %) et l'utilisateur trade 2-5 fois/jour EN SELECTIONNANT. La
question jamais mesuree :

    L'EV des trades est-elle CONCENTREE dans un sous-ensemble identifiable
    par les features disponibles a l'entree ?

Cette sonde :
  1. Rejoue la machine a etats spot de diag_spot_mirror_exit (C1 : achat
     cross_up+MTF_ALIGNE -> vente cross_down/invalidation 4h/timeout 7j,
     sans SL) sur BTCUSDT_BINANCE et DOGEUSDT_BINANCE, splits train/val/test.
  2. Persiste CHAQUE trade en JSONL (logs/validation/conditional_edge_trades/)
     avec le vecteur de features a l'instant d'entree.
  3. Analyse par DECILES : pour chaque feature, EV brute + IC95 block
     bootstrap par decile, sur chaque split (train = decouverte, val/test =
     confirmation).
  4. Modele JOINT : HistGradientBoostingRegressor (max_depth<=3, reg forte)
     entraine sur TRAIN UNIQUEMENT pour predire pnl_brut ; applique gele a
     val et test ; mesure l'EV du decile superieur PREDIT.

PRE-ENREGISTREMENT (avant execution)
------------------------------------
Decision sur le modele joint UNIQUEMENT (l'analyse par deciles est
exploratoire, train-only pour la decouverte) :
  4 tests = 2 actifs x {val, test}, alpha = 0.05/4 = 0.0125.
GO si, sur val ET test, pour au moins un actif :
    EV_brut(top decile predit) > 0.15 %
    ET IC95_bas > 0.04 %   (= cout maker aller-retour reel 0.02% x 2)
    ET n >= 200 par split
    ET meme signe de l'EV top-decile sur BTC ET DOGE (coherence inter-actifs)
Sinon NO-GO : l'edge n'est pas recuperable a partir de ces features a cet
horizon -> la fermeture OHLCV/5m devient honnete, la suite est
microstructure ou arret.

Discipline : aucune re-optimisation sur val/test ; le modele est gele apres
le train ; frais non reduits (on juge l'EV BRUTE contre un plancher de
cout reel, pas contre un cout qu'on aurait negocie).

Lecture seule. Sortie JSON logs/validation/conditional_edge_<ts>.json +
JSONL par trade + resume stdout.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_mtf_confluence import load_aligned, OUT_DIR  # noqa: E402
from diag_mtf_confluence_v2 import cross_up_events  # noqa: E402
from diag_spot_mirror_exit import (  # noqa: E402
    cross_down_events, get_masks_mirror, spot_cycle, boot_means, ci_p,
    STRIDE, H_MAX)
from _asset_guard import get_launcher_assets  # noqa: E402

# ---------------------------------------------------------------- parametres
ASSETS = list(get_launcher_assets())
SPLITS = ["train", "val", "test"]

# Features a l'entree (colonnes du df aligne 5m/1h/4h, suffixees par TF).
FEATURES = [
    "atr_pct", "atr_pct_1h", "atr_pct_4h",
    "ema_20_ratio", "ema_50_ratio_1h", "ema_100_ratio_4h",
    "macdh_12_26_9", "macdh_21_42_9_1h", "macdh_26_52_18_4h",
    "rsi_14", "rsi_21_1h", "rsi_28_4h",
    "adx_14", "adx_14_1h", "adx_14_4h",
    "bb_percent_b_20_2", "bb_percent_b_20_2_1h", "bb_percent_b_20_2_4h",
    "bb_width_20_2_4h",
    "volume_ratio_20", "volume_ratio_20_1h", "volume_ratio_20_4h",
    "volatility_ratio_14_50", "volatility_ratio_14_50_1h",
    "vwap_ratio", "vwap_ratio_1h",
    "log_return_1h", "log_return_4h",
    "di_delta_4h", "obv_slope_4h",
]

N_DECILES = 10
N_MIN_ABS = 30
N_MIN_POWER = 200                 # puissance pre-enregistree par split OOS
TOP_DECILE_EV_MIN = 0.0015        # 0.15 % — seuil economique GO
TOP_DECILE_IC_LOW_MIN = 0.0004    # 0.04 % — plancher = cout maker RT reel
N_TESTS = 4                       # 2 actifs x {val, test}
ALPHA = 0.05 / N_TESTS            # 0.0125

GBM_PARAMS = dict(max_depth=3, learning_rate=0.05, max_iter=300,
                  l2_regularization=1.0, min_samples_leaf=50,
                  early_stopping=False, random_state=20260916)

TRADES_DIR = OUT_DIR / "conditional_edge_trades"


# ---------------------------------------------------------------- pipeline
def run_state_machine(asset, split):
    """Rejoue la machine a etats spot C1 et collecte trades + features."""
    df = load_aligned(asset, split)
    if df is None:
        return None
    df = df.dropna(subset=["ema_100_ratio_4h", "ema_50_ratio_1h",
                           "macdh_12_26_9", "atr_pct"])
    n = len(df)
    if n < H_MAX + 50:
        print(f"  [skip] {asset}/{split}: rows={n} < H_MAX+50", flush=True)
        return None

    close = df["close"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    regime4h = df["ema_100_ratio_4h"].to_numpy(np.float64)
    atr = df["atr_pct"].to_numpy(np.float64)
    ev_up = cross_up_events(df)
    ev_dn = cross_down_events(df)
    aligne_up, _ = get_masks_mirror(df)

    ent, ext, pnl, reasons = spot_cycle(
        close, low, regime4h, atr, ev_up, ev_dn, aligne_up,
        n, H_MAX, k_sl=None)                     # C1 : sans SL

    feats = {}
    missing = []
    for f in FEATURES:
        if f in df.columns:
            feats[f] = df[f].to_numpy(np.float64)
        else:
            missing.append(f)
    if missing:
        print(f"  [warn] features absentes ({asset}/{split}): {missing}",
              flush=True)
    # heure UTC de la barre d'entree (cyclique)
    idx = df.index
    hour = (idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0)
    feats["hour_utc_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feats["hour_utc_cos"] = np.cos(2 * np.pi * hour / 24.0)
    feat_names = sorted(feats.keys())

    X = np.column_stack([feats[f][ent] for f in feat_names])
    ts_entry = idx[ent].astype("datetime64[ms]").astype(str)

    print(f"  [run] {asset}/{split}: {len(ent)} trades, "
          f"{len(feat_names)} features, EVbrut={pnl.mean():+.4%}", flush=True)
    return {"asset": asset, "split": split, "n": int(len(ent)),
            "X": X, "y": pnl, "feat_names": feat_names,
            "entries": ent, "exits": ext, "reasons": reasons,
            "ts_entry": ts_entry, "holding": ext - ent}


def persist_trades(cell):
    """JSONL : un trade par ligne, features a l'entree."""
    TRADES_DIR.mkdir(parents=True, exist_ok=True)
    path = TRADES_DIR / f"{cell['asset']}_{cell['split']}.jsonl"
    with open(path, "w") as fh:
        for i in range(cell["n"]):
            rec = {"asset": cell["asset"], "split": cell["split"],
                   "entry_ts": cell["ts_entry"][i],
                   "entry_idx": int(cell["entries"][i]),
                   "exit_idx": int(cell["exits"][i]),
                   "holding_bars": int(cell["holding"][i]),
                   "exit_reason": str(cell["reasons"][i]),
                   "pnl_brut": float(cell["y"][i]),
                   "features": {f: float(cell["X"][i, j])
                                for j, f in enumerate(cell["feat_names"])}}
            fh.write(json.dumps(rec) + "\n")
    return str(path)


# ------------------------------------------------------------- analyse deciles
def decile_analysis(y, X, feat_names):
    """EV brute + IC95 bootstrap par decile de chaque feature."""
    out = {}
    n = len(y)
    for j, f in enumerate(feat_names):
        x = X[:, j]
        edges = np.quantile(x, np.linspace(0, 1, N_DECILES + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        dec = np.clip(np.digitize(x, edges[1:-1]), 0, N_DECILES - 1)
        rows = []
        for d in range(N_DECILES):
            m = dec == d
            if m.sum() < N_MIN_ABS:
                rows.append({"decile": d, "n": int(m.sum()),
                             "skipped": f"n<{N_MIN_ABS}"})
                continue
            mean = float(y[m].mean())
            mb = boot_means(y[m])
            lo, hi, p = ci_p(mb)
            rows.append({"decile": d, "n": int(m.sum()), "ev_brut": mean,
                         "ci95_low": lo, "ci95_high": hi, "p_boot": p})
        out[f] = rows
    return out


def spread_score(dec_rows):
    """Amplitude max entre meilleur et pire decile (decouverte, train)."""
    evs = [(r["ev_brut"], r["ci95_low"]) for r in dec_rows
           if "ev_brut" in r]
    if len(evs) < 5:
        return None
    best = max(evs, key=lambda t: t[0])
    worst = min(evs, key=lambda t: t[0])
    return {"spread": best[0] - worst[0], "best_ev": best[0],
            "best_ci95_low": best[1]}


# ------------------------------------------------------------- modele joint
def joint_model(train, val, test):
    """GBM gele sur train, applique a val/test ; EV du top decile predit."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(**GBM_PARAMS)
    model.fit(train["X"], train["y"])
    res = {}
    for name, cell in (("val", val), ("test", test)):
        if cell is None or cell["n"] < N_MIN_ABS:
            res[name] = {"skipped": "cellule absente ou n<30"}
            continue
        score = model.predict(cell["X"])
        thr = np.quantile(score, 1.0 - 1.0 / N_DECILES)
        top = score >= thr
        y_top = cell["y"][top]
        if len(y_top) < N_MIN_ABS:
            res[name] = {"n_top": int(len(y_top)), "skipped": "n_top<30"}
            continue
        mb = boot_means(y_top)
        lo, hi, p = ci_p(mb)
        res[name] = {
            "n_total": int(cell["n"]), "n_top": int(len(y_top)),
            "ev_brut_top": float(y_top.mean()),
            "ev_brut_all": float(cell["y"].mean()),
            "ci95_low": lo, "ci95_high": hi, "p_boot": p,
            "power_ok": bool(len(y_top) >= N_MIN_POWER or
                             cell["n"] >= N_MIN_POWER),
            "go_cell": bool(y_top.mean() > TOP_DECILE_EV_MIN
                            and lo > TOP_DECILE_IC_LOW_MIN
                            and p < ALPHA)}
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[params] features={len(FEATURES)}+2h  deciles={N_DECILES}  "
          f"alpha={ALPHA:.4f} (0.05/{N_TESTS})  GO: EVtop>{TOP_DECILE_EV_MIN:.2%} "
          f"ET IC95bas>{TOP_DECILE_IC_LOW_MIN:.2%} ET n>={N_MIN_POWER} "
          f"ET val&test", flush=True)

    data = {}
    trade_files = []
    for asset in ASSETS:
        for split in SPLITS:
            cell = run_state_machine(asset, split)
            if cell is not None:
                data[(asset, split)] = cell
                trade_files.append(persist_trades(cell))
    print(f"[persist] {len(trade_files)} fichiers JSONL trades", flush=True)

    # ---- analyse par deciles (exploratoire ; decouverte sur train) ------
    print("\n===== ANALYSE PAR DECILES (train = decouverte) =====")
    decile_report = {}
    for asset in ASSETS:
        for split in SPLITS:
            cell = data.get((asset, split))
            if cell is None:
                continue
            dec = decile_analysis(cell["y"], cell["X"], cell["feat_names"])
            decile_report[f"{asset}/{split}"] = dec
            if split == "train":
                spreads = {f: spread_score(rows) for f, rows in dec.items()}
                top = sorted(((f, s) for f, s in spreads.items() if s),
                             key=lambda t: -t[1]["spread"])[:6]
                print(f"\n  {asset}/train — plus forts spreads deciles :")
                for f, s in top:
                    print(f"    {f:<28} spread={s['spread']:+.4%}  "
                          f"best_ev={s['best_ev']:+.4%}  "
                          f"best_IC95bas={s['best_ci95_low']:+.4%}",
                          flush=True)

    # ---- modele joint (decision, gele sur train) -------------------------
    print("\n===== MODELE JOINT GBM (fit train, evalue val/test) =====")
    joint_report = {}
    for asset in ASSETS:
        tr, va, te = (data.get((asset, "train")), data.get((asset, "val")),
                      data.get((asset, "test")))
        if tr is None:
            continue
        jr = joint_model(tr, va, te)
        joint_report[asset] = jr
        for split in ("val", "test"):
            r = jr.get(split, {})
            if "ev_brut_top" not in r:
                print(f"  {asset}/{split}: {r}", flush=True)
                continue
            tag = ("GO" if r["go_cell"] else
                   "sous-puissant" if (r["ev_brut_top"] > TOP_DECILE_EV_MIN
                                       and r["ci95_low"] >
                                       TOP_DECILE_IC_LOW_MIN)
                   else "fail")
            print(f"  {asset}/{split}: top-decile EVbrut="
                  f"{r['ev_brut_top']:+.4%} (all={r['ev_brut_all']:+.4%})  "
                  f"IC95bas={r['ci95_low']:+.4%}  p={r['p_boot']:.4f}  "
                  f"n_top={r['n_top']}/{r['n_total']}  {tag}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = {"generated_utc": stamp,
           "params": {"features": FEATURES, "n_deciles": N_DECILES,
                      "alpha": ALPHA, "n_tests": N_TESTS,
                      "top_decile_ev_min": TOP_DECILE_EV_MIN,
                      "top_decile_ic_low_min": TOP_DECILE_IC_LOW_MIN,
                      "n_min_power": N_MIN_POWER, "gbm": GBM_PARAMS,
                      "state_machine": "spot_mirror C1 (achat cross_up+"
                      "MTF_ALIGNE -> vente cross_down/invalidation 4h/"
                      "timeout 2016), stride interne sequentiel"},
           "trade_files": trade_files,
           "deciles": decile_report,
           "joint": joint_report}
    path = OUT_DIR / f"conditional_edge_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nRapport JSON : {path}")

    # ---- verdict pre-enregistre ------------------------------------------
    print("\n===== VERDICT PRE-ENREGISTRE (ETAPE 1) =====")
    go_assets = []
    signs = {}
    for asset, jr in joint_report.items():
        va, te = jr.get("val", {}), jr.get("test", {})
        ok = bool(va.get("go_cell")) and bool(te.get("go_cell"))
        if ok:
            go_assets.append(asset)
        for split, r in (("val", va), ("test", te)):
            if "ev_brut_top" in r:
                signs.setdefault(split, []).append(
                    np.sign(r["ev_brut_top"]))
    coherent = all(len(s) == len(ASSETS) and (s > 0).all()
                   for s in (np.asarray(v) for v in signs.values())
                   ) if signs else False
    if go_assets and coherent:
        print(f"==> GO : edge concentre recuperable ({', '.join(go_assets)}), "
              "signe coherent val/test et inter-actifs. Le GO PPO est "
              "justifie sur base mesuree -> ETAPES 2-5.")
    elif go_assets:
        print(f"==> GO PARTIEL : {', '.join(go_assets)} passent val+test mais "
              "coherence inter-actifs non satisfaite. Approfondir avant GO.")
    else:
        print("==> NO-GO : l'EV des trades n'est pas recuperable de facon "
              "concentree a partir de ces features a cet horizon. Fermeture "
              "honnete de la branche OHLCV/5m ; la suite est microstructure "
              "ou arret.")


if __name__ == "__main__":
    main()
