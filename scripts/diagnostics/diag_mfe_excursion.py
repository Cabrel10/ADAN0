#!/usr/bin/env python3
"""diag_mfe_excursion.py — distribution MFE/MAE longue echeance + sorties structurelles.

Pourquoi cette sonde existe (hypothese a tester, verbatim du brief) :
  R1-R4 ont mesure des EV avec geometrie SL/TP fixe sur H <= 288 barres (24h).
  Verdict NO-GO robuste sur CET espace. Mais rien n'a mesure la distribution
  COMPLETE des excursions jusqu'a 7 jours : si le marche offre des mouvements
  enormes (p99 MFE 7j ~ 8-12 %) et que l'architecture TP fixe sort a +1,2 %,
  le probleme n'est pas l'absence d'opportunite mais la conservation des
  rares trajectoires qui paient toutes les autres ("runners").

Trois experiences, AUCUNE strategie nouvelle, AUCUN reward, AUCUN PPO :

  EXP-1  Distribution MFE/MAE a H in {12, 48, 288, 576, 2016} barres 5m
         (1h / 4h / 24h / 48h / 7j) sur les MEMES entrees candidates que R4
         (cross-up MACD 5m, non superposees stride=40). Quantiles p50/p90/p95/
         p99, temps d'atteinte du pic, concentration (part du top 5 % / top 1 %
         dans l'energie MFE totale), P(MFE > 3x / 10x couts) conditionnee a
         l'etat MTF (aligne A_trend_full vs contradictoire E_countertrend).
         Classification des trajectoires A-E (bruit / petit edge / runner /
         faux signal / piege-a-SL).

  EXP-2  Test #5 du brief — "faut-il meme un TP prix fixe ?"
         Memes entrees (A_trend_full). Trois regles de sortie :
           B1  TP fixe 1,2 % / SL fixe 0,6 % (la geometrie du debat "3x frais")
           B2  SL = 0,5 x ATR, PAS de TP, sortie = invalidation du regime 4h
               (ema_100_ratio_4h repasse < 1) ou H_max = 2016
           B3  SL = 3 x ATR (large), meme invalidation structurelle
         Couts RT 0,5 % deduits. Compare : EV net, distribution, skew,
         part du top 5 % dans le PnL total, EV par barre.

  EXP-3  Test #2 du brief — delai d'entree post-trigger.
         Trigger cross-up MACD a t, entree reelle a t + d pour
         d in {0, 2, 5, 10, 20, 40}. Mesure MFE/MAE a 48 barres depuis
         l'entree reelle + EV nette (geometrie fixe B1). L'entree retardee
         capture-t-elle la correction de sur-reaction ?

Honnetete statistique :
  - univers launcher UNIQUEMENT via _asset_guard (jamais de liste locale) ;
  - memes entrees non superposees que le protocole v2 (stride=40) ;
  - train = mesure principale, val/test = confirmation des ordres de grandeur ;
  - les distributions longues sont autocorrelees (fenetres qui se chevauchent
    a l'interieur d'un trade) : on rapporte des DISTRIBUTIONS, pas des tests
    d'hypothese d'EV — le block bootstrap reste utilise pour EXP-2 (PnL par
    trade independant a stride 40... en pratique les sorties longues se
    chevauchent : IC a lire comme indicatif, verdict sur ordres de grandeur).

Lecture seule. Sortie JSON logs/validation/ + resume stdout.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_mtf_confluence import load_aligned, OUT_DIR  # noqa: E402
from diag_mtf_confluence_v2 import (  # noqa: E402
    cross_up_events, select_nonoverlapping, block_bootstrap_ci, wilson_ci,
)
from _asset_guard import get_launcher_assets  # noqa: E402

# ---------------------------------------------------------------- parametres
FEES_RT = 0.004
SLIPPAGE_RT = 0.001
COST_RT = FEES_RT + SLIPPAGE_RT          # 0,5 % round-trip (conservateur)
THRESH_DEBAT = 3 * FEES_RT               # 1,2 % — le "TP = 3x frais" du debat
THRESH_3X_COST = 3 * COST_RT             # 1,5 %
THRESH_10X_COST = 10 * COST_RT           # 5,0 %

H_MEASURE = [12, 48, 288, 576, 2016]     # 1h, 4h, 24h, 48h, 7j
H_MAX = max(H_MEASURE)
STRIDE = 40                              # identique au protocole v2
DELAYS = [0, 2, 5, 10, 20, 40]           # EXP-3

ASSETS = list(get_launcher_assets())
SPLITS = ["train", "val", "test"]

TP_FIXE, SL_FIXE = 0.012, 0.006          # B1 — la geometrie du debat
K_SL_B2, K_SL_B3 = 0.5, 3.0              # B2/B3 — SL en multiples d'ATR entree


# ------------------------------------------------------------- coeur vectoriel
def excursion_cumulative(high, low, close, entries, h_max, snapshots):
    """Un seul passage h=1..h_max ; MFE/MAE courants + snapshots aux horizons.

    Retourne dict {H: (mfe, mae)} + time-to-peak (barres du MFE a h_max)
    + rendement final a h_max. Vectorise sur les entrees a chaque h."""
    entries = entries[entries + h_max < len(close)]
    if len(entries) == 0:
        return {}, None, None, entries
    entry = close[entries]
    mfe = np.zeros(len(entries))
    mae = np.zeros(len(entries))
    ttp = np.zeros(len(entries), dtype=np.int64)
    snap = {H: None for H in snapshots}
    for h in range(1, h_max + 1):
        hh = high[entries + h] / entry - 1.0
        ll = low[entries + h] / entry - 1.0
        better = hh > mfe
        ttp[better] = h
        mfe = np.maximum(mfe, hh)
        mae = np.minimum(mae, ll)
        if h in snap:
            snap[h] = (mfe.copy(), mae.copy())
    final = close[entries + h_max] / entry - 1.0
    return snap, ttp, final, entries


def quantile_block(x):
    if len(x) == 0:
        return {}
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    return {"p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p90": float(q[4]), "p95": float(q[5]),
            "p99": float(q[6]), "mean": float(x.mean())}


def concentration(mfe):
    """Part de l'energie MFE (somme des MFE positifs) portee par le top 5%/1%."""
    e = np.maximum(mfe, 0.0)
    tot = e.sum()
    if tot <= 0 or len(e) < 20:
        return {"top5_share": None, "top1_share": None}
    s = np.sort(e)[::-1]
    k5 = max(1, int(0.05 * len(s)))
    k1 = max(1, int(0.01 * len(s)))
    return {"top5_share": float(s[:k5].sum() / tot),
            "top1_share": float(s[:k1].sum() / tot)}


def prob_thresholds(mfe):
    n = len(mfe)
    if n == 0:
        return {}
    return {"P_mfe_gt_1.2pct": float((mfe > THRESH_DEBAT).mean()),
            "P_mfe_gt_3xcost": float((mfe > THRESH_3X_COST).mean()),
            "P_mfe_gt_10xcost": float((mfe > THRESH_10X_COST).mean()),
            "n": int(n)}


def classify_trajectories(mfe, mae, final, h_label):
    """Classes A-E du brief (mesurees a l'horizon de reference h_label).

    A bruit       : MFE < couts ET |MAE| < couts
    B petit edge  : MFE > 3x couts MAIS final < 0,5 x MFE (retour rapide)
    C runner      : MFE > 3x couts ET final >= 0,5 x MFE ET MAE > -couts
    D faux signal : MAE <= -couts ET MFE < couts
    E piege-a-SL  : MAE <= -couts ET MFE > 3x couts (le SL classique le tue)
    """
    n = len(mfe)
    if n == 0:
        return {}
    a = (mfe < COST_RT) & (mae > -COST_RT)
    big = mfe > THRESH_3X_COST
    b = big & (final < 0.5 * mfe)
    c = big & (final >= 0.5 * mfe) & (mae > -COST_RT)
    d = (mae <= -COST_RT) & (mfe < COST_RT)
    e = (mae <= -COST_RT) & big
    other = ~(a | b | c | d | e)
    out = {"horizon": h_label, "n": int(n)}
    for name, mask in (("A_bruit", a), ("B_petit_edge", b), ("C_runner", c),
                       ("D_faux_signal", d), ("E_piege_a_sl", e),
                       ("autres", other)):
        out[name] = {"share": float(mask.mean()), "n": int(mask.sum())}
        if mask.any():
            out[name]["mfe_med"] = float(np.median(mfe[mask]))
            out[name]["mae_med"] = float(np.median(mae[mask]))
            out[name]["final_med"] = float(np.median(final[mask]))
    return out


# ------------------------------------------------------------- EXP-2 : sorties
def first_passage_fixe(high, low, close, entries, h_max, sl, tp):
    """B1 : premier passage pessimiste TP/SL fixes. Retourne (pnl, ht, sortie)."""
    n = len(entries)
    entry = close[entries]
    pnl = np.zeros(n)
    ht = np.full(n, h_max, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    reason = np.array(["timeout"] * n, dtype=object)
    for h in range(1, h_max + 1):
        hh = high[entries + h] / entry - 1.0
        ll = low[entries + h] / entry - 1.0
        both = (~done) & (hh >= tp) & (ll <= -sl)
        pnl[both] = -sl; ht[both] = h; reason[both] = "sl_ambigu"; done |= both
        only_tp = (~done) & (hh >= tp)
        pnl[only_tp] = tp; ht[only_tp] = h; reason[only_tp] = "tp"; done |= only_tp
        only_sl = (~done) & (ll <= -sl)
        pnl[only_sl] = -sl; ht[only_sl] = h; reason[only_sl] = "sl"; done |= only_sl
    rest = ~done
    pnl[rest] = close[entries[rest] + h_max] / entry[rest] - 1.0
    return pnl, ht, reason


def sortie_invalidation(high, low, close, regime4h, entries, h_max, atr_entry,
                        k_sl):
    """B2/B3 : SL = k_sl x ATR, PAS de TP, sortie quand le regime 4h s'inverse
    (ema_100_ratio_4h < 1.0 — l'entree exigeait > 1.0) ou timeout h_max."""
    n = len(entries)
    entry = close[entries]
    sl = k_sl * atr_entry
    pnl = np.zeros(n)
    ht = np.full(n, h_max, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    reason = np.array(["timeout"] * n, dtype=object)
    for h in range(1, h_max + 1):
        ll = low[entries + h] / entry - 1.0
        hit_sl = (~done) & (ll <= -sl)
        pnl[hit_sl] = -sl[hit_sl]; ht[hit_sl] = h
        reason[hit_sl] = "sl"; done |= hit_sl
        inv = (~done) & (regime4h[entries + h] < 1.0)
        pnl[inv] = close[entries[inv] + h] / entry[inv] - 1.0
        ht[inv] = h; reason[inv] = "invalidation_4h"; done |= inv
        if done.all():
            break
    rest = ~done
    pnl[rest] = close[entries[rest] + h_max] / entry[rest] - 1.0
    return pnl, ht, reason


def resume_pnl(pnl_net, ht, reason, label):
    n = len(pnl_net)
    if n < 30:
        return {"label": label, "n": int(n), "skipped": "n<30"}
    mean, lo, hi, p = block_bootstrap_ci(pnl_net)
    winners = pnl_net > 0
    out = {"label": label, "n": int(n),
           "ev_net": mean, "ci95": [lo, hi], "p_boot": p,
           "wr": float(winners.mean()),
           "pnl_med": float(np.median(pnl_net)),
           "pnl_p90": float(np.quantile(pnl_net, 0.90)),
           "pnl_p99": float(np.quantile(pnl_net, 0.99)),
           "skew": float(((pnl_net - pnl_net.mean()) ** 3).mean()
                         / (pnl_net.std() ** 3 + 1e-18)),
           "holding_med": float(np.median(ht)),
           "ev_per_bar": float(mean / max(np.median(ht), 1.0)),
           "exit_reasons": {r: float((reason == r).mean())
                            for r in np.unique(reason)}}
    pos = pnl_net[pnl_net > 0]
    if len(pos) >= 20 and pos.sum() > 0:
        s = np.sort(pos)[::-1]
        k5 = max(1, int(0.05 * len(s)))
        out["top5_winners_share_of_gains"] = float(s[:k5].sum() / pos.sum())
    return out


# --------------------------------------------------------------- assemblage
def get_masks(df):
    """Masques d'etat MTF au moment de l'entree (memes definitions que R1)."""
    f4 = df["ema_100_ratio_4h"].to_numpy(np.float64)
    f1 = df["ema_50_ratio_1h"].to_numpy(np.float64)
    m5 = df["macdh_12_26_9"].to_numpy(np.float64)
    aligne = (f4 > 1.0) & (f1 > 1.0) & (m5 > 0)
    contradictoire = (f4 < 1.0) & (m5 > 0)
    return aligne, contradictoire


def run_split(asset, split, df):
    n = len(df)
    if n < H_MAX + 50:
        return {"asset": asset, "split": split, "skipped": f"rows={n} < H_MAX+50"}
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    regime4h = df["ema_100_ratio_4h"].to_numpy(np.float64)
    atr = df["atr_pct"].to_numpy(np.float64)

    events = cross_up_events(df)
    base_entries = select_nonoverlapping(events, n, STRIDE)
    base_entries = base_entries[base_entries + H_MAX < n]
    aligne, contradictoire = get_masks(df)

    res = {"asset": asset, "split": split, "rows_5m": n,
           "n_trigger_events": int(events.sum()),
           "n_entries_stride40": int(len(base_entries))}

    # ---------------- EXP-1 : distributions MFE/MAE longue echeance --------
    buckets = {"ALL": base_entries,
               "MTF_ALIGNE": base_entries[aligne[base_entries]],
               "MTF_CONTRADICTOIRE": base_entries[contradictoire[base_entries]]}
    exp1 = {}
    for label, entries in buckets.items():
        snap, ttp, final, entries = excursion_cumulative(
            high, low, close, entries, H_MAX, H_MEASURE)
        if len(entries) < 30:
            exp1[label] = {"n": int(len(entries)), "skipped": "n<30"}
            continue
        b = {"n": int(len(entries)), "horizons": {}}
        for H in H_MEASURE:
            mfe, mae = snap[H]
            hkey = {12: "1h", 48: "4h", 288: "24h", 576: "48h", 2016: "7j"}[H]
            b["horizons"][hkey] = {
                "mfe": quantile_block(mfe), "mae": quantile_block(mae),
                **prob_thresholds(mfe)}
            if H == 2016:
                b["concentration_7j"] = concentration(mfe)
                b["ttp_7j_barres"] = quantile_block(ttp.astype(float))
                b["classes_7j"] = classify_trajectories(
                    mfe, mae, final, "7j")
        exp1[label] = b
    res["exp1_distributions"] = exp1

    # ---- EXP-2 : test #5 — TP fixe vs invalidation structurelle (4h) -----
    ent = buckets["MTF_ALIGNE"]
    exp2 = {}
    if len(ent) >= 30:
        pnl, ht, r = first_passage_fixe(high, low, close, ent, 288,
                                        SL_FIXE, TP_FIXE)
        exp2["B1_tp_fixe_1.2_sl_0.6_H288"] = resume_pnl(
            pnl - COST_RT, ht, r, "B1")
        for k_sl, name in ((K_SL_B2, "B2_sl_0.5atr_invalidation"),
                           (K_SL_B3, "B3_sl_3atr_invalidation")):
            pnl, ht, r = sortie_invalidation(
                high, low, close, regime4h, ent, H_MAX,
                atr[ent], k_sl)
            exp2[name] = resume_pnl(pnl - COST_RT, ht, r, name)
    else:
        exp2["skipped"] = f"n_aligne={len(ent)} < 30"
    res["exp2_sortie_structurelle_vs_tp_fixe"] = exp2

    # ---- EXP-3 : test #2 — delai d'entree post-trigger (H=48) ------------
    exp3 = {}
    for d in DELAYS:
        delayed = base_entries + d
        delayed = delayed[delayed + 48 < n]
        delayed = delayed[events[delayed - d]]  # re-ancre sur le trigger
        if len(delayed) < 30:
            exp3[f"delay_{d}"] = {"n": int(len(delayed)), "skipped": "n<30"}
            continue
        snap, _, _, delayed = excursion_cumulative(
            high, low, close, delayed, 48, [48])
        mfe, mae = snap[48]
        pnl, ht, r = first_passage_fixe(high, low, close, delayed, 48,
                                        SL_FIXE, TP_FIXE)
        pnl_net = pnl - COST_RT
        mean, lo, hi, _ = block_bootstrap_ci(pnl_net)
        exp3[f"delay_{d}"] = {
            "n": int(len(delayed)),
            "mfe_4h_med": float(np.median(mfe)),
            "mae_4h_med": float(np.median(mae)),
            "ev_net_B1_4h": mean, "ci95": [lo, hi],
            "wr": float((pnl_net > 0).mean())}
    res["exp3_delai_entree"] = exp3
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for asset in ASSETS:
        for split in SPLITS:
            df = load_aligned(asset, split)
            if df is None:
                continue
            df = df.dropna(subset=["ema_100_ratio_4h", "ema_50_ratio_1h",
                                   "macdh_12_26_9", "atr_pct"])
            print(f"[run] {asset}/{split}: {len(df)} barres 5m", flush=True)
            results.append(run_split(asset, split, df))
            r = results[-1]
            if "exp1_distributions" in r:
                for label, b in r["exp1_distributions"].items():
                    if "horizons" not in b:
                        continue
                    h7 = b["horizons"].get("7j", {})
                    print(f"  {label:<20} n={b['n']:>6}  "
                          f"MFE7j p50={h7.get('mfe', {}).get('p50', float('nan')):+.3%} "
                          f"p95={h7.get('mfe', {}).get('p95', float('nan')):+.3%} "
                          f"p99={h7.get('mfe', {}).get('p99', float('nan')):+.3%} "
                          f"P(>3xco)={h7.get('P_mfe_gt_3xcost', float('nan')):.2%}",
                          flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = {"generated_utc": stamp,
           "params": {"fees_rt": FEES_RT, "slippage_rt": SLIPPAGE_RT,
                      "cost_rt": COST_RT, "H_measure": H_MEASURE,
                      "stride": STRIDE, "delays": DELAYS,
                      "tp_fixe": TP_FIXE, "sl_fixe": SL_FIXE,
                      "k_sl_b2": K_SL_B2, "k_sl_b3": K_SL_B3,
                      "thresh_debat_3x_frais": THRESH_DEBAT},
           "results": results}
    path = OUT_DIR / f"mfe_excursion_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nRapport JSON : {path}")

    # ---------------- verdict synthetique vs attentes du brief -------------
    print("\n===== VERDICT vs attentes du brief (train) =====")
    attentes = [("p50 MFE 1h ~ 0,2 %", "1h", "mfe", "p50"),
                ("p90 MFE 4h ~ 1,5 %", "4h", "mfe", "p90"),
                ("p95 MFE 24h ~ 3 %", "24h", "mfe", "p95"),
                ("p99 MFE 7j ~ 8-12 %", "7j", "mfe", "p99")]
    for r in results:
        if r.get("split") != "train" or "exp1_distributions" not in r:
            continue
        b = r["exp1_distributions"].get("ALL", {})
        print(f"  {r['asset']} (ALL, n={b.get('n')}):")
        for nom, hk, fam, q in attentes:
            v = b.get("horizons", {}).get(hk, {}).get(fam, {}).get(q)
            print(f"    {nom:<26} mesure = {v:+.3%}" if v is not None
                  else f"    {nom:<26} n/a")
        conc = b.get("concentration_7j", {})
        print(f"    top5% -> part energie MFE 7j : {conc.get('top5_share')}")
        print(f"    top1% -> part energie MFE 7j : {conc.get('top1_share')}")
        for cond in ("MTF_ALIGNE", "MTF_CONTRADICTOIRE"):
            bb = r["exp1_distributions"].get(cond, {})
            p3 = bb.get("horizons", {}).get("24h", {}).get("P_mfe_gt_3xcost")
            if p3 is not None:
                print(f"    P(MFE24h > 3x couts | {cond}) = {p3:.2%}")


if __name__ == "__main__":
    main()
