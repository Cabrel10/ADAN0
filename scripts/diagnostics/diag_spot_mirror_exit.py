#!/usr/bin/env python3
"""diag_spot_mirror_exit.py — sonde MIROIR SPOT PUR (achat ET vente, jamais short).

Pourquoi cette sonde existe
---------------------------
Trois faits mesures dans les sondes precedentes rendent cette experience
obligatoire avant toute autre :

  1. `fee_sensitivity_run.log` : EVbrut == BE_cost partout ; le meilleur
     budget de frais OOS est +0,041 % (DOGE B2 test) — il n'y a quasi pas
     d'edge BRUT long a proteger. La piste "baisser les frais" est
     structurellement morte pour l'entree longue seule.
  2. `mtf_confluence_v2_run_BINANCE.log` : monotonie b1 < 0 (BTC p=0,0005 ;
     DOGE p=0,011) — plus la confluence MTF haussiere est forte, plus l'EV
     baisse legerement. Le "biais predictif positif faible" est infirme.
  3. Angle mort structurel : TOUTES les sondes (R1-R4, MFE/MAE, LOW-MAE,
     fee_sensitivity) sont LONG-ONLY (cross_up, MFE vers le haut, SL dessous
     TP dessus). Le constat N6 "l'EV suit le signe du B&H du split" est
     alors presque une tautologie : un systeme qui ne peut qu'acheter a
     mecaniquement une EV correlee au marche.

Contrainte utilisateur (decision d'univers) : **TRADING SPOT PUR**.
Pas de short. Pas de perp. Un signal baissier ne s'encaisse PAS en vendant
a decouvert : il vaut comme signal de **VENTE de l'actif detenu** (ou de
non-achat) — la valeur evitee vaut -E[r_fwd]. En spot pur, la strategie
complete est le CYCLE : ACHETER sur signal haussier, VENDRE sur signal
baissier, etre FLAT le reste du temps. C'est exactement ce cycle qui n'a
jamais ete mesure en dehors du PPO (dont les sorties etaient forcees par
des boites TP/SL ou par 99 % HOLD).

Ce que cette sonde mesure (AUCUNE strategie nouvelle optimisee, AUCUN reward,
AUCUN PPO, lecture seule) :

  EXP-A  Valeur predictive du signal de VENTE miroir.
         cross_down MACD 5m (miroir exact de cross_up_events) + MTF BAISSIER
         (ema_100_ratio_4h < 1 ET ema_50_ratio_1h < 1 ET macdh < 0, miroir
         exact de MTF_ALIGNE). Entrees non superposees stride 40 (protocole
         identique a fee_sensitivity). Mesure : r_fwd a H_REF=288 (24h,
         fixe a priori) + descriptif multi-horizons {48, 288, 576, 2016}.
         EV_vente = -E[r_fwd]. Controles : bucket UP+ALIGNE (miroir) et
         BLIND (toutes barres stride 40) sur les memes splits, car N6 a
         montre que le signe du split domine : la statistique decisive est
         AUSSI r_fwd(BLIND) - r_fwd(DOWN) > 0 (le signal bat l'aveugle).

  EXP-C  La strategie SPOT COMPLETE achat->vente, en machine a etats
         sequentielle (trades non superposes par construction) :
           ENTREE  : cross_up 5m ET MTF_ALIGNE, execution a close[t]
           SORTIE  : premier evenement parmi
                     - cross_down 5m           (vente sur signal miroir)
                     - ema_100_ratio_4h < 1.0  (invalidation structurelle)
                     - SL = k_sl x ATR entree  (variante C2 ; C1 sans SL)
                     - timeout H_MAX = 2016 barres (7 j)
           C1 : pas de SL. C2 : SL = 3 x ATR (le B3 le moins mauvais).
         PnL BRUT simule une fois ; EV nette a cout c = EV_brut - c
         (translation exacte, meme methode que diag_fee_sensitivity).
         Grille de couts RT : 0,40 % / 0,10 % / 0,05 % / 0.
         B&H du split rapporte pour le contexte N6.

PRE-ENREGISTREMENT (avant toute execution)
------------------------------------------
Espace de tests du verdict :
  Q1 (signal de vente) : 2 actifs x {train, val, test} = 6 tests
       statistique : EV_vente(H=288) ; PASS si EV_vente > 0 ET
       IC95_bas(bootstrap) > 0 ET p < alpha.
  Q2 (strategie spot complete) : 2 variantes (C1, C2) x 2 actifs x
       {val, test} = 8 cellules au cout de deblocage 0,10 % RT ;
       PASS si EV_net > 0 ET IC95_bas > 0 ET p < alpha.
  alpha = 0.05 / 14 = 0.003571  (Bonferroni sur l'espace total rapporte,
       meme discipline que les 0.05/12 de fee_sensitivity ; PAS de seuil
       plus genereux a posteriori).
Puissance : n >= 200 exige pour un GO (N_MIN_POWER du projet). Si
  30 <= n < 200 et IC95_bas > 0 : "SIGNAL SOUS-PUISSANT — prolonger
  l'historique avant GO", ce n'est PAS un GO. n < 30 : non rapporte.

GO global si : Q1 passe sur >= 1 actif (val ET test, signe coherent train)
OU Q2 passe sur >= 1 (variante, actif) en val ET test.
NO-GO sinon. En cas de NO-GO : l'architecture OHLCV/5m/spot est close pour
les deux directions ; il ne reste que microstructure (nouvelle information)
ou arret de la branche.

Honnetete statistique : univers launcher UNIQUEMENT via _asset_guard ;
aucune liste d'actifs locale ; train sert de descriptif, le verdict Q2 se
joue sur val ET test ; les trades EXP-C longue duree ont des PnL
autocorreles d'une fenetre a l'autre — block bootstrap (blocs de 5,
B=2000) obligatoire, IC a lire comme indicatif.

Lecture seule. Sortie JSON logs/validation/spot_mirror_<ts>.json + stdout.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_mtf_confluence import load_aligned, OUT_DIR  # noqa: E402
from diag_mtf_confluence_v2 import (  # noqa: E402
    cross_up_events, select_nonoverlapping)
from _asset_guard import get_launcher_assets  # noqa: E402

# ---------------------------------------------------------------- parametres
ASSETS = list(get_launcher_assets())
SPLITS = ["train", "val", "test"]

STRIDE = 40                     # identique au protocole v2 / fee_sensitivity
H_REF = 288                     # horizon de reference Q1 (24h), fixe a priori
H_DESC = [48, 288, 576, 2016]   # descriptif multi-horizons (pas de verdict)
H_MAX = 2016                    # timeout strategie spot (7 j), = fee_sensitivity
K_SL_C2 = 3.0                   # SL variante C2 (= B3, le moins mauvais)

COST_GRID = [0.004, 0.001, 0.0005, 0.0]   # RT : actuel / deblocage / maker / 0
COST_UNLOCK = 0.001             # 0,10 % RT — borne de la condition de deblocage

N_BOOT = 2000
BLOCK = 5
N_MIN_ABS = 30                  # plancher absolu pour rapporter un chiffre
N_MIN_POWER = 200               # puissance pre-enregistree pour un GO

# Pre-enregistrement : 6 tests Q1 + 8 cellules Q2 = 14 -> Bonferroni
N_TESTS_TOTAL = 14
ALPHA = 0.05 / N_TESTS_TOTAL    # 0.003571

RNG = np.random.default_rng(20260916)


# ---------------------------------------------------------------- utilitaires
def cross_down_events(df):
    """Trigger evenementiel 5m : croisement BAISSIER du histogramme MACD.

    Miroir exact de cross_up_events (diag_mtf_confluence_v2).
    """
    macdh = df["macdh_12_26_9"].to_numpy(np.float64)
    ev = np.zeros(len(df), dtype=bool)
    ev[1:] = (macdh[1:] < 0) & (macdh[:-1] >= 0)
    return ev


def get_masks_mirror(df):
    """Masques d'etat MTF : aligne haussier (R1) et son miroir baissier."""
    f4 = df["ema_100_ratio_4h"].to_numpy(np.float64)
    f1 = df["ema_50_ratio_1h"].to_numpy(np.float64)
    m5 = df["macdh_12_26_9"].to_numpy(np.float64)
    aligne_up = (f4 > 1.0) & (f1 > 1.0) & (m5 > 0)
    aligne_down = (f4 < 1.0) & (f1 < 1.0) & (m5 < 0)
    return aligne_up, aligne_down


def boot_means(x, n_boot=N_BOOT, block=BLOCK):
    """Moyennes bootstrap en blocs (meme schema que diag_fee_sensitivity)."""
    n = len(x)
    if n < 5:
        return None
    n_blocks = int(np.ceil(n / block))
    starts = RNG.integers(0, max(1, n - block + 1), size=(n_boot, n_blocks))
    arange = np.arange(block)[None, :]
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = (starts[b][:, None] + arange).ravel()[:n]
        idx = np.clip(idx, 0, n - 1)
        means[b] = x[idx].mean()
    return means


def ci_p(means):
    """IC95 + p bilaterale vs 0 a partir de moyennes bootstrap."""
    if means is None:
        return (np.nan, np.nan, np.nan)
    lo, hi = np.quantile(means, [0.025, 0.975])
    p_two = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return (float(lo), float(hi), float(min(1.0, p_two)))


# ---------------------------------------------------------------- EXP-A
def forward_returns(close, entries, H):
    """r_fwd(t, H) = close[t+H]/close[t] - 1 sur les entrees valides."""
    e = entries[entries + H < len(close)]
    if len(e) == 0:
        return np.array([]), e
    return close[e + H] / close[e] - 1.0, e


def exp_a_bucket(close, entries, label):
    """Valeur predictive d'un bucket : EV_vente a H_REF + descriptif."""
    n_raw = len(entries)
    out = {"label": label, "n_trigger": int(n_raw)}
    if n_raw < N_MIN_ABS:
        out["skipped"] = f"n<{N_MIN_ABS}"
        return out
    r_ref, e_ref = forward_returns(close, entries, H_REF)
    if len(r_ref) < N_MIN_ABS:
        out["skipped"] = f"n(H={H_REF})<{N_MIN_ABS}"
        return out
    ev_vente = -r_ref                       # spot pur : valeur de la VENTE
    m = boot_means(ev_vente)
    lo, hi, p = ci_p(m)
    out["H_ref"] = H_REF
    out["n"] = int(len(r_ref))
    out["power_ok"] = bool(len(r_ref) >= N_MIN_POWER)
    out["ev_vente"] = float(ev_vente.mean())        # = -E[r_fwd]
    out["r_fwd_mean"] = float(r_ref.mean())
    out["r_fwd_med"] = float(np.median(r_ref))
    out["ci95_ev_vente"] = [lo, hi]
    out["p_boot"] = p
    out["pass_q1"] = bool(ev_vente.mean() > 0 and lo > 0 and p < ALPHA)
    # descriptif multi-horizons (aucun verdict)
    desc = {}
    for H in H_DESC:
        r_h, _ = forward_returns(close, entries, H)
        if len(r_h) >= N_MIN_ABS:
            desc[str(H)] = {"n": int(len(r_h)),
                            "r_fwd_mean": float(r_h.mean()),
                            "ev_vente": float(-r_h.mean())}
    out["horizons_desc"] = desc
    return out


# ---------------------------------------------------------------- EXP-C
def spot_cycle(close, low, regime4h, atr, ev_up, ev_dn, aligne_up,
               n, h_max, k_sl=None):
    """Machine a etats SPOT PUR : achat cross_up+ALIGNE -> vente sur
    cross_down / invalidation 4h / (SL optionnel) / timeout. FLAT sinon.

    Jamais short : entre deux trades la position est 0. Trades sequentiels,
    non superposes par construction. Execution a la cloture de la barre du
    signal (convention identique aux autres sondes).
    """
    entries, exits, pnl, reasons = [], [], [], []
    i = 0
    last = n - 1
    while i < last:
        if ev_up[i] and aligne_up[i]:
            entry = close[i]
            sl = k_sl * atr[i] if k_sl is not None else None
            h_end = min(h_max, last - i)
            exit_i = i + h_end
            reason = "timeout"
            for h in range(1, h_end + 1):
                j = i + h
                if sl is not None and low[j] <= entry * (1.0 - sl):
                    exit_i, reason = j, "sl"
                    break
                if ev_dn[j]:
                    exit_i, reason = j, "cross_down"
                    break
                if regime4h[j] < 1.0:
                    exit_i, reason = j, "invalidation_4h"
                    break
            p = (-sl if reason == "sl" else close[exit_i] / entry - 1.0)
            entries.append(i)
            exits.append(exit_i)
            pnl.append(p)
            reasons.append(reason)
            i = exit_i + 1
        else:
            i += 1
    return (np.array(entries), np.array(exits), np.array(pnl),
            np.array(reasons, dtype=object))


def sweep_costs_spot(pnl_brut, label):
    """EV nette a chaque cout RT par translation exacte (cf fee_sensitivity)."""
    n = len(pnl_brut)
    out = {"label": label, "n": int(n),
           "ev_brut": float(pnl_brut.mean()),
           "be_cost_rt": float(pnl_brut.mean()),
           "wr_brut": float((pnl_brut > 0).mean()),
           "power_ok": bool(n >= N_MIN_POWER),
           "levels": {}}
    means = boot_means(pnl_brut)
    if means is None:
        out["skipped"] = "n<5"
        return out
    for c in COST_GRID:
        m_net = means - c
        lo, hi, p = ci_p(m_net)
        out["levels"][f"{c:.4%}"] = {
            "ev_net": float(pnl_brut.mean() - c),
            "ci95_low": lo, "ci95_high": hi, "p_boot": p}
    return out


# ---------------------------------------------------------------- pipeline
def run_cell(asset, split):
    df = load_aligned(asset, split)
    if df is None:
        return None
    df = df.dropna(subset=["ema_100_ratio_4h", "ema_50_ratio_1h",
                           "macdh_12_26_9", "atr_pct"])
    n = len(df)
    print(f"[run] {asset}/{split}: {n} barres 5m", flush=True)
    if n < H_MAX + 50:
        return {"asset": asset, "split": split,
                "skipped": f"rows={n} < H_MAX+50"}

    close = df["close"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    regime4h = df["ema_100_ratio_4h"].to_numpy(np.float64)
    atr = df["atr_pct"].to_numpy(np.float64)

    ev_up = cross_up_events(df)
    ev_dn = cross_down_events(df)
    aligne_up, aligne_down = get_masks_mirror(df)

    res = {"asset": asset, "split": split, "rows_5m": n,
           "n_cross_up": int(ev_up.sum()),
           "n_cross_down": int(ev_dn.sum()),
           "bnh_split": float(close[-1] / close[0] - 1.0)}

    # ---- EXP-A : valeur du signal de vente (miroir) --------------------
    ent_dn = select_nonoverlapping(ev_dn & aligne_down, n, STRIDE)
    ent_up = select_nonoverlapping(ev_up & aligne_up, n, STRIDE)
    ent_blind = select_nonoverlapping(np.ones(n, dtype=bool), n, STRIDE)
    exp_a = {"DOWN_BAISSIER": exp_a_bucket(close, ent_dn, "DOWN_BAISSIER"),
             "UP_ALIGNE": exp_a_bucket(close, ent_up, "UP_ALIGNE"),
             "BLIND": exp_a_bucket(close, ent_blind, "BLIND")}
    # statistique decisive N6-proof : le signal DOWN bat-il l'aveugle ?
    r_dn, _ = forward_returns(close, ent_dn, H_REF)
    r_bl, _ = forward_returns(close, ent_blind, H_REF)
    if len(r_dn) >= N_MIN_ABS and len(r_bl) >= N_MIN_ABS:
        m_dn = boot_means(r_dn)
        m_bl = boot_means(r_bl)
        if m_dn is not None and m_bl is not None:
            diff = m_bl - m_dn          # > 0 : le signal DOWN est pire que l'aveugle
            lo, hi, p = ci_p(diff)
            exp_a["DELTA_BLIND_minus_DOWN"] = {
                "delta": float(r_bl.mean() - r_dn.mean()),
                "ci95": [lo, hi], "p_boot": p,
                "signal_beats_blind": bool(lo > 0)}
    res["exp_a_signal_vente"] = exp_a
    for lbl, b in exp_a.items():
        if lbl.startswith("DELTA") or "ev_vente" not in b:
            continue
        print(f"  A/{lbl:<14} n={b['n']:>5}  r_fwd={b['r_fwd_mean']:+.4%}  "
              f"EV_vente={b['ev_vente']:+.4%}  IC95bas={b['ci95_ev_vente'][0]:+.4%}  "
              f"p={b['p_boot']:.4f}  {'PASS-Q1' if b['pass_q1'] else 'fail'}",
              flush=True)
    if "DELTA_BLIND_minus_DOWN" in exp_a:
        d = exp_a["DELTA_BLIND_minus_DOWN"]
        print(f"  A/DELTA BLIND-DOWN = {d['delta']:+.4%}  IC95={d['ci95']}  "
              f"{'signal>aveugle' if d['signal_beats_blind'] else 'PAS mieux que aveugle'}",
              flush=True)

    # ---- EXP-C : strategie spot complete achat->vente -------------------
    exp_c = {}
    for name, k_sl in (("C1_pas_de_sl", None), ("C2_sl_3atr", K_SL_C2)):
        ent, ext, pnl, reasons = spot_cycle(
            close, low, regime4h, atr, ev_up, ev_dn, aligne_up,
            n, H_MAX, k_sl)
        if len(pnl) < N_MIN_ABS:
            exp_c[name] = {"n": int(len(pnl)), "skipped": f"n<{N_MIN_ABS}"}
            print(f"  C/{name:<14} n={len(pnl)}  (ignore)", flush=True)
            continue
        s = sweep_costs_spot(pnl, name)
        s["holding_med"] = float(np.median(ext - ent))
        s["exit_reasons"] = {r: float((reasons == r).mean())
                             for r in np.unique(reasons)}
        exp_c[name] = s
        lvl = s["levels"].get(f"{COST_UNLOCK:.4%}", {})
        print(f"  C/{name:<14} n={s['n']:>5}  EVbrut={s['ev_brut']:+.4%}  "
              f"BE={s['be_cost_rt']:+.4%}  "
              f"EV@0.10%RT={lvl.get('ev_net', float('nan')):+.4%}  "
              f"IC95bas={lvl.get('ci95_low', float('nan')):+.4%}  "
              f"hold_med={s['holding_med']:.0f}b", flush=True)
    res["exp_c_strategie_spot"] = exp_c
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[params] stride={STRIDE}  H_ref={H_REF}  H_max={H_MAX}  "
          f"k_sl_C2={K_SL_C2}  cost_grid={['%.4f' % c for c in COST_GRID]}  "
          f"n_boot={N_BOOT} block={BLOCK}", flush=True)
    print(f"[pre-enregistre] Q1: 6 tests + Q2: 8 cellules = {N_TESTS_TOTAL} "
          f"-> alpha Bonferroni = {ALPHA:.6f}  (GO exige n>={N_MIN_POWER}, "
          f"IC95bas>0, p<alpha, val ET test)", flush=True)

    results = []
    for asset in ASSETS:
        for split in SPLITS:
            r = run_cell(asset, split)
            if r is not None:
                results.append(r)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = {"generated_utc": stamp,
           "contrainte": "SPOT PUR — achat et vente de l'actif detenu, "
                         "jamais short a decouvert",
           "params": {"stride": STRIDE, "H_ref": H_REF, "H_desc": H_DESC,
                      "H_max": H_MAX, "k_sl_C2": K_SL_C2,
                      "cost_grid_rt": COST_GRID, "cost_unlock_rt": COST_UNLOCK,
                      "n_boot": N_BOOT, "block": BLOCK,
                      "n_min_abs": N_MIN_ABS, "n_min_power": N_MIN_POWER,
                      "n_tests_total": N_TESTS_TOTAL, "alpha_bonferroni": ALPHA},
           "results": results}
    path = OUT_DIR / f"spot_mirror_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nRapport JSON : {path}")

    # ---------------- verdict pre-enregistre ---------------------------
    print("\n===== VERDICT PRE-ENREGISTRE =====")
    print(f"(alpha Bonferroni = {ALPHA:.6f} = 0.05/{N_TESTS_TOTAL})\n")

    print("-- Q1 : le signal de VENTE (cross_down + MTF baissier) vaut-il "
          "quelque chose ? --")
    q1_pass_any = False
    for asset in ASSETS:
        row = []
        pass_val = pass_test = False
        signes = {}
        for split in SPLITS:
            cell = next((r for r in results
                         if r.get("asset") == asset
                         and r.get("split") == split), None)
            b = ((cell or {}).get("exp_a_signal_vente", {})
                 .get("DOWN_BAISSIER", {}))
            if "ev_vente" not in b:
                row.append(f"{split}:n/a")
                continue
            signes[split] = np.sign(b["ev_vente"])
            ok = (b["pass_q1"] and b.get("power_ok", False))
            if split == "val":
                pass_val = ok
            if split == "test":
                pass_test = ok
            tag = ("PASS" if ok else
                   "sous-puissant" if (b["pass_q1"] and not b["power_ok"])
                   else "fail")
            row.append(f"{split}[EVv={b['ev_vente']:+.4%} "
                       f"lo={b['ci95_ev_vente'][0]:+.4%} "
                       f"p={b['p_boot']:.4f} n={b['n']} {tag}]")
        coherent = (len(signes) == 3
                    and signes.get("train", 0) > 0
                    and signes.get("val", 0) > 0
                    and signes.get("test", 0) > 0)
        go = pass_val and pass_test and coherent
        q1_pass_any |= go
        print(f"  {asset:<18} {'  '.join(row)}  -> {'GO' if go else 'NO-GO'}")

    print("\n-- Q2 : la strategie SPOT complete (achat cross_up -> vente "
          "cross_down/invalidation) est-elle rentable a 0,10 % RT ? --")
    q2_pass_any = False
    for name in ("C1_pas_de_sl", "C2_sl_3atr"):
        for asset in ASSETS:
            row, passes = [], {}
            for split in ("val", "test"):
                cell = next((r for r in results
                             if r.get("asset") == asset
                             and r.get("split") == split), None)
                s = ((cell or {}).get("exp_c_strategie_spot", {})
                     .get(name, {}))
                lvl = s.get("levels", {}).get(f"{COST_UNLOCK:.4%}")
                if lvl is None:
                    row.append(f"{split}:n/a")
                    passes[split] = False
                    continue
                ok = (lvl["ev_net"] > 0 and lvl["ci95_low"] > 0
                      and lvl["p_boot"] < ALPHA and s.get("power_ok", False))
                passes[split] = ok
                tag = ("PASS" if ok else
                       "sous-puissant" if (lvl["ev_net"] > 0
                                           and lvl["ci95_low"] > 0
                                           and lvl["p_boot"] < ALPHA)
                       else "FAIL")
                row.append(f"{split}[EV={lvl['ev_net']:+.4%} "
                           f"lo={lvl['ci95_low']:+.4%} p={lvl['p_boot']:.4f} "
                           f"n={s['n']} {tag}]")
            go = passes.get("val") and passes.get("test")
            q2_pass_any |= bool(go)
            print(f"  {name:<14} {asset:<18} {'  '.join(row)}  -> "
                  f"{'GO' if go else 'NO-GO'}")

    print("\n==> " + (
        "GO : au moins une question passe le protocole pre-enregistre. "
        "Voir cellules ci-dessus."
        if (q1_pass_any or q2_pass_any) else
        "NO-GO : ni le signal de vente miroir ni la strategie spot complete "
        "ne passent le protocole pre-enregistre (alpha=0.05/14, val ET test, "
        "n>=200). L'architecture OHLCV/5m/spot est close DANS LES DEUX "
        "DIRECTIONS ; il ne reste que microstructure (nouvelle information) "
        "ou arret de la branche."))


if __name__ == "__main__":
    main()
