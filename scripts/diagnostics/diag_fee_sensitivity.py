#!/usr/bin/env python3
"""diag_fee_sensitivity.py — sensibilite de l'EV aux couts round-trip.

Pourquoi cette sonde existe
---------------------------
Le verdict MFE/excursion (RAPPORT_MFE_EXCURSION.md, commit 540c15e/31a46a6)
classe les voies de deblocage par leverage mesure :

  1. COUTS : 0,5 % RT mangent 2-4x le MFE 4h median (0,62 % BTC).
     "Toute reduction de fees_rt/slippage_rt deplace directement l'EV
      de toutes les variantes."

Et ETAT_DU_CODE §9 fixe la condition de deblocage explicite :

  "frais <= 0,10 % A/R avec signal net demontre sur backtest non superpose
   val ET test".

Aucune mesure n'a encore balaye le cout. Toutes les EV publiees sont a
cout fixe 0,5 % RT. Cette sonde comble ce trou.

Methode (exacte, pas une approximation)
---------------------------------------
Pour une regle de sortie donnee, le PnL brut par trade est simule UNE fois
(cout = 0). L'EV nette a un cout round-trip c est exactement :

    EV_net(c) = mean(PnL_brut) - c

Le block bootstrap est calcule UNE fois sur les PnL bruts ; les IC95 et
p-values a chaque niveau de cout s'obtiennent en translatant les moyennes
bootstrap : means_net = means_brut - c. Aucune re-simulation par niveau.

Grille de couts RT : 0,5 % (actuel) -> 0 (plancher theorique), incluant
0,10 % = la borne haute de la condition de deblocage.

Regles testees (identiques a EXP2 du rapport MFE, memes entrees MTF_ALIGNE,
stride 40, univers launcher garanti par _asset_guard) :

  B1  TP +1,2 % / SL -0,6 %, timeout 288 barres (24h) — la boite actuelle
  B2  SL 0,5 x ATR + sortie sur invalidation 4h, H_max 2016 (7j)
  B3  SL 3,0 x ATR + sortie sur invalidation 4h, H_max 2016 (7j)

Sortie : logs/validation/fee_sensitivity_<ts>.json + verdict stdout.

Criteres de verdict (pre-enregistres) :
  DEBLOQUE si au moins une regle a, sur val ET test, a cout RT <= 0,10 % :
    - EV nette > 0, ET
    - borne basse IC95 bootstrap > 0 (equivalent p_boot < 0,05).
  Sinon NO-GO maintenu, et on rapporte le cout break-even par cellule
  (= EV brute : le cout maximal que la strategie pourrait supporter).

Caveat herite du protocole MFE : les entrees sont non superposees (stride
40) mais les sorties longues (B2/B3, holding median ~42-48 barres) se
chevauchent partiellement — IC a lire comme indicatif, verdict sur ordres
de grandeur et coherence val/test.

Lecture seule. AUCUNE modification du reward, du fee gate, ni du code de
production.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_mtf_confluence import load_aligned, OUT_DIR  # noqa: E402
from diag_mtf_confluence_v2 import (  # noqa: E402
    cross_up_events, select_nonoverlapping, N_BOOT, BLOCK)
from diag_mfe_excursion import (  # noqa: E402
    first_passage_fixe, sortie_invalidation, get_masks,
    TP_FIXE, SL_FIXE, K_SL_B2, K_SL_B3, STRIDE, H_MAX)
from _asset_guard import get_launcher_assets  # noqa: E402

# ------------------------------------------------------------------ parametres
ASSETS = list(get_launcher_assets())
SPLITS = ["train", "val", "test"]

# Grille de couts round-trip (fraction). 0.005 = actuel, 0.001 = borne de
# la condition de deblocage ("frais <= 0,10 % A/R").
COST_GRID = [0.005, 0.004, 0.003, 0.002, 0.0015, 0.001, 0.0005, 0.0]
COST_UNLOCK = 0.001          # 0,10 % RT — borne haute de la condition
H_B1 = 288                   # timeout B1 (24h), identique a EXP2

RNG = np.random.default_rng(20260910)


# ------------------------------------------------------------------ bootstrap
def boot_means(pnl, n_boot=N_BOOT, block=BLOCK):
    """Moyennes bootstrap en blocs sur le PnL BRUT (une seule fois)."""
    n = len(pnl)
    if n < 5:
        return None
    n_blocks = int(np.ceil(n / block))
    starts = RNG.integers(0, max(1, n - block + 1), size=(n_boot, n_blocks))
    arange = np.arange(block)[None, :]
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = (starts[b][:, None] + arange).ravel()[:n]
        idx = np.clip(idx, 0, n - 1)
        means[b] = pnl[idx].mean()
    return means


def sweep_costs(pnl_brut, means_brut, label):
    """Traduit les moyennes bootstrap brutes en EV nette a chaque cout."""
    n = len(pnl_brut)
    out = {"label": label, "n": int(n),
           "ev_brut": float(pnl_brut.mean()),
           "break_even_cost_rt": float(pnl_brut.mean()),  # EV=0 ici
           "wr_brut": float((pnl_brut > 0).mean()),
           "levels": {}}
    if means_brut is None:
        out["skipped"] = "n<5"
        return out
    for c in COST_GRID:
        m_net = means_brut - c
        lo, hi = np.quantile(m_net, [0.025, 0.975])
        p_two = 2 * min((m_net <= 0).mean(), (m_net >= 0).mean())
        out["levels"][f"{c:.4%}"] = {
            "ev_net": float(pnl_brut.mean() - c),
            "ci95": [float(lo), float(hi)],
            "p_boot": float(min(1.0, p_two))}
    return out


# ------------------------------------------------------------------ simulation
def simulate_rules(df):
    """Simule B1/B2/B3 a cout nul sur les entrees MTF_ALIGNE (protocole EXP2).

    Retourne {asset-independent dict regle -> (pnl_brut, ht, reason)}."""
    n = len(df)
    if n < H_MAX + 50:
        return None, f"rows={n} < H_MAX+50"
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    regime4h = df["ema_100_ratio_4h"].to_numpy(np.float64)
    atr = df["atr_pct"].to_numpy(np.float64)

    events = cross_up_events(df)
    entries = select_nonoverlapping(events, n, STRIDE)
    entries = entries[entries + H_MAX < n]
    aligne, _ = get_masks(df)
    ent = entries[aligne[entries]]
    if len(ent) < 30:
        return None, f"n_aligne={len(ent)} < 30"

    rules = {}
    pnl, ht, r = first_passage_fixe(high, low, close, ent, H_B1,
                                    SL_FIXE, TP_FIXE)
    rules["B1_tp_fixe_1.2_sl_0.6_H288"] = (pnl, ht, r)
    for k_sl, name in ((K_SL_B2, "B2_sl_0.5atr_invalidation"),
                       (K_SL_B3, "B3_sl_3atr_invalidation")):
        pnl, ht, r = sortie_invalidation(
            high, low, close, regime4h, ent, H_MAX, atr[ent], k_sl)
        rules[name] = (pnl, ht, r)
    return (rules, ent), None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[params] cost_grid={['%.4f' % c for c in COST_GRID]}  "
          f"cost_unlock={COST_UNLOCK:.4f}  n_boot={N_BOOT}  block={BLOCK}  "
          f"stride={STRIDE}  H_max={H_MAX}", flush=True)

    results = []
    for asset in ASSETS:
        for split in SPLITS:
            df = load_aligned(asset, split)
            if df is None:
                continue
            df = df.dropna(subset=["ema_100_ratio_4h", "ema_50_ratio_1h",
                                   "macdh_12_26_9", "atr_pct"])
            print(f"[run] {asset}/{split}: {len(df)} barres 5m", flush=True)
            sim, err = simulate_rules(df)
            if sim is None:
                results.append({"asset": asset, "split": split,
                                "skipped": err})
                print(f"  [skip] {err}", flush=True)
                continue
            rules, ent = sim
            cell = {"asset": asset, "split": split,
                    "rows_5m": int(len(df)), "n_entries": int(len(ent)),
                    "rules": {}}
            for name, (pnl, ht, reason) in rules.items():
                means = boot_means(pnl)
                s = sweep_costs(pnl, means, name)
                s["holding_med"] = float(np.median(ht))
                s["exit_reasons"] = {
                    r: float((reason == r).mean())
                    for r in np.unique(reason)}
                cell["rules"][name] = s
                be = s.get("break_even_cost_rt", float("nan"))
                ev_unlock = s.get("levels", {}).get(
                    f"{COST_UNLOCK:.4%}", {}).get("ev_net", float("nan"))
                print(f"  {name:<32} n={s['n']:>5}  "
                      f"EVbrut={s.get('ev_brut', float('nan')):+.4%}  "
                      f"BE_cost={be:+.4%}  "
                      f"EV@0.10%RT={ev_unlock:+.4%}", flush=True)
            results.append(cell)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = {"generated_utc": stamp,
           "params": {"cost_grid_rt": COST_GRID, "cost_unlock_rt": COST_UNLOCK,
                      "n_boot": N_BOOT, "block": BLOCK, "stride": STRIDE,
                      "H_max": H_MAX, "H_B1": H_B1,
                      "tp_fixe": TP_FIXE, "sl_fixe": SL_FIXE,
                      "k_sl_b2": K_SL_B2, "k_sl_b3": K_SL_B3,
                      "entries": "cross_up MACD 5m + MTF_ALIGNE, stride 40"},
           "results": results}
    path = OUT_DIR / f"fee_sensitivity_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nRapport JSON : {path}")

    # ------------------------------------------------- verdict pre-enregistre
    print("\n===== VERDICT — condition de deblocage =====")
    print(f"(regle EV>0 ET IC95 bas > 0 a cout RT <= {COST_UNLOCK:.2%}, "
          "sur val ET test)\n")
    verdicts = {}
    for name in ("B1_tp_fixe_1.2_sl_0.6_H288", "B2_sl_0.5atr_invalidation",
                 "B3_sl_3atr_invalidation"):
        verdicts[name] = {}
        for asset in ASSETS:
            row = {}
            for split in SPLITS:
                cell = next((r for r in results
                             if r.get("asset") == asset
                             and r.get("split") == split), None)
                rule = (cell or {}).get("rules", {}).get(name)
                if rule is None or "levels" not in rule:
                    row[split] = None
                    continue
                lvl = rule["levels"].get(f"{COST_UNLOCK:.4%}")
                row[split] = {
                    "ev": lvl["ev_net"], "lo": lvl["ci95"][0],
                    "p": lvl["p_boot"],
                    "pass": bool(lvl["ev_net"] > 0 and lvl["ci95"][0] > 0)}
            verdicts[name][asset] = row
            fmt = lambda s: ("n/a" if row.get(s) is None else
                             f"EV={row[s]['ev']:+.4%} lo={row[s]['lo']:+.4%} "
                             f"p={row[s]['p']:.3f} "
                             f"{'PASS' if row[s]['pass'] else 'FAIL'}")
            print(f"  {name:<32} {asset:<18} "
                  f"val[{fmt('val')}]  test[{fmt('test')}]")

    any_unlock = any(
        cell.get("val") and cell.get("test")
        and cell["val"]["pass"] and cell["test"]["pass"]
        for rule in verdicts.values() for cell in rule.values())
    print("\n==> " + ("DEBLOQUAGE ECONOMIQUE : au moins une regle passe "
                       "val ET test a 0,10 % RT."
                       if any_unlock else
                       "NO-GO MAINTENU : aucune regle ne passe val ET test "
                       "a 0,10 % RT. Voir break-even par cellule dans le JSON."))


if __name__ == "__main__":
    main()
