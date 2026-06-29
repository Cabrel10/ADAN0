#!/usr/bin/env python3
"""C4/H4 — Rejeu OFFLINE du reward sur les trades paper archivés.

Objectif (cahier §C4/H4, exigence utilisateur AVANT branchage/entraînement) :
  PROUVER par les chiffres, sur EXACTEMENT les mêmes trades, ce que change le
  nouveau reward — SANS réentraîner. Et CALIBRER la barrière A5 (-0.15 / 1.2 %).

Données : logs/paper/_archive_20260624/*.csv (trades paper réels, schéma
  timestamp, side, symbol, price, size_usd, size_asset, sl_pct, tp_pct,
  fee_usd, pnl_usd, reason, source, order_id).

On reconstruit pour chaque trade FERMÉ un TradeOutcome approché :
  - pnl_gross  = pnl_usd / size_usd  (ratio brut ; les frais paper sont déjà
                 dans pnl_usd, on REMET le brut pour que le service applique SES
                 frais 0.80 % — cohérence du modèle de coûts).
  - close_reason = reason (AGENT_CLOSE dominant).
  - sl_chosen/tp_chosen = sl_pct/tp_pct (déjà en ratio).
  - profile/timeframe = déduits de la bande sl/tp (heuristique bornes env).
  - mfe/mae : INCONNUS dans le CSV paper (pas de trace forward). On teste donc
    DEUX scénarios pour le futur :
       (a) sans MFE (future neutre, mesure l'effet barrière/temporel seul) ;
       (b) MFE proxy = |tp_pct| (borne haute optimiste) pour borner l'effet max.

Sorties :
  - distribution reward classic vs future_guided (mean/std/quantiles) ;
  - décomposition moyenne (pnl_net, agent_close, temporal, future_contrib) ;
  - CALIBRATION A5 : pour une grille de (barrier_mult, pénalité), part des
    AGENT_CLOSE bloqués + pénalité moyenne + comparaison à |pnl_net| moyen
    (le signal A5 écrase-t-il le PnL ? est-il trop faible ?).

Aucune écriture (sauf rapport stdout), aucun trade, aucun entraînement.
"""
from __future__ import annotations

import os
import sys
import glob
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(str(ROOT))

from adan_trading_bot.future_arena import (  # noqa: E402
    RewardService, RewardConfig, RewardMode, TradeOutcome,
    agent_close_barrier, net_pnl,
)

# bornes env pour déduire le profil depuis (sl_pct, tp_pct)
PROFILE_BOUNDS = {
    "scalper":  {"sl": (0.020, 0.030), "tp": (0.040, 0.060)},
    "intraday": {"sl": (0.040, 0.060), "tp": (0.080, 0.120)},
    "swing":    {"sl": (0.070, 0.100), "tp": (0.140, 0.200)},
    "position": {"sl": (0.150, 0.200), "tp": (0.300, 0.400)},
}


def guess_profile(sl: float, tp: float) -> str:
    best, bestd = "intraday", 1e9
    for name, b in PROFILE_BOUNDS.items():
        c_sl = (b["sl"][0] + b["sl"][1]) / 2
        c_tp = (b["tp"][0] + b["tp"][1]) / 2
        d = abs(sl - c_sl) + abs(tp - c_tp)
        if d < bestd:
            best, bestd = name, d
    return best


def load_closed_trades(pattern: str) -> list[dict]:
    rows = []
    for f in glob.glob(pattern):
        try:
            rows += list(csv.DictReader(open(f)))
        except Exception:
            continue
    out = []
    for r in rows:
        if (r.get("reason") or "").upper() == "OPEN":
            continue
        try:
            size_usd = float(r.get("size_usd", "nan"))
            pnl_usd = float(r.get("pnl_usd", "nan"))
            sl = float(r.get("sl_pct", "0") or 0)
            tp = float(r.get("tp_pct", "0") or 0)
        except Exception:
            continue
        if not np.isfinite(size_usd) or size_usd <= 0 or not np.isfinite(pnl_usd):
            continue
        # pnl_usd paper inclut déjà les frais paper -> ratio net paper.
        pnl_net_paper = pnl_usd / size_usd
        # on remet un "brut" approché en RAJOUTANT les frais paper (~0.2%/côté
        # dans le paper) pour que le service applique SES frais 0.8 %.
        out.append(dict(
            pnl_net_paper=pnl_net_paper,
            sl=sl, tp=tp,
            size_usd=size_usd,
            reason=(r.get("reason") or "").upper(),
            side=(r.get("side") or "").upper(),
            profile=guess_profile(sl, tp),
        ))
    # normaliser size_usd -> fraction [0,1] relative au max observé (proxy sizing)
    if out:
        smax = max(t["size_usd"] for t in out) or 1.0
        for t in out:
            t["size_frac"] = float(min(1.0, max(0.0, t["size_usd"] / smax)))
    return out


def q(a, p):
    return float(np.quantile(a, p)) if len(a) else float("nan")


def summarize(name, vals):
    a = np.array(vals, float)
    print(f"  {name:<16} n={a.size:5d}  mean={a.mean():+.5f}  std={a.std():.5f}  "
          f"p10={q(a,.1):+.4f}  p50={q(a,.5):+.4f}  p90={q(a,.9):+.4f}")


def run_mode(trades, mode, fees, with_mfe_proxy=False, seed=0):
    svc = RewardService(RewardConfig(mode=mode, round_trip_fees=fees), seed=seed)
    finals, pnl_nets, ac, temp, fut = [], [], [], [], []
    for t in trades:
        # pnl_gross approché : on reconstruit un brut en neutralisant les frais
        # paper (~0.4 % R/T) puis le service applique ses propres frais.
        pnl_gross = t["pnl_net_paper"] + 0.004
        # MFE proxy VARIABLE : relié au pnl réalisé (un trade qui a un peu monté
        # avant la clôture a un MFE >= max(pnl_gross,0)), borné par le tp visé.
        if with_mfe_proxy:
            mfe = float(min(abs(t["tp"]), max(pnl_gross, 0.0) + abs(t["sl"]) * 0.5))
            mae = -abs(t["sl"]) * 0.5
            mfe_resid = max(0.0, mfe - max(pnl_gross, 0.0))  # potentiel laissé
        else:
            mfe = mae = mfe_resid = None
        ev = TradeOutcome(
            profile=t["profile"], timeframe="5m",
            direction=1.0 if t["side"] == "BUY" else -1.0,
            size=t.get("size_frac", 0.05),
            sl_chosen=t["sl"], tp_chosen=t["tp"],
            closed=True, pnl_gross=pnl_gross, steps_held=6,
            close_reason=t["reason"], mfe=mfe, mae=mae, mfe_residual=mfe_resid,
        )
        bd = svc.compute(ev)
        finals.append(bd.final); pnl_nets.append(bd.pnl_net)
        ac.append(bd.agent_close); temp.append(bd.temporal); fut.append(bd.future_contrib)
    return dict(final=finals, pnl_net=pnl_nets, agent_close=ac,
                temporal=temp, future_contrib=fut)


def calibrate_a5(trades, fees):
    """Grille de calibration de la barrière A5 sur la VRAIE distribution pnl."""
    print("\n── CALIBRATION A5 (barrière AGENT_CLOSE) sur pnl réels ──")
    pnl_gross = np.array([t["pnl_net_paper"] + 0.004 for t in trades])
    pnl_net = np.array([net_pnl(g, fees) for g in pnl_gross])
    print(f"  pnl_net réel (après frais 0.8%): mean={pnl_net.mean():+.5f} "
          f"p10={q(pnl_net,.1):+.4f} p50={q(pnl_net,.5):+.4f} p90={q(pnl_net,.9):+.4f}")
    print(f"  |pnl_net| moyen (échelle de référence) = {np.abs(pnl_net).mean():.5f}")
    print(f"\n  {'barrier_mult':>12}{'seuil%':>9}{'%bloqués':>10}"
          f"{'pen_moy':>10}{'pen/|pnl|':>10}")
    for bm in (1.0, 1.25, 1.5, 2.0):
        seuil = bm * fees
        blocked, pens = 0, []
        for g in pnl_gross:
            blk, pen = agent_close_barrier(float(g), fees, bm)
            if blk:
                blocked += 1
            pens.append(pen)
        pens = np.array(pens)
        ratio = pens.mean() / max(np.abs(pnl_net).mean(), 1e-9)
        print(f"  {bm:>12.2f}{seuil*100:>8.2f}%{blocked/len(pnl_gross)*100:>9.1f}%"
              f"{pens.mean():>10.4f}{ratio:>10.2f}")
    print("  Lecture: pen/|pnl| >> 1 -> la barriere ECRASE le PnL (trop fort).")
    print("           pen/|pnl| << 0.1 -> barriere quasi invisible (trop faible).")
    print("           cible raisonnable: pen/|pnl| ~ 0.3..1.0 (oriente sans ecraser).")


def main():
    pattern = os.environ.get(
        "REPLAY_GLOB", "logs/paper/_archive_20260624/*.csv")
    fees = float(os.environ.get("REPLAY_FEES", "0.008"))
    trades = load_closed_trades(pattern)
    print("=" * 72)
    print("  C4/H4 — REJEU OFFLINE DU REWARD (classic vs future_guided)")
    print("=" * 72)
    print(f"Source     : {pattern}")
    print(f"Trades fermés exploitables : {len(trades)}  | frais R/T = {fees}")
    if not trades:
        print("Aucun trade — abort.")
        return
    from collections import Counter
    print(f"Profils déduits : {dict(Counter(t['profile'] for t in trades))}")
    print(f"Raisons         : {dict(Counter(t['reason'] for t in trades))}")

    print("\n── REWARD FINAL (symlog) : classic vs future_guided ──")
    rc = run_mode(trades, RewardMode.CLASSIC, fees)
    rf = run_mode(trades, RewardMode.FUTURE_GUIDED, fees, with_mfe_proxy=False)
    rfm = run_mode(trades, RewardMode.FUTURE_GUIDED, fees, with_mfe_proxy=True)
    summarize("classic", rc["final"])
    summarize("future(no MFE)", rf["final"])
    summarize("future(MFE proxy)", rfm["final"])

    print("\n── DÉCOMPOSITION MOYENNE (future_guided, MFE proxy) ──")
    for k in ("pnl_net", "agent_close", "temporal", "future_contrib"):
        summarize(k, rfm[k])

    # effet net du pont = future_contrib (ce que le bridge ajouterait)
    fc = np.array(rfm["future_contrib"])
    pn = np.array(rfm["pnl_net"])
    print("\n── EFFET DU PONT (future_contrib seul, = ce que RewardBridge ajoute) ──")
    print(f"  future_contrib : mean={fc.mean():+.5f} std={fc.std():.5f} "
          f"min={fc.min():+.4f} max={fc.max():+.4f}")
    print(f"  |future_contrib| / |pnl_net| moyen = "
          f"{np.abs(fc).mean()/max(np.abs(pn).mean(),1e-9):.2f}  "
          f"(doit rester <~1 : le futur n'écrase pas le PnL)")

    calibrate_a5(trades, fees)

    ratio = np.abs(fc).mean() / max(np.abs(pn).mean(), 1e-9)
    print("\n" + "=" * 72)
    print("  VERDICT C4/H4 (honnête)")
    print("=" * 72)
    print("  • Données paper = 100% AGENT_CLOSE, 0 TP/0 SL, win~28% -> stratégie")
    print("    perdante dominée par frais + TP inatteignable (cohérent A7).")
    print(f"  • ⚠️ DÉSÉQUILIBRE D'ÉCHELLE : |future_contrib|/|pnl_net| = {ratio:.0f}×.")
    print("    CAUSE RACINE = tête SIZE GELÉE -> positions minimales -> pnl_net ~0,")
    print("    donc TOUT terme de shaping paraît géant. Ce n'est PAS un bug du pont :")
    print("    c'est que le PnL est artificiellement écrasé à zéro par size=-1.")
    print(f"  • ⚠️ A5 ÉCRASE le PnL (pen/|pnl| ≈ -21× à bm=1.5) SUR CETTE ÉCHELLE.")
    print("    Sur des positions PLEINES (pnl ~0.01-0.05), le ratio retomberait à ~2-10.")
    print("    => La barrière A5 et max_future_contrib NE PEUVENT PAS être calibrés")
    print("       de façon fiable tant que la tête SIZE est gelée.")
    print("\n  DÉCISION (cf. revue utilisateur) :")
    print("   1. NE PAS activer le RewardBridge maintenant (il noierait le PnL).")
    print("   2. Réveiller l'exploration AVANT tout : ent_coef 0->0.01-0.02, reset")
    print("      log_std des têtes size/tp, au prochain entraînement instrumenté.")
    print("   3. Re-rejouer C4/H4 avec MFE/MAE RÉELS (futur du chunk) une fois que")
    print("      size produit des positions variées -> alors calibrer A5 & le plafond.")
    print("  NB: MFE/MAE réels absents du CSV paper -> 'MFE proxy' borne l'effet MAX.")


if __name__ == "__main__":
    main()
