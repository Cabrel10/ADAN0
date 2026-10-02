#!/usr/bin/env python3
"""
Moniteur d'APPRENTISSAGE ADAN0 (post-forensic, focus qualite pas vitesse).
Lit le log d'entrainement et agrege:
  - FPS (table SB3) + progression steps
  - metriques PPO: ep_rew_mean, entropy_loss, explained_variance, approx_kl,
    clip_fraction, value_loss, std
  - activite trading: OPEN / CLOSE, WR (win rate), profit factor (si PnL dispo),
    STERILE_SELL
Ecrit un rapport lisible. Lecture seule.

Usage: python monitor_learning.py <LOGFILE>
"""
import os
import re
import sys


def num(s):
    try:
        return float(s)
    except Exception:
        return None


def main():
    log = sys.argv[1]
    if not os.path.exists(log):
        print(f"log absent: {log}")
        sys.exit(1)
    txt = open(log, errors="ignore").read()

    # FPS / steps (tables SB3)
    fps = re.findall(r"fps\s*\|\s*([\d.]+)", txt)
    tot = re.findall(r"total_timesteps\s*\|\s*(\d+)", txt)
    iters = re.findall(r"iterations\s*\|\s*(\d+)", txt)
    elapsed = re.findall(r"time_elapsed\s*\|\s*(\d+)", txt)

    rew = re.findall(r"ep_rew_mean\s*\|\s*([-\d.e+]+)", txt)
    ent = re.findall(r"entropy_loss\s*\|\s*([-\d.e+]+)", txt)
    ev = re.findall(r"explained_variance\s*\|\s*([-\d.e+]+)", txt)
    kl = re.findall(r"approx_kl\s*\|\s*([-\d.e+]+)", txt)
    clipf = re.findall(r"clip_fraction\s*\|\s*([-\d.e+]+)", txt)
    vloss = re.findall(r"value_loss\s*\|\s*([-\d.e+]+)", txt)
    std = re.findall(r"\bstd\s*\|\s*([-\d.e+]+)", txt)

    # steps env
    steps = re.findall(r"\[STEP (\d+)\]", txt)
    max_step = max((int(s) for s in steps), default=0)

    # trades
    n_open = txt.count("TRADE_AUDIT_OPEN")
    n_close = txt.count("TRADE_AUDIT_CLOSE")
    n_sterile = txt.count("STERILE_SELL")

    # PnL des CLOSE si present (cherche 'pnl=' ou 'realized')
    close_pnls = re.findall(r"TRADE_AUDIT_CLOSE[^\n]*?pnl[=:\s]+([-\d.]+)", txt,
                            re.IGNORECASE)
    wins = losses = 0
    gross_win = gross_loss = 0.0
    for p in close_pnls:
        v = num(p)
        if v is None:
            continue
        if v > 0:
            wins += 1
            gross_win += v
        elif v < 0:
            losses += 1
            gross_loss += abs(v)

    print("=" * 60)
    print(f"MONITEUR APPRENTISSAGE — {os.path.basename(log)}")
    print("=" * 60)
    print(f"max_step env       : {max_step}")
    if tot:
        print(f"total_timesteps    : {tot[-1]}  (iterations={iters[-1] if iters else '?'})")
    if fps and elapsed:
        print(f"fps (derniere table): {fps[-1]}   time_elapsed={elapsed[-1]}s")
        # fps instantane entre 2 dernieres tables
        if len(tot) >= 2 and len(elapsed) >= 2:
            dts = int(tot[-1]) - int(tot[-2])
            dte = int(elapsed[-1]) - int(elapsed[-2])
            if dte > 0:
                print(f"fps instantane     : {dts/dte:.2f}  (Δsteps={dts}/Δt={dte}s)")
    print("-" * 60)
    print("METRIQUES PPO (derniere -> avant-derniere -> ...):")
    def show(name, arr, n=4):
        if arr:
            vals = arr[-n:][::-1]
            print(f"  {name:20s}: {' <- '.join(vals)}")
    show("ep_rew_mean", rew)
    show("entropy_loss", ent)
    show("explained_variance", ev)
    show("approx_kl", kl)
    show("clip_fraction", clipf)
    show("value_loss", vloss)
    show("std (policy)", std)
    print("-" * 60)
    print("ACTIVITE TRADING:")
    print(f"  OPEN={n_open}  CLOSE={n_close}  STERILE_SELL={n_sterile}")
    if close_pnls:
        tot_t = wins + losses
        wr = 100 * wins / tot_t if tot_t else 0
        pf = (gross_win / gross_loss) if gross_loss > 0 else float('inf')
        print(f"  trades fermes avec PnL={len(close_pnls)}  WR={wr:.1f}%  "
              f"profit_factor={pf:.2f}  (wins={wins}/losses={losses})")
    else:
        print("  (PnL par trade non logge dans TRADE_AUDIT_CLOSE — WR/PF indispo ici)")
    print("=" * 60)

    # Diagnostic sante apprentissage
    print("DIAGNOSTIC:")
    if ev:
        evv = num(ev[-1])
        if evv is not None:
            if evv > 0.1:
                print(f"  [OK] explained_variance={evv:.3f}>0.1 — la value function apprend.")
            elif evv > -0.1:
                print(f"  [~]  explained_variance={evv:.3f} ~0 — value function neutre/debut.")
            else:
                print(f"  [!]  explained_variance={evv:.3f}<0 — value function n'aide pas (a surveiller).")
    if ent and len(ent) >= 2:
        e0, e1 = num(ent[0]), num(ent[-1])
        if e0 is not None and e1 is not None:
            trend = "monte (exploration↑)" if e1 > e0 else "baisse (exploitation↑)"
            print(f"  [i]  entropy_loss {e0:.2f} -> {e1:.2f} ({trend}).")
    if n_close == 0 and max_step > 1000:
        print("  [!]  Aucun trade ferme — risque de politique inerte.")
    elif n_close > 0:
        print(f"  [OK] Le modele ouvre ET ferme des positions ({n_close} fermetures).")


if __name__ == "__main__":
    main()
