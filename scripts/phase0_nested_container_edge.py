#!/usr/bin/env python3
"""
phase0_nested_container_edge.py — ADAN-System-One · PHASE 0
=============================================================

Protocole de validation empirique IMMÉDIAT (workflow §V) :
prouver l'existence de l'edge des « contenants imbriqués » AVANT toute
machinerie PyTorch. Si ce filtre structurel simple ne sépare pas, tout le
socle physique du workflow tombe.

Hypothèse testée (SHORT — balayage d'extrême en fin de cycle 1h)
------------------------------------------------------------------
À la clôture d'une bougie 5m à l'indice t :

1. La 5m clôture en phase k ∈ {11, 12} de son contenant 1h
   (minute 50 ou 55 — fin de formation de la bougie 1h).
2. SWEEP : high[t] dépasse le plus haut des 10 barres 5m précédentes.
3. RÉINTÉGRATION : close[t] revient sous ce plus haut précédent.
4. MÈCHE DE REJET : mèche supérieure ≥ 40 % du range de la bougie.

Simulation (sans aucune donnée future) :
   - Entrée SHORT à l'open de la 5m suivante (t+1).
   - SL = high[t] × 1.0005  (au-dessus de la mèche + 0.05 %)
   - TP = entry − 2.5 × (SL − entry)   (expansion 2.5R vers le bas du contenant)
   - Frais : 0.40 % aller-retour, déduits SYSTÉMATIQUEMENT.
   - Sortie au plus tard après 288 barres (24 h) → time-stop au close.
   - Dans une barre touchant SL et TP simultanément : pire cas (SL) — biais
     conservateur assumé.

Splits temporels (aucune fuite) :
   train  < 2022-01-01  |  validation 2022→2023  |  test ≥ 2024-01-01

Verdict : EV nette par trade, win rate, profit factor, séparation vs baseline
(short naïf à la même phase sans condition de sweep).
"""

import sys
import numpy as np
import pandas as pd

PARQUET = "data/processed/BTCUSDT_binance/BTCUSDT_5m_featured.parquet"
FEES_RT = 0.0040          # 0.40 % aller-retour
SL_BUFFER = 0.0005        # +0.05 % au-dessus de la mèche
TP_R = 2.5                # expansion visée : 2.5 × risque
WICK_MIN = 0.40           # mèche de rejet ≥ 40 % du range
LOOKBACK_SWEEP = 10       # nouveau plus haut vs 10 barres
MAX_HOLD = 288            # time-stop : 24 h (288 × 5m)
PHASES = {11, 12}         # fin de contenant 1h (minutes 50 et 55)


def run(df, use_sweep_filter):
    """Exécute le backtest. use_sweep_filter=False → baseline (même phase,
    sans conditions sweep/réintégration/mèche)."""
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    idx = df.index
    n = len(df)

    # Phase k dans le contenant 1h : minute//5 + 1  (1..12)
    phase = idx.minute.to_numpy() // 5 + 1

    # Plus haut des 10 barres précédentes (rolling décalé d'une barre)
    prev_high = (
        pd.Series(h).rolling(LOOKBACK_SWEEP).max().shift(1).to_numpy()
    )

    trades = []  # (timestamp_signal, r_net, issue)
    i = LOOKBACK_SWEEP
    while i < n - MAX_HOLD - 1:
        if phase[i] in PHASES:
            signal = True
            if use_sweep_filter:
                rng = h[i] - l[i]
                wick_up = h[i] - max(o[i], c[i])
                signal = (
                    np.isfinite(prev_high[i])
                    and rng > 0
                    and h[i] > prev_high[i]          # sweep du plus haut
                    and c[i] < prev_high[i]          # réintégration du range
                    and wick_up >= WICK_MIN * rng    # mèche de rejet ≥ 40 %
                )
            if signal:
                entry = o[i + 1]
                sl = h[i] * (1.0 + SL_BUFFER)
                risk = sl - entry
                if risk > 0:
                    tp = entry - TP_R * risk
                    r_net, bars, issue = -1.0, MAX_HOLD, "time-stop"
                    closed = False
                    for j in range(i + 1, i + 1 + MAX_HOLD):
                        hit_sl = h[j] >= sl
                        hit_tp = l[j] <= tp
                        if hit_sl and hit_tp:      # ambiguïté → pire cas
                            r_net, bars, issue = -1.0, j - i, "SL(ambigu)"
                            closed = True
                            break
                        if hit_sl:
                            r_net, bars, issue = -1.0, j - i, "SL"
                            closed = True
                            break
                        if hit_tp:
                            r_net, bars, issue = TP_R, j - i, "TP"
                            closed = True
                            break
                    if not closed:
                        exit_price = c[i + MAX_HOLD]
                        r_net = (entry - exit_price) / risk
                        bars, issue = MAX_HOLD, "time-stop"
                    # Frais : 0.40 % du nominal, ramenés en R
                    fees_r = (FEES_RT * entry) / risk
                    trades.append((idx[i], r_net - fees_r, issue))
                    i += bars  # pas de chevauchement : prochain signal après sortie
                    continue
        i += 1

    tr = pd.DataFrame(trades, columns=["ts", "r_net", "issue"])
    return tr


def stats(tr, name):
    if tr.empty:
        return {"split": name, "trades": 0}
    wins = tr[tr.r_net > 0]
    losses = tr[tr.r_net <= 0]
    pf = wins.r_net.sum() / abs(losses.r_net.sum()) if len(losses) and losses.r_net.sum() != 0 else np.nan
    return {
        "split": name,
        "trades": len(tr),
        "win_rate_%": round(100 * len(wins) / len(tr), 1),
        "EV_nette_R": round(tr.r_net.mean(), 4),
        "médiane_R": round(tr.r_net.median(), 3),
        "total_R": round(tr.r_net.sum(), 1),
        "profit_factor": round(pf, 2) if np.isfinite(pf) else None,
        "TP_%": round(100 * (tr.issue == "TP").mean(), 1),
        "SL_%": round(100 * tr.issue.str.startswith("SL").mean(), 1),
        "timeStop_%": round(100 * (tr.issue == "time-stop").mean(), 1),
    }


def main():
    print("=" * 72)
    print("ADAN-SYSTEM-ONE · PHASE 0 — Edge des contenants imbriqués (BTCUSDT 5m)")
    print("=" * 72)
    df = pd.read_parquet(PARQUET, columns=["open", "high", "low", "close"])
    df = df[df["high"] >= df[["open", "close"]].max(axis=1)]  # garde OHLC saines
    print(f"Données : {len(df):,} bougies 5m  ({df.index[0]} → {df.index[-1]})")

    SPLITS = {
        "train      (<2022)": (None, "2022-01-01"),
        "validation (22-23)": ("2022-01-01", "2024-01-01"),
        "test       (≥2024)": ("2024-01-01", None),
    }

    for label, use_filter in [("FILTRE SWEEP (thèse)", True),
                              ("BASELINE naïve (même phase, sans sweep)", False)]:
        print(f"\n── {label} " + "─" * (46 - len(label)))
        tr = run(df, use_filter)
        rows = []
        for sname, (d0, d1) in SPLITS.items():
            sub = tr
            if d0: sub = sub[sub.ts >= d0]
            if d1: sub = sub[sub.ts < d1]
            rows.append(stats(sub, sname))
        res = pd.DataFrame(rows).set_index("split")
        print(res.to_string())
        if use_filter:
            test = rows[2]
            print("\n── VERDICT (split test, jamais vu) " + "─" * 36)
            ev = test.get("EV_nette_R", 0) or 0
            if test.get("trades", 0) >= 30 and ev > 0:
                print(f"✅ GO : EV nette = +{ev}R/trade sur {test['trades']} trades "
                      f"(PF={test['profit_factor']}) — edge des contenants CONFIRMÉ.")
            else:
                print(f"❌ NO-GO : EV nette = {ev}R/trade ({test.get('trades', 0)} trades) "
                      "— le socle structurel ne sépare pas assez en l'état.")


if __name__ == "__main__":
    sys.exit(main())
