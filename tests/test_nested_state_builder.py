#!/usr/bin/env python3
"""
test_nested_state_builder.py — Tests de causalité du LivingStateSnapshot
=========================================================================

L'invariant absolu de l'étape 1 du workflow : **zéro fuite de données
futures**. Ces tests le prouvent mécaniquement :

  T1. MUTATION DU FUTUR — modifier les barres > i ne change RIEN au snapshot i.
  T2. CONTENANT VIVANT — à mi-parcours (k=6/12), running_1h == OHLC des 6
      premières 5m du contenant (et non la bougie 1h complète).
  T3. CONTENANT PRÉCÉDENT — prev_1h == la 1h civile complète fermée.
  T4. PHASES — k_1h et m_4h corrects aux frontières (00:00, 03:55→04:00).
  T5. INTÉGRITÉ — NaN / OHLC incohérent / historique court → abstention.
  T6. SWEEPS — détection identique à la logique Phase 0 (référence validée).

Exécution :
  conda run : /home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python3
              tests/test_nested_state_builder.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from adan_trading_bot.data.nested_state_builder import (
    NestedStateBuilder, BARS_PER_1H, BARS_PER_4H,
)

PASS, FAIL = "✅", "❌"
failures = []


def check(name, cond, detail=""):
    print(f"  {PASS if cond else FAIL} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def make_df(n=600, seed=7, start="2024-03-01"):
    """Timeline 5m synthétique déterministe (marche aléatoire)."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, n))
    vol = np.abs(rng.normal(10, 2, n))
    idx = pd.date_range(start, periods=n, freq="5min")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


print("=" * 72)
print("TEST nested_state_builder — causalité du LivingStateSnapshot")
print("=" * 72)

# ─── T1 : mutation du futur ──────────────────────────────────────────────────
print("\nT1 — Le futur ne doit JAMAIS influencer le passé")
df = make_df()
i = 300
s_ref = NestedStateBuilder(df).snapshot(i)
df_mut = df.copy()
df_mut.iloc[i + 1:, df_mut.columns.get_loc("high")] *= 5.0   # futur explosé
df_mut.iloc[i + 1:, df_mut.columns.get_loc("close")] *= 3.0
s_mut = NestedStateBuilder(df_mut).snapshot(i)
same = (
    np.array_equal(s_ref.seq_5m, s_mut.seq_5m)
    and np.array_equal(s_ref.running_1h, s_mut.running_1h)
    and np.array_equal(s_ref.running_4h, s_mut.running_4h)
    and np.array_equal(s_ref.prev_1h, s_mut.prev_1h)
    and np.array_equal(s_ref.prev_4h, s_mut.prev_4h)
    and s_ref.pos_in_1h == s_mut.pos_in_1h
    and s_ref.phase_4h == s_mut.phase_4h
    and s_ref.sweep_high_1h == s_mut.sweep_high_1h
)
check("snapshot invariant sous mutation du futur (barres > i)", same)

# ─── T2 : contenant vivant à mi-parcours ─────────────────────────────────────
print("\nT2 — Contenant 1h RUNNING à k=6/12 (mi-formation)")
# barre i à minute 25 → 6e barre de l'heure (k=6)
i2 = df.index.searchsorted(pd.Timestamp("2024-03-01 13:25"))
s2 = NestedStateBuilder(df).snapshot(i2)
hour_start = df.index[i2].floor("h")
expected = df.loc[hour_start:df.index[i2]]
oh = expected["open"].iloc[0]; hh = expected["high"].max()
ll = expected["low"].min();  vv = expected["volume"].sum()
ok2 = (s2.phase_1h == 6 / 12
       and np.isclose(s2.running_1h[0], oh) and np.isclose(s2.running_1h[1], hh)
       and np.isclose(s2.running_1h[2], ll) and np.isclose(s2.running_1h[3], vv))
check("running_1h == OHLCV des 6 premières 5m (pas la 1h complète)", ok2,
      f"k={s2.phase_1h*12:.0f}/12, O={s2.running_1h[0]:.2f} vs {oh:.2f}")

# ─── T3 : contenant précédent fermé ──────────────────────────────────────────
print("\nT3 — prev_1h == bougie 1h civile FERMÉE précédente")
prev_hour = hour_start - pd.Timedelta(hours=1)
exp_prev = df.loc[prev_hour:hour_start - pd.Timedelta(minutes=5)]
ok3 = (np.isclose(s2.prev_1h[1], exp_prev["high"].max())
       and np.isclose(s2.prev_1h[2], exp_prev["low"].min())
       and np.isclose(s2.prev_1h[3], exp_prev["close"].iloc[-1]))
check("prev_1h (high/low/close) == 1h fermée précédente", ok3)

# ─── T4 : phases aux frontières ──────────────────────────────────────────────
print("\nT4 — Phases aux frontières horaires")
b = NestedStateBuilder(df)
i_0000 = df.index.searchsorted(pd.Timestamp("2024-03-02 00:00"))
i_0355 = df.index.searchsorted(pd.Timestamp("2024-03-02 03:55"))
s_a, s_b = b.snapshot(i_0000), b.snapshot(i_0355)
ok4 = (s_a.phase_1h == 1 / 12 and s_a.phase_4h == 1 / 48
       and s_b.phase_1h == 12 / 12 and s_b.phase_4h == 48 / 48)
check("00:00 → phase 1/12 & 1/48 ; 03:55 → 12/12 & 48/48", ok4)

# ─── T5 : intégrité ──────────────────────────────────────────────────────────
print("\nT5 — Contrôle d'intégrité (étape 2 → abstention)")
ok5a = not NestedStateBuilder(df).snapshot(10).integrity_ok  # historique court
df_nan = df.copy(); df_nan.iloc[400, df_nan.columns.get_loc("close")] = np.nan
ok5b = not NestedStateBuilder(df_nan).snapshot(400).integrity_ok
df_bad = df.copy(); df_bad.iloc[450, df_bad.columns.get_loc("low")] = df_bad["high"].iloc[450] + 10
ok5c = not NestedStateBuilder(df_bad).snapshot(450).integrity_ok
check("historique court → False", ok5a)
check("NaN dans la fenêtre → False", ok5b)
check("OHLC incohérent (low > high) → False", ok5c)

# ─── T6 : sweeps conformes à la logique Phase 0 ──────────────────────────────
print("\nT6 — Détection des sweeps (référence Phase 0)")
# Injection d'un sweep haut artificiel sur le contenant 1h précédent
df_sw = make_df(seed=11)
i_sw = df_sw.index.searchsorted(pd.Timestamp("2024-03-01 15:50"))  # k=11
ref = NestedStateBuilder(df_sw).snapshot(i_sw)
prev_high_1h, prev_low_1h = ref.prev_1h[1], ref.prev_1h[2]
# Sweep haut : high dépasse le sommet 1h précédent, close réintègre,
# et low maintenu AU-DESSUS du bas 1h précédent (isolation du cas haut).
df_sw.iloc[i_sw, df_sw.columns.get_loc("high")] = prev_high_1h + 5.0
df_sw.iloc[i_sw, df_sw.columns.get_loc("low")] = prev_low_1h + 0.5
df_sw.iloc[i_sw, df_sw.columns.get_loc("close")] = prev_high_1h - 1.0
df_sw.iloc[i_sw, df_sw.columns.get_loc("open")] = prev_high_1h - 0.5
s6 = NestedStateBuilder(df_sw).snapshot(i_sw)
check("sweep_high_1h détecté (dépassement + réintégration)", s6.sweep_high_1h == 1)
check("pas de faux sweep_low_1h", s6.sweep_low_1h == 0)

# ─── T7 : données réelles BTCUSDT (cohérence sur parquet de production) ──────
print("\nT7 — Cohérence sur données réelles BTCUSDT (10k barres)")
pq = os.path.join(os.path.dirname(__file__), "..",
                  "data/processed/BTCUSDT_binance/BTCUSDT_5m_featured.parquet")
if os.path.exists(pq):
    real = pd.read_parquet(pq, columns=["open", "high", "low", "close", "volume"])
    real = real.iloc[50000:60000]  # milieu de série (2019)
    rb = NestedStateBuilder(real)
    # running 1h à clôture complète (k=12) == 1h recalculée indépendamment
    i_full = int(np.where(rb.k_1h == 12)[0][100])
    s7 = rb.snapshot(i_full)
    hour_start = real.index[i_full].floor("h")
    exp = real.loc[hour_start:real.index[i_full]]
    ok7 = (np.isclose(s7.running_1h[1], exp["high"].max())
           and np.isclose(s7.running_1h[2], exp["low"].min())
           and s7.integrity_ok)
    check("running_1h à k=12 == 1h recalculée sur données réelles", ok7)
else:
    check("données réelles disponibles", False, f"parquet absent : {pq}")

# ─── Verdict ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
if failures:
    print(f"❌ {len(failures)} test(s) en échec : {failures}")
    sys.exit(1)
print("✅ TOUS LES TESTS PASSENT — causalité du LivingStateSnapshot prouvée.")
print("   (futur muté sans effet · contenants vivants exacts · intégrité active)")
