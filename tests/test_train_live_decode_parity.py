#!/usr/bin/env python3
"""test_train_live_decode_parity.py — Prouve que le décodage SL/TP est IDENTIQUE
entre l'env d'entraînement et execution_engine (live/paper).

On ne charge PAS l'env complet (trop lourd) : on réimplémente la formule de
référence de l'env (la source de vérité, _BOUNDS + (raw+1)/2 + clip + R/R + fee
gate + ATR scalper floor) et on la compare à ExecutionEngine.decode_action.

Si ce test passe, une action PPO donnée produit le MÊME ordre SL/TP en backtest
et en paper/live. C'est la garantie anti-divergence demandée par l'audit.
"""
import sys
import os
import importlib.util
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# ── Référence = table _BOUNDS de l'env (copiée ici comme oracle) ──
ENV_BOUNDS = {
    "scalper":  {"sl": (0.003, 0.012), "tp": (0.005, 0.020)},
    "intraday": {"sl": (0.005, 0.020), "tp": (0.008, 0.040)},
    "swing":    {"sl": (0.010, 0.035), "tp": (0.015, 0.070)},
    "position": {"sl": (0.020, 0.060), "tp": (0.030, 0.120)},
}


def env_reference_decode(profile, sl_raw, tp_raw, atr_pct=0.002):
    """Réplique EXACTE de multi_asset_chunked_env.py:7451-7540."""
    b = ENV_BOUNDS.get(profile, ENV_BOUNDS["intraday"])
    sl_lo, sl_hi = b["sl"]
    tp_lo, tp_hi = b["tp"]
    tp_lo = max(tp_lo, 0.006)
    norm_sl = (sl_raw + 1.0) / 2.0
    sl_pct = float(np.clip(sl_lo + norm_sl * (sl_hi - sl_lo), sl_lo, sl_hi))
    norm_tp = (tp_raw + 1.0) / 2.0
    tp_pct = float(np.clip(tp_lo + norm_tp * (tp_hi - tp_lo), tp_lo, tp_hi))
    if tp_pct < sl_pct * 1.5:
        tp_pct = float(min(sl_pct * 1.5, tp_hi))
    if profile == "scalper":
        min_scalp_sl = max(0.006, 3.0 * atr_pct)
        if sl_pct < min_scalp_sl:
            sl_pct = min_scalp_sl
            if tp_pct < sl_pct * 1.5:
                tp_pct = float(min(sl_pct * 1.5, tp_hi))
    return sl_pct, tp_pct


def main():
    from adan_trading_bot.trading.execution_engine import ExecutionEngine

    # ── Vérif 1 : la table _PROFILE_BOUNDS est identique à ENV_BOUNDS ──
    for prof, b in ENV_BOUNDS.items():
        live_b = ExecutionEngine._PROFILE_BOUNDS[prof]
        assert tuple(live_b["sl"]) == tuple(b["sl"]), f"_PROFILE_BOUNDS sl {prof} divergent"
        assert tuple(live_b["tp"]) == tuple(b["tp"]), f"_PROFILE_BOUNDS tp {prof} divergent"
    print("[BOUNDS OK] _PROFILE_BOUNDS == env _BOUNDS pour les 4 profils.")

    # ── ATR floor: l'env utilise context[0]; on passe un context_vector cohérent ──
    atr_pct = 0.002
    cv = np.zeros(17, dtype=np.float32)
    cv[0] = atr_pct          # ATR/close ratio
    cv[3] = 0.5              # bull_prob neutre (pas d'effet hors stochastic)

    grid = np.linspace(-1.0, 1.0, 9)
    n_checks = 0
    max_diff = 0.0
    for profile in ENV_BOUNDS:
        ee = ExecutionEngine(symbol="BTCUSDT", profile=profile,
                             stochastic_sltp=False, mode="paper")
        for sl_raw in grid:
            for tp_raw in grid:
                action = np.array([0.8, 0.0, 0.0, sl_raw, tp_raw], dtype=np.float32)
                dec = ee.decode_action(action, context_vector=cv)
                live_sl, live_tp = dec["sl_pct"], dec["tp_pct"]
                ref_sl, ref_tp = env_reference_decode(profile, sl_raw, tp_raw, atr_pct)
                d = max(abs(live_sl - ref_sl), abs(live_tp - ref_tp))
                max_diff = max(max_diff, d)
                assert d < 1e-9, (
                    f"DIVERGENCE {profile} sl_raw={sl_raw:.2f} tp_raw={tp_raw:.2f}: "
                    f"live=({live_sl:.5f},{live_tp:.5f}) ref=({ref_sl:.5f},{ref_tp:.5f})"
                )
                n_checks += 1
    print(f"[PARITY OK] {n_checks} combinaisons (4 profils × 9×9) — "
          f"décodage SL/TP identique env↔live (max_diff={max_diff:.2e}).")


if __name__ == "__main__":
    main()
    print("ALL DECODE-PARITY CHECKS PASSED")
