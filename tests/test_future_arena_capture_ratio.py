"""FINDING #4 / revue utilisateur — Future Arena CAPTURE-RATIO philosophy.

Prouve la nouvelle philosophie (le marche n'a pas de TP maximum) :

  1. Un MEME TP est juge differemment selon le POTENTIEL REEL (MFE) :
       - flat market  (MFE 1.8%, capture 89%) -> tp_q HAUT.
       - bullrun       (MFE 18%,  capture  9%) -> tp_q NEGATIF ("argent laisse").
  2. Un TP au-dela du MFE (jamais touche) est penalise.
  3. La capture optimale (~70%) pique au max.
  4. Le SL est juge contre le MAE reel observe (pas une cible fixe).
  5. Le FALLBACK (pas de futur) reste fonctionnel via la cible config.
  6. Les cibles sont surchargeables depuis config (anti-dette).

Chargement isole (evite le __init__ lourd du package). Run:
    python tests/test_future_arena_capture_ratio.py
"""
import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(__file__)
_FA = os.path.join(_HERE, "..", "src", "adan_trading_bot", "future_arena")


def _load_fa():
    sys.modules.setdefault("adan_trading_bot", types.ModuleType("adan_trading_bot"))
    pkg = types.ModuleType("fa")
    pkg.__path__ = [_FA]
    sys.modules["fa"] = pkg
    mods = {}
    for sub in ("future_zones", "escalation", "reward_service", "reward_bridge"):
        spec = importlib.util.spec_from_file_location(
            f"fa.{sub}", os.path.join(_FA, f"{sub}.py")
        )
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"fa.{sub}"] = m
        spec.loader.exec_module(m)
        mods[sub] = m
    return mods


def main():
    fa = _load_fa()
    rs = fa["reward_service"]
    rb = fa["reward_bridge"]
    tp_quality, sl_quality = rs.tp_quality, rs.sl_quality

    # ── 1. SAME TP, DIFFERENT MARKET ──────────────────────────────────────────
    flat = tp_quality(0.016, 0.030, mfe=0.018)   # capture ~0.89
    bull = tp_quality(0.016, 0.030, mfe=0.18)    # capture ~0.09
    print(f"[1] TP=1.6% FLAT(MFE=1.8%,cap=0.89): tp_q={flat:+.3f}")
    print(f"    TP=1.6% BULL(MFE=18%, cap=0.09): tp_q={bull:+.3f}")
    assert flat > bull, "flat must score higher than bull for same small TP"
    assert flat > 0.4, "capturing 89% must be rewarded"
    assert bull < 0.0, "capturing 9% in a bullrun must be penalized"

    # ── 2. TP beyond real potential (never touched) ───────────────────────────
    utopian = tp_quality(0.10, 0.030, mfe=0.02)  # capture 5.0
    print(f"[2] TP=10% MFE=2% (cap=5.0): tp_q={utopian:+.3f}")
    assert utopian < 0.0, "unreachable TP must be penalized"

    # ── 3. optimal capture peaks ──────────────────────────────────────────────
    opt = tp_quality(0.014, 0.030, mfe=0.02)     # capture 0.70
    print(f"[3] TP=1.4% MFE=2% (cap=0.70): tp_q={opt:+.3f}")
    assert opt > 0.9, "optimal capture must peak"

    # ── 4. SL judged by real MAE ──────────────────────────────────────────────
    sl_tight = sl_quality(0.003, 0.012, mae=0.008)   # SL < MAE
    sl_opt = sl_quality(0.010, 0.012, mae=0.008)     # ~1.25x MAE
    sl_wide = sl_quality(0.030, 0.012, mae=0.008)    # > 2.5x MAE
    print(f"[4] SL tight={sl_tight:+.3f} opt={sl_opt:+.3f} wide={sl_wide:+.3f}")
    assert sl_tight < 0 and sl_opt > 0 and sl_wide < sl_opt

    # ── 5. fallback works (no future data) ────────────────────────────────────
    fb = tp_quality(0.030, 0.030, mfe=None)
    print(f"[5] fallback no-MFE TP=3% target=3%: tp_q={fb:+.3f}")
    assert fb > 0.8, "fallback (gauss) must reward proximity to target"

    # ── 6. config-driven targets (anti-dette) ─────────────────────────────────
    rs.configure_targets(targets={"scalper": [0.005, 0.009]}, tf_scale={"1h": 2.0})
    assert rs.profile_tf_targets("scalper", "5m") == (0.005, 0.009)
    assert abs(rs.profile_tf_targets("scalper", "1h")[1] - 0.018) < 1e-9
    print("[6] config-driven targets applied OK")

    # ── 7. bridge end-to-end still active + capped ────────────────────────────
    cfg = {"reward_shaping": {"future_reward": {
        "enabled": True, "mode": "future_guided",
        "round_trip_fees": 0.005, "max_future_contrib": 0.60,
        "targets": {"intraday": [0.012, 0.030]},
        "weights": {"w_tp": 0.20, "w_eqs": 0.35},
    }}}
    bridge = rb.RewardBridge.from_config(cfg, seed=0)
    assert not bridge.is_noop
    c = bridge.contribution(
        profile="intraday", timeframe="5m", closed=True, pnl_gross=0.01,
        steps_held=10, close_reason="TP", direction=1.0, size=0.3,
        sl_chosen=0.008, tp_chosen=0.016, mfe=0.018, mae=0.008,
    )
    assert abs(c) <= 0.60 + 1e-6
    print(f"[7] bridge active, contrib={c:+.4f} (within cap), w_tp={bridge.config.w_tp}")

    print("\nALL CAPTURE-RATIO CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
