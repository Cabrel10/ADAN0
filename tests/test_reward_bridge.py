"""Tests du RewardBridge — pont ADDITIF ENV ↔ reward_service.

Style auto-exécutable (sans pytest), cohérent avec test_reward_service.py.

Invariants critiques :
  * classic / désactivé   → contribution == 0.0 EXACTEMENT (no-op).
  * future_guided actif    → contribution finie, bornée par max_future_contrib.
  * ne lève JAMAIS         → entrées invalides renvoient 0.0, last_error rempli.
  * from_config            → lit le bloc YAML, défaut sûr (désactivé) si absent.
  * contribution EXCLUT le pnl_net (pas de double comptage avec l'env).
"""
from __future__ import annotations

import math
import traceback

from adan_trading_bot.future_arena import RewardBridge
from adan_trading_bot.future_arena.reward_service import (
    RewardConfig, RewardMode,
)


def _approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ── 1. mode classic = no-op absolu ───────────────────────────────────────────
def test_classic_is_zero():
    br = RewardBridge(RewardConfig(mode=RewardMode.CLASSIC), enabled=True)
    out = br.contribution(closed=True, pnl_gross=0.05, close_reason="TP",
                          direction=1.0, size=0.5)
    assert out == 0.0
    assert br.is_noop is True


def test_disabled_is_zero_even_in_future_mode():
    br = RewardBridge(RewardConfig(mode=RewardMode.FUTURE_GUIDED), enabled=False)
    out = br.contribution(closed=True, pnl_gross=0.05, close_reason="AGENT_CLOSE",
                          direction=1.0, size=0.5, mfe=0.03, mae=-0.01)
    assert out == 0.0
    assert br.is_noop is True


# ── 2. future_guided actif produit une contribution finie et bornée ──────────
def test_future_guided_produces_bounded_finite():
    cfg = RewardConfig(mode=RewardMode.FUTURE_GUIDED, max_future_contrib=0.60)
    br = RewardBridge(cfg, enabled=True, seed=0)
    out = br.contribution(
        profile="intraday", timeframe="5m",
        closed=True, pnl_gross=0.02, steps_held=8, close_reason="AGENT_CLOSE",
        direction=1.0, size=0.4, sl_chosen=0.05, tp_chosen=0.10,
        mfe=0.04, mae=-0.015, mfe_residual=0.03, near_green=False,
    )
    assert math.isfinite(out)
    assert -0.60 - 1e-9 <= out <= 0.60 + 1e-9
    assert br.is_noop is False
    assert br.n_active == 1


def test_contribution_excludes_pnl_net():
    """La contribution ne doit PAS contenir le PnL net (évite double comptage)."""
    cfg = RewardConfig(mode=RewardMode.FUTURE_GUIDED)
    br = RewardBridge(cfg, enabled=True, seed=1)
    out = br.contribution(closed=True, pnl_gross=10.0, close_reason="TP",
                          direction=1.0, size=0.5, mfe=0.2, mae=-0.01)
    bd = br.last_breakdown()
    assert bd is not None
    # le terme renvoyé == future_contrib, jamais pnl_net (qui serait énorme ici).
    assert _approx(out, bd.future_contrib)
    assert abs(out) <= 0.60 + 1e-9
    assert bd.pnl_net != 0.0  # le service calcule le pnl, mais on ne le renvoie pas


# ── 3. robustesse : ne lève jamais ───────────────────────────────────────────
def test_never_raises_on_garbage():
    cfg = RewardConfig(mode=RewardMode.FUTURE_GUIDED)
    br = RewardBridge(cfg, enabled=True)
    out = br.contribution(profile=None, timeframe=None, closed="oui",  # type: ignore
                          pnl_gross="x", size=object())  # type: ignore
    assert out == 0.0
    assert br.last_error is not None


# ── 4. from_config ────────────────────────────────────────────────────────────
def test_from_config_absent_block_is_noop():
    br = RewardBridge.from_config({})
    assert br.is_noop is True
    assert br.contribution(closed=True, pnl_gross=0.05) == 0.0


def test_from_config_none_is_noop():
    br = RewardBridge.from_config(None)
    assert br.is_noop is True


def test_from_config_enabled_future_guided():
    cfg = {
        "reward_shaping": {
            "future_reward": {
                "enabled": True,
                "mode": "future_guided",
                "max_future_contrib": 0.5,
            }
        },
        "environment": {"commission": 0.004},
    }
    br = RewardBridge.from_config(cfg, seed=0)
    assert br.enabled is True
    assert br.config.mode == RewardMode.FUTURE_GUIDED
    # round_trip_fees déduit de 2×commission = 0.008
    assert _approx(br.config.round_trip_fees, 0.008)
    assert _approx(br.config.max_future_contrib, 0.5)
    assert br.is_noop is False


def test_from_config_explicit_fees_override_commission():
    cfg = {
        "reward_shaping": {"future_reward": {
            "enabled": True, "mode": "future_guided", "round_trip_fees": 0.012}},
        "environment": {"commission": 0.004},
    }
    br = RewardBridge.from_config(cfg)
    assert _approx(br.config.round_trip_fees, 0.012)


def test_from_config_fees_fallback_default():
    cfg = {"reward_shaping": {"future_reward": {
        "enabled": True, "mode": "future_guided"}}}
    br = RewardBridge.from_config(cfg)
    assert _approx(br.config.round_trip_fees, 0.008)


def test_from_config_unknown_mode_falls_back_classic():
    cfg = {"reward_shaping": {"future_reward": {
        "enabled": True, "mode": "wildcard"}}}
    br = RewardBridge.from_config(cfg)
    assert br.config.mode == RewardMode.CLASSIC
    assert br.is_noop is True  # classic = no-op même si enabled


def test_from_config_fees_from_trading_rules():
    cfg = {"reward_shaping": {"future_reward": {
        "enabled": True, "mode": "future_guided"}},
        "trading_rules": {"commission_pct": 0.003}}
    br = RewardBridge.from_config(cfg)
    assert _approx(br.config.round_trip_fees, 0.006)


# ── 5. snapshot / déterminisme ───────────────────────────────────────────────
def test_snapshot_shape():
    br = RewardBridge.from_config({"reward_shaping": {"future_reward": {
        "enabled": True, "mode": "future_guided"}}}, seed=0)
    br.contribution(closed=True, pnl_gross=0.01, close_reason="AGENT_CLOSE",
                    direction=1.0, size=0.3, mfe=0.02, mae=-0.01)
    snap = br.snapshot()
    for k in ("enabled", "mode", "round_trip_fees", "max_future_contrib",
              "n_calls", "n_active", "service"):
        assert k in snap
    assert snap["mode"] == "future_guided"
    assert snap["n_calls"] == 1


def test_determinism_same_seed_same_output():
    kw = dict(closed=True, pnl_gross=0.005, steps_held=3,
              close_reason="AGENT_CLOSE", direction=1.0, size=0.4,
              mfe=0.03, mae=-0.02, mfe_residual=0.025, near_green=False)
    a = RewardBridge(RewardConfig(mode=RewardMode.FUTURE_GUIDED), enabled=True, seed=42)
    b = RewardBridge(RewardConfig(mode=RewardMode.FUTURE_GUIDED), enabled=True, seed=42)
    seq = [dict(kw, pnl_gross=0.005 + i * 0.001) for i in range(5)]
    out_a = [a.contribution(**s) for s in seq]
    out_b = [b.contribution(**s) for s in seq]
    assert out_a == out_b


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passés")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
