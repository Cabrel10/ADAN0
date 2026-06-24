"""Tests du reward-service guidé par le futur (Lot C1/C2/C3, cahier §3.4/§3.5/§10).

Couvre les garde-fous critiques de la revue utilisateur :
  - frais SPOT 0.80 % réellement déduits (PnL NET) ;
  - SL/TP cibles PAR (profil × timeframe) ;
  - le futur NE DOMINE JAMAIS le PnL réel (anti-oracle, plafond) ;
  - barrière AGENT_CLOSE dynamique = pénalité gradient (≠ no-op) ;
  - efficacité temporelle ;
  - pénalité « zone verte ratée / contresens » progressive (escalation) ;
  - reward non prévisible (escalation stateful, pas de mur statique) ;
  - mode classic vs future_guided.
"""

from __future__ import annotations

import math

from adan_trading_bot.future_arena import (
    RewardMode,
    RewardConfig,
    RewardService,
    TradeOutcome,
    ROUND_TRIP_FEES_DEFAULT,
    profile_tf_targets,
    net_pnl,
    entry_quality_score,
    sizing_quality,
    agent_close_barrier,
    temporal_efficiency,
)
from adan_trading_bot.future_arena.future_zones import (
    CriticalPoint, PivotDirection, Zone,
)


def _green(direction=PivotDirection.LOW, mfe=0.05, mae=0.01):
    return CriticalPoint(
        idx=10, direction=direction, price=100.0, timestamp=None,
        mfe=mfe, mae=mae, zone=Zone.GREEN, quality_score=0.9, confidence=0.9,
    )


# ── frais & PnL net ──────────────────────────────────────────────────────────
def test_fees_default_is_080_pct():
    assert abs(ROUND_TRIP_FEES_DEFAULT - 0.008) < 1e-12


def test_net_pnl_deducts_round_trip_fees():
    # +1.0 % brut → +0.2 % net après 0.80 % de frais.
    assert abs(net_pnl(0.010, 0.008) - 0.002) < 1e-12
    # micro-gain +0.3 % brut → NÉGATIF net (le cœur de la faille #6).
    assert net_pnl(0.003, 0.008) < 0


# ── SL/TP par profil × timeframe ──────────────────────────────────────────────
def test_targets_scale_with_timeframe():
    sl5, tp5 = profile_tf_targets("scalper", "5m")
    sl1h, tp1h = profile_tf_targets("scalper", "1h")
    sl4h, tp4h = profile_tf_targets("scalper", "4h")
    assert sl5 < sl1h < sl4h
    assert tp5 < tp1h < tp4h


def test_targets_scale_with_profile():
    sl_sc, _ = profile_tf_targets("scalper", "5m")
    sl_pos, _ = profile_tf_targets("position", "5m")
    assert sl_pos > sl_sc  # position a des bandes plus larges


# ── EQS / sizing purs ─────────────────────────────────────────────────────────
def test_eqs_green_positive_red_negative():
    assert entry_quality_score(0.06, 0.01, Zone.GREEN) > 0
    assert entry_quality_score(0.002, 0.02, Zone.RED) < 0


def test_sizing_oversize_in_red_is_penalized():
    s_big = sizing_quality(0.9, mfe=0.002, mae=0.02, zone=Zone.RED)
    s_small = sizing_quality(0.1, mfe=0.002, mae=0.02, zone=Zone.RED)
    assert s_big < s_small <= 0


# ── barrière AGENT_CLOSE = pénalité gradient (≠ no-op) ────────────────────────
def test_agent_close_barrier_blocks_micro_gain_with_gradient():
    blocked, pen = agent_close_barrier(0.003, 0.008, barrier_mult=1.5)
    assert blocked is True
    assert pen < 0          # un VRAI gradient négatif, pas un simple rejet


def test_agent_close_barrier_allows_real_profit():
    blocked, pen = agent_close_barrier(0.05, 0.008, barrier_mult=1.5)
    assert blocked is False
    assert pen == 0.0


# ── efficacité temporelle ─────────────────────────────────────────────────────
def test_temporal_efficiency_grows_with_holding():
    early = temporal_efficiency(0.05, steps_held=2, tau=12.0)
    late = temporal_efficiency(0.05, steps_held=60, tau=12.0)
    assert 0 <= early < late <= 0.05


def test_temporal_efficiency_zero_on_loss():
    assert temporal_efficiency(-0.05, steps_held=60, tau=12.0) == 0.0


# ── GARDE-FOU ANTI-ORACLE : le futur ne domine jamais le PnL réel ─────────────
def test_future_never_dominates_pnl():
    cfg = RewardConfig(mode=RewardMode.FUTURE_GUIDED, max_future_contrib=0.6)
    svc = RewardService(cfg, seed=0)
    # trade parfait en zone verte MAIS perte réelle nette : le PnL doit peser.
    ev = TradeOutcome(
        profile="scalper", timeframe="5m", direction=1.0, size=0.5,
        sl_chosen=0.012, tp_chosen=0.030, closed=True, pnl_gross=-0.02,
        steps_held=20, close_reason="SL",
        mfe=0.06, mae=0.01, near_green=True, nearest_green=_green(),
    )
    bd = svc.compute(ev)
    # contribution future bornée
    assert abs(bd.future_contrib) <= cfg.max_future_contrib + 1e-9
    # le PnL net réel reste fortement négatif (perte non masquée par les zones)
    assert bd.pnl_net < 0


# ── pénalité « zone verte ratée » progressive (escalation) ────────────────────
def test_missed_green_is_progressive_not_static():
    cfg = RewardConfig(mode=RewardMode.FUTURE_GUIDED)
    svc = RewardService(cfg, seed=0)
    # HOLD répété près d'un 🟢 (direction ~0) → pénalité qui ESCALADE.
    pens = []
    for _ in range(12):
        ev = TradeOutcome(
            profile="scalper", timeframe="5m", direction=0.0, size=0.0,
            closed=False, near_green=True, nearest_green=_green(),
            mfe=0.06, mae=0.01,
        )
        pens.append(svc.compute(ev).missed_green)
    nonzero = [p for p in pens if p < 0]
    # rien à la 1ʳᵉ occurrence (grâce) puis pénalités présentes et croissantes
    assert pens[0] == 0.0
    assert len(nonzero) >= 3
    mags = [abs(p) for p in nonzero]
    assert mags[-1] > mags[0]


def test_aligned_trade_in_green_repays_debt():
    cfg = RewardConfig(mode=RewardMode.FUTURE_GUIDED)
    svc = RewardService(cfg, seed=0)
    # on accumule de la dette en HOLD près d'un 🟢...
    for _ in range(10):
        svc.compute(TradeOutcome(
            profile="scalper", timeframe="5m", direction=0.0, size=0.0,
            closed=False, near_green=True, nearest_green=_green(),
            mfe=0.06, mae=0.01))
    debt = svc._esc.repetitions("hold_in_green")
    assert debt > 0
    # ... puis on prend le trade DANS LE BON SENS (long sur pivot LOW) → reset.
    svc.compute(TradeOutcome(
        profile="scalper", timeframe="5m", direction=1.0, size=0.5,
        closed=False, near_green=True, nearest_green=_green(PivotDirection.LOW),
        mfe=0.06, mae=0.01))
    assert svc._esc.repetitions("hold_in_green") < debt


# ── mode classic ignore le futur ──────────────────────────────────────────────
def test_classic_mode_ignores_future_terms():
    svc = RewardService(RewardConfig(mode=RewardMode.CLASSIC), seed=0)
    ev = TradeOutcome(
        profile="scalper", timeframe="5m", direction=1.0, size=0.9,
        sl_chosen=0.5, tp_chosen=0.0001, closed=False,
        near_green=True, nearest_green=_green(), mfe=0.06, mae=0.01)
    bd = svc.compute(ev)
    assert bd.eqs == 0.0 and bd.sl_q == 0.0 and bd.tp_q == 0.0
    assert bd.future_contrib == 0.0


# ── reproductibilité (même seed → même trajectoire) ──────────────────────────
def test_reproducible_with_seed():
    def run():
        svc = RewardService(RewardConfig(), seed=123)
        out = []
        for _ in range(15):
            out.append(svc.compute(TradeOutcome(
                profile="swing", timeframe="1h", direction=0.0, size=0.0,
                closed=False, near_green=True, nearest_green=_green(),
                mfe=0.06, mae=0.01)).final)
        return out
    assert run() == run()


# ── le step PnL gagnant net est bien positif final ───────────────────────────
def test_real_win_yields_positive_reward():
    svc = RewardService(RewardConfig(mode=RewardMode.FUTURE_GUIDED), seed=0)
    ev = TradeOutcome(
        profile="scalper", timeframe="5m", direction=1.0, size=0.4,
        sl_chosen=0.012, tp_chosen=0.030, closed=True, pnl_gross=0.04,
        steps_held=18, close_reason="TP",
        mfe=0.05, mae=0.01, near_green=True,
        nearest_green=_green(PivotDirection.LOW))
    bd = svc.compute(ev)
    assert bd.final > 0
    assert bd.pnl_net > 0


def _run_all():
    import traceback
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
