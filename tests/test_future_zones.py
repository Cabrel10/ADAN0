"""Tests du module future_arena.future_zones (Lot B1/B5).

Lancer avec l'env conda trading_env :
  /home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python \
      -m pytest tests/test_future_zones.py -v
ou directement :
  .../trading_env/bin/python tests/test_future_zones.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adan_trading_bot.future_arena import (
    Zone,
    PivotDirection,
    ZoneConfig,
    wick_ratios,
    detect_pivots,
    compute_mfe_mae,
    classify_zone,
    build_critical_points,
)


def _ohlc(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


# ── wick_ratios ────────────────────────────────────────────────────────────
def test_wick_ratios_basic():
    # bougie : open=10 high=14 low=8 close=11 -> body_top=11, body_bot=10
    w_up, w_down = wick_ratios([10], [14], [8], [11])
    assert abs(w_up[0] - (14 - 11) / (14 - 8)) < 1e-9   # 3/6 = 0.5
    assert abs(w_down[0] - (10 - 8) / (14 - 8)) < 1e-9  # 2/6 ≈ 0.333


def test_wick_ratios_flat_candle():
    # high == low : pas de division par zéro
    w_up, w_down = wick_ratios([10], [10], [10], [10])
    assert w_up[0] == 0.0 and w_down[0] == 0.0


# ── detect_pivots ──────────────────────────────────────────────────────────
def test_detect_pivots_simple_v_shape():
    # un creux net au milieu : prix descend puis remonte
    closes = [100, 98, 96, 94, 90, 94, 96, 98, 100]
    rows = [(c, c + 0.5, c - 0.5, c) for c in closes]
    df = _ohlc(rows)
    cfg = ZoneConfig(fractal_k=2, min_swing_pct=0.001)
    pivots = detect_pivots(df, cfg)
    assert any(p.direction == PivotDirection.LOW for p in pivots), pivots


def test_detect_pivots_too_short():
    df = _ohlc([(1, 1, 1, 1), (1, 1, 1, 1)])
    assert detect_pivots(df, ZoneConfig(fractal_k=2)) == []


# ── compute_mfe_mae ────────────────────────────────────────────────────────
def test_mfe_mae_long():
    # entrée à 100, le futur monte à 110 (high) puis a touché 98 (low)
    closes = [100, 105, 100]
    rows = [
        (100, 100, 100, 100),  # idx 0 = entrée
        (104, 110, 98, 105),   # futur 1 : high 110, low 98
        (105, 106, 103, 100),  # futur 2
    ]
    df = _ohlc(rows)
    mfe, mae = compute_mfe_mae(df, 0, PivotDirection.LOW, horizon=5)
    assert abs(mfe - 0.10) < 1e-9   # (110-100)/100
    assert abs(mae - 0.02) < 1e-9   # (100-98)/100


def test_mfe_mae_short():
    rows = [
        (100, 100, 100, 100),  # idx 0 = entrée short
        (100, 103, 90, 92),    # favorable = baisse -> low 90 ; adverse = high 103
    ]
    df = _ohlc(rows)
    mfe, mae = compute_mfe_mae(df, 0, PivotDirection.HIGH, horizon=5)
    assert abs(mfe - 0.10) < 1e-9   # (100-90)/100
    assert abs(mae - 0.03) < 1e-9   # (103-100)/100


def test_mfe_mae_end_of_chunk():
    df = _ohlc([(100, 100, 100, 100)])
    mfe, mae = compute_mfe_mae(df, 0, PivotDirection.LOW, horizon=5)
    assert mfe == 0.0 and mae == 0.0  # pas de futur


# ── classify_zone ──────────────────────────────────────────────────────────
def test_classify_green():
    cfg = ZoneConfig()
    zone, q = classify_zone(mfe=0.04, mae=0.01, cfg_or=cfg) if False else classify_zone(0.04, 0.01, cfg)
    assert zone == Zone.GREEN
    assert 0.0 <= q <= 1.0


def test_classify_red():
    cfg = ZoneConfig()
    zone, _ = classify_zone(0.005, 0.02, cfg)  # rr = 0.25 <= 0.8
    assert zone == Zone.RED


def test_classify_orange():
    cfg = ZoneConfig()
    zone, _ = classify_zone(0.012, 0.011, cfg)  # rr ≈ 1.09, entre 0.8 et 1.5
    assert zone == Zone.ORANGE


# ── build_critical_points (intégration) ────────────────────────────────────
def test_build_critical_points_count_bounded():
    rng = np.random.default_rng(42)
    n = 288  # une journée 5m
    price = 100 + np.cumsum(rng.normal(0, 0.3, n))
    rows = [(p, p + abs(rng.normal(0, 0.4)), p - abs(rng.normal(0, 0.4)), p) for p in price]
    df = _ohlc(rows)
    cfg = ZoneConfig(max_points=15)
    pts = build_critical_points(df, cfg, regime=2)
    assert len(pts) <= 15
    assert all(p.regime == 2 for p in pts)
    assert all(0.0 <= p.quality_score <= 1.0 for p in pts)
    # triés par index
    assert pts == sorted(pts, key=lambda c: c.idx)


def _run_all():
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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
