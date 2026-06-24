"""Tests du module future_arena.wick_stats (Lot B2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from adan_trading_bot.future_arena import (
    compute_wick_distribution,
    compute_distributions_by_regime,
    derive_sltp_targets,
    sl_quality,
    tp_quality,
)


def _ohlc(rows, regime=None):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    if regime is not None:
        df["regime"] = regime
    return df


def test_distribution_basic():
    # bougies avec mèche haute constante de ~1% du prix
    rows = [(100, 101, 100, 100.5) for _ in range(50)]
    dist = compute_wick_distribution(_ohlc(rows), "5m")
    assert dist.n_samples == 50
    # mèche haute = (101 - 100.5)/100.5 ≈ 0.497%
    assert 0.004 < dist.percentile_up(50) < 0.006


def test_distribution_empty():
    dist = compute_wick_distribution(_ohlc([]), "5m")
    assert dist.n_samples == 0


def test_distribution_missing_col_raises():
    df = pd.DataFrame({"open": [1], "high": [1], "low": [1]})  # pas de close
    try:
        compute_wick_distribution(df, "5m")
        assert False, "doit lever ValueError"
    except ValueError:
        pass


def test_distributions_by_regime_splits():
    rows_a = [(100, 100.5, 100, 100.2) for _ in range(30)]
    rows_b = [(100, 102, 98, 100) for _ in range(30)]
    df = pd.concat([_ohlc(rows_a, regime=0), _ohlc(rows_b, regime=1)], ignore_index=True)
    dists = compute_distributions_by_regime(df, "5m", regime_col="regime")
    assert set(dists.keys()) == {0, 1}
    # régime 1 a des mèches bien plus grandes que régime 0
    assert dists[1].percentile_up(50) > dists[0].percentile_up(50)


def test_distributions_no_regime_col():
    rows = [(100, 101, 99, 100) for _ in range(20)]
    dists = compute_distributions_by_regime(_ohlc(rows), "1h")
    assert set(dists.keys()) == {None}


def test_derive_sltp_long_vs_short():
    # mèches basses plus grandes que hautes -> SL long > SL short
    rows = [(100, 100.5, 98, 100) for _ in range(100)]  # grande mèche basse
    dist = compute_wick_distribution(_ohlc(rows), "5m")
    long_t = derive_sltp_targets(dist, "long", sl_percentile=90)
    short_t = derive_sltp_targets(dist, "short", sl_percentile=90)
    assert long_t.sl_target_pct > short_t.sl_target_pct
    assert long_t.sl_target_pct >= long_t.noise_floor_pct  # jamais sous le bruit


def test_derive_sltp_bad_direction():
    dist = compute_wick_distribution(_ohlc([(100, 101, 99, 100)] * 10), "5m")
    try:
        derive_sltp_targets(dist, "sideways")
        assert False
    except ValueError:
        pass


def test_sl_quality_peaks_at_target():
    from adan_trading_bot.future_arena import SLTPTargets
    t = SLTPTargets("5m", None, "long", sl_target_pct=0.02, tp_target_pct=0.04,
                    sl_percentile=90, tp_percentile=75, noise_floor_pct=0.005)
    q_good = sl_quality(0.02, t)      # pile sur la cible
    q_too_tight = sl_quality(0.005, t)
    q_too_wide = sl_quality(0.08, t)
    assert q_good > 0.99
    assert q_too_tight < q_good
    assert q_too_wide < q_good


def test_tp_quality():
    q_good = tp_quality(tp_chosen=0.04, mfe_future=0.04)
    q_tiny = tp_quality(tp_chosen=0.005, mfe_future=0.04)   # laisse de l'argent
    q_huge = tp_quality(tp_chosen=0.20, mfe_future=0.04)    # irréaliste
    assert q_good > 0.99
    assert q_tiny < q_good and q_huge < q_good
    assert tp_quality(0.04, mfe_future=0.0) == 0.0


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
