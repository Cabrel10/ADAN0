"""TRAIN-only ATR/SL abstention audit. No tuning, trading, GPU or training."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from adan_trading_bot.data.nested_state_builder import NestedStateBuilder
from adan_trading_bot.data import canonical_atr
from adan_trading_bot.offline.labeler_mfe_mae import compute_true_atr_1h, PARQUET
from adan_trading_bot.features.feature_registry import get_feature_registry
from adan_trading_bot.features.feature_availability_contract import FeatureAvailabilityContract
from adan_trading_bot.policy.geometry_engine import compute_sl_bounds, generate_candidate_grid


def audit(path=PARQUET):
    frame = pd.read_parquet(path, columns=['open', 'high', 'low', 'close', 'volume'])
    frame = frame[(frame.index >= '2017-01-01') & (frame.index < '2022-01-01')]
    b = NestedStateBuilder(frame)
    actual, labeler = b.canonical_atr_1h.values, compute_true_atr_1h(b)
    np.testing.assert_allclose(actual, labeler, atol=0, rtol=0, equal_nan=True)
    fraction = actual / b.c
    available = np.isfinite(fraction) & (b.c > 0)
    minimum = np.full(b.n, np.nan); maximum = minimum.copy()
    for i in np.flatnonzero(available):
        minimum[i], maximum[i] = compute_sl_bounds(float(fraction[i]))
    low = available & (fraction < 0.012 / 2.5)
    high = available & (fraction > 0.03)
    invalid_interval = available & (minimum > maximum)
    assert np.array_equal(invalid_interval, low | high)
    ns = frame.index.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    finite = np.isfinite(b.bars_5m).all(axis=1)
    finite_window = pd.Series(finite).rolling(49, min_periods=49).min().eq(1).to_numpy()
    intervals = np.r_[True, np.diff(ns) == 300_000_000_000]
    clock_window = pd.Series(intervals).rolling(48, min_periods=48).min().eq(1).to_numpy()
    ohlc = (b.h >= np.maximum(b.o, b.c)) & (b.l <= np.minimum(b.o, b.c)) & (b.l <= b.h) & (b.v >= 0)
    previous = np.isfinite(b.prev_1h).all(axis=1) & np.isfinite(b.prev_4h).all(axis=1)
    integrity = finite_window & clock_window & ohlc & previous & (np.arange(b.n) >= b.MIN_BARS)
    contract = FeatureAvailabilityContract(get_feature_registry())
    rng = np.random.default_rng(1729)
    indices = np.sort(rng.choice(np.arange(192, b.n), size=500, replace=False))
    for i in indices:
        snap = b.snapshot(int(i))
        assert snap.integrity_ok == bool(integrity[i])
        plans = generate_candidate_grid(snap, 'LONG', availability_contract=contract)
        expected = bool(integrity[i] and available[i] and not invalid_interval[i])
        assert bool(plans) == expected, (int(i), expected, len(plans))
        assert all(minimum[i] <= p.sl_pct <= maximum[i] and p.admissible() for p in plans)
    years = {}
    for year in sorted(frame.index.year.unique()):
        mask = frame.index.year == year
        count = int(mask.sum()); valid = int((mask & available).sum())
        declined = int((mask & (~available | invalid_interval)).sum())
        years[str(year)] = {'decisions': count, 'atr_unavailable': int((mask & ~available).sum()),
                           'atr_below_0_48_percent': int((mask & low).sum()),
                           'atr_above_3_percent': int((mask & high).sum()),
                           'sl_infeasible_given_atr_available': int((mask & invalid_interval).sum()) / valid if valid else None,
                           'atr_sl_abstention_rate_all_decisions': declined / count,
                           'atr_sl_feasible_decisions': count - declined,
                           'integrity_plus_atr_sl_admissible_decisions': int((mask & integrity & available & ~invalid_interval).sum()),
                           'atr_fraction_quantiles': {str(q): float(np.quantile(fraction[mask & available], q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)}}
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''): digest.update(chunk)
    return {'gate': 'ATR_BRIDGE_MEASUREMENT_NOT_FINAL_GEOMETRY_SELECTION', 'dataset': path,
            'dataset_sha256': digest.hexdigest(), 'rows': b.n,
            'start': str(frame.index[0]), 'end': str(frame.index[-1]), 'split': 'TRAIN_ONLY',
            'definition': canonical_atr.ATR_DEFINITION,
            'canonical_source_sha256': hashlib.sha256(Path(canonical_atr.__file__).read_bytes()).hexdigest(),
            'registry_sha256': hashlib.sha256(Path('config/feature_registry.json').read_bytes()).hexdigest(),
            'snapshot_labeler_comparison_count': b.n, 'divergences': 0,
            'candidate_and_integrity_sample_count': 500, 'seed': 1729,
            'atr_sl_abstention_rate': float((~available | invalid_interval).mean()),
            'years': years, 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
            'interpretation': 'ATR/SL-only abstention is measured before model/EV/fills/portfolio gates. Large abstention is a constraint finding, not fixed via TEST. TP3.5R remains exploratory baseline; TP_MAX unresolved.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    target = args.report.resolve()
    if not target.is_relative_to(Path('/home/ubuntu/webapp')) or not target.parent.is_dir():
        raise ValueError('Existing report parent must be in workspace')
    result = audit(); target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
