"""Dataset-level independent validation of build_plan_dataset on real TRAIN.

The production dataset (states + plans) is rebuilt and every row is re-derived
from RAW OHLCV with scalar code that does not call the builder, the snapshot,
canonical_atr, the geometry engine or compute_plan_outcomes:
  - state clock + raw bar values + independent ATR (validate_labeler.reference_atr)
  - SL bounds recomputed from that independent ATR
  - outcomes recomputed by an independent scalar loop
  - purge/gap/partition checks recomputed from raw timestamps
No training, no VAL/TEST use, no tuning.
"""
import argparse
import hashlib
import inspect
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from adan_trading_bot.offline.build_plan_dataset import build
from adan_trading_bot.offline.labeler_mfe_mae import PARQUET
from adan_trading_bot.offline.validate_labeler import reference_atr

FIELDS = ['Y_WIN', 'Y_TP_FIRST', 'Y_SL_FIRST', 'TIMEOUT', 'MFE', 'MAE', 'TIME_TO_TP', 'TIME_TO_SL', 'NET_RETURN']
FLOATS = {'MFE', 'MAE', 'NET_RETURN'}


def scalar_outcome(raw, decision_pos, direction, sl, tp_r, horizon, fee):
    # raw: dict of plain float arrays (open/high/low/close); scalar loop, no vectorized production code.
    entry = float(raw['open'][decision_pos + 1]); short = direction == 'SHORT'
    stop = entry * (1 + sl) if short else entry * (1 - sl)
    target = entry * (1 - sl * tp_r) if short else entry * (1 + sl * tp_r)
    mfe = mae = 0.0; t_tp = t_sl = 0
    for k in range(1, horizon + 1):
        hi = float(raw['high'][decision_pos + k]); lo = float(raw['low'][decision_pos + k])
        mfe = max(mfe, (entry - lo) / entry if short else (hi - entry) / entry)
        mae = max(mae, (hi - entry) / entry if short else (entry - lo) / entry)
        if not t_tp and ((lo <= target) if short else (hi >= target)): t_tp = k
        if not t_sl and ((hi >= stop) if short else (lo <= stop)): t_sl = k
    sl_first = int(bool(t_sl) and (not t_tp or t_sl <= t_tp))
    tp_first = int(bool(t_tp) and (not t_sl or t_tp < t_sl))
    if not (sl_first or tp_first):
        exit_price = float(raw['close'][decision_pos + horizon])
    elif tp_first:
        exit_price = target
    else:
        gap_open = float(raw['open'][decision_pos + t_sl])
        exit_price = max(stop, gap_open) if short else min(stop, gap_open)
    gross = ((entry - exit_price) if short else (exit_price - entry)) / (entry * sl)
    net = gross - fee / sl
    return {'Y_WIN': int(net > 0), 'Y_TP_FIRST': tp_first, 'Y_SL_FIRST': sl_first,
            'TIMEOUT': int(not (sl_first or tp_first)), 'MFE': mfe, 'MAE': mae,
            'TIME_TO_TP': t_tp, 'TIME_TO_SL': t_sl, 'NET_RETURN': net}


def validate(max_states, stride, horizons=(48, 144, 288), mutate=False):
    raw_all = pd.read_parquet(PARQUET, columns=['open', 'high', 'low', 'close', 'volume'])
    train = raw_all[(raw_all.index >= '2017-01-01') & (raw_all.index < '2022-01-01')]
    states, plans, manifest = build(raw_all, split='train', max_states=max_states,
                                    stride=stride, horizons=horizons)
    arrays = {c: train[c].to_numpy(dtype=float) for c in ('open', 'high', 'low', 'close')}
    if mutate:
        # Oracle self-test: corrupt exactly one row per field; each must be detected once.
        plans = plans.copy()
        for row, field in enumerate(FIELDS):
            column = plans.columns.get_loc(field)
            plans.iloc[row, column] = plans.iloc[row, column] + (0.5 if field in FLOATS else 1)
    pos = pd.Series(np.arange(len(train)), index=train.index)
    clock = train.index.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    gap_after = set(train.index[np.flatnonzero(np.diff(clock) != 300_000_000_000)])
    errors = {k: 0 for k in FIELDS}; maxima = {k: 0.0 for k in FIELDS}
    checks = {'state_raw_mismatch': 0, 'state_atr_mismatch': 0, 'state_nonfinite': 0,
              'bounds_mismatch': 0, 'plan_outside_bounds': 0, 'window_crosses_gap': 0,
              'label_after_partition': 0, 'entry_not_next_bar': 0, 'label_clock_mismatch': 0,
              'outcome_columns_in_state': 0, 'orphan_plan': 0, 'unexpected_tp': 0, 'short_in_spot': 0, 'fee_mismatch': 0, 'duplicate_plan_id': 0, 'sl_below_cost_floor': 0}
    examples = []
    import yaml
    rules = yaml.safe_load(open('config/config.yaml'))['trading_rules']
    assert rules['futures_enabled'] is False, 'oracle independently confirms spot config'
    cost_rt = 2 * (float(rules['commission_pct']) + float(rules['slippage_pct']))
    checks['duplicate_plan_id'] = int(plans.plan_id.duplicated().sum())
    checks['sl_below_cost_floor'] = int((plans.sl_pct < cost_rt / 0.30 - 1e-12).sum())
    leaks = {'MFE', 'MAE', 'Y_WIN', 'Y_TP_FIRST', 'Y_SL_FIRST', 'NET_RETURN', 'TIMEOUT', 'TIME_TO_TP', 'TIME_TO_SL'}
    checks['outcome_columns_in_state'] = len(leaks & set(states.columns))
    state_index = states.set_index('state_id')
    atr_cache = {}
    for _, s in states.iterrows():
        p = int(pos[s.decision_open_ts]); bar = train.iloc[p]
        if not np.isfinite(s[manifest['feature_names']].to_numpy(dtype=float)).all(): checks['state_nonfinite'] += 1
        for f in ('open', 'high', 'low', 'close', 'volume'):
            if s[f'bar_5m.{f}'] != float(bar[f]): checks['state_raw_mismatch'] += 1
        if s['seq_5m.lag_3.close'] != float(train.close.iloc[p - 3]): checks['state_raw_mismatch'] += 1
        hour = s.decision_open_ts.floor('h')
        if hour not in atr_cache:
            atr_cache[hour] = reference_atr(train, p)
        atr = atr_cache[hour]
        if not np.isclose(s['c1h.atr_1h'], atr, rtol=0, atol=1e-9): checks['state_atr_mismatch'] += 1
        if not np.isclose(s['c1h.atr_1h_pct'], atr / float(bar.close), rtol=0, atol=1e-12): checks['state_atr_mismatch'] += 1
    for _, r in plans.iterrows():
        if r.state_id not in state_index.index: checks['orphan_plan'] += 1; continue
        s = state_index.loc[r.state_id]; p = int(pos[s.decision_open_ts])
        fraction = reference_atr(train, p) / float(train.close.iloc[p])
        sl_min, sl_max = max(0.012, fraction, cost_rt / 0.30), min(0.030, 2.5 * fraction)
        if not (np.isclose(r.sl_min_bound, sl_min, atol=1e-12, rtol=0) and np.isclose(r.sl_max_bound, sl_max, atol=1e-12, rtol=0)):
            checks['bounds_mismatch'] += 1
        if not (sl_min - 1e-12 <= r.sl_pct <= sl_max + 1e-12): checks['plan_outside_bounds'] += 1
        if r.tp_r != 3.5: checks['unexpected_tp'] += 1
        if r.entry_timestamp != train.index[p + 1]: checks['entry_not_next_bar'] += 1
        end = p + int(r.horizon)
        if r.outcome_end_ts != train.index[end] or r.label_available_ts != train.index[end] + pd.Timedelta(minutes=5):
            checks['label_clock_mismatch'] += 1
        if any(ts in gap_after for ts in train.index[p:end]): checks['window_crosses_gap'] += 1
        if r.label_available_ts >= pd.Timestamp('2022-01-01'): checks['label_after_partition'] += 1
        truth = scalar_outcome(arrays, p, r.direction, float(r.sl_pct), float(r.tp_r), int(r.horizon), float(r.fees_rt))
        if r.direction != 'LONG': checks['short_in_spot'] += 1
        if not np.isclose(float(r.fees_rt), cost_rt, rtol=0, atol=1e-15): checks['fee_mismatch'] += 1
        for f in FIELDS:
            a, e = float(r[f]), float(truth[f]); d = abs(a - e); maxima[f] = max(maxima[f], d)
            same = np.isclose(a, e, atol=1e-12, rtol=0) if f in FLOATS else a == e
            if not same:
                errors[f] += 1
                if len(examples) < 20: examples.append({'plan_id': r.plan_id, 'field': f, 'actual': a, 'reference': e})
    per_geometry = plans.groupby(['direction', 'horizon']).agg(
        n=('plan_id', 'size'), tp_first=('Y_TP_FIRST', 'mean'), sl_first=('Y_SL_FIRST', 'mean'),
        timeout=('TIMEOUT', 'mean'), win=('Y_WIN', 'mean'), net_r=('NET_RETURN', 'mean')).reset_index()
    per_sl = plans.assign(sl_bucket=plans.sl_pct.round(4)).groupby('sl_bucket').agg(
        n=('plan_id', 'size'), tp_first=('Y_TP_FIRST', 'mean'), sl_first=('Y_SL_FIRST', 'mean'),
        timeout=('TIMEOUT', 'mean'), win=('Y_WIN', 'mean')).reset_index()
    counts = manifest['counters']
    return {'gate': 'GATE4_DATASET_LEVEL_ORACLE', 'split': 'TRAIN_ONLY', 'max_states': max_states, 'stride': stride,
            'horizons': list(horizons), 'states': len(states), 'plans': len(plans),
            'plans_per_state': {'mean': float(plans.groupby('state_id').size().mean()),
                                'min': int(plans.groupby('state_id').size().min()),
                                'max': int(plans.groupby('state_id').size().max())},
            'decision_abstention': {**counts, 'admitted_states': len(states),
                                    'abstention_rate': 1 - len(states) / counts['selected_decisions']},
            'field_divergences': errors, 'max_absolute_error': maxima, 'contract_checks': checks,
            'examples': examples, 'per_direction_horizon': per_geometry.to_dict('records'),
            'per_sl': per_sl.to_dict('records'), 'manifest_excerpt': {k: v for k, v in manifest.items() if k != 'feature_names'},
            'feature_count': len(manifest['feature_names']),
            'reference_sha256': hashlib.sha256((inspect.getsource(scalar_outcome) + inspect.getsource(reference_atr)).encode()).hexdigest(),
            'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
            'interpretation': 'Arithmetic/contract proof of the dataset pipeline under hypothetical next-open fill. Rows are overlapping alternative plans, not trades; rates are descriptive TRAIN statistics, not edge.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-states', type=int, default=300); ap.add_argument('--stride', type=int, default=12)
    ap.add_argument('--report', type=Path, required=True); ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    target = args.report.resolve()
    if not target.is_relative_to(Path('/home/ubuntu/webapp')) or not target.parent.is_dir():
        raise ValueError('Report parent must exist in workspace')
    if args.self_test:
        corrupted = validate(args.max_states, args.stride, mutate=True)
        detected = corrupted['field_divergences']
        assert all(detected[f] == 1 for f in FIELDS), detected
        print('ORACLE_SELF_TEST_PASS', detected)
    result = validate(args.max_states, args.stride)
    result['oracle_self_test'] = 'one corruption per field injected and detected exactly once' if args.self_test else 'not run'
    target.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps({k: result[k] for k in ('states', 'plans', 'plans_per_state', 'decision_abstention', 'field_divergences', 'contract_checks')}, indent=2, default=str))
    if any(result['field_divergences'].values()) or any(result['contract_checks'].values()):
        raise SystemExit(1)
