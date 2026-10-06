"""GATE4: full TRAIN label audit, including legacy imbalance and current semantics.

No TEST/VAL selection, training, label rewriting or class-weight workaround.
Production compute_labels is called separately for continuous TRAIN segments.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from adan_trading_bot.data.nested_state_builder import NestedStateBuilder
from adan_trading_bot.offline import labeler_mfe_mae as labeler


def distribution(values, *, binary=None):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Distribution requires a nonempty finite label vector')
    if not np.equal(values, np.round(values)).all():
        raise ValueError('Categorical labels must be integer-valued')
    labels, counts = np.unique(values, return_counts=True)
    probs = counts / len(values)
    entropy = float(-(probs * np.log(probs)).sum())
    if binary is None:
        binary = set(labels).issubset({0, 1})
    if binary and not set(labels).issubset({0, 1}):
        raise ValueError('Binary target contains nonbinary classes')
    prevalence = float(values.mean()) if binary else None
    return {'n': len(values), 'counts': {str(int(k)): int(v) for k, v in zip(labels, counts)},
            'prevalence': prevalence, 'majority_accuracy': float(probs.max()),
            'majority_class': int(labels[np.argmax(counts)]), 'entropy_nats': entropy,
            'entropy_bits': entropy / np.log(2),
            'train_fitted_constant_nll': entropy,
            'train_fitted_constant_brier': prevalence * (1 - prevalence) if binary else float(1 - (probs**2).sum())}


def stats(frame):
    categorical = {'y_regime', 'y_direction', 'y_qualite_setup', 'y_conviction'}
    result = {'rows': len(frame), 'distributions': {name: distribution(frame[name], binary=name not in categorical) for name in labeler.LABEL_COLS
                                      if name.startswith('y_') and name in frame},
              'continuous': {name: {str(q): float(frame[name].quantile(q)) for q in (0, .1, .5, .9, .99, 1)}
                             for name in ('mfe', 'mae', 'net_return', 'time_to_tp', 'time_to_sl') if name in frame}}
    if 'y_win' in frame:
        tp, sl = frame.y_tp_first.to_numpy(), frame.y_sl_first.to_numpy()
        result['invariants'] = {
            'both_first_events': int(((tp == 1) & (sl == 1)).sum()),
            'y_win_not_tp_first': int((frame.y_win.to_numpy() != tp).sum()),
            'negative_excursions': int(((frame.mfe < 0) | (frame.mae < 0)).sum()),
            'timeout_count': int(((tp == 0) & (sl == 0)).sum()),
            'timeout_net_return_nonzero': int(((tp == 0) & (sl == 0) & (frame.net_return != 0)).sum()),
            'touch_time_inconsistent': int(((tp == 1) & ((frame.time_to_tp < 1) | (frame.time_to_tp > 288))).sum()
                                         + ((sl == 1) & ((frame.time_to_sl < 1) | (frame.time_to_sl > 288))).sum()),
            'aucune_direction_evaluated_as_long': int((frame.y_direction == 2).sum())}
        assert result['invariants']['both_first_events'] == 0
        assert result['invariants']['y_win_not_tp_first'] == 0
        assert result['invariants']['negative_excursions'] == 0
        assert result['invariants']['touch_time_inconsistent'] == 0
    return result


def audit():
    source = pd.read_parquet(labeler.PARQUET, columns=['open', 'high', 'low', 'close', 'volume'])
    train = source[(source.index >= '2017-01-01') & (source.index < '2022-01-01')]
    ns = train.index.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    boundaries = np.r_[0, np.flatnonzero(np.diff(ns) != 300_000_000_000) + 1, len(train)]
    parts, segments = [], []
    warmup, horizon = 288, labeler.MAX_HOLD  # >=240 volume baseline: no bfilled regime warmup
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment = train.iloc[start:end]
        record = {'start': str(segment.index[0]), 'end': str(segment.index[-1]), 'bars': len(segment),
                  'warmup_excluded': min(warmup, len(segment)), 'horizon_tail_excluded': min(horizon, max(0, len(segment)-warmup))}
        if len(segment) <= warmup + horizon:
            record['labeled'] = 0; segments.append(record); continue
        b = NestedStateBuilder(segment)
        output = labeler.compute_labels(b, labeler.build_features(b))
        keep = np.arange(warmup, len(segment) - horizon)
        labels = pd.DataFrame({k: output[k][keep] for k in labeler.LABEL_COLS}, index=segment.index[keep])
        assert (output['plan_valid'][keep] == 1).all()
        labels['year'] = labels.index.year
        labels['phase_1h'] = b.k_1h[keep]
        labels['phase_4h'] = b.m_4h[keep]
        # Measure old denominator-scale mismatch against the SAME causal stream.
        atr5 = labeler.atr14(b.o, b.h, b.l, b.c)
        fh = pd.Series(b.h[::-1]).rolling(12).max().to_numpy()[::-1]
        fl = pd.Series(b.l[::-1]).rolling(12).min().to_numpy()[::-1]
        future_range = np.roll(fh-fl, -1)
        labels['old_expansion_reconstruction'] = (future_range[keep] > 1.5 * atr5[keep]).astype(int)
        labels['old_range4h_to_atr5'] = (b.run_4h[1][keep] - b.run_4h[2][keep]) / atr5[keep]
        labels['old_trap_reconstruction'] = (labels.old_range4h_to_atr5 > 1.8).astype(int)
        record['labeled'] = len(labels); segments.append(record); parts.append(labels)
    current = pd.concat(parts)
    old = pd.read_parquet('data/labeled/train.parquet')
    if 'ts' in old:
        assert (pd.to_datetime(old.ts) < pd.Timestamp('2022-01-01')).all()
    old_stats = stats(old)
    conditionals = {}
    for field in ('year', 'phase_1h', 'y_regime', 'y_direction', 'y_sweep_confirme'):
        conditionals[field] = {}
        for value, subset in current.groupby(field):
            conditionals[field][str(int(value))] = {'n': len(subset), 'tp_first': float(subset.y_tp_first.mean()),
                'sl_first': float(subset.y_sl_first.mean()), 'win': float(subset.y_win.mean()),
                'expansion': float(subset.y_expansion_imminente.mean()), 'sweep': float(subset.y_sweep_confirme.mean()),
                'mfe_mean': float(subset.mfe.mean()), 'mae_mean': float(subset.mae.mean()),
                'net_return_r_mean': float(subset.net_return.mean())}
    report = {'gate': 'LABEL_AUDIT_NOT_TRAINING_AUTHORIZATION', 'split': 'TRAIN_ONLY',
              'current_version': labeler.LABEL_VERSION, 'dataset': labeler.PARQUET,
              'start': str(train.index[0]), 'end': str(train.index[-1]), 'source_rows': len(train),
              'labeled_rows': len(current), 'segments': segments, 'warmup_per_segment': warmup,
              'horizon': horizon, 'purge_rule': 'future outcomes cannot cross gaps or TRAIN boundary; last 288 excluded per segment',
              'legacy_artifact': {'path': 'data/labeled/train.parquet',
                                  'sha256': hashlib.sha256(Path('data/labeled/train.parquet').read_bytes()).hexdigest(),
                                  'note': 'Actual file read today; not assumed to be the original 97%/94% historical label version'},
              'dataset_sha256': hashlib.sha256(Path(labeler.PARQUET).read_bytes()).hexdigest(),
              'legacy_labels': old_stats, 'current_labels': stats(current), 'conditionals': conditionals,
              'old_imbalance_diagnostics': {'reconstructed_expansion_atr5_prevalence': float(current.old_expansion_reconstruction.mean()),
                    'current_expansion_atr1h_prevalence': float(current.y_expansion_imminente.mean()),
                    'reconstructed_trap_range4h_over_atr5_prevalence': float(current.old_trap_reconstruction.mean()),
                    'current_trap_prevalence': float((current.y_regime == 3).mean()),
                    'range4h_over_atr5_quantiles': {str(q): float(current.old_range4h_to_atr5.quantile(q)) for q in (.1,.5,.9,.99)}},
              'findings': ['OLD expansion compares 12-bar future range to 1-bar-scale ATR5; near-universal expansion is a scale mismatch, not alpha.',
                    'OLD TRAP compares growing 4h range to ATR5 and threshold1.8; phase/window scale inflates TRAP.',
                    'Current Y_WIN is TP-first by construction, not complete net-profit classification; timeouts are assigned zero return without mark-to-market/fees.',
                    'Current MFE/MAE stop on the first exit event and include the full exit bar; not full-horizon excursion and not ordered intrabar excursions.',
                    'time_to_tp/sl are first exit times only; other potential touch is censored, zero is not a time.',
                    'AUCUNE defaults to a LONG economic label; targets are NOT truly candidate-conditioned.',
                    'Regime volume baseline bfill and EMA np.roll are unsafe in warmup; audited rows exclude >=288 warmup, not silently feeding them to a model.',
                    'quality/conviction embed future TP/reintegration/expansion labels; truth only, never STATE inputs.',
                    'Regime/sweep/expansion correlations and mean returns here are descriptive associations, not causal usefulness or an OOS alpha claim.',
                    'CLASS WEIGHTS do not resolve semantic defects. Training/GPU/500K remain BLOCKED.'],
              'seed': None, 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
              'labeler_sha256': hashlib.sha256(Path(labeler.__file__).read_bytes()).hexdigest()}
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args(); target = args.report.resolve()
    if not target.is_relative_to(Path('/home/ubuntu/webapp')) or not target.parent.is_dir():
        raise ValueError('Report parent must exist in workspace')
    result = audit(); target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'rows':result['labeled_rows'], 'old_vs_new':result['old_imbalance_diagnostics'],
                      'invariants':result['current_labels']['invariants']}, indent=2))
