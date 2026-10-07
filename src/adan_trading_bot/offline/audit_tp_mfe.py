"""TRAIN-only conditional MFE audit to inform (not select) TP_MAX.

For each admitted plan (direction x horizon x SL bucket), MFE in R units is
MFE_fraction / sl_pct over the FULL candidate horizon. We report quantiles and
P(MFE >= k R) for k in 3.5..6. A candidate TP_MAX rule is DERIVED only as the
largest TP grid value whose TRAIN hit-before-horizon support is >= a minimum
count; it is recorded as a TRAIN-derived proposal, NOT a validated geometry.
Hit-first ordering vs SL is NOT implied by MFE; first-touch economics remain in
the plan labels. No VAL/TEST, no tuning loop.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adan_trading_bot.offline.build_plan_dataset import build
from adan_trading_bot.offline.labeler_mfe_mae import PARQUET

GRID = (3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
MIN_SUPPORT = 200


def audit(max_states, stride):
    raw = pd.read_parquet(PARQUET, columns=['open', 'high', 'low', 'close', 'volume'])
    states, plans, manifest = build(raw, split='train', max_states=max_states, stride=stride)
    plans = plans.assign(mfe_r=plans.MFE / plans.sl_pct, mae_r=plans.MAE / plans.sl_pct,
                         sl_bucket=pd.cut(plans.sl_pct, [0.0119, 0.0151, 0.0201, 0.0301], labels=['1.2-1.5%', '1.5-2.0%', '2.0-3.0%']))
    cells = []
    for (direction, horizon, bucket), g in plans.groupby(['direction', 'horizon', 'sl_bucket'], observed=True):
        row = {'direction': direction, 'horizon': int(horizon), 'sl_bucket': str(bucket), 'n': len(g),
               'mfe_r_quantiles': {str(q): float(g.mfe_r.quantile(q)) for q in (.25, .5, .75, .9, .95)},
               'p_mfe_ge': {str(k): float((g.mfe_r >= k).mean()) for k in GRID},
               'p_sl_touched': float((g.TIME_TO_SL > 0).mean()), 'p_tp35_first': float(g.Y_TP_FIRST.mean())}
        supported = [k for k in GRID if (g.mfe_r >= k).sum() >= MIN_SUPPORT]
        row['train_proposed_tp_max'] = max(supported) if supported else None
        cells.append(row)
    return {'gate': 'TP_MAX_TRAIN_MFE_AUDIT_PROPOSAL_ONLY', 'split': 'TRAIN_ONLY', 'states': len(states), 'plans': len(plans),
            'min_support_count': MIN_SUPPORT, 'grid': list(GRID), 'cells': cells,
            'registry_sha256': manifest['registry_sha256'], 'outcome_function_sha256': manifest['outcome_function_sha256'],
            'interpretation': 'MFE reachability is necessary, not sufficient, for TP: SL may be touched first. Proposal must be confirmed on VAL with first-touch economics before any geometry lock. TEST untouched.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-states', type=int, default=3000); ap.add_argument('--stride', type=int, default=12)
    ap.add_argument('--report', type=Path, required=True); args = ap.parse_args()
    target = args.report.resolve()
    if not target.is_relative_to(Path('/home/ubuntu/webapp')) or not target.parent.is_dir():
        raise ValueError('Report parent must exist in workspace')
    result = audit(args.max_states, args.stride)
    target.write_text(json.dumps(result, indent=2) + '\n')
    for c in result['cells']:
        print(c['direction'], c['horizon'], c['sl_bucket'], c['n'], 'median', round(c['mfe_r_quantiles']['0.5'], 2),
              'P>=3.5', round(c['p_mfe_ge']['3.5'], 3), 'P>=5', round(c['p_mfe_ge']['5.0'], 3), 'tp35_first', round(c['p_tp35_first'], 3),
              'proposed', c['train_proposed_tp_max'])
