"""SPOT LONG-only TRAIN/VAL MFE and first-touch audit, never a TP_MAX lock.

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


def audit(max_states, stride, split="train", dataset_dir=None):
    raw = pd.read_parquet(PARQUET, columns=['open', 'high', 'low', 'close', 'volume'])
    from adan_trading_bot.offline.build_plan_dataset import RANGES
    from adan_trading_bot.offline.validate_plan_dataset import scalar_outcome
    from adan_trading_bot.policy.market_contract import load_market_contract
    if split not in RANGES:
        raise ValueError('TEST is forbidden for TP selection')
    market = load_market_contract()
    if dataset_dir is None:
        states, plans, manifest = build(raw, split=split, max_states=max_states, stride=stride)
    else:
        directory = Path(dataset_dir)
        manifest = json.loads((directory/'manifest.json').read_text())
        if manifest['split'] != split or manifest.get('test_touched'):
            raise ValueError('Split mismatch')
        for name in ('states','plans'):
            if hashlib.sha256((directory/f'{name}.parquet').read_bytes()).hexdigest() != manifest[f'{name}_sha256']:
                raise ValueError('Dataset hash mismatch')
        states = pd.read_parquet(directory/'states.parquet')
        plans = pd.read_parquet(directory/'plans.parquet')
    market.validate_dataset(plans,manifest)
    if not plans.plan_id.is_unique or not states.state_id.is_unique:
        raise ValueError('Duplicate IDs; no silent deduplication')
    lo, hi = RANGES[split]
    raw = raw[(raw.index >= lo) & (raw.index < hi)]
    arrays = {c:raw[c].to_numpy(dtype=float) for c in ('open','high','low','close')}
    positions = pd.Series(np.arange(len(raw)),index=raw.index)
    grid_outcomes = {}
    # Fixed grid, chosen before looking at VAL. Independent first-touch scalar.
    for tp in GRID:
        grid_outcomes[tp] = pd.DataFrame([scalar_outcome(arrays,int(positions[r.decision_timestamp]),'LONG',
                       float(r.sl_pct),tp,int(r.horizon),market.cost_rt) for r in plans.itertuples()],index=plans.index)

    plans = plans.assign(mfe_r=plans.MFE / plans.sl_pct, mae_r=plans.MAE / plans.sl_pct,
                         sl_bucket=pd.cut(plans.sl_pct, [0.0119, 0.0151, 0.0201, 0.0301], labels=['1.2-1.5%', '1.5-2.0%', '2.0-3.0%']))
    cells = []
    for (direction, horizon, bucket), g in plans.groupby(['direction', 'horizon', 'sl_bucket'], observed=True):
        row = {'direction': direction, 'horizon': int(horizon), 'sl_bucket': str(bucket), 'n': len(g),
               'mfe_r_quantiles': {str(q): float(g.mfe_r.quantile(q)) for q in (.25, .5, .75, .9, .95)},
               'p_mfe_ge': {str(k): float((g.mfe_r >= k).mean()) for k in GRID},
               'p_sl_touched': float((g.TIME_TO_SL > 0).mean()), 'p_tp35_first': float(g.Y_TP_FIRST.mean()),
               'states':int(g.state_id.nunique()),'coverage_fraction':float(len(g)/len(plans)),
               'tp_grid_economics':{str(tp):{key:float(grid_outcomes[tp].loc[g.index,key].mean())
                    for key in ('Y_TP_FIRST','Y_SL_FIRST','TIMEOUT','Y_WIN','NET_RETURN')} for tp in GRID}}
        supported = [k for k in GRID if (g.mfe_r >= k).sum() >= MIN_SUPPORT]
        row['train_proposed_tp_max'] = max(supported) if supported and split == 'train' else None
        cells.append(row)
    return {'gate':'SPOT_TP_MAX_TRAIN_VAL_COMPARISON_REQUIRED','split':split.upper(),'market':'SPOT', 'states': len(states), 'plans': len(plans),
            'min_support_count': MIN_SUPPORT, 'grid': list(GRID), 'cells': cells,
            'manifest':manifest,'market_contract_sha256':market.sha256(),'action_space_contract_sha256':market.action_space_sha256(),
            'fee_verified':market.fee_verified,'fees_rt':market.cost_rt,'saved_dataset':str(dataset_dir) if dataset_dir else None,
            'baseline_rates':{key:float(plans[key].mean()) for key in ('Y_TP_FIRST','Y_SL_FIRST','TIMEOUT','Y_WIN','NET_RETURN')},
            'profitable_timeouts':int(((plans.TIMEOUT==1)&(plans.Y_WIN==1)).sum()),
            'win_minus_tp_first':float(plans.Y_WIN.mean()-plans.Y_TP_FIRST.mean()),
            'states_per_phase':states.decision_open_ts.dt.minute.floordiv(5).add(1).value_counts().sort_index().to_dict(),
            'registry_sha256': manifest['registry_sha256'], 'outcome_function_sha256': manifest['outcome_function_sha256'],
            'interpretation': 'MFE reachability is necessary, not sufficient, for TP: SL may be touched first. Proposal must be confirmed on VAL with first-touch economics before any geometry lock. TEST untouched.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-states', type=int, default=3000); ap.add_argument('--stride', type=int, default=12)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--split',choices=('train','val'),default='train');ap.add_argument('--dataset-dir',type=Path)
    args = ap.parse_args()
    target = args.report.resolve()
    if not target.is_relative_to(Path('/home/ubuntu/webapp')) or not target.parent.is_dir():
        raise ValueError('Report parent must exist in workspace')
    result = audit(args.max_states, args.stride, split=args.split,dataset_dir=args.dataset_dir)
    target.write_text(json.dumps(result, indent=2) + '\n')
    for c in result['cells']:
        print(c['direction'], c['horizon'], c['sl_bucket'], c['n'], 'median', round(c['mfe_r_quantiles']['0.5'], 2),
              'P>=3.5', round(c['p_mfe_ge']['3.5'], 3), 'P>=5', round(c['p_mfe_ge']['5.0'], 3), 'tp35_first', round(c['p_tp35_first'], 3),
              'proposed', c['train_proposed_tp_max'])
