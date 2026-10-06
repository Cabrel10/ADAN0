"""Normalized state/plan/outcome dataset, NEVER an all-registry flat tensor.

TRAIN/VAL only. Hypothetical filled-next-open labels; maker fill probabilities,
exchange lot/tick filters and portfolio walk-forward performance remain gates.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from adan_trading_bot.data.nested_state_builder import NestedStateBuilder
from adan_trading_bot.features.feature_registry import get_feature_registry
from adan_trading_bot.features.feature_availability_contract import FeatureAvailabilityContract, FeatureAvailabilityError
from adan_trading_bot.policy.geometry_engine import generate_candidate_grid
from adan_trading_bot.policy.risk_engine import load_micro_capital_regime, size_micro_position
from adan_trading_bot.offline.labeler_mfe_mae import compute_plan_outcomes, PARQUET

VERSION = 'plan-conditioned-conditional-fill-v1'
RANGES = {'train': ('2017-01-01','2022-01-01'), 'val': ('2022-01-01','2024-01-01')}


def build(frame, *, split='train', max_states=5000, stride=12, horizons=(48,144,288), regime=None):
    if split not in RANGES: raise ValueError('TEST dataset construction/evaluation is not authorized')
    if max_states <= 0 or stride <= 0 or not horizons or any(not isinstance(h,int) or h <= 0 for h in horizons):
        raise ValueError('Positive sample/stride/integer horizons required')
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError('Source clock must be unique and ordered')
    lo, hi = RANGES[split]
    frame = frame[(frame.index >= lo) & (frame.index < hi)]
    if frame.empty: raise ValueError('No rows in authorized partition')
    regime = load_micro_capital_regime() if regime is None else regime
    registry = get_feature_registry(); contract = FeatureAvailabilityContract(registry)
    names = contract.eligible_names()
    if not names: raise ValueError('No verified named features')
    clock = frame.index.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    cuts = np.r_[0,np.flatnonzero(np.diff(clock)!=300_000_000_000)+1,len(frame)]
    warmup, horizon_max = max(288,max(horizons)), max(horizons)
    segments = [frame.iloc[start:end] for start,end in zip(cuts[:-1],cuts[1:])]
    candidates = [(segment_id,i) for segment_id,segment in enumerate(segments)
                  for i in range(warmup,len(segment)-horizon_max,stride)]
    if not candidates: raise ValueError('No complete causal state/horizon window')
    # Deterministically cover the whole partition, not only early liquid/illiquid bars.
    if len(candidates)>max_states:
        candidates = [candidates[j] for j in np.linspace(0,len(candidates)-1,max_states,dtype=int)]
    grouped = {}
    for segment_id,i in candidates: grouped.setdefault(segment_id,[]).append(i)
    state_rows, plan_rows, counters = [], [], {'selected_decisions':len(candidates),'integrity_veto':0,
        'atr_unavailable_veto':0,'sl_interval_veto':0,'risk_veto':0}
    for segment_id, indices in grouped.items():
        segment=segments[segment_id];b=NestedStateBuilder(segment)
        pending_plans, pending_indices, pending_metadata = [], [], []
        for i in indices:
            # This is an explicitly synthetic flat micro-capital SCENARIO, not a
            # claimed historical account ledger or 745 unknowns replaced by zeros.
            portfolio=np.array([regime.initial_capital,0.,0.,0.,0.])
            snapshot=b.snapshot(i,portfolio=portfolio)
            if not snapshot.integrity_ok: counters['integrity_veto']+=1;continue
            try: values=contract.materialize(snapshot,names)
            except FeatureAvailabilityError: counters['atr_unavailable_veto']+=1;continue
            state_id=f'{split}:BTCUSDT:{snapshot.timestamp.isoformat()}'
            admitted=[]
            for direction in ('LONG','SHORT'):
                for horizon in horizons:
                    for candidate in generate_candidate_grid(snapshot,direction,availability_contract=contract,horizon=horizon):
                        sizing=size_micro_position(regime,regime.initial_capital,candidate.sl_pct)
                        if not sizing.ok: counters['risk_veto']+=1;continue
                        context={'scenario':'SYNTHETIC_FLAT_INITIAL_MICRO_CAPITAL',
                            'cash_usd':regime.initial_capital,'equity_usd':regime.initial_capital,
                            'exposure_usd':0.,'positions_open':0,'trades_today':0,'cooldown_bars':0,
                            'notional_usd':sizing.size_usd,'nominal_risk_usd':sizing.risk_usd,
                            'risk_budget_usd':regime.initial_capital*regime.risk_per_trade,
                            'allocation_fraction':sizing.f_applied,'min_order_usd':regime.effective_min_notional,
                            'config_sha256':regime.config_sha256}
                        candidate=replace(candidate,portfolio_state=context,execution_mode='CONDITIONAL_FILLED_NEXT_OPEN')
                        assert candidate.admissible()
                        admitted.append((candidate,context))
            if not admitted: counters['sl_interval_veto']+=1;continue
            state_rows.append({'state_id':state_id,'decision_open_ts':snapshot.timestamp,
                'decision_close_ts':snapshot.timestamp+pd.Timedelta(minutes=5),
                'segment_id':segment_id,'atr_available_at':snapshot.atr_1h.available_at,**values})
            for plan,context in admitted:
                pending_plans.append(plan);pending_indices.append(i)
                pending_metadata.append({'state_id':state_id,'segment_id':segment_id,
                    'sl_min_bound':plan.sl_min_bound,'sl_max_bound':plan.sl_max_bound,
                    'tp_status':plan.tp_status,'portfolio_context_json':json.dumps(context,sort_keys=True)})
        if pending_plans:
            outcomes=compute_plan_outcomes(b,pending_indices,pending_plans,fees_rt=.0008)
            for number,(row,metadata) in enumerate(zip(outcomes,pending_metadata)):
                row=dict(row);row.pop('portfolio_state')
                row.update(metadata)
                key='|'.join(str(row[x]) for x in ('state_id','direction','sl_pct','tp_r','horizon','execution_mode','portfolio_context_json'))
                row['plan_id']=hashlib.sha256(key.encode()).hexdigest()
                row['outcome_end_ts']=segment.index[pending_indices[number]+pending_plans[number].horizon]
                row['fees_rt']=.0008
                plan_rows.append(row)
    states,plans=pd.DataFrame(state_rows),pd.DataFrame(plan_rows)
    if states.empty or plans.empty: raise ValueError('No admitted states/candidates; never fabricate rows')
    if not states.state_id.is_unique or not plans.plan_id.is_unique: raise AssertionError('Duplicate state/plan IDs')
    if not set(plans.state_id).issubset(set(states.state_id)): raise AssertionError('Missing relational state')
    if not np.isfinite(states[names].to_numpy()).all(): raise AssertionError('NaN/unknown feature injected')
    if (plans.outcome_end_ts >= pd.Timestamp(hi)).any(): raise AssertionError('Outcome crosses partition boundary')
    if not ((plans.sl_min_bound<=plans.sl_pct)&(plans.sl_pct<=plans.sl_max_bound)).all(): raise AssertionError('Candidate violates bounds')
    if not ((plans.Y_TP_FIRST+plans.Y_SL_FIRST+plans.TIMEOUT)==1).all(): raise AssertionError('Outcome event partition invalid')
    source = hashlib.sha256(inspect.getsource(compute_plan_outcomes).encode()).hexdigest()
    metadata={'version':VERSION,'split':split,'state_rows':len(states),'plan_rows':len(plans),
        'feature_names':names,'registry_sha256':hashlib.sha256(Path('config/feature_registry.json').read_bytes()).hexdigest(),
        'micro_capital_regime':asdict(regime),'max_states':max_states,'stride_bars':stride,'horizons':list(horizons),
        'counters':counters,'warmup_bars':warmup,'purged_tail_bars_per_segment':horizon_max,
        'continuous_segments':len(segments),'start':str(states.decision_open_ts.min()),'end':str(states.decision_open_ts.max()),
        'outcome_function_sha256':source,'fees_rt_assumption':.0008,
        'entry_assumption':'FILLED_AT_NEXT_OPEN_CONDITIONAL_ONLY_NOT_A_MAKER_FILL_MODEL',
        'portfolio_assumption':'EXOGENOUS_SYNTHETIC_FLAT_MICRO_SCENARIO_NOT_HISTORICAL_LEDGER',
        'tp_status':'BASELINE_3_5R_ONLY_CONDITIONAL_TP_MAX_UNRESOLVED',
        'training_authorized':False,'live_trading_authorized':False,'test_touched':False,
        'seed':None,'selection':'deterministic evenly spaced causal decisions across authorized split',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
        'label_rates':{key:float(plans[key].mean()) for key in ('Y_TP_FIRST','Y_SL_FIRST','Y_WIN','TIMEOUT')},
        'interpretation':'Rows are alternative hypothetical plans, NOT executed trades. Horizons overlap; purge prevents split/gap leakage but does not make within-TRAIN rows independent. No tuning or OOS alpha inference.'}
    return states,plans,metadata


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split',choices=tuple(RANGES),default='train')
    ap.add_argument('--max-states',type=int,default=5000);ap.add_argument('--stride',type=int,default=12)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(Path('/home/ubuntu/webapp')) or not output.parent.is_dir():raise ValueError('Output parent must exist within workspace')
    if output.exists():raise ValueError('Refuse overwriting versioned dataset directory')
    raw=pd.read_parquet(PARQUET,columns=['open','high','low','close','volume'])
    states,plans,metadata=build(raw,split=args.split,max_states=args.max_states,stride=args.stride)
    metadata['git_commit']=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
    digest=hashlib.sha256()
    with open(PARQUET,'rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    metadata['dataset_path']=PARQUET;metadata['dataset_sha256']=digest.hexdigest()
    output.mkdir()
    states.to_parquet(output/'states.parquet',index=False)
    plans.to_parquet(output/'plans.parquet',index=False)
    metadata['states_sha256']=hashlib.sha256((output/'states.parquet').read_bytes()).hexdigest()
    metadata['plans_sha256']=hashlib.sha256((output/'plans.parquet').read_bytes()).hexdigest()
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k!='feature_names'},indent=2))
