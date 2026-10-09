"""Independent raw-bar oracle for candidate-conditioned hypothetical-fill truth."""
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from adan_trading_bot.data.nested_state_builder import NestedStateBuilder
from adan_trading_bot.policy.geometry_engine import PlanCandidate
from adan_trading_bot.offline.labeler_mfe_mae import compute_plan_outcomes

from adan_trading_bot.policy.market_contract import load_market_contract, MarketContractError
FEE = load_market_contract().cost_rt
FIELDS = ['Y_WIN','Y_TP_FIRST','Y_SL_FIRST','MFE','MAE','TIME_TO_TP','TIME_TO_SL','NET_RETURN','TIMEOUT']


def naive(frame, i, plan, fee):
    entry = float(frame.open.iloc[i + 1]); short = plan.direction == 'SHORT'
    stop = entry * (1 + plan.sl_pct if short else 1 - plan.sl_pct)
    target = entry * (1 - plan.sl_pct * plan.tp_r if short else 1 + plan.sl_pct * plan.tp_r)
    mfe = mae = 0.; t_tp = t_sl = 0
    for j in range(i + 1, i + plan.horizon + 1):
        hi, lo = float(frame.high.iloc[j]), float(frame.low.iloc[j])
        mfe = max(mfe, (entry - lo if short else hi - entry) / entry)
        mae = max(mae, (hi - entry if short else entry - lo) / entry)
        if t_tp == 0 and (lo <= target if short else hi >= target): t_tp = j - i
        if t_sl == 0 and (hi >= stop if short else lo <= stop): t_sl = j - i
    sl = int(t_sl != 0 and (t_tp == 0 or t_sl <= t_tp))
    tp = int(t_tp != 0 and (t_sl == 0 or t_tp < t_sl))
    timeout = not (sl or tp)
    close = float(frame.close.iloc[i + plan.horizon])
    if timeout:
        exit_price = close
    elif tp:
        exit_price = target
    else:
        open_at_stop = float(frame.open.iloc[i + t_sl])
        exit_price = max(stop, open_at_stop) if short else min(stop, open_at_stop)
    gross = (entry - exit_price if short else exit_price - entry) / (entry * plan.sl_pct)
    net = gross - fee / plan.sl_pct
    return dict(Y_WIN=int(net > 0), Y_TP_FIRST=tp, Y_SL_FIRST=sl, MFE=mfe, MAE=mae,
                TIME_TO_TP=t_tp, TIME_TO_SL=t_sl, NET_RETURN=net, TIMEOUT=int(timeout))


def frame():
    return pd.DataFrame({'open':100.,'high':100.2,'low':99.8,'close':100.,'volume':1.},
                        index=pd.date_range('2020-01-01',periods=800,freq='5min'))


class PlanLabelTests(unittest.TestCase):
    def compare(self, data, indices, plans, fee=FEE):
        actual = compute_plan_outcomes(NestedStateBuilder(data), indices, plans, fees_rt=fee)
        for i, plan, row in zip(indices, plans, actual):
            expected = naive(data, i, plan, fee)
            for field in FIELDS:
                self.assertAlmostEqual(float(row[field]),float(expected[field]),places=12,msg=field)
        return actual

    def test_entry_bar_touch_same_bar_ambiguity_and_symmetry(self):
        for direction in ('LONG',):
            for event in ('tp','sl','both'):
                data=frame(); plan=PlanCandidate(direction,.012,3.5,horizon=12)
                if event in ('tp','both'): data.iloc[301,data.columns.get_loc('high' if direction=='LONG' else 'low')]=104.2 if direction=='LONG' else 95.8
                if event in ('sl','both'): data.iloc[301,data.columns.get_loc('low' if direction=='LONG' else 'high')]=98.8 if direction=='LONG' else 101.2
                row=self.compare(data,[300],[plan])[0]
                self.assertEqual(row['Y_SL_FIRST'],int(event in ('sl','both')))
                self.assertEqual(row['Y_TP_FIRST'],int(event=='tp'))
                self.assertEqual(row['TIME_TO_TP'],1 if event in ('tp','both') else 0)

    def test_full_horizon_excursion_and_later_touch_not_exit_censored(self):
        data=frame();data.iloc[301,data.columns.get_loc('low')]=98.8
        data.iloc[305,data.columns.get_loc('high')]=120
        row=self.compare(data,[300],[PlanCandidate('LONG',.012,3.5,horizon=12)])[0]
        self.assertEqual(row['Y_SL_FIRST'],1);self.assertEqual(row['TIME_TO_TP'],5)
        self.assertEqual(row['MFE'],.2);self.assertEqual(row['Y_WIN'],0)

    def test_profitable_timeout_win_is_not_tp_first(self):
        data=frame();data.iloc[312,data.columns.get_loc('close')]=101
        data.iloc[312,data.columns.get_loc('high')]=101
        row=self.compare(data,[300],[PlanCandidate('LONG',.012,3.5,horizon=12)])[0]
        self.assertEqual(row['TIMEOUT'],1);self.assertEqual(row['Y_WIN'],1)
        self.assertEqual(row['Y_TP_FIRST'],0);self.assertGreater(row['NET_RETURN'],0)

    def test_costs_make_flat_timeout_loss(self):
        row=self.compare(frame(),[300],[PlanCandidate('LONG',.012,3.5,horizon=12)])[0]
        self.assertEqual(row['TIMEOUT'],1);self.assertEqual(row['Y_WIN'],0)
        self.assertAlmostEqual(row['NET_RETURN'],-FEE/.012)

    def test_same_state_different_plan_targets_and_horizons(self):
        data=frame();data.iloc[301,data.columns.get_loc('low')]=98.7
        plans=[PlanCandidate('LONG',.012,3.5,horizon=12),PlanCandidate('LONG',.02,3.5,horizon=12),PlanCandidate('LONG',.012,3.5,horizon=288)]
        rows=self.compare(data,[300]*3,plans)
        self.assertEqual(rows[0]['Y_SL_FIRST'],1);self.assertEqual(rows[1]['Y_SL_FIRST'],0)
        self.assertEqual([x['horizon'] for x in rows],[12,12,288])
        self.assertTrue(all(x['entry_assumption']=='FILLED_AT_NEXT_OPEN' for x in rows))

    def test_stop_gap_is_not_capped_at_one_R_loss(self):
        for direction, price in (('LONG',90.),):
            data=frame();data.iloc[302,:4]=[price,price+.2,price-.2,price]
            row=self.compare(data,[300],[PlanCandidate(direction,.012,3.5,horizon=12)])[0]
            self.assertEqual(row['TIME_TO_SL'],2)
            self.assertLess(row['NET_RETURN'],-8.)
            self.assertEqual(row['exit_price'],price)

    def test_seeded_raw_reference_grid(self):
        rng=np.random.default_rng(1729);data=frame()
        mid=100*np.exp(np.cumsum(rng.normal(0,.001,len(data))))
        data['open']=mid;data['close']=mid;data['high']=mid*1.003;data['low']=mid*.997
        indices=list(range(192,392,10));plans=[PlanCandidate('LONG',.012 if j%3 else .02,3.5,horizon=12 if j%2 else 288) for j in range(len(indices))]
        self.compare(data,indices,plans)

    def test_short_rejected_by_spot_contract(self):
        with self.assertRaises(MarketContractError):
            compute_plan_outcomes(NestedStateBuilder(frame()),[300],[PlanCandidate('SHORT',.012,3.5,horizon=12)],fees_rt=FEE)

    def test_incomplete_invalid_or_unsupported_inputs_rejected(self):
        data=frame();b=NestedStateBuilder(data);plan=PlanCandidate('LONG',.012,3.5,horizon=12)
        for index, bad in [(799,plan),(300,replace(plan,sl_pct=0)),(300,replace(plan,horizon=12.5))]:
            with self.assertRaises(ValueError):compute_plan_outcomes(b,[index],[bad],fees_rt=FEE)
        with self.assertRaises(ValueError):compute_plan_outcomes(b,[300],[plan],fees_rt=FEE,entry_assumption='GUARANTEED_MAKER_FILL')
        with self.assertRaises(ValueError):compute_plan_outcomes(NestedStateBuilder(data.drop(data.index[250])),[300],[plan],fees_rt=FEE)


def validate_real_train(n=1000, seed=1729):
    """Direct production vs independent scalar reference on unique TRAIN entries.

    Explicit direction/SL/TP/horizon vary independently of sweep labels. Samples
    validate arithmetic, not predictive edge: horizons overlap, no OOS claim.
    """
    import hashlib
    import inspect
    from collections import Counter
    from adan_trading_bot.offline.labeler_mfe_mae import PARQUET
    if n <= 0 or n > 20000:
        raise ValueError('Positive real sample count <=20000 required')
    source = pd.read_parquet(PARQUET, columns=['open','high','low','close','volume'])
    train = source[(source.index >= '2017-01-01') & (source.index < '2022-01-01')]
    rng = np.random.default_rng(seed); remaining = n
    errors = {field: 0 for field in FIELDS}; maxima = {field: 0. for field in FIELDS}
    examples, timestamps, all_rows, windows = [], [], [], []
    for block, start in enumerate(np.linspace(0, len(train)-5000, 12, dtype=int)):
        for offset in range(int(start), min(int(start)+20000, len(train)-5000+1), 500):
            sample = train.iloc[offset:offset+5000]
            clock = sample.index.to_numpy(dtype='datetime64[ns]').astype(np.int64)
            if (np.diff(clock)==300_000_000_000).all(): break
        else: raise ValueError('No continuous TRAIN stratum')
        count = remaining // (12-block); remaining -= count
        indices = np.sort(rng.choice(np.arange(288,len(sample)-288),size=count,replace=False))
        plans = [PlanCandidate('LONG',
                               (.012,.015,.02,.025,.03)[(block+j)%5],
                               (3.5,4.,5.)[(block+j)%3],horizon=(12,48,144,288)[(block+j)%4],
                               execution_mode='CONDITIONAL_FILLED_NEXT_OPEN') for j in range(count)]
        rows = compute_plan_outcomes(NestedStateBuilder(sample),indices,plans,fees_rt=FEE)
        for index, plan, row in zip(indices,plans,rows):
            truth = naive(sample,int(index),plan,FEE)
            timestamps.append(str(sample.index[index+1])); all_rows.append(row)
            for field in FIELDS:
                actual,expected=float(row[field]),float(truth[field])
                delta=abs(actual-expected);maxima[field]=max(maxima[field],delta)
                same=actual==expected if field not in ('MFE','MAE','NET_RETURN') else np.isclose(actual,expected,atol=1e-12,rtol=0)
                if not same:
                    errors[field]+=1
                    if len(examples)<20:examples.append({'field':field,'entry':str(row['entry_timestamp']),'actual':actual,'reference':expected})
        windows.append({'start':str(sample.index[0]),'end':str(sample.index[-1]),'decisions':count})
    assert len(timestamps)==n and len(set(timestamps))==n
    distributions={field:dict(Counter(str(x[field]) for x in all_rows)) for field in ('Y_TP_FIRST','Y_SL_FIRST','Y_WIN','TIMEOUT')}
    return {'split':'TRAIN_ONLY','n':n,'seed':seed,'atol':1e-12,'rtol':0,'divergences':errors,
            'max_absolute_error':maxima,'examples':examples,'windows':windows,'distributions':distributions,
            'win_differs_from_tp_first':sum(x['Y_WIN']!=x['Y_TP_FIRST'] for x in all_rows),
            'stop_losses_worse_than_minus_1R_plus_fees':sum(x['Y_SL_FIRST'] and x['NET_RETURN'] < -1-FEE/x['sl_pct']-1e-12 for x in all_rows),
            'entry_timestamp_sha256':hashlib.sha256('\n'.join(timestamps).encode()).hexdigest(),
            'production_function_sha256':hashlib.sha256(inspect.getsource(compute_plan_outcomes).encode()).hexdigest(),
            'reference_function_sha256':hashlib.sha256(inspect.getsource(naive).encode()).hexdigest(),
            'interpretation':'Hypothetical filled entry, explicit costs and conservative stop gaps; maker fill probability and portfolio OOS are NOT validated.'}


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    ap=argparse.ArgumentParser();ap.add_argument('--real-n',type=int,default=0);ap.add_argument('--report',type=Path)
    args=ap.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PlanLabelTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    if args.real_n:
        report=validate_real_train(args.real_n)
        print(json.dumps(report,indent=2))
        if args.report:
            path=args.report.resolve()
            if not path.is_relative_to(Path('/home/ubuntu/webapp')) or not path.parent.is_dir():raise ValueError('Report outside workspace')
            path.write_text(json.dumps(report,indent=2)+'\n')
        if any(report['divergences'].values()):raise SystemExit(1)

