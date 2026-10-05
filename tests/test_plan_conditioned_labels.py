"""Independent raw-bar oracle for candidate-conditioned hypothetical-fill truth."""
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from adan_trading_bot.data.nested_state_builder import NestedStateBuilder
from adan_trading_bot.policy.geometry_engine import PlanCandidate
from adan_trading_bot.offline.labeler_mfe_mae import compute_plan_outcomes

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
    def compare(self, data, indices, plans, fee=.0008):
        actual = compute_plan_outcomes(NestedStateBuilder(data), indices, plans, fees_rt=fee)
        for i, plan, row in zip(indices, plans, actual):
            expected = naive(data, i, plan, fee)
            for field in FIELDS:
                self.assertAlmostEqual(float(row[field]),float(expected[field]),places=12,msg=field)
        return actual

    def test_entry_bar_touch_same_bar_ambiguity_and_symmetry(self):
        for direction in ('LONG','SHORT'):
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
        self.assertAlmostEqual(row['NET_RETURN'],-.0008/.012)

    def test_same_state_different_plan_targets_and_horizons(self):
        data=frame();data.iloc[301,data.columns.get_loc('low')]=98.7
        plans=[PlanCandidate('LONG',.012,3.5,horizon=12),PlanCandidate('LONG',.02,3.5,horizon=12),PlanCandidate('SHORT',.012,3.5,horizon=288)]
        rows=self.compare(data,[300]*3,plans)
        self.assertEqual(rows[0]['Y_SL_FIRST'],1);self.assertEqual(rows[1]['Y_SL_FIRST'],0)
        self.assertEqual([x['direction'] for x in rows],['LONG','LONG','SHORT'])
        self.assertTrue(all(x['entry_assumption']=='FILLED_AT_NEXT_OPEN' for x in rows))

    def test_stop_gap_is_not_capped_at_one_R_loss(self):
        for direction, price in (('LONG',90.),('SHORT',110.)):
            data=frame();data.iloc[302,:4]=[price,price+.2,price-.2,price]
            row=self.compare(data,[300],[PlanCandidate(direction,.012,3.5,horizon=12)])[0]
            self.assertEqual(row['TIME_TO_SL'],2)
            self.assertLess(row['NET_RETURN'],-8.)
            self.assertEqual(row['exit_price'],price)

    def test_seeded_raw_reference_grid(self):
        rng=np.random.default_rng(1729);data=frame()
        mid=100*np.exp(np.cumsum(rng.normal(0,.001,len(data))))
        data['open']=mid;data['close']=mid;data['high']=mid*1.003;data['low']=mid*.997
        indices=list(range(192,392,10));plans=[PlanCandidate('LONG' if j%2 else 'SHORT',.012 if j%3 else .02,3.5,horizon=12 if j%2 else 288) for j in range(len(indices))]
        self.compare(data,indices,plans)

    def test_incomplete_invalid_or_unsupported_inputs_rejected(self):
        data=frame();b=NestedStateBuilder(data);plan=PlanCandidate('LONG',.012,3.5,horizon=12)
        for index, bad in [(799,plan),(300,replace(plan,direction='AUCUNE')),(300,replace(plan,sl_pct=0)),(300,replace(plan,horizon=12.5))]:
            with self.assertRaises(ValueError):compute_plan_outcomes(b,[index],[bad],fees_rt=.0008)
        with self.assertRaises(ValueError):compute_plan_outcomes(b,[300],[plan],fees_rt=.0008,entry_assumption='GUARANTEED_MAKER_FILL')
        with self.assertRaises(ValueError):compute_plan_outcomes(NestedStateBuilder(data.drop(data.index[250])),[300],[plan],fees_rt=.0008)


if __name__=='__main__':unittest.main(verbosity=2)
