"""Relational dataset clock/purge/provenance and feature/label separation tests."""
import json
import unittest
import numpy as np
import pandas as pd
from adan_trading_bot.offline.build_plan_dataset import build


def fixture(start='2020-01-01', n=1200):
    return pd.DataFrame({'open':100.,'high':101.,'low':99.,'close':100.,'volume':1.},
                         index=pd.date_range(start,periods=n,freq='5min'))


class DatasetContractTests(unittest.TestCase):
    def test_named_states_join_explicit_plans_and_no_label_input(self):
        states,plans,manifest=build(fixture(),max_states=5,horizons=(12,48))
        self.assertEqual(len(states),5)
        self.assertEqual(len(manifest['feature_names']),283)
        self.assertEqual(len(plans),5*3*2*2)
        self.assertTrue(states.state_id.is_unique);self.assertTrue(plans.plan_id.is_unique)
        self.assertTrue(set(plans.state_id).issubset(set(states.state_id)))
        self.assertFalse(set(states.columns)&{'MFE','MAE','Y_WIN','Y_TP_FIRST','Y_SL_FIRST','NET_RETURN'})
        self.assertFalse(any(x.startswith(('config.','y_')) for x in manifest['feature_names']))
        self.assertTrue(np.isfinite(states[manifest['feature_names']].to_numpy()).all())
        self.assertTrue(((plans.sl_min_bound<=plans.sl_pct)&(plans.sl_pct<=plans.sl_max_bound)).all())
        self.assertTrue((plans.tp_r==3.5).all())
        self.assertTrue((plans.label_available_ts==plans.outcome_end_ts+pd.Timedelta(minutes=5)).all())
        self.assertTrue((states.decision_close_ts==states.decision_open_ts+pd.Timedelta(minutes=5)).all())
        for value in plans.portfolio_context_json:
            context=json.loads(value)
            self.assertEqual(context['scenario'],'SYNTHETIC_FLAT_INITIAL_MICRO_CAPITAL')
            self.assertEqual(context['positions_open'],0)
            self.assertAlmostEqual(context['allocation_fraction'],.8)
            self.assertGreaterEqual(context['notional_usd'],11.)
            self.assertLessEqual(context['nominal_risk_usd'],context['risk_budget_usd'])
        self.assertFalse(manifest['training_authorized']);self.assertFalse(manifest['test_touched'])

    def test_future_mutation_does_not_change_same_timestamp_named_state(self):
        data=fixture();before,_,meta=build(data,max_states=5,horizons=(12,48))
        first=before.iloc[0];changed=data.copy();mask=changed.index>first.decision_open_ts
        changed.loc[mask,['open','high','low','close']]*=2
        after,_,_=build(changed,max_states=5,horizons=(12,48))
        match=after[after.state_id==first.state_id].iloc[0]
        np.testing.assert_array_equal(first[meta['feature_names']].to_numpy(dtype=float),
                                      match[meta['feature_names']].to_numpy(dtype=float))

    def test_partition_boundary_and_gaps_are_purged(self):
        data=fixture('2021-12-28',1600)
        states,plans,meta=build(data,max_states=8,horizons=(12,48))
        self.assertTrue((plans.label_available_ts<pd.Timestamp('2022-01-01')).all())
        self.assertTrue((states.decision_open_ts<pd.Timestamp('2022-01-01')).all())
        gap=fixture(n=2400).drop(fixture(n=2400).index[1100])
        states,plans,meta=build(gap,max_states=10,horizons=(12,48))
        self.assertEqual(meta['continuous_segments'],2)
        boundary=pd.Timestamp('2020-01-04 19:40')
        first_segment=plans[plans.segment_id==0]
        self.assertTrue((first_segment.label_available_ts<boundary).all())
        self.assertTrue((states[states.segment_id==1].decision_open_ts>=gap.index[1100]+pd.Timedelta(minutes=5*288)).all())

    def test_test_or_unavailable_data_cannot_be_fabricated(self):
        with self.assertRaises(ValueError):build(fixture(),split='test')
        with self.assertRaises(ValueError):build(fixture(n=100),max_states=5)
        data=fixture();data['high']=100.1;data['low']=99.9
        with self.assertRaises(ValueError):build(data,max_states=5)  # ATR .2%, empty SL interval
        duplicate=pd.concat([fixture().iloc[:10],fixture().iloc[9:]])
        with self.assertRaises(ValueError):build(duplicate,max_states=5)


if __name__=='__main__':unittest.main(verbosity=2)
