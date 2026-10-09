"""GATE 5 grouped perception contract tests (no training)."""
import json
import unittest

import numpy as np
import pandas as pd
import torch

from adan_trading_bot.data.nested_state_builder import NestedStateBuilder
from adan_trading_bot.features.feature_registry import get_feature_registry
from adan_trading_bot.features.feature_availability_contract import FeatureAvailabilityContract
from adan_trading_bot.features.relation_graph import RelationGraph
from adan_trading_bot.models.grouped_perception import (
    build_layout, encode_state, encode_plan, GroupedRelationalPerception, PerceptionContractError, SEQ_FIELDS)
from adan_trading_bot.offline.build_plan_dataset import build


def bars(n=1200, seed=7):
    rng = np.random.default_rng(seed); mid = 100 * np.exp(np.cumsum(rng.normal(0, .002, n)))
    return pd.DataFrame({'open': mid, 'high': mid * 1.006, 'low': mid * .994, 'close': mid,
                         'volume': rng.uniform(1, 3, n)}, index=pd.date_range('2020-01-01', periods=n, freq='5min'))


class GroupedPerceptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = get_feature_registry(); cls.contract = FeatureAvailabilityContract(cls.registry)
        cls.names = cls.contract.eligible_names(); cls.layout = build_layout(cls.names, cls.contract)
        data = bars()
        graph = RelationGraph(cls.registry).qualify_on_raw_train(data.iloc[:600], cls.contract)
        cls.graph = graph
        cls.states, cls.plans, cls.meta = build(data, max_states=6, horizons=(48,))
        cls.data = data

    def test_exact_partition_of_all_admitted_names(self):
        self.assertEqual(len(self.layout.names), 283)
        self.assertEqual(sorted(self.layout.names), sorted(self.names))
        self.assertEqual((len(self.layout.bar), self.layout.seq_lags, len(self.layout.c1h), len(self.layout.c4h)), (8, 35, 16, 14))

    def test_unknown_or_config_or_label_names_rejected(self):
        for bad in ('config.agent.batch_size', 'y_win', 'rsi_14', 'c4h.atr_4h'):
            with self.assertRaises(PerceptionContractError):
                build_layout(self.names + [bad], self.contract)
        with self.assertRaises(PerceptionContractError):
            build_layout([n for n in self.names if n != 'seq_5m.lag_7.close'], self.contract)

    def test_missing_extra_and_nonfinite_values_raise_never_zero(self):
        snap = NestedStateBuilder(self.data).snapshot(400)
        values = self.contract.materialize(snap, self.names)
        encoded = encode_state(self.layout, values)
        self.assertEqual(encoded['seq'].shape, (35, len(SEQ_FIELDS)))
        broken = dict(values); broken.pop('c1h.atr_1h')
        with self.assertRaises(PerceptionContractError): encode_state(self.layout, broken)
        extra = dict(values); extra['c4h.atr_4h'] = 1.0
        with self.assertRaises(PerceptionContractError): encode_state(self.layout, extra)
        bad = dict(values); bad['bar_5m.high'] = float('nan')
        with self.assertRaises(PerceptionContractError): encode_state(self.layout, bad)

    def test_scale_invariance_and_causal_sequence_order(self):
        snap = NestedStateBuilder(self.data).snapshot(400)
        values = self.contract.materialize(snap, self.names)
        scaled_data = self.data.copy(); scaled_data[['open', 'high', 'low', 'close']] *= 37.0; scaled_data['volume'] *= 5
        scaled = self.contract.materialize(NestedStateBuilder(scaled_data).snapshot(400), self.names)
        a, b = encode_state(self.layout, values), encode_state(self.layout, scaled)
        for key in a: np.testing.assert_allclose(a[key], b[key], atol=1e-5)
        close_index = SEQ_FIELDS.index('close')
        self.assertAlmostEqual(float(a['seq'][-1, close_index]), float(np.log(values['seq_5m.lag_1.close'] / values['bar_5m.close'])), places=6)
        self.assertAlmostEqual(float(a['seq'][0, close_index]), float(np.log(values['seq_5m.lag_35.close'] / values['bar_5m.close'])), places=6)

    def test_plan_encoder_rejects_outcomes_and_out_of_bounds(self):
        row = self.plans.iloc[0].to_dict(); context = json.loads(row['portfolio_context_json'])
        with self.assertRaises(PerceptionContractError): encode_plan(row, context)   # row still contains outcomes
        clean = {k: row[k] for k in ('direction', 'sl_pct', 'tp_r', 'horizon', 'sl_min_bound', 'sl_max_bound')}
        plan, port = encode_plan(clean, context)
        self.assertEqual(plan.shape, (6,)); self.assertEqual(port.shape, (4,))
        with self.assertRaises(PerceptionContractError): encode_plan(dict(clean, sl_pct=clean['sl_max_bound'] + .001), context)
        with self.assertRaises(PerceptionContractError): encode_plan(dict(clean, direction='SHORT'), context)

    def test_only_verified_intra_group_edges_and_forward_backward(self):
        torch.manual_seed(1729)
        model = GroupedRelationalPerception(self.layout, self.graph, self.contract)
        self.assertGreater(model.bar.edge_index.shape[1] + model.c1h.edge_index.shape[1] + model.c4h.edge_index.shape[1], 0)
        snap = NestedStateBuilder(self.data).snapshot(400)
        enc = encode_state(self.layout, self.contract.materialize(snap, self.names))
        row = self.plans.iloc[0].to_dict()
        plan, port = encode_plan({k: row[k] for k in ('direction', 'sl_pct', 'tp_r', 'horizon', 'sl_min_bound', 'sl_max_bound')},
                                 json.loads(row['portfolio_context_json']))
        t = lambda a: torch.tensor(a).unsqueeze(0).repeat(3, *([1] * a.ndim))
        z = model(t(enc['bar']), t(enc['seq']), t(enc['c1h']), t(enc['c4h']), t(plan), t(port))
        self.assertEqual(z.shape, (3, 128)); self.assertTrue(torch.isfinite(z).all())
        z.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))

    def test_temporal_encoder_is_causal(self):
        torch.manual_seed(0); model = GroupedRelationalPerception(self.layout, self.graph, self.contract).seq
        x = torch.randn(2, 35, 7); y = x.clone(); y[:, -1] += 5.0
        a = model.conv[0](x.transpose(1, 2))[..., :-2]; b = model.conv[0](y.transpose(1, 2))[..., :-2]
        torch.testing.assert_close(a[..., :-1], b[..., :-1])   # changing newest step never alters earlier outputs

    def test_plan_changes_z_for_same_state(self):
        torch.manual_seed(3); model = GroupedRelationalPerception(self.layout, self.graph, self.contract).eval()
        enc = encode_state(self.layout, self.contract.materialize(NestedStateBuilder(self.data).snapshot(400), self.names))
        t = lambda a: torch.tensor(a).unsqueeze(0)
        p1 = torch.tensor([[1, .018, 3.5, 1., 0., .5]]); p2 = torch.tensor([[1, .03, 3.5, .17, 1.5, 0.]])
        port = torch.tensor([[.8, .04, .8, 0.]])
        z1 = model(t(enc['bar']), t(enc['seq']), t(enc['c1h']), t(enc['c4h']), p1, port)
        z2 = model(t(enc['bar']), t(enc['seq']), t(enc['c1h']), t(enc['c4h']), p2, port)
        self.assertGreater(float((z1 - z2).abs().max()), 1e-4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
