"""GATE 3 topology/provenance/message passing tests; no training or alpha claim."""
import json
import unittest
from dataclasses import replace

import torch

from adan_trading_bot.features.feature_registry import FeatureRegistry, get_feature_registry
from adan_trading_bot.features.relation_graph import (
    RelationGraph, RelationEdge, EdgeType, EdgeStatus, DependencyOrigin, SparseRelationConv, RelationalPerception,
)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.registry = get_feature_registry()
        self.graph = RelationGraph(self.registry)

    def test_sparse_topology(self):
        summary = self.graph.validate()
        self.assertEqual(summary['total_nodes'], 1026)
        self.assertEqual(summary['total_edges'], 540)
        self.assertEqual(self.graph.get_edge_index().shape, (2, 540))
        self.assertGreater(summary['sparsity_pct'], 99.9)

    def test_open_aggregations_are_not_silently_dropped(self):
        edges = {(x.source, x.target, x.edge_type) for x in self.graph.edges}
        for target in ('c1h.open', 'c4h.open'):
            self.assertIn(('bar_5m.open', target, EdgeType.AGGREGATES), edges)

    def test_membership_and_temporal_semantics(self):
        for edge in self.graph.edges:
            if edge.edge_type == EdgeType.SAME_CONTAINER:
                self.assertEqual(edge.source.split('.')[0], edge.target.split('.')[0])
            if edge.target.startswith('seq_5m.lag_'):
                self.assertNotEqual(edge.source.split('.')[0], 'bar_5m')

    def test_bad_edges_rejected(self):
        template = self.graph.edges[0]
        for edge in [replace(template, source='unknown'),
                     replace(template, target='unknown'),
                     replace(template, target=template.source),
                     replace(template, weight=float('nan')),
                     replace(template, edge_type='bogus'),
                     replace(template, origin=DependencyOrigin.PREDICTIVE_DEPENDENCY)]:
            with self.assertRaises(ValueError):
                self.graph.add_edge(edge)
        with self.assertRaises(ValueError):
            self.graph.add_edge(template)

    def test_empirical_train_only_provenance(self):
        edge = RelationEdge('bar_5m.close', 'ema_9', EdgeType.EMPIRICAL_DEPENDENCY,
                            DependencyOrigin.PREDICTIVE_DEPENDENCY)
        provenance = dict(split='TRAIN', start='2017-08-17', end='2021-12-31',
                          sample_count=1000, method='test-fixture-only', freeze_id='test-not-a-measurement')
        for bad in [None, {}, dict(provenance, split='VAL'),
                    dict(provenance, end='2022-01-01'),
                    dict(provenance, sample_count=0), dict(provenance, freeze_id='')]:
            with self.assertRaises(ValueError):
                self.graph.add_edge(replace(edge, training_provenance=bad))
        self.graph.add_edge(replace(edge, training_provenance=provenance))
        self.assertEqual(self.graph.validate()['by_origin']['PREDICTIVE_DEPENDENCY'], 1)
        # The edge above is a synthetic validation fixture, not fitted evidence.

    def test_declarations_do_not_auto_verify_constraints(self):
        from adan_trading_bot.features.feature_availability_contract import FeatureAvailabilityContract
        contract = FeatureAvailabilityContract(self.registry)
        self.assertEqual(self.graph.constraint_edges(contract), [])
        self.assertTrue(all(e.status == EdgeStatus.UNVERIFIED for e in self.graph.edges))
        with self.assertRaises(ValueError):
            self.graph.get_edge_index(verified_only=True)
        with self.assertRaises(RuntimeError):
            RelationalPerception(self.registry, self.graph)

    def test_qualified_constraints_and_reconciled_atr_lineage(self):
        import pandas as pd
        from adan_trading_bot.features.feature_availability_contract import FeatureAvailabilityContract
        contract = FeatureAvailabilityContract(self.registry)
        frame = pd.DataFrame({'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 1.},
                             index=pd.date_range('2020-01-01', periods=600, freq='5min'))
        qualified = self.graph.qualify_on_raw_train(frame, contract)
        self.assertEqual(len(qualified.constraint_edges(contract)), 283)
        self.assertEqual(qualified.get_edge_index(True, contract).shape, (2, 283))
        self.assertEqual(qualified.get_edge_weights(True, contract).shape, (283,))
        for edge in qualified.constraint_edges(contract):
            self.assertNotEqual(edge.status, EdgeStatus.UNVERIFIED)
            self.assertNotIn(self.registry[edge.source].category, ('LABEL_ONLY', 'CONFIG_ONLY', 'UNRESOLVED'))
            self.assertNotIn(self.registry[edge.target].category, ('LABEL_ONLY', 'CONFIG_ONLY', 'UNRESOLVED'))
        for edge in qualified.edges:
            if edge.target == 'c1h.atr_1h':
                self.assertIn('prev_', edge.source)
                self.assertNotEqual(edge.source_time_offset_bars, 0)
                self.assertEqual(edge.status, EdgeStatus.UNVERIFIED)
        with self.assertRaises(ValueError):
            qualified.qualify_on_raw_train(frame.set_axis(pd.date_range('2022-01-01', periods=600, freq='5min')), contract)

    def test_forged_edge_verification_rejected(self):
        template = self.graph.edges[0]
        with self.assertRaises(ValueError):
            self.graph.add_edge(replace(template, status=EdgeStatus.VERIFIED_DETERMINISTIC))
        with self.assertRaises(ValueError):
            self.graph.add_edge(replace(template, status=EdgeStatus.VERIFIED_PLAN,
                                        verification={'passed': True, 'source': template.source,
                                                      'target': template.target, 'test': 'fake'}))

    def test_predictive_train_edge_is_not_a_hard_causal_constraint(self):
        from adan_trading_bot.features.feature_availability_contract import FeatureAvailabilityContract
        contract = FeatureAvailabilityContract(self.registry)
        proof = {'passed': True, 'test': 'synthetic-provenance-fixture-not-data-evidence',
                 'source': 'bar_5m.close', 'target': 'bar_5m.open',
                 'producer_sha256': contract.producer_hash, 'adapter_sha256': contract.adapter_hash}
        edge = RelationEdge('bar_5m.close', 'bar_5m.open', EdgeType.EMPIRICAL_DEPENDENCY,
                            DependencyOrigin.PREDICTIVE_DEPENDENCY,
                            training_provenance={'split': 'TRAIN', 'start': '2017-08-17', 'end': '2021-12-31',
                                                 'method': 'test-fixture', 'sample_count': 10, 'freeze_id': 'test-only'},
                            status=EdgeStatus.PREDICTIVE_TRAIN_ONLY, verification=proof)
        graph = RelationGraph(self.registry, edges=[edge])
        self.assertEqual(graph.constraint_edges(contract), [])

    def test_explicit_empty_graph(self):
        graph = RelationGraph(self.registry, edges=[])
        self.assertEqual(graph.get_edge_index().shape, (2, 0))
        self.assertEqual(graph.validate()['total_edges'], 0)

    def test_unresolved_registry_parent_rejected(self):
        entries = self.registry.all_variables()
        entries[0] = replace(entries[0], dependances_deterministes=['not-declared'])
        with self.assertRaises(ValueError):
            RelationGraph(FeatureRegistry(entries))

    def test_sparse_scatter_equals_independent_loop_and_backprop(self):
        torch.manual_seed(1729)
        layer = SparseRelationConv(3, 4)
        x = torch.randn(2, 1026, 3, requires_grad=True)
        indices, weights = self.graph.get_edge_index(), self.graph.get_edge_weights()
        actual = layer(x, indices, weights)
        expected = layer.lin_self(x)
        for j in range(indices.shape[1]):
            src, target = int(indices[0, j]), int(indices[1, j])
            expected[:, target] = expected[:, target] + layer.lin_neighbor(x[:, src]) * weights[j]
        expected = torch.nn.functional.gelu(expected + layer.bias)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(layer.lin_neighbor.weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
