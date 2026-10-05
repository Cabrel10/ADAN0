"""GATE 2 named availability boundary and future mutation regressions."""
import unittest
from dataclasses import replace

import pandas as pd

from adan_trading_bot.data.nested_state_builder import NestedStateBuilder
from adan_trading_bot.features.feature_registry import FeatureRegistry, get_feature_registry
from adan_trading_bot.features.feature_availability_contract import (
    FeatureAvailabilityContract, FeatureAvailabilityError,
)


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.registry = get_feature_registry()
        self.contract = FeatureAvailabilityContract(self.registry)
        self.frame = pd.DataFrame({'open': 100., 'high': 101., 'low': 99.,
                                   'close': 100., 'volume': 1.},
                                  index=pd.date_range('2020-01-01', periods=600, freq='5min'))

    def test_every_declaration_has_exclusive_category_and_status(self):
        entries = self.registry.all_variables()
        self.assertEqual(len(entries), 1026)
        self.assertEqual(sum(x.future_safe == 'VERIFIED' for x in entries), 283)
        self.assertEqual(sum(x.future_safe == 'UNKNOWN' for x in entries), 743)
        self.assertEqual(sum(x.future_safe == 'UNSAFE' for x in entries), 0)
        self.assertEqual(sum(x.status == 'UNRESOLVED' for x in entries), 743)
        for entry in entries:
            self.assertTrue(entry.reason)
            self.assertIsNotNone(entry.missing_source)
            self.assertIsNotNone(entry.lineage)
            if entry.status == 'UNRESOLVED':
                self.assertTrue(entry.missing_runtime_mapping)
                self.assertFalse(entry.available_at_t)
                self.assertEqual(entry.future_safe, 'UNKNOWN')

    def test_no_1026_auto_input(self):
        eligible = set(self.contract.eligible_names())
        self.assertEqual(len(eligible), 283)
        for entry in self.registry.all_variables():
            if entry.name in eligible:
                self.contract.require(entry.name)
            else:
                with self.assertRaises(FeatureAvailabilityError):
                    self.contract.require(entry.name)

    def test_all_label_and_config_sources_prohibited(self):
        for entry in self.registry.all_variables():
            if entry.source.startswith('labeler.') or entry.name.startswith('config.'):
                self.assertIn(entry.category, ('LABEL_ONLY', 'CONFIG_ONLY'))
                with self.assertRaises(FeatureAvailabilityError):
                    self.contract.require(entry.name)

    def test_materialization_and_future_mutation(self):
        i = 300
        before = self.contract.materialize(NestedStateBuilder(self.frame).snapshot(i))
        edited = self.frame.copy()
        edited.iloc[i + 1:, :4] *= 7
        edited.iloc[i + 1:, 4] *= 4
        after = self.contract.materialize(NestedStateBuilder(edited).snapshot(i))
        self.assertEqual(before, after)
        self.assertEqual(len(before), 283)
        self.assertNotIn('rsi_7', before)
        with self.assertRaises(FeatureAvailabilityError):
            self.contract.materialize(NestedStateBuilder(self.frame).snapshot(i), ['rsi_7'])

    def test_integrity_and_duplicate_names_rejected(self):
        with self.assertRaises(FeatureAvailabilityError):
            self.contract.materialize(NestedStateBuilder(self.frame).snapshot(10))
        with self.assertRaises(FeatureAvailabilityError):
            self.contract.materialize(NestedStateBuilder(self.frame).snapshot(300), ['bar_5m.close'] * 2)

    def test_forged_label_source_cannot_bypass_contract(self):
        good = self.registry['bar_5m.close']
        bad = replace(good, source='labeler.future_close')
        registry = FeatureRegistry([bad])
        with self.assertRaises(FeatureAvailabilityError):
            FeatureAvailabilityContract(registry).require(bad.name)

    def test_stale_proof_or_unsafe_status_rejected(self):
        good = self.registry['bar_5m.close']
        for bad in (replace(good, verification=dict(good.verification, producer_sha256='stale')),
                    replace(good, future_safe='UNSAFE')):
            with self.assertRaises(FeatureAvailabilityError):
                FeatureAvailabilityContract(FeatureRegistry([bad])).require(bad.name)

    def test_missing_resolver_cannot_be_filled_with_zero(self):
        good = self.registry['bar_5m.close']
        bad = replace(good, name='not.mapped', verification=dict(good.verification, feature='not.mapped'))
        contract = FeatureAvailabilityContract(FeatureRegistry([bad]))
        with self.assertRaises(FeatureAvailabilityError):
            contract.materialize(NestedStateBuilder(self.frame).snapshot(300), ['not.mapped'])

    def test_atr_semantics_are_explicit_and_not_automatically_resolved(self):
        for name in ('c1h.atr_1h', 'c1h.atr_1h_pct'):
            entry = self.contract.require(name)
            self.assertEqual(entry.future_safe, 'VERIFIED')
            self.assertTrue(entry.verification.get('bridge_tests'))
        for name in ('c4h.atr_4h', 'c4h.atr_4h_pct'):
            entry = self.registry[name]
            self.assertEqual(entry.future_safe, 'UNKNOWN')
            self.assertEqual(entry.status, 'UNRESOLVED')
            self.assertIn('COMPLETE', entry.atr_definition['formula'])
            self.assertIn('lag', entry.atr_definition['formula'])
            self.assertIsNotNone(entry.atr_definition['snapshot_consistency'])
            for parent in entry.lineage['parents']:
                self.assertNotIn('running_high', parent['name'])
                self.assertNotIn('running_low', parent['name'])
            with self.assertRaises(FeatureAvailabilityError):
                self.contract.require(name)

    def test_canonical_hour_atr_is_lagged_complete_and_not_a_phase_proxy(self):
        import numpy as np
        from adan_trading_bot.offline.labeler_mfe_mae import compute_true_atr_1h
        from adan_trading_bot.policy.geometry_engine import _atr_pct_from_snapshot
        b = NestedStateBuilder(self.frame)
        atr = compute_true_atr_1h(b)
        self.assertTrue(np.isnan(atr[:168]).all())
        self.assertEqual(float(atr[168]), 2.0)
        edited = self.frame.copy()
        edited.iloc[301:, :4] *= 7
        revised = compute_true_atr_1h(NestedStateBuilder(edited))
        np.testing.assert_allclose(atr[:301], revised[:301], atol=0, rtol=0, equal_nan=True)
        with self.assertRaises(ValueError):
            _atr_pct_from_snapshot(b.snapshot(300))
        self.assertEqual(self.registry['atr_14'].atr_definition['definition_version'], 'legacy-5m-rma14')
        self.assertEqual(self.registry['c1h.atr_1h'].atr_definition['definition_version'], 'systemone-1h-sma14-tr-lag1-v1')

    def test_legacy_training_entrypoint_blocks_before_loading_data(self):
        from unittest.mock import patch
        from adan_trading_bot.offline import train_calibrated_judgments as trainer
        with patch('sys.argv', ['trainer', '--steps', '1']), patch.object(trainer, 'load_split') as loader:
            with self.assertRaises(FeatureAvailabilityError):
                trainer.main()
            loader.assert_not_called()
        with self.assertRaises(RuntimeError):
            trainer.MlpEncoder()

    def test_training_remains_blocked(self):
        self.assertFalse(self.contract.training_authorized)
        with self.assertRaises(FeatureAvailabilityError):
            self.contract.require_training()


if __name__ == '__main__':
    unittest.main(verbosity=2)
