"""ATR source, causal provenance and runtime bridge tests. No training."""
import unittest
import numpy as np
import pandas as pd

from adan_trading_bot.data.canonical_atr import compute_true_atr_1h, ATR_DEFINITION
from adan_trading_bot.data.nested_state_builder import NestedStateBuilder
from adan_trading_bot.offline import labeler_mfe_mae as labeler


def bars(n=1200, spread=1.0):
    return pd.DataFrame({'open': 100., 'high': 100. + spread, 'low': 100. - spread,
                         'close': 100., 'volume': 1.},
                        index=pd.date_range('2020-01-01', periods=n, freq='5min'))


class ATRBridgeTests(unittest.TestCase):
    def test_unique_function_and_every_timestamp_equality(self):
        self.assertIs(labeler.compute_true_atr_1h, compute_true_atr_1h)
        b = NestedStateBuilder(bars())
        truth = labeler.compute_true_atr_1h(b)
        actual = np.array([b.snapshot(i).atr_1h.value for i in range(b.n)])
        np.testing.assert_allclose(actual, truth, atol=0, rtol=0, equal_nan=True)

    def test_provenance_window_unit_lag_availability(self):
        b = NestedStateBuilder(bars()); i = 301; observation = b.snapshot(i).atr_1h
        self.assertTrue(observation.available)
        self.assertEqual(observation.definition, ATR_DEFINITION)
        self.assertEqual(observation.lag_containers, 1)
        self.assertEqual(observation.unit, 'quote_price')
        self.assertEqual(len(observation.source_hours), 14)
        self.assertEqual(observation.available_at, b.ts[i].floor('h'))
        self.assertEqual(observation.source_hours[-1].start + pd.Timedelta(hours=1), observation.available_at)
        self.assertEqual(observation.source_hours[0].start + pd.Timedelta(hours=14), observation.available_at)
        self.assertEqual(observation.value, 2.0)
        self.assertEqual(observation.fraction, 0.02)  # fraction, NOT 2 percent-points

    def test_future_mutation(self):
        frame = bars(); i = 301; original = NestedStateBuilder(frame).snapshot(i).atr_1h
        changed = frame.copy(); changed.iloc[i + 1:, :4] *= 7; changed.iloc[i + 1:, 4] *= 5
        revised = NestedStateBuilder(changed).snapshot(i).atr_1h
        self.assertEqual(original, revised)

    def test_elapsed_current_hour_mutation_including_t(self):
        frame = bars(); i = 307; hour = frame.index[i].floor('h')
        original = NestedStateBuilder(frame).snapshot(i).atr_1h
        changed = frame.copy()
        mask = (frame.index >= hour) & (frame.index <= frame.index[i])
        changed.loc[mask, ['open', 'high', 'low', 'close']] *= 3
        revised = NestedStateBuilder(changed).snapshot(i).atr_1h
        self.assertEqual(original.value, revised.value)
        self.assertEqual(original.source_hours, revised.source_hours)
        self.assertEqual(original.available_at, revised.available_at)
        # Normalization intentionally uses close_t; absolute ATR, not ATR/close,
        # must be invariant to changing the current close itself.
        self.assertAlmostEqual(revised.fraction, original.fraction / 3)

    def test_warmup_nan_and_partial_initial_hour(self):
        b = NestedStateBuilder(bars())
        for i in (0, 11, 97, 167):
            a = b.snapshot(i).atr_1h
            self.assertFalse(a.available); self.assertTrue(np.isnan(a.value))
            self.assertTrue(np.isnan(a.fraction)); self.assertIsNone(a.available_at)
        self.assertTrue(b.snapshot(168).atr_1h.available)
        partial = NestedStateBuilder(bars().iloc[5:])
        self.assertFalse(partial.snapshot(174).atr_1h.available)
        self.assertTrue(partial.snapshot(175).atr_1h.available)

    def test_true_range_uses_previous_close(self):
        frame = bars(); frame.iloc[240:252, :4] += 10
        b = NestedStateBuilder(frame); a = b.snapshot(252).atr_1h
        self.assertEqual(a.source_hours[-1].true_range, 11.0)
        self.assertEqual(a.value, (13 * 2 + 11) / 14)
        self.assertEqual(b.snapshot(251).atr_1h.value, 2.0)

    def test_gap_resets_14_complete_hour_warmup(self):
        frame = bars().drop(bars().index[245]); b = NestedStateBuilder(frame)
        at = lambda t: b.snapshot(int(b.ts.get_loc(pd.Timestamp(t)))).atr_1h
        self.assertFalse(at('2020-01-01 21:00').available)
        self.assertFalse(at('2020-01-02 10:55').available)
        self.assertTrue(at('2020-01-02 11:00').available)
        np.testing.assert_allclose(b.canonical_atr_1h.values, labeler.compute_true_atr_1h(b),
                                   rtol=0, atol=0, equal_nan=True)

    def test_invalid_hour_resets_without_zero_or_bfill(self):
        frame = bars(); frame.iloc[245, frame.columns.get_loc('high')] = np.nan
        b = NestedStateBuilder(frame)
        self.assertFalse(b.snapshot(252).atr_1h.available)
        self.assertTrue(np.isnan(b.snapshot(252).atr_1h.value))
        self.assertTrue(b.snapshot(420).atr_1h.available)


if __name__ == '__main__':
    unittest.main(verbosity=2)
