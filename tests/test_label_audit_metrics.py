"""Independent expected values for label-audit baselines; not learned models."""
import math
import unittest
from adan_trading_bot.offline.audit_labels import distribution


class LabelAuditMetricsTests(unittest.TestCase):
    def test_binary_majority_entropy_nll_and_brier(self):
        result = distribution([0, 0, 0, 1], binary=True)
        expected_entropy = -(.75 * math.log(.75) + .25 * math.log(.25))
        self.assertEqual(result['counts'], {'0': 3, '1': 1})
        self.assertEqual(result['majority_accuracy'], .75)
        self.assertEqual(result['prevalence'], .25)
        self.assertAlmostEqual(result['entropy_nats'], expected_entropy)
        self.assertAlmostEqual(result['train_fitted_constant_nll'], expected_entropy)
        self.assertAlmostEqual(result['train_fitted_constant_brier'], .1875)
        self.assertAlmostEqual(result['entropy_bits'], expected_entropy / math.log(2))

    def test_multiclass_brier_is_sum_over_class_probabilities(self):
        result = distribution([0, 0, 0, 1], binary=False)
        self.assertIsNone(result['prevalence'])
        self.assertEqual(result['train_fitted_constant_brier'], .375)
        result = distribution([0, 1, 2, 3], binary=False)
        self.assertEqual(result['majority_accuracy'], .25)
        self.assertEqual(result['train_fitted_constant_brier'], .75)
        self.assertEqual(result['entropy_bits'], 2.)

    def test_constant_label_and_invalid_vectors(self):
        result = distribution([0, 0, 0], binary=True)
        self.assertEqual(result['majority_accuracy'], 1.)
        self.assertEqual(result['entropy_nats'], 0.)
        self.assertEqual(result['train_fitted_constant_brier'], 0.)
        for values in ([], [float('nan')], [0., .5], [[0, 1]]):
            with self.assertRaises(ValueError): distribution(values)
        with self.assertRaises(ValueError): distribution([0, 2], binary=True)


if __name__ == '__main__': unittest.main(verbosity=2)
