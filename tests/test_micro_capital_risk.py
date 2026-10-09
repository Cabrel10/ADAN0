"""Current micro-capital configuration and sizing, separate from legacy Kelly."""
import unittest
from adan_trading_bot.policy.risk_engine import load_micro_capital_regime, size_micro_position


class MicroCapitalRiskTests(unittest.TestCase):
    def setUp(self): self.regime = load_micro_capital_regime()

    def test_real_config_and_explicit_min_order_reconciliation(self):
        self.assertEqual(self.regime.initial_capital, 20.5)
        self.assertEqual((self.regime.allocation_min,self.regime.allocation_max), (.7,.9))
        self.assertEqual(self.regime.risk_per_trade, .04)
        self.assertEqual(self.regime.max_positions, 1)
        self.assertEqual(self.regime.configured_min_notional, 5.)
        self.assertEqual(self.regime.effective_min_notional, 11.)
        self.assertTrue(self.regime.config_sha256)

    def test_micro_notional_and_nominal_risk_budget(self):
        result = size_micro_position(self.regime,20.5,.02)
        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.size_usd,16.4)
        self.assertAlmostEqual(result.f_applied,.8)
        self.assertAlmostEqual(result.risk_usd,.328)
        self.assertLessEqual(result.risk_usd,20.5*.04)
        for allocation in (.7,.9):
            result=size_micro_position(self.regime,20.5,.03,allocation_fraction=allocation)
            self.assertTrue(result.ok)
            self.assertAlmostEqual(result.size_usd,20.5*allocation)
            self.assertLessEqual(result.risk_usd,20.5*.04)

    def test_one_position_cash_risk_and_minimum_conflicts_abstain(self):
        cases = [dict(positions_open=1),dict(exposure_current=11.),dict(available_cash=10.),dict(allocation_fraction=.2)]
        for args in cases:
            self.assertFalse(size_micro_position(self.regime,20.5,.02,**args).ok)
        self.assertFalse(size_micro_position(self.regime,20.5,.08).ok)
        self.assertFalse(size_micro_position(self.regime,12.,.02).ok)
        self.assertFalse(size_micro_position(self.regime,float('nan'),.02).ok)
        self.assertFalse(size_micro_position(self.regime,20.5,0.).ok)


if __name__=='__main__':unittest.main(verbosity=2)
