"""SPOT strict contract: LONG entries only, configured per-side costs, fail on SHORT."""
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd

from adan_trading_bot.policy.market_contract import load_market_contract, MarketContractError
from adan_trading_bot.data.nested_state_builder import NestedStateBuilder
from adan_trading_bot.features.feature_registry import get_feature_registry
from adan_trading_bot.features.feature_availability_contract import FeatureAvailabilityContract
from adan_trading_bot.policy.geometry_engine import generate_candidate_grid, compute_geometry, PlanCandidate
from adan_trading_bot.offline.labeler_mfe_mae import compute_plan_outcomes
from adan_trading_bot.offline.build_plan_dataset import build


def bars(spread=1.5, n=1200):
    return pd.DataFrame({'open': 100., 'high': 100. + spread, 'low': 100. - spread, 'close': 100., 'volume': 1.},
                        index=pd.date_range('2020-01-01', periods=n, freq='5min'))


class SpotContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market = load_market_contract(); cls.contract = FeatureAvailabilityContract(get_feature_registry())

    def test_config_is_spot_long_only_with_per_side_costs(self):
        m = self.market
        self.assertEqual(m.market, 'SPOT'); self.assertEqual(m.entry_directions, ('LONG',))
        self.assertEqual(m.actions, ('BUY', 'SELL_EXIT', 'HOLD'))
        self.assertEqual((m.commission_per_side, m.slippage_per_side), (0.002, 0.0005))
        self.assertAlmostEqual(m.cost_rt, 0.005); self.assertAlmostEqual(m.min_sl_for_costs, 0.005 / 0.30)
        self.assertEqual(len(m.sha256()), 64)

    def test_short_rejected_everywhere(self):
        snap = NestedStateBuilder(bars()).snapshot(301)
        with self.assertRaises(MarketContractError):
            generate_candidate_grid(snap, 'SHORT', availability_contract=self.contract)
        self.assertFalse(compute_geometry('SHORT', 100., 101., snapshot=snap, availability_contract=self.contract).viable)
        with self.assertRaises(MarketContractError):
            compute_plan_outcomes(NestedStateBuilder(bars()), [300], [PlanCandidate('SHORT', .02, 3.5, horizon=12)], fees_rt=self.market.cost_rt)
        self.assertFalse(PlanCandidate('SHORT', .02, 3.5, horizon=12).admissible())

    def test_long_candidates_respect_cost_floor(self):
        snap = NestedStateBuilder(bars()).snapshot(301)   # ATR 3% -> bounds [3%,3%]
        plans = generate_candidate_grid(snap, 'LONG', availability_contract=self.contract)
        self.assertTrue(plans and all(p.direction == 'LONG' and p.sl_pct >= self.market.min_sl_for_costs - 1e-15 for p in plans))
        low = NestedStateBuilder(bars(spread=0.7)).snapshot(301)  # ATR 1.4% -> [1.667%, 3%]
        plans = generate_candidate_grid(low, 'LONG', availability_contract=self.contract)
        self.assertAlmostEqual(min(p.sl_pct for p in plans), 0.005 / 0.30)
        self.assertTrue(all(p.cost_r_ok for p in plans) if hasattr(plans[0], 'cost_r_ok') else True)

    def test_futures_config_refused_without_execution_path(self):
        import tempfile, yaml, pathlib
        config = yaml.safe_load(open('config/config.yaml')); config['trading_rules']['futures_enabled'] = True
        with tempfile.TemporaryDirectory(dir='/home/ubuntu/webapp') as d:
            path = pathlib.Path(d) / 'c.yaml'; path.write_text(yaml.safe_dump(config))
            with self.assertRaises(MarketContractError): load_market_contract(str(path))

    def test_short_in_spot_dataset_fails_immediately_even_with_valid_hashes(self):
        _, plans, manifest = build(bars(), max_states=2, horizons=(12,))
        plans.loc[plans.index[-1], 'direction'] = 'SHORT'
        with self.assertRaisesRegex(MarketContractError, 'SHORT'):
            self.market.validate_dataset(plans, manifest)

    def test_risk_rejects_short(self):
        from adan_trading_bot.policy.risk_engine import size_micro_position, load_micro_capital_regime
        with self.assertRaises(MarketContractError):
            size_micro_position(load_micro_capital_regime(), 20.5, .02, direction='SHORT')

    def test_paper_execution_buy_sell_exit_only_and_configured_costs(self):
        import tempfile
        from adan_trading_bot.trading.execution_engine import ExecutionEngine
        with tempfile.TemporaryDirectory(dir='/home/ubuntu/webapp') as d:
            engine = ExecutionEngine(log_dir=d)
            with self.assertRaises(MarketContractError):
                engine._execute_open('SELL', 100., .8, .02, .07, 0.)
            self.assertIsNone(engine._execute_close(100., 'AGENT_CLOSE', 0.))
            trade = engine._execute_open('BUY', 100., .8, .02, .07, 0.)
            self.assertIsNotNone(trade)
            self.assertEqual(engine.position.side, 'BUY')
            self.assertAlmostEqual(trade.fee_usd, trade.size_usd * self.market.commission_per_side)
            self.assertAlmostEqual(trade.price, 100.*(1+self.market.slippage_per_side))
            closed = engine._execute_close(101., 'SELL_EXIT', 1.)
            self.assertEqual(closed.side, 'SELL')
            self.assertIsNone(engine.position)
            self.assertGreaterEqual(engine.cash, 0.)

    def test_unverified_fees_block_production(self):
        with self.assertRaisesRegex(MarketContractError, 'NO-GO'):
            replace(self.market, fee_verified=False).require_verified_fees()

    def test_spot_dataset_contains_no_short_and_uses_configured_cost(self):
        states, plans, manifest = build(bars(), max_states=5, horizons=(12, 48))
        self.assertEqual(set(plans.direction), {'LONG'})
        self.assertTrue(np.allclose(plans.fees_rt, 0.005))
        self.assertEqual(manifest['market_contract']['market'], 'SPOT')
        self.assertEqual(len(manifest['market_contract_sha256']), 64)
        if (plans.direction == 'SHORT').any():
            self.fail('SHORT plan present in SPOT dataset')


if __name__ == '__main__':
    unittest.main(verbosity=2)
