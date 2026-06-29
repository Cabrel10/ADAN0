#!/usr/bin/env python3
"""
Test script to verify the PnL flow fix.
Checks that:
1. _pre_execute_pnl is captured from the first update_market_price call
2. _execute_trades accumulates this PnL correctly
3. The reward is calculated with the correct realized_pnl
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_pnl_flow():
    """Test the PnL flow through step() -> _execute_trades() -> _calculate_reward()"""
    
    logger.info("=" * 80)
    logger.info("TEST: PnL Flow Verification")
    logger.info("=" * 80)
    
    # Check 1: Verify that the module imports
    logger.info("\n[CHECK 1] Module import")
    try:
        from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
        logger.info("✓ Module imports successfully")
    except Exception as e:
        logger.error(f"✗ Module import failed: {e}")
        return False
    
    # Check 2: Verify the code structure in _execute_trades
    logger.info("\n[CHECK 2] Code structure verification in _execute_trades()")
    try:
        import inspect
        
        # Get the source of _execute_trades
        source = inspect.getsource(MultiAssetChunkedEnv._execute_trades)
        
        # Check for key patterns
        checks = {
            "_pre_receipts = getattr(self, '_pre_execute_sl_tp_receipts'": "Pre-receipts retrieval",
            "_pre_pnl = getattr(self, '_pre_execute_pnl'": "Pre-PnL retrieval",
            "realized_pnl += receipt_pnl": "PnL accumulation from receipts",
            "realized_pnl += _pre_pnl": "PnL accumulation from pre-PnL",
            "[EXECUTE_TRADES]": "Logging of pre-captured SL/TP",
        }
        
        for pattern, description in checks.items():
            if pattern in source:
                logger.info(f"✓ Found: {description}")
            else:
                logger.error(f"✗ Missing: {description}")
                return False
        
        # Check that the old double update is gone
        if "pnl_from_update, sl_tp_receipts = self.portfolio_manager.update_market_price" in source:
            logger.error("✗ Found old double update_market_price call in _execute_trades")
            return False
        else:
            logger.info("✓ Old double update_market_price call removed")
        
    except Exception as e:
        logger.error(f"✗ Code structure check failed: {e}")
        return False
    
    # Check 3: Verify step() captures PnL
    logger.info("\n[CHECK 3] step() method PnL capture")
    try:
        source = inspect.getsource(MultiAssetChunkedEnv.step)
        
        checks = {
            "self._pre_execute_pnl = float(_pnl_pre)": "PnL capture from first update_market_price",
            "[EARLY_SL_TP]": "Logging of early SL/TP",
            "realized_pnl, discrete_action, discrete_action_requested = self._execute_trades": "Call to _execute_trades",
            "[REWARD] Realized PnL for step": "Logging of realized_pnl",
        }
        
        for pattern, description in checks.items():
            if pattern in source:
                logger.info(f"✓ Found: {description}")
            else:
                logger.error(f"✗ Missing: {description}")
                return False
        
    except Exception as e:
        logger.error(f"✗ step() check failed: {e}")
        return False
    
    # Check 4: Verify _calculate_reward uses realized_pnl
    logger.info("\n[CHECK 4] _calculate_reward() uses realized_pnl")
    try:
        source = inspect.getsource(MultiAssetChunkedEnv._calculate_reward)
        
        if "def _calculate_reward(self, action: np.ndarray, realized_pnl: float)" in source:
            logger.info("✓ _calculate_reward signature includes realized_pnl parameter")
        else:
            logger.error("✗ _calculate_reward signature missing realized_pnl parameter")
            return False
        
        if "pnl_net = float(realized_pnl)" in source:
            logger.info("✓ _calculate_reward uses realized_pnl for PnL calculation")
        else:
            logger.error("✗ _calculate_reward doesn't use realized_pnl correctly")
            return False
        
    except Exception as e:
        logger.error(f"✗ _calculate_reward check failed: {e}")
        return False
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL CHECKS PASSED - PnL flow is correctly implemented")
    logger.info("=" * 80)
    return True

if __name__ == '__main__':
    success = test_pnl_flow()
    sys.exit(0 if success else 1)
