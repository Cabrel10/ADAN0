import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.adan_trading_bot.common.enhanced_config_manager import get_config_manager
from src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
from src.adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine

def analyze_trading_conditions():
    config_manager = get_config_manager(config_dir='config', enable_hot_reload=False, force_new=True)
    config = {
        'trading': config_manager.get_config('trading'),
        'dbe': config_manager.get_config('dbe'),
        'environment': config_manager.get_config('environment'),
        'agent': config_manager.get_config('agent'),
        'data': config_manager.get_config('data'),
        'paths': {'processed_data_dir': '/tmp/processed_data'} # Dummy path
    }
    
    print("=== ANALYSE DES CONDITIONS DE TRADING ===")
    print("\n1. CONFIGURATION DES PALIERS DE CAPITAL:")
    
    capital_tiers = config['trading']['capital_tiers']
    for tier in capital_tiers:
        print(f"\n📊 Palier: {tier['name']}")
        print(f"   Capital min/max: {tier['min_capital']}/{tier['max_capital']} USDT")
        print(f"   Taille position: {tier['max_position_size_pct']}%")
        print(f"   SL: {tier.get('stop_loss_pct', 'N/A')}")
        print(f"   TP: {tier.get('take_profit_pct', 'N/A')}")
    
    print("\n2. CONFIGURATION DES TIMEFRAMES:")
    timeframe_config = config['trading']['trading_rules']['frequency']
    for tf in ['5m', '1h', '4h']:
        limits = timeframe_config
        print(f"\n⏰ Timeframe: {tf}")
        print(f"   Positions max: {limits['max_positions'][tf]}")
        print(f"   Fréquence min: {limits['min_positions'][tf]}")
        print(f"   Durée max: {config['trading']['trading_rules']['duration_tracking'][tf]['max_duration_steps']} steps")
    
    print("\n3. TEST DES TRANSITIONS DE PALIERS:")
    # Test simulation de différents niveaux de capital
    test_capitals = [15, 25, 50, 150, 500, 1500]
    
    for capital in test_capitals:
        pm = PortfolioManager(config=config, worker_id=0)
        pm.initial_equity = capital
        pm.reset()
        current_tier = pm.get_current_tier()
        
        dbe = DynamicBehaviorEngine(config=config['dbe'], finance_manager=pm, worker_id=0)
        risk_params = dbe.compute_dynamic_modulation('bull', 0.1, 10)
        pm.update_risk_parameters(risk_params, tier=current_tier)

        print(f"\n💰 Capital: {capital} USDT → Palier: {current_tier['name']}")
        print(f"   Taille position: {pm.pos_size_pct*100:.2f}%")
        print(f"   SL/TP appliqués: SL={pm.sl_pct*100:.2f}%, TP={pm.tp_pct*100:.2f}%")

if __name__ == "__main__":
    analyze_trading_conditions()
