
import yaml
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.adan_trading_bot.common.enhanced_config_manager import get_config_manager

def validate_trading_config():
    """Validation complète de la configuration de trading"""
    
    config_manager = get_config_manager(config_dir='config', enable_hot_reload=False)
    trading_config = config_manager.get_config('trading')
    
    errors = []
    warnings = []
    
    if not trading_config:
        errors.append("Le fichier de configuration 'trading.yaml' est manquant ou vide.")
    else:
        # 1. Validation des paliers de capital
        if 'capital_tiers' in trading_config:
            capital_tiers = trading_config['capital_tiers']
            previous_max = 0
            for tier_config in capital_tiers:
                current_max = tier_config.get('max_capital')
                if current_max is None:
                    current_max = float('inf')
                
                if current_max <= previous_max:
                    errors.append(f"Palier {tier_config['name']}: capital max ({current_max}) doit être > au palier précédent ({previous_max})")
                
                if tier_config['max_position_size_pct'] > 100:
                    errors.append(f"Palier {tier_config['name']}: taille position ({tier_config['max_position_size_pct']}%) > 100%")
                
                previous_max = current_max
        else:
            errors.append("La section 'capital_tiers' est manquante dans trading.yaml")

        # 2. Validation des timeframes
        if 'trading_rules' in trading_config and 'frequency' in trading_config['trading_rules']:
            timeframe_limits = trading_config['trading_rules']['frequency']
            if 'min_positions' in timeframe_limits and 'max_positions' in timeframe_limits:
                for tf in timeframe_limits['min_positions']:
                    if tf in timeframe_limits['max_positions']:
                        if timeframe_limits['max_positions'][tf] < timeframe_limits['min_positions'][tf]:
                            warnings.append(f"Timeframe {tf}: max_positions ({timeframe_limits['max_positions'][tf]}) < min_positions ({timeframe_limits['min_positions'][tf]})")
        else:
            warnings.append("La section 'frequency' ou 'trading_rules' est manquante dans trading.yaml")

    # 3. Affichage des résultats
    print("=== VALIDATION CONFIGURATION TRADING ===\n")
    
    if errors:
        print("❌ ERREURS CRITIQUES:")
        for error in errors:
            print(f"   - {error}")
        print()
    
    if warnings:
        print("⚠️  AVERTISSEMENTS:")
        for warning in warnings:
            print(f"   - {warning}")
        print()
    
    if not errors and not warnings:
        print("✅ CONFIGURATION VALIDE - Tous les paramètres sont correctement alignés")
    
    return len(errors) == 0

if __name__ == "__main__":
    validate_trading_config()
