#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Translates agent actions into trade orders."""
import numpy as np
from ..common.utils import get_logger

logger = get_logger()

class ActionTranslator:
    def __init__(self, assets: list[str]):
        self.assets = assets

    def translate_action(self, action: np.ndarray, portfolio_manager, current_prices: dict[str, float]) -> list[dict]:
        """Translates a raw action from the agent into a list of trade orders.
        
        Args:
            action: Array of action values between -1 and 1 for each asset
            portfolio_manager: PortfolioManager instance
            current_prices: Dictionary of current prices for each asset
            
        Returns:
            List of trade orders, each as a dictionary with 'asset', 'units', and 'price'
        """
        orders = []
        
        # Vérification des dimensions de l'action
        if len(action) != len(self.assets):
            raise ValueError(f"Action dimensions ({len(action)}) do not match number of assets ({len(self.assets)})")
            
        # Vérification des prix manquants
        for asset in self.assets:
            if asset not in current_prices:
                raise KeyError(f"Missing price for asset {asset}")
        
        tier = portfolio_manager.get_current_tier()
        position_size_pct = tier.get('max_position_size_pct', 0.5)
        
        # Utiliser le capital total pour le calcul de la position
        total_capital = portfolio_manager.initial_capital
        
        for i, asset in enumerate(self.assets):            
            action_value = action[i]
            current_price = current_prices[asset]  # On sait que le prix existe grâce à la vérification plus haut
            
            # Buy signal
            if action_value > 0.5:  
                # Calculer la taille de la position en fonction du capital total
                position_value = total_capital * position_size_pct
                units = position_value / current_price
                
                # Vérifier les tailles minimales
                min_trade_size = portfolio_manager.config.get('trading_rules', {}).get('min_trade_size', 0.0001)
                min_notional = portfolio_manager.config.get('trading_rules', {}).get('min_notional_value', 10.0)
                
                if units >= min_trade_size and (units * current_price) >= min_notional:
                    orders.append({
                        'asset': asset, 
                        'units': units, 
                        'price': current_price,
                        'action': 'buy'
                    })
            
            # Sell signal
            elif action_value < -0.5:
                # Vérifier si nous avons une position à vendre
                position = portfolio_manager.positions.get(asset)
                if position and position.is_open:
                    # Vendre un pourcentage de la position basé sur la force du signal
                    sell_pct = abs(action_value)  # Entre 0.5 et 1.0
                    units_to_sell = position.size * sell_pct
                    
                    orders.append({
                        'asset': asset, 
                        'units': -units_to_sell,  # Négatif pour une vente
                        'price': current_price,
                        'action': 'sell'
                    })
        
        return orders
