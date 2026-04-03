import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RiskManager:
    """Gestionnaire de risque avec protections avancées"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_daily_drawdown = config.get('max_daily_drawdown', 0.05)  # 5%
        self.max_position_risk = config.get('max_position_risk', 0.02)     # 2%
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.10)   # 10%
        
        # Sécurité numérique
        self.portfolio_peak = config.get('initial_capital', 10000.0)
        self.daily_peak = self.portfolio_peak
        
    def validate_trade(
        self, 
        portfolio_value: float,
        position_size: float, 
        entry_price: float,
        stop_loss: float
    ) -> bool:
        """
        Valide un trade avec multiples vérifications de sécurité
        """
        # Vérifications numériques de base
        if not all(np.isfinite(x) for x in [portfolio_value, position_size, entry_price, stop_loss]):
            logger.error("Trade validation: valeurs non finies")
            return False
            
        if portfolio_value <= 0:
            logger.error(f"Trade validation: capital invalide: {portfolio_value}")
            return False
            
        # Vérification risque position
        position_risk = abs(entry_price - stop_loss) * position_size
        position_risk_pct = position_risk / portfolio_value if portfolio_value > 0 else 0
        
        if position_risk_pct > self.max_position_risk:
            logger.warning(
                f"Risque position trop élevé: {position_risk_pct:.2%} > {self.max_position_risk:.2%}"
            )
            return False
            
        # Vérification drawdown quotidien
        current_drawdown = (self.daily_peak - portfolio_value) / self.daily_peak if self.daily_peak > 0 else 0
        if current_drawdown > self.max_daily_drawdown:
            logger.warning(f"Drawdown quotidien dépassé: {current_drawdown:.2%} > {self.max_daily_drawdown:.2%}")
            return False
            
        return True
        
    def update_peak(self, portfolio_value: float):
        """Met à jour les peaks avec sécurité"""
        if np.isfinite(portfolio_value):
            self.portfolio_peak = max(self.portfolio_peak, portfolio_value)
            self.daily_peak = max(self.daily_peak, portfolio_value)
