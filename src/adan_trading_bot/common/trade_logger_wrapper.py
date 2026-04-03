#!/usr/bin/env python3
"""
Wrapper pour logger automatiquement les trades via CentralLogger
Sans modifier portfolio_manager.py
"""
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from .central_logger import logger as central_logger
    CENTRAL_LOGGER_AVAILABLE = True
except ImportError:
    CENTRAL_LOGGER_AVAILABLE = False
    central_logger = None


class TradeLoggerWrapper:
    """Wrapper pour intercepter et logger les trades"""
    
    def __init__(self, portfolio_manager):
        self.pm = portfolio_manager
        self._original_open = portfolio_manager.open_position
        self._original_close = portfolio_manager.close_position
        
        # Remplacer les méthodes
        portfolio_manager.open_position = self._wrapped_open
        portfolio_manager.close_position = self._wrapped_close
    
    def _wrapped_open(self, asset: str, entry_price: float, size: float, 
                      stop_loss_pct: float, take_profit_pct: float, 
                      open_step: int, open_time: Optional[Any] = None,
                      timeframe: str = "5m", risk_horizon: float = 0.0) -> Dict[str, Any]:
        """Wrapper pour open_position avec logging"""
        
        # Appeler la méthode originale
        result = self._original_open(
            asset=asset,
            entry_price=entry_price,
            size=size,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            open_step=open_step,
            open_time=open_time,
            timeframe=timeframe,
            risk_horizon=risk_horizon
        )
        
        # Logger le trade
        if CENTRAL_LOGGER_AVAILABLE and central_logger:
            try:
                central_logger.trade(
                    action="BUY",
                    symbol=asset,
                    quantity=size,
                    price=entry_price,
                    pnl=None,
                    source="portfolio_manager"
                )
            except Exception as e:
                logger.debug(f"Erreur log trade ouverture: {e}")
        
        return result
    
    def _wrapped_close(self, asset: str, current_price: float, current_step: int,
                       reason: str = "TP", current_time: Optional[Any] = None) -> Dict[str, Any]:
        """Wrapper pour close_position avec logging"""
        
        # Appeler la méthode originale
        result = self._original_close(
            asset=asset,
            current_price=current_price,
            current_step=current_step,
            reason=reason,
            current_time=current_time
        )
        
        # Logger le trade de fermeture
        if CENTRAL_LOGGER_AVAILABLE and central_logger:
            try:
                pnl = result.get('realized_pnl', 0.0) if isinstance(result, dict) else None
                central_logger.trade(
                    action="SELL",
                    symbol=asset,
                    quantity=result.get('size', 0.0) if isinstance(result, dict) else 0.0,
                    price=current_price,
                    pnl=pnl,
                    source="portfolio_manager"
                )
            except Exception as e:
                logger.debug(f"Erreur log trade fermeture: {e}")
        
        return result


def enable_trade_logging(portfolio_manager):
    """Active le logging des trades pour un portfolio_manager"""
    try:
        TradeLoggerWrapper(portfolio_manager)
        logger.info("✅ Trade logging activé")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur activation trade logging: {e}")
        return False
