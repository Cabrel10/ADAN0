"""
Module de gestion des risques pour l'apprentissage continu ADAN.
Surveille et contrôle les risques pendant le trading en temps réel.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class SafetyManager:
    """
    Gestionnaire de sécurité pour l'apprentissage continu.
    Surveille les risques et peut arrêter l'apprentissage en cas de danger.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le gestionnaire de sécurité.
        
        Args:
            config: Configuration complète du système
        """
        self.config = config
        
        # Configuration des risques
        risk_config = config.get('risk_management', {})
        
        # Limites de capital
        self.max_daily_loss = risk_config.get('max_daily_loss', 100.0)
        self.max_position_value = risk_config.get('max_position_value', 50.0)
        self.max_total_exposure = risk_config.get('max_total_exposure', 200.0)
        self.emergency_stop_loss = risk_config.get('emergency_stop_loss', 0.15)
        
        # Limites d'apprentissage
        self.max_daily_trades = risk_config.get('max_daily_trades', 100)
        self.max_consecutive_losses = risk_config.get('max_consecutive_losses', 5)
        self.max_drawdown_percent = risk_config.get('max_drawdown_percent', 10.0)
        self.max_negative_reward_streak = risk_config.get('max_negative_reward_streak', 10)
        
        # Seuils d'alerte
        alert_config = risk_config.get('alert_thresholds', {})
        self.loss_alert_percent = alert_config.get('loss_percent', 5)
        self.unusual_activity_detection = alert_config.get('unusual_activity', True)
        
        # État du gestionnaire
        self.daily_trades_count = 0
        self.consecutive_losses = 0
        self.negative_reward_streak = 0
        self.daily_start_capital = None
        self.peak_capital = None
        self.trade_history = []
        self.reward_history = []
        self.alerts_raised = []
        
        # État d'arrêt d'urgence
        self.emergency_stop_active = False
        self.stop_learning_active = False
        self.last_reset_date = datetime.now().date()
        
        logger.info(f"✅ SafetyManager initialized")
        logger.info(f"📊 Max daily loss: ${self.max_daily_loss}")
        logger.info(f"📊 Max position value: ${self.max_position_value}")
        logger.info(f"📊 Emergency stop loss: {self.emergency_stop_loss:.1%}")
    
    def check_safety_conditions(self, 
                               proposed_action: Dict[str, Any], 
                               current_state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Vérifie les conditions de sécurité avant l'exécution d'une action.
        
        Args:
            proposed_action: Action proposée par l'agent
            current_state: État actuel du système (capital, positions, etc.)
            
        Returns:
            Tuple[bool, str]: (allowed, reason)
        """
        try:
            # Reset quotidien si nécessaire
            self._check_daily_reset()
            
            # Vérifier l'arrêt d'urgence
            if self.emergency_stop_active:
                return False, "Emergency stop is active"
            
            # Vérifier l'arrêt d'apprentissage
            if self.stop_learning_active:
                return False, "Learning stop is active"
            
            # Initialiser le capital de référence si nécessaire
            current_capital = current_state.get('capital', 0)
            if self.daily_start_capital is None:
                self.daily_start_capital = current_capital
                self.peak_capital = current_capital
            
            # Mettre à jour le pic de capital
            if current_capital > self.peak_capital:
                self.peak_capital = current_capital
            
            # 1. Vérifier le nombre de trades quotidiens
            if self.daily_trades_count >= self.max_daily_trades:
                self._raise_alert("DAILY_TRADE_LIMIT", f"Daily trade limit reached: {self.daily_trades_count}")
                return False, f"Daily trade limit reached ({self.max_daily_trades})"
            
            # 2. Vérifier la perte quotidienne
            daily_loss = self.daily_start_capital - current_capital
            if daily_loss > self.max_daily_loss:
                self._raise_alert("DAILY_LOSS_LIMIT", f"Daily loss limit exceeded: ${daily_loss:.2f}")
                self._activate_emergency_stop("Daily loss limit exceeded")
                return False, f"Daily loss limit exceeded: ${daily_loss:.2f}"
            
            # 3. Vérifier le stop loss d'urgence (drawdown depuis le pic)
            drawdown = (self.peak_capital - current_capital) / self.peak_capital if self.peak_capital > 0 else 0
            if drawdown > self.emergency_stop_loss:
                self._raise_alert("EMERGENCY_STOP_LOSS", f"Emergency stop loss triggered: {drawdown:.1%}")
                self._activate_emergency_stop("Emergency stop loss triggered")
                return False, f"Emergency stop loss triggered: {drawdown:.1%}"
            
            # 4. Vérifier la taille de position
            if proposed_action.get('amount', 0) > self.max_position_value:
                self._raise_alert("POSITION_SIZE_LIMIT", f"Position size too large: ${proposed_action.get('amount', 0):.2f}")
                return False, f"Position size too large: ${proposed_action.get('amount', 0):.2f}"
            
            # 5. Vérifier l'exposition totale
            total_exposure = self._calculate_total_exposure(current_state)
            if total_exposure > self.max_total_exposure:
                self._raise_alert("TOTAL_EXPOSURE_LIMIT", f"Total exposure too high: ${total_exposure:.2f}")
                return False, f"Total exposure too high: ${total_exposure:.2f}"
            
            # 6. Vérifier les pertes consécutives
            if self.consecutive_losses >= self.max_consecutive_losses:
                self._raise_alert("CONSECUTIVE_LOSSES", f"Too many consecutive losses: {self.consecutive_losses}")
                self._activate_learning_stop("Too many consecutive losses")
                return False, f"Too many consecutive losses: {self.consecutive_losses}"
            
            # 7. Vérifier le streak de récompenses négatives
            if self.negative_reward_streak >= self.max_negative_reward_streak:
                self._raise_alert("NEGATIVE_REWARD_STREAK", f"Too many negative rewards: {self.negative_reward_streak}")
                self._activate_learning_stop("Too many negative rewards")
                return False, f"Too many negative rewards: {self.negative_reward_streak}"
            
            # 8. Alertes non bloquantes
            self._check_alert_conditions(current_state, daily_loss, drawdown)
            
            return True, "All safety conditions passed"
            
        except Exception as e:
            logger.error(f"❌ Error checking safety conditions: {e}")
            return False, f"Safety check error: {str(e)}"
    
    def record_trade_result(self, trade_result: Dict[str, Any], reward: float):
        """
        Enregistre le résultat d'un trade pour le suivi des risques.
        
        Args:
            trade_result: Résultat du trade
            reward: Récompense associée
        """
        try:
            self.daily_trades_count += 1
            
            # Enregistrer dans l'historique
            trade_record = {
                'timestamp': time.time(),
                'result': trade_result,
                'reward': reward,
                'is_profitable': trade_result.get('pnl', 0) > 0 if 'pnl' in trade_result else reward > 0
            }
            
            self.trade_history.append(trade_record)
            self.reward_history.append(reward)
            
            # Limiter la taille des historiques
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
            if len(self.reward_history) > 1000:
                self.reward_history = self.reward_history[-1000:]
            
            # Mettre à jour les compteurs de streak
            if trade_record['is_profitable']:
                self.consecutive_losses = 0  # Reset des pertes consécutives
            else:
                self.consecutive_losses += 1
            
            if reward < 0:
                self.negative_reward_streak += 1
            else:
                self.negative_reward_streak = 0  # Reset du streak négatif
            
            # Log périodique
            if self.daily_trades_count % 10 == 0:
                self._log_safety_status()
            
        except Exception as e:
            logger.error(f"❌ Error recording trade result: {e}")
    
    def _check_daily_reset(self):
        """Vérifie et effectue le reset quotidien si nécessaire."""
        current_date = datetime.now().date()
        
        if current_date > self.last_reset_date:
            logger.info(f"🔄 Daily reset - New trading day: {current_date}")
            
            # Reset des compteurs quotidiens
            self.daily_trades_count = 0
            self.daily_start_capital = None
            self.consecutive_losses = 0
            self.negative_reward_streak = 0
            self.last_reset_date = current_date
            
            # Désactiver l'arrêt d'apprentissage (mais pas l'arrêt d'urgence)
            if self.stop_learning_active:
                logger.info("🔄 Learning stop reset for new day")
                self.stop_learning_active = False
            
            # Nettoyer les anciennes alertes
            self.alerts_raised = [alert for alert in self.alerts_raised 
                                if alert['timestamp'] > time.time() - 86400]  # Garder 24h
    
    def _calculate_total_exposure(self, current_state: Dict[str, Any]) -> float:
        """Calcule l'exposition totale actuelle."""
        try:
            positions = current_state.get('positions', {})
            total_exposure = 0.0
            
            for asset_id, position in positions.items():
                # Estimer la valeur de la position
                # Pour simplifier, on utilise la quantité * prix (si disponible)
                qty = position.get('qty', 0)
                price = position.get('price', 0)
                position_value = abs(qty * price)
                total_exposure += position_value
            
            return total_exposure
            
        except Exception as e:
            logger.error(f"❌ Error calculating total exposure: {e}")
            return 0.0
    
    def _check_alert_conditions(self, current_state: Dict[str, Any], daily_loss: float, drawdown: float):
        """Vérifie les conditions d'alerte non bloquantes."""
        try:
            # Alerte de perte
            if daily_loss > 0 and (daily_loss / self.daily_start_capital) * 100 > self.loss_alert_percent:
                self._raise_alert("LOSS_WARNING", f"Daily loss warning: {daily_loss:.2f} ({(daily_loss/self.daily_start_capital)*100:.1f}%)")
            
            # Alerte de drawdown
            if drawdown > 0.05:  # 5% de drawdown
                self._raise_alert("DRAWDOWN_WARNING", f"Drawdown warning: {drawdown:.1%}")
            
            # Détection d'activité inhabituelle
            if self.unusual_activity_detection and len(self.trade_history) >= 10:
                recent_trades = self.trade_history[-10:]
                recent_time_span = recent_trades[-1]['timestamp'] - recent_trades[0]['timestamp']
                
                if recent_time_span < 300:  # 10 trades en moins de 5 minutes
                    self._raise_alert("UNUSUAL_ACTIVITY", "High frequency trading detected")
            
        except Exception as e:
            logger.error(f"❌ Error checking alert conditions: {e}")
    
    def _raise_alert(self, alert_type: str, message: str):
        """Lève une alerte de sécurité."""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'message': message,
            'datetime': datetime.now().isoformat()
        }
        
        self.alerts_raised.append(alert)
        
        logger.warning(f"🚨 SAFETY ALERT [{alert_type}]: {message}")
        
        # Limiter le nombre d'alertes stockées
        if len(self.alerts_raised) > 100:
            self.alerts_raised = self.alerts_raised[-100:]
    
    def _activate_emergency_stop(self, reason: str):
        """Active l'arrêt d'urgence."""
        self.emergency_stop_active = True
        self._raise_alert("EMERGENCY_STOP", f"Emergency stop activated: {reason}")
        logger.critical(f"🚨 EMERGENCY STOP ACTIVATED: {reason}")
    
    def _activate_learning_stop(self, reason: str):
        """Active l'arrêt d'apprentissage (moins sévère que l'arrêt d'urgence)."""
        self.stop_learning_active = True
        self._raise_alert("LEARNING_STOP", f"Learning stop activated: {reason}")
        logger.warning(f"⚠️ LEARNING STOP ACTIVATED: {reason}")
    
    def _log_safety_status(self):
        """Log le statut de sécurité actuel."""
        try:
            current_capital = self.peak_capital or 0
            daily_loss = (self.daily_start_capital - current_capital) if self.daily_start_capital else 0
            drawdown = ((self.peak_capital - current_capital) / self.peak_capital) if self.peak_capital else 0
            
            logger.info(f"🛡️ Safety Status:")
            logger.info(f"   📊 Daily trades: {self.daily_trades_count}/{self.max_daily_trades}")
            logger.info(f"   💰 Daily loss: ${daily_loss:.2f}/${self.max_daily_loss:.2f}")
            logger.info(f"   📉 Drawdown: {drawdown:.1%}/{self.emergency_stop_loss:.1%}")
            logger.info(f"   🔴 Consecutive losses: {self.consecutive_losses}/{self.max_consecutive_losses}")
            logger.info(f"   ⬇️ Negative streak: {self.negative_reward_streak}/{self.max_negative_reward_streak}")
            logger.info(f"   🚨 Emergency stop: {'ACTIVE' if self.emergency_stop_active else 'INACTIVE'}")
            logger.info(f"   ⚠️ Learning stop: {'ACTIVE' if self.stop_learning_active else 'INACTIVE'}")
            
        except Exception as e:
            logger.error(f"❌ Error logging safety status: {e}")
    
    def get_safety_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'état de sécurité."""
        try:
            current_capital = self.peak_capital or 0
            daily_loss = (self.daily_start_capital - current_capital) if self.daily_start_capital else 0
            drawdown = ((self.peak_capital - current_capital) / self.peak_capital) if self.peak_capital else 0
            
            recent_alerts = [alert for alert in self.alerts_raised 
                           if alert['timestamp'] > time.time() - 3600]  # Dernière heure
            
            return {
                'emergency_stop_active': self.emergency_stop_active,
                'learning_stop_active': self.stop_learning_active,
                'daily_trades_count': self.daily_trades_count,
                'max_daily_trades': self.max_daily_trades,
                'daily_loss': daily_loss,
                'max_daily_loss': self.max_daily_loss,
                'current_drawdown': drawdown,
                'emergency_stop_threshold': self.emergency_stop_loss,
                'consecutive_losses': self.consecutive_losses,
                'max_consecutive_losses': self.max_consecutive_losses,
                'negative_reward_streak': self.negative_reward_streak,
                'max_negative_reward_streak': self.max_negative_reward_streak,
                'recent_alerts_count': len(recent_alerts),
                'total_alerts_today': len([a for a in self.alerts_raised 
                                         if datetime.fromtimestamp(a['timestamp']).date() == datetime.now().date()]),
                'recent_alerts': recent_alerts[-5:],  # 5 dernières alertes
                'peak_capital': self.peak_capital,
                'daily_start_capital': self.daily_start_capital
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting safety summary: {e}")
            return {}
    
    def reset_emergency_stop(self, reason: str = "Manual reset"):
        """Remet à zéro l'arrêt d'urgence (à utiliser avec précaution)."""
        if self.emergency_stop_active:
            self.emergency_stop_active = False
            self._raise_alert("EMERGENCY_STOP_RESET", f"Emergency stop reset: {reason}")
            logger.warning(f"🔄 Emergency stop reset: {reason}")
    
    def reset_learning_stop(self, reason: str = "Manual reset"):
        """Remet à zéro l'arrêt d'apprentissage."""
        if self.stop_learning_active:
            self.stop_learning_active = False
            self._raise_alert("LEARNING_STOP_RESET", f"Learning stop reset: {reason}")
            logger.info(f"🔄 Learning stop reset: {reason}")
    
    def is_trading_allowed(self) -> bool:
        """Vérifie si le trading est autorisé."""
        return not self.emergency_stop_active
    
    def is_learning_allowed(self) -> bool:
        """Vérifie si l'apprentissage est autorisé."""
        return not self.emergency_stop_active and not self.stop_learning_active