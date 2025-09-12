"""
Système d'optimisation continue pour le Dynamic Behavior Engine.
Implémente la tâche 9.2.2 - Optimisation continue DBE.
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque
import threading
import time
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from .adaptive_dbe import AdaptiveDBE, DBEParameters, MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class OptimizationMetrics:
    """Métriques pour l'optimisation continue"""
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 1.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    def get_composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calcule un score composite pondéré"""
        if weights is None:
            weights = {
                'sharpe_ratio': 0.3,
                'total_return': 0.2,
                'max_drawdown': -0.2,  # Négatif car on veut minimiser
                'win_rate': 0.15,
                'calmar_ratio': 0.15
            }
        
        score = 0.0
        for metric, weight in weights.items():
            if hasattr(self, metric):
                value = getattr(self, metric)
                score += weight * value
        
        return score


class PerformanceFeedbackLoop:
    """Boucle de rétroaction pour l'optimisation continue"""
    
    def __init__(self, window_size: int = 100, optimization_frequency: int = 50):
        self.window_size = window_size
        self.optimization_frequency = optimization_frequency
        
        # Historique des performances
        self.performance_history = deque(maxlen=window_size)
        self.parameter_history = deque(maxlen=window_size)
        self.timestamp_history = deque(maxlen=window_size)
        
        # Modèle de prédiction (Gaussian Process)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        self.model_trained = False
        
        # Statistiques d'optimisation
        self.optimization_count = 0
        self.last_optimization_time = time.time()
        self.optimization_improvements = []
        
    def add_performance_sample(self, params: DBEParameters, 
                             metrics: OptimizationMetrics) -> None:
        """Ajoute un échantillon de performance"""
        self.performance_history.append(metrics)
        self.parameter_history.append(params.to_dict())
        self.timestamp_history.append(time.time())
        
        # Réentraîner le modèle périodiquement
        if len(self.performance_history) >= 10 and len(self.performance_history) % 10 == 0:
            self._retrain_model()
    
    def _retrain_model(self) -> None:
        """Réentraîne le modèle de prédiction"""
        if len(self.performance_history) < 5:
            return
        
        try:
            # Préparer les données d'entraînement
            X = []
            y = []
            
            for params, metrics in zip(self.parameter_history, self.performance_history):
                # Utiliser les paramètres clés comme features
                features = [
                    params['risk_threshold_low'],
                    params['risk_threshold_medium'],
                    params['risk_threshold_high'],
                    params['volatility_threshold_low'],
                    params['volatility_threshold_high'],
                    params['max_drawdown_threshold'],
                    params['learning_rate']
                ]
                X.append(features)
                y.append(metrics.get_composite_score())
            
            X = np.array(X)
            y = np.array(y)
            
            # Entraîner le modèle
            self.gp_model.fit(X, y)
            self.model_trained = True
            
            logger.debug(f"GP model retrained with {len(X)} samples")
            
        except Exception as e:
            logger.warning(f"Failed to retrain GP model: {e}")
    
    def suggest_parameter_optimization(self, current_params: DBEParameters) -> Optional[DBEParameters]:
        """Suggère une optimisation des paramètres"""
        if not self.model_trained or len(self.performance_history) < self.optimization_frequency:
            return None
        
        try:
            # Définir l'espace de recherche
            param_bounds = [
                (0.1, 0.5),   # risk_threshold_low
                (0.3, 0.8),   # risk_threshold_medium
                (0.6, 0.95),  # risk_threshold_high
                (0.005, 0.02), # volatility_threshold_low
                (0.02, 0.1),  # volatility_threshold_high
                (0.05, 0.3),  # max_drawdown_threshold
                (0.001, 0.1)  # learning_rate
            ]
            
            # Fonction objectif
            def objective(x):
                try:
                    # Prédire la performance avec ces paramètres
                    pred_score, _ = self.gp_model.predict([x], return_std=True)
                    return -pred_score[0]  # Minimiser (donc négatif)
                except:
                    return 1000  # Pénalité en cas d'erreur
            
            # Optimisation
            current_features = [
                current_params.risk_threshold_low,
                current_params.risk_threshold_medium,
                current_params.risk_threshold_high,
                current_params.volatility_threshold_low,
                current_params.volatility_threshold_high,
                current_params.max_drawdown_threshold,
                current_params.learning_rate
            ]
            
            result = minimize(objective, current_features, bounds=param_bounds, method='L-BFGS-B')
            
            if result.success:
                # Créer nouveaux paramètres
                optimized_params = current_params.to_dict()
                param_names = [
                    'risk_threshold_low', 'risk_threshold_medium', 'risk_threshold_high',
                    'volatility_threshold_low', 'volatility_threshold_high',
                    'max_drawdown_threshold', 'learning_rate'
                ]
                
                for i, param_name in enumerate(param_names):
                    optimized_params[param_name] = result.x[i]
                
                self.optimization_count += 1
                self.last_optimization_time = time.time()
                
                # Calculer l'amélioration prédite
                current_score = self.gp_model.predict([current_features])[0]
                optimized_score = self.gp_model.predict([result.x])[0]
                improvement = optimized_score - current_score
                
                self.optimization_improvements.append(improvement)
                
                logger.info(f"Parameter optimization suggested improvement: {improvement:.4f}")
                
                return DBEParameters.from_dict(optimized_params)
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
        
        return None
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'optimisation"""
        return {
            'optimization_count': self.optimization_count,
            'last_optimization_time': self.last_optimization_time,
            'avg_improvement': np.mean(self.optimization_improvements) if self.optimization_improvements else 0.0,
            'total_samples': len(self.performance_history),
            'model_trained': self.model_trained,
            'recent_performance_trend': self._calculate_performance_trend()
        }
    
    def _calculate_performance_trend(self) -> float:
        """Calcule la tendance de performance récente"""
        if len(self.performance_history) < 10:
            return 0.0
        
        recent_scores = [metrics.get_composite_score() for metrics in list(self.performance_history)[-10:]]
        return np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]


class AutoAdjustmentSystem:
    """Système d'auto-ajustement en temps réel"""
    
    def __init__(self, adjustment_sensitivity: float = 0.1):
        self.adjustment_sensitivity = adjustment_sensitivity
        self.adjustment_history = deque(maxlen=1000)
        self.performance_thresholds = {
            'excellent': 0.8,
            'good': 0.4,
            'poor': 0.0,
            'critical': -0.5
        }
        
    def auto_adjust_parameters(self, current_params: DBEParameters,
                             recent_performance: OptimizationMetrics,
                             market_conditions: Dict[str, float]) -> DBEParameters:
        """Ajuste automatiquement les paramètres basé sur la performance récente"""
        
        adjusted_params = current_params.to_dict()
        adjustments_made = []
        
        performance_score = recent_performance.get_composite_score()
        
        # Ajustements basés sur la performance
        if performance_score < self.performance_thresholds['critical']:
            # Performance critique - ajustements drastiques
            adjusted_params['risk_threshold_low'] *= 0.8
            adjusted_params['risk_threshold_medium'] *= 0.8
            adjusted_params['risk_threshold_high'] *= 0.8
            adjusted_params['max_drawdown_threshold'] *= 0.7
            adjustments_made.append('critical_risk_reduction')
            
        elif performance_score < self.performance_thresholds['poor']:
            # Performance faible - ajustements conservateurs
            adjusted_params['risk_threshold_low'] *= 0.9
            adjusted_params['risk_threshold_medium'] *= 0.9
            adjusted_params['max_drawdown_threshold'] *= 0.85
            adjustments_made.append('conservative_adjustment')
            
        elif performance_score > self.performance_thresholds['excellent']:
            # Excellente performance - être plus agressif
            adjusted_params['risk_threshold_low'] *= 1.05
            adjusted_params['risk_threshold_medium'] *= 1.05
            adjusted_params['max_drawdown_threshold'] *= 1.1
            adjustments_made.append('aggressive_adjustment')
        
        # Ajustements basés sur les conditions de marché
        volatility = market_conditions.get('volatility', 0.02)
        drawdown = market_conditions.get('current_drawdown', 0.0)
        
        if volatility > 0.05:  # Haute volatilité
            adjusted_params['volatility_threshold_high'] = min(0.1, volatility * 1.2)
            adjusted_params['learning_rate'] *= 0.8  # Apprentissage plus lent
            adjustments_made.append('high_volatility_adjustment')
            
        if drawdown > 0.1:  # Drawdown élevé
            adjusted_params['max_drawdown_threshold'] = max(0.05, drawdown * 0.8)
            adjusted_params['risk_threshold_high'] *= 0.85
            adjustments_made.append('high_drawdown_adjustment')
        
        # Appliquer les contraintes
        adjusted_params = self._apply_parameter_constraints(adjusted_params)
        
        # Enregistrer l'ajustement
        if adjustments_made:
            self.adjustment_history.append({
                'timestamp': time.time(),
                'performance_score': performance_score,
                'adjustments': adjustments_made,
                'market_conditions': market_conditions.copy()
            })
            
            logger.info(f"Auto-adjustments made: {adjustments_made}")
        
        return DBEParameters.from_dict(adjusted_params)
    
    def _apply_parameter_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Applique les contraintes sur les paramètres"""
        constraints = {
            'risk_threshold_low': (0.1, 0.5),
            'risk_threshold_medium': (0.3, 0.8),
            'risk_threshold_high': (0.6, 0.95),
            'volatility_threshold_low': (0.005, 0.02),
            'volatility_threshold_high': (0.02, 0.1),
            'max_drawdown_threshold': (0.05, 0.3),
            'learning_rate': (0.001, 0.1)
        }
        
        for param_name, (min_val, max_val) in constraints.items():
            if param_name in params:
                params[param_name] = np.clip(params[param_name], min_val, max_val)
        
        return params
    
    def get_adjustment_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'ajustement"""
        if not self.adjustment_history:
            return {'total_adjustments': 0}
        
        adjustment_types = {}
        for record in self.adjustment_history:
            for adj_type in record['adjustments']:
                adjustment_types[adj_type] = adjustment_types.get(adj_type, 0) + 1
        
        return {
            'total_adjustments': len(self.adjustment_history),
            'adjustment_types': adjustment_types,
            'recent_adjustments': list(self.adjustment_history)[-5:],
            'adjustment_frequency': len(self.adjustment_history) / max(1, (time.time() - self.adjustment_history[0]['timestamp']) / 3600)
        }


class ContinuousOptimizationDBE:
    """DBE avec optimisation continue et auto-ajustement"""
    
    def __init__(self, initial_params: Optional[DBEParameters] = None,
                 optimization_enabled: bool = True,
                 auto_adjustment_enabled: bool = True,
                 save_path: str = "logs/continuous_dbe"):
        
        # Composants principaux
        self.adaptive_dbe = AdaptiveDBE(initial_params, adaptation_enabled=True, save_path=save_path)
        self.feedback_loop = PerformanceFeedbackLoop()
        self.auto_adjustment = AutoAdjustmentSystem()
        
        # Configuration
        self.optimization_enabled = optimization_enabled
        self.auto_adjustment_enabled = auto_adjustment_enabled
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Métriques et historique
        self.performance_tracker = deque(maxlen=1000)
        self.optimization_log = []
        self.last_optimization_check = time.time()
        self.optimization_interval = 300  # 5 minutes
        
        # Threading pour optimisation asynchrone
        self.optimization_lock = threading.Lock()
        self.optimization_thread = None
        self.stop_optimization = False
        
        logger.info("ContinuousOptimizationDBE initialized")
    
    def update(self, market_data: Dict[str, Any], 
               performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Met à jour le système avec optimisation continue.
        
        Args:
            market_data: Données de marché actuelles
            performance_metrics: Métriques de performance
            
        Returns:
            Modulation DBE optimisée
        """
        # Convertir les métriques
        opt_metrics = OptimizationMetrics(
            sharpe_ratio=performance_metrics.get('sharpe_ratio', 0.0),
            total_return=performance_metrics.get('total_return', 0.0),
            max_drawdown=performance_metrics.get('max_drawdown', 0.0),
            volatility=performance_metrics.get('volatility', 0.02),
            win_rate=performance_metrics.get('win_rate', 0.5),
            profit_factor=performance_metrics.get('profit_factor', 1.0)
        )
        
        # Calculer ratios supplémentaires
        if opt_metrics.max_drawdown > 0:
            opt_metrics.calmar_ratio = opt_metrics.total_return / opt_metrics.max_drawdown
        if opt_metrics.volatility > 0:
            opt_metrics.sortino_ratio = opt_metrics.total_return / opt_metrics.volatility
        
        # Ajouter à l'historique de performance
        self.performance_tracker.append(opt_metrics)
        self.feedback_loop.add_performance_sample(self.adaptive_dbe.params, opt_metrics)
        
        # Auto-ajustement en temps réel si activé
        if self.auto_adjustment_enabled:
            market_conditions = self.adaptive_dbe.regime_detector.get_market_conditions()
            adjusted_params = self.auto_adjustment.auto_adjust_parameters(
                self.adaptive_dbe.params, opt_metrics, market_conditions
            )
            self.adaptive_dbe.params = adjusted_params
        
        # Mise à jour du DBE adaptatif
        modulation = self.adaptive_dbe.update(market_data, performance_metrics)
        
        # Vérification d'optimisation périodique
        current_time = time.time()
        if (self.optimization_enabled and 
            current_time - self.last_optimization_check > self.optimization_interval):
            self._trigger_optimization_check()
            self.last_optimization_check = current_time
        
        # Enrichir la modulation avec les informations d'optimisation
        modulation.update({
            'optimization_enabled': self.optimization_enabled,
            'auto_adjustment_enabled': self.auto_adjustment_enabled,
            'performance_score': opt_metrics.get_composite_score(),
            'optimization_stats': self.feedback_loop.get_optimization_stats(),
            'adjustment_stats': self.auto_adjustment.get_adjustment_stats()
        })
        
        return modulation
    
    def _trigger_optimization_check(self) -> None:
        """Déclenche une vérification d'optimisation"""
        if self.optimization_thread and self.optimization_thread.is_alive():
            return  # Optimisation déjà en cours
        
        self.optimization_thread = threading.Thread(target=self._run_optimization_check)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
    
    def _run_optimization_check(self) -> None:
        """Exécute la vérification d'optimisation en arrière-plan"""
        with self.optimization_lock:
            try:
                # Suggérer une optimisation
                optimized_params = self.feedback_loop.suggest_parameter_optimization(
                    self.adaptive_dbe.params
                )
                
                if optimized_params:
                    # Évaluer si l'optimisation est bénéfique
                    current_score = self.performance_tracker[-1].get_composite_score() if self.performance_tracker else 0.0
                    
                    # Log de l'optimisation
                    self.optimization_log.append({
                        'timestamp': time.time(),
                        'current_params': self.adaptive_dbe.params.to_dict(),
                        'optimized_params': optimized_params.to_dict(),
                        'current_score': current_score,
                        'optimization_triggered': True
                    })
                    
                    # Appliquer l'optimisation (avec prudence)
                    if len(self.performance_tracker) > 50:  # Assez d'historique
                        self.adaptive_dbe.params = optimized_params
                        logger.info("Parameters optimized based on performance feedback")
                
            except Exception as e:
                logger.error(f"Optimization check failed: {e}")
    
    def force_optimization(self) -> bool:
        """Force une optimisation immédiate"""
        try:
            optimized_params = self.feedback_loop.suggest_parameter_optimization(
                self.adaptive_dbe.params
            )
            
            if optimized_params:
                old_params = self.adaptive_dbe.params.to_dict()
                self.adaptive_dbe.params = optimized_params
                
                logger.info("Forced parameter optimization completed")
                
                # Log de l'optimisation forcée
                self.optimization_log.append({
                    'timestamp': time.time(),
                    'type': 'forced_optimization',
                    'old_params': old_params,
                    'new_params': optimized_params.to_dict()
                })
                
                return True
            
        except Exception as e:
            logger.error(f"Forced optimization failed: {e}")
        
        return False
    
    def get_comprehensive_performance_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance complet"""
        if not self.performance_tracker:
            return {'error': 'No performance data available'}
        
        # Statistiques de performance
        recent_performances = list(self.performance_tracker)[-50:]  # 50 dernières
        
        performance_stats = {
            'avg_sharpe_ratio': np.mean([p.sharpe_ratio for p in recent_performances]),
            'avg_total_return': np.mean([p.total_return for p in recent_performances]),
            'avg_max_drawdown': np.mean([p.max_drawdown for p in recent_performances]),
            'avg_win_rate': np.mean([p.win_rate for p in recent_performances]),
            'performance_volatility': np.std([p.get_composite_score() for p in recent_performances]),
            'performance_trend': np.polyfit(range(len(recent_performances)), 
                                          [p.get_composite_score() for p in recent_performances], 1)[0]
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_parameters': self.adaptive_dbe.params.to_dict(),
            'performance_stats': performance_stats,
            'optimization_stats': self.feedback_loop.get_optimization_stats(),
            'adjustment_stats': self.auto_adjustment.get_adjustment_stats(),
            'adaptive_dbe_stats': self.adaptive_dbe.get_comprehensive_stats(),
            'total_performance_samples': len(self.performance_tracker),
            'optimization_log_size': len(self.optimization_log),
            'system_health': self._assess_system_health()
        }
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Évalue la santé du système d'optimisation"""
        health = {
            'status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # Vérifier la performance récente
        if len(self.performance_tracker) > 10:
            recent_scores = [p.get_composite_score() for p in list(self.performance_tracker)[-10:]]
            avg_recent_score = np.mean(recent_scores)
            
            if avg_recent_score < -0.5:
                health['status'] = 'critical'
                health['issues'].append('Poor recent performance')
                health['recommendations'].append('Consider parameter reset or manual intervention')
            elif avg_recent_score < 0.0:
                health['status'] = 'warning'
                health['issues'].append('Below-average performance')
                health['recommendations'].append('Monitor closely and consider optimization')
        
        # Vérifier la fréquence d'optimisation
        opt_stats = self.feedback_loop.get_optimization_stats()
        if opt_stats['optimization_count'] == 0 and len(self.performance_tracker) > 100:
            health['issues'].append('No optimizations performed despite sufficient data')
            health['recommendations'].append('Check optimization system functionality')
        
        return health
    
    def save_optimization_state(self) -> str:
        """Sauvegarde l'état complet du système d'optimisation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        state = {
            'timestamp': timestamp,
            'comprehensive_report': self.get_comprehensive_performance_report(),
            'recent_optimization_log': self.optimization_log[-20:],  # 20 dernières optimisations
            'configuration': {
                'optimization_enabled': self.optimization_enabled,
                'auto_adjustment_enabled': self.auto_adjustment_enabled,
                'optimization_interval': self.optimization_interval
            }
        }
        
        filepath = self.save_path / f"continuous_optimization_state_{timestamp}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"Continuous optimization state saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save optimization state: {e}")
            return ""
    
    def shutdown(self) -> None:
        """Arrêt propre du système d'optimisation"""
        self.stop_optimization = True
        
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5.0)
        
        # Sauvegarde finale
        self.save_optimization_state()
        
        logger.info("ContinuousOptimizationDBE shutdown completed")