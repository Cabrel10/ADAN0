
"""
Dynamic Behavior Engine (DBE) - Module de contrôle adaptatif pour l'agent de trading.

Ce module implémente un système de modulation dynamique des paramètres de trading
en fonction des performances et des conditions de marché en temps réel.
"""
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import json
import os
from datetime import datetime
from dataclasses import dataclass, field
import logging
import pickle
from pathlib import Path
from functools import lru_cache

import yaml

from ..common.utils import get_logger
from ..common.replay_logger import ReplayLogger

logger = get_logger(__name__)

# Fichier de log pour le suivi des décisions du DBE
DBE_LOG_FILE = os.getenv("DBE_LOG_FILE", "dbe_replay.jsonl")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                         np.int16, np.int32, np.int64, np.uint8,
                         np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


class DBELogger(logging.Logger):
    """Logger personnalisé pour le Dynamic Behavior Engine."""

    def log_decision(self, step_index, modulation_dict, context_metrics,
                    performance_metrics, additional_info):
        """
        Enregistre une décision du DBE.

        Args:
            step_index: Numéro d'étape
            modulation_dict: Dictionnaire des paramètres de modulation
            context_metrics: Métriques de contexte (régime de marché, etc.)
            performance_metrics: Métriques de performance
            additional_info: Informations supplémentaires
        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'step': step_index,
            'modulation': modulation_dict,
            'context': context_metrics,
            'performance': performance_metrics,
            'additional_info': additional_info
        }

        # Enregistrer dans un fichier JSONL
        log_dir = Path('logs/dbe_decisions')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'dbe_decisions.jsonl'

        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry, cls=NpEncoder) + '\n')
        except Exception as e:
            self.error(f"Erreur lors de l'écriture du log de décision: {e}")


# Configurer le logger personnalisé
logging.setLoggerClass(DBELogger)

@dataclass
class DBESnapshot:
    """
    Snapshot de l'état du Dynamic Behavior Engine (DBE) à un instant T.

    Attributes:
        timestamp: Horodatage du snapshot
        step: Numéro d'étape actuel
        market_regime: Régime de marché détecté (ex: 'BULL', 'BEAR', 'NEUTRAL')
        risk_level: Niveau de risque actuel (0.0 à 2.0)
        sl_pct: Pourcentage de stop-loss (0.0 à 1.0)
        tp_pct: Pourcentage de take-profit (0.0 à 1.0)
        position_size_pct: Taille de position en pourcentage du capital (0.0 à 1.0)
        reward_boost: Facteur d'amplification des récompenses
        penalty_inaction: Pénalité pour inaction
        metrics: Dictionnaire de métriques supplémentaires
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)
    step: int = 0
    market_regime: str = "NEUTRAL"
    risk_level: float = 1.0
    sl_pct: float = 0.02
    tp_pct: float = 0.04
    position_size_pct: float = 0.1
    reward_boost: float = 1.0
    penalty_inaction: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

DEFAULT_CONFIG = {
    # Détection des régimes de marché
    'market_regime_detection': {
        'adx_threshold': 25,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'atr_multiplier': 1.8,
        'ema_fast': 20,
        'ema_slow': 50,
        'regime_confidence_threshold': 0.7
    },

    # Paramètres par régime de marché
    'regime_parameters': {
        'bull': {
            'sl_multiplier': 1.1,
            'tp_multiplier': 1.3,
            'position_size_multiplier': 0.8
        },
        'bear': {
            'sl_multiplier': 0.7,
            'tp_multiplier': 0.9,
            'position_size_multiplier': 0.6
        },
        'volatile': {
            'sl_multiplier': 0.6,
            'tp_multiplier': 0.7,
            'position_size_multiplier': 0.5
        },
        'sideways': {
            'sl_multiplier': 0.9,
            'tp_multiplier': 1.1,
            'position_size_multiplier': 0.7
        }
    },

    # Gestion des risques
    'risk_management': {
        'min_trade_value': 11.0,  # Minimum de 11 USDT par trade
        'max_drawdown_pct': 4.0,  # Drawdown maximum absolu de 4%
        'max_leverage': 1.0
    },

    # Gestion de la volatilité
    'volatility_management': {
        'lookback': 21,  # Période de lookback en jours
        'min_volatility': 0.01,  # Volatilité minimale
        'max_volatility': 0.20  # Volatilité maximale
    },

    # Paramètres de risque de base
    'risk_parameters': {
        'min_risk_level': 0.3,
        'max_risk_level': 2.0,
        'risk_increment': 0.1,
        'risk_decrement': 0.05,
        'max_drawdown': 0.1,
        'max_position_size': 0.5,
        'initial_sl_pct': 0.02,
        'initial_tp_pct': 0.04,
        'initial_position_size': 0.1,
        'initial_risk': 0.5,
        'base_sl_pct': 0.02,
        'base_tp_pct': 0.04,
        'max_sl_pct': 0.10,
        'min_sl_pct': 0.005,
        'daily_loss_limit': 0.05,  # 5% de perte journalière max
        'weekly_loss_limit': 0.15  # 15% de perte hebdomadaire max
    },

    # Paramètres de lissage
    'smoothing': {
        'initial_factor': 0.1,
        'min_factor': 0.01,
        'max_factor': 0.5,
        'adaptation_rate': 0.01
    },

    # Paramètres de performance
    'performance': {
        'lookback_window': 100,
        'min_trades_for_metrics': 10,
        'risk_free_rate': 0.0,
        'annualization_factor': 252
    },

    # Paramètres d'apprentissage
    'learning': {
        'learning_rate_range': [1e-5, 1e-3],
        'ent_coef_range': [0.001, 0.1],
        'gamma_range': [0.9, 0.999],
        'batch_size': 64,
        'n_steps': 2048,
        'n_epochs': 10,
        'clip_range': 0.2
    }
}

class DynamicBehaviorEngine:
    """
    Moteur de comportement dynamique avancé qui ajuste les paramètres de trading
    en fonction des conditions de marché, de la performance du portefeuille
    et de l'état interne de l'agent.
    """

    # Constante pour le montant minimum d'un trade (en USDT)
    MIN_TRADE = 11.0

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 finance_manager: Optional[Any] = None):
        """
        Initialise le DBE avec la configuration fournie.

        Args:
            config: Dictionnaire de configuration (optionnel)
            finance_manager: Instance de FinanceManager (optionnel)
        """
        # Fusion de la configuration par défaut avec celle fournie
        self.config = self._merge_configs(DEFAULT_CONFIG, config or {})

        # Référence au gestionnaire de portefeuille
        self.finance_manager = finance_manager

        # État actuel du marché
        self.current_regime = 'neutral'
        self.regime_confidence = 0.0
        self.last_regime_change = datetime.utcnow()

        # Métriques de performance
        self.portfolio_value_history = []
        self.daily_returns = []
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0

        # Initialisation des paramètres de trading
        self.current_sl_multiplier = 1.0
        self.current_tp_multiplier = 1.0
        self.current_position_size_multiplier = 1.0

        # Log available config keys for debugging
        if 'position_sizing' not in self.config:
            logger.error(f"'position_sizing' key not found in DBE config. Available keys: {list(self.config.keys())}")

        self.max_position_size = self.config['position_sizing']['max_position_size']

        # Paramètres de lissage
        self.smoothing_factor = self.config.get('smoothing', {}).get('initial_factor', 0.1)
        self.smoothed_params = {
            'sl_pct': self.config['position_sizing'].get('initial_sl_pct', 0.02),
            'tp_pct': self.config['position_sizing'].get('initial_tp_pct', 0.04),
            'position_size': self.config['position_sizing'].get('initial_position_size', 0.1),
            'risk_level': 1.0
        }

        # Initialisation du logger personnalisé
        self.logger = logging.getLogger(f"dbe.{self.__class__.__name__}")

        # S'assurer que le logger est bien de type DBELogger
        if not isinstance(self.logger, DBELogger):
            self.logger.__class__ = DBELogger

        # Initialisation des historiques
        self.decision_history = []  # Historique des décisions prises
        self.trade_history = []     # Historique des trades

        # Initialisation des états des workers
        self.worker_states = {}

        # Initialisation de l'état
        self.state = {
            'current_step': 0,
            'last_trade_step': 0,
            'consecutive_losses': 0,
            'consecutive_wins': 0,
            'last_win': False,
            'last_reward': 0.0,
            'drawdown': 0.0,
            'current_risk_level': 1.0,  # Niveau de risque initial (1.0 = neutre)
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'recovery_factor': 0.0,
            'expectancy': 0.0,
            'avg_trade': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'equity_curve': [],
            'returns': [],
            'drawdowns': [],
            'position_duration': 0,  # Durée de la position actuelle en pas de temps
            'market_conditions': {},
            'performance_metrics': {  # Ajout des métriques de performance initiales
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'avg_trade': 0.0,
                'expectancy': 0.0
            }
        }

        # Chargement de la configuration externe si elle existe
        dbe_config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'dbe_config.yaml'
        if dbe_config_path.exists():
            with open(dbe_config_path, 'r') as f:
                dbe_config = yaml.safe_load(f) or {}
                if dbe_config:
                    self.config = self._merge_configs(self.config, dbe_config)

    def _merge_configs(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Fusionne récursivement deux dictionnaires de configuration."""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def detect_market_regime(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """Détecte le régime de marché actuel."""
        adx = market_data.get('adx', 0)
        rsi = market_data.get('rsi', 50)
        ema_fast = market_data.get('ema_fast', 0)
        ema_slow = market_data.get('ema_slow', 0)

        if adx > self.config['market_regime_detection']['adx_threshold']:
            if ema_fast > ema_slow:
                return 'bull', 0.7 + (0.3 * (adx / 100))
            else:
                return 'bear', 0.7 + (0.3 * (adx / 100))
        else:
            if rsi > 70 or rsi < 30:
                return 'volatile', 0.8
            return 'sideways', 0.9

    def _get_capital_tier(self, portfolio_value: float) -> Optional[Dict[str, Any]]:
        """Determines the capital tier based on the portfolio value."""
        capital_tiers = self.config.get('capital_tiers')
        if not capital_tiers or not isinstance(capital_tiers, list):
            logger.warning("Configuration 'capital_tiers' manquante ou invalide.")
            return None
        for tier in capital_tiers:
            min_capital = tier.get('min_capital', 0)
            max_capital = tier.get('max_capital')
            if max_capital is None: # For the highest tier, max_capital can be null
                max_capital = float('inf')

            if min_capital <= portfolio_value < max_capital:
                logger.debug(f"Capital {portfolio_value:.2f} USDT correspond au palier: {tier.get('name')}")
                return tier
        logger.warning(f"Aucun palier de capital trouvé pour un portefeuille de {portfolio_value:.2f} USDT.")
        return None

    def update_risk_parameters(self, market_data: Dict[str, Any], portfolio_value: float) -> Dict[str, float]:
        """
        Met à jour les paramètres de risque en fonction du régime de marché détecté
        et des conditions actuelles du marché, en utilisant la logique des paliers de capital.
        """
        try:
            # --- Nouvelle logique de Paliers de Capital ---
            tier = self._get_capital_tier(portfolio_value)

            # Détection du régime de marché
            new_regime, confidence = self.detect_market_regime(market_data)
            if (confidence > self.config.get('market_regime_detection', {}).get('regime_confidence_threshold', 0.7) and
                new_regime != self.current_regime):
                self.current_regime = new_regime
                logger.info(f"Changement de régime détecté: {new_regime.upper()} (Confiance: {confidence*100:.1f}%)")

            regime_params = self.config.get('regime_parameters', {}).get(self.current_regime, {})
            risk_params = self.config.get('risk_parameters', {})

            # --- Calcul de la taille de position basé sur les paliers ---
            position_size_pct = risk_params.get('initial_position_size', 0.1) # Default

            if tier:
                exposure_range = tier.get('exposure_range', [1, 10])
                min_exposure_pct = exposure_range[0] / 100.0
                max_exposure_pct = exposure_range[1] / 100.0

                risk_level = self.state.get('current_risk_level', 1.0)
                # Normaliser le niveau de risque (0.3-2.0) en un facteur (0-1)
                normalized_risk = (risk_level - 0.3) / (2.0 - 0.3)

                # Calculer la taille de position en %% en utilisant l'intervalle du palier
                position_size_pct = min_exposure_pct + (max_exposure_pct - min_exposure_pct) * normalized_risk

                # Appliquer la contrainte max du palier
                max_tier_size_pct = tier.get('max_position_size_pct', 90) / 100.0
                position_size_pct = min(position_size_pct, max_tier_size_pct)

                # Vérifier la valeur minimale du trade
                min_trade_value = self.config.get('risk_management', {}).get('min_trade_value', 11.0)
                if portfolio_value > 0 and (position_size_pct * portfolio_value) < min_trade_value:
                    # Si la taille calculée est inférieure au minimum, on prend le minimum,
                    # mais seulement si cela ne dépasse pas la taille max du palier.
                    if (min_trade_value / portfolio_value) <= max_tier_size_pct:
                        position_size_pct = min_trade_value / portfolio_value
                    else:
                        # Conflit de configuration, on plafonne à la taille max
                        position_size_pct = max_tier_size_pct
                        logger.warning(f"Conflit de configuration pour le palier {tier.get('name')}: min_trade_value "
                                     f"({min_trade_value} USDT) est supérieur à max_position_size_pct "
                                     f"({max_tier_size_pct*100}%%). Plafonnement à la taille maximale.")
            else:
                # Fallback à l'ancienne logique si aucun palier n'est trouvé
                position_size_multiplier = regime_params.get('position_size_multiplier', 1.0)
                risk_level = self.state.get('current_risk_level', 1.0)
                position_size_pct = np.clip(
                    risk_params.get('initial_position_size', 0.1) * position_size_multiplier * risk_level,
                    risk_params.get('min_position_size', 0.01),
                    risk_params.get('max_position_size', 0.5)
                )

            # --- Calcul du SL/TP (logique existante) ---
            close_price = market_data.get('close', 1.0)
            atr = market_data.get('atr', 0.0)
            volatility = atr / close_price if close_price > 0 else 0.0
            vol_management = self.config.get('volatility_management', {})
            min_vol = vol_management.get('min_volatility', 0.01)
            max_vol = vol_management.get('max_volatility', 0.20)
            vol_adjustment = np.clip((volatility - min_vol) / (max_vol - min_vol + 1e-6) * 1.5 + 0.5, 0.5, 2.0)

            base_sl_pct = risk_params.get('base_sl_pct', 0.02)
            sl_multiplier = regime_params.get('sl_multiplier', 1.0)
            min_sl_pct = risk_params.get('min_sl_pct', 0.005)
            max_sl_pct = risk_params.get('max_sl_pct', 0.10)
            stop_loss_pct = np.clip(base_sl_pct * sl_multiplier * vol_adjustment, min_sl_pct, max_sl_pct)

            base_tp_pct = risk_params.get('base_tp_pct', 0.04)
            tp_multiplier = regime_params.get('tp_multiplier', 1.0)
            take_profit_pct = base_tp_pct * tp_multiplier

            return {
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'position_size_pct': position_size_pct,
                'regime': self.current_regime,
                'regime_confidence': confidence,
                'volatility': volatility,
                'vol_adjustment': vol_adjustment,
                'risk_level': self.state.get('current_risk_level', 1.0)
            }

        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des paramètres de risque: {e}", exc_info=True)
            # Retour de valeurs par défaut en cas d'erreur
            return {
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'position_size_pct': 0.1,
                'regime': self.current_regime,
                'regime_confidence': 0.0,
                'volatility': 0.0,
                'vol_adjustment': 1.0,
                'risk_level': 1.0,
            }

    def deep_update(self, d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self.deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def update(self, dbe_config):
        self.deep_update(self.config, dbe_config)

        # Merge the provided config argument, which takes the highest precedence
        if config:
            self.config.update(config)

        # Initialisation du gestionnaire financier
        self.finance_manager = finance_manager

        # État interne
        self.state = {
            'current_step': 0,
            'drawdown': 0.0,
            'winrate': 0.0,
            'volatility': 0.0,
            'market_regime': 'NEUTRAL',
            'last_trade_pnl': 0.0,
            'consecutive_losses': 0,
            'position_duration': 0,
            'current_risk_level': 1.0,  # Niveau de risque initial (1.0 = neutre)
            'max_risk_level': 2.0,
            'min_risk_level': 0.5,
            'last_modulation': {},
            'performance_metrics': {}
        }

        # Historique des trades et des décisions
        self.trade_history: List[Dict[str, Any]] = []
        self.decision_history: List[DBESnapshot] = []
        self.win_rates = []
        self.drawdowns = []
        self.position_durations = []
        self.pnl_history = []
        self.trade_results = []

        # Initialisation du logger de relecture
        log_config = self.config.get('logging', {})
        self.logger = ReplayLogger(
            log_dir=log_config.get('log_dir', 'logs/dbe'),
            compression=log_config.get('compression', 'gzip')
        )

        # Configuration du niveau de log
        log_level = log_config.get('log_level', 'INFO').upper()
        logging.getLogger().setLevel(getattr(logging, log_level))

        # Initialisation des paramètres de lissage
        self.smoothing_factor = self.config.get('smoothing_factor', 0.1)  # Facteur de lissage exponentiel
        self.smoothed_params = {
            'sl_pct': self.config.get('risk_parameters', {}).get('base_sl_pct', 0.02),
            'tp_pct': self.config.get('risk_parameters', {}).get('base_tp_pct', 0.04)
        }

        # Configuration de la persistance d'état
        self.state_persistence_enabled = config.get('state_persistence', {}).get('enabled', True) if config else True
        self.state_save_path = config.get('state_persistence', {}).get('save_path', 'logs/dbe/state') if config else 'logs/dbe/state'
        self.state_save_interval = config.get('state_persistence', {}).get('save_interval', 100) if config else 100

        logger.info("🚀 Dynamic Behavior Engine initialisé (version avancée)")
        logger.info(f"Configuration: {json.dumps(self._serialize_config(), indent=2)}")
        logger.info(f"Persistance d'état: {'Activée' if self.state_persistence_enabled else 'Désactivée'}")

    def _serialize_config(self) -> Dict[str, Any]:
        """Sérialise la configuration pour le logging."""
        # Crée une copie profonde pour éviter de modifier la configuration originale
        config = self.config.copy()

        # Éviter de logger des informations sensibles
        if 'api_keys' in config:
            config['api_keys'] = {k: '***' for k in config['api_keys']}

        return config

    def update_state(self, live_metrics: Dict[str, Any]) -> None:
        """
        Met à jour l'état interne du DBE avec les dernières métriques.

        Args:
            live_metrics: Dictionnaire des métriques en temps réel
        """
        try:
            # Mise à jour des métriques de base
            self.state['current_step'] += 1
            self.state['volatility'] = live_metrics.get('volatility', 0.0)

            # Détection du régime de marché avec les paramètres individuels pour le cache
            rsi = live_metrics.get('rsi')
            adx = live_metrics.get('adx')
            ema_ratio = live_metrics.get('ema_ratio')
            atr = live_metrics.get('atr', 0.0)
            atr_pct = live_metrics.get('atr_pct')

            self.state['market_regime'] = self._detect_market_regime(
                rsi=rsi, adx=adx, ema_ratio=ema_ratio, atr=atr, atr_pct=atr_pct
            )

            # Mise à jour des métriques de performance si le gestionnaire financier est disponible
            if self.finance_manager:
                portfolio_metrics = self.finance_manager.get_performance_metrics()
                self.state['winrate'] = portfolio_metrics.get('win_rate', 0.0)
                self.state['drawdown'] = portfolio_metrics.get('max_drawdown', 0.0) * 100  # En pourcentage

                # Mise à jour de l'historique des trades
                if 'recent_trades' in portfolio_metrics:
                    self.trade_history.extend(portfolio_metrics['recent_trades'])

                    # Mise à jour des pertes consécutives
                    recent_trades = portfolio_metrics['recent_trades']
                    if recent_trades and not recent_trades[-1].get('is_win', False):
                        self.state['consecutive_losses'] += 1
                    else:
                        self.state['consecutive_losses'] = 0

            # Ajustement du niveau de risque en fonction des performances
            self._adjust_risk_level()

            # Adaptation du facteur de lissage
            self._adapt_smoothing_factor()

            # Journalisation de l'état actuel
            logger.debug(f"État DBE mis à jour - Régime: {self.state['market_regime']}, "
                       f"Risque: {self.state['current_risk_level']:.2f}, "
                       f"Winrate: {self.state['winrate']*100:.1f}%")

        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de l'état du DBE: {e}", exc_info=True)
            raise

    def _process_trade_result(self, trade_result: Dict[str, Any]) -> None:
        """Traite le résultat d'un trade et met à jour les métriques."""
        self.state['last_trade_pnl'] = trade_result.get('pnl_pct', 0.0)

        # Mise à jour du nombre de pertes consécutives
        if trade_result.get('pnl_pct', 0) <= 0:
            self.state['consecutive_losses'] += 1
        else:
            self.state['consecutive_losses'] = 0

        # Mise à jour de la durée de position
        if 'position_duration' in trade_result:
            self.position_durations.append(trade_result['position_duration'])
            self.state['position_duration'] = np.mean(self.position_durations[-100:]) if self.position_durations else 0

        # Mise à jour du taux de réussite
        if 'is_win' in trade_result:
            self.win_rates.append(1 if trade_result['is_win'] else 0)
            self.state['winrate'] = np.mean(self.win_rates[-100:]) if self.win_rates else 0.0

        # Mise à jour du drawdown
        if 'drawdown' in trade_result:
            self.drawdowns.append(trade_result['drawdown'])
            self.state['drawdown'] = np.mean(self.drawdowns[-100:]) if self.drawdowns else 0.0

        # Ajout à l'historique des trades
        self.trade_history.append({
            'timestamp': datetime.utcnow(),
            'pnl_pct': trade_result.get('pnl_pct', 0.0),
            'is_win': trade_result.get('is_win', False),
            'position_duration': trade_result.get('position_duration', 0),
            'drawdown': trade_result.get('drawdown', 0.0),
            'market_regime': self.state['market_regime']
        })

    @property
    def market_regime(self) -> str:
        """Get current market regime."""
        return self.state.get('market_regime', 'NEUTRAL')

    @property
    def current_step(self) -> int:
        """Get current step."""
        return self.state.get('current_step', 0)

    @property
    def risk_level(self) -> float:
        """Get current risk level."""
        return self.state.get('current_risk_level', 1.0)

    def save_state(self, filepath: str) -> None:
        """Save the current state to a file."""
        state_data = {
            'state': self.state,
            'trade_history': self.trade_history,
            'decision_history': [vars(snapshot) for snapshot in self.decision_history],
            'win_rates': self.win_rates,
            'drawdowns': self.drawdowns,
            'position_durations': self.position_durations
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state_data, f)

        logger.info(f"DBE state saved to {filepath}")

    def load_state(self, filepath: str) -> None:
        """Load state from a file."""
        with open(filepath, 'rb') as f:
            state_data = pickle.load(f)

        self.state = state_data['state']
        self.trade_history = state_data['trade_history']
        self.decision_history = [DBESnapshot(**snapshot) for snapshot in state_data['decision_history']]
        self.win_rates = state_data['win_rates']
        self.drawdowns = state_data['drawdowns']
        self.position_durations = state_data['position_durations']

        logger.info(f"DBE state loaded from {filepath}")

    def get_status(self) -> str:
        """Get current status as a formatted string."""
        return f"DBE Status - Step: {self.current_step}, Regime: {self.market_regime}, Risk: {self.risk_level:.2f}"

    def on_trade_closed(self, trade_result: Dict[str, Any]) -> None:
        """Process a closed trade result."""
        self._process_trade_result(trade_result)

    def compute_dynamic_modulation(self) -> Dict[str, Any]:
        """
        Calcule la modulation dynamique des paramètres de trading.

        Returns:
            Dictionnaire contenant les paramètres modulés
        """
        try:
            # Initialisation des paramètres de base
            mod = {
                'sl_pct': self.config.get('risk_parameters', {}).get('base_sl_pct', 0.02),
                'tp_pct': self.config.get('risk_parameters', {}).get('base_tp_pct', 0.04),
                'reward_boost': 1.0,
                'penalty_inaction': 0.0,
                'position_size_pct': self.config.get('position_sizing', {}).get('base_position_size', 0.1),
                'leverage': self.config.get('position_sizing', {}).get('base_leverage', 1.0),
                'risk_mode': 'NORMAL',  # 'DEFENSIVE', 'NORMAL', 'AGGRESSIVE'
                'learning_rate': None,
                'ent_coef': None,
                'gamma': None
            }

            # Calcul des paramètres de risque
            self._compute_risk_parameters(self.state, mod)

            # Application des modulations spécifiques au régime de marché
            self._apply_market_regime_modulation(mod)

            # Ajustement des paramètres d'apprentissage
            self._adjust_learning_parameters(mod)

            # Validation et ajustement final des paramètres
            self._validate_parameters(mod)

            # Création d'un snapshot de la décision
            snapshot = DBESnapshot(
                step=self.state['current_step'],
                market_regime=self.state['market_regime'],
                risk_level=self.state['current_risk_level'],
                sl_pct=mod['sl_pct'],
                tp_pct=mod['tp_pct'],
                position_size_pct=mod['position_size_pct'],
                reward_boost=mod['reward_boost'],
                penalty_inaction=mod['penalty_inaction'],
                metrics=self.state['performance_metrics'].copy()
            )

            # Ajout à l'historique des décisions
            self.decision_history.append(snapshot)
            self.state['last_modulation'] = mod.copy()

            return mod

        except Exception as e:
            logger.error(f"Erreur lors du calcul de la modulation: {e}", exc_info=True)
            # Retourner une modulation sécurisée en cas d'erreur
            return {
                'sl_pct': 0.02,
                'tp_pct': 0.04,
                'reward_boost': 1.0,
                'penalty_inaction': 0.0,
                'position_size_pct': 0.1,
                'risk_mode': 'NORMAL',
                'error': str(e)
            }



    def _apply_market_regime_modulation(self, mod: Dict[str, Any]) -> None:
        """Applique les modulations spécifiques au régime de marché."""
        regime = self.state['market_regime'].upper()
        mode_config = self.config.get('modes', {}).get(regime.lower(), {})

        if not mode_config:
            return

        # Application des multiplicateurs
        mod['sl_pct'] *= mode_config.get('sl_multiplier', 1.0)
        mod['tp_pct'] *= mode_config.get('tp_multiplier', 1.0)
        mod['position_size_pct'] *= mode_config.get('position_size_multiplier', 1.0)
        mod['risk_mode'] = regime

    def _adjust_learning_parameters(self, mod: Dict[str, Any]) -> None:
        """Ajuste les paramètres d'apprentissage en fonction du risque."""
        learning_config = self.config.get('learning', {})

        # Récupération et validation des plages de valeurs
        lr_range = [max(1e-8, float(x)) for x in learning_config.get('learning_rate_range', [1e-5, 1e-3])]  # Minimum 1e-8 pour éviter les valeurs trop petites
        ent_coef_range = [max(1e-8, float(x)) for x in learning_config.get('ent_coef_range', [0.001, 0.1])]
        gamma_range = [max(0.1, min(float(x), 0.999)) for x in learning_config.get('gamma_range', [0.9, 0.999])]  # Gamma entre 0.1 et 0.999

        # Ajustement basé sur le niveau de risque avec clamping
        risk_factor = max(0.1, min(float(self.state['current_risk_level']), 10.0))  # Limite le facteur de risque entre 0.1 et 10.0

        try:
            # Calcul du learning rate avec clamping pour éviter les valeurs négatives ou trop élevées
            base_lr = lr_range[0] + (lr_range[1] - lr_range[0]) * (risk_factor - 1.0)
            mod['learning_rate'] = max(1e-8, min(base_lr, lr_range[1] * 2.0))  # Ne dépasse pas le double du max

            # Calcul de l'entropy coefficient avec clamping
            ent_coef = ent_coef_range[0] + (ent_coef_range[1] - ent_coef_range[0]) * (1.0 / max(0.1, risk_factor))
            mod['ent_coef'] = max(1e-8, min(ent_coef, ent_coef_range[1] * 2.0))

            # Calcul du gamma avec clamping
            gamma = gamma_range[0] + (gamma_range[1] - gamma_range[0]) * (min(risk_factor, 2.0) - 1.0)
            mod['gamma'] = max(0.1, min(gamma, 0.999))

            # Logging détaillé pour le débogage
            logger.debug(
                f"Learning params - Risk: {risk_factor:.2f}, "
                f"LR: {mod['learning_rate']:.2e}, "
                f"EntCoef: {mod['ent_coef']:.4f}, "
                f"Gamma: {mod['gamma']:.3f}"
            )

        except (TypeError, ValueError) as e:
            logger.error(f"Erreur dans le calcul des paramètres d'apprentissage: {e}")
            # Valeurs par défaut sécurisées en cas d'erreur
            mod['learning_rate'] = lr_range[0]
            mod['ent_coef'] = ent_coef_range[0]
            mod['gamma'] = gamma_range[0]

    def _validate_parameters(self, mod: Dict[str, Any]) -> None:
        """Valide et contraint les paramètres dans des limites acceptables."""
        risk_params = self.config.get('risk_parameters', {})
        pos_params = self.config.get('position_sizing', {})

        # Contraintes sur les SL/TP
        mod['sl_pct'] = np.clip(
            mod['sl_pct'],
            risk_params.get('min_sl_pct', 0.005),
            risk_params.get('max_sl_pct', 0.10)
        )

        # TP minimum = 1.5x SL pour assurer un ratio risque/rendement positif
        min_tp_ratio = 1.5
        mod['tp_pct'] = max(mod['tp_pct'], mod['sl_pct'] * min_tp_ratio)

        # Contraintes sur la taille de position
        mod['position_size_pct'] = np.clip(
            mod['position_size_pct'],
            pos_params.get('min_position_size', 0.01),
            pos_params.get('max_position_size', 0.30)
        )

    def _log_decision(self, snapshot: DBESnapshot, mod: Dict[str, Any]) -> None:
        """Journalise la décision prise par le DBE."""
        decision_data = {
            'step': snapshot.step,
            'market_regime': snapshot.market_regime,
            'risk_level': snapshot.risk_level,
            'modulation': {
                'sl_pct': snapshot.sl_pct,
                'tp_pct': snapshot.tp_pct,
                'position_size_pct': snapshot.position_size_pct,
                'reward_boost': snapshot.reward_boost,
                'penalty_inaction': snapshot.penalty_inaction,
                'learning_rate': mod.get('learning_rate'),
                'ent_coef': mod.get('ent_coef'),
                'gamma': mod.get('gamma')
            },
            'performance_metrics': snapshot.metrics,
            'timestamp': snapshot.timestamp.isoformat()
        }

        # Utilisation du ReplayLogger pour enregistrer la décision
        self.logger.log_decision(
            step_index=snapshot.step,
            modulation_dict=decision_data['modulation'],
            context_metrics={
                'market_regime': snapshot.market_regime,
                'risk_level': snapshot.risk_level,
                'drawdown': self.state.get('drawdown', 0.0),
                'winrate': self.state.get('winrate', 0.0),
                'volatility': self.state.get('volatility', 0.0)
            },
            performance_metrics=snapshot.metrics,
            additional_info={
                'consecutive_losses': self.state['consecutive_losses'],
                'position_duration': self.state['position_duration']
            }
        )

        logger.info(
            f"DBE Decision - Step: {snapshot.step} | "
            f"Regime: {snapshot.market_regime} | "
            f"SL: {snapshot.sl_pct*100:.2f}% | "
            f"TP: {snapshot.tp_pct*100:.2f}% | "
            f"PosSize: {snapshot.position_size_pct*100:.1f}% | "
            f"Winrate: {self.state['winrate']*100:.1f}%"
        )

    @lru_cache(maxsize=128)
    def _detect_market_regime(self, rsi: float, adx: float, ema_ratio: float, atr: float, atr_pct: float) -> str:
        """
        Détecte le régime de marché actuel à partir des indicateurs techniques.

        Args:
            rsi: Indice de force relative (0-100)
            adx: Average Directional Index (0-100)
            ema_ratio: Ratio EMA rapide / lente
            atr: Average True Range
            atr_pct: ATR en pourcentage du prix

        Returns:
            Chaîne identifiant le régime de marché
        """
        try:
            # Nettoyage des entrées
            rsi = float(rsi) if rsi is not None else 50.0
            adx = float(adx) if adx is not None else 20.0
            ema_ratio = float(ema_ratio) if ema_ratio is not None else 1.0
            atr_pct = float(atr_pct) if atr_pct is not None else 0.0

            # Détection du régime de marché
            if adx > 25:  # Marché avec tendance
                if ema_ratio > 1.005:  # Tendance haussière
                    return 'BULL'
                elif ema_ratio < 0.995:  # Tendance baissière
                    return 'BEAR'

            # Marché sans tendance
            if atr_pct > 0.02:  # Volatilité élevée
                return 'VOLATILE'
            else:
                return 'SIDEWAYS'

        except Exception as e:
            logger.error(f"Erreur lors de la détection du régime de marché: {e}")
            return 'UNKNOWN'

    def _adjust_risk_level(self) -> None:
        """
        Ajuste dynamiquement le niveau de risque en fonction des performances récentes
        et des conditions de marché.
        """
        try:
            # Récupération des paramètres de risque depuis la configuration
            risk_params = self.config.get('risk_parameters', {})
            min_risk = risk_params.get('min_risk_level', 0.3)
            max_risk = risk_params.get('max_risk_level', 2.0)

            # Initialisation de l'état si nécessaire
            self.state.setdefault('current_risk_level', risk_params.get('initial_risk', 0.5))
            self.state.setdefault('win_rate', 0.0)
            self.state.setdefault('drawdown', 0.0)
            self.state.setdefault('consecutive_losses', 0)
            self.state.setdefault('sharpe_ratio', 0.0)
            self.state.setdefault('sortino_ratio', 0.0)
            self.state.setdefault('volatility', 0.0)

            # Récupération des métriques de performance
            portfolio_metrics = self.finance_manager.get_performance_metrics() if self.finance_manager else {}
            win_rate = portfolio_metrics.get('win_rate', self.state['win_rate'])
            drawdown = portfolio_metrics.get('drawdown', self.state['drawdown'])

            # Facteurs d'ajustement du risque
            # 1. Facteur basé sur le taux de réussite (win rate)
            target_win_rate = 0.6  # Cible de 60% de trades gagnants
            win_rate_factor = (win_rate / target_win_rate) ** 2  # Carré pour un effet non linéaire

            # 2. Facteur basé sur le drawdown
            max_allowed_drawdown = risk_params.get('max_drawdown', 0.1)  # 10% par défaut
            drawdown_factor = 1.0 - min(drawdown / (max_allowed_drawdown * 2), 0.5)  # Réduction jusqu'à 50%

            # 3. Facteur basé sur le ratio de Sharpe
            sharpe_ratio = portfolio_metrics.get('sharpe_ratio', 0.0)
            sharpe_factor = 1.0 + (sharpe_ratio / 2.0)  # Augmente le risque avec un meilleur Sharpe

            # 4. Facteur basé sur la volatilité
            vol_management = self.config.get('volatility_management', {})
            min_vol = vol_management.get('min_volatility', 0.01)
            max_vol = vol_management.get('max_volatility', 0.20)
            current_vol = portfolio_metrics.get('volatility', min_vol)
            vol_factor = 1.0 - ((current_vol - min_vol) / (max_vol - min_vol + 1e-6)) * 0.5  # Réduction jusqu'à 50%

            # 5. Facteur basé sur les pertes consécutives
            loss_streak_factor = 1.0 / (1.0 + self.state.get('consecutive_losses', 0) * 0.2)

            # Calcul du nouveau niveau de risque
            base_risk = self.state['current_risk_level']
            new_risk = base_risk * win_rate_factor * drawdown_factor * sharpe_factor * vol_factor * loss_streak_factor

            # Lissage pour éviter les changements trop brutaux
            smoothing = self.config.get('smoothing', {})
            alpha = smoothing.get('adaptation_rate', 0.1)  # Vitesse d'adaptation
            smoothed_risk = (1.0 - alpha) * base_risk + alpha * new_risk

            # Application des limites
            self.state['current_risk_level'] = np.clip(smoothed_risk, min_risk, max_risk)

            # Mise à jour des métriques dans l'état
            self.state.update({
                'win_rate': win_rate,
                'drawdown': drawdown,
                'sharpe_ratio': sharpe_ratio,
                'volatility': current_vol
            })

            # Journalisation détaillée
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Ajustement du risque - Niveau: {self.state['current_risk_level']:.2f} | "
                    f"Win Rate: {win_rate*100:.1f}% | "
                    f"Drawdown: {drawdown*100:.1f}% | "
                    f"Sharpe: {sharpe_ratio:.2f} | "
                    f"Volatilité: {current_vol*100:.1f}% | "
                    f"Pertes consécutives: {self.state.get('consecutive_losses', 0)}"
                )

        except Exception as e:
            logger.error(f"Erreur lors de l'ajustement du niveau de risque: {e}")
            logger.exception("Détails de l'erreur:")
            # En cas d'erreur, on conserve le niveau de risque actuel

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques de performance actuelles.

        Returns:
            Dictionnaire des métriques de performance
        """
        if not self.finance_manager:
            return {}

        # Récupération des métriques du gestionnaire financier
        portfolio_metrics = self.finance_manager.get_performance_metrics()

        # Calcul des métriques avancées
        if self.trade_history:
            recent_trades = self.trade_history[-100:]  # 100 derniers trades
            pnls = tuple(t['pnl_pct'] for t in recent_trades if 'pnl_pct' in t)
            wins = [t for t in recent_trades if t.get('is_win', False)]
            losses = [t for t in recent_trades if not t.get('is_win', True)]

            avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0.0
            avg_loss = abs(np.mean([t['pnl_pct'] for t in losses])) if losses else 0.0
            win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')

            # Utilisation des méthodes mises en cache
            risk_free_rate = 0.0  # Taux sans risque (peut être paramétré)
            sharpe_ratio = self._calculate_sharpe_ratio(pnls, risk_free_rate) if pnls else 0.0
            sortino_ratio = self._calculate_sortino_ratio(pnls, risk_free_rate) if pnls else 0.0
        else:
            avg_win = avg_loss = win_loss_ratio = sharpe_ratio = sortino_ratio = 0.0

        # Construction du dictionnaire de résultats
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'step': self.state['current_step'],
            'portfolio': {
                'total_value': portfolio_metrics.get('total_capital', 0.0),
                'free_cash': portfolio_metrics.get('free_capital', 0.0),
                'invested': portfolio_metrics.get('invested_capital', 0.0),
                'total_return': portfolio_metrics.get('total_return', 0.0),
                'max_drawdown': portfolio_metrics.get('max_drawdown', 0.0),
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio
            },
            'trading': {
                'total_trades': portfolio_metrics.get('trade_count', 0),
                'win_rate': portfolio_metrics.get('win_rate', 0.0) * 100,  # en pourcentage
                'avg_win_pct': avg_win * 100,  # en pourcentage
                'avg_loss_pct': avg_loss * 100,  # en pourcentage
                'win_loss_ratio': win_loss_ratio,
                'consecutive_losses': self.state['consecutive_losses'],
                'avg_trade_duration': self.state.get('position_duration', 0)
            },
            'risk': {
                'current_risk_level': self.state['current_risk_level'],
                'market_regime': self.state['market_regime'],
                'current_volatility': self.state.get('volatility', 0.0),
                'current_drawdown': self.state.get('drawdown', 0.0)
            }
        }

        # Mise à jour des métriques de performance dans l'état
        self.state['performance_metrics'] = metrics

        return metrics

    @lru_cache(maxsize=128)
    def _calculate_sharpe_ratio(self, returns_tuple: Tuple[float, ...], risk_free_rate: float = 0.0) -> float:
        """Calcule le ratio de Sharpe annualisé avec mise en cache des résultats.

        Args:
            returns_tuple: Tuple des rendements (doit être hashable pour le cache)
            risk_free_rate: Taux sans risque annuel (par défaut: 0.0)

        Returns:
            Ratio de Sharpe annualisé
        """
        if not returns_tuple:
            return 0.0

        returns = np.array(returns_tuple)
        excess_returns = returns - risk_free_rate / 252  # Taux sans risque journalier
        std_dev = np.std(excess_returns)

        # Éviter la division par zéro
        if std_dev < 1e-9:
            return 0.0

        sharpe = np.mean(excess_returns) / std_dev * np.sqrt(252)
        return float(sharpe)

    @lru_cache(maxsize=128)
    def _calculate_sortino_ratio(self, returns_tuple: Tuple[float, ...], risk_free_rate: float = 0.0) -> float:
        """Calcule le ratio de Sortino annualisé avec mise en cache des résultats.

        Args:
            returns_tuple: Tuple des rendements (doit être hashable pour le cache)
            risk_free_rate: Taux sans risque annuel (par défaut: 0.0)

        Returns:
            Ratio de Sortino annualisé
        """
        if not returns_tuple:
            return 0.0

        returns = np.array(returns_tuple)
        excess_returns = returns - risk_free_rate / 252  # Taux sans risque journalier
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0

        downside_std = np.std(downside_returns)

        # Éviter la division par zéro
        if downside_std < 1e-9:
            return 0.0

        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
        return float(sortino)

    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retourne l'historique des décisions prises par le DBE.

        Args:
            limit: Nombre maximum de décisions à retourner

        Returns:
            Liste des décisions au format dictionnaire
        """
        # Sélection des décisions les plus récentes
        recent_decisions = self.decision_history[-limit:] if self.decision_history else []

        # Conversion des snapshots en dictionnaires
        return [{
            'timestamp': d.timestamp.isoformat(),
            'step': d.step,
            'market_regime': d.market_regime,
            'risk_level': d.risk_level,
            'sl_pct': d.sl_pct,
            'tp_pct': d.tp_pct,
            'position_size_pct': d.position_size_pct,
            'reward_boost': d.reward_boost,
            'penalty_inaction': d.penalty_inaction,
            'metrics': d.metrics
        } for d in recent_decisions]

    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retourne l'historique des trades effectués.

        Args:
            limit: Nombre maximum de trades à retourner

        Returns:
            Liste des trades au format dictionnaire
        """
        # Sélection des trades les plus récents
        recent_trades = self.trade_history[-limit:] if self.trade_history else []

        # Conversion des timestamps en chaînes
        return [{
            'timestamp': t['timestamp'].isoformat() if hasattr(t['timestamp'], 'isoformat') else str(t['timestamp']),
            'pnl_pct': t.get('pnl_pct', 0.0),
            'is_win': t.get('is_win', False),
            'position_duration': t.get('position_duration', 0),
            'drawdown': t.get('drawdown', 0.0),
            'market_regime': t.get('market_regime', 'UNKNOWN')
        } for t in recent_trades]

    def save_state(self, filepath: Union[str, Path]) -> bool:
        """
        Sauvegarde l'état actuel du DBE dans un fichier.

        Args:
            filepath: Chemin vers le fichier de sauvegarde

        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        try:
            state = {
                'state': self.state,
                'trade_history': self.trade_history,
                'decision_history': [d.__dict__ for d in self.decision_history],
                'win_rates': self.win_rates,
                'drawdowns': self.drawdowns,
                'position_durations': self.position_durations,
                'config': self.config
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            logger.info(f"État du DBE sauvegardé dans {filepath}")
            return True

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'état du DBE: {e}")
            return False

    @classmethod
    def load_state(cls, filepath: Union[str, Path], finance_manager: Optional[Any] = None) -> Optional['DynamicBehaviorEngine']:
        """
        Charge un état précédemment sauvegardé.

        Args:
            filepath: Chemin vers le fichier de sauvegarde
            finance_manager: Instance de FinanceManager (optionnel)

        Returns:
            Une instance de DynamicBehaviorEngine avec l'état chargé, ou None en cas d'erreur
        """
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            # Création d'une nouvelle instance avec la configuration sauvegardée
            dbe = cls(config=state.get('config', {}), finance_manager=finance_manager)

            # Restauration de l'état
            dbe.state = state.get('state', {})
            dbe.trade_history = state.get('trade_history', [])
            dbe.decision_history = [DBESnapshot(**d) for d in state.get('decision_history', [])]
            dbe.win_rates = state.get('win_rates', [])
            dbe.drawdowns = state.get('drawdowns', [])
            dbe.position_durations = state.get('position_durations', [])

            logger.info(f"État du DBE chargé depuis {filepath}")
            return dbe

        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'état du DBE: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """
        Retourne un résumé de l'état actuel du DBE.

        Returns:
            Dictionnaire contenant les informations de statut
        """
        if not self.finance_manager:
            portfolio_value = 0.0
            free_cash = 0.0
        else:
            metrics = self.finance_manager.get_performance_metrics()
            portfolio_value = metrics.get('total_capital', 0.0)
            free_cash = metrics.get('free_capital', 0.0)

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'step': self.state['current_step'],
            'market_regime': self.state['market_regime'],
            'risk_level': self.state['current_risk_level'],
            'portfolio_value': portfolio_value,
            'free_cash': free_cash,
            'drawdown': self.state['drawdown'],
            'winrate': self.state['winrate'],
            'consecutive_losses': self.state['consecutive_losses'],
            'last_modulation': self.state.get('last_modulation', {}),
            'total_decisions': len(self.decision_history),
            'total_trades': len(self.trade_history)
        }

    def reset(self) -> None:
        """Réinitialise l'état interne du DBE."""
        # Réinitialisation de l'état
        self.state = {
            'current_step': 0,
            'drawdown': 0.0,
            'winrate': 0.0,
            'volatility': 0.0,
            'market_regime': 'NEUTRAL',
            'last_trade_pnl': 0.0,
            'consecutive_losses': 0,
            'position_duration': 0,
            'current_risk_level': 1.0,
            'max_risk_level': 2.0,
            'min_risk_level': 0.5,
            'last_modulation': {},
            'performance_metrics': {}
        }

        # Réinitialisation des historiques
        self.trade_history = []
        self.decision_history = []
        self.win_rates = []
        self.drawdowns = []
        self.position_durations = []
        self.pnl_history = []
        self.trade_results = []

        # Réinitialisation des paramètres lissés aux valeurs de base
        self.smoothed_params = {
            'sl_pct': self.config.get('risk_parameters', {}).get('base_sl_pct', 0.02),
            'tp_pct': self.config.get('risk_parameters', {}).get('base_tp_pct', 0.04)
        }

        # Réinitialisation du gestionnaire financier si disponible
        if self.finance_manager:
            self.finance_manager.reset()

        logger.info("DBE réinitialisé")

    def reset_for_new_chunk(self, continuity=True):
        """
        Réinitialise le DBE pour un nouveau chunk avec contrôle de continuité.

        Args:
            continuity: Si True, préserve l'historique pour accumulation d'expérience.
                       Si False, fait un reset complet (ancien comportement).
        """
        if continuity:
            logger.info("[DBE CONTINUITY] Garder histoire – Append only, no reset.")

            # Préserver TOUTE l'expérience accumulée (volatility_history, régimes, etc.)
            # Seulement reset des métriques temporaires du step/chunk actuel
            self.state['current_step'] = 0
            self.state['last_trade_pnl'] = 0.0
            self.state['position_duration'] = 0

            # GARDER: volatility_history, regime, trade_history, win_rates, etc.
            # GARDER: smoothed_params, current_risk_level pour continuité

            # Étendre l'historique au lieu de le resetter
            if hasattr(self, 'new_vol_data') and self.new_vol_data:
                if not hasattr(self, 'volatility_history'):
                    self.volatility_history = []
                self.volatility_history.extend(self.new_vol_data)
                self.new_vol_data = []  # Clear new data after appending

            # Garantir que volatility_history existe
            if not hasattr(self, 'volatility_history'):
                self.volatility_history = []

            # Stocker la volatilité actuelle si disponible
            current_vol = self.state.get('volatility', 0.0)
            if current_vol > 0.0 and (not self.volatility_history or self.volatility_history[-1] != current_vol):
                self.volatility_history.append(current_vol)
                logger.debug(f"[VOL ACCUMULATION] Volatilité ajoutée: {current_vol:.4f}")

            logger.info(f"[DBE CONTINUITY] Histoire préservée. Vol history: {len(getattr(self, 'volatility_history', []))} points")

        else:
            # Ancien reset (garder pour hard reset rare)
            logger.info("[DBE FULL RESET] Reset complet de l'expérience")
            self._reset_for_new_chunk_legacy()

    def _reset_for_new_chunk_legacy(self) -> None:
        """
        Réinitialisation complète (ancien comportement) - utilisé seulement pour hard reset.
        """
        self.state['current_step'] = 0
        self.state['last_trade_pnl'] = 0.0
        self.state['consecutive_losses'] = 0
        self.state['position_duration'] = 0
        self.state['volatility'] = 0.0
        self.state['market_regime'] = 'NEUTRAL'
        self.state['trend_strength'] = 0.0

        # Reset complet de l'historique (perte d'expérience)
        if hasattr(self, 'volatility_history'):
            self.volatility_history = []
        if hasattr(self, 'trade_history'):
            self.trade_history = []
        self.current_regime = 'neutral'

        logger.info("🔄 DBE: Reset complet effectué (perte d'expérience)")

    def _adapt_smoothing_factor(self) -> None:
        """
        Adapte le facteur de lissage (smoothing_factor) en fonction des performances récentes.
        - Réduit le lissage (augmente smoothing_factor) si les performances sont bonnes (winrate élevé, faible drawdown).
        - Augmente le lissage (diminue smoothing_factor) si les performances sont mauvaises (winrate faible, drawdown élevé).
        """
        current_winrate = self.state.get('winrate', 0.0)
        current_drawdown = self.state.get('drawdown', 0.0)

        # Paramètres de configuration pour l'adaptation du lissage
        adapt_config = self.config.get('smoothing_adaptation', {
            'min_smoothing': 0.01,
            'max_smoothing': 0.5,
            'winrate_threshold_good': 0.6,
            'winrate_threshold_bad': 0.4,
            'drawdown_threshold_good': 5.0, # in percent
            'drawdown_threshold_bad': 15.0, # in percent
            'adaptation_rate': 0.01
        })

        min_smoothing = adapt_config['min_smoothing']
        max_smoothing = adapt_config['max_smoothing']
        winrate_threshold_good = adapt_config['winrate_threshold_good']
        winrate_threshold_bad = adapt_config['winrate_threshold_bad']
        drawdown_threshold_good = adapt_config['drawdown_threshold_good']
        drawdown_threshold_bad = adapt_config['drawdown_threshold_bad']
        adaptation_rate = adapt_config['adaptation_rate']

        new_smoothing_factor = self.smoothing_factor

        # Ajustement basé sur le winrate
        if current_winrate > winrate_threshold_good:
            new_smoothing_factor += adaptation_rate # Reduce smoothing (faster adaptation)
        elif current_winrate < winrate_threshold_bad:
            new_smoothing_factor -= adaptation_rate # Increase smoothing (slower adaptation)

        # Ajustement basé sur le drawdown
        if current_drawdown < drawdown_threshold_good: # Lower drawdown is good
            new_smoothing_factor += adaptation_rate
        elif current_drawdown > drawdown_threshold_bad: # Higher drawdown is bad
            new_smoothing_factor -= adaptation_rate

        # Clip the smoothing factor to stay within bounds
        self.smoothing_factor = np.clip(new_smoothing_factor, min_smoothing, max_smoothing)
        logger.debug(f"Smoothing factor adapted to: {self.smoothing_factor:.3f} (Winrate: {current_winrate:.2f}, Drawdown: {current_drawdown:.2f})")

    def calculate_trade_parameters(
        self,
        capital: float,
        worker_pref_pct: float,
        tier_config: Optional[Dict[str, Any]] = None,
        current_price: Optional[float] = None,
        asset_volatility: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calcule les paramètres de trade en fonction du capital, des préférences du worker
        et de la configuration du palier.
        """
        logger.debug(f"[DBE_CALC] Entrée: capital={capital:.2f}, worker_pref={worker_pref_pct:.2f}, tier_config?={tier_config is not None}, price={current_price}, vol={asset_volatility}")
        try:
            if not tier_config or not isinstance(tier_config, dict):
                logger.warning("[DBE_CALC] Échec: tier_config est manquant ou n'est pas un dictionnaire.")
                return {'feasible': False, 'reason': 'Configuration de palier manquante ou invalide'}

            logger.debug(f"[DBE_CALC] tier_config reçu: {tier_config}")

            risk_params = self.compute_dynamic_modulation()
            aggressivity = risk_params.get('aggressivity', 0.5)
            logger.debug(f"[DBE_CALC] Agressivité calculée: {aggressivity:.2f}")

            exposure_range = tier_config.get('exposure_range')
            if not exposure_range or not isinstance(exposure_range, list) or len(exposure_range) != 2:
                logger.warning(f"[DBE_CALC] Échec: 'exposure_range' invalide ou manquant dans le palier {tier_config.get('name')}. Reçu: {exposure_range}")
                return {'feasible': False, 'reason': f"exposure_range invalide pour le palier {tier_config.get('name')}"}

            logger.debug(f"[DBE_CALC] Palier '{tier_config.get('name')}': exposure_range={exposure_range}")

            min_position_pct = exposure_range[0] / 100.0
            max_position_pct = exposure_range[1] / 100.0
            logger.debug(f"[DBE_CALC] Intervalle d'exposition: min={min_position_pct:.2%}, max={max_position_pct:.2%}")

            position_pct = min_position_pct + (max_position_pct - min_position_pct) * aggressivity
            logger.debug(f"[DBE_CALC] Taille de position (basée sur l'agressivité): {position_pct:.2%}")

            max_position = tier_config.get('max_position_size_pct', 0.5) / 100.0
            position_pct = min(max_position, position_pct)
            logger.debug(f"[DBE_CALC] Taille de position (après contrainte max du palier de {max_position:.2%}): {position_pct:.2%}")

            position_size_usdt = capital * position_pct
            logger.debug(f"[DBE_CALC] Taille de position (en USDT): {position_size_usdt:.2f}")

            min_trade_value = tier_config.get('min_trade_value', 11.0)
            if position_size_usdt < min_trade_value:
                logger.warning(f"[DBE_CALC] Taille calculée ({position_size_usdt:.2f} USDT) < min_trade_value ({min_trade_value} USDT). Ajustement.")
                if capital < min_trade_value:
                    logger.warning(f"[DBE_CALC] Échec: Capital ({capital:.2f} USDT) insuffisant pour le trade minimum de {min_trade_value} USDT.")
                    return {'feasible': False, 'reason': f'Capital insuffisant (min {min_trade_value} USDT requis)'}
                position_pct = min_trade_value / capital
                position_size_usdt = min_trade_value
                logger.debug(f"[DBE_CALC] Taille de position ajustée au minimum: {position_pct:.2%} ({position_size_usdt:.2f} USDT)")

            sl_pct = risk_params.get('sl_pct', 0.02)
            tp_pct = risk_params.get('tp_pct', sl_pct * 2)

            # Log the decision with the final calculated position size
            snapshot = DBESnapshot(
                step=self.state['current_step'],
                market_regime=risk_params.get('regime', 'neutral'),
                risk_level=risk_params.get('risk_level', 1.0),
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                position_size_pct=position_pct, # Use the calculated value
                reward_boost=risk_params.get('reward_boost', 1.0),
                penalty_inaction=risk_params.get('penalty_inaction', 0.0),
                metrics=self.state.get('performance_metrics', {}).copy()
            )
            self._log_decision(snapshot, risk_params)

            return {
                'feasible': True,
                'position_size_pct': position_pct,
                'position_size_usdt': position_size_usdt,
                'sl_pct': sl_pct,
                'tp_pct': tp_pct,
                'aggressivity': aggressivity,
                'risk_level': risk_params.get('risk_level', 1.0),
                'capital': capital,
                'tier': tier_config.get('name', 'unknown'),
                'regime': risk_params.get('regime', 'neutral'),
                'volatility': risk_params.get('volatility', 0.0),
                'risk_per_trade_pct': tier_config.get('risk_per_trade_pct', 0.01)
            }

        except Exception as e:
            logger.error(f"Erreur dans calculate_trade_parameters: {e}", exc_info=True)
            return {'feasible': False, 'reason': f'Erreur de calcul: {str(e)}'}

    def check_reset_conditions(self, worker_id: str) -> Tuple[bool, str]:
        """
        Vérifie les conditions de full reset pour un worker.

        Args:
            worker_id: Identifiant du worker

        Returns:
            Tuple[bool, str]: (True si reset nécessaire, raison du reset)
        """
        # Initialiser l'état du worker s'il n'existe pas
        if worker_id not in self.worker_states:
            self.worker_states[worker_id] = {
                'initial_capital': self.finance_manager.get_balance(worker_id) if self.finance_manager else 0.0,
                'cumulative_loss': 0.0,
                'last_trade_ts': None,
                'consecutive_losses': 0,
                'trade_history': []
            }

        state = self.worker_states[worker_id]

        # 1) Capital total < MIN_TRADE
        if state['initial_capital'] < self.MIN_TRADE:
            return True, f"capital_below_min_trade ({state['initial_capital']:.2f} < {self.MIN_TRADE})"

        # 2) Position invendable + solde insuffisant
        any_untradable = any(p.get('value', 0) < self.MIN_TRADE for p in state['trade_history'])
        if any_untradable and state['initial_capital'] < self.MIN_TRADE:
            return True, f"untradable_position_and_low_cash (pos<{self.MIN_TRADE} and cash<{self.MIN_TRADE})"

        # 3) Cumulative loss >= palier max_drawdown
        if hasattr(self, 'determine_tier'):
            tier = self.determine_tier(state['initial_capital'])
            if tier and 'max_drawdown' in tier and state['cumulative_loss'] >= tier['max_drawdown']:
                return True, f"max_drawdown_reached (loss: {state['cumulative_loss']:.2f} >= {tier['max_drawdown']})"

        # 4) Vérifier d'autres conditions de reset si nécessaire
        # ...

        # Aucune condition de reset détectée
        return False, ""

    def perform_full_reset(self, worker_id: str, restore_capital: float = None) -> None:
        """
        Effectue un reset complet du worker.

        Args:
            worker_id: Identifiant du worker
            restore_capital: Montant de capital à restaurer (optionnel)
        """
        if worker_id not in self.worker_states:
            self.worker_states[worker_id] = {}

        state = self.worker_states[worker_id]

        # 1) Fermer les positions ouvertes
        if self.finance_manager:
            try:
                positions = self.finance_manager.get_open_positions(worker_id)
                for pos in positions:
                    try:
                        self.finance_manager.force_close_position(
                            worker_id,
                            pos.get('symbol', ''),
                            pos.get('qty', 0)
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Échec de la fermeture forcée pour {worker_id} {pos.get('symbol', '')}: {str(e)}"
                        )
            except Exception as e:
                self.logger.error(f"Erreur lors de la récupération des positions: {str(e)}")

        # 2) Définir le nouveau capital
        if restore_capital is None:
            # Utiliser la valeur par défaut de la configuration ou 20.0 USDT
            restore_capital = self.config.get('default_reset_capital', 20.0)
            if isinstance(restore_capital, dict):
                restore_capital = restore_capital.get(worker_id, 20.0)

        if self.finance_manager:
            self.finance_manager.set_balance(worker_id, restore_capital)

        # 3) Réinitialiser l'état du worker (mode partiel pour conserver la mémoire longue)
        # On ne réinitialise que le capital et les compteurs d'épisode, pas l'historique.
        state.update({
            'initial_capital': restore_capital,
            'last_trade_ts': None,
            'consecutive_losses': 0
        })
        # NOTE: 'cumulative_loss' et 'trade_history' sont intentionnellement conservés
        # pour permettre au DBE d'apprendre des échecs passés.

        self.logger.info(f"[RESET PARTIEL] Worker {worker_id} -> capital restauré à {restore_capital:.2f} USDT. Mémoire des erreurs conservée.")

    def reset_flow(self, worker_id: str) -> bool:
        """
        Vérifie les conditions de reset et effectue un reset si nécessaire.

        Args:
            worker_id: Identifiant du worker

        Returns:
            bool: True si un reset a été effectué, False sinon
        """
        should_reset, reason = self.check_reset_conditions(worker_id)
        if should_reset:
            self.logger.warning(f"[RESET] Condition de reset détectée pour {worker_id}: {reason}")
            self.perform_full_reset(worker_id)
            return True
        return False

    def on_trade_closed(self, trade_result: Dict[str, Any]) -> None:
        """
        Met à jour l'état après la fermeture d'un trade.

        Args:
            trade_result: Résultat du trade fermé
        """
        worker_id = trade_result.get('worker_id')
        if not worker_id or worker_id not in self.worker_states:
            return

        state = self.worker_states[worker_id]

        # Mettre à jour l'historique des trades
        state['last_trade_ts'] = datetime.utcnow()
        state['trade_history'].append(trade_result)

        # Mettre à jour les pertes cumulées
        if 'pnl' in trade_result and trade_result['pnl'] < 0:
            state['cumulative_loss'] += abs(trade_result['pnl'])
            state['consecutive_losses'] += 1
        else:
            state['consecutive_losses'] = 0

    def __del__(self):
        """Nettoyage à la destruction de l'instance."""
        status = self.get_status()
        return (
            f"DBE Status (Step: {status['step']})\n"
            f"Portfolio: ${status['portfolio']['total_value']:,.2f} "
            f"(Return: {status['portfolio']['total_return_pct']:.2f}%)\n"
            f"Trades: {status['trading']['total_trades']} "
            f"(Win Rate: {status['trading']['win_rate']:.1f}%)\n"
            f"Risk: {status['risk']['current_risk_level']:.2f} "
            f"(Regime: {status['risk']['market_regime']})\n"
            f"Drawdown: {status['risk']['current_drawdown']:.2f}% | "
            f"Volatility: {status['risk']['volatility']:.4f}"
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration actuelle du DBE.

        Returns:
            Dictionnaire de configuration
        """
        # Retourne une copie pour éviter les modifications accidentelles
        return self.config.copy()

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Met à jour la configuration du DBE.

        Args:
            new_config: Dictionnaire contenant les nouvelles valeurs de configuration
        """
        # Mise à jour récursive de la configuration
        def deep_update(current: Dict[str, Any], new: Dict[str, Any]) -> None:
            for key, value in new.items():
                if key in current and isinstance(current[key], dict) and isinstance(value, dict):
                    deep_update(current[key], value);
                else:
                    current[key] = value

        # Application de la mise à jour
        deep_update(self.config, new_config)
        logger.info("Configuration du DBE mise à jour")

        # Mise à jour du niveau de log si nécessaire
        if 'logging' in new_config and 'log_level' in new_config['logging']:
            log_level = new_config['logging']['log_level'].upper()
            logging.getLogger().setLevel(getattr(logging, log_level))

    def __del__(self):
        if hasattr(self, 'dbe_log_file') and not self.dbe_log_file.closed:
            self.dbe_log_file.close()

    def _compute_risk_parameters(self, state: Dict[str, Any], mod: Dict[str, Any]) -> None:
        """
        Calcule les paramètres de risque dynamiques (SL/TP).

        Args:
            state: Dictionnaire contenant l'état actuel
            mod: Dictionnaire à mettre à jour avec les nouveaux paramètres de risque
        """
        try:
            if state is None or mod is None:
                logger.warning("State ou mod est None dans _compute_risk_parameters")
                return

            # Récupération des configurations
            risk_cfg = self.config.get('risk_parameters', {})
            regime_params = self.config.get('regime_parameters', {}).get(self.current_regime, {})

            # Paramètres de base
            base_sl = float(risk_cfg.get('base_sl_pct', 0.02))
            base_tp = float(risk_cfg.get('base_tp_pct', 0.04))

            # Initialisation des valeurs d'état avec des valeurs par défaut si manquantes
            current_drawdown = float(state.get('drawdown', 0.0))
            volatility = float(state.get('volatility', 0.0))
            win_rate = float(state.get('win_rate', 0.5))  # 50% par défaut
            sharpe_ratio = float(state.get('sharpe_ratio', 0.0))
            current_step = int(state.get('current_step', 0))

            # ACCUMULATION VOLATILITÉ: Stocker dans l'historique
            if volatility > 0.0:
                if not hasattr(self, 'volatility_history'):
                    self.volatility_history = []
                # Éviter les doublons
                if not self.volatility_history or abs(self.volatility_history[-1] - volatility) > 1e-6:
                    self.volatility_history.append(volatility)
                    logger.debug(f"[VOL HISTORY] Ajouté: {volatility:.4f}, Total: {len(self.volatility_history)} points")

            # Récupération des multiplicateurs spécifiques au régime
            sl_multiplier = float(regime_params.get('sl_multiplier', 1.0))
            tp_multiplier = float(regime_params.get('tp_multiplier', 1.0))

            # 1. Ajustement basé sur le drawdown
            max_drawdown = float(risk_cfg.get('max_drawdown', 0.1))  # 10% par défaut
            drawdown_factor = 1.0 - min(current_drawdown / (max_drawdown * 2), 0.5)  # Réduction jusqu'à 50%

            # 2. Ajustement basé sur la volatilité
            vol_management = self.config.get('volatility_management', {})
            min_vol = float(vol_management.get('min_volatility', 0.01))
            max_vol = float(vol_management.get('max_volatility', 0.20))
            vol_factor = 1.0 - ((volatility - min_vol) / (max_vol - min_vol + 1e-6)) * 0.5  # Réduction jusqu'à 50%

            # 3. Ajustement basé sur le win rate
            target_win_rate = 0.6  # Cible de 60% de trades gagnants
            win_rate_factor = (win_rate / target_win_rate) ** 2  # Effet non linéaire

            # 4. Ajustement basé sur le ratio de Sharpe
            sharpe_factor = 1.0 + (max(0, sharpe_ratio) / 2.0)  # Améliore le risque avec un meilleur Sharpe

            # Calcul des nouveaux paramètres avec contraintes
            min_sl = float(risk_cfg.get('min_sl_pct', 0.005))  # 0.5% minimum
            max_sl = float(risk_cfg.get('max_sl_pct', 0.10))    # 10% maximum
            min_tp = float(risk_cfg.get('min_tp_pct', 0.01))    # 1% minimum
            max_tp = float(risk_cfg.get('max_tp_pct', 0.20))    # 20% maximum

            # Calcul des nouvelles valeurs
            new_sl = base_sl * sl_multiplier * drawdown_factor * vol_factor
            new_tp = base_tp * tp_multiplier * win_rate_factor * sharpe_factor

            # Application des limites
            new_sl = max(min_sl, min(max_sl, new_sl))
            new_tp = max(min_tp, min(max_tp, new_tp))

            # Vérification de l'initialisation de smoothed_params
            if not hasattr(self, 'smoothed_params'):
                self.smoothed_params = {
                    'sl_pct': base_sl,
                    'tp_pct': base_tp,
                    'position_size': 0.1,
                    'risk_level': 1.0
                }

            # Application du lissage exponentiel
            smoothing = self.config.get('smoothing', {}).get('adaptation_rate', 0.1)

            # Mise à jour des paramètres lissés
            for param, new_val in [('sl_pct', new_sl), ('tp_pct', new_tp)]:
                if param in self.smoothed_params:
                    self.smoothed_params[param] = (
                        (1.0 - smoothing) * self.smoothed_params[param] +
                        smoothing * new_val
                    )
                else:
                    self.smoothed_params[param] = new_val

            # Calcul du coefficient d'agressivité (0-1) basé sur plusieurs facteurs
            # 1. Facteur de confiance (winrate récent)
            winrate_factor = min(1.0, max(0.0, (win_rate - 0.4) / 0.6))  # 0% à 100% pour winrate de 0.4 à 1.0

            # 2. Facteur de drawdown (pénalise les périodes de pertes)
            drawdown_factor = max(0.0, 1.0 - (current_drawdown / 0.1))  # 100% à 0% pour drawdown de 0% à 10%

            # 3. Facteur de volatilité (pénalise la volatilité élevée)
            volatility_factor = 1.0 - min(1.0, volatility * 5)  # 100% à 0% pour volatilité de 0% à 20%

            # 4. Facteur de régime de marché
            regime_factors = {
                'bull': 1.0,
                'bear': 0.3,
                'volatile': 0.5,
                'sideways': 0.7,
                'neutral': 0.8
            }
            regime_factor = regime_factors.get(self.current_regime.lower(), 0.5)

            # Calcul final du coefficient d'agressivité (0-1)
            aggressivity = (winrate_factor * 0.4 +
                          drawdown_factor * 0.3 +
                          volatility_factor * 0.2 +
                          regime_factor * 0.1)

            # Lissage du coefficient d'agressivité
            if 'aggressivity' not in self.smoothed_params:
                self.smoothed_params['aggressivity'] = 0.5  # Valeur par défaut

            smoothing = self.config.get('smoothing', {}).get('adaptation_rate', 0.1)
            self.smoothed_params['aggressivity'] = (
                (1.0 - smoothing) * self.smoothed_params['aggressivity'] +
                smoothing * aggressivity
            )

            # Mise à jour du dictionnaire de sortie avec les valeurs lissées
            mod.update({
                'sl_pct': self.smoothed_params.get('sl_pct', base_sl),
                'tp_pct': self.smoothed_params.get('tp_pct', base_tp),
                'risk_level': self.state.get('current_risk_level', 1.0),
                'regime': self.current_regime,
                'volatility': volatility,
                'aggressivity': self.smoothed_params['aggressivity']
            })

            # Journalisation des changements importants (tous les 50 pas)
            if current_step > 0 and current_step % 50 == 0:
                logger.info(
                    f"🔧 Paramètres de risque - "
                    f"Régime: {self.current_regime.upper()} | "
                    f"Drawdown: {current_drawdown:.2f}% | "
                    f"Volatilité: {volatility:.2f}% | "
                    f"Win Rate: {win_rate:.1f}% | "
                    f"Sharpe: {sharpe_ratio:.2f}\n"
                    f"SL: {new_sl:.2f}% (lissé: {mod['sl_pct']:.2f}%) | "
                    f"TP: {new_tp:.2f}% (lissé: {mod['tp_pct']:.2f}%) | "
                    f"Niveau de risque: {mod['risk_level']:.2f}"
                )

        except Exception as e:
            logger.error(f"Erreur dans _compute_risk_parameters: {str(e)}", exc_info=True)
            # En cas d'erreur, on utilise les valeurs par défaut
            mod.update({
                'sl_pct': risk_cfg.get('base_sl_pct', 0.02),
                'tp_pct': risk_cfg.get('base_tp_pct', 0.04),
                'risk_level': 1.0,
                'regime': self.current_regime,
                'volatility': 0.0
            })

    def _compute_reward_modulation(self, mod: Dict[str, Any]) -> None:
        """Calcule la modulation des récompenses."""
        # Paramètres configurables
        reward_config = self.config.get('reward', {})
        winrate_threshold = reward_config.get('winrate_threshold', 0.55)
        max_boost = reward_config.get('max_boost', 2.0)

        # Reward boost basé sur le winrate
        if self.state.get('winrate', 0.0) > winrate_threshold:
            boost_factor = min(
                max_boost,
                1.0 + (self.state['winrate'] - winrate_threshold) * 5.0
            )
            mod['reward_boost'] = boost_factor
        else:
            mod['reward_boost'] = 1.0

        # Pénalité d'inaction progressive
        inaction_factor = reward_config.get('inaction_factor', 0.1)
        action_freq = self.state.get('action_frequency', 1.0) # Default to 1 to avoid penalty if not present
        min_action_freq = reward_config.get('min_action_frequency', 0.1)

        if action_freq < min_action_freq and self.state.get('market_regime') in ['BULL', 'BEAR']:
            # Pénalité progressive basée sur la fréquence d'action
            mod['penalty_inaction'] = -inaction_factor * (min_action_freq - action_freq) * 10
        else:
            mod['penalty_inaction'] = 0.0

    def _compute_position_sizing(self, mod: Dict[str, Any]) -> None:
        """
        Calcule la taille de position dynamique.

        Args:
            mod: Dictionnaire des paramètres modulés à mettre à jour
        """
        sizing_cfg = self.config.get('position_sizing', {})
        base_size = sizing_cfg.get('base_position_size', 0.1)  # 10% par défaut

        # Ajustement basé sur la confiance (winrate récent)
        confidence_factor = min(2.0, max(0.5, self.state['winrate'] / 0.5))  # 0.5-2.0x

        # Ajustement basé sur le drawdown
        drawdown_factor = 1.0 - (self.state['drawdown'] / 100.0 * 2)  # Réduit la taille avec le drawdown

        # Ajustement basé sur la volatilité
        vol_factor = 1.0 / (1.0 + self.state['volatility'] * 10)  # Réduit la taille avec la volatilité

        # Calcul final avec limites
        mod['position_size_pct'] = max(
            sizing_cfg.get('min_position_size', 0.01),
            min(
                sizing_cfg.get('max_position_size', 0.3),
                base_size * confidence_factor * drawdown_factor * vol_factor
            )
        )

    def _compute_risk_mode(self, mod: Dict[str, Any]) -> None:
        """
        Détermine le mode de risque global (DEFENSIVE, NORMAL, AGGRESSIVE).

        Args:
            mod: Dictionnaire des paramètres modulés à mettre à jour
        """
        # Mode défensif si drawdown élevé ou pertes consécutives
        if self.state['drawdown'] > 10.0 or self.state['consecutive_losses'] >= 3:
            mod['risk_mode'] = "DEFENSIVE"
            mod['position_size_pct'] *= 0.5 # Reduce position size
            mod['sl_pct'] *= 1.2 # Tighten stop loss
        # Mode agressif si bonnes performances et faible drawdown
        elif self.state['winrate'] > 0.7 and self.state['drawdown'] < 2.0:
            mod['risk_mode'] = "AGGRESSIVE"
            mod['position_size_pct'] *= 1.2 # Increase position size
            mod['tp_pct'] *= 1.2 # Loosen take profit
        else:
            mod['risk_mode'] = "NORMAL"

    def _apply_market_regime_modifiers(self, mod: Dict[str, Any]) -> None:
        """
        Applique des ajustements spécifiques au régime de marché.

        Args:
            mod: Dictionnaire des paramètres modulés à mettre à jour
        """
        regime = self.state.get('market_regime', 'NORMAL')
        regime_cfg = self.config.get('modes', {}).get(regime.lower(), {})

        if not regime_cfg:
            return

        mod['position_size_pct'] *= regime_cfg.get('position_size_multiplier', 1.0)
        mod['sl_pct'] *= regime_cfg.get('sl_multiplier', 1.0)
        mod['tp_pct'] *= regime_cfg.get('tp_multiplier', 1.0)

        # Specific adjustments for trending markets
        if regime == "BULL" or regime == "BEAR":
            mod['trailing_stop'] = True # Activate trailing stop in trending markets

    def reset_chunk(self) -> None:
        """Réinitialise les métriques au début d'un nouveau chunk."""
        # Conserver certaines métriques (comme le winrate) mais réinitialiser les autres
        self.state.update({
            'current_step': 0,
            'chunk_optimal_pnl': 0.0,
            'position_size_pct': self.config.get('position_sizing', {}).get('base_position_size', 0.1)
        })
        logger.info("🔄 DBE: Nouveau chunk - réinitialisation des métriques")

    def _log_dbe_state(self, modulation: Dict[str, Any]) -> None:
        """
        Logs the current state and modulation of the DBE to a JSONL file.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": self.state.get('current_step', 0),
            "drawdown": self.state.get('drawdown', 0.0),
            "winrate": self.state.get('winrate', 0.0),
            "volatility": self.state.get('volatility', 0.0),
            "market_regime": self.state.get('market_regime', 'NORMAL'),
            "sl_pct": modulation.get('sl_pct', 0.0),
            "tp_pct": modulation.get('tp_pct', 0.0),
            "reward_boost": modulation.get('reward_boost', 0.0),
            "penalty_inaction": modulation.get('penalty_inaction', 0.0),
            "position_size_pct": modulation.get('position_size_pct', 0.0),
            "risk_mode": modulation.get('risk_mode', 'NORMAL')
        }
        try:
            self.dbe_log_file.write(json.dumps(log_entry, cls=NpEncoder) + '\n')
            self.dbe_log_file.flush() # Ensure data is written to disk immediately
        except Exception as e:
            logger.error(f"Error writing to DBE log file: {e}")
