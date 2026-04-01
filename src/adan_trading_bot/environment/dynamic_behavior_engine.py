import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import yaml

# ------------------------------------------------------------------
# SOTA 2025: Gaussian HMM for statistical regime detection
# ------------------------------------------------------------------
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

logger = logging.getLogger(__name__)

# Number of hidden states for the Markov regime model
N_HMM_STATES = 3
# Minimum observations required to fit the HMM
HMM_MIN_OBS = 30
# Rolling window size for HMM input
HMM_WINDOW = 120


@dataclass
class DBESnapshot:
    """Snapshot des décisions du DBE pour logging et historique."""
    step: int
    market_regime: str
    risk_level: float
    sl_pct: float
    tp_pct: float
    position_size_pct: float
    reward_boost: float = 1.0
    penalty_inaction: float = 0.0
    metrics: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.metrics is None:
            self.metrics = {}

class DynamicBehaviorEngine:
    def __init__(self, config: Dict[str, Any] = None, worker_id: int = 0, **kwargs):
        """Initialise le DynamicBehaviorEngine."""
        self.config = config or {}
        self.worker_id = worker_id
        self.env = None
        self.current_regime = "sideways"
        self.state = {"current_step": 0}
        self.logger = logging.getLogger(f"dbe.{self.__class__.__name__}")
        self.finance_manager = kwargs.get("finance_manager")
        self.smart_logger = kwargs.get("smart_logger")
        self._active_hunts = {}
        self.decision_history = []
        self.trade_history = []

    def log_info(self, message, step=None):
        """Log un message avec le système intelligent SmartLogger."""
        if self.smart_logger:
            try:
                self.smart_logger.smart_info(logger, message, step)
            except Exception:
                logger.info(message)
        else:
            logger.info(message)

        # Initialisation des paramètres de trading (suite)
        # NOTE: This block is legacy init code that runs inside log_info.
        # Using .get() with safe defaults to prevent KeyError.
        if not hasattr(self, '_legacy_init_done'):
            self._legacy_init_done = True
            self.current_position_size_multiplier = 1.0

            pos_sizing = self.config.get("position_sizing", {})

            self.max_position_size = pos_sizing.get("max_position_size", 0.9)

            # Paramètres de lissage
            self.smoothing_factor = self.config.get("smoothing", {}).get(
                "initial_factor", 0.1
            )
            self.smoothed_params = {
                "sl_pct": pos_sizing.get("initial_sl_pct", 0.02),
                "tp_pct": pos_sizing.get("initial_tp_pct", 0.04),
                "position_size": pos_sizing.get(
                    "initial_position_size", 0.1
                ),
                "risk_level": 1.0,
            }

            # Configuration de fréquence des positions
            self.frequency_config = self.config.get("trading_rules", {}).get(
                "frequency", {}
            )

        # Initialisation du logger personnalisé
        self.logger = logging.getLogger(f"dbe.{self.__class__.__name__}")

        # Initialisation des historiques
        self.decision_history = []  # Historique des décisions prises
        self.trade_history = []  # Historique des trades

        # Initialisation des états des workers
        self.worker_states = {}

        # Initialisation de l'état avec gestion d'erreur
        try:
            self.state = {
                "current_step": 0,
                "last_trade_step": 0,
                "consecutive_losses": 0,
                "consecutive_wins": 0,
                "last_win": False,
                "last_reward": 0.0,
                "drawdown": 0.0,
                "current_risk_level": 1.0,  # Niveau de risque initial (1.0 = neutre)
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "volatility": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "recovery_factor": 0.0,
                "expectancy": 0.0,
                "avg_trade": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "equity_curve": [],
                "returns": [],
                "drawdowns": [],
                "position_duration": 0,  # Durée de la position actuelle en pas de temps
                "market_conditions": {},
                "performance_metrics": {  # Ajout des métriques de performance initiales
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "max_drawdown": 0.0,
                    "volatility": 0.0,
                    "avg_trade": 0.0,
                    "expectancy": 0.0,
                },
                # Champs additionnels requis par le code
                "market_regime": "NEUTRAL",
                "winrate": 0.0,
                "last_trade_pnl": 0.0,
                "trend_strength": 0.0,
                "last_modulation": {},
            }
        except Exception as e:
            logger.error(f"Error initializing DBE state: {e}")
            # Fallback: initialize minimal state
            self.state = {
                "current_step": 0,
                "market_regime": "NEUTRAL",
                "current_risk_level": 1.0,
                "winrate": 0.0,
                "win_rate": 0.0,
                "drawdown": 0.0,
                "volatility": 0.0,
                "consecutive_losses": 0,
                "position_duration": 0,
                "last_trade_pnl": 0.0,
                "trend_strength": 0.0,
                "last_modulation": {},
                "performance_metrics": {},
            }

        # Chargement de la configuration externe si elle existe
        dbe_config_path = (
            Path(__file__).parent.parent.parent.parent / "config" / "dbe_config.yaml"
        )
        if dbe_config_path.exists():
            with open(dbe_config_path, "r") as f:
                dbe_config = yaml.safe_load(f) or {}
                if dbe_config:
                    self.config = self._merge_configs(self.config, dbe_config)

    def _merge_configs(
        self,
        result: Dict[str, Any],
        update: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries recursively.

        Args:
            result: Base configuration dictionary
            update: Configuration dictionary to merge into result

        Returns:
            Merged configuration dictionary
        """
        result = result.copy()
        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def update_state(self, live_metrics: Dict[str, Any]) -> None:
        """
        Update the DBE state with live metrics from the environment.

        Args:
            live_metrics: Dictionary containing current market and portfolio metrics
        """
        try:
            # Ensure state exists, but don't reinitialize if it already exists
            if not hasattr(self, "state") or self.state is None:
                logger.info(
                    f"[DBE Worker {self.worker_id}] Initializing DBE state for the first time"
                )
                self.state = {
                    "current_step": 0,
                    "market_regime": "NEUTRAL",
                    "current_risk_level": 1.0,
                    "winrate": 0.0,
                    "win_rate": 0.0,
                    "drawdown": 0.0,
                    "volatility": 0.0,
                    "consecutive_losses": 0,
                    "position_duration": 0,
                    "last_trade_pnl": 0.0,
                    "trend_strength": 0.0,
                    "last_modulation": {},
                    "performance_metrics": {},
                    "initialized": True,
                    "initialization_time": time.time(),
                }
            else:
                logger.debug(
                    f"[DBE Worker {self.worker_id}] State already exists, updating..."
                )

            # Increment step counter
            self.state["current_step"] += 1

            # Update market data
            if "rsi" in live_metrics:
                self.state["rsi"] = live_metrics["rsi"]
            if "adx" in live_metrics:
                self.state["adx"] = live_metrics["adx"]
            if "volatility" in live_metrics:
                self.state["volatility"] = live_metrics["volatility"]

            # Update portfolio metrics
            if "win_rate" in live_metrics:
                self.state["win_rate"] = live_metrics["win_rate"]
                self.state["winrate"] = live_metrics["win_rate"]
            if "drawdown" in live_metrics:
                self.state["drawdown"] = live_metrics["drawdown"]
            if "current_drawdown" in live_metrics:
                self.state["drawdown"] = live_metrics["current_drawdown"]
            if "max_drawdown" in live_metrics:
                self.state["max_drawdown"] = live_metrics["max_drawdown"]
            if "sharpe_ratio" in live_metrics:
                self.state["sharpe_ratio"] = live_metrics["sharpe_ratio"]
            if "sortino_ratio" in live_metrics:
                self.state["sortino_ratio"] = live_metrics["sortino_ratio"]

        except Exception as e:
            logger.error(f"Error in update_state: {e}")

    # ------------------------------------------------------------------
    # SOTA 2025: HMM-based probabilistic regime detection
    # ------------------------------------------------------------------
    def _init_hmm(self) -> None:
        """Lazy-initialise the Gaussian HMM model and observation buffer."""
        if not hasattr(self, '_hmm_model'):
            self._hmm_fitted = False
            self._hmm_obs_buffer: list = []  # list of [log_return, volatility]
            if HMM_AVAILABLE:
                self._hmm_model = GaussianHMM(
                    n_components=N_HMM_STATES,
                    covariance_type="full",
                    n_iter=50,
                    random_state=42,
                    tol=0.01,
                )
            else:
                self._hmm_model = None
            # Last probability vector (uniform prior)
            self._hmm_probs = np.ones(N_HMM_STATES, dtype=np.float32) / N_HMM_STATES

    def _update_hmm(self, log_return: float, volatility: float) -> np.ndarray:
        """Feed a new observation and return the state-probability vector.

        Returns:
            np.ndarray of shape (3,) summing to 1.0.
        """
        self._init_hmm()
        self._hmm_obs_buffer.append([log_return, volatility])

        # Keep a rolling window
        if len(self._hmm_obs_buffer) > HMM_WINDOW:
            self._hmm_obs_buffer = self._hmm_obs_buffer[-HMM_WINDOW:]

        if self._hmm_model is None or len(self._hmm_obs_buffer) < HMM_MIN_OBS:
            return self._hmm_probs.copy()

        try:
            X = np.array(self._hmm_obs_buffer, dtype=np.float64)
            # Re-fit periodically (every 20 new obs) for adaptivity
            if not self._hmm_fitted or len(self._hmm_obs_buffer) % 20 == 0:
                self._hmm_model.fit(X)
                self._hmm_fitted = True

            # Posterior probability of last observation
            posteriors = self._hmm_model.predict_proba(X)
            self._hmm_probs = posteriors[-1].astype(np.float32)
        except Exception as e:
            logger.debug(f"HMM update fallback: {e}")

        return self._hmm_probs.copy()

    def get_regime_probabilities(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Public API: return the 3-state HMM probability vector.

        This vector is intended to be injected into the context_vector
        for the FiLM conditioning layer.
        """
        self._init_hmm()
        close = market_data.get("close", 0.0)
        prev_close = market_data.get("prev_close", close)
        volatility = market_data.get("volatility", 0.0)

        if prev_close > 0 and close > 0:
            log_ret = float(np.log(close / prev_close))
        else:
            log_ret = 0.0

        return self._update_hmm(log_ret, volatility)

    def detect_market_regime(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """Detect market regime using the Gaussian HMM posterior.

        Returns a (regime_label, confidence) tuple for backward compatibility.
        The regime label is derived from the highest-probability hidden state.
        State mapping (sorted by mean log-return after fit):
            - State with highest mean return → 'bull'
            - State with lowest  mean return → 'bear'
            - Middle state                   → 'sideways'
        If the HMM is not yet fitted, falls back to simple heuristics.
        """
        probs = self.get_regime_probabilities(market_data)
        best_state = int(np.argmax(probs))
        confidence = float(probs[best_state])

        # Derive human-readable label from HMM means if fitted
        if hasattr(self, '_hmm_model') and self._hmm_model is not None and self._hmm_fitted:
            try:
                means = self._hmm_model.means_[:, 0]  # log-return means
                order = np.argsort(means)  # ascending: bear, sideways, bull
                labels = {int(order[0]): 'bear', int(order[1]): 'sideways', int(order[2]): 'bull'}
                regime = labels.get(best_state, 'sideways')
            except Exception:
                regime = ['bear', 'sideways', 'bull'][best_state]
        else:
            # Heuristic fallback while HMM is warming up
            adx = market_data.get("adx", 0)
            ema_fast = market_data.get("ema_fast", 0)
            ema_slow = market_data.get("ema_slow", 0)
            if adx > 25:
                regime = "bull" if ema_fast > ema_slow else "bear"
                confidence = 0.7 + 0.3 * (adx / 100.0)
            else:
                regime = "sideways"
                confidence = 0.9

        worker_key = getattr(self, "worker_key", f"w{self.worker_id}")
        logger.debug(
            f"[HMM_REGIME] Worker={worker_key} | probs={probs} | "
            f"regime={regime} (conf={confidence:.2f})"
        )

        return regime, confidence

    def _get_capital_tier(self, portfolio_value: float) -> Optional[Dict[str, Any]]:
        """Determines the capital tier based on the portfolio value."""
        capital_tiers = self.config.get("capital_tiers")
        if not capital_tiers or not isinstance(capital_tiers, list):
            logger.warning("Configuration 'capital_tiers' manquante ou invalide.")
            return None
        for tier in capital_tiers:
            min_capital = tier.get("min_capital", 0)
            max_capital = tier.get("max_capital")
            if max_capital is None:  # For the highest tier, max_capital can be null
                max_capital = float("inf")

            if min_capital <= portfolio_value < max_capital:
                logger.debug(
                    f"Capital {portfolio_value:.2f} USDT correspond au palier: {tier.get('name')}"
                )
                return tier
        logger.warning(
            f"Aucun palier de capital trouvé pour un portefeuille de {portfolio_value:.2f} USDT."
        )
        return None

    def get_capital_tier(self, portfolio_value: float) -> Optional[Dict[str, Any]]:
        """Public alias for _get_capital_tier (OMEGA compatibility)."""
        return self._get_capital_tier(portfolio_value)

    def update_risk_parameters(
        self,
        market_data: Dict[str, Any],
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Met à jour les paramètres de risque en respectant les capital_tiers et la modulation de régime simple.
        """
        try:
            # 1. Déterminer le palier de capital
            tier = self._get_capital_tier(portfolio_value)
            if not tier:
                raise ValueError("Impossible de déterminer le palier de capital.")
            
            tier_limit_pct = tier.get("max_position_size_pct", 90.0) / 100.0

            # 2. Déterminer le régime de marché
            regime, confidence = self.detect_market_regime(market_data)
            self.current_regime = regime

            # 3. Récupérer la taille de position de base du worker
            # Note: cette valeur n'est pas directement dans la config worker, on prend une valeur par défaut
            base_position_pct = self.config.get("risk_parameters", {}).get("initial_position_size", 0.7)

            # 4. Appliquer la modulation de régime (+/- 10%)
            regime_multipliers = {'bull': 1.10, 'bear': 0.90}
            regime_mult = regime_multipliers.get(regime, 1.0)
            modulated_size_pct = base_position_pct * regime_mult
            
            # 5. Appliquer la contrainte du capital_tier
            final_position_size_pct = min(modulated_size_pct, tier_limit_pct)

            # 6. SL/TP restent immuables
            risk_params = self.config.get("risk_parameters", {})
            stop_loss_pct = float(risk_params.get("base_sl_pct", 0.02))
            take_profit_pct = float(risk_params.get("base_tp_pct", 0.04))

            logger.info(
                f"[DBE_MOD_V2 Worker {self.worker_id}] Base:{base_position_pct:.2%} | Regime:'{regime}'(x{regime_mult}) | Modulated:{modulated_size_pct:.2%} | TierLimit:{tier_limit_pct:.2%} -> Final:{final_position_size_pct:.2%}"
            )

            return {
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "position_size_pct": final_position_size_pct,
                "regime": self.current_regime,
                "regime_confidence": confidence,
                "risk_level": self.state.get("current_risk_level", 1.0),
            }

        except Exception as e:
            logger.error(f"Erreur dans update_risk_parameters: {e}", exc_info=True)
            return { "stop_loss_pct": 0.02, "take_profit_pct": 0.04, "position_size_pct": 0.1, "regime": "error" }

        
    def _normalize_tier_key(self, tier_value: Any) -> str:
        """Normalise le nom de palier vers: Micro, Small, Medium, High, Enterprise."""
        if isinstance(tier_value, dict):
            name = tier_value.get("name", "").lower()
        else:
            name = str(tier_value).lower()
        mapping = {
            "micro capital": "Micro",
            "micro": "Micro",
            "small capital": "Small",
            "small": "Small",
            "medium capital": "Medium",
            "medium": "Medium",
            "high capital": "High",
            "high": "High",
            "enterprise": "Enterprise",
        }
        return mapping.get(name, "Micro")

    def _get_tier_based_parameters(self, worker_key: str, current_tier: Any) -> Tuple[float, float, float]:
        """Récupère SL/TP/PosSize de base depuis trading_parameters (source unique de vérité Optuna).

        HIÉRARCHIE :
        1. Charger trading_parameters (base Optuna - source unique)
        2. Retourner les valeurs pures (pas de modulation ici)
        
        La modulation DBE sera appliquée dans compute_dynamic_modulation()
        """
        tier_key = self._normalize_tier_key(current_tier)

        # Récupère la config du worker depuis config.yaml (workers section)
        worker_config: Dict[str, Any] = self.config.get("workers", {}).get(worker_key, {})

        # Si pas trouvé dans config.yaml, essaie depuis env.worker_config (fallback)
        if not worker_config:
            worker_config = getattr(self.env, "worker_config", {}) or {}

        logger.debug(
            f"[TIER_PARAMS_V2] {worker_key} | tier_key={tier_key} | source={'config.yaml' if self.config.get('workers', {}).get(worker_key) else 'env.worker_config'}"
        )

        # SOURCE UNIQUE DE VÉRITÉ : trading_parameters (Optuna)
        trading_params = worker_config.get("trading_parameters", {})
        
        base_sl = float(trading_params.get("stop_loss_pct", 0.02))
        base_tp = float(trading_params.get("take_profit_pct", 0.04))
        base_pos = float(trading_params.get("position_size_pct", 0.1))

        logger.debug(
            f"[TIER_PARAMS_V2] {worker_key} | Optuna base: SL={base_sl:.2%}, TP={base_tp:.2%}, Pos={base_pos:.2%}"
        )

        # Pas d'aggressiveness_decay ici - sera appliqué dans compute_dynamic_modulation()
        # Pas de modulation de régime ici - sera appliquée dans compute_dynamic_modulation()
        
        return float(base_sl), float(base_tp), float(base_pos)

    def _compute_regime_modulation(self, base_sl: float, base_tp: float, pos_size: float, regime: str) -> Tuple[float, float, float]:
        """Applique la modulation par régime de marché."""
        reg_key = (regime or "sideways").lower()
        regime_params = self.config.get("regime_parameters", {}).get(reg_key, {})

        final_sl = float(base_sl) * float(regime_params.get("sl_multiplier", 1.0))
        final_tp = float(base_tp) * float(regime_params.get("tp_multiplier", 1.0))
        final_pos_size = float(pos_size) * float(regime_params.get("position_size_multiplier", 1.0))

        # Clamp SL/TP génériques
        final_sl = max(min(final_sl, 0.20), 0.005)
        final_tp = max(min(final_tp, 0.25), 0.01)

        # Clamp de sécurité générique sur pos_size (sera re-clampé au cap du tier plus bas)
        final_pos_size = max(min(final_pos_size, 0.95), 0.01)

        return final_sl, final_tp, final_pos_size

    def compute_dynamic_modulation(self, env=None, risk_horizon: float = 0.0) -> Dict[str, Any]:
        """Orchestre le calcul des paramètres en respectant la hiérarchie Optuna -> DBE -> Environnement.
        
        FLUX :
        1. Charger base Optuna (trading_parameters)
        2. Appliquer multiplicateurs DBE (±15% max)
        3. Clamp par hard_constraints (min/max absolus)
        4. Clamp par tier (max_position_size_pct)
        """
        try:
            if env:
                self.set_env_reference(env)
            
            if not hasattr(self, "env") or self.env is None:
                logger.warning("[DBE_V2] Référence 'env' manquante, retour des paramètres par défaut")
                return self._get_default_modulation()

            # Récupère worker_key
            worker_key = getattr(self.env, "worker_name", None)
            if not worker_key:
                try:
                    worker_key = self.env.worker_config.get("name", f"w{getattr(self, 'worker_id', 0)}")
                except Exception:
                    worker_key = f"w{getattr(self, 'worker_id', 0)}"

            # Récupère current_tier
            try:
                current_tier = self.env.portfolio.get_current_tier()
            except Exception:
                current_tier = "Micro"

            # Récupère regime
            regime = str(getattr(self, "current_regime", "sideways")).lower()

            # ÉTAPE 1 : Charger base Optuna
            base_sl, base_tp, base_pos = self._get_tier_based_parameters(worker_key, current_tier)
            
            # ÉTAPE 2 : Appliquer multiplicateurs DBE (±15% max)
            regime_params = self.config.get("dbe", {}).get("regime_parameters", {}).get(regime, {})
            
            # Convertir multiplicateurs absolus en relatifs et borner à ±15%
            pos_mult_raw = float(regime_params.get("position_size_multiplier", 1.0))
            sl_mult_raw = float(regime_params.get("sl_multiplier", 1.0))
            tp_mult_raw = float(regime_params.get("tp_multiplier", 1.0))
            
            # Convertir en ajustement relatif et borner à ±15%
            pos_adjustment = min(max(pos_mult_raw - 1.0, -0.15), 0.15)
            sl_adjustment = min(max(sl_mult_raw - 1.0, -0.15), 0.15)
            tp_adjustment = min(max(tp_mult_raw - 1.0, -0.15), 0.15)
            
            # Appliquer ajustements
            adjusted_pos = base_pos * (1 + pos_adjustment)
            adjusted_sl = base_sl * (1 + sl_adjustment)
            adjusted_tp = base_tp * (1 + tp_adjustment)
            
            logger.debug(
                f"[DBE_V2_MOD] {worker_key} | Regime={regime} | Adjustments: Pos{pos_adjustment:+.1%}, SL{sl_adjustment:+.1%}, TP{tp_adjustment:+.1%}"
            )
            
            # ÉTAPE 3 : Clamp par hard_constraints (min/max absolus)
            hard_constraints = self.config.get("environment", {}).get("hard_constraints", {})
            
            sl_min = float(hard_constraints.get("stop_loss_pct", {}).get("min", 0.005))
            sl_max = float(hard_constraints.get("stop_loss_pct", {}).get("max", 0.20))
            tp_min = float(hard_constraints.get("take_profit_pct", {}).get("min", 0.01))
            tp_max = float(hard_constraints.get("take_profit_pct", {}).get("max", 0.50))
            
            adjusted_sl = max(min(adjusted_sl, sl_max), sl_min)
            adjusted_tp = max(min(adjusted_tp, tp_max), tp_min)
            
            # ÉTAPE 4 : Clamp par tier (max_position_size_pct)
            try:
                if isinstance(current_tier, dict):
                    tier_cap_pct = float(current_tier.get("max_position_size_pct", 90.0)) / 100.0
                else:
                    tier_cap_pct = 0.90
            except Exception:
                tier_cap_pct = 0.90
            
            adjusted_pos = min(adjusted_pos, tier_cap_pct)
            adjusted_pos = max(adjusted_pos, 0.01)  # Min 1%

            tier_name = current_tier if isinstance(current_tier, str) else getattr(current_tier, "name", str(current_tier))
            logger.info(
                f"[DBE_V2_FINAL] {worker_key} | Tier={tier_name} | Regime={regime} | Final: SL={adjusted_sl:.2%}, TP={adjusted_tp:.2%}, Pos={adjusted_pos:.2%}"
            )

            mod = {
                "sl_pct": float(adjusted_sl),
                "tp_pct": float(adjusted_tp),
                "position_size_pct": float(adjusted_pos),
                "regime": regime,
                "regime_confidence": 0.8,  # Default confidence
            }
            return mod
        except Exception as e:
            logger.error(f"Erreur dans compute_dynamic_modulation: {e}", exc_info=True)
            return { "sl_pct": 0.02, "tp_pct": 0.04, "position_size_pct": 0.1, "regime": "error" }

    def _calculate_frequency_adjustment(
        self,
        positions_count: Dict[str, int],
    ) -> Dict[str, float]:
        """
        Calcule les ajustements de paramètres basés sur la fréquence des positions.

        Args:
            positions_count: Dictionnaire avec les compteurs par timeframe

        Returns:
            Dict avec les multiplicateurs d'ajustement
        """
        if not self.frequency_config:
            return {
                "position_size_multiplier": 1.0,
                "sl_multiplier": 1.0,
                "tp_multiplier": 1.0,
            }

        adjustment = {
            "position_size_multiplier": 1.0,
            "sl_multiplier": 1.0,
            "tp_multiplier": 1.0,
        }

        # Vérifier chaque timeframe individuellement
        for timeframe in ["5m", "1h", "4h"]:
            if timeframe in self.frequency_config:
                tf_config = self.frequency_config[timeframe]
                min_pos = tf_config.get("min_positions", 0)
                max_pos = tf_config.get("max_positions", 999)
                current_count = positions_count.get(timeframe, 0)

                # Ajustements en fonction du régime de marché et de la fréquence
                if current_count < min_pos:
                    # Pas assez de positions : encourager plus de trades
                    if self.current_regime == "bull":
                        adjustment["position_size_multiplier"] *= (
                            1.1  # Augmenter taille position
                        )
                        adjustment["sl_multiplier"] *= 1.05  # SL moins strict
                    elif self.current_regime == "neutral":
                        adjustment["position_size_multiplier"] *= 1.05
                    # En bear, rester conservateur même si pas assez de positions

                elif current_count > max_pos:
                    # Trop de positions : limiter les trades
                    adjustment["position_size_multiplier"] *= 0.8  # Réduire taille
                    adjustment["sl_multiplier"] *= (
                        0.9  # SL plus strict pour fermer plus vite
                    )
                    adjustment["tp_multiplier"] *= 0.95  # TP plus court

        # Ajustement pour le total journalier
        total_min = self.frequency_config.get("total_daily_min", 5)
        total_max = self.frequency_config.get("total_daily_max", 15)
        daily_total = positions_count.get("daily_total", 0)

        if daily_total < total_min:
            # Encourager plus de trades au global
            adjustment["position_size_multiplier"] *= 1.1
            if self.current_regime in ["bull", "neutral"]:
                adjustment["sl_multiplier"] *= 1.1  # SL moins strict
        elif daily_total > total_max:
            # Limiter les trades au global
            adjustment["position_size_multiplier"] *= 0.7
            adjustment["sl_multiplier"] *= 0.8  # SL plus strict
            adjustment["tp_multiplier"] *= 0.9  # TP plus court

        # S'assurer que les ajustements restent dans des bornes raisonnables
        adjustment["position_size_multiplier"] = np.clip(
            adjustment["position_size_multiplier"], 0.3, 2.0
        )
        adjustment["sl_multiplier"] = np.clip(adjustment["sl_multiplier"], 0.5, 1.5)
        adjustment["tp_multiplier"] = np.clip(adjustment["tp_multiplier"], 0.7, 1.5)

        return adjustment

    def set_env_reference(self, env):
        """
        Définit la référence à l'environnement pour accéder aux compteurs de fréquence.

        Args:
            env: Instance de MultiAssetChunkedEnv
        """
        self.env = env

    def deep_update(self, d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self.deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    @property
    def market_regime(self) -> str:
        """Get current market regime."""
        return self.state.get("market_regime", "NEUTRAL")

    @property
    def current_step(self) -> int:
        """Get current step."""
        return self.state.get("current_step", 0)

    @property
    def risk_level(self) -> float:
        """Get current risk level."""
        return self.state.get("current_risk_level", 1.0)

    def on_trade_closed(self, trade_result: Dict[str, Any]) -> None:
        """Process a closed trade result."""
        self._process_trade_result(trade_result)

    def _get_position_size(self, regime: str, portfolio_manager) -> float:
        # 1. Récupérer la taille du palier (70–90%) depuis capital_tiers
        tier = portfolio_manager.get_current_tier()
        tier_size = 0.70  # Default fallback
        
        if isinstance(tier, dict):
            # Essayer d'extraire la taille de position depuis exposure_range
            exposure_range = tier.get("exposure_range")
            if exposure_range and isinstance(exposure_range, (list, tuple)) and len(exposure_range) >= 2:
                tier_size = float(exposure_range[1]) / 100.0  # exposure_range est en pourcentage
            else:
                # Fallback: utiliser pos_size_pct de portfolio_manager
                tier_size = getattr(portfolio_manager, 'pos_size_pct', 0.70) or 0.70

        # 2. Appliquer un multiplicateur léger selon régime
        multipliers = {
            "bear": 0.8,      # 70% → 56%
            "bull": 1.0,      # 70% → 70%
            "neutral": 0.9
        }
        multiplier = multipliers.get(regime, 0.9)

        # 3. Appliquer + clamp
        final = tier_size * multiplier
        final = np.clip(final, 0.50, 0.90)  # Jamais <50%, jamais >90%

        return final

    def _get_default_modulation(self) -> Dict[str, Any]:
        """Returns a safe, default modulation dictionary in case of critical errors."""
        self.logger.error("[DBE] Critical error, returning default modulation.")
        return {
            "sl_pct": self.config.get("risk_parameters", {}).get("base_sl_pct", 0.02),
            "tp_pct": self.config.get("risk_parameters", {}).get("base_tp_pct", 0.04),
            "reward_boost": 1.0,
            "penalty_inaction": 0.0,
            "position_size_pct": 0.1,
            "risk_mode": "DEFENSIVE",
            "error": "Default modulation due to critical error",
        }

    def _apply_market_regime_modulation(self, mod: Dict[str, Any]) -> None:
        """Applique les modulations spécifiques au régime de marché."""
        regime = self.state["market_regime"].upper()
        mode_config = self.config.get("modes", {}).get(regime.lower(), {})

        if not mode_config:
            return

        # Application des multiplicateurs
        # ⚠️ IMPORTANT: SL/TP sont IMMUABLES - pas de modulation par régime
        # mod["sl_pct"] *= mode_config.get("sl_multiplier", 1.0)  # DISABLED
        # mod["tp_pct"] *= mode_config.get("tp_multiplier", 1.0)  # DISABLED
        mod["position_size_pct"] *= mode_config.get("position_size_multiplier", 1.0)
        mod["risk_mode"] = regime

    def _adjust_position_size_aggressively(self, mod: Dict[str, Any], env) -> None:
        """Ajuste la taille de position de manière agressive pour forcer plus de trades."""
        regime = self.detect_regime()

        # Get frequency configuration and counts
        frequency_config = env.config.get("trading_rules", {}).get("frequency", {})
        positions_count = getattr(env, "positions_count", {})
        last_trade_steps_by_tf = getattr(env, "last_trade_steps_by_tf", {})
        current_step = getattr(env, "current_step", 0)

        # Base position size
        base_position_size = (
            mod.get("position_size_pct", 0.1) * 100
        )  # Convert to percentage

        for tf in ["5m", "1h", "4h"]:
            count = positions_count.get(tf, 0)
            min_pos = frequency_config.get("min_positions", {}).get(tf, 1)
            steps_since_last_trade = current_step - last_trade_steps_by_tf.get(tf, 0)
            force_trade_steps = frequency_config.get("force_trade_steps", 50)

            # Aggressive position size adjustments
            if (
                count < min_pos
                and regime in ["bull", "neutral"]
                and steps_since_last_trade > force_trade_steps
            ):
                base_position_size = min(
                    base_position_size * 1.3, 100.0
                )  # Increase by 30%
                logger.info(
                    f"[DBE_POSITION Worker {getattr(self, 'worker__id', 0)}] Increasing position size by 30% for {tf} - insufficient trades"
                )
            elif (
                count > frequency_config.get("max_positions", {}).get(tf, 10)
                and regime == "bear"
            ):
                base_position_size = max(
                    base_position_size * 0.7, 10.0
                )  # Decrease by 30%
                logger.info(
                    f"[DBE_POSITION Worker {getattr(self, 'worker_id', 0)}] Decreasing position size by 30% for {tf} - excessive trades"
                )

        # Update the modulation
        mod["position_size_pct"] = base_position_size / 100.0  # Convert back to decimal

        # Log the decision
        logger.info(
            f"[DBE_DECISION Worker {getattr(self, 'worker_id', 0)}] Step: {current_step} | Regime: {regime} | "
            f"SL: {mod.get('sl_pct', 0.02) * 100:.2f}% | TP: {mod.get('tp_pct', 0.04) * 100:.2f}% | "
            f"PosSize: {base_position_size:.1f}% | Counts: {positions_count}"
        )

    def detect_regime(self):
        """Detect current market regime."""
        # Simple regime detection - can be enhanced with actual market data
        if hasattr(self.state, "market_regime"):
            return self.state.get("market_regime", "neutral").lower()

        # Fallback regime detection based on performance
        performance = self.state.get("performance_metrics", {})
        winrate = performance.get("win_rate", 50.0)

        if winrate > 60:
            return "bull"
        elif winrate < 40:
            return "bear"
        else:
            return "neutral"

    def _adjust_learning_parameters(self, mod: Dict[str, Any]) -> None:
        """Ajuste les paramètres d'apprentissage en fonction du risque."""
        learning_config = self.config.get("learning", {})

        # Récupération et validation des plages de valeurs
        lr_range = [
            max(1e-8, float(x))
            for x in learning_config.get("learning_rate_range", [1e-5, 1e-3])
        ]  # Minimum 1e-8 pour éviter les valeurs trop petites
        ent_coef_range = [
            max(1e-8, float(x))
            for x in learning_config.get("ent_coef_range", [0.001, 0.1])
        ]
        gamma_range = [
            max(0.1, min(float(x), 0.999))
            for x in learning_config.get("gamma_range", [0.9, 0.999])
        ]  # Gamma entre 0.1 et 0.999

        # Ajustement basé sur le niveau de risque avec clamping
        risk_factor = max(
            0.1, min(float(self.state["current_risk_level"]), 10.0)
        )

        try:
            # Calcul du learning rate avec clamping pour éviter les valeurs négatives ou trop élevées
            base_lr = lr_range[0] + (lr_range[1] - lr_range[0]) * (risk_factor - 1.0)
            mod["learning_rate"] = max(
                1e-8, min(base_lr, lr_range[1] * 2.0) # Ne dépasse pas le double du max
            )

            # Calcul de l'entropy coefficient avec clamping
            ent_coef = ent_coef_range[0] + (ent_coef_range[1] - ent_coef_range[0]) * (
                1.0 / max(0.1, risk_factor)
            )
            mod["ent_coef"] = max(1e-8, min(ent_coef, ent_coef_range[1] * 2.0))

            # Calcul du gamma avec clamping
            gamma = gamma_range[0] + (gamma_range[1] - gamma_range[0]) * (
                min(risk_factor, 2.0) - 1.0
            )
            mod["gamma"] = max(0.1, min(gamma, 0.999))

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
            mod["learning_rate"] = lr_range[0]
            mod["ent_coef"] = ent_coef_range[0]
            mod["gamma"] = gamma_range[0]

    def _validate_parameters(self, mod: Dict[str, Any]) -> None:
        """Valide et contraint les paramètres dans des limites acceptables."""
        risk_params = self.config.get("risk_parameters", {})
        pos_params = self.config.get("position_sizing", {})

        # ⚠️ IMPORTANT: SL/TP sont IMMUABLES - DBE ne les modifie PAS
        # Les valeurs SL/TP sont déterminées par Optuna pour chaque worker
        # et appliquées directement par le portfolio manager
        # Ne pas modifier mod["sl_pct"] et mod["tp_pct"] ici

        # Contraintes sur la taille de position
        mod["position_size_pct"] = np.clip(
            mod["position_size_pct"],
            pos_params.get("min_position_size", 0.01),
            pos_params.get("max_position_size", 0.30),
        )

    def _log_decision(self, snapshot: Dict[str, Any], mod: Dict[str, Any]) -> None:
        """Journalise la décision prise par le DBE."""
        decision_data = {
            "step": snapshot.step,
            "market_regime": snapshot.market_regime,
            "risk_level": snapshot.risk_level,
            "modulation": {
                "sl_pct": snapshot.sl_pct,
                "tp_pct": snapshot.tp_pct,
                "position_size_pct": snapshot.position_size_pct,
                "reward_boost": snapshot.reward_boost,
                "penalty_inaction": snapshot.penalty_inaction,
                "learning_rate": mod.get("learning_rate"),
                "ent_coef": mod.get("ent_coef"),
                "gamma": mod.get("gamma"),
            },
            "performance_metrics": snapshot.metrics,
            "timestamp": snapshot.timestamp.isoformat(),
        }

        # Utilisation du ReplayLogger pour enregistrer la décision
        self.logger.log_decision(
            step_index=snapshot.step,
            modulation_dict=decision_data["modulation"],
            context_metrics={
                "market_regime": snapshot.market_regime,
                "risk_level": snapshot.risk_level,
                "drawdown": self.state.get("drawdown", 0.0),
                "winrate": self.state.get("winrate", 0.0),
                "volatility": self.state.get("volatility", 0.0),
            },
            performance_metrics=snapshot.metrics,
            additional_info={
                "consecutive_losses": self.state["consecutive_losses"],
                "position_duration": self.state["position_duration"],
            },
        )

        self.log_info(
            f"DBE Decision - Step: {snapshot.step} | "
            f"Regime: {snapshot.market_regime} | "
            f"SL: {snapshot.sl_pct * 100:.2f}% | "
            f"TP: {snapshot.tp_pct * 100:.2f}% | "
            f"PosSize: {snapshot.position_size_pct * 100:.1f}% | "
            f"Winrate: {self.state['winrate'] * 100:.1f}%"
        )

    @lru_cache(maxsize=128)
    def _detect_market_regime(
        self,
        rsi: float,
        adx: float,
        ema_ratio: float,
        atr: float,
        atr_pct: float,
    ) -> str:
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
                    return "BULL"
                elif ema_ratio < 0.995:  # Tendance baissière
                    return "BEAR"

            # Marché sans tendance
            if atr_pct > 0.02:  # Volatilité élevée
                return "VOLATILE"
            else:
                return "SIDEWAYS"

        except Exception as e:
            logger.error(f"Erreur lors de la détection du régime de marché: {e}")
            return "UNKNOWN"

    def _adjust_risk_level(self) -> None:
        """
        Ajuste dynamiquement le niveau de risque avec une formule additive robuste.
        Cette version est conçue pour être stable même avec des métriques d'entrée anormales.
        """
        try:
            # Récupération des configurations avec des valeurs par défaut robustes
            risk_params = self.config.get("risk_parameters", {})
            min_risk = float(risk_params.get("min_risk_level", 0.3))
            max_risk = float(risk_params.get("max_risk_level", 2.0))
            base_risk_level = (min_risk + max_risk) / 2.0  # Point de départ neutre

            # --- 1. Sanitisation des Métriques d'Entrée ---
            portfolio_metrics = (
                self.finance_manager.get_metrics() if self.finance_manager else {}
            )

            # S'assurer que le win_rate est un ratio (0-1)
            raw_win_rate = portfolio_metrics.get(
                "win_rate", self.state.get("win_rate", 0.5)
            )
            win_rate = raw_win_rate / 100.0 if raw_win_rate > 1.0 else raw_win_rate
            win_rate = np.clip(win_rate, 0.0, 1.0)

            drawdown = np.clip(
                portfolio_metrics.get("drawdown", self.state.get("drawdown", 0.0)),
                0.0,
                1.0,
            )

            sharpe_ratio = portfolio_metrics.get(
                "sharpe_ratio", self.state.get("sharpe_ratio", 0.0)
            )
            sharpe_ratio = np.nan_to_num(
                sharpe_ratio, nan=0.0, posinf=2.0, neginf=-2.0
            )  # Contrôler les valeurs extrêmes

            consecutive_losses = self.state.get("consecutive_losses", 0)

            # --- 2. Calcul des Scores Normalisés (-1 à +1) ---

            # Score de Win Rate (centré autour de 55%)
            win_rate_score = np.clip((win_rate - 0.55) / 0.2, -1.0, 1.0)  # De 35% à 75%

            # Score de Drawdown (pénalité non-linéaire)
            max_allowed_drawdown = float(risk_params.get("max_drawdown", 0.1))
            drawdown_score = (
                1.0 - (drawdown / max_allowed_drawdown) ** 0.5
            )  # Racine carrée pour pénaliser plus fortement au début
            drawdown_score = np.clip(
                drawdown_score * 2 - 1, -1.0, 1.0
            )  # Mapper sur [-1, 1]

            # Score de Sharpe Ratio (borné avec tanh pour la stabilité)
            sharpe_score = np.tanh(
                sharpe_ratio / 2.0
            )  # tanh mappe sur [-1, 1], divisé par 2 pour adoucir

            # Score des Pertes Consécutives (pénalité exponentielle)
            loss_streak_score = (
                np.exp(-consecutive_losses / 5.0) * 2 - 1
            )  # De +1 (0 perte) à -1 (beaucoup de pertes)

            # --- 3. Combinaison Additive Pondérée ---
            weights = {
                "win_rate": 0.3,
                "drawdown": 0.4,
                "sharpe": 0.2,
                "loss_streak": 0.1,
            }

            performance_score = (
                weights["win_rate"] * win_rate_score
                + weights["drawdown"] * drawdown_score
                + weights["sharpe"] * sharpe_score
                + weights["loss_streak"] * loss_streak_score
            )
            performance_score = np.clip(performance_score, -1.0, 1.0)

            # --- 4. Mapper le Score au Niveau de Risque ---
            risk_range = (max_risk - min_risk) / 2.0
            target_risk = base_risk_level + performance_score * risk_range

            # --- 5. Lissage Exponentiel pour la Stabilité ---
            alpha = self.config.get("smoothing", {}).get("adaptation_rate", 0.1)
            current_risk = self.state.get("current_risk_level", base_risk_level)
            smoothed_risk = (1.0 - alpha) * current_risk + alpha * target_risk

            # --- 6. Application Finale des Bornes ---
            self.state["current_risk_level"] = np.clip(
                smoothed_risk, min_risk, max_risk
            )

            # --- 7. Journalisation pour le Débogage ---
            if self.state["current_step"] % 50 == 0:  # Log toutes les 50 étapes
                logger.info(
                    f"RISK_ADJUST | Step: {self.state['current_step']} | "
                    f"WinRate: {win_rate:.2f} (Score: {win_rate_score:.2f}) | "
                    f"Drawdown: {drawdown:.2f} (Score: {drawdown_score:.2f}) | "
                    f"Sharpe: {sharpe_ratio:.2f} (Score: {sharpe_score:.2f}) | "
                    f"Perf Score: {performance_score:.2f} -> "
                    f"Risk Level: {self.state['current_risk_level']:.2f}"
                )

        except Exception as e:
            logger.error(
                f"Erreur lors de l'ajustement du niveau de risque: {e}", exc_info=True
            )
            # En cas d'erreur, on revient à un niveau de risque conservateur
            self.state["current_risk_level"] = min_risk

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques de performance actuelles.

        Returns:
            Dictionnaire des métriques de performance
        """
        if not self.finance_manager:
            return {}

        # Récupération des métriques du gestionnaire financier
        portfolio_metrics = self.finance_manager.get_metrics()

        # Calcul des métriques avancées
        if self.trade_history:
            recent_trades = self.trade_history[-100:]  # 100 derniers trades
            pnls = tuple(t["pnl_pct"] for t in recent_trades if "pnl_pct" in t)
            wins = [t for t in recent_trades if t.get("is_win", False)]
            losses = [t for t in recent_trades if not t.get("is_win", True)]

            avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0.0
            avg_loss = abs(np.mean([t["pnl_pct"] for t in losses])) if losses else 0.0
            win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else float("inf")

            # Utilisation des méthodes mises en cache
            risk_free_rate = 0.0  # Taux sans risque (peut être paramétré)
            sharpe_ratio = (
                self._calculate_sharpe_ratio(pnls, risk_free_rate) if pnls else 0.0
            )
            sortino_ratio = (
                self._calculate_sortino_ratio(pnls, risk_free_rate) if pnls else 0.0
            )
        else:
            avg_win = avg_loss = win_loss_ratio = sharpe_ratio = sortino_ratio = 0.0

        # Construction du dictionnaire de résultats
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": self.state["current_step"],
            "portfolio": {
                "total_value": portfolio_metrics.get("total_capital", 0.0),
                "free_cash": portfolio_metrics.get("free_capital", 0.0),
                "invested": portfolio_metrics.get("invested_capital", 0.0),
                "total_return": portfolio_metrics.get("total_return", 0.0),
                "max_drawdown": portfolio_metrics.get("max_drawdown", 0.0),
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
            },
            "trading": {
                "total_trades": portfolio_metrics.get("trade_count", 0),
                "win_rate": portfolio_metrics.get("win_rate", 0.0),  # en pourcentage
                "avg_win_pct": avg_win * 100,  # en pourcentage
                "avg_loss_pct": avg_loss * 100,  # en pourcentage
                "win_loss_ratio": win_loss_ratio,
                "consecutive_losses": self.state["consecutive_losses"],
                "avg_trade_duration": self.state.get("position_duration", 0),
            },
            "risk": {
                "current_risk_level": self.state["current_risk_level"],
                "market_regime": self.state["market_regime"],
                "current_volatility": self.state.get("volatility", 0.0),
                "current_drawdown": self.state.get("drawdown", 0.0),
            },
        }

        # Mise à jour des métriques de performance dans l'état
        self.state["performance_metrics"] = metrics

        return metrics

    @lru_cache(maxsize=128)
    def _calculate_sharpe_ratio(
        self,
        returns_tuple: Tuple[float, ...],
        risk_free_rate: float = 0.0,
    ) -> float:
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

        sharpe = (
            np.mean(excess_returns) / std_dev * np.sqrt(365)
        )  # 365 days for crypto (24/7 trading)
        return float(sharpe)

    @lru_cache(maxsize=128)
    def _calculate_sortino_ratio(
        self,
        returns_tuple: Tuple[float, ...],
        risk_free_rate: float = 0.0,
    ) -> float:
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
            return float("inf") if np.mean(excess_returns) > 0 else 0.0

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
        recent_decisions = (
            self.decision_history[-limit:] if self.decision_history else []
        )

        # Conversion des snapshots en dictionnaires
        return [
            {
                "timestamp": d.timestamp.isoformat(),
                "step": d.step,
                "market_regime": d.market_regime,
                "risk_level": d.risk_level,
                "sl_pct": d.sl_pct,
                "tp_pct": d.tp_pct,
                "position_size_pct": d.position_size_pct,
                "reward_boost": d.reward_boost,
                "penalty_inaction": d.penalty_inaction,
                "metrics": d.metrics,
            }
            for d in recent_decisions
        ]

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
        return [
            {
                "timestamp": t["timestamp"].isoformat()
                if hasattr(t["timestamp"], "isoformat")
                else str(t["timestamp"]),
                "pnl_pct": t.get("pnl_pct", 0.0),
                "is_win": t.get("is_win", False),
                "position_duration": t.get("position_duration", 0),
                "drawdown": t.get("drawdown", 0.0),
                "market_regime": t.get("market_regime", "UNKNOWN"),
            }
            for t in recent_trades
        ]

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
                "state": self.state,
                "trade_history": self.trade_history,
                "decision_history": [d.__dict__ for d in self.decision_history],
                "win_rates": self.win_rates,
                "drawdowns": self.drawdowns,
                "position_durations": self.position_durations,
                "config": self.config,
            }

            with open(filepath, "wb") as f:
                pickle.dump(state, f)

            self.log_info(f"État du DBE sauvegardé dans {filepath}")
            return True

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'état du DBE: {e}")
            return False

    @classmethod
    def load_state(
        cls,
        filepath: Union[str, Path],
        finance_manager: Optional[Any] = None,
    ) -> Optional["DynamicBehaviorEngine"]:
        """
        Charge un état précédemment sauvegardé.

        Args:
            filepath: Chemin vers le fichier de sauvegarde
            finance_manager: Instance de FinanceManager (optionnel)

        Returns:
            Une instance de DynamicBehaviorEngine avec l'état chargé, ou None en cas d'erreur
        """
        try:
            with open(filepath, "rb") as f:
                state = pickle.load(f)

            # Création d'une nouvelle instance avec la configuration sauvegardée
            dbe = cls(config=state.get("config", {}), finance_manager=finance_manager)

            # Restauration de l'état
            dbe.state = state.get("state", {})
            dbe.trade_history = state.get("trade_history", [])
            dbe.decision_history = [
                DBESnapshot(**d) for d in state.get("decision_history", [])
            ]
            dbe.win_rates = state.get("win_rates", [])
            dbe.drawdowns = state.get("drawdowns", [])
            dbe.position_durations = state.get("position_durations", [])

            self.log_info(f"État du DBE chargé depuis {filepath}")
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
            metrics = self.finance_manager.get_metrics()
            portfolio_value = metrics.get("total_capital", 0.0)
            free_cash = metrics.get("free_capital", 0.0)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": self.state["current_step"],
            "market_regime": self.state["market_regime"],
            "risk_level": self.state["current_risk_level"],
            "portfolio_value": portfolio_value,
            "free_cash": free_cash,
            "drawdown": self.state["drawdown"],
            "winrate": self.state["winrate"],
            "consecutive_losses": self.state["consecutive_losses"],
            "last_modulation": self.state.get("last_modulation", {}),
            "total_decisions": len(self.decision_history),
            "total_trades": len(self.trade_history),
        }

    def start_hunt(
        self,
        worker_id: int,
        asset: str,
        hunting_timeframe: str,
        duration_steps: int,
        start_step: int,
    ):
        """
        Enregistre le début d'une nouvelle traque pour un worker spécifique.
        """
        if worker_id not in self._active_hunts:
            self._active_hunts[worker_id] = {
                "asset": asset,
                "hunting_timeframe": hunting_timeframe,
                "duration_steps": duration_steps,
                "start_step": start_step,
                "active": True,
            }
            self.log_info(
                f"[HUNT STARTED] Worker {worker_id} starting hunt for {asset} on {hunting_timeframe} for {duration_steps} steps."
            )

    def end_hunt(self, worker_id: int):
        """
        Marque une traque comme terminée pour un worker spécifique.
        """
        if worker_id in self._active_hunts:
            hunt_info = self._active_hunts.pop(worker_id)
            self.log_info(
                f"[HUNT ENDED] Worker {worker_id} ended hunt for {hunt_info.get('asset')}."
            )

    def is_hunting(self, worker_id: int) -> bool:
        """
        Vérifie si un worker est actuellement en traque.
        """
        hunt_info = self._active_hunts.get(worker_id)
        if not hunt_info:
            return False

        # Vérifie si la traque a expiré
        current_step = self.state.get("current_step", 0)
        start_step = hunt_info.get("start_step", 0)
        duration = hunt_info.get("duration_steps", 0)
        if current_step > start_step + duration:
            self.log_info(
                f"[HUNT EXPIRED] Worker {worker_id} hunt for {hunt_info.get('asset')} expired."
            )
            self.end_hunt(worker_id)
            return False

        return True

    def get_hunt_info(self, worker_id: int) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations sur la traque en cours pour un worker donné.
        """
        if self.is_hunting(worker_id):  # Ceci vérifie aussi l'expiration
            return self._active_hunts.get(worker_id)
        return None

    def reset(self) -> None:
        """Réinitialise l'état interne du DBE."""
        # Réinitialisation de l'état
        self.state = {
            "current_step": 0,
            "drawdown": 0.0,
            "winrate": 0.0,
            "volatility": 0.0,
            "market_regime": "NEUTRAL",
            "last_trade_pnl": 0.0,
            "consecutive_losses": 0,
            "position_duration": 0,
            "current_risk_level": 1.0,
            "max_risk_level": 2.0,
            "min_risk_level": 0.5,
            "last_modulation": {},
            "performance_metrics": {},
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
            "sl_pct": self.config.get("risk_parameters", {}).get("base_sl_pct", 0.02),
            "tp_pct": self.config.get("risk_parameters", {}).get("base_tp_pct", 0.04),
        }

        # Réinitialisation du gestionnaire financier si disponible
        if self.finance_manager:
            self.finance_manager.reset()

        # SOTA 2025: Reset HMM state
        if hasattr(self, '_hmm_obs_buffer'):
            self._hmm_obs_buffer = []
            self._hmm_fitted = False
            self._hmm_probs = np.ones(N_HMM_STATES, dtype=np.float32) / N_HMM_STATES

        self.log_info("DBE réinitialisé")

    def reset_for_new_chunk(self, continuity=True):
        if continuity:
            # Only log from primary worker to avoid duplication
            worker_id = getattr(self, "worker_id", 0)
            self.log_info(
                "[DBE CONTINUITY] Préservation historique – Append volatility_history."
            )
            if hasattr(self, "new_vol_data"):
                self.volatility_history.extend(self.new_vol_data)  # Accumule données
            # Pas de reset pour regime, sl, tp, etc.
        else:
            # Réservé pour reset complet (rare)
            # Only log from primary worker to avoid duplication
            worker_id = getattr(self, "worker_id", 0)
            if hasattr(self, "smart_logger"):
                self.smart_logger.smart_warning(
                    logger,
                    "[DBE FULL RESET] Réinitialisation complète – Perte historique.",
                )
            else:
                logger.warning(
                    f"[Worker {worker_id}] [DBE FULL RESET] Réinitialisation complète – Perte historique."
                )
            self.volatility_history = []
            self.regime = "neutral"
            self.sl_pct = 0.02
            self.tp_pct = 0.0394

    def _reset_for_new_chunk_legacy(self) -> None:
        """
        Réinitialisation complète (ancien comportement) - utilisé seulement pour hard reset.
        """
        self.state["current_step"] = 0
        self.state["last_trade_pnl"] = 0.0
        self.state["consecutive_losses"] = 0
        self.state["position_duration"] = 0
        self.state["volatility"] = 0.0
        self.state["market_regime"] = "NEUTRAL"
        self.state["trend_strength"] = 0.0

        # Reset complet de l'historique (perte d'expérience)
        if hasattr(self, "volatility_history"):
            self.volatility_history = []
        if hasattr(self, "trade_history"):
            self.trade_history = []
        self.current_regime = "neutral"

        self.log_info("🔄 DBE: Reset complet effectué (perte d'expérience)")

    def _adapt_smoothing_factor(self) -> None:
        """
        Adapte le facteur de lissage (smoothing_factor) en fonction des performances récentes.
        - Réduit le lissage (augmente smoothing_factor) si les performances sont bonnes (winrate élevé, faible drawdown).
        - Augmente le lissage (diminue smoothing_factor) si les performances sont mauvaises (winrate faible, drawdown élevé).
        """
        current_winrate = self.state.get("winrate", 0.0)
        current_drawdown = self.state.get("drawdown", 0.0)

        # Paramètres de configuration pour l'adaptation du lissage
        adapt_config = self.config.get(
            "smoothing_adaptation",
            {
                "min_smoothing": 0.01,
                "max_smoothing": 0.5,
                "winrate_threshold_good": 0.6,
                "winrate_threshold_bad": 0.4,
                "drawdown_threshold_good": 5.0,  # in percent
                "drawdown_threshold_bad": 15.0,  # in percent
                "adaptation_rate": 0.01,
            },
        )

        min_smoothing = adapt_config["min_smoothing"]
        max_smoothing = adapt_config["max_smoothing"]
        winrate_threshold_good = adapt_config["winrate_threshold_good"]
        winrate_threshold_bad = adapt_config["winrate_threshold_bad"]
        drawdown_threshold_good = adapt_config["drawdown_threshold_good"]
        drawdown_threshold_bad = adapt_config["drawdown_threshold_bad"]
        adaptation_rate = adapt_config["adaptation_rate"]

        new_smoothing_factor = self.smoothing_factor

        # Ajustement basé sur le winrate
        if current_winrate > winrate_threshold_good:
            new_smoothing_factor += (
                adaptation_rate  # Reduce smoothing (faster adaptation)
            )
        elif current_winrate < winrate_threshold_bad:
            new_smoothing_factor -= (
                adaptation_rate  # Increase smoothing (slower adaptation)
            )

        # Ajustement basé sur le drawdown
        if current_drawdown < drawdown_threshold_good:  # Lower drawdown is good
            new_smoothing_factor += adaptation_rate
        elif current_drawdown > drawdown_threshold_bad:  # Higher drawdown is bad
            new_smoothing_factor -= adaptation_rate

        # Clip the smoothing factor to stay within bounds
        self.smoothing_factor = np.clip(
            new_smoothing_factor, min_smoothing, max_smoothing
        )
        logger.debug(
            f"Smoothing factor adapted to: {self.smoothing_factor:.3f} (Winrate: {current_winrate:.2f}, Drawdown: {current_drawdown:.2f})"
        )

    def calculate_trade_parameters(
        self,
        capital: float,
        worker_pref_pct: float,
        tier_config: Optional[Dict[str, Any]] = None,
        current_price: Optional[float] = None,
        asset_volatility: Optional[float] = None,
        dbe_modulation: Optional[Dict[str, Any]] = None,
        risk_horizon: float = 0.0,
        desired_position_size: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calcule les paramètres de trade en fonction du capital, des préférences du worker,
        de la configuration du palier et de l'horizon de risque choisi par l'agent.

        Args:
            capital: Capital total disponible
            worker_pref_pct: Préférence du worker (score de confiance d'achat/vente)
            tier_config: Configuration du palier de capital
            current_price: Prix actuel de l'actif
            asset_volatility: Volatilité de l'actif
            dbe_modulation: Modulation du DBE
            risk_horizon: Horizon de risque choisi par l'agent (-1: court terme, 1: long terme)
            desired_position_size: Taille de position désirée par l'agent (-1: petite, 1: grande)
        """
        logger.debug(
            f"[DBE_CALC] Entrée: capital={capital:.2f}, worker_pref={worker_pref_pct:.2f}, tier_config?={tier_config is not None}, price={current_price}, vol={asset_volatility}, risk_horizon={risk_horizon:.2f}, desired_size={desired_position_size:.2f}"
        )
        try:
            if not tier_config or not isinstance(tier_config, dict):
                logger.warning(
                    "[DBE_CALC] Échec: tier_config est manquant ou n'est pas un dictionnaire."
                )
                return {
                    "feasible": False,
                    "reason": "Configuration de palier manquante ou invalide",
                }

            logger.debug(f"[DBE_CALC] tier_config reçu: {tier_config}")

            # HIÉRARCHIE V2 : Utiliser directement les paramètres de compute_dynamic_modulation()
            risk_params = self.compute_dynamic_modulation()
            
            # Récupérer les paramètres finaux (déjà appliqués par la hiérarchie)
            position_pct = float(risk_params.get("position_size_pct", 0.1))
            sl_pct = float(risk_params.get("sl_pct", 0.02))
            tp_pct = float(risk_params.get("tp_pct", 0.04))
            
            logger.debug(
                f"[DBE_CALC_V2] Paramètres de compute_dynamic_modulation: Pos={position_pct:.2%}, SL={sl_pct:.2%}, TP={tp_pct:.2%}"
            )

            # Vérifier notional ≥ 11 USDT (hard_constraint)
            position_size_usdt = capital * position_pct
            min_trade_value = float(
                self.config.get("environment", {}).get("hard_constraints", {}).get("min_order_value_usdt", 11.0)
            )
            
            if position_size_usdt < min_trade_value:
                logger.warning(
                    f"[DBE_CALC_V2] Notional calculé ({position_size_usdt:.2f} USDT) < min_trade_value ({min_trade_value} USDT). Vérification."
                )
                if capital < min_trade_value:
                    logger.warning(
                        f"[DBE_CALC_V2] Échec: Capital ({capital:.2f} USDT) insuffisant pour le trade minimum de {min_trade_value} USDT."
                    )
                    return {
                        "feasible": False,
                        "reason": f"Capital insuffisant (min {min_trade_value} USDT requis)",
                    }
                # Remontée à min_trade_value
                position_pct = min_trade_value / capital
                position_size_usdt = min_trade_value
                logger.debug(
                    f"[DBE_CALC_V2] Notional ajusté au minimum: {position_pct:.2%} ({position_size_usdt:.2f} USDT)"
                )

            # Journaliser la décision
            logger.debug(
                f"[DBE_CALC] Paramètres de risque (immuables): SL={sl_pct:.2%}, TP={tp_pct:.2%}"
            )

            # Log the decision with the final calculated position size
            snapshot = DBESnapshot(
                step=self.state["current_step"],
                market_regime=risk_params.get("regime", "neutral"),
                risk_level=risk_params.get("risk_level", 1.0),
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                position_size_pct=position_pct,  # Use the calculated value
                reward_boost=risk_params.get("reward_boost", 1.0),
                penalty_inaction=risk_params.get("penalty_inaction", 0.0),
                metrics=self.state.get("performance_metrics", {}).copy(),
            )
            try:
                self._log_decision(snapshot, risk_params)
            except (AttributeError, TypeError):
                # Fallback: just log to decision_history
                self.decision_history.append(snapshot)

            # Définir aggressivity par défaut si non défini
            if 'aggressivity' not in locals():
                aggressivity = 1.0

            return {
                "feasible": True,
                "position_size_pct": position_pct,
                "position_size_usdt": position_size_usdt,
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "aggressivity": aggressivity,
                "risk_level": risk_params.get("risk_level", 1.0),
                "capital": capital,
                "tier": tier_config.get("name", "unknown"),
                "regime": risk_params.get("regime", "neutral"),
                "volatility": risk_params.get("volatility", 0.0),
                "risk_per_trade_pct": tier_config.get("risk_per_trade_pct", 0.01),
                "risk_horizon": risk_horizon,
                "desired_position_size": desired_position_size,
            }

        except Exception as e:
            logger.error(f"Erreur dans calculate_trade_parameters: {e}", exc_info=True)
            return {"feasible": False, "reason": f"Erreur de calcul: {str(e)}"}

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
                "initial_capital": self.finance_manager.get_balance(worker_id)
                if self.finance_manager
                else 0.0,
                "cumulative_loss": 0.0,
                "last_trade_ts": None,
                "consecutive_losses": 0,
                "trade_history": [],
            }

        state = self.worker_states[worker_id]

        # 1) Capital total < MIN_TRADE
        if state["initial_capital"] < self.MIN_TRADE:
            return (
                True,
                f"capital_below_min_trade ({state['initial_capital']:.2f} < {self.MIN_TRADE})",
            )

        # 2) Position invendable + solde insuffisant
        any_untradable = any(
            p.get("value", 0) < self.MIN_TRADE for p in state["trade_history"]
        )
        if any_untradable and state["initial_capital"] < self.MIN_TRADE:
            return (
                True,
                f"untradable_position_and_low_cash (pos<{self.MIN_TRADE} and cash<{self.MIN_TRADE})",
            )

        # 3) Cumulative loss >= palier max_drawdown
        if hasattr(self, "determine_tier"):
            tier = self.determine_tier(state["initial_capital"])
            if (
                tier
                and "max_drawdown" in tier
                and state["cumulative_loss"] >= tier["max_drawdown"]
            ):
                return (
                    True,
                    f"max_drawdown_reached (loss: {state['cumulative_loss']:.2f} >= {tier['max_drawdown']})",
                )

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
                            worker_id, pos.get("symbol", ""), pos.get("qty", 0)
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Échec de la fermeture forcée pour {worker_id} {pos.get('symbol', '')}: {str(e)}"
                        )
            except Exception as e:
                self.logger.error(
                    f"Erreur lors de la récupération des positions: {str(e)}"
                )

        # 2) Définir le nouveau capital
        if restore_capital is None:
            # Utiliser la valeur par défaut de la configuration ou 20.0 USDT
            restore_capital = self.config.get("default_reset_capital", 20.0)
            if isinstance(restore_capital, dict):
                restore_capital = restore_capital.get(worker_id, 20.0)

        if self.finance_manager:
            self.finance_manager.set_balance(worker_id, restore_capital)

        # 3) Réinitialiser l'état du worker (mode partiel pour conserver la mémoire longue)
        # On ne réinitialise que le capital et les compteurs d'épisode, pas l'historique.
        state.update(
            {
                "initial_capital": restore_capital,
                "last_trade_ts": None,
                "consecutive_losses": 0,
            }
        )
        # NOTE: 'cumulative_loss' et 'trade_history' sont intentionnellement conservés
        # pour permettre au DBE d'apprendre des échecs passés.

        self.logger.info(
            f"[RESET PARTIEL] Worker {worker_id} -> capital restauré à {restore_capital:.2f} USDT. Mémoire des erreurs conservée."
        )

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
            self.logger.warning(
                f"[RESET] Condition de reset détectée pour {worker_id}: {reason}"
            )
            self.perform_full_reset(worker_id)
            return True
        return False

    def on_trade_closed(self, trade_result: Dict[str, Any]) -> None:
        """
        Met à jour l'état après la fermeture d'un trade.

        Args:
            trade_result: Résultat du trade fermé
        """
        worker_id = trade_result.get("worker_id")
        if not worker_id or worker_id not in self.worker_states:
            return

        state = self.worker_states[worker_id]

        # Mettre à jour l'historique des trades
        state["last_trade_ts"] = datetime.now(timezone.utc)
        state["trade_history"].append(trade_result)

        # Mettre à jour les pertes cumulées
        if "pnl" in trade_result and trade_result["pnl"] < 0:
            state["cumulative_loss"] += abs(trade_result["pnl"])
            state["consecutive_losses"] += 1
        else:
            state["consecutive_losses"] = 0

    def __del__(self):
        """Nettoyage à la destruction de l'instance."""
        try:
            status = self.get_status()
            portfolio = (
                status.get("portfolio", {})
                if isinstance(status.get("portfolio"), dict)
                else {}
            )
            trading = (
                status.get("trading", {})
                if isinstance(status.get("trading"), dict)
                else {}
            )
            risk = (
                status.get("risk", {}) if isinstance(status.get("risk"), dict) else {}
            )
            step = status.get("step", 0)
            total_value = portfolio.get("total_value", 0.0)
            total_return_pct = portfolio.get("total_return_pct", 0.0)
            total_trades = trading.get("total_trades", 0)
            win_rate = trading.get("win_rate", 0.0)
            current_risk_level = risk.get("current_risk_level", 0.0)
            market_regime = risk.get("market_regime", "UNKNOWN")
            current_drawdown = risk.get("current_drawdown", 0.0)
            volatility = risk.get("volatility", 0.0)
            return (
                f"DBE Status (Step: {step})\n"
                f"Portfolio: ${total_value:,.2f} (Return: {total_return_pct:.2f}%)\n"
                f"Trades: {total_trades} (Win Rate: {win_rate:.1f}%)\n"
                f"Risk: {current_risk_level:.2f} (Regime: {market_regime})\n"
                f"Drawdown: {current_drawdown:.2f}% | Volatility: {volatility:.4f}"
            )
        except Exception:
            return "DBE destroyed"

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
                if (
                    key in current
                    and isinstance(current[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_update(current[key], value)
                else:
                    current[key] = value

        # Application de la mise à jour
        deep_update(self.config, new_config)
        self.log_info("Configuration du DBE mise à jour")

        # Mise à jour du niveau de log si nécessaire
        if "logging" in new_config and "log_level" in new_config["logging"]:
            log_level = new_config["logging"]["log_level"].upper()
            logging.getLogger().setLevel(getattr(logging, log_level))

    def _compute_risk_parameters(
        self,
        state: Dict[str, Any],
        mod: Dict[str, Any],
        risk_horizon: float = 0.0,
    ) -> None:
        """
        Calcule les paramètres de risque dynamiques (SL/TP).

        Args:
            state: Dictionnaire contenant l'état actuel
            mod: Dictionnaire à mettre à jour avec les nouveaux paramètres de risque
            risk_horizon: Horizon de risque choisi par l'agent (-1: court terme, 1: long terme)
        """
        try:
            if state is None or mod is None:
                logger.warning("State ou mod est None dans _compute_risk_parameters")
                return

            # Récupération des configurations
            risk_cfg = self.config.get("risk_parameters", {})
            regime_params = self.config.get("regime_parameters", {}).get(
                self.current_regime, {}
            )

            # Paramètres de base
            base_sl = float(risk_cfg.get("base_sl_pct", 0.02))
            base_tp = float(risk_cfg.get("base_tp_pct", 0.04))
            
            # ⚠️ IMPORTANT: SL/TP sont IMMUABLES - pas de modulation dynamique
            # Les valeurs SL/TP sont déterminées par Optuna pour chaque worker
            # et appliquées directement par le portfolio manager
            # Forcer les valeurs de base sans modulation
            mod["sl_pct"] = base_sl
            mod["tp_pct"] = base_tp
            return  # EXIT EARLY - pas de modulation SL/TP

            # Initialisation des valeurs d'état avec des valeurs par défaut si manquantes
            current_drawdown = float(state.get("drawdown", 0.0))
            volatility = float(state.get("volatility", 0.0))
            win_rate = float(state.get("win_rate", 0.5))  # 50% par défaut
            sharpe_ratio = float(state.get("sharpe_ratio", 0.0))
            current_step = int(state.get("current_step", 0))

            # ACCUMULATION VOLATILITÉ: Stocker dans l'historique
            if volatility > 0.0:
                if not hasattr(self, "volatility_history"):
                    self.volatility_history = []
                # Éviter les doublons
                if (
                    not self.volatility_history
                    or abs(self.volatility_history[-1] - volatility) > 1e-6
                ):
                    self.volatility_history.append(volatility)
                    logger.debug(
                        f"[VOL HISTORY] Ajouté: {volatility:.4f}, Total: {len(self.volatility_history)} points"
                    )

            # Récupération des multiplicateurs spécifiques au régime
            sl_multiplier = float(regime_params.get("sl_multiplier", 1.0))
            tp_multiplier = float(regime_params.get("tp_multiplier", 1.0))

            # === NOUVEAU: Ajustement basé sur l'horizon de risque de l'agent ===
            # risk_horizon est entre -1 (court terme) et 1 (long terme)
            # Pour le SL: plus l'horizon est court (-1), plus le SL est serré (multiplicateur > 1)
            #             plus l'horizon est long (1), plus le SL est large (multiplicateur < 1)
            sl_multiplier_rh = 1.0 - (
                risk_horizon * 0.3
            )  # Ex: -1 -> 1.3, 0 -> 1.0, 1 -> 0.7
            # Pour le TP: plus l'horizon est court (-1), plus le TP est serré (multiplicateur < 1)
            #             plus l'horizon est long (1), plus le TP est large (multiplicateur > 1)
            tp_multiplier_rh = 1.0 + (
                risk_horizon * 0.3
            )  # Ex: -1 -> 0.7, 0 -> 1.0, 1 -> 1.3

            sl_multiplier *= sl_multiplier_rh
            tp_multiplier *= tp_multiplier_rh
            # ==================================================================

            # 1. Ajustement basé sur le drawdown
            max_drawdown = float(risk_cfg.get("max_drawdown", 0.1))  # 10% par défaut
            drawdown_factor = 1.0 - min(
                current_drawdown / (max_drawdown * 2), 0.5
            )  # Réduction jusqu'à 50%

            # 2. Ajustement basé sur la volatilité
            vol_management = self.config.get("volatility_management", {})
            min_vol = float(vol_management.get("min_volatility", 0.01))
            max_vol = float(vol_management.get("max_volatility", 0.20))
            vol_factor = (
                1.0 - ((volatility - min_vol) / (max_vol - min_vol + 1e-6)) * 0.5
            )  # Réduction jusqu'à 50%

            # 3. Ajustement basé sur le win rate
            target_win_rate = 0.6  # Cible de 60% de trades gagnants
            win_rate_factor = (win_rate / target_win_rate) ** 2  # Effet non linéaire

            # 4. Ajustement basé sur le ratio de Sharpe
            sharpe_factor = 1.0 + (
                max(0, sharpe_ratio) / 2.0
            )  # Améliore le risque avec un meilleur Sharpe

            # Calcul des nouveaux paramètres avec contraintes
            min_sl = float(risk_cfg.get("min_sl_pct", 0.005))  # 0.5% minimum
            max_sl = float(risk_cfg.get("max_sl_pct", 0.10))  # 10% maximum
            min_tp = float(risk_cfg.get("min_tp_pct", 0.01))  # 1% minimum
            max_tp = float(risk_cfg.get("max_tp_pct", 0.20))  # 20% maximum

            # Calcul des nouvelles valeurs
            new_sl = base_sl * sl_multiplier * drawdown_factor * vol_factor
            new_tp = base_tp * tp_multiplier * win_rate_factor * sharpe_factor

            # Application des limites
            new_sl = max(min_sl, min(max_sl, new_sl))
            new_tp = max(min_tp, min(max_tp, new_tp))

            # Vérification de l'initialisation de smoothed_params
            if not hasattr(self, "smoothed_params"):
                self.smoothed_params = {
                    "sl_pct": base_sl,
                    "tp_pct": base_tp,
                    "position_size": 0.1,
                    "risk_level": 1.0,
                }

            # Application du lissage exponentiel
            smoothing = self.config.get("smoothing", {}).get("adaptation_rate", 0.1)

            # Mise à jour des paramètres lissés
            for param, new_val in [("sl_pct", new_sl), ("tp_pct", new_tp)]:
                if param in self.smoothed_params:
                    self.smoothed_params[param] = (
                        1.0 - smoothing
                    ) * self.smoothed_params[param] + smoothing * new_val
                else:
                    self.smoothed_params[param] = new_val

            # Calcul du coefficient d'agressivité (0-1) basé sur plusieurs facteurs
            # 1. Facteur de confiance (winrate récent)
            winrate_factor = min(
                1.0, max(0.0, (win_rate - 0.4) / 0.6) # 0% à 100% pour winrate de 0.4 à 1.0
            )

            # 2. Facteur de drawdown (pénalise les périodes de pertes)
            drawdown_factor = 1.0 - (
                self.state["drawdown"] / 100.0 * 2
            )  # Réduit la taille avec le drawdown

            # 3. Facteur de volatilité (pénalise la volatilité élevée)
            vol_factor = 1.0 / (
                1.0 + self.state["volatility"] * 10
            )  # Réduit la taille avec la volatilité

            # 4. Facteur de régime de marché
            regime_factors = {
                "bull": 1.0,
                "bear": 0.3,
                "volatile": 0.5,
                "sideways": 0.7,
                "neutral": 0.8,
            }
            regime_factor = regime_factors.get(self.current_regime.lower(), 0.5)

            # Calcul final du coefficient d'agressivité (0-1)
            aggressivity = (
                winrate_factor * 0.4
                + drawdown_factor * 0.3
                + volatility_factor * 0.2
                + regime_factor * 0.1
            )

            # Lissage du coefficient d'agressivité
            if "aggressivity" not in self.smoothed_params:
                self.smoothed_params["aggressivity"] = 0.5  # Valeur par défaut

            smoothing = self.config.get("smoothing", {}).get("adaptation_rate", 0.1)
            self.smoothed_params["aggressivity"] = (
                1.0 - smoothing
            ) * self.smoothed_params["aggressivity"] + smoothing * aggressivity

            # Mise à jour du dictionnaire de sortie avec les valeurs lissées
            mod.update(
                {
                    "sl_pct": self.smoothed_params.get("sl_pct", base_sl),
                    "tp_pct": self.smoothed_params.get("tp_pct", base_tp),
                    "risk_level": self.state.get("current_risk_level", 1.0),
                    "regime": self.current_regime,
                    "volatility": volatility,
                    "aggressivity": self.smoothed_params["aggressivity"],
                }
            )

            # Journalisation des changements importants (tous les 50 pas)
            if current_step > 0 and current_step % 50 == 0:
                logger.info(
                    f"🔧 Paramètres de risque - "
                    f"Régime: {self.current_regime.upper()} | "
                    f"Drawdown: {current_drawdown:.2f}% | "
                    f"Volatilité: {volatility:.2f}% | "
                    f"Win Rate: {win_rate:.1f}% | "
                    f"Sharpe: {sharpe_ratio:.2f} | "
                    f"SL: {new_sl:.2f}% (lissé: {mod['sl_pct']:.2f}%) | "
                    f"TP: {new_tp:.2f}% (lissé: {mod['tp_pct']:.2f}%) | "
                    f"Niveau de risque: {mod['risk_level']:.2f}"
                )

        except Exception as e:
            logger.error(
                f"Erreur dans _compute_risk_parameters: {str(e)}", exc_info=True
            )
            # En cas d'erreur, on utilise les valeurs par défaut
            mod.update(
                {
                    "sl_pct": risk_cfg.get("base_sl_pct", 0.02),
                    "tp_pct": risk_cfg.get("base_tp_pct", 0.04),
                    "risk_level": 1.0,
                    "regime": self.current_regime,
                    "volatility": 0.0,
                }
            )

    def _compute_reward_modulation(self, mod: Dict[str, Any]) -> None:
        """Calcule la modulation des récompenses."""
        # Paramètres configurables
        reward_config = self.config.get("reward", {})
        winrate_threshold = reward_config.get("winrate_threshold", 0.55)
        max_boost = reward_config.get("max_boost", 2.0)

        # Reward boost basé sur le winrate
        if self.state.get("winrate", 0.0) > winrate_threshold:
            boost_factor = min(
                max_boost, 1.0 + (self.state["winrate"] - winrate_threshold) * 5.0
            )
            mod["reward_boost"] = boost_factor
        else:
            mod["reward_boost"] = 1.0

        # Pénalité d'inaction progressive
        inaction_factor = reward_config.get("inaction_factor", 0.1)
        action_freq = self.state.get(
            "action_frequency", 1.0
        )  # Default to 1 to avoid penalty if not present
        min_action_freq = reward_config.get("min_action_frequency", 0.1)

        if action_freq < min_action_freq and self.state.get("market_regime") in [
            "BULL",
            "BEAR",
        ]:
            # Pénalité progressive basée sur la fréquence d'action
            mod["penalty_inaction"] = (
                -inaction_factor * (min_action_freq - action_freq) * 10
            )
        else:
            mod["penalty_inaction"] = 0.0

    def _compute_position_sizing(self, mod: Dict[str, Any]) -> None:
        """
        Calcule la taille de position dynamique.

        Args:
            mod: Dictionnaire des paramètres modulés à mettre à jour
        """
        sizing_cfg = self.config.get("position_sizing", {})
        base_size = sizing_cfg.get("base_position_size", 0.1)  # 10% par défaut

        # Ajustement basé sur la confiance (winrate récent)
        confidence_factor = min(2.0, max(0.5, self.state["winrate"] / 0.5))  # 0.5-2.0x

        # Ajustement basé sur le drawdown
        drawdown_factor = 1.0 - (
            self.state["drawdown"] / 100.0 * 2
        )  # Réduit la taille avec le drawdown

        # Ajustement basé sur la volatilité
        vol_factor = 1.0 / (
            1.0 + self.state["volatility"] * 10
        )  # Réduit la taille avec la volatilité

        # Calcul final avec limites
        mod["position_size_pct"] = max(
            sizing_cfg.get("min_position_size", 0.01),
            min(
                sizing_cfg.get("max_position_size", 0.3),
                base_size * confidence_factor * drawdown_factor * vol_factor,
            ),
        )

    def _compute_risk_mode(self, mod: Dict[str, Any]) -> None:
        """
        Détermine le mode de risque global (DEFENSIVE, NORMAL, AGGRESSIVE).

        Args:
            mod: Dictionnaire des paramètres modulés à mettre à jour
        """
        # Mode défensif si drawdown élevé ou pertes consécutives
        if self.state["drawdown"] > 10.0 or self.state["consecutive_losses"] >= 3:
            mod["risk_mode"] = "DEFENSIVE"
            mod["position_size_pct"] *= 0.5  # Reduce position size
            # ⚠️ IMPORTANT: SL/TP sont IMMUABLES - pas de modulation
            # mod["sl_pct"] *= 1.2  # DISABLED - Tighten stop loss
        # Mode agressif si bonnes performances et faible drawdown
        elif self.state["winrate"] > 0.7 and self.state["drawdown"] < 2.0:
            mod["risk_mode"] = "AGGRESSIVE"
            mod["position_size_pct"] *= 1.2  # Increase position size
            # ⚠️ IMPORTANT: SL/TP sont IMMUABLES - pas de modulation
            # mod["tp_pct"] *= 1.2  # DISABLED - Loosen take profit
        else:
            mod["risk_mode"] = "NORMAL"

    def _apply_market_regime_modifiers(self, mod: Dict[str, Any]) -> None:
        """
        Applique des ajustements spécifiques au régime de marché.

        Args:
            mod: Dictionnaire des paramètres modulés à mettre à jour
        """
        regime = self.state.get("market_regime", "NORMAL")
        regime_cfg = self.config.get("modes", {}).get(regime.lower(), {})

        if not regime_cfg:
            return

        mod["position_size_pct"] *= regime_cfg.get("position_size_multiplier", 1.0)
        # ⚠️ IMPORTANT: SL/TP sont IMMUABLES - pas de modulation par régime
        # mod["sl_pct"] *= regime_cfg.get("sl_multiplier", 1.0)  # DISABLED
        # mod["tp_pct"] *= regime_cfg.get("tp_multiplier", 1.0)  # DISABLED

        # Specific adjustments for trending markets
        if regime == "BULL" or regime == "BEAR":
            mod["trailing_stop"] = True  # Activate trailing stop in trending markets

    def reset_chunk(self) -> None:
        """Réinitialise les métriques au début d'un nouveau chunk."""
        # Conserver certaines métriques (comme le winrate) mais réinitialiser les autres
        self.state.update(
            {
                "current_step": 0,
                "chunk_optimal_pnl": 0.0,
                "position_size_pct": self.config.get("position_sizing", {}).get(
                    "base_position_size", 0.1
                ),
            }
        )
        self.log_info("🔄 DBE: Nouveau chunk - réinitialisation des métriques")

    def _log_dbe_state(self, modulation: Dict[str, Any]) -> None:
        """
        Logs the current state and modulation of the DBE to a JSONL file.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": self.state.get("current_step", 0),
            "drawdown": self.state.get("drawdown", 0.0),
            "winrate": self.state.get("winrate", 0.0),
            "volatility": self.state.get("volatility", 0.0),
            "market_regime": self.state.get("market_regime", "NORMAL"),
            "sl_pct": modulation.get("sl_pct", 0.0),
            "tp_pct": modulation.get("tp_pct", 0.0),
            "reward_boost": modulation.get("reward_boost", 0.0),
            "penalty_inaction": modulation.get("penalty_inaction", 0.0),
            "position_size_pct": modulation.get("position_size_pct", 0.0),
            "risk_mode": modulation.get("risk_mode", "NORMAL"),
        }
        try:
            self.dbe_log_file.write(json.dumps(log_entry, cls=NpEncoder) + "\n")
            self.dbe_log_file.flush()  # Ensure data is written to disk immediately
        except Exception as e:
            logger.error(f"Error writing to DBE log file: {e}")