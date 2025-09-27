"""
Patch Gugu & March - Système de Récompenses d'Excellence pour ADAN

Ce module implémente des mécanismes avancés de récompense qui guident ADAN
vers l'excellence plutôt que de simplement punir l'échec.

Philosophie :
- Gugu : Récompenser la maîtrise technique (Sharpe, consistance, confluence)
- March : Encourager la joie de la victoire (séries gagnantes, momentum)

Auteur: L'Homme en Noir
Date: 2024
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExcellenceMetrics:
    """Métriques pour le calcul des bonus d'excellence"""

    sharpe_ratio: float = 0.0
    profit_factor: float = 1.0
    win_rate: float = 0.5
    winning_streak: int = 0
    total_trades: int = 0
    recent_pnl_history: List[float] = None
    current_drawdown: float = 0.0
    timeframe_signals: Dict[str, bool] = None

    def __post_init__(self):
        if self.recent_pnl_history is None:
            self.recent_pnl_history = []
        if self.timeframe_signals is None:
            self.timeframe_signals = {"5m": False, "1h": False, "4h": False}


class GuguMarchExcellenceRewards:
    """
    Système de récompenses d'excellence qui transforme ADAN d'un survivant
    en un maître du trading.
    """

    def __init__(self, config: Dict):
        """
        Initialise le système de récompenses d'excellence

        Args:
            config: Configuration des bonus d'excellence du config.yaml
        """
        self.config = config.get("excellence_bonuses", {})

        # Configuration du bonus de série de victoires (March)
        self.winning_streak_config = self.config.get("winning_streak_bonus", {})
        self.streak_enabled = self.winning_streak_config.get("enabled", True)
        self.min_streak = self.winning_streak_config.get("min_streak", 3)
        self.streak_multiplier = self.winning_streak_config.get(
            "streak_multiplier", 0.3
        )
        self.max_streak_bonus = self.winning_streak_config.get("max_streak_bonus", 2.0)

        # Configuration du bonus de confluence (Gugu)
        self.confluence_config = self.config.get("confluence_bonus", {})
        self.confluence_enabled = self.confluence_config.get("enabled", True)
        self.multi_tf_weight = self.confluence_config.get("multi_timeframe_weight", 0.4)
        self.trend_alignment_weight = self.confluence_config.get(
            "trend_alignment_weight", 0.2
        )

        # Configuration de la récompense de consistance (Gugu)
        self.consistency_config = self.config.get("consistency_reward", {})
        self.consistency_enabled = self.consistency_config.get("enabled", True)
        self.min_trades_for_bonus = self.consistency_config.get(
            "min_trades_for_bonus", 10
        )
        self.sharpe_threshold = self.consistency_config.get("sharpe_threshold", 1.5)
        self.consistency_multiplier = self.consistency_config.get(
            "consistency_multiplier", 0.5
        )

        # Historique pour le calcul de la consistance
        self.performance_history = []
        self.last_winning_streak = 0

        logger.info(f"[GUGU-MARCH] Excellence Rewards System initialized")
        logger.info(
            f"[GUGU-MARCH] Streak bonus: {self.streak_enabled}, Confluence: {self.confluence_enabled}, Consistency: {self.consistency_enabled}"
        )

    def calculate_excellence_bonus(
        self, base_reward: float, metrics: ExcellenceMetrics, trade_won: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calcule le bonus d'excellence total à ajouter à la récompense de base

        Args:
            base_reward: Récompense de base du trade/step
            metrics: Métriques de performance actuelles
            trade_won: Si ce step correspond à un trade gagnant

        Returns:
            Tuple de (bonus_total, détail_des_bonus)
        """
        bonus_breakdown = {}
        total_bonus = 0.0

        # 1. Bonus de Performance Sharpe (Guide de Gugu)
        sharpe_bonus = self._calculate_sharpe_excellence_bonus(metrics.sharpe_ratio)
        if sharpe_bonus > 0:
            bonus_breakdown["sharpe_excellence"] = sharpe_bonus
            total_bonus += sharpe_bonus

        # 2. Bonus de Série de Victoires (Encouragement de March)
        if trade_won:
            streak_bonus = self._calculate_winning_streak_bonus(
                metrics.winning_streak, base_reward
            )
            if streak_bonus > 0:
                bonus_breakdown["winning_streak"] = streak_bonus
                total_bonus += streak_bonus

        # 3. Bonus de Confluence Multi-Timeframe (Vision du Monde)
        confluence_bonus = self._calculate_confluence_bonus(
            metrics.timeframe_signals, base_reward
        )
        if confluence_bonus > 0:
            bonus_breakdown["confluence"] = confluence_bonus
            total_bonus += confluence_bonus

        # 4. Bonus de Consistance (Maîtrise de Gugu)
        consistency_bonus = self._calculate_consistency_bonus(metrics)
        if consistency_bonus > 0:
            bonus_breakdown["consistency"] = consistency_bonus
            total_bonus += consistency_bonus

        # 5. Bonus de Profit Factor (Efficacité Globale)
        pf_bonus = self._calculate_profit_factor_bonus(
            metrics.profit_factor, base_reward
        )
        if pf_bonus > 0:
            bonus_breakdown["profit_factor"] = pf_bonus
            total_bonus += pf_bonus

        # Log détaillé si bonus significatif
        if total_bonus > 0.1:
            logger.info(
                f"[GUGU-MARCH] Excellence bonus: {total_bonus:.4f} | Breakdown: {bonus_breakdown}"
            )

        return total_bonus, bonus_breakdown

    def _calculate_sharpe_excellence_bonus(self, sharpe_ratio: float) -> float:
        """
        Bonus exponentiel pour Sharpe Ratio élevé (Guide de Gugu)

        Philosophie : Un Sharpe > 1.0 indique une maîtrise technique.
        Plus il est élevé, plus la récompense doit être disproportionnée.
        """
        if sharpe_ratio <= 1.0:
            return 0.0

        # Bonus exponentiel : (Sharpe - 1.0)² * multiplicateur
        excess_sharpe = sharpe_ratio - 1.0
        sharpe_bonus = (excess_sharpe**2) * 0.5

        # Plafonner à un bonus raisonnable
        return min(sharpe_bonus, 2.0)

    def _calculate_winning_streak_bonus(
        self, streak_count: int, base_reward: float
    ) -> float:
        """
        Bonus multiplicateur pour séries de victoires (Encouragement de March)

        Philosophie : La joie et la fierté d'une série de victoires doivent
        être récompensées de manière croissante.
        """
        if (
            not self.streak_enabled
            or streak_count < self.min_streak
            or base_reward <= 0
        ):
            return 0.0

        # Multiplicateur croissant avec la série
        streak_multiplier = min(
            (streak_count - self.min_streak + 1) * self.streak_multiplier,
            self.max_streak_bonus,
        )

        streak_bonus = base_reward * streak_multiplier

        logger.debug(
            f"[MARCH] Winning streak {streak_count}: +{streak_bonus:.4f} bonus"
        )
        return streak_bonus

    def _calculate_confluence_bonus(
        self, timeframe_signals: Dict[str, bool], base_reward: float
    ) -> float:
        """
        Bonus pour confluence multi-timeframe (Vision du Monde)

        Philosophie : Un trade confirmé sur plusieurs échelles de temps
        est de meilleure qualité et doit être récompensé.
        """
        if not self.confluence_enabled or not timeframe_signals or base_reward <= 0:
            return 0.0

        # Compter les timeframes en confluence
        active_signals = sum(1 for signal in timeframe_signals.values() if signal)

        if active_signals < 2:  # Pas de confluence
            return 0.0

        # Bonus proportionnel au nombre de timeframes alignés
        confluence_strength = (active_signals - 1) / len(timeframe_signals)
        confluence_bonus = base_reward * self.multi_tf_weight * confluence_strength

        logger.debug(
            f"[GUGU] Confluence {active_signals}/{len(timeframe_signals)} TFs: +{confluence_bonus:.4f} bonus"
        )
        return confluence_bonus

    def _calculate_consistency_bonus(self, metrics: ExcellenceMetrics) -> float:
        """
        Bonus pour consistance des performances (Maîtrise de Gugu)

        Philosophie : La régularité dans l'excellence doit être
        récompensée au-delà des performances ponctuelles.
        """
        if (
            not self.consistency_enabled
            or metrics.total_trades < self.min_trades_for_bonus
        ):
            return 0.0

        if metrics.sharpe_ratio < self.sharpe_threshold:
            return 0.0

        # Bonus basé sur la combinaison Sharpe élevé + nombre de trades
        trade_volume_factor = min(
            metrics.total_trades / 50.0, 1.0
        )  # Plafonné à 50 trades
        consistency_bonus = (
            (metrics.sharpe_ratio - self.sharpe_threshold)
            * self.consistency_multiplier
            * trade_volume_factor
        )

        return consistency_bonus

    def _calculate_profit_factor_bonus(
        self, profit_factor: float, base_reward: float
    ) -> float:
        """
        Bonus pour Profit Factor élevé

        Philosophie : PF > 1.0 = profitable, plus c'est élevé,
        plus c'est remarquable.
        """
        if profit_factor <= 1.0 or base_reward <= 0:
            return 0.0

        # Bonus logarithmique pour éviter l'explosion
        pf_excess = profit_factor - 1.0
        pf_bonus = base_reward * np.log(1 + pf_excess) * 0.3

        return min(pf_bonus, base_reward * 1.0)  # Plafonné à 100% du base_reward

    def update_streak_counter(self, trade_won: bool, trade_lost: bool = False) -> int:
        """
        Met à jour le compteur de série de victoires

        Args:
            trade_won: True si le trade est gagnant
            trade_lost: True si le trade est perdant

        Returns:
            Nouveau compte de série de victoires
        """
        if trade_won:
            self.last_winning_streak += 1
        elif trade_lost:
            if self.last_winning_streak >= self.min_streak:
                logger.info(
                    f"[MARCH] End of winning streak: {self.last_winning_streak} trades"
                )
            self.last_winning_streak = 0

        return self.last_winning_streak

    def analyze_timeframe_confluence(
        self,
        signals_5m: Dict,
        signals_1h: Dict,
        signals_4h: Dict,
        current_action: float,
    ) -> Dict[str, bool]:
        """
        Analyse la confluence entre différents timeframes

        Args:
            signals_5m: Signaux du timeframe 5m
            signals_1h: Signaux du timeframe 1h
            signals_4h: Signaux du timeframe 4h
            current_action: Action actuelle prise (-1 à 1)

        Returns:
            Dict indiquant quels timeframes sont en confluence avec l'action
        """
        confluence_signals = {}
        action_direction = (
            1 if current_action > 0.1 else (-1 if current_action < -0.1 else 0)
        )

        if action_direction == 0:  # Pas d'action claire
            return {"5m": False, "1h": False, "4h": False}

        # Analyser chaque timeframe pour confluence
        for tf_name, signals in [
            ("5m", signals_5m),
            ("1h", signals_1h),
            ("4h", signals_4h),
        ]:
            if not signals:
                confluence_signals[tf_name] = False
                continue

            # Exemple de logique de confluence (à adapter selon vos indicateurs)
            rsi = signals.get("rsi", 50)
            macd_histogram = signals.get("macd_histogram", 0)

            if action_direction > 0:  # Action d'achat
                # Confluence haussière : RSI pas en surachat + MACD positif
                confluence_signals[tf_name] = rsi < 70 and macd_histogram > 0
            else:  # Action de vente
                # Confluence baissière : RSI pas en survente + MACD négatif
                confluence_signals[tf_name] = rsi > 30 and macd_histogram < 0

        return confluence_signals

    def get_performance_summary(self) -> Dict[str, float]:
        """
        Retourne un résumé des performances du système d'excellence
        """
        return {
            "current_winning_streak": self.last_winning_streak,
            "streak_bonus_active": self.last_winning_streak >= self.min_streak,
            "performance_history_length": len(self.performance_history),
            "system_enabled": True,
        }

    def reset_metrics(self):
        """Remet à zéro les métriques (utile en cas de reset du portfolio)"""
        self.last_winning_streak = 0
        self.performance_history.clear()
        logger.info("[GUGU-MARCH] Metrics reset")


# Fonction utilitaire pour intégration facile
def create_excellence_rewards_system(config: Dict) -> GuguMarchExcellenceRewards:
    """
    Crée et initialise le système de récompenses d'excellence

    Args:
        config: Configuration globale du bot

    Returns:
        Instance configurée du système d'excellence
    """
    reward_config = config.get("reward_shaping", {})
    return GuguMarchExcellenceRewards(reward_config)


# Exemple d'utilisation dans un environnement de trading
def example_integration():
    """
    Exemple d'intégration dans le système de récompense existant
    """
    config = {
        "reward_shaping": {
            "excellence_bonuses": {
                "winning_streak_bonus": {
                    "enabled": True,
                    "min_streak": 3,
                    "streak_multiplier": 0.3,
                    "max_streak_bonus": 2.0,
                },
                "confluence_bonus": {
                    "enabled": True,
                    "multi_timeframe_weight": 0.4,
                    "trend_alignment_weight": 0.2,
                },
                "consistency_reward": {
                    "enabled": True,
                    "min_trades_for_bonus": 10,
                    "sharpe_threshold": 1.5,
                    "consistency_multiplier": 0.5,
                },
            }
        }
    }

    excellence_system = create_excellence_rewards_system(config)

    # Simulation d'utilisation
    metrics = ExcellenceMetrics(
        sharpe_ratio=1.8,
        profit_factor=1.15,
        winning_streak=4,
        total_trades=25,
        timeframe_signals={"5m": True, "1h": True, "4h": False},
    )

    base_reward = 0.5  # Récompense de base d'un trade
    total_bonus, breakdown = excellence_system.calculate_excellence_bonus(
        base_reward, metrics, trade_won=True
    )

    final_reward = base_reward + total_bonus
    print(
        f"Base reward: {base_reward}, Excellence bonus: {total_bonus}, Final: {final_reward}"
    )
    print(f"Bonus breakdown: {breakdown}")


if __name__ == "__main__":
    # Test du système
    example_integration()
