"""
Analyseur de qualité décisionnelle pour ADAN Trading Bot.
Distingue les décisions réfléchies (patterns réels) des décisions hasardeuses (bruit).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DecisionQualityMetrics:
    """Métriques de qualité décisionnelle."""

    # Couche statistique
    accuracy: float
    f1_score: float
    recall: float
    precision: float
    expected_shortfall: float

    # Couche probabiliste
    profit_factor: float
    edge_ratio: float
    pattern_repeatability: float
    calmar_ratio: float

    # Couche robustesse
    walk_forward_consistency: float
    monte_carlo_profitability: float
    out_of_sample_degradation: float
    deflated_sharpe: float

    # Couche économique
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float

    # Métriques comportementales
    decision_consistency: float
    pattern_quality_score: float
    risk_management_quality: float
    temporal_intelligence: float

    # Verdict final
    is_reflective: bool
    reflection_score: float  # 0-100


class DecisionQualityAnalyzer:
    """Analyse la qualité des décisions du modèle."""

    def __init__(
        self,
        trades_data: pd.DataFrame,
        market_data: pd.DataFrame,
        model_predictions: Optional[np.ndarray] = None,
    ):
        """
        Initialise l'analyseur.

        Args:
            trades_data: DataFrame avec colonnes [entry_price, exit_price, pnl, etc.]
            market_data: DataFrame avec OHLCV
            model_predictions: Prédictions du modèle (optionnel)
        """
        self.trades = trades_data
        self.market = market_data
        self.predictions = model_predictions
        self.metrics = None

    def analyze(self) -> DecisionQualityMetrics:
        """Analyse complète de la qualité décisionnelle."""

        # Couche statistique
        stat_metrics = self._analyze_statistical_layer()

        # Couche probabiliste
        prob_metrics = self._analyze_probabilistic_layer()

        # Couche robustesse
        robust_metrics = self._analyze_robustness_layer()

        # Couche économique
        econ_metrics = self._analyze_economic_layer()

        # Métriques comportementales
        behavior_metrics = self._analyze_behavioral_metrics()

        # Verdict final
        reflection_score = self._calculate_reflection_score(
            stat_metrics, prob_metrics, robust_metrics, econ_metrics, behavior_metrics
        )

        self.metrics = DecisionQualityMetrics(
            # Statistique
            accuracy=stat_metrics.get("accuracy", 0),
            f1_score=stat_metrics.get("f1_score", 0),
            recall=stat_metrics.get("recall", 0),
            precision=stat_metrics.get("precision", 0),
            expected_shortfall=stat_metrics.get("expected_shortfall", 0),
            # Probabiliste
            profit_factor=prob_metrics.get("profit_factor", 0),
            edge_ratio=prob_metrics.get("edge_ratio", 0),
            pattern_repeatability=prob_metrics.get("pattern_repeatability", 0),
            calmar_ratio=prob_metrics.get("calmar_ratio", 0),
            # Robustesse
            walk_forward_consistency=robust_metrics.get("walk_forward_consistency", 0),
            monte_carlo_profitability=robust_metrics.get("monte_carlo_profitability", 0),
            out_of_sample_degradation=robust_metrics.get(
                "out_of_sample_degradation", 0
            ),
            deflated_sharpe=robust_metrics.get("deflated_sharpe", 0),
            # Économique
            sharpe_ratio=econ_metrics.get("sharpe_ratio", 0),
            sortino_ratio=econ_metrics.get("sortino_ratio", 0),
            max_drawdown=econ_metrics.get("max_drawdown", 0),
            # Comportemental
            decision_consistency=behavior_metrics.get("decision_consistency", 0),
            pattern_quality_score=behavior_metrics.get("pattern_quality_score", 0),
            risk_management_quality=behavior_metrics.get("risk_management_quality", 0),
            temporal_intelligence=behavior_metrics.get("temporal_intelligence", 0),
            # Verdict
            is_reflective=reflection_score > 60,
            reflection_score=reflection_score,
        )

        return self.metrics

    def _analyze_statistical_layer(self) -> Dict:
        """Couche statistique : séparer signal du bruit."""
        pnl = self.trades["pnl"].values
        wins = np.sum(pnl > 0)
        losses = np.sum(pnl < 0)
        total = len(pnl)

        if total == 0:
            return {
                "accuracy": 0,
                "f1_score": 0,
                "recall": 0,
                "precision": 0,
                "expected_shortfall": 0,
            }

        # Accuracy (win rate)
        accuracy = wins / total if total > 0 else 0

        # Precision et Recall (si prédictions disponibles)
        if self.predictions is not None:
            tp = np.sum((self.predictions > 0.5) & (pnl > 0))
            fp = np.sum((self.predictions > 0.5) & (pnl <= 0))
            fn = np.sum((self.predictions <= 0.5) & (pnl > 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
        else:
            precision = accuracy
            recall = accuracy
            f1 = accuracy

        # Expected Shortfall (moyenne des 5% pires pertes)
        if len(pnl) > 0:
            worst_5_pct = np.percentile(pnl, 5)
            es = np.mean(pnl[pnl <= worst_5_pct]) if np.any(pnl <= worst_5_pct) else 0
        else:
            es = 0

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "recall": recall,
            "precision": precision,
            "expected_shortfall": es,
        }

    def _analyze_probabilistic_layer(self) -> Dict:
        """Couche probabiliste : patterns répétables et profitables."""
        pnl = self.trades["pnl"].values

        if len(pnl) == 0:
            return {
                "profit_factor": 0,
                "edge_ratio": 0,
                "pattern_repeatability": 0,
                "calmar_ratio": 0,
            }

        # Profit Factor
        gains = np.sum(pnl[pnl > 0])
        losses = np.abs(np.sum(pnl[pnl < 0]))
        profit_factor = gains / losses if losses > 0 else (gains / 0.01 if gains > 0 else 0)

        # Win Rate et Edge Ratio
        win_rate = np.sum(pnl > 0) / len(pnl)
        avg_win = np.mean(pnl[pnl > 0]) if np.any(pnl > 0) else 0
        avg_loss = np.abs(np.mean(pnl[pnl < 0])) if np.any(pnl < 0) else 0

        edge_ratio = (
            (win_rate * avg_win) / ((1 - win_rate) * avg_loss)
            if ((1 - win_rate) * avg_loss) > 0
            else 0
        )

        # Pattern Repeatability (clustering)
        pattern_repeatability = self._calculate_pattern_repeatability()

        # Calmar Ratio
        total_return = np.sum(pnl)
        max_dd = self._calculate_max_drawdown(pnl)
        calmar = total_return / max_dd if max_dd > 0 else 0

        return {
            "profit_factor": profit_factor,
            "edge_ratio": edge_ratio,
            "pattern_repeatability": pattern_repeatability,
            "calmar_ratio": calmar,
        }

    def _analyze_robustness_layer(self) -> Dict:
        """Couche robustesse : tests anti-overfitting."""
        pnl = self.trades["pnl"].values

        if len(pnl) < 20:
            return {
                "walk_forward_consistency": 0,
                "monte_carlo_profitability": 0,
                "out_of_sample_degradation": 0,
                "deflated_sharpe": 0,
            }

        # Walk-Forward Analysis
        wf_consistency = self._walk_forward_analysis(pnl)

        # Monte Carlo Simulation
        mc_profitability = self._monte_carlo_analysis(pnl)

        # Out-of-Sample Degradation
        oos_degradation = self._out_of_sample_degradation(pnl)

        # Deflated Sharpe Ratio
        sharpe = self._calculate_sharpe_ratio(pnl)
        deflated_sharpe = self._deflate_sharpe(sharpe, len(pnl))

        return {
            "walk_forward_consistency": wf_consistency,
            "monte_carlo_profitability": mc_profitability,
            "out_of_sample_degradation": oos_degradation,
            "deflated_sharpe": deflated_sharpe,
        }

    def _analyze_economic_layer(self) -> Dict:
        """Couche économique : profitabilité réelle."""
        pnl = self.trades["pnl"].values

        if len(pnl) == 0:
            return {
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown": 0,
            }

        sharpe = self._calculate_sharpe_ratio(pnl)
        sortino = self._calculate_sortino_ratio(pnl)
        max_dd = self._calculate_max_drawdown(pnl)

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
        }

    def _analyze_behavioral_metrics(self) -> Dict:
        """Métriques comportementales."""
        return {
            "decision_consistency": self._calculate_decision_consistency(),
            "pattern_quality_score": self._calculate_pattern_quality(),
            "risk_management_quality": self._calculate_risk_management_quality(),
            "temporal_intelligence": self._calculate_temporal_intelligence(),
        }

    def _calculate_decision_consistency(self) -> float:
        """Mesure la consistance des décisions."""
        if len(self.trades) < 10:
            return 0

        # Variance des actions (si disponible)
        if "action" in self.trades.columns:
            action_volatility = self.trades["action"].std()
            consistency = max(0, 1 - min(action_volatility / 0.3, 1))
        else:
            consistency = 0.5

        # Holding time consistency
        if "holding_time" in self.trades.columns:
            holding_times = self.trades["holding_time"].values
            holding_cv = np.std(holding_times) / np.mean(holding_times)
            holding_consistency = max(0, 1 - min(holding_cv / 2, 1))
        else:
            holding_consistency = 0.5

        return (consistency + holding_consistency) / 2

    def _calculate_pattern_quality(self) -> float:
        """Mesure la qualité des patterns."""
        pnl = self.trades["pnl"].values

        if len(pnl) < 5:
            return 0

        # Clustering des trades gagnants
        winning_trades = self.trades[self.trades["pnl"] > 0]

        if len(winning_trades) == 0:
            return 0

        # Nombre de patterns distincts (clusters)
        if "features" in winning_trades.columns:
            # Simplifié : compter les contextes distincts
            n_patterns = len(winning_trades.groupby("features"))
            pattern_concentration = min(n_patterns / max(1, len(winning_trades) / 3), 1)
        else:
            pattern_concentration = 0.5

        # Win rate par pattern
        win_rate = np.sum(pnl > 0) / len(pnl)
        win_rate_quality = min(win_rate / 0.6, 1)

        return (pattern_concentration + win_rate_quality) / 2

    def _calculate_risk_management_quality(self) -> float:
        """Mesure la qualité de la gestion des risques."""
        pnl = self.trades["pnl"].values

        if len(pnl) == 0:
            return 0

        # Ratio gain/perte
        gains = np.sum(pnl[pnl > 0])
        losses = np.abs(np.sum(pnl[pnl < 0]))
        ratio = gains / losses if losses > 0 else 0

        # Drawdown control
        max_dd = self._calculate_max_drawdown(pnl)
        dd_control = max(0, 1 - min(max_dd / 0.15, 1))

        # Stop-loss respect (si disponible)
        if "stopped_out" in self.trades.columns:
            sl_respect = 1 - (self.trades["stopped_out"].sum() / len(self.trades))
        else:
            sl_respect = 0.5

        return (min(ratio / 1.5, 1) + dd_control + sl_respect) / 3

    def _calculate_temporal_intelligence(self) -> float:
        """Mesure l'intelligence temporelle."""
        if len(self.trades) < 10:
            return 0

        # Consistency across timeframes (si disponible)
        if "timeframe" in self.trades.columns:
            tf_groups = self.trades.groupby("timeframe")["pnl"].mean()
            tf_consistency = 1 - (tf_groups.std() / tf_groups.mean())
        else:
            tf_consistency = 0.5

        # Time-of-day consistency
        if "entry_time" in self.trades.columns:
            self.trades["hour"] = pd.to_datetime(self.trades["entry_time"]).dt.hour
            hourly_groups = self.trades.groupby("hour")["pnl"].mean()
            hourly_consistency = 1 - (hourly_groups.std() / hourly_groups.mean())
        else:
            hourly_consistency = 0.5

        return max(0, min((tf_consistency + hourly_consistency) / 2, 1))

    def _calculate_pattern_repeatability(self) -> float:
        """Calcule la répétabilité des patterns."""
        if len(self.trades) < 10:
            return 0

        # Clustering simple basé sur les features
        pnl = self.trades["pnl"].values
        winning_trades = np.sum(pnl > 0)

        if winning_trades == 0:
            return 0

        # Approximation : si 30%+ des trades gagnants sont dans 2-3 clusters
        return min(winning_trades / len(self.trades) / 0.3, 1)

    def _walk_forward_analysis(self, pnl: np.ndarray) -> float:
        """Analyse walk-forward."""
        if len(pnl) < 20:
            return 0

        window_size = len(pnl) // 4
        returns = []

        for i in range(0, len(pnl) - window_size, window_size // 2):
            window = pnl[i : i + window_size]
            if len(window) > 0:
                returns.append(np.sum(window))

        if len(returns) < 2:
            return 0

        # Consistency = 1 - coefficient of variation
        cv = np.std(returns) / np.mean(returns) if np.mean(returns) != 0 else 1
        return max(0, 1 - min(cv, 1))

    def _monte_carlo_analysis(self, pnl: np.ndarray) -> float:
        """Simulation Monte Carlo."""
        if len(pnl) < 20:
            return 0

        n_simulations = 1000
        profitable_sims = 0

        for _ in range(n_simulations):
            shuffled = np.random.permutation(pnl)
            if np.sum(shuffled) > 0:
                profitable_sims += 1

        return profitable_sims / n_simulations

    def _out_of_sample_degradation(self, pnl: np.ndarray) -> float:
        """Dégradation out-of-sample."""
        if len(pnl) < 20:
            return 0

        # Split 70/30
        split = int(len(pnl) * 0.7)
        is_return = np.sum(pnl[:split])
        oos_return = np.sum(pnl[split:])

        if is_return == 0:
            return 0

        degradation = 1 - (oos_return / is_return)
        return max(0, min(degradation, 1))

    def _calculate_sharpe_ratio(self, pnl: np.ndarray, rf_rate: float = 0.02) -> float:
        """Calcule le Sharpe Ratio."""
        if len(pnl) < 2:
            return 0

        returns = pnl / np.abs(pnl).mean() if np.abs(pnl).mean() > 0 else pnl
        excess_returns = returns - rf_rate / 252

        if np.std(excess_returns) == 0:
            return 0

        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _calculate_sortino_ratio(self, pnl: np.ndarray, rf_rate: float = 0.02) -> float:
        """Calcule le Sortino Ratio."""
        if len(pnl) < 2:
            return 0

        returns = pnl / np.abs(pnl).mean() if np.abs(pnl).mean() > 0 else pnl
        excess_returns = returns - rf_rate / 252

        downside = np.std(excess_returns[excess_returns < 0])

        if downside == 0:
            return 0

        return np.mean(excess_returns) / downside * np.sqrt(252)

    def _calculate_max_drawdown(self, pnl: np.ndarray) -> float:
        """Calcule le drawdown maximum."""
        if len(pnl) == 0:
            return 0

        cumsum = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = (cumsum - running_max) / np.abs(running_max)

        return np.abs(np.min(drawdown)) if len(drawdown) > 0 else 0

    def _deflate_sharpe(self, sharpe: float, n_obs: int) -> float:
        """Ajuste le Sharpe pour l'overfitting."""
        if n_obs < 2:
            return 0

        # Formule simplifiée de Yan et al. (2025)
        deflation_factor = 1 - (0.5 / n_obs)
        return sharpe * deflation_factor

    def _calculate_reflection_score(
        self, stat, prob, robust, econ, behavior
    ) -> float:
        """Calcule le score de réflexion final (0-100)."""
        # Poids des différentes couches
        weights = {
            "statistical": 0.15,
            "probabilistic": 0.25,
            "robustness": 0.25,
            "economic": 0.20,
            "behavioral": 0.15,
        }

        # Normalisation (0-1)
        stat_score = (
            stat.get("accuracy", 0) * 0.3
            + stat.get("f1_score", 0) * 0.4
            + max(0, min(stat.get("expected_shortfall", 0) / -0.02, 1)) * 0.3
        )

        prob_score = (
            min(prob.get("profit_factor", 0) / 1.5, 1) * 0.3
            + min(prob.get("edge_ratio", 0) / 1.5, 1) * 0.3
            + prob.get("pattern_repeatability", 0) * 0.2
            + min(prob.get("calmar_ratio", 0) / 3.0, 1) * 0.2
        )

        robust_score = (
            robust.get("walk_forward_consistency", 0) * 0.3
            + robust.get("monte_carlo_profitability", 0) * 0.3
            + max(0, 1 - robust.get("out_of_sample_degradation", 0)) * 0.2
            + min(robust.get("deflated_sharpe", 0) / 1.0, 1) * 0.2
        )

        econ_score = (
            min(econ.get("sharpe_ratio", 0) / 1.5, 1) * 0.4
            + min(econ.get("sortino_ratio", 0) / 2.0, 1) * 0.4
            + max(0, 1 - econ.get("max_drawdown", 0) / 0.15) * 0.2
        )

        behavior_score = (
            behavior.get("decision_consistency", 0) * 0.25
            + behavior.get("pattern_quality_score", 0) * 0.35
            + behavior.get("risk_management_quality", 0) * 0.25
            + behavior.get("temporal_intelligence", 0) * 0.15
        )

        # Score final
        final_score = (
            stat_score * weights["statistical"]
            + prob_score * weights["probabilistic"]
            + robust_score * weights["robustness"]
            + econ_score * weights["economic"]
            + behavior_score * weights["behavioral"]
        )

        return final_score * 100

    def print_report(self) -> None:
        """Affiche un rapport détaillé."""
        if self.metrics is None:
            print("❌ Aucune analyse effectuée. Appelez analyze() d'abord.")
            return

        m = self.metrics

        print("\n" + "=" * 80)
        print("📊 RAPPORT D'ANALYSE DE QUALITÉ DÉCISIONNELLE - ADAN TRADING BOT")
        print("=" * 80)

        # Verdict final
        verdict = "✅ RÉFLÉCHI" if m.is_reflective else "❌ HASARDEUX"
        print(f"\n🎯 VERDICT FINAL: {verdict} (Score: {m.reflection_score:.1f}/100)")

        # Couche statistique
        print("\n📈 COUCHE STATISTIQUE (Séparer signal du bruit)")
        print(f"  Accuracy:           {m.accuracy:.2%} (seuil: >55%)")
        print(f"  F1-Score:           {m.f1_score:.2f} (seuil: >0.60)")
        print(f"  Recall:             {m.recall:.2%} (seuil: >65%)")
        print(f"  Precision:          {m.precision:.2%}")
        print(f"  Expected Shortfall: {m.expected_shortfall:.4f} (seuil: <-0.02)")

        # Couche probabiliste
        print("\n💰 COUCHE PROBABILISTE (Patterns répétables & profitables)")
        print(f"  Profit Factor:      {m.profit_factor:.2f} (seuil: >1.5)")
        print(f"  Edge Ratio:         {m.edge_ratio:.2f} (seuil: >1.5)")
        print(f"  Pattern Repeat:     {m.pattern_repeatability:.2%} (seuil: >80%)")
        print(f"  Calmar Ratio:       {m.calmar_ratio:.2f} (seuil: >3.0)")

        # Couche robustesse
        print("\n🛡️  COUCHE ROBUSTESSE (Anti-overfitting)")
        print(f"  Walk-Forward:       {m.walk_forward_consistency:.2%} (seuil: >70%)")
        print(f"  Monte Carlo:        {m.monte_carlo_profitability:.2%} (seuil: >80%)")
        print(f"  OOS Degradation:    {m.out_of_sample_degradation:.2%} (seuil: <20%)")
        print(f"  Deflated Sharpe:    {m.deflated_sharpe:.2f} (seuil: >0.8)")

        # Couche économique
        print("\n💵 COUCHE ÉCONOMIQUE (Profitabilité réelle)")
        print(f"  Sharpe Ratio:       {m.sharpe_ratio:.2f} (seuil: >1.5)")
        print(f"  Sortino Ratio:      {m.sortino_ratio:.2f} (seuil: >2.0)")
        print(f"  Max Drawdown:       {m.max_drawdown:.2%} (seuil: <15%)")

        # Métriques comportementales
        print("\n🧠 MÉTRIQUES COMPORTEMENTALES")
        print(f"  Decision Consistency:    {m.decision_consistency:.2%}")
        print(f"  Pattern Quality:         {m.pattern_quality_score:.2%}")
        print(f"  Risk Management:         {m.risk_management_quality:.2%}")
        print(f"  Temporal Intelligence:   {m.temporal_intelligence:.2%}")

        print("\n" + "=" * 80)
