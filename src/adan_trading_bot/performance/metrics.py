"""
Module de calcul et de suivi des métriques de performance du portefeuille.
"""
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Classe pour le calcul et le suivi des métriques de performance."""

    def __init__(self, metrics_dir: str = "logs/metrics"):
        """
        Initialise le suivi des métriques.

        Args:
            metrics_dir: Répertoire pour enregistrer les métriques
        """
        self.returns: List[float] = []
        self.drawdowns: List[float] = []
        self.equity_curve: List[float] = []
        self.trades: List[Dict] = []
        self.start_time = datetime.utcnow()

        # Créer le répertoire des métriques si nécessaire
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Fichier de log structuré
        self.metrics_file = self.metrics_dir / f"metrics_{self.start_time.strftime('%Y%m%d_%H%M%S')}.jsonl"

        # En-tête du fichier de métriques
        self._log_metrics({"event": "initialization", "timestamp": self.start_time.isoformat()})

    def update_trade(self, trade_result: Dict) -> None:
        """Met à jour les métriques avec les résultats d'un trade."""
        self.trades.append(trade_result)

        if 'pnl_pct' in trade_result and trade_result['pnl_pct'] is not None:
            self.returns.append(trade_result['pnl_pct'] / 100)  # Conversion en décimal

        if 'equity' in trade_result:
            self.equity_curve.append(trade_result['equity'])

        self._log_metrics({
            "event": "trade_closed",
            "timestamp": datetime.utcnow().isoformat(),
            **trade_result
        })

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calcule le ratio de Sharpe annualisé."""
        if not self.returns:
            return 0.0

        returns = np.array(self.returns)
        excess_returns = returns - (risk_free_rate / 252)  # Taux sans risque quotidien

        if np.std(excess_returns) == 0:
            return 0.0

        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calcule le ratio de Sortino annualisé."""
        if not self.returns:
            return 0.0

        returns = np.array(self.returns)
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        return np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)

    def calculate_profit_factor(self) -> float:
        """Calcule le profit factor (gross profit / gross loss)."""
        if not self.trades:
            return 0.0

        gross_profit = sum(t['pnl'] for t in self.trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t.get('pnl', 0) < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def calculate_calmar_ratio(self) -> float:
        """Calcule le ratio de Calmar (CAGR / Max Drawdown)."""
        if not self.equity_curve:
            return 0.0

        cagr = self._calculate_cagr()
        max_dd = self.calculate_max_drawdown()

        if max_dd == 0:
            return 0.0

        return cagr / abs(max_dd)

    def calculate_max_drawdown(self) -> float:
        """Calcule le drawdown maximum en pourcentage."""
        if not self.equity_curve:
            return 0.0

        peak = self.equity_curve[0]
        max_dd = 0.0

        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd * 100  # En pourcentage

    def _calculate_cagr(self) -> float:
        """Calcule le taux de croissance annuel composé."""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0

        start_value = self.equity_curve[0]
        end_value = self.equity_curve[-1]
        years = (datetime.utcnow() - self.start_time).days / 365.25

        if years <= 0 or start_value == 0:
            return 0.0

        return (end_value / start_value) ** (1 / years) - 1

    def get_metrics_summary(self) -> Dict[str, float]:
        """Retourne un résumé des métriques de performance."""
        return {
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "sortino_ratio": self.calculate_sortino_ratio(),
            "profit_factor": self.calculate_profit_factor(),
            "calmar_ratio": self.calculate_calmar_ratio(),
            "max_drawdown": self.calculate_max_drawdown(),
            "cagr": self._calculate_cagr() * 100,  # En pourcentage
            "total_return": (self.equity_curve[-1] / self.equity_curve[0] - 1) * 100 if self.equity_curve else 0.0,
            "win_rate": (len([t for t in self.trades if t.get('pnl', 0) > 0]) / len(self.trades) * 100)
                        if self.trades else 0.0,
            "total_trades": len(self.trades),
            "winning_trades": len([t for t in self.trades if t.get('pnl', 0) > 0]),
            "losing_trades": len([t for t in self.trades if t.get('pnl', 0) < 0]),
        }

    def _log_metrics(self, data: Dict) -> None:
        """Écrit les métriques dans le fichier de log structuré."""
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"[METRICS] Erreur lors de l'écriture des métriques: {e}")

    def log_periodic_update(self) -> None:
        """Journalise une mise à jour périodique des métriques."""
        metrics = self.get_metrics_summary()
        self._log_metrics({
            "event": "periodic_update",
            "timestamp": datetime.utcnow().isoformat(),
            **metrics
        })

        # Log dans la console avec un format lisible
        logger.info(
            "[METRICS] "
            f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
            f"Sortino: {metrics['sortino_ratio']:.2f} | "
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Max DD: {metrics['max_drawdown']:.2f}% | "
            f"CAGR: {metrics['cagr']:.2f}% | "
            f"Win Rate: {metrics['win_rate']:.1f}% | "
            f"Trades: {metrics['total_trades']} ({metrics['winning_trades']}W/{metrics['losing_trades']}L)"
        )
