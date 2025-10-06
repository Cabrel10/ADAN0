"""
Module de calcul et de suivi des métriques de performance du portefeuille.
"""
import numpy as np
from typing import List, Dict, Optional
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import logging

from ..utils.smart_logger import create_smart_logger

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Classe pour le calcul et le suivi des métriques de performance."""

    def __init__(self, config=None, worker_id=0, metrics_dir: str = "logs/metrics"):
        """
        Initialise le suivi des métriques.

        Args:
            config: Configuration du système
            worker_id: ID du worker pour les logs
            metrics_dir: Répertoire pour enregistrer les métriques
        """
        self.returns = deque(maxlen=10000)
        self.drawdowns = deque(maxlen=10000)
        self.equity_curve = deque(maxlen=10000)
        self.trades = deque(maxlen=5000)
        self.start_time = datetime.utcnow()

        # Worker ID pour éviter la duplication de logs
        self.worker_id = worker_id

        # Initialiser le SmartLogger pour ce worker
        self.smart_logger = create_smart_logger(worker_id, total_workers=4, logger_name="performance_metrics")

        self.config = config or {}
        self.closed_positions = deque(maxlen=5000)

        # Cache pour éviter les doublons de logs
        self._last_logs = {}
        import time
        self._time_module = time

        # État courant des positions ouvertes (synchronisé par le portefeuille)
        self._current_open_positions = []
        self._current_prices = {}

        # Compteurs de fréquence des positions par timeframe
        self.positions_frequency = {
            '5m': 0,
            '1h': 0,
            '4h': 0,
            'daily_total': 0
        }
        self.frequency_history = deque(maxlen=1000)  # Historique des fréquences par jour

        # Compteurs d'activités de trading
        self.trade_attempts_total = 0
        self.valid_trade_attempts = 0
        self.invalid_trade_attempts = 0
        self.executed_trades_opened = 0

        # Créer le répertoire des métriques si nécessaire
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Fichier de log structuré
        self.metrics_file = self.metrics_dir / f"metrics_{self.start_time.strftime('%Y%m%d_%H%M%S')}.jsonl"

        # En-tête du fichier de métriques
        self._log_metrics({"event": "initialization", "timestamp": self.start_time.isoformat()})

    def __getstate__(self):
        """Préparer l'état pour le pickling, en excluant les loggers et modules."""
        state = self.__dict__.copy()
        # Exclure les attributs non-sérialisables de manière sécurisée
        state.pop('smart_logger', None)
        state.pop('_time_module', None)
        if 'metrics_file' in state:
            state['_metrics_file_path'] = str(state.pop('metrics_file'))
        return state

    def __setstate__(self, state):
        """Restaurer l'état après le unpickling et ré-initialiser les loggers."""
        if '_metrics_file_path' in state:
            state['metrics_file'] = Path(state['_metrics_file_path'])
            del state['_metrics_file_path']

        self.__dict__.update(state)
        
        self.smart_logger = create_smart_logger(
            getattr(self, 'worker_id', 0), 
            total_workers=4, 
            logger_name="performance_metrics"
        )
        import time
        self._time_module = time

    def log_info(self, message, step=None):
        """Log un message avec le système intelligent SmartLogger."""
        self.smart_logger.smart_info(logger, message, step)

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
            "worker_id": self.worker_id,
            **trade_result
        })

    # --- Nouveaux enregistreurs pour tentatives et exécutions ---
    def record_trade_attempt(self, valid: bool, reason: Optional[str] = None, context: Optional[Dict] = None) -> None:
        """Enregistre une tentative de trade (valide/invalide)."""
        self.trade_attempts_total += 1
        if valid:
            self.valid_trade_attempts += 1
        else:
            self.invalid_trade_attempts += 1
        payload = {
            "event": "trade_attempt",
            "timestamp": datetime.utcnow().isoformat(),
            "worker_id": self.worker_id,
            "valid": bool(valid),
        }
        if reason:
            payload["reason"] = str(reason)
        if context:
            try:
                payload.update({k: v for k, v in context.items() if k not in ("prices",)})
            except Exception:
                pass
        self._log_metrics(payload)

    def record_trade_rejection(self, reason: str, context: Optional[Dict] = None) -> None:
        """Spécialisation: enregistrement d'un rejet de trade (invalide)."""
        self.record_trade_attempt(valid=False, reason=reason, context=context)

    def record_trade_open(self, receipt: Dict) -> None:
        """Enregistre l'ouverture effective d'un trade."""
        self.executed_trades_opened += 1
        payload = {
            "event": "trade_opened",
            "timestamp": datetime.utcnow().isoformat(),
            "worker_id": self.worker_id,
        }
        try:
            serializable = {k: v for k, v in receipt.items() if k not in ("prices",)}
            payload.update(serializable)
        except Exception:
            pass
        self._log_metrics(payload)

    def calculate_unrealized_pnl(self, open_positions=None, current_prices=None) -> Dict[str, float]:
        """
        Calcule le PnL non réalisé des positions ouvertes.

        Args:
            open_positions: Liste des positions ouvertes
            current_prices: Dict des prix actuels {asset: price}

        Returns:
            Dict avec le PnL non réalisé et les métriques associées
        """
        if not open_positions or not current_prices:
            return {
                'unrealized_pnl': 0.0,
                'unrealized_pnl_pct': 0.0,
                'open_positions_count': 0,
                'unrealized_winners': 0,
                'unrealized_losers': 0
            }

        total_unrealized = 0.0
        unrealized_winners = 0
        unrealized_losers = 0

        for position in open_positions:
            if not position or not hasattr(position, 'asset'):
                continue

            asset = position.asset
            if asset not in current_prices:
                continue

            current_price = current_prices[asset]
            entry_price = getattr(position, 'entry_price', 0)
            size = getattr(position, 'size', 0)

            if entry_price > 0 and size != 0:
                # Calcul PnL non réalisé (position longue)
                position_pnl = (current_price - entry_price) * size
                total_unrealized += position_pnl

                if position_pnl > 0:
                    unrealized_winners += 1
                elif position_pnl < 0:
                    unrealized_losers += 1

        # Calcul pourcentage basé sur le capital total si disponible
        total_equity = self.equity_curve[-1] if self.equity_curve else 1000.0
        unrealized_pct = (total_unrealized / total_equity) * 100 if total_equity > 0 else 0.0

        return {
            'unrealized_pnl': total_unrealized,
            'unrealized_pnl_pct': unrealized_pct,
            'open_positions_count': len(open_positions),
            'unrealized_winners': unrealized_winners,
            'unrealized_losers': unrealized_losers
        }

    def update_open_positions_metrics(self, open_positions=None, current_prices=None) -> None:
        """
        Met à jour les métriques avec les positions ouvertes actuelles.

        Args:
            open_positions: Liste des positions ouvertes du portfolio
            current_prices: Dict des prix actuels {asset: price}
        """
        # Stocker les données des positions ouvertes pour get_metrics_summary
        self._current_open_positions = open_positions or []
        self._current_prices = current_prices or {}

    def record_equity_snapshot(self, equity: Optional[float]) -> None:
        """Enregistre la valeur d'équité courante dans la courbe d'équité."""
        if equity is None:
            return

        try:
            equity_value = float(equity)
        except (TypeError, ValueError):
            return

        if self.equity_curve and self.equity_curve[-1] == equity_value:
            return

        self.equity_curve.append(equity_value)

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calcule le ratio de Sharpe annualisé."""
        if not self.returns or len(self.returns) < 10:
            return 0.0

        try:
            # Convertir le deque en array numpy de manière sécurisée
            returns = np.array(list(self.returns))

            if len(returns) == 0:
                logger.debug(f"[Worker {self.worker_id}] [METRICS] No returns available for Sharpe calculation")
                return 0.0

            excess_returns = returns - (risk_free_rate / 252)  # Taux sans risque quotidien

            # Vérification plus stricte pour éviter les divisions par zéro
            std = np.std(excess_returns)
            if std <= 1e-10:
                logger.debug(f"[Worker {self.worker_id}] [METRICS] Zero volatility detected, Sharpe = 0")
                return 0.0

            sharpe_ratio = np.mean(excess_returns) / std * np.sqrt(365)

            # Vérifier que le résultat est valide
            if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
                logger.warning(f"[Worker {self.worker_id}] [METRICS] Invalid Sharpe ratio detected: {sharpe_ratio}")
                return 0.0

            # Clipper les valeurs extrêmes pour éviter les aberrations
            clipped_sharpe = float(np.clip(sharpe_ratio, -10.0, 10.0))

            logger.debug(f"[Worker {self.worker_id}] [METRICS] Sharpe: {clipped_sharpe:.4f} | returns={len(returns)} | std={std:.6f}")
            return clipped_sharpe

        except Exception as e:
            logger.error(f"[Worker {self.worker_id}] [METRICS] Error calculating Sharpe ratio: {e}")
            return 0.0

    def calculate_metrics(self, positions_count=None, trade_attempts=None, invalid_trade_attempts=None):
        """Calculate comprehensive performance metrics with worker identification."""
        # Filtrer les trades fermés
        closed_trades = [trade for trade in self.trades if trade.get('action') == 'close']
        total_trades = len(closed_trades)
        wins = sum(1 for trade in closed_trades if trade.get('pnl', 0) > 0)
        losses = sum(1 for trade in closed_trades if trade.get('pnl', 0) < 0)
        neutrals = total_trades - wins - losses
        winrate = (wins / max(1, total_trades)) * 100 if total_trades > 0 else 0.0

        open_positions_count = 0
        positions_str = "Positions: N/A"
        if positions_count:
            open_positions_count = positions_count.get('open_positions', positions_count.get('daily_total', 0))
            positions_str = (
                f"Positions: 5m:{positions_count.get('5m', 0)}, 1h:{positions_count.get('1h', 0)}, "
                f"4h:{positions_count.get('4h', 0)}, Total:{open_positions_count}"
            )

        # Format Max DD with more precision for small values
        max_dd_value = self.calculate_max_drawdown()
        max_dd_str = f"{max_dd_value:.3f}" if max_dd_value < 1.0 else f"{max_dd_value:.2f}"

        self.log_info(
            f"[METRICS] Sharpe: {self.calculate_sharpe_ratio():.2f} | Sortino: {self.calculate_sortino_ratio():.2f} | "
            f"Profit Factor: {self.calculate_profit_factor():.2f} | Max DD: {max_dd_str}% | "
            f"CAGR: {self.calculate_cagr():.2f}% | Win Rate: {winrate:.1f}% | Trades: {total_trades} "
            f"({wins}W/{losses}L/{neutrals}N) | {positions_str}"
        )

        return {
            'sharpe': self.calculate_sharpe_ratio(),
            'sortino': self.calculate_sortino_ratio(),
            'profit_factor': self.calculate_profit_factor(),
            'max_dd': self.calculate_max_drawdown(),
            'cagr': self.calculate_cagr(),
            'winrate': winrate,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'neutrals': neutrals,
            'positions_count': positions_count,
            'open_positions_count': open_positions_count,
            # Tentatives et exécutions
            'trade_attempts': trade_attempts if trade_attempts is not None else self.trade_attempts_total,
            'invalid_trade_attempts': invalid_trade_attempts if invalid_trade_attempts is not None else self.invalid_trade_attempts,
            'valid_trade_attempts': self.valid_trade_attempts,
            'executed_trades_opened': self.executed_trades_opened,
        }

    def close_position(self, position, exit_price):
        """Log a closed position with worker identification."""
        pnl = (exit_price - position['entry_price']) * position['size']
        pnl_pct = (pnl / (position['entry_price'] * position['size'])) * 100 if position['entry_price'] > 0 else 0.0

        closed_pos = {
            'asset': position['asset'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_price': position['entry_price'],
            'exit_price': exit_price
        }

        self.closed_positions.append(closed_pos)

        logger.info(f"[POSITION FERMÉE Worker {self.worker_id}] {position['asset']} - Taille: {position['size']:.6f} | "
                    f"Entrée: {position['entry_price']:.2f} | Sortie: {exit_price:.2f} | "
                    f"PnL: {pnl:+.2f} ({pnl_pct:.2f}%)")

        return closed_pos

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calcule le ratio de Sortino annualisé."""
        if not self.returns or len(self.returns) < 10:
            return 0.0

        try:
            # Convertir le deque en array numpy de manière sécurisée
            returns = np.array(list(self.returns))

            if len(returns) == 0:
                logger.debug(f"[Worker {self.worker_id}] [METRICS] No returns available for Sortino calculation")
                return 0.0

            excess_returns = returns - (risk_free_rate / 252)
            downside_returns = excess_returns[excess_returns < 0]

            if len(downside_returns) == 0:
                logger.debug(f"[Worker {self.worker_id}] [METRICS] No downside returns for Sortino calculation")
                return 0.0

            # Vérification plus stricte pour éviter les divisions par zéro
            downside_std = np.std(downside_returns)
            if downside_std <= 1e-10:
                logger.debug(f"[Worker {self.worker_id}] [METRICS] Zero downside volatility detected, Sortino = 0")
                return 0.0

            sortino_ratio = (np.mean(returns) - risk_free_rate/365) / downside_std * np.sqrt(365)

            # Vérifier que le résultat est valide
            if np.isnan(sortino_ratio) or np.isinf(sortino_ratio):
                logger.warning(f"[Worker {self.worker_id}] [METRICS] Invalid Sortino ratio detected: {sortino_ratio}")
                return 0.0

            # Clipper les valeurs extrêmes pour éviter les aberrations
            clipped_sortino = float(np.clip(sortino_ratio, -10.0, 10.0))

            logger.debug(f"[Worker {self.worker_id}] [METRICS] Sortino: {clipped_sortino:.4f} | downside_returns={len(downside_returns)} | downside_std={downside_std:.6f}")
            return clipped_sortino

        except Exception as e:
            logger.error(f"[Worker {self.worker_id}] [METRICS] Error calculating Sortino ratio: {e}")
            return 0.0

    def calculate_profit_factor(self) -> float:
        """Calcule le profit factor."""
        # Filtrer les trades fermés seulement
        closed_trades = [trade for trade in self.trades if trade.get('action') == 'close']

        if not closed_trades:
            return 0.0

        # Ignore trades neutres demandé
        pnls = [trade['pnl'] for trade in closed_trades]
        wins = sum(pnl for pnl in pnls if pnl > 0)
        losses = sum(-pnl for pnl in pnls if pnl < 0)
        profit_factor = wins / losses if losses > 0 else 1.0 if wins > 0 else 0.0

        return profit_factor


    def calculate_cagr(self) -> float:
        """Calcule le CAGR (Compound Annual Growth Rate)."""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0

        start_value = self.equity_curve[0]
        end_value = self.equity_curve[-1]

        if start_value <= 0:
            return 0.0

        # Approximation: nombre de jours basé sur le nombre de points
        days = len(self.equity_curve)
        years = days / 365.25

        if years <= 0:
            return 0.0

        cagr = ((end_value / start_value) ** (1 / years) - 1) * 100
        return float(np.clip(cagr, -100.0, 500.0))  # Limiter les valeurs extrêmes

    def calculate_profit_factor_legacy(self) -> float:
        """Calcule le profit factor (gross profit / gross loss) - méthode legacy."""
        if not self.trades or len(self.trades) < 5:
            return 0.0

        gross_profit = sum(t['pnl'] for t in self.trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t.get('pnl', 0) < 0))

        if gross_loss == 0 or gross_loss < 1e-8:
            # Éviter les valeurs infinies, retourner une valeur élevée mais raisonnable
            return min(100.0, gross_profit) if gross_profit > 0 else 0.0

        profit_factor = gross_profit / gross_loss
        # Clipper pour éviter les valeurs extrêmes
        return float(np.clip(profit_factor, 0.0, 100.0))

    def calculate_calmar_ratio(self) -> float:
        """Calcule le ratio de Calmar (CAGR / Max Drawdown)."""
        if not self.equity_curve or len(self.equity_curve) < 10:
            return 0.0

        cagr = self._calculate_cagr()
        max_dd = self.calculate_max_drawdown()

        if max_dd == 0 or abs(max_dd) < 1e-4:
            return 0.0

        calmar = cagr / abs(max_dd)
        # Clipper les valeurs extrêmes
        return float(np.clip(calmar, -10.0, 10.0))

    def calculate_max_drawdown(self, equity_curve: Optional[List[float]] = None) -> float:
        """Calcule le drawdown maximum en pourcentage à partir d'une courbe d'équité."""
        curve = equity_curve if equity_curve is not None else list(self.equity_curve)

        if not curve or len(curve) < 2:
            return 0.0

        equity_array = np.array(curve, dtype=np.float32)

        if not np.all(np.isfinite(equity_array)):
            logger.warning("Courbe d'équité contient des valeurs non finies, nettoyage en cours.")
            equity_array = equity_array[np.isfinite(equity_array)]

        if len(equity_array) < 2:
            return 0.0

        peak_array = np.maximum.accumulate(equity_array)
        non_zero_peaks = np.where(peak_array == 0, 1.0, peak_array)
        drawdown_array = (peak_array - equity_array) / non_zero_peaks
        max_dd = np.max(drawdown_array) if len(drawdown_array) > 0 else 0.0

        return float(max_dd * 100)

    def update_position_frequency(self, timeframe: str, count: int = 1) -> None:
        """
        Met à jour les compteurs de fréquence des positions par timeframe.

        Args:
            timeframe: Timeframe concerné ('5m', '1h', '4h')
            count: Nombre de positions à ajouter (par défaut 1)
        """
        if timeframe in self.positions_frequency:
            self.positions_frequency[timeframe] += count
            self.positions_frequency['daily_total'] += count

    def reset_daily_frequency(self) -> None:
        """Reset les compteurs de fréquence pour un nouveau jour."""
        # Sauvegarder dans l'historique avant reset
        self.frequency_history.append(self.positions_frequency.copy())

        # Reset des compteurs
        self.positions_frequency = {
            '5m': 0,
            '1h': 0,
            '4h': 0,
            'daily_total': 0
        }

    def calculate_frequency_compliance(self) -> Dict[str, float]:
        """
        Calcule la conformité aux objectifs de fréquence.

        Returns:
            Dict contenant les métriques de conformité
        """
        # Objectifs configurables (pourraient venir de la config)
        targets = {
            '5m': {'min': 6, 'max': 15},
            '1h': {'min': 3, 'max': 10},
            '4h': {'min': 1, 'max': 3},
            'daily_total': {'min': 5, 'max': 15}
        }

        compliance_metrics = {}

        for timeframe, target in targets.items():
            current_count = self.positions_frequency.get(timeframe, 0)
            min_target = target['min']
            max_target = target['max']

            # Calculer la conformité (0.0 = non conforme, 1.0 = parfaitement conforme)
            if min_target <= current_count <= max_target:
                compliance = 1.0
            elif current_count < min_target:
                compliance = current_count / min_target if min_target > 0 else 0.0
            else:  # current_count > max_target
                compliance = max_target / current_count if current_count > 0 else 0.0

            compliance_metrics[f'frequency_compliance_{timeframe}'] = compliance
            compliance_metrics[f'frequency_target_min_{timeframe}'] = min_target
            compliance_metrics[f'frequency_target_max_{timeframe}'] = max_target

        # Compliance globale (moyenne des compliances individuelles)
        individual_compliances = [
            compliance_metrics[f'frequency_compliance_{tf}']
            for tf in ['5m', '1h', '4h', 'daily_total']
        ]
        compliance_metrics['frequency_compliance_overall'] = np.mean(individual_compliances)

        return compliance_metrics

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
        # Calculer le win rate de manière sécurisée avec diagnostic détaillé
        win_rate = 0.0
        # Ne compter que les trades fermés pour les métriques de performance
        closed_trades = [t for t in self.trades if t.get('action') == 'close']
        winning_trades = len([t for t in closed_trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in closed_trades if t.get('pnl', 0) < 0])
        neutral_trades = len([t for t in closed_trades if t.get('pnl', 0) == 0])
        total_trades = len(closed_trades)

        # Diagnostic détaillé pour comprendre les incohérences
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            # Clipper entre 0 et 100%
            win_rate = min(100.0, max(0.0, win_rate))

        max_dd = self.calculate_max_drawdown()
        if total_trades < 5:
            max_dd = 0.0

        # Calcul des métriques de fréquence
        frequency_metrics = self.calculate_frequency_compliance()

        # Calcul des métriques des positions ouvertes (PnL non réalisé)
        open_positions = getattr(self, '_current_open_positions', [])
        current_prices = getattr(self, '_current_prices', {})
        unrealized_metrics = self.calculate_unrealized_pnl(open_positions, current_prices)

        # Calcul du win rate combiné (trades fermés + positions ouvertes gagnantes)
        combined_winning = winning_trades + unrealized_metrics['unrealized_winners']
        combined_losing = losing_trades + unrealized_metrics['unrealized_losers']
        combined_total = combined_winning + combined_losing + neutral_trades
        combined_win_rate = (combined_winning / combined_total * 100) if combined_total > 0 else win_rate

        return {
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "sortino_ratio": self.calculate_sortino_ratio(),
            "profit_factor": self.calculate_profit_factor(),
            "calmar_ratio": self.calculate_calmar_ratio(),
            "max_drawdown": max_dd,
            "cagr": self._calculate_cagr() * 100,  # En pourcentage
            "total_return": (self.equity_curve[-1] / self.equity_curve[0] - 1) * 100 if self.equity_curve else 0.0,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "neutral_trades": neutral_trades,
            # Métriques positions ouvertes
            "unrealized_pnl": unrealized_metrics['unrealized_pnl'],
            "unrealized_pnl_pct": unrealized_metrics['unrealized_pnl_pct'],
            "open_positions_count": unrealized_metrics['open_positions_count'],
            "unrealized_winners": unrealized_metrics['unrealized_winners'],
            "unrealized_losers": unrealized_metrics['unrealized_losers'],
            # Métriques combinées
            "combined_win_rate": combined_win_rate,
            "combined_total_positions": combined_total,
            # Métriques de fréquence
            "daily_positions_5m": self.positions_frequency.get('5m', 0),
            "daily_positions_1h": self.positions_frequency.get('1h', 0),
            "daily_positions_4h": self.positions_frequency.get('4h', 0),
            "daily_positions_total": self.positions_frequency.get('daily_total', 0),
            # Compteurs d'activité
            "trade_attempts_total": self.trade_attempts_total,
            "invalid_trade_attempts": self.invalid_trade_attempts,
            "valid_trade_attempts": self.valid_trade_attempts,
            "executed_trades_opened": self.executed_trades_opened,
            **frequency_metrics
        }

    def _log_metrics(self, data: Dict) -> None:
        """Écrit les métriques dans le fichier de log structuré en gérant la sérialisation."""
        try:
            serializable_data = json.loads(json.dumps(data, default=str))
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(serializable_data) + '\n')
        except Exception as e:
            if hasattr(self, 'smart_logger'):
                self.smart_logger.smart_error(logger, f"[METRICS] Erreur lors de l'écriture des métriques: {e}")
            else:
                logger.error(f"[Worker {getattr(self, 'worker_id', 0)}] [METRICS] Erreur lors de l'écriture des métriques: {e}")

    def log_periodic_update(self, step: int) -> None:
        """Journalise une mise à jour périodique des métriques."""
        metrics = self.get_metrics_summary()
        self._log_metrics({
            "event": "periodic_update",
            "timestamp": datetime.utcnow().isoformat(),
            **metrics
        })

        # Log dans la console avec un format lisible
        # Utiliser SmartLogger pour logging intelligent entre workers
        metrics_summary = (
            "[METRICS] "
            f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
            f"Sortino: {metrics['sortino_ratio']:.2f} | "
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Max DD: {metrics['max_drawdown']:.2f}% | "
            f"CAGR: {metrics['cagr']:.2f}% | "
            f"Win Rate: {metrics['win_rate']:.1f}% (Combined: {metrics['combined_win_rate']:.1f}%)\n"
            f"Trades: {metrics['total_trades']} ({metrics['winning_trades']}W/{metrics['losing_trades']}L) | "
            f"Open: {metrics['open_positions_count']} ({metrics['unrealized_winners']}W/{metrics['unrealized_losers']}L)\n"
            f"Unrealized PnL: {metrics['unrealized_pnl']:.2f} USDT ({metrics['unrealized_pnl_pct']:.2f}%) | "
            f"Positions: 5m:{metrics['daily_positions_5m']}, 1h:{metrics['daily_positions_1h']}, "
            f"4h:{metrics['daily_positions_4h']}, Total:{metrics['daily_positions_total']}"
        )

        if hasattr(self, 'smart_logger'):
            self.smart_logger.smart_info(logger, metrics_summary, step)
        else:
            logger.info(f"[Worker {getattr(self, 'worker_id', 0)}] {metrics_summary}")

            # Write to CSV file
            csv_path = self.metrics_dir / "performance_summary.csv"
            csv_exists = csv_path.exists()

            with open(csv_path, 'a') as f:
                if not csv_exists:
                    f.write("timestamp,step,sharpe,max_dd,win_rate,combined_win_rate,total_trades,open_positions,unrealized_pnl,cagr,profit_factor,pos_5m,pos_1h,pos_4h,pos_total\n")

                f.write(
                    f"{datetime.utcnow().isoformat()},"
                    f"{step},"
                    f"{metrics['sharpe_ratio']:.3f},"
                    f"{metrics['max_drawdown']:.2f},"
                    f"{metrics['win_rate']:.1f},"
                    f"{metrics['combined_win_rate']:.1f},"
                    f"{metrics['total_trades']},"
                    f"{metrics['open_positions_count']},"
                    f"{metrics['unrealized_pnl']:.2f},"
                    f"{metrics['cagr']:.2f},"
                    f"{metrics['profit_factor']:.2f},"
                    f"{metrics['daily_positions_5m']},"
                    f"{metrics['daily_positions_1h']},"
                    f"{metrics['daily_positions_4h']},"
                    f"{metrics['daily_positions_total']}\n"
                )
