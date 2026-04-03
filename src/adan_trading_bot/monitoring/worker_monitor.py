"""
Module de surveillance des performances des workers dans un environnement d'apprentissage distribué.

Ce module fournit une classe pour suivre et analyser les performances de chaque worker
dans un système de trading automatisé.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from threading import RLock


@dataclass
class TradeRecord:
    """Classe pour enregistrer les détails d'un trade."""
    timestamp: float
    worker_id: str
    symbol: str
    action: str  # 'buy' ou 'sell'
    price: float
    quantity: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    duration: Optional[float] = None
    

@dataclass
class WorkerStats:
    """Classe pour suivre les statistiques d'un worker."""
    worker_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak: float = 0.0
    current_balance: float = 0.0
    trade_history: List[TradeRecord] = field(default_factory=list)
    
    def update_stats(self, trade: TradeRecord) -> None:
        """Met à jour les statistiques avec un nouveau trade."""
        self.total_trades += 1
        self.total_pnl += trade.pnl if trade.pnl else 0.0
        self.current_balance += trade.pnl if trade.pnl else 0.0
        
        # Mise à jour du drawdown
        if self.current_balance > self.peak:
            self.peak = self.current_balance
        else:
            drawdown = (self.peak - self.current_balance) / (self.peak + 1e-8)
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Mise à jour des trades gagnants/perdants
        if trade.pnl is not None:
            if trade.pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
        
        # Ajout à l'historique
        self.trade_history.append(trade)
    
    @property
    def win_rate(self) -> float:
        """Calcule le taux de réussite des trades."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def avg_pnl(self) -> float:
        """Calcule le PnL moyen par trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades
    
    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des statistiques du worker."""
        return {
            'worker_id': self.worker_id,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'avg_pnl': self.avg_pnl,
            'max_drawdown': self.max_drawdown,
            'current_balance': self.current_balance
        }


class WorkerMonitor:
    """
    Classe pour surveiller et analyser les performances des workers.
    
    Cette classe est thread-safe et peut être utilisée dans un environnement multi-threadé.
    """
    
    def __init__(self):
        """Initialise le moniteur de workers."""
        self._workers: Dict[str, WorkerStats] = {}
        self._lock = RLock()
    
    def record_trade(
        self,
        worker_id: str,
        symbol: str,
        action: str,
        price: float,
        quantity: float,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        duration: Optional[float] = None
    ) -> None:
        """
        Enregistre un nouveau trade pour un worker donné.
        
        Args:
            worker_id: Identifiant unique du worker
            symbol: Symbole de l'actif tradé
            action: Type d'action ('buy' ou 'sell')
            price: Prix du trade
            quantity: Quantité tradée
            pnl: Profit/Perte du trade (optionnel)
            pnl_pct: Profit/Perte en pourcentage (optionnel)
            duration: Durée de la position en secondes (optionnel)
        """
        trade = TradeRecord(
            timestamp=time.time(),
            worker_id=worker_id,
            symbol=symbol,
            action=action,
            price=price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration=duration
        )
        
        with self._lock:
            if worker_id not in self._workers:
                self._workers[worker_id] = WorkerStats(worker_id=worker_id)
            
            self._workers[worker_id].update_stats(trade)
    
    def get_stats(self, worker_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère les statistiques pour un worker spécifique ou pour tous les workers.
        
        Args:
            worker_id: Identifiant du worker (optionnel)
            
        Returns:
            Un dictionnaire contenant les statistiques demandées
        """
        with self._lock:
            if worker_id is not None:
                if worker_id in self._workers:
                    return self._workers[worker_id].get_summary()
                return {}
            
            # Retourne les statistiques pour tous les workers
            return {wid: worker.get_summary() for wid, worker in self._workers.items()}
    
    def get_worker_ids(self) -> List[str]:
        """Retourne la liste des identifiants des workers suivis."""
        with self._lock:
            return list(self._workers.keys())
    
    def get_trade_history(
        self,
        worker_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des trades pour un worker ou pour tous les workers.
        
        Args:
            worker_id: Identifiant du worker (optionnel)
            limit: Nombre maximum de trades à retourner (optionnel)
            
        Returns:
            Une liste de dictionnaires contenant les détails des trades
        """
        with self._lock:
            if worker_id is not None:
                if worker_id in self._workers:
                    trades = self._workers[worker_id].trade_history
                    if limit:
                        trades = trades[-limit:]
                    return [self._trade_to_dict(t) for t in trades]
                return []
            
            # Récupère les trades de tous les workers
            all_trades = []
            for worker in self._workers.values():
                worker_trades = worker.trade_history
                if limit:
                    worker_trades = worker_trades[-limit:]
                all_trades.extend([self._trade_to_dict(t) for t in worker_trades])
            
            # Trie par timestamp
            all_trades.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return all_trades
    
    def _trade_to_dict(self, trade: TradeRecord) -> Dict[str, Any]:
        """Convertit un objet TradeRecord en dictionnaire."""
        return {
            'timestamp': trade.timestamp,
            'worker_id': trade.worker_id,
            'symbol': trade.symbol,
            'action': trade.action,
            'price': trade.price,
            'quantity': trade.quantity,
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'duration': trade.duration
        }
    
    def get_summary_table(self) -> str:
        """
        Génère un tableau récapitulatif des performances de tous les workers.
        
        Returns:
            Une chaîne formatée contenant le tableau récapitulatif
        """
        with self._lock:
            if not self._workers:
                return "Aucun worker suivi pour le moment."
            
            # En-tête du tableau
            headers = [
                'Worker ID', 'Trades', 'Win Rate', 'Total PnL', 
                'Avg PnL', 'Max Drawdown', 'Balance'
            ]
            rows = []
            
            # Données de chaque worker
            for worker_id, stats in self._workers.items():
                summary = stats.get_summary()
                rows.append([
                    worker_id[:8],  # Affiche seulement les 8 premiers caractères de l'ID
                    summary['total_trades'],
                    f"{summary['win_rate']:.1f}%",
                    f"{summary['total_pnl']:.2f}",
                    f"{summary['avg_pnl']:.4f}",
                    f"{summary['max_drawdown']:.2%}",
                    f"{summary['current_balance']:.2f}"
                ])
            
            # Calcule les totaux
            totals = [
                'TOTAL',
                sum(row[1] for row in rows),  # Total des trades
                f"{sum(1 for row in rows if float(row[2].rstrip('%')) > 50) / len(rows) * 100:.1f}%",  # Win rate moyen
                f"{sum(float(row[3]) for row in rows):.2f}",  # PnL total
                f"{sum(float(row[4]) for row in rows) / len(rows):.4f}",  # PnL moyen
                f"{max(float(row[5].rstrip('%')) / 100 for row in rows):.2%}",  # Max drawdown max
                f"{sum(float(row[6]) for row in rows):.2f}"  # Solde total
            ]
            
            # Largeurs des colonnes
            col_widths = [
                max(len(str(row[i])) for row in [headers] + rows + [totals])
                for i in range(len(headers))
            ]
            
            # Construction du tableau
            def make_row(row, is_header=False):
                return " | ".join(
                    f"{str(item):<{col_widths[i]}}" 
                    for i, item in enumerate(row)
                )
            
            # Ligne de séparation
            separator = "-" * (sum(col_widths) + 3 * (len(col_widths) - 1))
            
            # Construction du tableau complet
            table = [
                separator,
                make_row(headers),
                separator,
                *[make_row(row) for row in rows],
                separator,
                make_row(totals),
                separator
            ]
            
            return "\n".join(table)
