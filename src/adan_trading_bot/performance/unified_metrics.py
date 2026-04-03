#!/usr/bin/env python3
"""
🎯 CALCULATEUR DE MÉTRIQUES UNIFIÉ - Source unique de vérité
Résout: Calculs différents, incohérences, pas de validation
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .unified_metrics_db import UnifiedMetricsDB

class UnifiedMetrics:
    """Calculateur unique de métriques - SOURCE DE VÉRITÉ"""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db = UnifiedMetricsDB(db_path)
        self.trades = []
        self.returns = []
        self.portfolio_values = []
    
    # ========== AJOUTER DES DONNÉES ==========
    
    def add_trade(self, action: str, symbol: str, quantity: float, price: float, pnl: Optional[float] = None):
        """Ajouter un trade"""
        self.trades.append({
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'pnl': pnl
        })
        # Sauvegarder dans la base
        self.db.add_trade(action, symbol, quantity, price, pnl)
    
    def add_return(self, return_value: float):
        """Ajouter un return quotidien"""
        self.returns.append(return_value)
        self.db.add_metric('daily_return', return_value, 'unified_metrics')
    
    def add_portfolio_value(self, value: float):
        """Ajouter une valeur de portefeuille"""
        self.portfolio_values.append(value)
        self.db.add_metric('portfolio_value', value, 'unified_metrics')
    
    # ========== CALCULER LES MÉTRIQUES ==========
    
    def calculate_sharpe(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculer le Sharpe Ratio - MÉTHODE UNIQUE
        
        Sharpe = (Rendement moyen - Taux sans risque) / Écart-type
        Annualisé sur 252 jours de trading
        """
        if len(self.returns) < 2:
            return 0.0
        
        returns = np.array(self.returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualisé (252 jours de trading)
        sharpe = ((mean_return - risk_free_rate) / std_return) * np.sqrt(252)
        
        # Sauvegarder
        self.db.add_metric('sharpe_ratio', sharpe, 'unified_metrics')
        
        return sharpe
    
    def calculate_max_drawdown(self) -> float:
        """
        Calculer le Max Drawdown - MÉTHODE UNIQUE
        
        Drawdown = (Pic - Valeur actuelle) / Pic
        """
        if not self.portfolio_values:
            return 0.0
        
        values = np.array(self.portfolio_values)
        running_max = np.maximum.accumulate(values)
        drawdown = (running_max - values) / running_max
        max_dd = np.max(drawdown)
        
        # Sauvegarder
        self.db.add_metric('max_drawdown', max_dd, 'unified_metrics')
        
        return max_dd
    
    def calculate_win_rate(self) -> float:
        """
        Calculer le Win Rate - MÉTHODE UNIQUE
        
        Win Rate = Trades gagnants / Total trades
        """
        if not self.trades:
            return 0.0
        
        winning_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        total_trades = len(self.trades)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Sauvegarder
        self.db.add_metric('win_rate', win_rate, 'unified_metrics')
        
        return win_rate
    
    def calculate_profit_factor(self) -> float:
        """
        Calculer le Profit Factor - MÉTHODE UNIQUE
        
        Profit Factor = Gains totaux / Pertes totales
        """
        if not self.trades:
            return 0.0
        
        gains = sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0)
        losses = abs(sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0))
        
        if losses == 0:
            return 0.0
        
        profit_factor = gains / losses
        
        # Sauvegarder
        self.db.add_metric('profit_factor', profit_factor, 'unified_metrics')
        
        return profit_factor
    
    def calculate_total_return(self, initial_capital: float = 10000.0) -> float:
        """
        Calculer le rendement total - MÉTHODE UNIQUE
        
        Total Return = (Valeur finale - Valeur initiale) / Valeur initiale
        """
        if not self.portfolio_values:
            return 0.0
        
        final_value = self.portfolio_values[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Sauvegarder
        self.db.add_metric('total_return', total_return, 'unified_metrics')
        
        return total_return
    
    def calculate_calmar_ratio(self, initial_capital: float = 10000.0) -> float:
        """
        Calculer le Calmar Ratio - MÉTHODE UNIQUE
        
        Calmar Ratio = Rendement annuel / Max Drawdown
        """
        total_return = self.calculate_total_return(initial_capital)
        max_dd = self.calculate_max_drawdown()
        
        if max_dd == 0:
            return 0.0
        
        calmar = total_return / max_dd
        
        # Sauvegarder
        self.db.add_metric('calmar_ratio', calmar, 'unified_metrics')
        
        return calmar
    
    # ========== RAPPORTS ==========
    
    def get_report(self, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """Rapport complet - SOURCE UNIQUE DE VÉRITÉ"""
        
        report = {
            'summary': {
                'total_trades': len(self.trades),
                'total_returns_data_points': len(self.returns),
                'portfolio_values_data_points': len(self.portfolio_values),
            },
            'metrics': {
                'sharpe_ratio': self.calculate_sharpe(),
                'max_drawdown': self.calculate_max_drawdown(),
                'win_rate': self.calculate_win_rate(),
                'profit_factor': self.calculate_profit_factor(),
                'total_return': self.calculate_total_return(initial_capital),
                'calmar_ratio': self.calculate_calmar_ratio(initial_capital),
            },
            'consistency': self.db.validate_consistency(),
            'database_summary': self.db.get_summary()
        }
        
        return report
    
    def get_detailed_report(self, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """Rapport détaillé avec historique"""
        
        report = self.get_report(initial_capital)
        
        # Ajouter l'historique
        report['recent_trades'] = self.db.get_trades(limit=20)
        report['recent_metrics'] = {
            'sharpe': self.db.get_metrics('sharpe_ratio', limit=10),
            'drawdown': self.db.get_metrics('max_drawdown', limit=10),
            'win_rate': self.db.get_metrics('win_rate', limit=10),
        }
        report['recent_validations'] = self.db.get_validations(limit=10)
        
        return report
    
    # ========== VALIDATION ==========
    
    def validate_metrics(self) -> Dict[str, bool]:
        """Valider la cohérence des métriques"""
        
        validations = {}
        
        # Validation 1: Sharpe doit être entre -10 et 10
        sharpe = self.calculate_sharpe()
        validations['sharpe_range'] = -10 <= sharpe <= 10
        if not validations['sharpe_range']:
            self.db.add_validation('sharpe_range', False, f"Sharpe {sharpe} hors limites")
        
        # Validation 2: Drawdown doit être entre 0 et 1
        dd = self.calculate_max_drawdown()
        validations['drawdown_range'] = 0 <= dd <= 1
        if not validations['drawdown_range']:
            self.db.add_validation('drawdown_range', False, f"Drawdown {dd} hors limites")
        
        # Validation 3: Win rate doit être entre 0 et 1
        wr = self.calculate_win_rate()
        validations['win_rate_range'] = 0 <= wr <= 1
        if not validations['win_rate_range']:
            self.db.add_validation('win_rate_range', False, f"Win rate {wr} hors limites")
        
        # Validation 4: Profit factor doit être positif
        pf = self.calculate_profit_factor()
        validations['profit_factor_positive'] = pf >= 0
        if not validations['profit_factor_positive']:
            self.db.add_validation('profit_factor_positive', False, f"Profit factor {pf} négatif")
        
        # Validation 5: Cohérence base de données
        consistency = self.db.validate_consistency()
        validations['db_consistency'] = consistency['consistent']
        if not validations['db_consistency']:
            self.db.add_validation('db_consistency', False, f"Incohérence: {consistency}")
        
        return validations
    
    def print_report(self, initial_capital: float = 10000.0):
        """Afficher le rapport de manière lisible"""
        
        report = self.get_report(initial_capital)
        
        print("=" * 70)
        print("📊 RAPPORT MÉTRIQUES UNIFIÉES")
        print("=" * 70)
        print()
        
        print("📈 RÉSUMÉ:")
        for key, value in report['summary'].items():
            print(f"  {key}: {value}")
        print()
        
        print("📊 MÉTRIQUES:")
        for key, value in report['metrics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()
        
        print("✅ COHÉRENCE:")
        for key, value in report['consistency'].items():
            print(f"  {key}: {value}")
        print()
        
        print("=" * 70)

