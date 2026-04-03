#!/usr/bin/env python3
"""
🎯 BASE DE DONNÉES UNIFIÉE - Source unique pour TOUTES les métriques
Résout: Métriques incohérentes, calculs différents, pas de validation
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import threading

class UnifiedMetricsDB:
    """Base de données unique pour TOUTES les métriques"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: str = "metrics.db"):
        """Singleton - une seule instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: str = "metrics.db"):
        """Initialiser la base de données (singleton) et gérer un éventuel changement de fichier.

        Comportement:
        - Première initialisation: crée la connexion sur ``db_path`` et les tables nécessaires.
        - Appels suivants avec le *même* ``db_path``: ne font rien (singleton classique).
        - Appels suivants avec un **autre** ``db_path``: ferment proprement l'ancienne connexion,
          reconfigurent la base vers le nouveau fichier et recréent les tables.

        Cela permet aux tests (ex: ``test_foundations.db``) d'utiliser un fichier dédié tout en
        conservant une instance unique dans le processus.
        """

        # Si l'instance a déjà été initialisée, vérifier si le chemin demandé est différent
        if getattr(self, "_initialized", False):
            new_path = Path(db_path)
            if new_path != getattr(self, "db_path", new_path):
                # Changement explicite de fichier DB: reconfigurer proprement
                lock = getattr(self, "_lock", threading.Lock())
                with lock:
                    conn = getattr(self, "conn", None)
                    if conn is not None:
                        conn.close()
                    self.db_path = new_path
                    self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                    self.cursor = self.conn.cursor()
                    # (Re)créer les tables sur ce nouveau fichier
                    self._create_tables()
                    # S'assurer que le lock est bien défini pour les appels suivants
                    self._lock = lock
            return

        # Première initialisation du singleton
        self._initialized = True
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._lock = threading.Lock()
        self._create_tables()
    
    def _create_tables(self):
        """Créer les tables si elles n'existent pas"""
        
        # Table 1: Métriques
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                source TEXT,
                validation_status TEXT DEFAULT 'pending'
            )
        ''')
        
        # Table 2: Trades
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                action TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                pnl REAL,
                status TEXT DEFAULT 'executed',
                validation_status TEXT DEFAULT 'pending'
            )
        ''')
        
        # Table 3: Validations
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                check_name TEXT NOT NULL,
                passed BOOLEAN,
                details TEXT
            )
        ''')
        
        # Table 4: Synchronisations
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS synchronizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                component TEXT NOT NULL,
                status TEXT,
                details TEXT
            )
        ''')
        
        # Index pour performance
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_name ON metrics(metric_name)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_timestamp ON metrics(timestamp)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_symbol ON trades(symbol)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_timestamp ON trades(timestamp)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_validation_name ON validations(check_name)')
        
        self.conn.commit()
    
    # ========== MÉTRIQUES ==========
    
    def add_metric(self, name: str, value: float, source: str = "unknown") -> bool:
        """Ajouter une métrique"""
        try:
            with self._lock:
                self.cursor.execute(
                    'INSERT INTO metrics (metric_name, metric_value, source) VALUES (?, ?, ?)',
                    (name, value, source)
                )
                self.conn.commit()
            return True
        except Exception as e:
            print(f"❌ Erreur ajout métrique: {e}")
            return False
    
    def get_metrics(self, name: str, limit: int = 100) -> List[Dict]:
        """Récupérer les dernières métriques"""
        try:
            with self._lock:
                self.cursor.execute(
                    'SELECT id, timestamp, metric_name, metric_value, source FROM metrics WHERE metric_name = ? ORDER BY timestamp DESC LIMIT ?',
                    (name, limit)
                )
                rows = self.cursor.fetchall()
            
            return [
                {
                    'id': row[0],
                    'timestamp': row[1],
                    'name': row[2],
                    'value': row[3],
                    'source': row[4]
                }
                for row in rows
            ]
        except Exception as e:
            print(f"❌ Erreur lecture métriques: {e}")
            return []
    
    def get_latest_metric(self, name: str) -> Optional[float]:
        """Récupérer la dernière valeur d'une métrique"""
        try:
            with self._lock:
                self.cursor.execute(
                    'SELECT metric_value FROM metrics WHERE metric_name = ? ORDER BY timestamp DESC LIMIT 1',
                    (name,)
                )
                row = self.cursor.fetchone()
            return row[0] if row else None
        except Exception as e:
            print(f"❌ Erreur lecture dernière métrique: {e}")
            return None
    
    # ========== TRADES ==========
    
    def add_trade(self, action: str, symbol: str, quantity: float, price: float, pnl: Optional[float] = None) -> bool:
        """Ajouter un trade"""
        try:
            with self._lock:
                self.cursor.execute(
                    'INSERT INTO trades (action, symbol, quantity, price, pnl) VALUES (?, ?, ?, ?, ?)',
                    (action, symbol, quantity, price, pnl)
                )
                self.conn.commit()
            return True
        except Exception as e:
            print(f"❌ Erreur ajout trade: {e}")
            return False
    
    def get_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Récupérer les derniers trades"""
        try:
            with self._lock:
                if symbol:
                    self.cursor.execute(
                        'SELECT id, timestamp, action, symbol, quantity, price, pnl FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?',
                        (symbol, limit)
                    )
                else:
                    self.cursor.execute(
                        'SELECT id, timestamp, action, symbol, quantity, price, pnl FROM trades ORDER BY timestamp DESC LIMIT ?',
                        (limit,)
                    )
                rows = self.cursor.fetchall()
            
            return [
                {
                    'id': row[0],
                    'timestamp': row[1],
                    'action': row[2],
                    'symbol': row[3],
                    'quantity': row[4],
                    'price': row[5],
                    'pnl': row[6]
                }
                for row in rows
            ]
        except Exception as e:
            print(f"❌ Erreur lecture trades: {e}")
            return []
    
    def get_trade_count(self) -> int:
        """Compter le nombre total de trades"""
        try:
            with self._lock:
                self.cursor.execute('SELECT COUNT(*) FROM trades')
                result = self.cursor.fetchone()
            return result[0] if result else 0
        except Exception as e:
            print(f"❌ Erreur comptage trades: {e}")
            return 0
    
    # ========== VALIDATIONS ==========
    
    def add_validation(self, check_name: str, passed: bool, details: str = "") -> bool:
        """Ajouter une validation"""
        try:
            with self._lock:
                self.cursor.execute(
                    'INSERT INTO validations (check_name, passed, details) VALUES (?, ?, ?)',
                    (check_name, passed, details)
                )
                self.conn.commit()
            return True
        except Exception as e:
            print(f"❌ Erreur ajout validation: {e}")
            return False
    
    def get_validations(self, check_name: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Récupérer les validations"""
        try:
            with self._lock:
                if check_name:
                    self.cursor.execute(
                        'SELECT id, timestamp, check_name, passed, details FROM validations WHERE check_name = ? ORDER BY timestamp DESC LIMIT ?',
                        (check_name, limit)
                    )
                else:
                    self.cursor.execute(
                        'SELECT id, timestamp, check_name, passed, details FROM validations ORDER BY timestamp DESC LIMIT ?',
                        (limit,)
                    )
                rows = self.cursor.fetchall()
            
            return [
                {
                    'id': row[0],
                    'timestamp': row[1],
                    'check_name': row[2],
                    'passed': row[3],
                    'details': row[4]
                }
                for row in rows
            ]
        except Exception as e:
            print(f"❌ Erreur lecture validations: {e}")
            return []
    
    # ========== SYNCHRONISATIONS ==========
    
    def add_sync(self, component: str, status: str, details: str = "") -> bool:
        """Ajouter une synchronisation"""
        try:
            with self._lock:
                self.cursor.execute(
                    'INSERT INTO synchronizations (component, status, details) VALUES (?, ?, ?)',
                    (component, status, details)
                )
                self.conn.commit()
            return True
        except Exception as e:
            print(f"❌ Erreur ajout sync: {e}")
            return False
    
    # ========== VÉRIFICATIONS ==========
    
    def validate_consistency(self) -> Dict[str, Any]:
        """Vérifier la cohérence des données"""
        try:
            with self._lock:
                self.cursor.execute('SELECT COUNT(*) FROM trades')
                trades_count = self.cursor.fetchone()[0]
                
                self.cursor.execute('SELECT COUNT(*) FROM metrics')
                metrics_count = self.cursor.fetchone()[0]
                
                self.cursor.execute('SELECT COUNT(*) FROM validations')
                validations_count = self.cursor.fetchone()[0]
            
            return {
                'trades': trades_count,
                'metrics': metrics_count,
                'validations': validations_count,
                'consistent': trades_count > 0 and metrics_count > 0,
                'status': '✅ OK' if (trades_count > 0 and metrics_count > 0) else '⚠️ Données manquantes'
            }
        except Exception as e:
            print(f"❌ Erreur validation cohérence: {e}")
            return {'error': str(e)}
    
    def get_summary(self) -> Dict[str, Any]:
        """Résumé complet de la base de données"""
        consistency = self.validate_consistency()
        
        return {
            'database': str(self.db_path),
            'consistency': consistency,
            'latest_metrics': {
                'sharpe': self.get_latest_metric('sharpe_ratio'),
                'drawdown': self.get_latest_metric('max_drawdown'),
                'win_rate': self.get_latest_metric('win_rate')
            },
            'recent_trades': len(self.get_trades(limit=10)),
            'recent_validations': len(self.get_validations(limit=10))
        }
    
    def close(self):
        """Fermer la connexion"""
        if self.conn:
            self.conn.close()

