#!/usr/bin/env python3
"""
🎯 LOGGER CENTRALISÉ - Source unique pour TOUS les logs
Résout: Erreurs de transmission, logs dispersés, format incohérent
"""

import logging
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class CentralLogger:
    """Logger centralisé pour tout le projet ADAN"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton - une seule instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialiser le logger une seule fois"""
        if self._initialized:
            return
        
        self._initialized = True
        self.logger = logging.getLogger("ADAN_CENTRAL")
        self.logger.setLevel(logging.DEBUG)
        
        # Créer le répertoire des logs
        self.log_dir = Path("logs/central")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Handler Console (affichage)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        
        # 2. Handler Fichier (stockage)
        today = datetime.now().strftime('%Y%m%d')
        file_handler = logging.FileHandler(
            self.log_dir / f"adan_{today}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        
        # 3. Handler JSON (pour parsing)
        json_handler = logging.FileHandler(
            self.log_dir / f"adan_{today}.jsonl"
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        
        # Ajouter les handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(json_handler)
    
    # ========== MÉTHODES SPÉCIALISÉES ==========
    
    def trade(self, action: str, symbol: str, quantity: float, price: float, 
              pnl: Optional[float] = None, **kwargs):
        """Log un trade et le persiste dans la base de données"""
        msg = f"[TRADE] {action} {quantity} {symbol} @ ${price:.2f}"
        if pnl is not None:
            msg += f" | PnL: ${pnl:.2f}"
        self.logger.info(msg, extra={'trade_data': kwargs})
        
        # Persister dans la base de données
        try:
            from ..performance.unified_metrics_db import UnifiedMetricsDB
            db = UnifiedMetricsDB()
            db.add_trade(action, symbol, quantity, price, pnl)
        except Exception as e:
            self.logger.debug(f"[TRADE_DB] Erreur persistance trade: {e}")
    
    def metric(self, name: str, value: float, unit: str = "", **kwargs):
        """Log une métrique et la persiste dans la base de données"""
        msg = f"[METRIC] {name}: {value:.4f} {unit}"
        self.logger.info(msg, extra={'metric_data': kwargs})
        
        # Persister dans la base de données
        try:
            from ..performance.unified_metrics_db import UnifiedMetricsDB
            db = UnifiedMetricsDB()
            db.add_metric(name, value, source=kwargs.get('source', 'central_logger'))
        except Exception as e:
            self.logger.debug(f"[METRIC_DB] Erreur persistance métrique {name}: {e}")
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log une erreur"""
        self.logger.error(f"[ERROR] {message}", exc_info=exc_info, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log un avertissement"""
        self.logger.warning(f"[WARNING] {message}", extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log un debug"""
        self.logger.debug(f"[DEBUG] {message}", extra=kwargs)
    
    def sync(self, component: str, status: str, details: Dict[str, Any]):
        """Log une synchronisation et la persiste dans la base de données"""
        msg = f"[SYNC] {component}: {status}"
        self.logger.info(msg, extra={'sync_data': details})
        
        # Persister dans la base de données
        try:
            from ..performance.unified_metrics_db import UnifiedMetricsDB
            db = UnifiedMetricsDB()
            details_str = json.dumps(details) if isinstance(details, dict) else str(details)
            db.add_sync(component, status, details_str)
        except Exception as e:
            self.logger.debug(f"[SYNC_DB] Erreur persistance sync: {e}")
    
    def validation(self, check_name: str, passed: bool, details: str = ""):
        """Log une validation et la persiste dans la base de données"""
        status = "✅ PASS" if passed else "❌ FAIL"
        msg = f"[VALIDATION] {check_name}: {status}"
        if details:
            msg += f" | {details}"
        level = logging.INFO if passed else logging.WARNING
        self.logger.log(level, msg)
        
        # Persister dans la base de données
        try:
            from ..performance.unified_metrics_db import UnifiedMetricsDB
            db = UnifiedMetricsDB()
            db.add_validation(check_name, passed, details)
        except Exception as e:
            self.logger.debug(f"[VALIDATION_DB] Erreur persistance validation: {e}")


class JSONFormatter(logging.Formatter):
    """Formatter pour logs en JSON"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # Ajouter les données extra si présentes
        if hasattr(record, 'trade_data'):
            log_data['trade_data'] = record.trade_data
        if hasattr(record, 'metric_data'):
            log_data['metric_data'] = record.metric_data
        if hasattr(record, 'sync_data'):
            log_data['sync_data'] = record.sync_data
        
        return json.dumps(log_data, default=str)


# Instance globale
logger = CentralLogger()

