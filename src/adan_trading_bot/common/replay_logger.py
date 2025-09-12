"""
ReplayLogger - Module de journalisation avancée pour le Dynamic Behavior Engine (DBE).

Ce module offre une journalisation complète et structurée des décisions du DBE,
ainsi que des outils d'analyse pour comprendre les performances du système.
"""
import json
import zlib
import gzip
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
import logging

# Configuration du logging
logger = logging.getLogger(__name__)

class ReplayLogger:
    """
    Journalise les décisions du DBE dans un fichier JSONL pour analyse ultérieure.
    
    Chaque ligne du fichier représente une décision avec son contexte.
    """
    
    def __init__(self, log_dir: str = "logs/dbe", compression: str = 'gzip'):
        """
        Initialise le logger de relecture avancé.
        
        Args:
            log_dir: Répertoire de sortie pour les fichiers de log
            compression: Type de compression à utiliser ('none', 'gzip', 'zlib')
        """
        # Configuration
        self.compression = compression.lower()
        assert self.compression in ('none', 'gzip', 'zlib'), "Compression non supportée"
        
        # Création du répertoire de logs
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Création du chemin du fichier de log avec horodatage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"dbe_replay_{timestamp}"
        
        # Gestion des extensions de fichier selon la compression
        if self.compression == 'gzip':
            self.log_file = self.log_dir / f"{base_filename}.jsonl.gz"
        elif self.compression == 'zlib':
            self.log_file = self.log_dir / f"{base_filename}.jsonl.zlib"
        else:
            self.log_file = self.log_dir / f"{base_filename}.jsonl"
        
        # Initialisation des compteurs et états
        self.decision_count = 0
        self.episode_count = 0
        self.last_flush = datetime.now()
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_size = 100  # Nombre d'entrées avant vidage automatique
        
        # En-tête du fichier de log
        self._write_metadata()
        logger.info(f"ReplayLogger initialisé - Fichier: {self.log_file}")
    
    def _write_metadata(self) -> None:
        """Écrit les métadonnées au début du fichier de log."""
        import platform
        import socket
        import getpass
        
        metadata = {
            "type": "metadata",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "format": "jsonl",
            "compression": self.compression,
            "description": "DBE Decision Log - Enhanced Version",
            "system": {
                "hostname": socket.gethostname(),
                "username": getpass.getuser(),
                "platform": platform.platform(),
                "python_version": platform.python_version()
            },
            "fields": {
                "decision": ["timestamp", "step", "modulation", "context", "performance_metrics"],
                "episode_end": ["timestamp", "episode", "total_reward", "metrics"],
                "error": ["timestamp", "error_type", "message", "context"]
            }
        }
        
        self._write_line(metadata)
        logger.debug("Métadonnées du journal enregistrées")
    
    def log_decision(
        self, 
        step_index: int, 
        modulation_dict: Dict[str, Any], 
        context_metrics: Dict[str, Any],
        performance_metrics: Optional[Dict[str, Any]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        flush: bool = False
    ) -> None:
        """
        Enregistre une décision du DBE avec des métriques avancées.
        
        Args:
            step_index: Numéro de l'étape actuelle
            modulation_dict: Dictionnaire des paramètres modulés
            context_metrics: Métriques de contexte (état du marché, portefeuille, etc.)
            performance_metrics: Métriques de performance (ex: PnL, drawdown, etc.)
            additional_info: Informations supplémentaires optionnelles
            flush: Force l'écriture immédiate sur disque
        """
        log_entry = {
            "type": "decision",
            "timestamp": datetime.utcnow().isoformat() + 'Z',  # UTC avec Z pour le fuseau
            "step": step_index,
            "modulation": self._serialize(modulation_dict),
            "context": self._serialize(context_metrics),
            "performance_metrics": self._serialize(performance_metrics or {})
        }
        
        if additional_info:
            log_entry["additional_info"] = self._serialize(additional_info)
        
        # Ajout au buffer
        self.buffer.append(log_entry)
        self.decision_count += 1
        
        # Vérifier si on doit vider le buffer
        if len(self.buffer) >= self.buffer_size or flush or self._should_flush():
            self.flush()
            
        logger.debug(f"Décision enregistrée - Étape {step_index}")
        
        return log_entry  # Retourne l'entrée pour un traitement ultérieur si nécessaire
    
    def log_episode_end(
        self, 
        episode: int, 
        total_reward: float, 
        episode_metrics: Dict[str, Any],
        flush: bool = True
    ) -> None:
        """
        Enregistre la fin d'un épisode avec des statistiques détaillées.
        
        Args:
            episode: Numéro de l'épisode
            total_reward: Récompense totale de l'épisode
            episode_metrics: Métriques de l'épisode
            flush: Si True, force l'écriture immédiate sur disque
        """
        log_entry = {
            "type": "episode_end",
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "episode": episode,
            "total_reward": float(total_reward),  # S'assurer que c'est sérialisable
            "metrics": self._serialize(episode_metrics),
            "decision_count": self.decision_count,
            "duration_seconds": (datetime.utcnow() - self.last_flush).total_seconds()
        }
        
        # Ajout au buffer mais on flush immédiatement pour les fins d'épisode
        self.buffer.append(log_entry)
        self.episode_count += 1
        
        if flush:
            self.flush()
            
        logger.info(
            f"Fin d'épisode {episode} - "
            f"Récompense: {total_reward:.2f} - "
            f"Décisions: {self.decision_count}"
        )
        
        # Réinitialisation des compteurs pour le prochain épisode
        self.decision_count = 0
        self.last_flush = datetime.utcnow()
    
    def _write_line(self, data: Dict[str, Any]) -> None:
        """
        Écrit une ligne dans le fichier de log avec gestion de la compression.
        
        Args:
            data: Données à écrire (doivent être sérialisables en JSON)
        """
        try:
            line = json.dumps(data, ensure_ascii=False, default=str) + '\n'
            if self.compression == 'gzip':
                with gzip.open(self.log_file, 'at', encoding='utf-8') as f:
                    f.write(line)
            elif self.compression == 'zlib':
                with open(self.log_file, 'ab') as f:
                    compressed = zlib.compress(line.encode('utf-8'))
                    f.write(compressed + b'\n')
            else:  # Pas de compression
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(line)
                    
        except (IOError, TypeError, ValueError) as e:
            logger.error(f"Erreur lors de l'écriture dans le journal: {e}")
            # Essayer d'écrire dans un fichier de secours
            self._write_fallback(data)
    
    def _write_fallback(self, data: Dict[str, Any]) -> None:
        """Écrit dans un fichier de secours en cas d'erreur."""
        try:
            fallback_file = self.log_file.parent / f"{self.log_file.stem}_fallback.jsonl"
            with open(fallback_file, 'a', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, default=str)
                f.write('\n')
            logger.warning(f"Écriture de secours dans {fallback_file}")
        except Exception as e:
            logger.critical(f"Échec critique de l'écriture de secours: {e}")
    
    def flush(self) -> None:
        """Vide le buffer dans le fichier de log."""
        if not self.buffer:
            return
            
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                for entry in self.buffer:
                    line = json.dumps(entry, ensure_ascii=False, default=str)
                    f.write(line + '\n')
            
            logger.debug(f"Buffer vidé - {len(self.buffer)} entrées écrites")
            self.buffer = []
            
        except Exception as e:
            logger.error(f"Erreur lors du vidage du buffer: {e}")
    
    def _should_flush(self) -> bool:
        """Détermine si le buffer doit être vidé."""
        # Vérifier le temps écoulé depuis le dernier flush
        time_since_flush = (datetime.utcnow() - self.last_flush).total_seconds()
        return time_since_flush > 300  # 5 minutes
    
    @staticmethod
    def _serialize(data: Any) -> Any:
        """
        S'assure que les données sont sérialisables en JSON.
        
        Args:
            data: Données à sérialiser
            
        Returns:
            Données sérialisables
        """
        if data is None:
            return None
            
        if isinstance(data, (str, int, float, bool)) or data is None:
            return data
            
        if isinstance(data, (list, tuple)):
            return [ReplayLogger._serialize(item) for item in data]
            
        if isinstance(data, dict):
            return {str(k): ReplayLogger._serialize(v) for k, v in data.items()}
            
        if isinstance(data, np.integer):
            return int(data)
            
        if isinstance(data, np.floating):
            return float(data)
            
        if isinstance(data, np.ndarray):
            return data.tolist()
            
        if hasattr(data, 'isoformat'):  # Pour les objets datetime
            return data.isoformat()
            
        # Essayer de convertir en dict si possible
        if hasattr(data, '__dict__'):
            return ReplayLogger._serialize(data.__dict__)
            
        # Dernier recours : conversion en chaîne
        return str(data)
    
    def get_log_path(self) -> str:
        """
        Retourne le chemin du fichier de log actuel.
        
        Returns:
            Chemin absolu du fichier de log
        """
        return str(self.log_file.absolute())
    
    def load_logs(self, log_file: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Charge les logs dans un DataFrame pandas pour analyse.
        
        Args:
            log_file: Chemin vers le fichier de log (par défaut: dernier fichier créé)
            
        Returns:
            DataFrame contenant les logs
        """
        log_file = Path(log_file) if log_file else self.log_file
        
        if not log_file.exists():
            raise FileNotFoundError(f"Fichier de log non trouvé: {log_file}")
            
        logs = []
        
        try:
            if log_file.suffix == '.gz':
                import gzip
                with gzip.open(log_file, 'rt', encoding='utf-8') as f:
                    for line in f:
                        logs.append(json.loads(line))
            elif log_file.suffix == '.zlib':
                with open(log_file, 'rb') as f:
                    for line in f:
                        if line.strip():
                            decompressed = zlib.decompress(line).decode('utf-8')
                            logs.append(json.loads(decompressed))
            else:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            logs.append(json.loads(line))
            
            return pd.DataFrame(logs)
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des logs: {e}")
            raise
    
    def analyze_decision_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyse les motifs de décision à partir des logs.
        
        Args:
            df: DataFrame des logs
            
        Returns:
            Dictionnaire contenant les analyses
        """
        if df.empty:
            return {}
            
        # Filtrer uniquement les décisions
        decisions = df[df['type'] == 'decision'].copy()
        
        if decisions.empty:
            return {}
            
        # Extraire les métriques de performance
        decisions['pnl'] = decisions['performance_metrics'].apply(
            lambda x: x.get('pnl', 0) if isinstance(x, dict) else 0
        )
        
        # Calculer les statistiques de base
        stats = {
            'total_decisions': len(decisions),
            'avg_pnl': decisions['pnl'].mean(),
            'win_rate': (decisions['pnl'] > 0).mean(),
            'avg_holding_time': decisions['context'].apply(
                lambda x: x.get('holding_time', 0) if isinstance(x, dict) else 0
            ).mean(),
            'modulation_stats': {
                'avg_sl': decisions['modulation'].apply(
                    lambda x: x.get('sl_pct', 0) if isinstance(x, dict) else 0
                ).mean(),
                'avg_tp': decisions['modulation'].apply(
                    lambda x: x.get('tp_pct', 0) if isinstance(x, dict) else 0
                ).mean(),
            }
        }
        
        return stats


# Utilisation simplifiée pour les tests
if __name__ == "__main__":
    logger = ReplayLogger()
    
    # Exemple d'utilisation
    logger.log_decision(
        step_index=1,
        modulation_dict={"sl_pct": 0.02, "tp_pct": 0.04, "risk_mode": "NORMAL"},
        context_metrics={"volatility": 0.015, "drawdown": 0.05, "winrate": 0.55}
    )
    
    print(f"Logs enregistrés dans : {logger.get_log_path()}")

# Exemple d'utilisation avancée
if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(level=logging.INFO)
    
    # Création du logger avec compression gzip
    logger = ReplayLogger(compression='gzip')
    
    try:
        # Exemple d'enregistrement d'une décision
        decision = logger.log_decision(
            step_index=1,
            modulation_dict={
                "sl_pct": 0.02, 
                "tp_pct": 0.04,
                "risk_mode": "AGGRESSIVE"
            },
            context_metrics={
                "drawdown": 0.01, 
                "volatility": 0.15,
                "market_regime": "TRENDING_UP"
            },
            performance_metrics={
                "pnl": 15.50,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.05
            },
            additional_info={
                "reason": "Breakout détecté sur timeframe H1",
                "confidence": 0.85
            }
        )
        
        # Fin d'épisode
        logger.log_episode_end(
            episode=1,
            total_reward=150.75,
            episode_metrics={
                "win_rate": 0.65,
                "avg_trade": 12.50,
                "max_drawdown": 0.07,
                "sharpe_ratio": 1.1
            }
        )
        
        # Chargement et analyse des logs
        df = logger.load_logs()
        if not df.empty:
            analysis = logger.analyze_decision_patterns(df)
            print("\nAnalyse des décisions:")
            print(f"- Décisions totales: {analysis.get('total_decisions', 0)}")
            print(f"- PnL moyen: {analysis.get('avg_pnl', 0):.2f}")
            print(f"- Taux de réussite: {analysis.get('win_rate', 0) * 100:.1f}%")
            
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de l'exemple: {e}")
    finally:
        # S'assurer que tout est bien écrit sur le disque
        logger.flush()
    
    print(f"Logs enregistrés dans : {logger.get_log_path()}")
