"""
Système de monitoring de santé pour ADAN Trading Bot.
Implémente la tâche 10B.3.3.
"""

import psutil
import logging
import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import queue
import gc

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """États de santé"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Niveaux d'alerte"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """Métriques système"""
    timestamp: datetime
    
    # CPU
    cpu_percent: float
    cpu_count: int
    
    # Mémoire
    memory_total: int
    memory_available: int
    memory_percent: float
    memory_used: int
    
    # Disque
    disk_total: int
    disk_used: int
    disk_free: int
    disk_percent: float
    
    # Réseau
    network_sent: int
    network_recv: int
    
    # Processus
    process_count: int
    thread_count: int
    
    # Optionnel
    load_average: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class HealthAlert:
    """Alerte de santé système"""
    alert_id: str
    level: AlertLevel
    component: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        data = asdict(self)
        data['level'] = self.level.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data


class ComponentHealthChecker:
    """Vérificateur de santé pour un composant"""
    
    def __init__(self, name: str, check_function: Callable[[], Dict[str, Any]],
                 check_interval: int = 60):
        self.name = name
        self.check_function = check_function
        self.check_interval = check_interval
        self.last_check = None
        self.last_status = HealthStatus.UNKNOWN
        self.last_details = {}
    
    def check_health(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Vérifie la santé du composant"""
        try:
            details = self.check_function()
            status = details.get('status', HealthStatus.HEALTHY)
            
            self.last_check = datetime.now()
            self.last_status = status
            self.last_details = details
            
            return status, details
            
        except Exception as e:
            error_details = {
                'status': HealthStatus.CRITICAL,
                'error': str(e),
                'error_type': type(e).__name__
            }
            
            self.last_check = datetime.now()
            self.last_status = HealthStatus.CRITICAL
            self.last_details = error_details
            
            return HealthStatus.CRITICAL, error_details


class SystemHealthMonitor:
    """Moniteur de santé système"""
    
    def __init__(self, check_interval: int = 30, history_size: int = 1000):
        self.check_interval = check_interval
        self.history_size = history_size
        
        # Métriques et historique
        self.metrics_history: List[SystemMetrics] = []
        self.alerts: Dict[str, HealthAlert] = {}
        self.alert_history: List[HealthAlert] = []
        
        # Composants à surveiller
        self.component_checkers: Dict[str, ComponentHealthChecker] = {}
        
        # Seuils d'alerte
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'load_warning': None,  # Sera calculé selon le nombre de CPUs
            'load_critical': None
        }
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        self.metrics_callbacks: List[Callable] = []
        
        # Threading
        self.monitor_thread = None
        self.stop_monitoring = False
        self.metrics_queue = queue.Queue()
        
        # Initialisation
        self._setup_default_thresholds()
        self._register_default_components()
        
        logger.info("SystemHealthMonitor initialized")
    
    def _setup_default_thresholds(self) -> None:
        """Configure les seuils par défaut"""
        cpu_count = psutil.cpu_count()
        self.thresholds['load_warning'] = cpu_count * 0.8
        self.thresholds['load_critical'] = cpu_count * 1.2
    
    def _register_default_components(self) -> None:
        """Enregistre les composants par défaut"""
        # Vérificateur de base de données (si applicable)
        self.register_component(
            "database",
            self._check_database_health,
            check_interval=60
        )
        
        # Vérificateur de cache
        self.register_component(
            "cache",
            self._check_cache_health,
            check_interval=30
        )
        
        # Vérificateur de logs
        self.register_component(
            "logging",
            self._check_logging_health,
            check_interval=120
        )
    
    def register_component(self, name: str, check_function: Callable[[], Dict[str, Any]],
                          check_interval: int = 60) -> None:
        """Enregistre un composant à surveiller"""
        checker = ComponentHealthChecker(name, check_function, check_interval)
        self.component_checkers[name] = checker
        logger.info(f"Registered health checker for component: {name}")
    
    def start_monitoring(self) -> None:
        """Démarre le monitoring"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Monitoring already active")
            return
        
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("System health monitoring started")
    
    def stop_monitoring_process(self) -> None:
        """Arrête le monitoring"""
        self.stop_monitoring = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("System health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Boucle principale de monitoring"""
        while not self.stop_monitoring:
            try:
                # Collecter les métriques système
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Limiter l'historique
                if len(self.metrics_history) > self.history_size:
                    self.metrics_history.pop(0)
                
                # Vérifier les seuils
                self._check_system_thresholds(metrics)
                
                # Vérifier les composants
                self._check_components_health()
                
                # Notifier les callbacks
                self._notify_metrics_callbacks(metrics)
                
                # Nettoyer les anciennes alertes
                self._cleanup_old_alerts()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collecte les métriques système"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        try:
            load_average = list(psutil.getloadavg())
        except AttributeError:
            load_average = None  # Windows n'a pas getloadavg
        
        # Mémoire
        memory = psutil.virtual_memory()
        
        # Disque (partition racine)
        disk = psutil.disk_usage('/')
        
        # Réseau
        network = psutil.net_io_counters()
        
        # Processus
        process_count = len(psutil.pids())
        
        # Threads (approximation)
        thread_count = 0
        try:
            for proc in psutil.process_iter(['num_threads']):
                try:
                    thread_count += proc.info['num_threads'] or 0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except:
            thread_count = 0
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            load_average=load_average,
            memory_total=memory.total,
            memory_available=memory.available,
            memory_percent=memory.percent,
            memory_used=memory.used,
            disk_total=disk.total,
            disk_used=disk.used,
            disk_free=disk.free,
            disk_percent=disk.percent,
            network_sent=network.bytes_sent,
            network_recv=network.bytes_recv,
            process_count=process_count,
            thread_count=thread_count
        )
    
    def _check_system_thresholds(self, metrics: SystemMetrics) -> None:
        """Vérifie les seuils système"""
        # CPU
        if metrics.cpu_percent >= self.thresholds['cpu_critical']:
            self._create_alert(
                AlertLevel.CRITICAL,
                "system_cpu",
                f"CPU usage critical: {metrics.cpu_percent:.1f}%",
                {'cpu_percent': metrics.cpu_percent, 'threshold': self.thresholds['cpu_critical']}
            )
        elif metrics.cpu_percent >= self.thresholds['cpu_warning']:
            self._create_alert(
                AlertLevel.WARNING,
                "system_cpu",
                f"CPU usage high: {metrics.cpu_percent:.1f}%",
                {'cpu_percent': metrics.cpu_percent, 'threshold': self.thresholds['cpu_warning']}
            )
        else:
            self._resolve_alert("system_cpu")
        
        # Mémoire
        if metrics.memory_percent >= self.thresholds['memory_critical']:
            self._create_alert(
                AlertLevel.CRITICAL,
                "system_memory",
                f"Memory usage critical: {metrics.memory_percent:.1f}%",
                {'memory_percent': metrics.memory_percent, 'threshold': self.thresholds['memory_critical']}
            )
        elif metrics.memory_percent >= self.thresholds['memory_warning']:
            self._create_alert(
                AlertLevel.WARNING,
                "system_memory",
                f"Memory usage high: {metrics.memory_percent:.1f}%",
                {'memory_percent': metrics.memory_percent, 'threshold': self.thresholds['memory_warning']}
            )
        else:
            self._resolve_alert("system_memory")
        
        # Disque
        if metrics.disk_percent >= self.thresholds['disk_critical']:
            self._create_alert(
                AlertLevel.CRITICAL,
                "system_disk",
                f"Disk usage critical: {metrics.disk_percent:.1f}%",
                {'disk_percent': metrics.disk_percent, 'threshold': self.thresholds['disk_critical']}
            )
        elif metrics.disk_percent >= self.thresholds['disk_warning']:
            self._create_alert(
                AlertLevel.WARNING,
                "system_disk",
                f"Disk usage high: {metrics.disk_percent:.1f}%",
                {'disk_percent': metrics.disk_percent, 'threshold': self.thresholds['disk_warning']}
            )
        else:
            self._resolve_alert("system_disk")
        
        # Load average (si disponible)
        if metrics.load_average and len(metrics.load_average) > 0:
            load_1min = metrics.load_average[0]
            
            if load_1min >= self.thresholds['load_critical']:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    "system_load",
                    f"System load critical: {load_1min:.2f}",
                    {'load_1min': load_1min, 'threshold': self.thresholds['load_critical']}
                )
            elif load_1min >= self.thresholds['load_warning']:
                self._create_alert(
                    AlertLevel.WARNING,
                    "system_load",
                    f"System load high: {load_1min:.2f}",
                    {'load_1min': load_1min, 'threshold': self.thresholds['load_warning']}
                )
            else:
                self._resolve_alert("system_load")
    
    def _check_components_health(self) -> None:
        """Vérifie la santé des composants"""
        for name, checker in self.component_checkers.items():
            # Vérifier si c'est le moment de checker
            if (checker.last_check is None or 
                datetime.now() - checker.last_check >= timedelta(seconds=checker.check_interval)):
                
                status, details = checker.check_health()
                
                if status == HealthStatus.CRITICAL:
                    self._create_alert(
                        AlertLevel.CRITICAL,
                        f"component_{name}",
                        f"Component {name} is critical",
                        details
                    )
                elif status == HealthStatus.WARNING:
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"component_{name}",
                        f"Component {name} has warnings",
                        details
                    )
                else:
                    self._resolve_alert(f"component_{name}")
    
    def _check_database_health(self) -> Dict[str, Any]:
        """Vérifie la santé de la base de données"""
        # Placeholder - à implémenter selon la base utilisée
        return {
            'status': HealthStatus.HEALTHY,
            'connection_pool_size': 10,
            'active_connections': 2,
            'query_response_time_ms': 15
        }
    
    def _check_cache_health(self) -> Dict[str, Any]:
        """Vérifie la santé du cache"""
        try:
            # Vérifier l'usage mémoire du cache
            cache_size_mb = 0
            
            # Forcer le garbage collection pour avoir des stats précises
            gc.collect()
            
            return {
                'status': HealthStatus.HEALTHY,
                'cache_size_mb': cache_size_mb,
                'hit_rate': 0.85,  # Exemple
                'memory_usage_ok': cache_size_mb < 1000
            }
        except Exception as e:
            return {
                'status': HealthStatus.WARNING,
                'error': str(e)
            }
    
    def _check_logging_health(self) -> Dict[str, Any]:
        """Vérifie la santé du système de logs"""
        try:
            log_dir = Path("logs")
            
            if not log_dir.exists():
                return {
                    'status': HealthStatus.WARNING,
                    'message': "Log directory does not exist"
                }
            
            # Vérifier la taille des logs
            total_size = sum(f.stat().st_size for f in log_dir.rglob('*') if f.is_file())
            total_size_mb = total_size / (1024 * 1024)
            
            # Vérifier les permissions d'écriture
            test_file = log_dir / "health_check.tmp"
            try:
                test_file.write_text("test")
                test_file.unlink()
                write_ok = True
            except:
                write_ok = False
            
            status = HealthStatus.HEALTHY
            if total_size_mb > 1000:  # Plus de 1GB
                status = HealthStatus.WARNING
            if not write_ok:
                status = HealthStatus.CRITICAL
            
            return {
                'status': status,
                'total_size_mb': total_size_mb,
                'write_permissions': write_ok,
                'log_files_count': len(list(log_dir.rglob('*.log')))
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL,
                'error': str(e)
            }
    
    def _create_alert(self, level: AlertLevel, component: str, message: str, 
                     details: Dict[str, Any]) -> None:
        """Crée une alerte"""
        alert_id = f"{component}_{level.value}"
        
        # Éviter les doublons
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            return
        
        alert = HealthAlert(
            alert_id=alert_id,
            level=level,
            component=component,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
        
        self.alerts[alert_id] = alert
        self._notify_alert_callbacks(alert)
        
        logger.log(
            logging.CRITICAL if level == AlertLevel.CRITICAL else
            logging.ERROR if level == AlertLevel.ERROR else
            logging.WARNING,
            f"Health alert: {message}"
        )
    
    def _resolve_alert(self, component: str) -> None:
        """Résout les alertes d'un composant"""
        to_resolve = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.component == component and not alert.resolved
        ]
        
        for alert_id in to_resolve:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            
            # Déplacer vers l'historique
            self.alert_history.append(alert)
            del self.alerts[alert_id]
            
            logger.info(f"Health alert resolved: {alert.message}")
    
    def _cleanup_old_alerts(self) -> None:
        """Nettoie les anciennes alertes"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Nettoyer l'historique
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Récupère les métriques actuelles"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, hours: int = 1) -> List[SystemMetrics]:
        """Récupère l'historique des métriques"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp > cutoff_time
        ]
    
    def get_active_alerts(self) -> List[HealthAlert]:
        """Récupère les alertes actives"""
        return list(self.alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[HealthAlert]:
        """Récupère l'historique des alertes"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Récupère un résumé de la santé système"""
        current_metrics = self.get_current_metrics()
        active_alerts = self.get_active_alerts()
        
        # Déterminer le statut global
        if any(alert.level == AlertLevel.CRITICAL for alert in active_alerts):
            overall_status = HealthStatus.CRITICAL
        elif any(alert.level in [AlertLevel.ERROR, AlertLevel.WARNING] for alert in active_alerts):
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Statistiques des composants
        component_status = {}
        for name, checker in self.component_checkers.items():
            component_status[name] = {
                'status': checker.last_status.value if checker.last_status else 'unknown',
                'last_check': checker.last_check.isoformat() if checker.last_check else None,
                'details': checker.last_details
            }
        
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics.to_dict() if current_metrics else None,
            'active_alerts_count': len(active_alerts),
            'critical_alerts_count': len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
            'component_status': component_status,
            'uptime_hours': self._get_uptime_hours(),
            'monitoring_active': not self.stop_monitoring
        }
    
    def _get_uptime_hours(self) -> float:
        """Récupère l'uptime du système en heures"""
        try:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            return uptime.total_seconds() / 3600
        except:
            return 0.0
    
    def add_alert_callback(self, callback: Callable[[HealthAlert], None]) -> None:
        """Ajoute un callback pour les alertes"""
        self.alert_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable[[SystemMetrics], None]) -> None:
        """Ajoute un callback pour les métriques"""
        self.metrics_callbacks.append(callback)
    
    def _notify_alert_callbacks(self, alert: HealthAlert) -> None:
        """Notifie les callbacks d'alerte"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _notify_metrics_callbacks(self, metrics: SystemMetrics) -> None:
        """Notifie les callbacks de métriques"""
        for callback in self.metrics_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")
    
    def save_health_report(self, filepath: str = None) -> str:
        """Sauvegarde un rapport de santé"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"logs/health_report_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_system_health_summary(),
            'recent_metrics': [m.to_dict() for m in self.get_metrics_history(hours=1)],
            'active_alerts': [a.to_dict() for a in self.get_active_alerts()],
            'recent_alert_history': [a.to_dict() for a in self.get_alert_history(hours=6)]
        }
        
        # Créer le répertoire si nécessaire
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Health report saved to: {filepath}")
        return filepath
    
    def shutdown(self) -> None:
        """Arrêt propre du moniteur"""
        logger.info("Shutting down SystemHealthMonitor...")
        
        self.stop_monitoring()
        
        # Sauvegarder un rapport final
        try:
            self.save_health_report()
        except Exception as e:
            logger.error(f"Error saving final health report: {e}")
        
        logger.info("SystemHealthMonitor shutdown completed")