#!/usr/bin/env python3
"""
Système d'alerte intelligent pour le bot de trading ADAN.

Ce module fournit des fonctionnalités de surveillance et d'alerte pour détecter
les problèmes potentiels et les opportunités dans le système de trading.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from threading import Thread, Lock
from queue import Queue

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Niveaux de gravité des alertes."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Canaux de notification disponibles."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"

@dataclass
class AlertRule:
    """Règle d'alerte configurable."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    message_template: str
    level: AlertLevel = AlertLevel.WARNING
    cooldown: int = 300  # secondes entre les alertes identiques
    enabled: bool = True
    last_triggered: Optional[float] = None

    def should_trigger(self, context: Dict[str, Any]) -> bool:
        """Détermine si l'alerte doit être déclenchée."""
        if not self.enabled:
            return False

        # Vérifier le cooldown
        current_time = time.time()
        if (self.last_triggered is not None and
            current_time - self.last_triggered < self.cooldown):
            return False

        # Vérifier la condition
        try:
            if self.condition(context):
                self.last_triggered = current_time
                return True
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de la règle {self.name}: {e}")

        return False

    def format_message(self, context: Dict[str, Any]) -> str:
        """Formate le message d'alerte avec le contexte."""
        try:
            return self.message_template.format(**context)
        except KeyError as e:
            logger.error(f"Erreur de formatage du message pour {self.name}: {e}")
            return f"[ERREUR] Impossible de formater le message pour {self.name}"

@dataclass
class Alert:
    """Représente une alerte générée."""
    id: str
    title: str
    message: str
    level: AlertLevel
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'alerte en dictionnaire."""
        return {
            'id': self.id,
            'title': self.title,
            'message': self.message,
            'level': self.level.value,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'acknowledged': self.acknowledged
        }

    def to_json(self) -> str:
        """Sérialise l'alerte en JSON."""
        return json.dumps(self.to_dict())

class NotificationHandler:
    """Gère l'envoi des notifications via différents canaux."""

    def __init__(self, config: Dict[str, Any]):
        """Initialise le gestionnaire de notifications."""
        self.config = config
        self.email_config = config.get('email', {})
        self.slack_config = config.get('slack', {})
        self.webhook_config = config.get('webhook', {})

    def send_notification(self, alert: Alert, channels: List[AlertChannel]) -> bool:
        """Envoie une notification via les canaux spécifiés."""
        success = True

        for channel in channels:
            try:
                if channel == AlertChannel.EMAIL:
                    self._send_email(alert)
                elif channel == AlertChannel.SLACK:
                    self._send_slack(alert)
                elif channel == AlertChannel.WEBHOOK:
                    self._send_webhook(alert)
                elif channel == AlertChannel.LOG:
                    self._log_alert(alert)
            except Exception as e:
                logger.error(f"Échec de l'envoi de l'alerte via {channel}: {e}")
                success = False

        return success

    def _send_email(self, alert: Alert) -> None:
        """Envoie une alerte par email."""
        if not self.email_config.get('enabled', False):
            return

        msg = MIMEMultipart()
        msg['From'] = self.email_config.get('from')
        msg['To'] = ', '.join(self.email_config.get('to', []))
        msg['Subject'] = f"[{alert.level.upper()}] {alert.title}"

        body = f"""
        Alerte: {alert.title}
        Niveau: {alert.level.upper()}
        Heure: {datetime.fromtimestamp(alert.timestamp).isoformat()}

        Message:
        {alert.message}

        Métadonnées:
        {json.dumps(alert.metadata, indent=2, default=str)}
        """

        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(
            self.email_config.get('smtp_server'),
            self.email_config.get('smtp_port', 587)
        ) as server:
            if self.email_config.get('use_tls', True):
                server.starttls()
            if 'smtp_username' in self.email_config:
                server.login(
                    self.email_config['smtp_username'],
                    self.email_config.get('smtp_password', '')
                )
            server.send_message(msg)

    def _send_slack(self, alert: Alert) -> None:
        """Envoie une alerte sur Slack."""
        if not self.slack_config.get('enabled', False):
            return

        webhook_url = self.slack_config.get('webhook_url')
        if not webhook_url:
            return

        # Déterminer la couleur en fonction du niveau d'alerte
        color = {
            AlertLevel.INFO: "#36a64f",    # Vert
            AlertLevel.WARNING: "#f2c744", # Jaune
            AlertLevel.CRITICAL: "#e01e5a" # Rouge
        }.get(alert.level, "#757575")     # Gris par défaut

        payload = {
            "attachments": [{
                "fallback": f"[{alert.level.upper()}] {alert.title}: {alert.message}",
                "color": color,
                "title": f"{alert.title}",
                "text": alert.message,
                "fields": [
                    {
                        "title": "Niveau",
                        "value": alert.level.upper(),
                        "short": True
                    },
                    {
                        "title": "Date/Heure",
                        "value": datetime.fromtimestamp(alert.timestamp).isoformat(),
                        "short": True
                    }
                ],
                "footer": "ADAN Trading Bot",
                "ts": alert.timestamp
            }]
        }

        # Ajouter les métadonnées si présentes
        if alert.metadata:
            metadata_text = "\n".join(f"• *{k}*: {v}" for k, v in alert.metadata.items())
            payload["attachments"][0]["fields"].append({
                "title": "Détails",
                "value": metadata_text,
                "short": False
            })

        response = requests.post(
            webhook_url,
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()

    def _send_webhook(self, alert: Alert) -> None:
        """Envoie une alerte via un webhook personnalisé."""
        print("\n=== Début de _send_webhook ===")
        print(f"webhook_config: {self.webhook_config}")

        if not self.webhook_config.get('enabled', False):
            print("Webhook non activé dans la configuration")
            return

        webhook_url = self.webhook_config.get('url')
        if not webhook_url:
            print("Aucune URL de webhook configurée")
            return

        print(f"Préparation de la requête pour l'URL: {webhook_url}")

        payload = {
            'alert': alert.to_dict(),
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'adan_trading_bot'
        }

        headers = {'Content-Type': 'application/json'}
        if 'auth_token' in self.webhook_config:
            headers['Authorization'] = f"Bearer {self.webhook_config['auth_token']}"

        print(f"En-têtes de la requête: {headers}")
        print(f"Corps de la requête: {payload}")

        try:
            print("\nAppel à requests.post...")
            response = requests.post(
                webhook_url,
                json=payload,
                headers=headers,
                timeout=10
            )
            print(f"Réponse reçue: {response.status_code}")
            response.raise_for_status()
            print("Requête réussie!")
        except Exception as e:
            print(f"Erreur lors de l'envoi de la requête: {e}")
            raise

        print("=== Fin de _send_webhook ===\n")

    def _log_alert(self, alert: Alert) -> None:
        """Journalise l'alerte."""
        log_message = f"[{alert.level.upper()}] {alert.title}: {alert.message}"
        if alert.metadata:
            log_message += f"\nMétadonnées: {json.dumps(alert.metadata, default=str)}"

        if alert.level == AlertLevel.CRITICAL:
            logger.critical(log_message)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)

class AlertSystem:
    """Système d'alerte intelligent pour le bot de trading."""

    def __init__(self, config: Dict[str, Any]):
        """Initialise le système d'alerte."""
        self.config = config
        self.rules: Dict[str, AlertRule] = {}
        self.alert_history: List[Alert] = []
        self.max_history = config.get('max_history', 1000)
        self.notification_channels = [
            AlertChannel[ch.upper()]
            for ch in config.get('notification_channels', ['log'])
        ]

        # Initialisation du gestionnaire de notifications
        self.notification_handler = NotificationHandler(config)

        # File d'attente pour le traitement asynchrone des alertes
        self.alert_queue = Queue()
        self.processing_thread = Thread(target=self._process_alerts, daemon=True)
        self.processing_thread.start()

        # Verrou pour les opérations thread-safe
        self.lock = Lock()

        # Enregistrer les règles par défaut
        self._register_default_rules()

        logger.info("Système d'alerte initialisé")

    def add_rule(self, rule: AlertRule) -> None:
        """Ajoute une règle d'alerte au système."""
        with self.lock:
            self.rules[rule.name] = rule

    def remove_rule(self, rule_name: str) -> bool:
        """Supprime une règle d'alerte."""
        with self.lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                return True
            return False

    def enable_rule(self, rule_name: str, enabled: bool = True) -> bool:
        """Active ou désactive une règle d'alerte."""
        with self.lock:
            if rule_name in self.rules:
                self.rules[rule_name].enabled = enabled
                return True
            return False

    def evaluate_rules(self, context: Dict[str, Any]) -> List[Alert]:
        """Évalue toutes les règles d'alerte avec le contexte fourni."""
        alerts = []

        # Faire une copie des règles pour éviter les problèmes de concurrence
        with self.lock:
            rules = list(self.rules.values())

        for rule in rules:
            if rule.should_trigger(context):
                alert = Alert(
                    id=f"alert_{int(time.time())}_{len(self.alert_history)}",
                    title=rule.name,
                    message=rule.format_message(context),
                    level=rule.level,
                    metadata={
                        'rule': rule.name,
                        'context': context
                    }
                )
                alerts.append(alert)

                # Ajouter à l'historique
                with self.lock:
                    self.alert_history.append(alert)
                    # Limiter la taille de l'historique
                    if len(self.alert_history) > self.max_history:
                        self.alert_history = self.alert_history[-self.max_history:]

                # Mettre en file d'attente pour le traitement asynchrone
                self.alert_queue.put(alert)

        return alerts

    def _process_alerts(self) -> None:
        """Traite les alertes en arrière-plan."""
        while True:
            try:
                alert = self.alert_queue.get()
                if alert is None:  # Signal d'arrêt
                    break

                # Envoyer les notifications
                self.notification_handler.send_notification(
                    alert,
                    self.notification_channels
                )

                self.alert_queue.task_done()
            except Exception as e:
                logger.error(f"Erreur lors du traitement d'une alerte: {e}")

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        limit: int = 100,
        acknowledged: Optional[bool] = None
    ) -> List[Alert]:
        """Récupère les alertes selon les critères de filtrage."""
        with self.lock:
            alerts = self.alert_history.copy()

        # Filtrer par niveau
        if level is not None:
            alerts = [a for a in alerts if a.level == level]

        # Filtrer par statut d'acquittement
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        # Trier par date (plus récent en premier) et limiter
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        return alerts[:limit]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Marque une alerte comme acquittée."""
        with self.lock:
            for alert in self.alert_history:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    return True
        return False

    def _register_default_rules(self) -> None:
        """Enregistre les règles d'alerte par défaut."""
        # Règle pour la détection de forte baisse de performance
        self.add_rule(AlertRule(
            name="performance_degradation",
            condition=lambda ctx: (
                'performance_metrics' in ctx and
                'sharpe_ratio' in ctx['performance_metrics'] and
                ctx['performance_metrics']['sharpe_ratio'] < -1.0
            ),
            message_template=(
                "Dégradation des performances détectée. "
                "Ratio de Sharpe à {performance_metrics[sharpe_ratio]:.2f} "
                "(seuil: -1.0)"
            ),
            level=AlertLevel.WARNING,
            cooldown=3600  # 1 heure entre les alertes
        ))

        # Règle pour la détection de forte volatilité
        self.add_rule(AlertRule(
            name="high_volatility",
            condition=lambda ctx: (
                'market_metrics' in ctx and
                'volatility' in ctx['market_metrics'] and
                ctx['market_metrics']['volatility'] > 0.1
            ),
            message_template=(
                "Forte volatilité détectée: {market_metrics[volatility]:.2%} "
                "(seuil: 10.0%)"
            ),
            level=AlertLevel.WARNING
        ))

        # Règle pour les erreurs système
        self.add_rule(AlertRule(
            name="system_error",
            condition=lambda ctx: (
                'system_metrics' in ctx and
                'error_count' in ctx['system_metrics'] and
                ctx['system_metrics']['error_count'] > 0
            ),
            message_template=(
                "{system_metrics[error_count]} erreur(s) système détectée(s). "
                "Vérifiez les journaux pour plus de détails."
            ),
            level=AlertLevel.CRITICAL
        ))

    def shutdown(self) -> None:
        """Arrête le système d'alerte de manière propre."""
        # Envoyer un signal d'arrêt au thread de traitement
        self.alert_queue.put(None)
        self.processing_thread.join(timeout=5)
        logger.info("Système d'alerte arrêté")
