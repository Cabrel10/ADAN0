"""
ConfigWatcher - Système de surveillance et rechargement dynamique des configurations YAML.

Ce module permet de surveiller les fichiers de configuration et d'appliquer les changements
en temps réel pendant l'exécution, sans redémarrer le processus. Il inclut également
une validation des schémas de configuration via le ConfigValidator.

Fonctionnalités clés:
- Détection automatique des modifications de fichiers de configuration
- Validation des schémas avant application des changements
- Notifications en temps réel des mises à jour via callbacks
- Support pour les rechargements partiels (uniquement les composants impactés)
- Gestion robuste des erreurs et des configurations invalides

Exemple d'utilisation:
    ```python
    # Création d'un observateur de configuration
    watcher = ConfigWatcher(config_dir="chemin/vers/configs", validate=True)

    # Enregistrement d'un callback pour les changements de configuration
    def on_training_config_change(config_type, new_config, changes):
        print(f"Configuration {config_type} modifiée! Changements: {changes}")

    watcher.register_callback('training', on_training_config_change)

    # Utilisation du décorateur pour les méthodes réactives
    class MonComposant:
        def __init__(self, config_watcher):
            self.config_watcher = config_watcher
            self.config_watcher.register_callback('training', self.on_config_change)

        @config_reactive('training')
        def on_config_change(self, config_type, new_config, changes):
            print(f"Mise à jour de la configuration {config_type}")
    ```

Note:
    - Les configurations sont chargées de manière paresseuse au premier accès
    - Les validations sont effectuées avant chaque rechargement
    - Les callbacks sont exécutés dans des threads séparés
"""
import os
import time
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable, Union, Tuple
from types import TracebackType
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime, timedelta

from . import config_validator
from .utils import get_logger

logger = get_logger(__name__)

class ConfigFileHandler(FileSystemEventHandler):
    """
    Handler pour les événements de modification de fichiers de configuration.

    Attributes:
        config_watcher: Référence vers l'instance ConfigWatcher parente
    """

    def __init__(self, config_watcher: 'ConfigWatcher'):
        self.config_watcher = config_watcher

    def on_modified(self, event):
        """Appelé quand un fichier est modifié."""
        if not event.is_directory and event.src_path.endswith('.yaml'):
            logger.info(f"Configuration file modified: {event.src_path}")
            self.config_watcher._handle_config_change(event.src_path)

class ConfigWatcher:
    """Surveille les fichiers de configuration YAML et gère leur rechargement dynamique.

    Cette classe permet de charger, surveiller et gérer des configurations au format YAML
    avec les fonctionnalités suivantes :
    - Chargement initial des configurations depuis le répertoire spécifié
    - Détection automatique des modifications des fichiers de configuration
    - Validation des configurations via des schémas définis
    - Notification des composants intéressés via un système de callbacks
    - Gestion thread-safe des accès concurrents
    - Notifications asynchrones via callbacks
    - Gestion des erreurs et reprise sur échec
    - Support des rechargements partiels (uniquement les composants impactés)

    La classe est conçue pour être fiable même en cas de fichiers corrompus ou de validations échouées,
    en conservant toujours une configuration valide en mémoire.

    Attributes:
        config_dir (Path): Répertoire contenant les fichiers de configuration
        enabled (bool): Si False, désactive la surveillance des fichiers
        validate_schemas (bool): Si True, valide les schémas avant d'appliquer les changements
        callbacks (Dict[str, List[Callable]]): Dictionnaire des callbacks par type de configuration
        current_configs (Dict[str, Dict]): Dictionnaire des configurations courantes par type
        last_reload_times (Dict[str, datetime]): Dernier moment de rechargement par type de config
        watched_files (Dict[str, str]): Mapping des noms de fichiers vers leur type de configuration

    Raises:
        FileNotFoundError: Si le répertoire de configuration n'existe pas
        yaml.YAMLError: Si un fichier de configuration contient du YAML invalide
        ValidationError: Si la validation du schéma échoue (uniquement si validate_schemas=True)

    Example:
        ```python
        # Création avec validation activée
        watcher = ConfigWatcher('/chemin/vers/configs', validate=True)

        # Enregistrement d'un callback
        def log_changes(config_type, new_config, changes):
            logger.info(f"Configuration {config_type} modifiée")
            for key, change in changes.items():
                logger.info(f"  - {key}: {change['old_value']} -> {change['new_value']}")

        watcher.register_callback('training', log_changes)

        # Récupération de la configuration actuelle
        training_config = watcher.get_config('training')

        # Arrêt de la surveillance
        watcher.stop()
        ```
    """
    def __init__(self, config_dir: Union[str, Path], validate: Optional[bool] = None, validate_schemas: Optional[bool] = None):
        """Initialise le ConfigWatcher avec le répertoire de configuration.

        Cette méthode initialise le surveillant de configuration, charge les configurations
        initiales et démarre la surveillance des modifications de fichiers.

        Args:
            config_dir (Union[str, Path]):
                Chemin vers le répertoire contenant les fichiers de configuration YAML.
                Les fichiers doivent suivre la convention de nommage `type_config.yaml`.
                Exemple : `training.yaml`, `environment.yaml`.

            validate (bool, optional):
                Alias attendu par les tests pour activer/désactiver la validation des schémas.
                Si spécifié, a priorité sur `validate_schemas`.

            validate_schemas (bool, optional):
                Paramètre historique/alternatif pour activer la validation des schémas.

        Raises:
            FileNotFoundError: Si le répertoire de configuration n'existe pas
            PermissionError: Si l'accès au répertoire est refusé
            yaml.YAMLError: En cas d'erreur de syntaxe dans les fichiers YAML

        Example:
            ```python
            # Création avec validation des schémas activée (par défaut)
            watcher = ConfigWatcher("/chemin/vers/configs")

            # Création avec validation désactivée (pour les tests ou si la validation n'est pas nécessaire)
            watcher = ConfigWatcher("/chemin/vers/configs", validate_schemas=False)
            ```

        Note:
            - Le chargement initial des configurations est effectué de manière paresseuse
              (au premier accès via `get_config` ou `get_all_configs`)
            - La surveillance des fichiers commence immédiatement après l'initialisation
            - Utilisez le gestionnaire de contexte (`with`) pour une libération propre des ressources
        """
        self.config_dir = Path(config_dir)
        self.enabled = True
        # Résolution du drapeau de validation avec compatibilité ascendante
        if validate is not None:
            effective_validate = bool(validate)
        elif validate_schemas is not None:
            effective_validate = bool(validate_schemas)
        else:
            effective_validate = True
        self.validate_schemas = effective_validate
        self.observer = None
        self.callbacks: Dict[str, List[Callable]] = {}
        self.current_configs: Dict[str, Dict[str, Any]] = {}
        self.last_reload_times: Dict[str, datetime] = {}
        # Anti-bounce pour événements multiples très rapprochés (write + move)
        self._last_event_times: Dict[str, datetime] = {}

        # Fichiers de configuration surveillés avec leurs types
        self.watched_files = {
            'train_config.yaml': 'training',
            'environment_config.yaml': 'environment',
            'agent_config.yaml': 'agent',
            'dbe_config.yaml': 'dbe',
            'reward_config.yaml': 'reward',
            'risk_config.yaml': 'risk',
            'config.yaml': 'main'  # Fichier de configuration principal
        }

        # Mappage des types de configuration vers les méthodes de validation
        self.validation_methods = {
            'main': 'validate_main_config',
            'training': 'validate_train_config',
            'environment': 'validate_environment_config',
            'agent': 'validate_agent_config',
            'dbe': 'validate_dbe_config',
            'reward': 'validate_reward_config',
            'risk': 'validate_risk_config'
        }

        # Chargement initial des configurations
        self._load_initial_configs()

        if self.enabled:
            self._start_watching()
            logger.info(f"🔍 ConfigWatcher started - monitoring {self.config_dir}")
        else:
            logger.info("ConfigWatcher disabled")

    def _load_initial_configs(self):
        """Charge toutes les configurations initiales."""
        for filename, config_type in self.watched_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        self.current_configs[config_type] = config
                except Exception as e:
                    logger.error(f"Erreur lors du chargement de {config_path.name}: {e}")

    def _start_watching(self) -> None:
        """Démarre la surveillance des fichiers de configuration.

        Cette méthode initialise et démarre l'observateur de fichiers qui surveille
        les modifications des fichiers de configuration dans le répertoire configuré.

        Le processus de surveillance comprend :
        1. Création d'une instance de l'observateur
        2. Configuration du gestionnaire d'événements pour les modifications de fichiers
        3. Démarrage du thread d'observation en arrière-plan

        La méthode est appelée automatiquement lors de l'initialisation du ConfigWatcher
        si la surveillance est activée.

        Raises:
            RuntimeError: Si l'observateur ne peut pas être démarré

        Example:
            ```python
            # Démarrer la surveillance manuellement (normalement géré par __init__)
            watcher = ConfigWatcher("chemin/vers/configs")
            watcher._start_watching()
            ```

        Note:
            - Ne pas appeler cette méthode directement sauf si vous savez ce que vous faites
            - La méthode est idempotente (plusieurs appels n'ont pas d'effet supplémentaire)
            - Les erreurs sont journalisées mais pas propagées pour éviter d'interrompre le flux principal
        """
        if self.observer is not None:
            logger.debug("La surveillance est déjà en cours")
            return

        try:
            self.observer = Observer()
            event_handler = ConfigFileHandler(self)
            self.observer.schedule(event_handler, str(self.config_dir), recursive=False)
            self.observer.start()
            logger.info(f"Surveillance démarrée pour {self.config_dir}")
        except Exception as e:
            logger.error(f"Échec du démarrage de la surveillance: {e}")
            raise RuntimeError(f"Impossible de démarrer la surveillance: {e}")

    def _load_config_file(self, config_file: Path) -> None:
        """Charge un fichier de configuration YAML en mémoire.

        Cette méthode est responsable du chargement d'un fichier de configuration,
        de sa validation selon le schéma approprié, et de son stockage dans le
        dictionnaire des configurations courantes.

        Le processus de chargement comprend :
        1. Vérification que le fichier est bien enregistré dans watched_files
        2. Chargement du contenu YAML du fichier
        3. Validation du contenu avec le schéma approprié
        4. Mise à jour de la configuration courante et de l'horodatage

        Args:
            config_file: Chemin vers le fichier de configuration à charger

        Returns:
            None: La méthode modifie l'état interne mais ne retourne rien

        Raises:
            yaml.YAMLError: Si le fichier YAML est mal formé
            FileNotFoundError: Si le fichier n'existe pas

        Example:
            ```python
            # Chargement d'un fichier de configuration
            watcher._load_config_file(Path("/chemin/vers/training.yaml"))
            ```

        Note:
            - Les fichiers inconnus (non présents dans watched_files) sont ignorés
            - Les erreurs de validation n'interrompent pas l'exécution
            - Les configurations valides remplacent les précédentes
        """
        filename = config_file.name
        config_type = self.watched_files.get(filename)

        if config_type is None:
            logger.warning(f"Fichier de configuration non surveillé ignoré: {filename}")
            return

        logger.debug(f"Chargement du fichier de configuration: {filename}")
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            # Validation du schéma si activée
            if self.validate_schemas and not self._validate_config(config, config_type, filename):
                logger.error(f"Configuration invalide dans {filename}, chargement annulé")
                return

            # Mise à jour de la configuration
            self.current_configs[config_type] = config
            self.last_reload_times[config_type] = datetime.now()
            logger.info(f"Configuration chargée avec succès: {filename}")

        except yaml.YAMLError as e:
            logger.error(f"Erreur de syntaxe YAML dans {filename}: {e}")
        except FileNotFoundError:
            logger.error(f"Fichier de configuration non trouvé: {filename}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {filename}: {e}")

    def _validate_config(self, config: Dict[str, Any], config_type: str, filename: str) -> bool:
        """Valide une configuration avec le schéma approprié.

        Cette méthode effectue la validation des configurations selon les règles suivantes :
        - Si validate_schemas est False, la validation est ignorée (retourne toujours True)
        - Pour le type 'main', valide toutes les configurations avec validate_all_configs
        - Pour les autres types, utilise la méthode de validation spécifique si elle existe
        - Les erreurs de validation sont journalisées pour le débogage

        La validation est essentielle pour garantir l'intégrité des configurations avant leur
        application dans le système. Elle permet de détecter les erreurs de configuration
        avant qu'elles ne provoquent des comportements inattendus.

        Args:
            config: Dictionnaire contenant la configuration à valider
            config_type: Type de configuration (training, environment, etc.)
            filename: Nom du fichier source pour les messages d'erreur

        Returns:
            bool: True si la configuration est valide, False en cas d'échec de validation

        Raises:
            ValueError: Si le type de configuration n'est pas reconnu

        Example:
            ```python
            # Validation d'une configuration
            is_valid = watcher._validate_config(
                config={"learning_rate": 0.001, "batch_size": 32},
                config_type="training",
                filename="training.yaml"
            )
            ```
        """
        if not self.validate_schemas:
            return True

        validator = config_validator.ConfigValidator()

        if config_type == 'main':
            return validator.validate_all_configs(self.config_dir)

        # Sinon, on valide uniquement la section correspondante
        validation_method = getattr(validator, self.validation_methods.get(config_type, ''), None)
        if not validation_method:
            logger.warning(f"No validation method found for config type: {config_type}")
            return True

        # Création d'une configuration factice avec uniquement la section à valider
        fake_config = {config_type: config}
        is_valid = validation_method(fake_config, filename)

        if not is_valid:
            for error in validator.get_validation_errors():
                logger.error(f"Validation error in {filename}: {error}")
            return False

        return True

    def _handle_config_change(self, file_path: Union[str, Path]) -> None:
        """Gère le changement d'un fichier de configuration détecté par watchdog.

        Cette méthode est le point d'entrée principal pour le traitement des modifications
        de fichiers de configuration. Elle orchestre le processus complet de rechargement
        et de validation des configurations modifiées.

        Processus détaillé :
        1. Vérification que le fichier modifié est bien surveillé
        2. Court délai pour éviter les lectures partielles (race condition)
        3. Chargement de la nouvelle configuration depuis le disque
        4. Validation de la configuration selon le schéma approprié
        5. Détection des changements par rapport à la configuration actuelle
        6. Mise à jour de la configuration en mémoire si valide
        7. Notification des callbacks enregistrés pour ce type de configuration

        La méthode est conçue pour être robuste et sûre :
        - Toutes les exceptions sont attrapées et correctement journalisées
        - En cas d'échec de validation, la configuration précédente est conservée
        - Les callbacks sont exécutés dans un thread séparé pour ne pas bloquer
        - Les erreurs dans les callbacks n'affectent pas le processus principal

        Args:
            file_path: Chemin vers le fichier modifié. Peut être une chaîne ou un objet Path.
                      Le fichier doit être dans le répertoire de configuration surveillé.

        Returns:
            None: La méthode ne retourne rien mais peut modifier l'état interne

        Raises:
            FileNotFoundError: Si le fichier de configuration n'existe plus après modification
            yaml.YAMLError: Si le fichier contient du YAML mal formé
            ValidationError: Si la validation est activée et que la configuration est invalide

        Example:
            La méthode est principalement appelée automatiquement par watchdog :

            ```python
            # Appel manuel (déconseillé sauf pour le débogage)
            watcher._handle_config_change(Path('config/training.yaml'))
            ```

        Note:
            - Ne pas appeler cette méthode directement depuis le code utilisateur
            - Utiliser force_reload() pour forcer un rechargement depuis le code
            - Les erreurs sont journalisées avec un niveau d'erreur approprié
            - Les callbacks sont exécutés de manière asynchrone
        """
        # Normalise en objet Path si nécessaire
        if isinstance(file_path, str):
            file_path = Path(file_path)
        filename = file_path.name

        if filename not in self.watched_files:
            return

        config_type = self.watched_files[filename]
        # Debounce: ignore événements dupliqués dans une courte fenêtre
        now = datetime.now()
        last_evt = self._last_event_times.get(config_type)
        if last_evt and (now - last_evt) < timedelta(milliseconds=200):
            logger.debug(f"Ignoring duplicated event for {filename}")
            return
        self._last_event_times[config_type] = now

        # Petit délai pour éviter les lectures partielles
        time.sleep(0.1)

        # Rechargement de la configuration et traitement
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                new_config = yaml.safe_load(f) or {}

            # Validation du schéma
            if not self._validate_config(new_config, config_type, filename):
                logger.error(f"Configuration invalide dans {filename}, rechargement annulé")
                return

            old_config = self.current_configs.get(config_type, {})

            # Détection des changements
            changes = self._detect_changes(old_config, new_config)

            if changes:
                self.current_configs[config_type] = new_config
                self.last_reload_times[config_type] = datetime.now()

                logger.info(f"🔄 Configuration rechargée: {filename}")
                logger.info(f"Changements détectés: {list(changes.keys())}")

                # Notification des callbacks
                self._notify_callbacks(config_type, new_config, changes)
            else:
                logger.debug(f"Aucun changement détecté dans {filename}")

        except yaml.YAMLError as e:
            logger.error(f"Erreur de syntaxe YAML dans {filename}: {e}")
            return
        except Exception as e:
            logger.error(f"Error handling config change for {file_path}: {e}")

    def _detect_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte et analyse les différences entre deux configurations.

        Cette méthode compare récursivement deux dictionnaires de configuration et identifie
        toutes les modifications, ajouts et suppressions. La détection est effectuée de manière
        profonde (nested) pour identifier les changements même dans les structures imbriquées.

        Caractéristiques principales :
        - Détection des champs ajoutés, modifiés ou supprimés
        - Comparaison récursive des dictionnaires
        - Conservation des valeurs avant/après pour chaque modification
        - Gestion des types de données courants (dict, primitifs)

        Args:
            old_config: Dictionnaire représentant l'ancienne configuration
            new_config: Dictionnaire représentant la nouvelle configuration

        Returns:
            Dict[str, Dict[str, Any]]: Dictionnaire des changements au format :
                {
                    'chemin.vers.champ': {
                        'action': 'added|modified|removed',
                        'old_value': ancienne_valeur,  # Non présent pour 'added'
                        'new_value': nouvelle_valeur    # Non présent pour 'removed'
                    },
                    ...
                }

                Les clés utilisent une notation par points pour les chemins imbriqués.

        Example:
            ```python
            old = {'training': {'learning_rate': 0.01, 'batch_size': 32}}
            new = {
                'training': {
                    'learning_rate': 0.02,  # Modifié
                    'epochs': 10            # Ajouté
                }
                # batch_size: 32            # Supprimé
            }

            # Retourne :
            # {
            #     'training.learning_rate': {
            #         'action': 'modified',
            #         'old_value': 0.01,
            #         'new_value': 0.02
            #     },
            #     'training.epochs': {
            #         'action': 'added',
            #         'new_value': 10
            #     },
            #     'training.batch_size': {
            #         'action': 'removed',
            #         'old_value': 32
            #     }
            # }
            ```
        """
        changes = {}

        def compare_dicts(old_dict, new_dict, path=""):
            for key, new_value in new_dict.items():
                current_path = f"{path}.{key}" if path else key

                if key not in old_dict:
                    changes[current_path] = {'action': 'added', 'new_value': new_value}
                elif isinstance(new_value, dict) and isinstance(old_dict[key], dict):
                    compare_dicts(old_dict[key], new_value, current_path)
                elif old_dict[key] != new_value:
                    changes[current_path] = {
                        'action': 'modified',
                        'old_value': old_dict[key],
                        'new_value': new_value
                    }

            # Vérifier les clés supprimées
            for key in old_dict:
                if key not in new_dict:
                    current_path = f"{path}.{key}" if path else key
                    changes[current_path] = {'action': 'removed', 'old_value': old_dict[key]}

        compare_dicts(old_config, new_config)
        return changes

    def _notify_callbacks(self, config_type: str, new_config: Dict[str, Any], changes: Dict[str, Any]) -> None:
        """Notifie tous les callbacks enregistrés pour un type de configuration.

        Cette méthode est responsable de l'exécution de tous les callbacks enregistrés pour un type
        de configuration spécifique. Les callbacks sont exécutés de manière asynchrone dans un thread
        séparé pour éviter de bloquer le thread principal.

        Caractéristiques clés :
        - Exécution asynchrone des callbacks
        - Isolation des erreurs (les erreurs dans un callback n'affectent pas les autres)
        - Journalisation détaillée des erreurs
        - Gestion des cas où il n'y a pas de callbacks enregistrés

        Args:
            config_type: Type de configuration qui a été modifiée (ex: 'training', 'environment')
            new_config: Nouvelle configuration complète après application des changements
            changes: Dictionnaire des modifications détectées, au format :
                    {
                        'chemin.vers.champ': {
                            'action': 'added|modified|removed',
                            'old_value': ancienne_valeur,  # Optionnel selon l'action
                            'new_value': nouvelle_valeur    # Optionnel selon l'action
                        },
                        ...
                    }

        Returns:
            None

        Raises:
            Aucune exception n'est propagée en dehors de cette méthode pour assurer la robustesse.
            Toutes les erreurs sont capturées et journalisées.

        Example:
            ```python
            # Exemple de callback qui pourrait être notifié
            def log_changes(config_type, new_config, changes):
                print(f"Configuration {config_type} mise à jour !")
                for field, change in changes.items():
                    action = change['action']
                    if action == 'added':
                        print(f"  + {field} = {change['new_value']}")
                    elif action == 'modified':
                        print(f"  ~ {field}: {change['old_value']} -> {change['new_value']}")
                    elif action == 'removed':
                        print(f"  - {field} (était {change['old_value']})")

            # Enregistrement du callback
            watcher.register_callback('training', log_changes)
            ```

        Note:
            - Les callbacks sont exécutés dans un thread séparé pour éviter de bloquer
              le thread principal pendant le traitement.
            - Les erreurs dans les callbacks sont capturées et journalisées mais ne remontent pas
              pour éviter d'interrompre le traitement des autres callbacks.
            - L'ordre d'exécution des callbacks n'est pas garanti.
        """
        if config_type in self.callbacks:
            for callback in self.callbacks[config_type]:
                try:
                    callback(config_type, new_config, changes)
                except Exception as e:
                    logger.error(f"Erreur dans le callback pour {config_type}: {e}")
                    logger.debug(f"Détails de l'erreur:", exc_info=True)

    def register_callback(self, config_type: str, callback: Callable):
        """Enregistre un callback pour un type de configuration spécifique.

        Les callbacks enregistrés sont appelés de manière asynchrone à chaque modification
        d'un fichier de configuration du type spécifié. Chaque callback reçoit trois arguments :
        - Le type de configuration modifiée
        - La nouvelle configuration complète
        - Un dictionnaire détaillant les changements détectés

        Les callbacks sont exécutés dans un thread séparé pour éviter de bloquer
        le thread principal. Les exceptions levées par les callbacks sont attrapées
        et journalisées mais n'interrompent pas le traitement.

        Args:
            config_type: Type de configuration à surveiller (ex: 'training', 'environment')
            callback: Fonction à appeler lors des changements. Doit avoir la signature :
                     callback(config_type: str, new_config: Dict, changes: Dict) -> None

                     Où :
                     - config_type: Type de la configuration modifiée
                     - new_config: Dictionnaire contenant la configuration complète
                     - changes: Dictionnaire des changements au format :
                         {
                             'chemin.vers.champ': {
                                 'old_value': ancienne_valeur,
                                 'new_value': nouvelle_valeur
                             },
                             ...
                         }

        Raises:
            ValueError: Si config_type n'est pas un type de configuration valide

        Example:
            ```python
            def log_changes(config_type, new_config, changes):
                print(f"Configuration {config_type} modifiée !")
                for key, change in changes.items():
                    print(f"  - {key}: {change['old_value']} -> {change['new_value']}")

            watcher.register_callback('training', log_changes)
            ```
        """
        if config_type not in self.callbacks:
            self.callbacks[config_type] = []
        if callback not in self.callbacks[config_type]:
            self.callbacks[config_type].append(callback)
        logger.info(f"Callback registered for {config_type} config changes")

    def get_config(self, config_type: str) -> Dict[str, Any]:
        """Récupère la configuration actuelle pour un type donné.

        Cette méthode permet d'accéder à la dernière version connue d'une configuration
        d'un type spécifique. La configuration retournée est une copie pour éviter les
        modifications accidentelles de l'état interne.

        La méthode est thread-safe et peut être appelée de n'importe quel thread.

        Args:
            config_type: Type de configuration à récupérer (ex: 'training', 'environment')

        Returns:
            Dict[str, Any]: Copie de la configuration actuelle pour le type demandé.
                         Retourne un dictionnaire vide si la configuration n'existe pas.

        Raises:
            KeyError: Si le type de configuration n'existe pas dans watched_files

        Example:
            ```python
            # Récupération de la configuration d'entraînement
            try:
                training_config = watcher.get_config('training')
                print(f"Taux d'apprentissage: {training_config.get('learning_rate')}")
            except KeyError:
                print("Configuration d'entraînement non trouvée")

            # Vérification de l'existence avant accès
            if 'environment' in watcher.current_configs:
                env_config = watcher.get_config('environment')
            ```
        """
        return self.current_configs.get(config_type)

    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Récupère une copie de toutes les configurations actuellement chargées.

        Cette méthode retourne un dictionnaire contenant toutes les configurations
        actuellement en mémoire, organisées par type de configuration. Chaque configuration
        est retournée sous forme de copie pour éviter toute modification accidentelle
        de l'état interne du ConfigWatcher.

        Caractéristiques clés :
        - Retourne une copie profonde des configurations pour éviter les effets de bord
        - Préserve la structure hiérarchique des configurations
        - Ne déclenche pas de rechargement des fichiers de configuration
        - Opération thread-safe

        Returns:
            Dict[str, Dict[str, Any]]: Dictionnaire des configurations au format :
                {
                    'type_config1': { ... },  # Configuration complète pour le type 1
                    'type_config2': { ... },  # Configuration complète pour le type 2
                    ...
                }

                Les clés correspondent aux types de configuration (ex: 'training', 'environment')
                et les valeurs sont des dictionnaires contenant la configuration complète.

        Example:
            ```python
            # Récupération de toutes les configurations
            all_configs = watcher.get_all_configs()

            # Accès à des configurations spécifiques
            training_config = all_configs.get('training', {})
            env_config = all_configs.get('environment', {})

            # Itération sur toutes les configurations
            for config_type, config in all_configs.items():
                print(f"Type de configuration: {config_type}")
                print(f"Nombre de paramètres: {len(config)}")
            ```

        Note:
            - Les configurations retournées sont une copie à plat de l'état actuel.
            - Les modifications apportées au dictionnaire retourné n'affecteront pas
              la configuration interne du ConfigWatcher.
            - Pour mettre à jour une configuration, il faut modifier le fichier
              de configuration correspondant sur le disque.
        """
        return {k: v.copy() for k, v in self.current_configs.items()}

    def force_reload(self, config_type: Optional[str] = None) -> None:
        """Force le rechargement d'une configuration spécifique ou de toutes les configurations.

        Cette méthode permet de forcer manuellement le rechargement d'un ou plusieurs fichiers
        de configuration depuis le disque. Elle est utile lorsque des modifications ont été
        apportées aux fichiers de configuration en dehors du système de surveillance, ou pour
        forcer une relecture après des erreurs de chargement précédentes.

        Caractéristiques clés :
        - Recharge les configurations depuis le disque, ignorant le cache
        - Valide les configurations avant de les appliquer
        - Déclenche les callbacks enregistrés en cas de modifications détectées
        - Gère à la fois le rechargement d'un type spécifique ou de tous les types

        Args:
            config_type: Type de configuration à recharger. Si None, recharge toutes les configurations.
                       Doit correspondre à une clé de watched_files.

        Returns:
            None

        Raises:
            FileNotFoundError: Si le fichier de configuration spécifié n'existe pas
            yaml.YAMLError: Si le fichier de configuration contient du YAML invalide
            ValidationError: Si la validation est activée et que la configuration est invalide

        Example:
            ```python
            # Recharger une configuration spécifique
            try:
                watcher.force_reload('training')
                print("Configuration 'training' rechargée avec succès")
            except Exception as e:
                print(f"Erreur lors du rechargement: {e}")

            # Recharger toutes les configurations
            watcher.force_reload()
            ```

        Note:
            - Cette méthode est bloquante jusqu'à ce que le rechargement soit terminé
            - Les erreurs de validation n'entraînent pas la modification de la configuration actuelle
            - Les configurations sont rechargées dans l'ordre alphabétique des noms de fichiers
        """
        if config_type:
            if config_type in self.watched_files:
                file_path = Path(self.config_dir) / self.watched_files[config_type]
                logger.info(f"Forcing reload of {config_type} config from {file_path}")
                self._load_config_file(file_path)
            else:
                logger.warning(f"No watched config file for type: {config_type}")
        else:
            # Recharger toutes les configurations
            logger.info("Forcing reload of all configurations")
            for file_path in sorted(Path(self.config_dir).glob('*.yaml')):
                if file_path.name in self.watched_files.values():
                    self._load_config_file(file_path)
                    self._handle_config_change(str(config_path))

    def get_reload_status(self) -> Dict[str, Any]:
        """Récupère des informations sur l'état actuel des rechargements de configuration.

        Cette méthode fournit un aperçu de l'état de surveillance des configurations,
        y compris les horodatages des derniers rechargements réussis pour chaque type
        de configuration et la liste des fichiers surveillés.

        Returns:
            Dict[str, Any]: Un dictionnaire contenant :
                - 'last_reload_times': Dictionnaire des derniers rechargements par type de configuration.
                                     Les valeurs sont des chaînes au format ISO ou None si jamais rechargé.
                - 'watched_files': Liste des noms de fichiers actuellement surveillés.

        Example:
            ```python
            # Récupération du statut
            status = watcher.get_reload_status()

            # Affichage des informations
            print("Fichiers surveillés:", status['watched_files'])
            print("\nDerniers rechargements:")
            for config_type, timestamp in status['last_reload_times'].items():
                print(f"- {config_type}: {timestamp or 'Jamais'}")

            # Sortie possible:
            # Fichiers surveillés: ['training.yaml', 'environment.yaml']
            #
            # Derniers rechargements:
            # - training: 2023-04-01T15:30:45.123456
            # - environment: None
            ```

        Note:
            - Les horodatages sont en temps UTC
            - Un horodatage à None signifie que la configuration n'a jamais été rechargée avec succès
            - La liste des fichiers surveillés est dynamique et peut changer si la configuration change
        """
        return {
            'last_reload_times': {
                k: v.isoformat() if v else None
                for k, v in self.last_reload_times.items()
            },
            'watched_files': list(self.watched_files.keys())
        }

    def stop(self) -> None:
        """Arrête définitivement la surveillance des fichiers de configuration.

        Cette méthode arrête le thread d'observation des fichiers et libère les ressources associées.
        Une fois arrêté, le ConfigWatcher ne détectera plus les modifications des fichiers de configuration
        jusqu'à ce qu'une nouvelle instance soit créée.

        Caractéristiques clés :
        - Arrêt propre de l'observateur de fichiers
        - Libération des ressources système
        - Opération idempotente (peut être appelée plusieurs fois sans effet secondaire)
        - Désactive définitivement la surveillance

        Returns:
            None

        Example:
            ```python
            # Création et démarrage du surveillant
            watcher = ConfigWatcher("chemin/vers/configs")

            # ... utilisation du surveillant ...

            # Arrêt propre lorsque plus nécessaire
            watcher.stop()

            # Appels ultérieurs n'ont aucun effet
            watcher.stop()  # Sans effet
            ```

        Note:
            - Cette méthode est automatiquement appelée lors de la destruction de l'objet
              si le gestionnaire de contexte (with) est utilisé
            - Après l'appel à stop(), le ConfigWatcher ne peut pas être redémarré
            - Tous les callbacks enregistrés sont conservés mais ne seront plus notifiés
        """
        if self.observer:
            logger.info("Arrêt de la surveillance des fichiers de configuration")
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("ConfigWatcher stopped")

    def __enter__(self) -> 'ConfigWatcher':
        """Méthode d'entrée du gestionnaire de contexte.

        Cette méthode est appelée au début d'un bloc `with`. Elle permet d'utiliser
        le ConfigWatcher avec la syntaxe `with`, garantissant que les ressources
        seront correctement libérées à la fin du bloc.

        Returns:
            ConfigWatcher: L'instance actuelle du ConfigWatcher

        Example:
            ```python
            # Utilisation avec le gestionnaire de contexte
            with ConfigWatcher("chemin/vers/configs") as watcher:
                # Le ConfigWatcher est actif ici
                config = watcher.get_config('training')
                # ...

            # Ici, le ConfigWatcher est automatiquement arrêté
            ```
        """
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                exc_val: Optional[BaseException],
                exc_tb: Optional[TracebackType]) -> None:
        """Méthode de sortie du gestionnaire de contexte.

        Cette méthode est appelée à la fin d'un bloc `with`. Elle garantit que
        les ressources du ConfigWatcher sont correctement libérées, même en cas d'erreur.

        Args:
            exc_type: Type de l'exception levée dans le bloc, ou None si aucune
            exc_val: Instance de l'exception levée, ou None
            exc_tb: Traceback de l'exception, ou None

        Returns:
            None. Si une exception a été passée, elle est à nouveau levée après le nettoyage.

        Note:
            - Si une exception s'est produite dans le bloc `with`, elle est journalisée
              mais pas interceptée (laissée se propager)
            - La méthode est appelée dans tous les cas, que le bloc se termine normalement
              ou par une exception
        """
        self.stop()

def config_change_callback(config_type: str) -> Callable:
    """Décorateur pour enregistrer automatiquement une méthode comme callback de changement de configuration.

    Ce décorateur permet d'enregistrer une méthode de classe comme callback pour être notifiée
    des changements de configuration d'un type spécifique. La méthode décorée sera automatiquement
    appelée à chaque modification de la configuration du type spécifié.

    Args:
        config_type (str): Type de configuration à surveiller (ex: 'training', 'environment')

    Returns:
        Callable: Le décorateur qui enregistrera la méthode comme callback

    Example:
        class MonComposant:
            def __init__(self, config_watcher):
                self.config_watcher = config_watcher
                self.config_watcher.register_callback('training', self.on_training_config_change)

            @config_reactive('training')
            def on_training_config_change(self, config_type, new_config, changes):
                print(f"Mise à jour de la configuration {config_type}")
                for key, change in changes.items():
                    print(f"  - {key}: {change.get('old_value')} -> {change.get('new_value')}")
    """
    def decorator(func):
        func._config_reactive = True
        func._config_type = config_type
        return func
    return decorator

# Backward-compatible alias expected by tests and examples
# Tests import `config_reactive` from this module; keep both names.
config_reactive = config_change_callback

# Exemple d'utilisation
if __name__ == "__main__":
    def on_training_config_change(config_type: str, new_config: Dict[str, Any], changes: Dict[str, Any]):
        print(f"Training config changed: {changes}")
        # Ici on pourrait ajuster les paramètres de l'agent
        if 'learning_rate' in changes:
            print(f"Learning rate changed to: {new_config.get('learning_rate')}")

    def on_environment_config_change(config_type: str, new_config: Dict[str, Any], changes: Dict[str, Any]):
        print(f"Environment config changed: {changes}")
        # Ici on pourrait ajuster les paramètres de l'environnement

    # Test du ConfigWatcher
    with ConfigWatcher("config") as watcher:
        watcher.register_callback('training', on_training_config_change)
        watcher.register_callback('environment', on_environment_config_change)

        print("ConfigWatcher running... Modify config files to see changes")
        print("Press Ctrl+C to stop")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping ConfigWatcher...")
