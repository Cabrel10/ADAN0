# Fichiers Clés - ADAN Trading Bot

Ce document répertorie tous les fichiers clés du projet ADAN Trading Bot avec leur statut et leur objectif.

## Structure des Répertoires

```
bot/
├── config/                   # Fichiers de configuration
│   ├── config.yaml          # Configuration principale (monolithique actuelle)
│   ├── model.yaml           # Configuration des modèles (à créer)
│   ├── environment.yaml     # Configuration de l'environnement (à créer)
│   └── trading.yaml         # Paramètres de trading (à créer)
│
├── src/adan_trading_bot/
│   ├── common/              # Composants communs
│   │   ├── __init__.py
│   │   ├── config_loader.py      # Chargeur de configuration de base
│   │   ├── config_validator.py   # Validation de la configuration
│   │   ├── config_watcher.py     # Surveillance des changements
│   │   └── enhanced_config_manager.py  # Gestion avancée (à créer)
│   │
│   ├── trading/             # Logique de trading
│   │   ├── __init__.py
│   │   └── secure_api_manager.py  # Gestion sécurisée des API
│   │
│   └── utils/               # Utilitaires
│       ├── __init__.py
│       ├── timeout_manager.py     # Gestion des timeouts (à créer)
│       └── conda_validator.py     # Validation de l'environnement Conda (à créer)
│
├── tests/                   # Tests automatisés
│   ├── unit/               # Tests unitaires
│   ├── integration/        # Tests d'intégration
│   └── performance/        # Tests de performance
│
├── docs/                   # Documentation
│   ├── api/                # Documentation de l'API
│   ├── architecture/       # Diagrammes d'architecture
│   └── guides/             # Guides utilisateur
│
└── .env.example            # Exemple de variables d'environnement
```

## Détails des Fichiers

### Fichiers de Configuration

| Fichier | Statut | Description | Tâches Associées |
|---------|--------|-------------|------------------|
| `config/config.yaml` | :white_check_mark: Existant | Configuration monolithique actuelle | 2.2, 7.1 |
| `config/model.yaml` | :construction: À créer | Configuration des modèles d'IA | 2.2, 6.1, 6.2 |
| `config/environment.yaml` | :construction: À créer | Paramètres d'environnement | 2.2, 9.2 |
| `config/trading.yaml` | :construction: À créer | Paramètres de trading | 2.2, 4.1, 4.2 |

### Composants Communs

| Fichier | Statut | Description | Tâches Associées |
|---------|--------|-------------|------------------|
| `common/enhanced_config_manager.py` | :construction: À créer | Gestion avancée de la configuration avec hot-reload | 2.1, 7.1 |

### Infrastructure de Test

| Fichier | Statut | Description | Tâches Associées |
|---------|--------|-------------|------------------|
| `tests/conftest.py` | :construction: À créer | Configuration commune des tests | 10.1 |
| `tests/test_*.py` | :construction: À créer | Fichiers de tests unitaires | 10.1 |
| `tests/integration/` | :construction: À créer | Tests d'intégration | 10.1 |
| `tests/performance/` | :construction: À créer | Tests de performance | 10.1 |

### Documentation

| Fichier | Statut | Description | Tâches Associées |
|---------|--------|-------------|------------------|
| `docs/conf.py` | :construction: À créer | Configuration Sphinx | 10.2 |
| `docs/index.rst` | :construction: À créer | Page d'accueil de la documentation | 10.2 |
| `docs/architecture/` | :construction: À créer | Diagrammes d'architecture | 10.2 |
| `common/config_loader.py` | :white_check_mark: Existant | Chargement de base de la configuration | 2.1, 7.1 |
| `common/config_validator.py` | :white_check_mark: Existant | Validation du schéma de configuration | 2.1, 7.2 |
| `common/config_watcher.py` | :white_check_mark: Existant | Surveillance des changements de configuration | 2.1, 7.1 |

### Sécurité et API

| Fichier | Statut | Description | Tâches Associées |
|---------|--------|-------------|------------------|
| `trading/secure_api_manager.py` | :white_check_mark: Existant | Gestion sécurisée des appels API | 1.1, 1.2, 1.3 |
| `.env.example` | :construction: À créer | Modèle de variables d'environnement | 1.1, 1.2 |

### Utilitaires

| Fichier | Statut | Description | Tâches Associées |
|---------|--------|-------------|------------------|
| `utils/timeout_manager.py` | :construction: À créer | Gestion des timeouts des processus | 9.1, 9.3 |
| `utils/conda_validator.py` | :construction: À créer | Validation de l'environnement Conda | 9.2 |

## Légende des Statuts

- :white_check_mark: Existant - Le fichier existe déjà dans le codebase
- :construction: À créer - Le fichier doit être créé
- :warning: À modifier - Le fichier existe mais nécessite des modifications

## Règles de Gestion des Fichiers

1. **Création de Fichiers**
   - Tous les nouveaux fichiers doivent être documentés dans ce document
   - Suivre la structure de dossiers définie
   - Inclure des docstrings et commentaires

2. **Modification de Fichiers**
   - Mettre à jour ce document si la structure change
   - Documenter les changements majeurs
   - Maintenir la rétrocompatibilité quand c'est possible

3. **Sécurité**
   - Ne jamais commiter de clés API ou de secrets
   - Utiliser les variables d'environnement pour les données sensibles
   - Suivre les bonnes pratiques de sécurité définies dans `requirements.md`
