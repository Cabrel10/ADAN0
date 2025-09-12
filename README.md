# 🤖 ADAN Trading Bot

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub last commit](https://img.shields.io/github/last-commit/Cabrel10/ADAN)](https://github.com/Cabrel10/ADAN/commits/stable)
[![GitHub stars](https://img.shields.io/github/stars/Cabrel10/ADAN?style=social)](https://github.com/Cabrel10/ADAN/stargazers)

## 📝 Aperçu

ADAN (Adaptive Deep Algorithmic Network) est un système de trading algorithmique avancé conçu pour les marchés de cryptomonnaies. Basé sur des techniques d'apprentissage par renforcement profond (DRL), ADAN permet de développer et déployer des stratégies de trading adaptatives et évolutives avec une gestion avancée des risques.

> **Note** : Ce projet est en développement actif. Consultez la [branche stable](https://github.com/Cabrel10/ADAN/tree/stable) pour la dernière version testée.

## 👥 Contribution

Les contributions sont les bienvenues ! Voici comment contribuer :

1. **Signaler un bug** : Ouvrez une issue en détaillant le problème
2. **Proposer une amélioration** : Créez une issue pour discuter de votre idée
3. **Soumettre du code** :
   - Forkez le dépôt
   - Créez une branche pour votre fonctionnalité (`git checkout -b feature/ma-nouvelle-fonctionnalite`)
   - Committez vos changements (`git commit -am 'Ajout d'une nouvelle fonctionnalité'`)
   - Poussez vers la branche (`git push origin feature/ma-nouvelle-fonctionnalite`)
   - Ouvrez une Pull Request

### 🛠 Normes de code
- Suivez les conventions PEP 8
- Documentez votre code avec des docstrings
- Ajoutez des tests pour les nouvelles fonctionnalités
- Mettez à jour la documentation si nécessaire

## 🛠 Dépannage

### Problèmes courants

#### Erreur d'importation de modules
```bash
# Si vous obtenez des erreurs d'importation, assurez-vous que l'environnement est activé
conda activate trading_env

# Et que le package est installé en mode développement
pip install -e .
```

#### Problèmes de performances
- Réduisez le nombre de workers dans `config.yaml` si vous manquez de mémoire
- Activez le mode silencieux avec `ADAN_QUIET_AFTER_INIT=1`
- Désactivez l'affichage du tableau Rich avec `ADAN_RICH_STEP_TABLE=0`

#### Problèmes de données
- Vérifiez que les fichiers de données sont dans le bon format
- Assurez-vous d'avoir les permissions nécessaires sur le dossier `data/`

## 🚀 Fonctionnalités clés

### 🎯 Trading intelligent
- **Apprentissage par renforcement** : Implémentation d'algorithmes DRL avancés (PPO, SAC)
- **Gestion dynamique des risques** : Adaptation en temps réel aux conditions de marché
- **Trading multi-actifs** : Support natif pour BTC, ETH, SOL, XRP, ADA et plus
- **Multi-timeframes** : Analyse simultanée sur 5m, 1h et 4h

### ⚙️ Infrastructure
- **Entraînement parallèle** : Distribution sur plusieurs workers pour une formation rapide
- **Système de logs avancé** : Suivi détaillé avec Rich pour une meilleure visibilité
- **Gestion de la mémoire** : Chargement efficace des données par chunks
- **Sauvegarde automatique** : Checkpoints réguliers pour éviter les pertes de données

### ⚙️ Composants clés
- **StateBuilder** : Construction robuste d'observations multi-actifs et multi-timeframes
- **DataLoader** : Chargement efficace des données avec gestion de la mémoire
- **SharedExperienceBuffer** : Mémoire de rejeu d'expériences priorisées
- **TrainingOrchestrator** : Orchestrateur de l'entraînement distribué
- **Environnements de trading** : Simulation de marché pour le backtesting avec gestion automatique des réinitialisations
- **API d'échange** : Connecteurs pour différentes plateformes de trading

## 🛠 Installation

### Prérequis
- Python 3.10 ou supérieur
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou [Anaconda](https://www.anaconda.com/)
- [Git](https://git-scm.com/)
- [uv](https://github.com/astral-sh/uv) (installé via `pip install uv`)

### Installation

1. **Cloner le dépôt** :
```bash
git clone https://github.com/Cabrel10/ADAN.git
cd ADAN
git checkout stable  # Basculer sur la branche stable
```

2. **Créer et activer l'environnement Conda** :
```bash
conda create -n trading_env python=3.10 -y
conda activate trading_env
```

3. **Installer les dépendances avec uv pip** :
```bash
uv pip install -r requirements.txt
```

4. **Configuration initiale** :
```bash
cp .env.example .env
cp config/config.example.yaml config/config.yaml
```

> **Note** : Modifiez les fichiers `.env` et `config/config.yaml` selon vos besoins avant de lancer le bot.

## ⚙️ Variables d'environnement

Le projet utilise plusieurs variables d'environnement pour contrôler le comportement du système. Voici les principales :

| Variable | Valeur par défaut | Description |
|----------|------------------|-------------|
| `ADAN_QUIET_AFTER_INIT` | `1` | Active le mode silencieux après l'initialisation |
| `ADAN_RICH_STEP_TABLE` | `1` | Active l'affichage du tableau de suivi Rich |
| `ADAN_RICH_STEP_EVERY` | `10` | Fréquence d'affichage du tableau (en pas) |
| `ADAN_JSONL_EVERY` | `100` | Fréquence d'écriture des logs au format JSONL |
| `ADAN_LOG_LEVEL` | `INFO` | Niveau de journalisation (DEBUG, INFO, WARNING, ERROR) |
| `ADAN_DEVICE` | `auto` | Périphérique à utiliser pour l'entraînement (cpu, cuda, auto) |

Exemple d'utilisation :
```bash
# Activer les logs détaillés et l'affichage Rich
ADAN_QUIET_AFTER_INIT=0 ADAN_RICH_STEP_EVERY=5 python train.py

# Désactiver l'affichage Rich pour de meilleures performances
ADAN_RICH_STEP_TABLE=0 python train.py
```

## ⚙️ Configuration

La configuration s'effectue via le fichier `config/config.yaml`. Consultez le [Guide de configuration](docs/configuration_guide.md) pour les options détaillées.

## 🏗 Structure du projet

```
.
├── config/               # Fichiers de configuration
├── data/                 # Données brutes et traitées
│   ├── raw/             # Données brutes
│   └── processed/       # Données prétraitées
├── docs/                # Documentation technique
├── notebooks/           # Notebooks d'analyse et d'expérimentation
├── scripts/             # Scripts utilitaires
├── src/                 # Code source principal
│   └── adan_trading_bot/
│       ├── environment/  # Environnements de trading
│       ├── models/      # Modèles d'IA
│       ├── training/    # Logique d'entraînement
│       ├── utils/       # Utilitaires communs
│       └── __init__.py
└── tests/               # Tests automatisés
    ├── unit/           # Tests unitaires
    └── integration/    # Tests d'intégration
```

## 🚀 Démarrer

### Configuration minimale requise
- Python 3.8+
- 16GB de RAM recommandés
- 10GB d'espace disque pour les données

### Exemple de configuration
```yaml
# config/config.yaml
data:
  assets:
    - BTCUSDT
    - ETHUSDT
    - SOLUSDT
    - XRPUSDT
    - ADAUSDT
  timeframes:
    - 5m
    - 1h
    - 4h
  features:
    5m: [close, volume, rsi, bb_upper, bb_middle, bb_lower]
    1h: [close, volume, rsi, ema_20, ema_50]
    4h: [close, volume, atr, adx]
```

## 🧪 Exécution des tests

Pour exécuter tous les tests :
```bash
pytest
```

Pour exécuter une catégorie spécifique de tests :
```bash
pytest tests/unit/       # Tests unitaires
pytest tests/integration # Tests d'intégration
```

## ⏱️ Timeout d'entraînement et sauvegarde

ADAN supporte une interruption propre de l'entraînement via un timeout configurable, avec sauvegarde automatique d'un checkpoint.

### Utilisation en ligne de commande

```bash
python -m adan_trading_bot.training.trainer --config config/train_config.yaml --timeout 3600
# Interrompt l'entraînement après 3600 secondes (~1h) et sauvegarde models/best/timeout_checkpoint
```

### Utilisation programmatique

Le `TimeoutManager` fournit un context manager et un décorateur:

```python
from adan_trading_bot.utils.timeout_manager import TimeoutManager, TimeoutException

tm = TimeoutManager(timeout=1800)  # 30 min
try:
    with tm.limit():
        # Votre boucle d'entraînement
        agent.learn(total_timesteps=...
                    )
except TimeoutException:
    print("Entraînement interrompu par timeout")
```

Dans `trainer.py`, lorsque `--timeout` est spécifié, l'appel à `agent.learn()` est encapsulé dans `TimeoutManager`. En cas de timeout, un checkpoint est sauvegardé automatiquement dans `best_model_save_path/timeout_checkpoint`.

Notes:
- Sur Linux, l'implémentation utilise `SIGALRM`; sur autres plateformes, un fallback `threading.Timer` est appliqué.
- Le dossier de sauvegarde est créé si nécessaire.

## 🕵️ Surveillance et débogage

### Surveillance des performances

Le projet inclut plusieurs outils pour surveiller les performances et déboguer les problèmes :

1. **TensorBoard** : Visualisation des métriques d'entraînement
   ```bash
   tensorboard --logdir=./tensorboard_logs
   ```

2. **Logs structurés** : Les logs sont disponibles dans le dossier `logs/` au format JSONL pour une analyse facile

3. **Métriques en temps réel** : Les métriques clés sont affichées dans la console via Rich

### Débogage avancé

Pour activer le mode debug et obtenir plus d'informations :

```bash
# Activer les logs détaillés
ADAN_LOG_LEVEL=DEBUG python train.py

# Activer le mode pas à pas (pour le débogage interactif)
# Ajoutez cette ligne dans votre code :
# import pdb; pdb.set_trace()

# Pour inspecter l'état de l'environnement :
# env.render(mode='human')  # Affiche l'état actuel
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Voici comment contribuer :

1. Forkez le projet
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Poussez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## ⚡ Performances et optimisation

### Optimisation du GPU

Pour tirer le meilleur parti de votre matériel :

1. **CUDA** : Assurez-vous d'avoir CUDA installé pour l'accélération GPU
2. **Batch size** : Ajustez la taille des lots dans `config.yaml` en fonction de votre GPU
3. **Préchargement** : Activez le préchargement des données avec `preload: true` dans la configuration

### Optimisation de la mémoire

Pour gérer efficacement la mémoire :

1. **Chargement par chunks** : Activez le chargement par morceaux dans `config.yaml`
2. **Préallocation** : Désactivez la préallocation si nécessaire avec `preallocate_memory: false`
3. **Nettoyage** : Forcez le nettoyage de la mémoire avec `gc.collect()` dans les parties critiques

### Benchmarking

Pour mesurer les performances :

```python
from time import time
import torch

# Profiler le code
with torch.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
    # Votre code ici
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 🚀 Évolutivité et performances avancées

### Architecture évolutive

Le système est conçu pour évoluer avec vos besoins :

1. **Entraînement distribué** :
   - Utilisez plusieurs GPUs avec `DataParallel` ou `DistributedDataParallel`
   - Répartissez la charge sur plusieurs nœuds de calcul

2. **Traitement par lots** :
   - Ajustez la taille des lots pour optimiser l'utilisation du GPU
   - Utilisez le gradient accumulation pour les grands modèles

### Optimisations avancées

1. **Mélange de précision** :
   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()

   with autocast():
       outputs = model(inputs)
       loss = criterion(outputs, targets)

   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

2. **Compilation JIT** :
   ```python
   model = torch.jit.script(model)  # Pour les modèles
   func = torch.jit.script(my_function)  # Pour les fonctions
   ```

3. **Optimisation des données** :
   - Utilisez des datasets optimisés comme `WebDataset` pour les grands ensembles de données
   - Profitez du chargement asynchrone avec `DataLoader`

## 🔒 Sécurité et confidentialité

### Protection des clés API

1. **Ne jamais commiter de clés API** :
   - Utilisez toujours le fichier `.env` pour stocker les clés sensibles
   - Ajoutez `.env` à votre `.gitignore`
   - Utilisez des exemples de configuration pour les clés factices

2. **Permissions des clés** :
   - Créez des clés avec des permissions minimales nécessaires
   - Activez les restrictions d'adresse IP si possible
   - Régénérez régulièrement les clés

### Données sensibles

1. **Chiffrement** :
   - Activez le chiffrement pour les données sensibles
   - Utilisez des bibliothèques comme `cryptography` pour le chiffrement des données au repos

2. **Journalisation sécurisée** :
   - Ne logguez jamais de données sensibles
   - Utilisez des filtres pour masquer les informations sensibles dans les logs

### Bonnes pratiques

1. **Mises à jour de sécurité** :
   - Mettez à jour régulièrement les dépendances
   - Surveillez les vulnérabilités connues

2. **Audit de sécurité** :
   - Effectuez des audits de sécurité réguliers
   - Utilisez des outils comme `bandit` pour détecter les vulnérabilités Python

## 🚀 Déploiement en production

### Préparation pour la production

1. **Optimisation du modèle** :
   - Quantifiez le modèle pour réduire sa taille et améliorer les performances
   - Convertissez le modèle en TorchScript pour une inférence plus rapide
   - Testez les performances avec différents batch sizes

2. **Configuration du serveur** :
   - Utilisez des instances avec GPU pour l'inférence
   - Configurez le scaling automatique en fonction de la charge
   - Mettez en place un système de monitoring

### Options de déploiement

1. **API REST** :
   ```python
   from fastapi import FastAPI
   import torch

   app = FastAPI()
   model = torch.jit.load("model.pt")

   @app.post("/predict")
   async def predict(data: dict):
       with torch.no_grad():
           output = model(data["input"])
       return {"prediction": output.tolist()}
   ```

2. **Service gRPC** :
   - Meilleures performances que REST pour les appels fréquents
   - Support natif du streaming
   - Génération de code pour plusieurs langages

3. **Conteneurisation** :
   - Créez une image Docker pour un déploiement cohérent
   - Utilisez Kubernetes pour l'orchestration
   - Configurez des health checks

### Monitoring et maintenance

1. **Métriques clés** :
   - Latence d'inférence
   - Taux d'utilisation du GPU
   - Taux de requêtes par seconde
   - Taux d'erreur

2. **Alertes** :
   - Configurez des alertes pour les problèmes critiques
   - Surveillez la dérive des données
   - Suivez les performances au fil du temps

## 🛡 Gestion des erreurs et reprise après incident

### Stratégies de gestion des erreurs

1. **Gestion des erreurs d'entraînement** :
   - Sauvegardes automatiques périodiques
   - Reprise à partir du dernier checkpoint en cas d'échec
   - Journalisation détaillée des erreurs

2. **Gestion des erreurs d'inférence** :
   ```python
   try:
       with torch.no_grad():
           predictions = model(inputs)
   except RuntimeError as e:
       logger.error(f"Erreur d'inférence: {e}")
       # Stratégie de repli
       predictions = fallback_strategy(inputs)
   ```

### Reprise après incident

1. **Checkpoints** :
   - Sauvegardez régulièrement l'état du modèle et de l'optimiseur
   - Implémentez une rotation des sauvegardes

2. **Journalisation** :
   - Enregistrez les métriques clés à intervalles réguliers
   - Conservez les journaux d'erreur pour analyse ultérieure

3. **Auto-réparation** :
   - Détectez les états incohérents
   - Redémarrez les composants défaillants
   - Notifiez les administrateurs en cas de problème critique

### Tests de résistance

1. **Tests de charge** :
   - Simulez des pics de charge
   - Mesurez les performances sous contrainte

2. **Tests de résilience** :
   - Simulez des pannes de composants
   - Vérifiez la reprise après erreur

## 🔄 Intégration et déploiement continus (CI/CD)

### Configuration de base avec GitHub Actions

1. **Tests automatisés** :
   ```yaml
   # .github/workflows/tests.yml
   name: Tests

   on: [push, pull_request]

   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - name: Set up Python
           uses: actions/setup-python@v2
           with:
             python-version: '3.10'
         - name: Install dependencies
           run: |
             python -m pip install --upgrade pip
             pip install -r requirements.txt
             pip install pytest pytest-cov
         - name: Run tests
           run: |
             pytest --cov=./ --cov-report=xml
   ```

2. **Déploiement automatique** :
   ```yaml
   # .github/workflows/deploy.yml
   name: Deploy

   on:
     release:
       types: [published]

   jobs:
     deploy:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - name: Deploy to production
           run: |
             # Votre logique de déploiement ici
             echo "Deploying version ${GITHUB_REF#refs/tags/}"
   ```

### Bonnes pratiques CI/CD

1. **Tests parallèles** :
   - Exécutez les tests unitaires, d'intégration et de performance en parallèle
   - Utilisez des matrices de test pour différentes versions de Python

2. **Environnements de test** :
   - Créez des environnements de test isolés
   - Utilisez des fixtures pour les données de test

3. **Validation des modèles** :
   - Validez les performances du modèle avant le déploiement
   - Comparez avec le modèle précédent

4. **Rollback automatique** :
   - Implémentez une stratégie de rollback en cas d'échec
   - Surveillez les métriques après le déploiement

## 📊 Surveillance et métriques

### Métriques clés à surveiller

1. **Performances du modèle** :
   - Précision, rappel, F1-score
   - Latence d'inférence (moyenne, p95, p99)
   - Taux d'utilisation du GPU/CPU

2. **Métriques système** :
   - Utilisation de la mémoire
   - Charge CPU/GPU
   - Utilisation du disque et E/S

3. **Métriques métier** :
   - Taux de réussite des prédictions
   - Temps de réponse du système
   - Taux d'erreur par type

### Outils recommandés

1. **Prometheus + Grafana** :
   - Collecte et stockage des métriques
   - Tableaux de bord personnalisables
   - Alertes configurables

2. **Elastic Stack (ELK)** :
   - Centralisation des logs
   - Analyse en temps réel
   - Recherche puissante

3. **Datadog/New Relic** :
   - Solution tout-en-un
   - APM (Application Performance Monitoring)
   - Surveillance des infrastructures distribuées

### Exemple de configuration Prometheus

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'adan_trading_bot'
    static_configs:
      - targets: ['localhost:8000']
```

### Alertes recommandées

1. **Alertes critiques** :
   - Erreurs système
   - Temps d'indisponibilité
   - Perte de données

2. **Alertes de performance** :
   - Latence élevée
   - Utilisation élevée des ressources
   - Dérive des données

## 📚 Documentation du code

### Standards de documentation

1. **Docstrings** : Suivez le format Google Style pour les docstrings
   ```python
   def calculer_rendement(prix_initial: float, prix_final: float) -> float:
       """Calcule le rendement entre deux prix.

       Args:
           prix_initial (float): Prix initial de l'actif
           prix_final (float): Prix final de l'actif

       Returns:
           float: Rendement en pourcentage

       Raises:
           ValueError: Si le prix initial est nul ou négatif
       """
       if prix_initial <= 0:
           raise ValueError("Le prix initial doit être positif")
       return (prix_final - prix_initial) / prix_initial * 100
   ```

2. **Commentaires** :
   - Expliquez le "pourquoi" plutôt que le "comment"
   - Évitez les commentaires redondants
   - Mettez à jour les commentaires quand le code change

### Génération de documentation

1. **Sphinx** :
   - Installation : `pip install sphinx sphinx-rtd-theme`
   - Configuration : `sphinx-quickstart`
   - Génération : `make html`

2. **Exemple de configuration** :
   ```python
   # conf.py
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Pour le format Google Style
    'sphinx.ext.viewcode',
]

html_theme = 'sphinx_rtd_theme'
   ```

### Documentation des API

1. **FastAPI** (si utilisé) :
   - Documentation automatique sur `/docs` et `/redoc`
   - Utilisez les docstrings pour documenter les endpoints

2. **Exemple** :
   ```python
   @app.get("/api/portfolio/performance", response_model=PerformanceResponse)
   async def get_portfolio_performance(
       start_date: date,
       end_date: date = None,
       timeframe: str = "1d"
   ):
       """Récupère la performance du portefeuille sur une période donnée.

       Args:
           start_date: Date de début de la période
           end_date: Date de fin (défaut: aujourd'hui)
           timeframe: Période (1d, 1w, 1m, 1y)

       Returns:
           PerformanceResponse: Objet contenant les métriques de performance
       """
       # Implémentation...
   ```

## 🏗 Bonnes pratiques de développement

### Structure de code

1. **Organisation des modules** :
   ```
   src/
   ├── adan_trading_bot/
   │   ├── __init__.py
   │   ├── core/
   │   │   ├── __init__.py
   │   │   ├── models.py
   │   │   └── utils.py
   │   ├── data/
   │   └── trading/
   └── tests/
   ```

2. **Conventions de nommage** :
   - Classes : `CamelCase`
   - Variables et fonctions : `snake_case`
   - Constantes : `UPPER_SNAKE_CASE`
   - Fichiers de test : `test_*.py`

### Qualité du code

1. **Vérifications automatiques** :
   ```bash
   # Vérification du style
   black --check .
   flake8 .

   # Vérification des types
   mypy .

   # Tests
   pytest
   ```

2. **Git Hooks** :
   Utilisez `pre-commit` pour exécuter des vérifications avant chaque commit :
   ```yaml
   # .pre-commit-config.yaml
   repos:
   -   repo: https://github.com/psf/black
       rev: 22.3.0
       hooks:
       - id: black
         language_version: python3.10
   -   repo: https://github.com/PyCQA/flake8
       rev: 4.0.1
       hooks:
       - id: flake8
   ```

### Gestion des dépendances

1. **Mise à jour sécurisée** :
   ```bash
   # Mettre à jour une dépendance
   uv pip install --upgrade package_name

   # Geler les dépendances
   uv pip freeze > requirements.txt
   ```

2. **Environnements virtuels** :
   ```bash
   # Créer un environnement
   conda create -n trading_env python=3.10
   conda activate trading_env

   # Installer les dépendances
   uv pip install -r requirements.txt
   ```

## 🧪 Tests automatisés

### Structure des tests

```
tests/
├── unit/
│   ├── test_models.py
│   └── test_utils.py
├── integration/
│   └── test_data_pipeline.py
└── conftest.py
```

### Exemple de test unitaire

```python
import pytest
from adan_trading_bot.core.utils import calculer_rendement

def test_calcul_rendement_positif():
    """Teste le calcul de rendement avec des valeurs positives."""
    assert calculer_rendement(100, 110) == 10.0  # +10%

def test_calcul_rendement_negatif():
    """Teste le calcul de rendement avec une perte."""
    assert calculer_rendement(100, 90) == -10.0  # -10%

def test_calcul_rendement_prix_initial_zero():
    """Teste la levée d'exception avec un prix initial nul."""
    with pytest.raises(ValueError):
        calculer_rendement(0, 100)
```

### Tests d'intégration

```python
import pandas as pd
from adan_trading_bot.data.loader import DataLoader

def test_chargement_donnees(tmp_path):
    """Teste le chargement des données avec un fichier temporaire."""
    # Crée un fichier de test
    df_test = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=5),
        'close': [100, 101, 102, 101, 103]
    })
    test_file = tmp_path / "test_data.csv"
    df_test.to_csv(test_file, index=False)

    # Teste le chargement
    loader = DataLoader(str(test_file))
    df_loaded = loader.load()

    # Vérifications
    assert not df_loaded.empty
    assert 'close' in df_loaded.columns
    assert len(df_loaded) == 5
```

### Exécution des tests

```bash
# Tous les tests
pytest

# Tests unitaires uniquement
pytest tests/unit/

# Tests avec couverture de code
pytest --cov=adan_trading_bot --cov-report=html

# Tests en parallèle
pytest -n auto
```

### Intégration continue

Exemple de configuration GitHub Actions :

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python 3.10
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: |
        pytest --cov=./ --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## 🐞 Gestion des erreurs et journalisation

### Hiérarchie des exceptions

```python
class TradingError(Exception): pass
class DataError(TradingError): pass
class StrategyError(TradingError): pass
class ExecutionError(TradingError): pass
```

### Configuration du logging

```python
import logging

# Configuration de base
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Bonnes pratiques

1. **Niveaux de log** :
   - `DEBUG` : Détails pour le débogage
   - `INFO` : Suivi du flux normal
   - `WARNING` : Événements inattendus
   - `ERROR` : Échec d'une opération
   - `CRITICAL` : Erreur critique

2. **Journalisation structurée** :
   ```python
   logger.info("Ordre exécuté", extra={
       'type': 'order',
       'symbol': 'BTC/USD',
       'side': 'buy',
       'amount': 0.1,
       'price': 50000.0
   })
   ```

3. **Gestion des erreurs** :
   ```python
   try:
       execute_trade(order)
   except ExecutionError as e:
       logger.error("Échec de l'exécution de l'ordre",
                   exc_info=True,
                   extra={'order': order.to_dict()})
       raise
   ```

## ⚙️ Gestion de la configuration

### Fichier de configuration principal

```yaml
# config/config.yaml

# Paramètres généraux
general:
  env: "development"
  log_level: "INFO"
  timezone: "UTC"

# Paramètres de trading
trading:
  pairs:
    - "BTC/USD"
    - "ETH/USD"
  timeframe: "1h"
  max_position_size: 0.1  # 10% du capital
  risk_per_trade: 0.01    # 1% de risque par trade

# Paramètres de l'API
exchange:
  name: "binance"
  api_key: "${EXCHANGE_API_KEY}"  # Variable d'environnement
  api_secret: "${EXCHANGE_API_SECRET}"
  sandbox: true

# Paramètres du modèle
model:
  name: "lstm_predictor"
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
```

### Chargement de la configuration

```python
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

class Config:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = Path(config_path)
        self._load_config()

    def _load_config(self):
        # Charger les variables d'environnement
        load_dotenv()

        # Charger la configuration YAML
        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)

        # Remplacer les variables d'environnement
        self._resolve_env_vars(self._config)

    def _resolve_env_vars(self, config):
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    config[key] = os.getenv(env_var, '')
                elif isinstance(value, (dict, list)):
                    self._resolve_env_vars(value)
        elif isinstance(config, list):
            for item in config:
                if isinstance(item, (dict, list)):
                    self._resolve_env_vars(item)

    def __getattr__(self, name):
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Configuration '{name}' non trouvée")

# Utilisation
config = Config()
print(config.trading.pairs)  # Accès aux paramètres
```

### Bonnes pratiques

1. **Sécurité** :
   - Ne jamais commiter de clés API dans le dépôt
   - Utiliser des variables d'environnement pour les données sensibles
   - Créer un fichier `.env.example` comme modèle

2. **Validation** :
   - Valider la configuration au démarrage
   - Fournir des valeurs par défaut appropriées
   - Documenter tous les paramètres de configuration

3. **Environnements multiples** :
   - Maintenir des fichiers de configuration séparés pour dev/test/prod
   - Utiliser des variables d'environnement pour la surcharge

## 🔄 Versionnage sémantique

Le projet suit le [Semantic Versioning](https://semver.org/) (SemVer) :

### Format : `MAJEUR.MINEUR.CORRECTIF`

- **MAJEUR** : Changements incompatibles avec les versions précédentes
- **MINEUR** : Nouvelles fonctionnalités rétrocompatibles
- **CORRECTIF** : Corrections de bugs rétrocompatibles

### Exemples

- `1.0.0` : Première version stable
- `1.1.0` : Ajout de nouvelles fonctionnalités
- `1.1.1` : Correction de bugs
- `2.0.0` : Changements majeurs non rétrocompatibles

### Branches et versions

- `main` : Dernière version stable (production)
- `develop` : Prochaine version en développement
- `feature/*` : Nouvelles fonctionnalités
- `release/*` : Préparation d'une nouvelle version
- `hotfix/*` : Corrections critiques pour la production

### Mise à jour de version

1. Mettre à jour la version dans `setup.py` :
   ```python
   # setup.py
   setup(
       name="adan-trading-bot",
       version="1.0.0",  # Mettre à jour ici
       # ...
   )
   ```

2. Créer un tag Git :
   ```bash
   git tag -a v1.0.0 -m "Version 1.0.0"
   git push origin v1.0.0
   ```

3. Créer une release sur GitHub avec les notes de version

## 🔒 Sécurité

### Bonnes pratiques de sécurité

1. **Gestion des clés API** :
   - Utilisez toujours des variables d'environnement pour les clés API
   - Limitez les permissions des clés API au strict nécessaire
   - Régénérez régulièrement les clés API
   - Ne stockez jamais de clés API en clair dans le code

2. **Sécurité du code** :
   ```python
   # ❌ À éviter
   api_key = "ma-cle-api-tres-secrete"

   # ✅ À privilégier
   import os
   api_key = os.getenv("EXCHANGE_API_KEY")
   ```

3. **Protection contre les attaques** :
   - Validez et nettoyez toutes les entrées utilisateur
   - Utilisez des requêtes paramétrées pour les bases de données
   - Implémentez des limites de débit (rate limiting)
   - Utilisez HTTPS pour toutes les communications

4. **Audit de sécurité** :
   - Utilisez des outils d'analyse statique de code (bandit, safety)
   - Tenez les dépendances à jour (Dependabot, Renovate)
   - Effectuez des tests d'intrusion réguliers

### Exemple de configuration sécurisée

```bash
# .gitignore
.env
*.pem
*.key
__pycache__/
```

```bash
# .env.example
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_API_SECRET=your_api_secret_here
```

### Outils recommandés

1. **Analyse de sécurité** :
   - `bandit`: Détection des vulnérabilités Python
   - `safety`: Vérification des vulnérabilités des dépendances
   - `trivy`: Analyse des vulnérabilités des conteneurs

2. **Surveillance** :
   - Audit des logs d'accès
   - Détection d'intrusion
   - Surveillance des activités suspectes

## 📜 Licence

Distribué sous licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

## 🔄 Gestion des versions

Le projet suit le [Semantic Versioning](https://semver.org/) :
- **MAJOR** : Changements incompatibles avec les versions précédentes
- **MINOR** : Nouvelles fonctionnalités rétrocompatibles
- **PATCH** : Corrections de bugs et améliorations mineures

### Branches
- `main` : Branche de développement principale (instable)
- `stable` : Dernière version stable
- `feature/*` : Nouvelles fonctionnalités
- `bugfix/*` : Corrections de bugs

## ✅ Bonnes pratiques de développement

1. **Tests** :
   - Écrivez des tests unitaires pour les nouvelles fonctionnalités
   - Vérifiez la couverture de code avec `pytest --cov=src tests/`
   - Exécutez les tests avant chaque commit

2. **Documentation** :
   - Documentez les nouvelles fonctions avec des docstrings
   - Mettez à jour le README pour les changements majeurs
   - Ajoutez des exemples d'utilisation

3. **Style de code** :
   - Suivez PEP 8
   - Utilisez `black` pour le formatage
   - Vérifiez avec `flake8` avant de committer

4. **Gestion des dépendances** :
   - Ajoutez les nouvelles dépendances à `requirements.txt`
   - Utilisez des versions spécifiques pour la reproductibilité
   - Mettez à jour `setup.py` si nécessaire

## ⚠️ Avertissement

**Ce logiciel est fourni à des fins éducatives et de recherche uniquement.**

Le trading de cryptomonnaies comporte des risques importants de perte en capital. Ne tradez pas avec de l'argent que vous ne pouvez pas vous permettre de perdre. Les performances passées ne sont pas indicatives des résultats futurs. Les développeurs ne peuvent être tenus responsables des pertes éventuelles encourues lors de l'utilisation de ce logiciel.
