# 🚀 Guide de démarrage rapide ADAN Trading Bot

Ce guide vous aidera à configurer et exécuter ADAN Trading Bot en quelques étapes simples.

## 📋 Prérequis

- **Python 3.10+** - [Télécharger Python](https://www.python.org/downloads/)
- **Miniconda/Anaconda** - [Télécharger Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- **Git** - [Télécharger Git](https://git-scm.com/downloads)
- **uv pip** (installation plus rapide des dépendances) :
  ```bash
  pip install uv
  ```

## 🛠 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Cabrel10/ADAN.git
cd ADAN
git checkout stable  # Basculer sur la branche stable
```

### 2. Configuration de l'environnement

#### Créer et activer l'environnement Conda

```bash
# Créer l'environnement
conda create -n trading_env python=3.10 -y

# Activer l'environnement
conda activate trading_env
```

#### Installer les dépendances avec uv pip

```bash
uv pip install -r requirements.txt
```

### 3. Configuration initiale

1. Copier les fichiers de configuration :
   ```bash
   cp .env.example .env
   cp config/config.example.yaml config/config.yaml
   ```

2. Configurer les variables d'environnement (`.env`) :
   ```ini
   # Exemple de configuration minimale
   ADAN_QUIET_AFTER_INIT=1
   ADAN_RICH_STEP_TABLE=1
   ADAN_RICH_STEP_EVERY=10
   ```

3. Configurer le fichier `config/config.yaml` selon vos besoins.

2. Modifiez `config/config.yaml` selon vos besoins :
   - Configurer les paramètres du réseau de neurones
   - Définir les paires de trading
   - Configurer les paramètres de risque

## 🚦 Exécution

### Mode développement avec affichage détaillé

Pour exécuter en mode développement avec affichage détaillé des logs :

```bash
ADAN_QUIET_AFTER_INIT=0 ADAN_RICH_STEP_TABLE=1 \
python scripts/train_parallel_agents.py --config config/config.yaml
```

### Mode production (optimisé)

Pour exécuter en mode production avec optimisation des performances :

```bash
ADAN_QUIET_AFTER_INIT=1 ADAN_RICH_STEP_EVERY=50 \
python scripts/train_parallel_agents.py --config config/config.yaml
```

### Variables d'environnement utiles

| Variable | Valeur par défaut | Description |
|----------|------------------|-------------|
| `ADAN_QUIET_AFTER_INIT` | `1` | Active le mode silencieux après l'initialisation |
| `ADAN_RICH_STEP_TABLE` | `1` | Active l'affichage du tableau de suivi Rich |
| `ADAN_RICH_STEP_EVERY` | `10` | Fréquence d'affichage du tableau (en pas) |
| `ADAN_JSONL_EVERY` | `100` | Fréquence d'écriture des logs au format JSONL |

### Surveillance des performances

Pour surveiller les performances en temps réel :

1. Dans un premier terminal, lancez TensorBoard :
   ```bash
   tensorboard --logdir=./tensorboard_logs
   ```

2. Dans un deuxième terminal, lancez l'entraînement :
   ```bash
   python scripts/train_parallel_agents.py --config config/config.yaml
   ```

## 🧪 Tests

### Exécuter tous les tests

```bash
pytest
```

### Exécuter une catégorie spécifique de tests

```bash
# Tests unitaires
pytest tests/unit/

# Tests d'intégration
pytest tests/integration/

# Un test spécifique
pytest tests/unit/test_shared_buffer.py -v
```

### Couverture de code

```bash
pytest --cov=src tests/
```

## 🏗 Structure du projet

```
.
├── config/                 # Fichiers de configuration
│   ├── config.yaml         # Configuration principale
│   └── config.example.yaml # Exemple de configuration
├── data/                   # Données brutes et traitées
│   ├── raw/               # Données brutes (non versionnées)
│   └── processed/         # Données prétraitées (non versionnées)
├── docs/                  # Documentation technique
├── notebooks/             # Notebooks d'analyse
├── scripts/               # Scripts utilitaires
│   ├── prepare_adan_data.py
│   └── test_orchestrator_integration.py
├── src/                   # Code source
│   └── adan_trading_bot/
│       ├── environment/   # Environnements de trading
│       ├── models/        # Modèles d'IA
│       ├── training/      # Logique d'entraînement
│       ├── utils/         # Utilitaires communs
│       ├── __init__.py
│       └── main.py        # Point d'entrée principal
└── tests/                 # Tests automatisés
    ├── unit/             # Tests unitaires
    └── integration/      # Tests d'intégration
```

## 🛠 Développement

### Standards de code

Le projet utilise :
- **Black** pour le formatage du code
- **isort** pour le tri des imports
- **Flake8** pour le linting

Pour formater le code avant de commiter :

```bash
black .
isort .
flake8
```

### Workflow Git

1. Créez une branche pour votre fonctionnalité :
   ```bash
   git checkout -b feature/nom-de-la-fonctionnalite
   ```

2. Committez vos changements :
   ```bash
   git add .
   git commit -m "Description claire des modifications"
   ```

3. Poussez vos changements :
   ```bash
   git push origin feature/nom-de-la-fonctionnalite
   ```

4. Créez une Pull Request

## 🚨 Dépannage

### Problèmes courants

#### Erreurs d'importation
- Vérifiez que votre environnement virtuel est activé
- Vérifiez que vous êtes dans le bon répertoire
- Exécutez `pip install -e .` pour une installation en mode développement

#### Problèmes de configuration
- Vérifiez l'indentation YAML
- Vérifiez les chemins des fichiers
- Consultez les logs dans `logs/` pour plus de détails

#### Problèmes de performance
- Vérifiez l'utilisation de la mémoire
- Réduisez la taille des lots d'entraînement si nécessaire
- Vérifiez les paramètres de parallélisme

## 📚 Documentation complète

Pour une documentation plus détaillée, consultez :
- [Guide de configuration avancée](docs/configuration_guide.md)
- [Guide de développement](docs/development_guide.md)
- [Architecture du système](docs/architecture.md)

## ⚠️ Avertissement important

**Ce logiciel est fourni à des fins éducatives et de recherche uniquement.**

Le trading de cryptomonnaies comporte des risques importants de perte en capital. Ne tradez pas avec de l'argent que vous ne pouvez pas vous permettre de perdre. Les performances passées ne sont pas indicatives des résultats futurs. Les développeurs ne peuvent être tenus responsables des pertes éventuelles encourues lors de l'utilisation de ce logiciel.

Pour toute question ou problème, veuillez ouvrir une issue sur le dépôt du projet.
