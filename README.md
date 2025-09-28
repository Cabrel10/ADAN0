# 🚀 PROJET ADAN - Système de Trading Automatisé

ADAN (Autonomous Digital Asset Navigator) est un système avancé de trading algorithmique conçu pour le marché des cryptomonnaies. Cette version (ADAN001_clean) inclut des améliorations majeures en termes de stabilité, de performance et de fonctionnalités.

## 🌟 Fonctionnalités principales

- **Trading multi-timeframe** (5m, 1h, 4h)
- **Gestion avancée des risques** avec système de capital progressif
- **Modèles d'IA** entraînés avec renforcement profond (PPO)
- **Système de récompenses d'excellence** (GUGU & MARCH)
- **Optimisation des performances** avec gestion de la mémoire et du GPU
- **Tableau de bord TensorBoard** pour le suivi des performances

## 🚀 Démarrage rapide

### Prérequis

- Python 3.8+
- CUDA 11.8 (pour l'accélération GPU)
- Git
- Compte Binance (pour le trading en direct)

### Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/Cabrel10/ADAN0.git
cd ADAN0
git checkout ADAN001_clean
git submodule update --init --recursive
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
cd bot
pip install -e .
```

3. Configurer les variables d'environnement :
```bash
cp .env.example .env
# Éditer le fichier .env avec vos clés API
```

## 📊 Structure du projet

```
ADAN0/
├── bot/                     # Code principal du bot de trading
│   ├── config/             # Fichiers de configuration
│   ├── scripts/            # Scripts d'entraînement et d'évaluation
│   └── src/                # Code source Python
├── data/                   # Données brutes et traitées
│   ├── raw/               # Données brutes (CSV)
│   └── processed/         # Données traitées (Parquet)
├── models/                # Modèles entraînés
└── logs/                  # Journaux et métriques
```

## 🛠 Configuration

Consultez le fichier `bot/config/config.yaml` pour personnaliser les paramètres de trading, les stratégies et les modèles.

## 🚦 Exécution

### Entraînement du modèle
```bash
cd bot
python scripts/train_parallel_agents.py
```

### Backtesting
```bash
python scripts/run_backtest.py
```

### Trading en direct
```bash
python scripts/run_live_trading.py
```

## 📈 Monitoring

Pour visualiser les performances :
```bash
tensorboard --logdir=logs/
```

## 📚 Documentation complète

Consultez les fichiers suivants pour plus de détails :
- `INSTRUCTIONS_UTILISATION_CORRIGEE.md` : Guide d'utilisation détaillé
- `RAPPORT_CORRECTIONS_TENSORBOARD_DASHBOARD.md` : Documentation technique

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👥 Contribution

Les contributions sont les bienvenues ! Veuillez lire les directives de contribution avant de soumettre une pull request.

## 📞 Support

Pour toute question ou problème, veuillez ouvrir une issue sur le dépôt GitHub ou contacter l'équipe de développement.
