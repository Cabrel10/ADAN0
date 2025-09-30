# 🚀 PROJET ADAN - Système de Trading Automatisé

ADAN (Autonomous Digital Asset Navigator) est un système avancé de trading algorithmique conçu pour le marché des cryptomonnaies. Cette version (ADAN002_lstm) introduit l'architecture LSTM pour une meilleure modélisation des séries temporelles et une prise de décision contextuelle.

## 🌟 Fonctionnalités principales

- **Mémoire à long terme** avec architecture LSTM personnalisée
- **Trading multi-timeframe** (5m, 1h, 4h) avec gestion intelligente
- **Gestion avancée des risques** avec système de capital progressif
- **Modèles d'IA** avec PPO récurrent (RecurrentPPO)
- **Politique personnalisée** pour observations complexes (MultiInputLstmPolicy)
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

## 🧠 Architecture LSTM

La version ADAN002 introduit une architecture LSTM avancée pour la modélisation des séquences temporelles :

- **Mémoire à long terme** : Capture les dépendances à long terme dans les données de marché
- **Gestion des observations complexes** : Support natif des espaces d'observation de type dictionnaire
- **Entraînement parallèle** : Optimisé pour le multi-agents avec synchronisation des poids

### Caractéristiques techniques

- **Algorithme** : RecurrentPPO de sb3-contrib
- **Politique personnalisée** : `MultiInputLstmPolicy` pour les observations complexes
- **Taille de la mémoire** : 256 unités par défaut (configurable)
- **Couches LSTM** : 1 couche par défaut (configurable)

## ⚙️ Configuration avancée

### Configuration de l'entraînement LSTM

Le fichier `bot/config/config.yaml` contient les paramètres spécifiques à l'entraînement LSTM :

```yaml
agent:
  policy: "MultiInputLstmPolicy"
  policy_kwargs:
    lstm:
      lstm_hidden_size: 256  # Taille de la couche cachée LSTM
      n_lstm_layers: 1       # Nombre de couches LSTM empilées
      enable_critic_lstm: True  # Activer le LSTM pour le critique
      shared_lstm: False       # Partager les poids LSTM entre l'acteur et le critique
```

### Exécution de l'entraînement

Pour lancer un entraînement avec la nouvelle architecture LSTM :

```bash
python bot/scripts/train_parallel_agents.py \
  --config-path bot/config/config.yaml \
  --checkpoint-dir bot/checkpoints \
  --num-envs 4 \
  --sync-interval 10000
```

### Surveillance de l'entraînement

1. **TensorBoard** :
   ```bash
   tensorboard --logdir=./logs/
   ```

2. **Métriques clés** :
   - `loss/policy_gradient_loss` : Perte du gradient de politique
   - `loss/value_loss` : Perte de la fonction de valeur
   - `loss/loss` : Perte totale
   - `rollout/ep_rew_mean` : Récompense moyenne par épisode

## 📊 Structure du projet

```
ADAN0/
├── bot/                     # Code principal du bot de trading
│   ├── src/adan_trading_bot/model/
│   │   └── custom_policy.py  # Implémentation de MultiInputLstmPolicy
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

### Guides principaux
- `INSTRUCTIONS_UTILISATION_CORRIGEE.md` - Guide d'utilisation complet avec exemples
- `INSTRUCTIONS_COLAB.md` - Instructions pour exécuter sur Google Colab
- `ENVIRONMENT_SETUP.md` - Guide d'installation de l'environnement

### Rapports techniques
- `RAPPORT_CORRECTIONS_TENSORBOARD_DASHBOARD.md` - Corrections du système de monitoring
- `RAPPORT_CORRECTION_PROBLEME_4_LOGS_WORKERS.md` - Résolution des problèmes de logs
- `RAPPORT_CORRECTION_PROBLEME_5_METRIQUES_ZERO.md` - Correction des métriques à zéro
- `RAPPORT_FINAL_BUG_EQUITE_DRAWDOWN.md` - Analyse du bug de drawdown

### Autres documents
- `CORRECTIONS_APPLIQUEES_BUGS_CRITIQUES.md` - Liste des corrections majeures
- `CORRECTIONS_WORKER_FREQUENCY.md` - Optimisation de la fréquence des workers


Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👥 Contribution

Les contributions sont les bienvenues ! Veuillez lire les directives de contribution avant de soumettre une pull request.

## 🚀 Commandes Essentielles

### 1. Configuration Initiale
```bash
# Activer l'environnement conda
conda activate trading_env
# OU avec le chemin complet
/home/morningstar/miniconda3/envs/trading_env/bin/python

# Vérifier l'installation
tensorboard --version
```

### 2. Entraînement du Modèle
```bash
# Lancer l'entraînement standard (30 secondes de test)
timeout 30s /home/morningstar/miniconda3/envs/trading_env/bin/python \
  bot/scripts/train_parallel_agents.py \
  --config bot/config/config.yaml \
  --checkpoint-dir bot/checkpoints

# Reprendre un entraînement existant
--resume
```

### 3. Monitoring en Temps Réel
```bash
# Lancer le dashboard de monitoring
cd bot/scripts
python training_dashboard.py
# Accès : http://localhost:8050

# Visualiser les logs TensorBoard
tensorboard --logdir=reports/tensorboard_logs
```

### 4. Gestion des Checkpoints
```bash
# Lister les checkpoints disponibles
ls -lat bot/checkpoints/ | grep checkpoint_

# Valider l'intégrité des checkpoints
python test_tensorboard_checkpoint_validation.py

# Nettoyer les anciens checkpoints
find bot/checkpoints -name "checkpoint_*" -type d -mtime +7 -exec rm -rf {} \;
```

### 5. Surveillance des Performances
```bash
# Suivre les logs en temps réel
tail -f logs/training.log

# Vérifier l'utilisation des ressources
htop  # ou nvtop pour les GPUs

# Vérifier l'état des workers
python scripts/check_workers.py
```

### 6. Maintenance et Dépannage
```bash
# Nettoyer les fichiers temporaires
make clean

# Mettre à jour le dépôt
git pull
git submodule update --recursive

# Vérifier les dépendances
pip list | grep -E "tensorboard|stable-baselines3|gym|numpy"
