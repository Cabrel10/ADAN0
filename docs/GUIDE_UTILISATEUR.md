# 📖 Guide d'Utilisation ADAN0

Guide complet pour utiliser le bot de trading ADAN0 — de l'installation au live trading.

---

## 🚀 Démarrage Rapide (5 minutes)

### Installation

```bash
# Cloner le repo
git clone https://github.com/Cabrel10/ADAN0.git
cd ADAN0

# Installer les dépendances
pip install -e .
```

### Test Rapide (CPU, ~5 min)

```bash
PYTHONPATH=src python scripts/smoke_test.py
```

Cela lance un test sur **5000 steps** avec les données historiques. Résultat attendu: Portfolio > $15.

---

## 📊 Pipeline Complet (Colab - Recommandé)

### Étape 1: Configuration Colab

1. Ouvrir `notebooks/ADAN_Full_Training_H100.ipynb` dans Google Colab
2. Activer **GPU** (T4 minimum, H100 idéal)
3. Exécuter les cellules dans l'ordre

Le notebook gère tout automatiquement:
- ✅ Clone repo + installation
- ✅ Téléchargement données BTC (6 ans, CCXT/Bitget)
- ✅ Feature engineering (21 indicateurs/timeframe)
- ✅ Création splits train/test/val
- ✅ Entraînement 500K+ steps
- ✅ Backtest OOS déterministe
- ✅ Export modèle

---

## 🏃 Entraînement Local (GPU)

### Configuration

Éditer `config/config.yaml`:

```yaml
training:
  mode: heavy                    # ou sandbox
  steps: 500000                  # cible
  num_cpus: 8
  num_samples: 4                 # 4 workers
  checkpoint_freq: 5000
  
reward:
  pnl_scale: 100
  drawdown_penalty: 50.0
  
profiles:
  - name: scalper
    timeframe: 5m
    gamma: 0.95
  - name: intraday
    timeframe: 1h
    gamma: 0.99
  - name: swing
    timeframe: 4h
    gamma: 0.995
  - name: position
    timeframe: 4h
    gamma: 0.999
```

### Lancer l'entraînement

```bash
PYTHONPATH=src python scripts/train_parallel_agents.py \
  --mode heavy \
  --steps 500000 \
  --num-cpus 8 \
  --num-samples 4 \
  --checkpoint-dir checkpoints/my_run
```

**Monitoring en temps réel:**
```bash
PYTHONPATH=src python scripts/live_monitor.py --checkpoint-dir checkpoints/my_run
```

---

## 📥 Télécharger les Données

### Télécharger BTC historique (6 ans)

```bash
PYTHONPATH=src python scripts/download_ccxt_data.py \
  --exchange bitget \
  --pair BTC/USDT \
  --timeframe 5m \
  --years 6 \
  --output data/raw/btc_5m.parquet
```

**Résultat**: `data/raw/btc_5m.parquet` (~5GB)

### Options d'exchange
- `bitget` ✅ (recommandé, données clean)
- `binance` ✅ (données publiques)
- Autres: Kraken, Coinbase, etc.

---

## 🔨 Créer le Dataset

### Étape 1: Calculer les indicateurs

```bash
PYTHONPATH=src python scripts/compute_features_real.py \
  --input data/raw/btc_5m.parquet \
  --output data/processed/btc_features.parquet
```

Cela ajoute **21 indicateurs** par timeframe (5m, 1h, 4h):
- RSI, MACD, Bollinger Bands
- ATR, ADX, Stochastic
- SMA, EMA, VWAP
- Volume profile, etc.

### Étape 2: Créer les splits

```bash
PYTHONPATH=src python scripts/create_train_test_val_splits.py \
  --input data/processed/btc_features.parquet \
  --output-dir data/splits \
  --train-pct 0.70 \
  --test-pct 0.20 \
  --val-pct 0.10
```

**Résultat**: 3 fichiers (chronologiques, no data leakage)
- `train.parquet` (70%)
- `test.parquet` (20%)
- `val.parquet` (10%)

---

## 🤖 Entraînement PBT

### Mode Sandbox (rapide, CPU)

```bash
PYTHONPATH=src python scripts/train_parallel_agents.py \
  --mode sandbox \
  --steps 5000
```

- ⏱️ Durée: ~5 minutes (CPU)
- 💾 Peu de mémoire requis
- 🎯 Parfait pour tests/debug

### Mode Heavy (production, GPU)

```bash
PYTHONPATH=src python scripts/train_parallel_agents.py \
  --mode heavy \
  --steps 500000 \
  --gpu-per-trial 1 \
  --num-samples 4
```

- ⏱️ Durée: ~48 heures (H100)
- 💾 32GB+ GPU VRAM
- 🎯 Production-ready

### Reprise d'entraînement

```bash
PYTHONPATH=src python scripts/verify_checkpoint_resume.py \
  --checkpoint-dir checkpoints/my_run/checkpoint_050000
```

---

## ✅ Backtest Déterministe (OOS)

### Évaluer sur test split

```bash
PYTHONPATH=src python scripts/deterministic_backtest.py \
  --checkpoint checkpoints/my_run/checkpoint_500000 \
  --split test \
  --output results/backtest_test.json
```

**Résultat**: Statistiques détaillées
```json
{
  "total_trades": 156,
  "win_rate": 52.3,
  "sharpe_ratio": 2.1,
  "max_drawdown": 18.5,
  "total_return": 12.4,
  "...": "..."
}
```

### Comparer train vs test

```bash
# Backtest on train split (sanity check)
PYTHONPATH=src python scripts/deterministic_backtest.py \
  --checkpoint checkpoints/my_run/checkpoint_500000 \
  --split train \
  --output results/backtest_train.json

# Comparer: train vs test (overfitting?)
```

---

## 📈 Entraîner l'Oracle (optionnel)

L'oracle (HMM) détecte les régimes de marché (uptrend, downtrend, range).

```bash
PYTHONPATH=src python scripts/train_oracle.py \
  --data data/processed/btc_features.parquet \
  --output models/oracle.pkl \
  --states 3
```

Utilisé par `dynamic_behavior_engine` pour adapter les risques.

---

## 📝 Paper Trading (Recommandé avant live)

Simuler le trading avec la clé API (sans capital réel):

```bash
PYTHONPATH=src python scripts/paper_trading_monitor.py \
  --checkpoint checkpoints/my_run/checkpoint_500000 \
  --exchange bitget \
  --mode paper \
  --duration 2weeks
```

**Durant 2 semaines**:
- ✅ Vérifier la stabilité du modèle
- ✅ Détecter les bugs d'exécution
- ✅ Valider l'intégration exchange
- ✅ Tester la gestion des commissions réelles

**Métriques à surveiller**:
- Win rate stable > 50%
- Sharpe > 1.0
- Max drawdown < 20%
- Trades cohérents (pas de bugs)

---

## 🔴 Live Trading (Capital réel)

### Prérequis

- ✅ 2 semaines paper trading réussies
- ✅ Backtest OOS: Sharpe > 1.5, DD < 25%
- ✅ Capital minimal: $100 (micro positions)
- ✅ Monitoring continu

### Lancer

```bash
PYTHONPATH=src python scripts/run_bot.py \
  --checkpoint checkpoints/my_run/checkpoint_500000 \
  --exchange bitget \
  --mode live \
  --capital 100 \
  --max-positions 4 \
  --max-dd-stop 20
```

### Monitoring Live

```bash
PYTHONPATH=src python scripts/live_monitor.py \
  --mode live \
  --update-freq 1m
```

**Alarmes activées**:
- 🔔 Drawdown > 20% → Arrêt d'urgence
- 🔔 Erreur de connexion → Log + notification
- 🔔 PnL anormal → Spike detection

---

## 🔍 Vérifier la Santé du Modèle

### Test d'intégrité des checkpoints

```bash
PYTHONPATH=src python scripts/verify_checkpoint_config.py \
  --checkpoint checkpoints/my_run/checkpoint_500000
```

Output:
```
✓ Config matches
✓ Weights shape OK
✓ Normalization stats loaded
✓ Policy callable
```

### Test du flux PnL

```bash
PYTHONPATH=src python scripts/test_pnl_flow.py \
  --checkpoint checkpoints/my_run/checkpoint_500000 \
  --steps 100
```

Vérifie:
- ✅ PnL calculations correctes
- ✅ Pas d'NaN/Inf
- ✅ Position lifecycle cohérent

---

## 📂 Structure des Fichiers

```
ADAN0/
├── config/config.yaml              # ← Éditer ici
├── data/
│   ├── raw/btc_5m.parquet         # Données brutes
│   └── processed/
│       ├── btc_features.parquet    # + 21 indicateurs
│       └── splits/
│           ├── train.parquet
│           ├── test.parquet
│           └── val.parquet
├── checkpoints/my_run/
│   ├── checkpoint_050000/
│   ├── checkpoint_100000/
│   └── checkpoint_500000/          # ← Best model
├── models/
│   └── oracle.pkl                  # Régime oracle
├── results/
│   ├── backtest_train.json
│   ├── backtest_test.json
│   └── paper_trading_log.json
├── logs/
│   ├── adan_trading_bot.json       # Live logs
│   └── metrics/
│       ├── metrics_*.jsonl         # Trade records
│       └── performance_summary.csv
├── notebooks/
│   └── ADAN_Full_Training_H100.ipynb
└── scripts/
    ├── train_parallel_agents.py    # 1️⃣ Entraîner
    ├── deterministic_backtest.py   # 2️⃣ Backtest
    ├── train_oracle.py             # 3️⃣ Oracle
    ├── paper_trading_monitor.py    # 4️⃣ Paper trading
    ├── run_bot.py                  # 5️⃣ Live trading
    └── [verification scripts]
```

---

## ⚡ Raccourcis Courants

### Je veux juste tester rapidement

```bash
PYTHONPATH=src python scripts/smoke_test.py
```

### Je veux entraîner sur Colab

→ Ouvrir `notebooks/ADAN_Full_Training_H100.ipynb`

### Je veux backtest sur mes données

```bash
# 1. Télécharger
PYTHONPATH=src python scripts/download_ccxt_data.py ...

# 2. Features
PYTHONPATH=src python scripts/compute_features_real.py ...

# 3. Splits
PYTHONPATH=src python scripts/create_train_test_val_splits.py ...

# 4. Backtest
PYTHONPATH=src python scripts/deterministic_backtest.py --split test
```

### Je veux faire du paper trading

```bash
# Supposant checkpoint_500000 existant:
PYTHONPATH=src python scripts/paper_trading_monitor.py \
  --checkpoint checkpoints/my_run/checkpoint_500000 \
  --exchange bitget \
  --mode paper
```

### Je veux du live trading

```bash
PYTHONPATH=src python scripts/run_bot.py \
  --checkpoint checkpoints/my_run/checkpoint_500000 \
  --exchange bitget \
  --mode live \
  --capital 100
```

---

## 🆘 Troubleshooting

### "PYTHONPATH not found"
```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python scripts/train_parallel_agents.py ...
```

### "GPU out of memory"
- Réduire `num_samples` (workers) dans config
- Utiliser `--gpu-per-trial 0.5` (partage GPU)
- Ou utiliser CPU avec `--num-cpus 16`

### "No data downloaded"
```bash
# Vérifier la connexion
python -c "import ccxt; print(ccxt.bitget().load_markets())"
```

### "Backtest shows 0 trades"
- Vérifier le checkpoint est valide: `verify_checkpoint_config.py`
- Vérifier l'observation shape: `test_pnl_flow.py`
- Réduire action threshold dans config

### "Paper trading not connecting"
```bash
# Tester les credentials
python -c "
from ccxt import bitget
exchange = bitget({
    'apiKey': 'YOUR_KEY',
    'secret': 'YOUR_SECRET',
    'password': 'YOUR_PASSWORD'
})
print(exchange.fetch_balance())
"
```

---

## 📖 Documentation Détaillée

- `docs/TIER_BASED_REWARD_IMPLEMENTATION.md` — Système de reward
- `docs/COMPREHENSIVE_AUDIT_GUIDE.md` — Vérification données
- `config/config.yaml` — Tous les paramètres
- `src/adan_trading_bot/` — Code source documenté

---

## ✅ Checklist Avant Live

- [ ] Smoke test réussi (`smoke_test.py`)
- [ ] Entraînement > 50K steps
- [ ] Backtest OOS: Sharpe > 1.5
- [ ] Paper trading 2 semaines sans bug
- [ ] Oracle entraîné
- [ ] Config review (SL/TP, commissions, capital)
- [ ] Monitoring script prêt
- [ ] Credentials testées
- [ ] Contingency plan (manuel stop)

---

**Besoin d'aide?** → Voir `docs/` ou ouvrir une issue sur GitHub.

