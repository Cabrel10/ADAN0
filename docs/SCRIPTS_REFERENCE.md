# 📜 Référence Complète des Scripts

Vue d'ensemble des 14 scripts production-ready et leur utilisation.

---

## 📥 Acquisition Données

### `download_ccxt_data.py`

Télécharge les données de marché historiques via CCXT.

**Utilisation:**
```bash
PYTHONPATH=src python scripts/download_ccxt_data.py \
  --exchange bitget \
  --pair BTC/USDT \
  --timeframe 5m \
  --years 6 \
  --output data/raw/btc_5m.parquet
```

**Paramètres:**
- `--exchange`: bitget, binance, kraken, etc.
- `--pair`: BTC/USDT, ETH/USDT, etc.
- `--timeframe`: 5m, 1h, 4h, 1d
- `--years`: Nombre d'années d'historique
- `--output`: Chemin fichier parquet

**Résultat:** Fichier parquet avec OHLCV brut

**Durée:** 10-30 min (dépend du timeframe)

---

## 🔨 Préparation Dataset

### `compute_features_real.py`

Ajoute 21 indicateurs techniques par timeframe.

**Utilisation:**
```bash
PYTHONPATH=src python scripts/compute_features_real.py \
  --input data/raw/btc_5m.parquet \
  --output data/processed/btc_features.parquet
```

**Indicateurs calculés:**
- Momentum: RSI, MACD, Stochastic
- Volatilité: Bollinger Bands, ATR, Keltner
- Trend: SMA, EMA, VWAP, Ichimoku
- Volume: OBV, CMF, AD
- Custom: Divergence, Support/Résistance

**Résultat:** Parquet enrichi (taille ×2-3)

**Durée:** 5-15 min

---

### `create_train_test_val_splits.py`

Crée les splits d'entraînement chronologiques (no data leakage).

**Utilisation:**
```bash
PYTHONPATH=src python scripts/create_train_test_val_splits.py \
  --input data/processed/btc_features.parquet \
  --output-dir data/splits \
  --train-pct 0.70 \
  --test-pct 0.20 \
  --val-pct 0.10
```

**Résultat:**
- `train.parquet` (70%, première partie)
- `test.parquet` (20%, milieu)
- `val.parquet` (10%, fin)

**Important:** Chronologique, pas random shuffle

**Durée:** 1-2 min

---

## 🤖 Entraînement

### `train_parallel_agents.py`

Entraîne 4 workers PPO avec Population-Based Training.

**Utilisation - Sandbox (rapide):**
```bash
PYTHONPATH=src python scripts/train_parallel_agents.py \
  --mode sandbox \
  --steps 5000
```

**Utilisation - Heavy (production):**
```bash
PYTHONPATH=src python scripts/train_parallel_agents.py \
  --mode heavy \
  --steps 500000 \
  --num-cpus 8 \
  --num-samples 4 \
  --checkpoint-dir checkpoints/prod_run
```

**Paramètres clés:**
- `--mode`: sandbox (CPU) ou heavy (GPU)
- `--steps`: Nombre total de steps
- `--num-cpus`: CPUs à utiliser
- `--num-samples`: Nombre de workers (4 recommandé)
- `--checkpoint-dir`: Où sauver les checkpoints
- `--checkpoint-freq`: Sauvegarde tous les N steps (default: 5000)

**Workers:**
- W1 Scalper: 5m, gamma=0.95
- W2 Intraday: 1h, gamma=0.99
- W3 Swing: 4h, gamma=0.995
- W4 Position: 4h, gamma=0.999

**Résultat:**
- `checkpoints/prod_run/checkpoint_*/` (fichiers modèle)
- `logs/adan_trading_bot.json` (metrics)
- `logs/metrics/*.jsonl` (trade records)

**Durée:**
- Sandbox: 5 min (CPU)
- 50K steps: 12 hours (GPU)
- 500K steps: 5 days (GPU)

---

## 📊 Backtest & Validation

### `deterministic_backtest.py`

Évalue le modèle sur un split de données (train/test/val).

**Utilisation:**
```bash
PYTHONPATH=src python scripts/deterministic_backtest.py \
  --checkpoint checkpoints/prod_run/checkpoint_500000 \
  --split test \
  --output results/backtest_test.json
```

**Paramètres:**
- `--checkpoint`: Chemin au modèle entraîné
- `--split`: train, test, ou val
- `--output`: Fichier JSON résultat
- `--verbose`: Affiche détails trades

**Résultat:** JSON avec stats:
```json
{
  "total_trades": 156,
  "win_rate": 52.3,
  "sharpe_ratio": 2.1,
  "sortino_ratio": 3.2,
  "max_drawdown": 18.5,
  "profit_factor": 1.8,
  "total_return": 12.4,
  "realized_pnl": 254.50,
  "...": "..."
}
```

**Usage courant:**
```bash
# Sanity check: train
python scripts/deterministic_backtest.py --checkpoint ... --split train

# OOS evaluation: test
python scripts/deterministic_backtest.py --checkpoint ... --split test

# Final validation: val
python scripts/deterministic_backtest.py --checkpoint ... --split val
```

**Durée:** 2-5 min

---

### `checkpoint_manager.py`

Gère les checkpoints (validation, listing, cleanup).

**Utilisation:**
```bash
# Lister les checkpoints
python scripts/checkpoint_manager.py list --dir checkpoints/prod_run

# Valider un checkpoint
python scripts/checkpoint_manager.py validate --checkpoint checkpoints/prod_run/checkpoint_500000

# Copier le best checkpoint
python scripts/checkpoint_manager.py copy \
  --source checkpoints/prod_run/checkpoint_500000 \
  --dest models/best_model
```

**Résultat:** Diagnostic checkpoint ou copie

**Durée:** Instantané

---

### `verify_checkpoint_config.py`

Vérifie qu'un checkpoint est valide et conforme.

**Utilisation:**
```bash
PYTHONPATH=src python scripts/verify_checkpoint_config.py \
  --checkpoint checkpoints/prod_run/checkpoint_500000
```

**Vérifie:**
- ✅ Config cohérente
- ✅ Poids chargés
- ✅ Normalization stats OK
- ✅ Policy callable

**Résultat:** PASS/FAIL

**Durée:** 1 sec

---

### `verify_checkpoint_resume.py`

Teste la reprise d'entraînement.

**Utilisation:**
```bash
PYTHONPATH=src python scripts/verify_checkpoint_resume.py \
  --checkpoint checkpoints/prod_run/checkpoint_500000 \
  --test-steps 100
```

**Vérifie:**
- ✅ State dict loadable
- ✅ Optimizer state preserved
- ✅ Training resume sans crash

**Résultat:** PASS/FAIL

**Durée:** 1-2 min

---

### `test_pnl_flow.py`

Teste le flux PnL (calculations, positions, equity).

**Utilisation:**
```bash
PYTHONPATH=src python scripts/test_pnl_flow.py \
  --checkpoint checkpoints/prod_run/checkpoint_500000 \
  --steps 100
```

**Vérifie:**
- ✅ PnL math correct
- ✅ Pas de NaN/Inf
- ✅ Positions tracked
- ✅ Equity updates cohérentes

**Résultat:** Test suite result

**Durée:** 2-5 min

---

## 🌐 Trading en Simulation

### `paper_trading_monitor.py`

Simule le trading réel sans capital.

**Utilisation:**
```bash
PYTHONPATH=src python scripts/paper_trading_monitor.py \
  --checkpoint checkpoints/prod_run/checkpoint_500000 \
  --exchange bitget \
  --mode paper \
  --duration 2weeks \
  --initial-capital 1000
```

**Paramètres:**
- `--checkpoint`: Modèle à tester
- `--exchange`: bitget, binance, etc.
- `--mode`: paper (sim) ou live (réel)
- `--duration`: 1week, 2weeks, 1month
- `--initial-capital`: Montant simulation

**Résultat:**
- Live trades sur paper account
- Logs détaillés dans `logs/paper_trading_log.json`
- Métriques en temps réel

**Durée:** Selon duration (2 semaines = 2 semaines)

---

## 🔴 Trading Réel

### `run_bot.py`

Lance le bot de trading en direct.

**Utilisation:**
```bash
PYTHONPATH=src python scripts/run_bot.py \
  --checkpoint checkpoints/prod_run/checkpoint_500000 \
  --exchange bitget \
  --mode live \
  --capital 100 \
  --max-positions 4 \
  --max-dd-stop 20
```

**Paramètres:**
- `--checkpoint`: Modèle
- `--exchange`: Exchange
- `--mode`: live (capital réel)
- `--capital`: Montant initial ($)
- `--max-positions`: Max positions simultanées
- `--max-dd-stop`: Arrêt si DD > X%

**Résultat:**
- Trades en direct
- Logs en temps réel
- Monitoring continu

**Prérequis:**
- [ ] Paper trading 2 semaines OK
- [ ] Backtest OOS: Sharpe > 1.5
- [ ] Credentials validées
- [ ] Monitoring prêt

---

## 🎓 Oracle & Régimes

### `train_oracle.py`

Entraîne un modèle HMM pour détecter les régimes de marché.

**Utilisation:**
```bash
PYTHONPATH=src python scripts/train_oracle.py \
  --data data/processed/btc_features.parquet \
  --output models/oracle.pkl \
  --states 3
```

**Paramètres:**
- `--data`: Parquet avec features
- `--output`: Chemin modèle PKL
- `--states`: Nombre régimes (3-4 típico)

**Régimes détectés:**
- State 0: Uptrend (risque bas)
- State 1: Range/Mixed (risque moyen)
- State 2: Downtrend (risque haut)

**Utilisé par:** Dynamic Behavior Engine pour adapter SL/TP

**Résultat:** `models/oracle.pkl` (3-5 MB)

**Durée:** 5-10 min

---

## 📈 Monitoring & Logs

### `live_monitor.py`

Affiche les metrics en temps réel.

**Utilisation:**
```bash
PYTHONPATH=src python scripts/live_monitor.py \
  --mode training \
  --checkpoint-dir checkpoints/prod_run \
  --update-freq 10s
```

**Résultat:** Dashboard live avec:
- Sharpe, Sortino, Win Rate
- Portfolio value, Drawdown
- Trades par seconde
- Errors/warnings

**Fonctionne avec:**
- Training (lit les logs)
- Paper trading
- Live trading

---

## 🧪 Tests Rapides

### `smoke_test.py`

Test rapide complet du système.

**Utilisation:**
```bash
python scripts/smoke_test.py
```

**Teste:**
- ✅ Data loading
- ✅ Features OK
- ✅ Model forward pass
- ✅ Training step
- ✅ Backtest

**Résultat:** PASS/FAIL (5 min)

**Durée:** ~5 minutes

---

## 🎯 Décider Quel Script Utiliser

| Tâche | Script | Temps |
|-------|--------|-------|
| Test rapide | `smoke_test.py` | 5 min |
| DL données | `download_ccxt_data.py` | 20 min |
| Ajouter features | `compute_features_real.py` | 10 min |
| Créer splits | `create_train_test_val_splits.py` | 2 min |
| Entraîner (rapide) | `train_parallel_agents.py --mode sandbox` | 5 min |
| Entraîner (production) | `train_parallel_agents.py --mode heavy` | 5 days |
| Backtest | `deterministic_backtest.py` | 5 min |
| Valider checkpoint | `verify_checkpoint_config.py` | 1 sec |
| Tester PnL | `test_pnl_flow.py` | 2 min |
| Entraîner oracle | `train_oracle.py` | 10 min |
| Paper trading | `paper_trading_monitor.py` | 2 weeks |
| Live trading | `run_bot.py` | Ongoing |
| Monitor live | `live_monitor.py` | Real-time |

---

## 🆘 Scripts Supprimés (v2.0)

Les scripts suivants ont été supprimés (deprecated):
- ❌ Old diagnostic scripts (sessions S11-S15)
- ❌ Old download scripts (remplacés par `download_ccxt_data.py`)
- ❌ Old backtest engines (remplacés par `deterministic_backtest.py`)
- ❌ Old test suites (functionality in `smoke_test.py`)

Voir `docs/CHANGES.md` pour la liste complète.

---

**Question?** Voir `docs/GUIDE_UTILISATEUR.md` ou ouvrir issue.

