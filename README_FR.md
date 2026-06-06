# ADAN0 — Assistant Digital Autonome pour la Navigation d'Actifs

Bot de trading BTC/USDT basé sur l'apprentissage par renforcement PPO.

---

## Statut Actuel (S16 - Hysteresis Tier + PBT Entreprise, 2026-06-06)

**ENTRAÎNEMENT EN COURS**: Multi-worker PBT avec système de reward basé sur les tiers.

| Métrique | Valeur | Status |
|----------|--------|--------|
| Steps d'entraînement | 8,192 | ✅ En progression (cible: 500K+) |
| Workers actifs | 4 | ✅ Ray distribué |
| Taux de gain (récent) | 52.99% | ✅ Au-dessus du seuil rentable |
| Ratio de Sharpe | 6.18 | ✅ Excellent |
| Ratio de Sortino | 9.52 | ✅ Protection forte contre les baisses |
| Valeur du portefeuille | $37.05 | ✅ Préservation du capital |
| Shaping de reward | ZÉRO | ✅ Signal financier pur |

**Réalisations (S16)**:
- **Hysteresis tier**: 4 niveaux de progression du capital (Micro→Petit→Moyen→Large)
- Bonus de promotion **10x** + pénalités de stagnation soft pour comportement réaliste
- **Pipeline PBT Entreprise**: Sync VecNormalize, support gSDE, vérification d'intégrité
- Profils spécifiques: Scalper (5m), Intraday (1h), Swing (4h), Position (1d)
- Correction du bug MaxDD (double multiplication ×100)
- Suivi PnL corrégé (pas de double-comptage)
- Gestion des risques par tier: SL/TP dynamiques

**Développement actif**:
- PBT multi-worker: 4 agents évoluant indépendamment
- Conformité de fréquence: Limites de positions par timeframe appliquées
- AGENT_CLOSE: ~50% des trades fermés par décision (amélioration vs 40% MaxDuration)
- Sync des métriques temps réel: Sharpe, Sortino, taux de gain, PnL non réalisé

---

## Architecture

```
4 Workers (PBT - Population-Based Training)
  W1 Scalper:   5m  | gamma=0.95  | n_steps=512
  W2 Intraday:  1h  | gamma=0.99  | n_steps=2048
  W3 Swing:     4h  | gamma=0.995 | n_steps=8192
  W4 Position:  4h  | gamma=0.999 | n_steps=16384

Espace d'observation: Dict{5m:(20,21), 1h:(20,21), 4h:(20,21), context:(17,), portfolio:(20,)}
Espace d'action: Box(-1,1,(5,)) [direction, taille%, tf_pref, sl%, tp%]
Extracteur: ContextualTemporalFusionExtractor (CNN + Attention)
Exploration: gSDE, log_std_init=-0.5
```

---

## Démarrage Rapide

### GPU (Colab/Kaggle)

Ouvrir `notebooks/ADAN_Full_Training_H100.ipynb` dans Colab avec H100/A100.

Le notebook gère tout:
1. Clone + installation
2. Téléchargement données BTC (6 ans via CCXT)
3. 21 indicateurs techniques par timeframe
4. Splits train/test/val (70/20/10)
5. Entraînement 4 workers + Ray Tune PBT (500K+ steps)
6. Backtest OOS déterministe
7. Export pour paper trading

### Local (Test rapide)

```bash
# Installation
pip install -e .

# Test 5000 steps (CPU, ~5 min)
PYTHONPATH=src python scripts/train_parallel_agents.py --mode sandbox --steps 5000

# Backtest OOS
PYTHONPATH=src python scripts/deterministic_backtest.py --steps 500 --split test
```

### Entraînement lourd (GPU local)

```bash
PYTHONPATH=src python scripts/train_parallel_agents.py \
  --mode heavy \
  --steps 500000 \
  --num-cpus 8 \
  --num-samples 4 \
  --profiles scalper,intraday,swing,position \
  --checkpoint-dir checkpoints/heavy
```

---

## Formule de Reward (S16 Final - Hysteresis Tier)

```
reward = symlog(raw) + bonus_tier
raw = pnl_net_scaled - frais_trade - penalite_dd + time_decay + bonus_survie

Où:
  pnl_net_scaled = (realized_pnl - commission*1.5) * 100 / capital_initial
  time_decay = -0.001 (steps sans trade)
  frais_trade = pct_commission * 100 / capital_initial
  penalite_dd = 50.0 * drawdown_pct^2 (quadratique)
  bonus_survie = +0.001/step (prévient strategies "impossible game")
  symlog(x) = sign(x) * ln(|x| + 1)

BONUS TIER (10x):
  Promotion Micro: +5.0
  Promotion Petit: +10.0
  Promotion Moyen: +20.0
  Promotion Large: +40.0

PÉNALITÉS STAGNATION (soft - ÷4):
  Tier 1: -0.005/step
  Tier 2: -0.010/step
  Tier 3: -0.015/step
  Tier 4: -0.020/step

ZÉRO shaping. Progression du capital pur + PnL réalisé.
```

---

## Structure du Projet

```
ADAN0/
  config/config.yaml             # Configuration centrale
  scripts/
    train_parallel_agents.py     # Entraînement principal
    deterministic_backtest.py    # Évaluation OOS
    download_ccxt_data.py        # Téléchargement données
    compute_features_real.py     # Indicateurs techniques
    create_train_test_val_splits.py  # Splits chronologiques
  src/adan_trading_bot/
    environment/
      multi_asset_chunked_env.py # Environnement RL
      reward_calculator.py       # Calcul du reward
      dynamic_behavior_engine.py # Moteur DBE
      exogenous_regime_oracle.py # Oracle HMM
    data_processing/
      data_loader.py             # Chargeur données
      state_builder.py           # Construction observations
      feature_engineer.py        # Indicateurs
    agent/
      feature_extractors.py      # CNN + Attention
    portfolio/
      portfolio_manager.py       # Gestion positions
  notebooks/
    ADAN_Full_Training_H100.ipynb  # Pipeline complet
  .github/workflows/
    adan0_relay.yml              # CI/CD
  docs/                           # Documentation complète
```

---

## Documentation

Tous les rapports d'analyse, diagnostics et sessions dans `docs/`:

- `ANALYSIS_COMPLETE_BUG_AUDIT.md` — Vérification intégrité données
- `METRICS_DEEP_DIVE.md` — Deep dives performance
- `TIER_BASED_REWARD_IMPLEMENTATION.md` — Implémentation hysteresis
- `SESSION_*_*.md` — Rapports sessions d'entraînement
- `COMPREHENSIVE_AUDIT_GUIDE.md` — Guide complet documentation

---

## Règles Critiques

1. **Pas de reward shaping** — Seulement PnL réalisé des trades fermés
2. **Pas de PnL non réalisé dans reward** — Viole Ng 1999
3. **Rapports honnêtes** — Si le modèle perd, on le dit
4. **explained_variance > 0** avant production
5. **2 semaines paper trading** minimum avant capital réel

---

## Déploiement

- **Exchanges**: Bitget, Binance (CCXT)
- **Mode**: Paper trading en premier
- **Capital**: Micro ($0-50) → Small → Medium → Large
- **Risque**: Détecteur Pareto, SL/TP dynamiques

---

**Status**: Développement actif | **Dernière mise à jour**: 2026-06-06 | **Commit**: S16 Tier Hysteresis + Enterprise PBT
