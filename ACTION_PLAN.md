# ACTION PLAN - ADAN0 PBT Training

## 🎯 Objectif
Optimiser les 4 workers pour atteindre une performance stable et profitable avant déploiement en production.

---

## 📋 ANALYSE ACTUELLE

### ✅ Ce qui fonctionne
- **Swing Worker**: +$21.95 (1.49x profit factor) - EXCELLENT
- **Intraday Worker**: +$6.41 (1.10x profit factor) - BON
- **Scalper Worker**: +$2.35 (1.07x profit factor) - ACCEPTABLE
- **Système global**: +$20.48 PnL (tous les mécanismes actifs)

### ❌ Ce qui ne fonctionne pas
- **Position Worker**: -$10.23 (0.88x profit factor) - PROBLÉMATIQUE
- **Volatilité excessive**: Scalper/Intraday varient de $14 à $230
- **Taux victoire faible**: Tous < 50% (sauf Scalper 46.7%)

---

## 🔧 ACTIONS IMMÉDIATES (Aujourd'hui)

### 1. Continuer l'entraînement
```bash
cd ~/Documents/trading/ADAN0-main
bash restart_training.sh
```
- Cible: 500K+ steps (actuellement ~250K)
- Durée estimée: 8-12 heures
- Objectif: Convergence et stabilisation

### 2. Monitorer les workers
```bash
bash monitor_workers.sh
```
- Vérifier que tous les 4 workers progressent
- Surveiller Swing (meilleur performer)
- Détecter tout crash ou anomalie

### 3. Vérifier TensorBoard
```bash
tensorboard --logdir /mnt/new_data/t10_training/tb_workers --reload_interval 30
```
- Accès: http://localhost:6006
- Vérifier que Position worker forme une courbe (pas bloqué)
- Comparer les trajectoires des 4 workers

---

## 🎛️ OPTIMISATIONS À COURT TERME (Prochaines 24h)

### 1. Réduire l'agressivité de Position Worker
**Problème**: Ouvre 1084 trades (2x plus que les autres), perd $10.23

**Solution**:
```yaml
# config/config.yaml - Position worker
position_worker:
  max_trades_per_episode: 500  # Réduire de 1084 à 500
  position_size_multiplier: 0.5  # Réduire de 50%
  action_threshold: 0.6  # Augmenter de 0.5 (moins de trades)
```

### 2. Augmenter la sélectivité des trades (EV-Gate)
**Problème**: Taux victoire faible (< 50%), beaucoup de mauvais trades

**Solution**:
```yaml
# config/config.yaml - EV-Gate
ev_gate:
  p_min_required: 0.55  # Augmenter de 0.50 (plus sélectif)
  fee_pct: 0.002  # Déjà correct
```

### 3. Stabiliser Scalper et Intraday
**Problème**: Volatilité excessive ($14-$230), capital instable

**Solution**:
```yaml
# config/config.yaml - Risk Management
scalper:
  position_size_pct: 5  # Réduire de 10%
  max_concurrent_positions: 2  # Réduire de 3
  
intraday:
  position_size_pct: 5  # Réduire de 10%
  max_concurrent_positions: 1  # Réduire de 2
```

---

## 📊 MÉTRIQUES DE SUCCÈS

### Cibles à atteindre (après optimisations)
| Métrique | Actuel | Cible | Worker |
|----------|--------|-------|--------|
| PnL | +$20.48 | +$50+ | Global |
| Win Rate | 39% | 50%+ | Tous |
| Profit Factor | 1.10x | 1.5x+ | Tous |
| Volatilité | $215 | $50 | Scalper/Intraday |
| Position PnL | -$10.23 | +$5+ | Position |

---

## 🚀 DÉPLOIEMENT EN PRODUCTION (Semaine prochaine)

### Phase 1: Paper Trading (Swing Worker)
```bash
# Déployer Swing worker en paper trading
python scripts/paper_trading_monitor.py \
  --worker swing \
  --capital 1000 \
  --duration 7days
```

**Critères de succès**:
- PnL positif sur 7 jours
- Win rate > 50%
- Profit factor > 1.5x
- Pas de crash

### Phase 2: Live Trading (Swing Worker)
```bash
# Déployer Swing worker en live trading
python scripts/live_trading.py \
  --worker swing \
  --capital 100 \
  --risk-per-trade 1%
```

**Critères de succès**:
- PnL positif sur 30 jours
- Sharpe ratio > 1.5
- Max drawdown < 20%

### Phase 3: Ajouter Intraday (si Swing réussit)
```bash
# Ajouter Intraday worker après 30 jours de Swing
python scripts/live_trading.py \
  --workers swing,intraday \
  --capital 200 \
  --risk-per-trade 1%
```

---

## 📈 TIMELINE

| Date | Action | Durée | Responsable |
|------|--------|-------|-------------|
| 2026-04-04 | Continuer entraînement | 8-12h | Auto |
| 2026-04-05 | Analyser résultats | 2h | Manuel |
| 2026-04-05 | Appliquer optimisations | 1h | Manuel |
| 2026-04-05 | Relancer entraînement | 8-12h | Auto |
| 2026-04-06 | Vérifier convergence | 2h | Manuel |
| 2026-04-07 | Déployer Swing en paper | 7 jours | Auto |
| 2026-04-14 | Décision live trading | 1h | Manuel |

---

## 🔍 MONITORING CONTINU

### Logs à surveiller
```bash
# Position closures
tail -f /mnt/new_data/t10_training/logs/training.log | grep "POSITION FERMÉE"

# SL/TP triggers
tail -f /mnt/new_data/t10_training/logs/training.log | grep -E "STOP LOSS|TAKE PROFIT"

# Errors
tail -f /mnt/new_data/t10_training/logs/training.log | grep -E "ERROR|CRITICAL"
```

### Métriques à vérifier
- ✅ Tous les workers actifs (4 PIDs)
- ✅ Trades ouverts/fermés en augmentation
- ✅ PnL global positif
- ✅ Pas de crash ou exception
- ✅ SL/TP se déclenchent régulièrement

---

## 🛑 POINTS D'ARRÊT (Kill Switches)

Arrêter l'entraînement si:
1. **PnL global devient négatif** (-$50+)
2. **Crash répété** (> 3 fois en 1h)
3. **Taux victoire < 30%** (tous les workers)
4. **Capital tombe à $10** (trop bas)
5. **Erreur non-gérée** dans les logs

---

## 📞 ESCALADE

Si problème:
1. Vérifier les logs: `tail -100 /mnt/new_data/t10_training/logs/training.log`
2. Vérifier les workers: `ps aux | grep ray::ADAN_PBT_Worker`
3. Vérifier TensorBoard: http://localhost:6006
4. Redémarrer si nécessaire: `bash restart_training.sh`

---

## ✅ CHECKLIST AVANT PRODUCTION

- [ ] Swing worker: PnL > +$50, Win rate > 50%, Profit factor > 1.5x
- [ ] Intraday worker: PnL > +$20, Win rate > 45%, Profit factor > 1.2x
- [ ] Scalper worker: PnL > +$10, Win rate > 45%, Profit factor > 1.1x
- [ ] Position worker: PnL > +$5 (ou désactiver)
- [ ] Volatilité < $100 pour tous
- [ ] 500K+ steps complétés
- [ ] 7 jours de paper trading réussis
- [ ] Pas de crash en 24h
- [ ] Tous les mécanismes (Anti-Hack, EV-Gate, Kelly, ATR-SL) actifs

---

## 🎓 CONCLUSION

**Statut**: ⚠️ EN COURS D'OPTIMISATION

- Swing worker est prometteur → continuer
- Position worker problématique → réduire agressivité
- Scalper/Intraday volatiles → réduire position size
- Système stable → prêt pour extended training

**Prochaine étape**: Continuer entraînement à 500K+ steps avec optimisations appliquées.
