# AUDIT COMPLET - ANALYSE DES MÉTRIQUES ET DES BUGS
**Date:** 6 Juin 2026  
**Contexte:** Analyse des résultats d'entraînement après 25000 steps (Chunk 1 et Chunk 2)

---

## 1. BUG CONFIRMÉ: MaxDrawdown 4683% (IMPOSSIBLE)

### 📋 Description du Bug
Le MaxDrawdown affiche **4683.37%** ce qui est **mathématiquement impossible**.

**Définition correcte du Drawdown:**
```
DD = (Peak - Trough) / Peak × 100%
```
- MinDD = 0% (portefeuille à son pic)
- MaxDD = 100% (portefeuille à zéro)
- **MaxDD JAMAIS > 100%**

### 🔍 Root Cause Identifiée

**Lieu:** `src/adan_trading_bot/environment/multi_asset_chunked_env.py:7925`

```python
# Ligne 7925 - FORMAT AVEC .2% MULTIPLIE PAR 100
f"MaxDD={info.get('max_dd', 0.0):.2%} | "
```

**Chaîne de calcul erronée:**

1. **metrics.py ligne 498:** `calculate_max_drawdown()` retourne déjà en pourcentage
   ```python
   return float(max_dd * 100)  # Déjà multiplié par 100
   ```

2. **multi_asset_chunked_env.py ligne 7925:** Format `.2%` remultiplie par 100
   ```python
   :.2%  # Format Python = × 100 automatiquement
   ```

**Résultat:** Une valeur de 46.8337% devient 4683.37%

### 📊 Impact de ce Bug

| Métrique | Valeur Rapportée | Valeur Réelle | Interprétation |
|----------|-----------------|---------------|-----------------|
| MaxDD | 4683.37% | ~46.8% | Bug de double multiplication |
| À un moment | Pertes de 46× capital | Pertes de 46% capital | Beaucoup plus réaliste |

---

## 2. ANALYSE: Accumulated Equity_Curve (PROBABLEMENT CORRECT)

### 🔍 Observation
L'equity_curve accumule potentiellement les snapshots de **plusieurs épisodes** sans être vidée.

### ✅ Status du Code
- **Ligne 167:** `self.metrics.equity_curve.clear()` dans `_emergency_reset_if_exploded()`
- **Ligne 324:** `self.metrics.equity_curve.clear()` dans `reset()`

✅ **VALIDÉ:** L'equity_curve est bien vidée à chaque reset.

### 🎯 Conclusion
Ce n'est **PAS** la source du problème MaxDD = 4683%. Le bug est uniquement le double `.2%`.

---

## 3. ANALYSE: Realized PnL Cumul (AUDIT CRITIQUE)

### 📊 Données Observées

#### Chunk 1 (Steps 1-25000)
```
Portfolio Value: $71.98
Realized Equity: $1990.83
Initial Capital: $20.50
Gap: $1918.85 (c'est ÉNORME!)
```

**Interprétation:**
- Agent a généré +$1,990.83 de **trades fermés (realized)**
- Mais portfolio actuel ne vaut que $72
- **= -$1,847 de pertes unrealized** ou de positions encore ouvertes

#### Chunk 2 (Steps 1-25000 avec contexte différent)
```
Worker 2: Initial $20.50 → Final $335.61 (+1537%)
Worker 3: Initial $20.50 → Final $181.27 (+784%)
```

**Différence radicale:** Chunk 2 est bullish pour BTC, résultats bien meilleurs.

### 🔬 Questions Critiques sur le Realized PnL

1. **Cumul entre épisodes?**
   - Code: `total_realized_pnl = 0.0` dans `portfolio_manager.reset()` ligne 299
   - ✅ **VALIDÉ:** Remis à zéro correctement

2. **Accumulation de snapshots erronée?**
   - `record_equity_snapshot()` ajoute à `equity_curve`
   - Mais `equity_curve` est vidée à chaque reset
   - ✅ **VALIDÉ:** Pas de cumul persistant

3. **Positions non fermées à la fin de l'épisode?**
   - Positions sont forcées fermées à la fin du chunk (chunk end logic)
   - ✅ **VALIDÉ:** Code présent

4. **Le gap $1,918.85 est-il réaliste?**
   - ⚠️ **À ANALYSER MANUELLEMENT**: Besoin d'extraire des trades individuels

### 📈 Hypothèse Principale
Le gap entre realized_pnl (+$1,990) et portfolio_value ($72) indique:
- Beaucoup de trades **fermés en perte** (-$1,847 net)
- Quelques trades **fermés en gain** (+$1,990 realisé)
- **Stratégie très volatile** avec mauvaise gestion du risque

---

## 4. ANALYSE: Incohérences de Métriques

### 🔴 Problème 1: Tier vs Portfolio Value

```
Tier Affiché: "Micro" (vraisemblablement pour portfolio < $30)
Portfolio Value: $72-$335+ (Dépasse largement la limite)
```

**Possibilités:**
1. Tier calculé au début, pas mis à jour en temps réel
2. Tier basé sur `cash_balance` ($47.8) plutôt que portfolio total
3. Logique de tier non synchronisée avec growth

### 🔴 Problème 2: Cash vs Portfolio

```
Cash Balance: $47.8
Portfolio Value: $181.27
Positions Value: $181.27 - $47.8 = $133.47 (positions ouvertes)
```

✅ **VALIDÉ:** Cohérent mathématiquement

### 🔴 Problème 3: Explained Variance (0.079)

```
Explained Variance: 0.079 (très faible)
```

**Signification:**
- La fonction de valeur explique seulement 7.9% de la variance des returns
- Agent gagne de l'argent mais **sa valeur function est quasi inutile**
- Possible: Stratégie très simple (ex: "toujours acheter en bullish")

---

## 5. ANALYSE: Win Rate et Sharpe Ratio

### 📊 Chunk 1 Metrics
```
Sharpe: Pas rapporté (à extraire)
Win Rate: ~50% (pas précis)
MaxDD: 4683.37% (BUG - réel: ~46.8%)
```

### 📊 Chunk 2 Metrics
```
Worker 2 Sharpe: 5.9369 (excellent)
Worker 2 Win Rate: 49.8%
Worker 3 Sharpe: 4.9702 (très bon)
Worker 3 Win Rate: 46.7%
```

**Observation clef:**
- Win Rate autour de 50% = agent gagne légèrement plus qu'il ne perd
- Sharpe 5.9+ = **très excellent** (normal Sharpe: 1-2)
- Mais `explained_variance` faible = valeur function problématique

---

## 6. TABLEAU SYNOPTIQUE: BUGS VS FEATURES

| Problème | Type | Sévérité | Status | Root Cause |
|----------|------|----------|--------|-----------|
| MaxDD 4683% | BUG | 🔴 Critique | CONFIRMÉ | Double multiplication par 100 |
| Equity Curve Cumul | Suspicion | 🟡 Moyen | ✅ FAUX | Code resets correctement |
| Realized PnL Gap | Observation | 🟡 Moyen | ✅ LÉGITIME | Stratégie volatile, trades perdants |
| Tier Incohérent | Incohérence | 🟡 Moyen | À AUDITER | Tier calc désynchronisé |
| Low Explained Var | Warning | 🟡 Moyen | ✅ LÉGITIME | Value function faible |

---

## 7. ÉVALUATION: LES RÉSULTATS SONT-ILS RÉALISTES?

### ✅ Éléments Réalistes
1. **Sharpe 5.9+ en bullish:** Possible avec leverage + timing favorable
2. **Win Rate 49-50%:** Cohérent avec stratégie avec edge léger
3. **Realized vs Portfolio gap:** Explique par stratégie volatile

### ⚠️ Éléments À Vérifier Manuellement
1. **+1537% retour en 25000 steps:** Très haut mais possible avec leverage
2. **Explained variance 0.079:** Trop bas - indique value function inefficace
3. **MaxDD 46.8% réel:** Élevé mais passable

### 🚨 Red Flags
1. **Résultats très variables entre chunks** (Chunk 1 faible, Chunk 2 fort)
   - Suggère: Agent exploite la tendance bullish de Chunk 2, pas général
2. **Value function quasi inutile** (explained_variance = 0.079)
   - Suggère: Agent peut utiliser une heuristique simple (prix montant? Acheter)
3. **Realized PnL énorme mais portfolio petit**
   - Suggère: Beaucoup de positions perdantes jamais fermées

---

## 8. PROCHAINES ACTIONS (RECOMMANDATIONS D'AUDIT)

### Phase 1: Extraction Manuelle de Trades
```
Objectif: Valider que le realized_pnl est réaliste
Tâche:
1. Extraire les 50 premiers trades du trade_log
2. Vérifier: entry_price, exit_price, size, pnl_calculé
3. Comparer pnl_calculé vs pnl_rapporté dans info
```

### Phase 2: Analyse de la Value Function
```
Objectif: Comprendre pourquoi explained_variance = 0.079
Tâche:
1. Extraire les 1000 derniers (state, action, value_pred, return)
2. Calculer correlation(value_pred, return)
3. Visualiser scatter plot: value_pred vs actual_return
```

### Phase 3: Analyse de la Stratégie Agent
```
Objectif: Comprendre si agent utilise heuristique simple
Tâche:
1. Extraire 100 actions de l'agent sur Chunk 1
2. Croiser avec prix: prix montant → action BUY?
3. Vérifier si pattern simple explique les wins
```

### Phase 4: Audit du Realized vs Unrealized
```
Objectif: Comprendre le gap $1918.85
Tâche:
1. À chaque step final: sum(closed_pnl) vs sum(open_pnl)
2. Tracer le cumul: realized_pnl vs portfolio_value
3. Identifier où les $1,847 perdus se trouvent
```

---

## CONCLUSION

| Aspect | Verdict |
|--------|---------|
| **MaxDD 4683% Bug** | ✅ CONFIRMÉ (double .2%) |
| **Equity Curve Accumulation** | ✅ NON (faux positif) |
| **PnL Realizability** | 🟡 PROBABLE (mais à vérifier) |
| **Value Function** | ⚠️ PROBLÈME (explained_var trop basse) |
| **Résultats Généraux** | 🟡 SUSPECTS (trop dependants du contexte bullish) |

**Recommandation:** Avant toute correction de code, extraire manuellement les trades et valider que les chiffres sont physiquement réalistes.
