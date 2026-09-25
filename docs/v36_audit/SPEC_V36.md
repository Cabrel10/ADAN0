# SPÉCIFICATION V36 — définir CE QUE le reward doit apprendre

Date : 2026-08-23
Fondé sur : `DIAGNOSTIC_V35.md` (données réelles run 500k).
Principe directeur : **ne pas ajouter de règles ; re-hiérarchiser + isoler pour tester.**

---

## 1. LA DÉFINITION MINIMALE (le contrat manquant)

Avant tout paramètre, on fixe la grandeur unique que le **critic** doit approximer :

> **V(s) ≈ espérance de la variation future de la richesse nette du portefeuille
> (equity = cash + non-réalisé), nette de frais, exprimée en % du capital initial.**

Tout terme du reward doit être justifié comme :
- (A) **partie du signal financier** (variation d'equity, frais) → cœur, doit dominer ;
- (B) **contrainte de risque** bornée qui protège la survie (drawdown) → secondaire, plafonnée ;
- (C) **shaping pédagogique** à somme quasi-nulle et magnitude PLAFONNÉE bien SOUS le signal (A).

Un terme qui n'entre dans aucune catégorie est **retiré**, pas gardé "au cas où".

---

## 2. CE QUE V35 A VIOLÉ (rappel chiffré)

| Terme | Part V35 | Catégorie | Verdict |
|-------|--------:|-----------|---------|
| symmetry_penalty | 32.8% | C (forme) mais NÉG. pur | **viole** : plus gros que A, jamais positif |
| future_contrib | 22.4% | C (pédagogique) | **viole** : magnitude > A |
| drawdown_penalty | 20.5% | B (risque) | limite : ok si borné, mais > A |
| **pnl_reward** | **13.1%** | **A (signal)** | **doit être #1, il est #4** |
| closure_bonus | 9.9% | C | à réduire |

Cible V36 : **A (pnl) ≥ 50% de l'amplitude**, B ≤ 25%, chaque terme C ≤ 10%.

---

## 3. PLAN D'ABLATION (3 bras, config-only, MÊME architecture 1.7M)

Aucun changement de code pour les bras 1-2 : tout via `config/config.yaml`.
Runs COURTS (50k steps) pour comparer, PAS 500k. Seul le gagnant ira à 500k.

### V36-A — "Finance pure" (bras de contrôle)
Objectif : prouver que l'agent PEUT apprendre la rentabilité quand le signal domine.
```yaml
reward_shaping.future_reward.enabled: false      # FA hors reward (télémétrie ok)
trading_rules.symmetry_enforcement.enabled: false
trading_rules.close_intention_penalty.enabled: false
# garder: pnl_reward, drawdown_penalty (borné), frais. C'est tout.
```
Hypothèse : EV se stabilise POSITIF, reward des gains devient POSITIF, ratio SL/TP s'améliore.

### V36-B — "Finance + Future Arena borné"
Comme A, mais FA réactivé AVEC plafond fortement réduit pour rester sous le PnL :
```yaml
reward_shaping.future_reward.enabled: true
reward_shaping.future_reward.max_future_contrib: 0.15   # était 0.60
trading_rules.symmetry_enforcement.enabled: false
```
Hypothèse : si B ≈ A → FA neutre ; si B < A → FA nuit même borné ; si B > A → FA utile borné.

### V36-C — "Symmetry réconcilié avec free_sltp"
Comme A + symmetry réactivé mais NON contradictoire avec SL/TP libres :
```yaml
trading_rules.symmetry_enforcement.enabled: true
trading_rules.symmetry_enforcement.rr_tolerance: 1.5    # zone morte RR∈[0,3], n'écrase pas RR≈2
trading_rules.symmetry_enforcement.max_step_penalty: 0.03  # était 0.15
```
Hypothèse : teste si une pénalité de forme LÉGÈRE aide sans dominer.

---

## 4. MÉTRIQUES DE COMPARAISON (mêmes pour les 3 bras)

Sur les 50k steps de chaque bras, extraire du log + jsonl :
1. **explained_variance** finale (moyenne 5 derniers buckets) — stabilité critic.
2. **reward.total_mean des trades GAGNANTS** — doit devenir POSITIF (V35 = -0.137).
3. **ratio SL_HIT / TP_HIT** — V35 = 6:1, viser < 3:1.
4. **PnL cumulé** — V35 = -187$ ; viser > V35.
5. **part de pnl_reward dans l'amplitude** — viser ≥ 50%.
6. **a0_mean (biais BUY/SELL)** et **std** (exploration) — V35 = -0.415 / 1.06.
7. **corr(reward.total, realized_pnl)** — doit rester ≥ 0.8.

Règle GO/NO-GO (mandat) : un bras n'est PASS qu'après un run réel terminé + métriques,
jamais "process alive".

---

## 5. GARDE-FOUS (non négociables)
- Checkpoint V35 (`ppo_adan0_v35_500k.zip`) reste FIGÉ comme référence, jamais écrasé.
- Un seul paramètre-famille change par bras (isolation causale).
- Architecture identique (CNN+attention+FiLM, ~1.7M) sur les 3 bras : la seule variable
  expérimentale est le CHEMIN DU SIGNAL, pas la capacité.
- Backup config avant modif. Chaque bras = un fichier config dédié + un tag git.

---

## 6. ORDRE D'EXÉCUTION
1. Créer 3 configs dérivées (`config_v36a.yaml`, `_v36b`, `_v36c`) par override du bloc reward.
2. Smoke test 2k steps chacun (vérifier que ça démarre + reward hiérarchie attendue via jsonl).
3. Run 50k chacun (séquentiel, 1 GPU/CPU).
4. Audit comparatif (réutiliser `audit_reward_causal.py` sur chaque jsonl).
5. GO/NO-GO → le gagnant part à 500k.
