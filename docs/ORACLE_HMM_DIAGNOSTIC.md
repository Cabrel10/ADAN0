# ADAN0 — Diagnostic de l'Oracle HMM (audit, AUCUNE modification de code)

> Généré le 2026-06-22. Investigation empirique en réponse à la consigne :
> « diagnostiquer l'Oracle HMM — est-ce que le buffer HMM est réinitialisé à
> chaque épisode (ce qui ferait que les 500 observations minimales ne sont
> jamais atteintes) ? Ne rien modifier pour l'instant. Documenter uniquement
> ce qui est trouvé. »
>
> **Règle d'or appliquée : on ne suppose pas. On a lu le code complet, puis
> instrumenté le DBE en direct pour mesurer le comportement réel.**

---

## 0. Résumé exécutif (la cause racine)

L'Oracle HMM est **gelé à la valeur uniforme `[0.3333, 0.3333, 0.3333]`** sur
100 % des steps, dans CET environnement (le sandbox d'évaluation).

**La cause n'est PAS** un reset de buffer entre épisodes (hypothèse de départ).
**La cause EST** : le paquet Python `hmmlearn` **n'est pas installé** dans ce
sandbox, alors qu'il est bien déclaré dans `requirements.txt` (`hmmlearn>=0.3.0`).

Conséquence en chaîne :

```
hmmlearn absent
  → import échoue (try/except ImportError) → HMM_AVAILABLE = False   (dbe l.14-20)
  → _init_hmm() pose self._hmm_model = None                         (dbe l.52)
  → _update_hmm() : `if self._hmm_model is None: return uniform`     (dbe l.~365)
  → get_regime_probabilities() renvoie toujours [1/3, 1/3, 1/3]
  → ctx[3] (bull_prob) = 0.3333 constant
  → confidence = clip(0.3333) = 0.3333 constant                     (env l.6903-6904)
  → target_exposure_pct = exp_min + (exp_max-exp_min)*0.3333         (env l.6908)
     ⇒ TAILLE DE POSITION CONSTANTE, indépendante du marché ET du modèle.
```

---

## 1. Preuve empirique directe (probe instrumenté)

Script : `/tmp/probe_hmm_direct.py`. On instancie un `DynamicBehaviorEngine`
réel, on appelle `get_regime_probabilities()` 200 fois avec des observations
**volontairement variées** (régime haussier 0-70, range 70-130, baissier
130-200), et on trace la taille du buffer + les probabilités.

Sortie brute :

```
hmmlearn IMPORT FAILED: ModuleNotFoundError("No module named 'hmmlearn'")
HMM_AVAILABLE in module = False
N_HMM_STATES = 3 MIN_OBS = 60
buffer init len = 0 probs = [0.33333334 0.33333334 0.33333334]

=== RESULTS ===
buffer len: start=1 end=200            ← le buffer SE REMPLIT correctement
probs std per state: [4.17e-07 4.17e-07 4.17e-07]   ← variation nulle
probs at t=0   : [0.33333334 0.33333334 0.33333334]
probs at t=59  : [0.33333334 0.33333334 0.33333334]  ← avant MIN_OBS (attendu)
probs at t=65  : [0.33333334 0.33333334 0.33333334]  ← APRÈS MIN_OBS (anormal !)
probs at t=130 : [0.33333334 0.33333334 0.33333334]
probs at t=199 : [0.33333334 0.33333334 0.33333334]
GLOBALLY FROZEN at uniform? True
unique row count: 1
```

**Lecture critique :**

- Le buffer atteint bien 200 observations (> MIN_OBS=60 et même proche de
  HMM_WINDOW=500). **L'hypothèse « buffer vidé à chaque reset, jamais 60 obs »
  est donc RÉFUTÉE pour le flux continu.** En backtest, l'env utilise
  `reset_for_new_chunk(continuity=True)` (env l.3430/3458) qui **préserve** le
  buffer ; seul `reset()` (dbe l.2045) le vide, et il n'est pas appelé entre
  chunks d'un même run continu.
- Pourtant les probs restent figées à 1/3 **même après 60 obs**. La seule
  explication compatible avec le code : `self._hmm_model is None`, donc le bloc
  de fit (RobustScaler + LedoitWolf + jitter, dbe l.372-440) n'est **jamais
  exécuté**. Et `_hmm_model = None` parce que `HMM_AVAILABLE = False`.

---

## 2. Ce que le code prévoit (lecture complète, non supposée)

`dynamic_behavior_engine.py` :

```python
# l.14-20
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
```

```python
# _init_hmm(), l.~326-54
if HMM_AVAILABLE:
    self._hmm_model = GaussianHMM(n_components=N_HMM_STATES, ...)
    self._hmm_n_init = 10
else:
    self._hmm_model = None                       # ← ICI dans ce sandbox
self._hmm_probs = np.ones(N_HMM_STATES)/N_HMM_STATES   # = [1/3,1/3,1/3]
```

```python
# _update_hmm(), l.~360-366
self._hmm_obs_buffer.append([...])               # le buffer SE remplit
if self._hmm_model is None or len(buffer) < HMM_MIN_OBS:
    return self._hmm_probs.copy()                # ← sortie permanente ici
# (le vrai fit qui suit n'est jamais atteint)
```

Le code de fit lui-même est **sain et sophistiqué** (RobustScaler, features de
cumul de rendement scalées, LedoitWolf shrinkage, jitter anti-collinéarité 1e-2,
refit tous les 120 steps). Il n'y a **pas de bug logique** dans le fit ; il est
simplement court-circuité par l'absence de `hmmlearn`.

---

## 3. Chaîne d'appel vérifiée (qui nourrit le HMM)

- L'env appelle `self.dbe.get_regime_probabilities(market_data)` à
  **`multi_asset_chunked_env.py:5393`**, dans un `try/except Exception: pass`.
- `market_data` vient de `_get_current_market_data_for_hmm()` (env l.5431),
  fonction **robuste** (try interne, valeurs par défaut) — elle ne lève
  pratiquement jamais d'exception, donc le `except: pass` n'est **pas** la cause
  du gel (vérifié par lecture du corps complet).
- `update_state()` (dbe l.252), le point d'entrée appelé par l'env à l.4560,
  **ne nourrit PAS le HMM** lui-même : aucune référence à `_update_hmm` /
  `get_regime_probabilities` dans son corps (l.252-565, vérifié par `awk`). Le
  HMM n'est nourri QUE par le chemin l.5393 (construction de l'observation).

---

## 4. Incohérence entraînement ↔ évaluation (point de vigilance majeur)

`hmmlearn>=0.3.0` est listé dans `requirements.txt`. Donc :

- Sur la **VPS d'entraînement** (`/home/morningstar/...`), `hmmlearn` était
  **probablement installé** → les checkpoints 450k/500k ont pu être entraînés
  AVEC un HMM fonctionnel (régimes réels, confidence variable, taille de
  position dynamique).
- Dans **ce sandbox d'évaluation**, `hmmlearn` est **absent** → tous mes
  backtests de cette session ont tourné avec le HMM gelé.

**Implication directe sur la cohésion du backtest :** si les modèles ont été
entraînés avec une `confidence` HMM variable mais sont évalués avec une
`confidence` constante (0.3333), alors la composante « sizing » de leur
politique est évaluée hors de sa distribution d'entraînement. Cela peut à soi
seul expliquer une partie de la dégénérescence observée (taille quasi-identique
entre 450k et 500k, comportement « micro-TP »).

**On ne corrige rien ici** (consigne). On documente que **toute conclusion de
backtest produite dans ce sandbox doit être considérée comme faite avec Oracle
HMM désactivé**, donc non strictement représentative de l'environnement
d'entraînement.

---

## 5. Réconciliation des fichiers de résultats contradictoires

Deux familles de fichiers coexistent dans `logs/validation/` et donnent des
verdicts **opposés sur le même checkpoint/split** :

| Fichier                         | Harnais                       | 450k / test         | Verdict        |
|---------------------------------|-------------------------------|---------------------|----------------|
| `backtest_CORRECTED_450k_test`  | `backtest_fixed_capital.py`   | WR 83.6 % / PF 11.8 | POSITIVE_EDGE  |
| `paper_trading_450k`            | `paper_trading_monitor.py`    | WR 98.6 % / PF 0.75 | NO_EDGE        |

**Ils ne mesurent PAS la même chose** (vérifié par lecture des deux scripts) :

- `backtest_fixed_capital.py` instancie le **vrai `MultiAssetChunkedEnv`**
  (SL/TP dynamiques par profil, durée max, HMM, FiLM…). `best_trade = median =
  12 %` = TP de profil atteint.
- `paper_trading_monitor.py` est un **simulateur séparé** avec son propre
  `VirtualPortfolioManager`, SL/TP **fixes hardcodés** (`stop_loss_pct=0.02`,
  `take_profit_pct=0.03`, l.162) et un interpréteur d'action maison
  (`signal_raw > 0.33 → BUY`, l.383). `best_trade = median = 0.052 %` ⇒ les
  gains ne viennent pas du TP+3 % mais d'une micro-sortie systématique → motif
  « ramasser des centimes » qui est un **artefact de ce simulateur**, pas une
  propriété du modèle.

**Conclusion : le juge de référence est `backtest_fixed_capital.py`** (fidèle à
l'environnement d'entraînement). Les fichiers `paper_trading_*.json` produits
par `paper_trading_monitor.py` ne sont **pas comparables** et ne doivent pas
servir de critère de décision. (Et tous deux, ici, tournent avec HMM gelé —
cf. §4.)

---

## 6. Verdict d'utilisabilité 450k / 500k (état actuel)

Sur le **split `val`** (hors échantillon, juge fidèle `backtest_fixed_capital`),
même avec HMM gelé dans ce sandbox :

| Modèle | val WR | val PF  | val E/trade | Verdict        |
|--------|--------|---------|-------------|----------------|
| 450k   | 60.3 % | 1.18    | +0.173 %    | POSITIVE_EDGE  |
| 500k   | 67.1 % | **2.58**| +1.665 %    | POSITIVE_EDGE  |

→ **Les deux gardent un edge hors échantillon** (WR > ~49 % aléatoire), le 500k
nettement supérieur (PF 2.58). À ce stade, **les modèles ne sont PAS jugés
inutilisables** : la priorité reste donc de les **observer en paper trading live
réel 3 jours** (`run_bot.py --mode paper`, `deterministic=False`), AVANT toute
ré-architecture de l'Oracle.

---

## 7. Ce qu'il faut faire (proposé, NON exécuté — consigne « ne rien modifier »)

1. **Installer `hmmlearn`** dans l'environnement d'évaluation pour ré-aligner
   eval↔train, puis **relancer le backtest val** des deux modèles avec HMM ACTIF
   et comparer aux chiffres ci-dessus (sépare l'effet « HMM gelé » de l'effet
   « politique du modèle »).
2. **Ne PAS** réécrire l'Oracle (mines/force-trade sur bougies futures) tant que
   1) le paper trading live 3 jours n'a pas tranché l'utilisabilité, et 2) on n'a
   pas confirmé que les modèles ont bien été entraînés AVEC hmmlearn.

> Rappel hiérarchie utilisateur : les améliorations Oracle ne se font **que si**
> 450k et 500k se révèlent inutilisables en live.
