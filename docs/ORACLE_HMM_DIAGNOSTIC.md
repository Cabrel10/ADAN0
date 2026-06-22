# ADAN0 — Diagnostic de l'Oracle HMM (CORRIGÉ — audit, AUCUNE modification de code)

> Généré le 2026-06-22. **CORRECTION D'UNE ERREUR DE DIAGNOSTIC PRÉCÉDENTE.**

## 0. Correction critique : l'Oracle HMM N'EST PAS gelé

Un diagnostic antérieur de cette même session concluait à tort que l'Oracle HMM
était « mort-né / gelé à 0.3333 ». **C'était une erreur d'environnement Python
de ma part**, pas un défaut du système ADAN0.

| Élément        | Diagnostic précédent (FAUX)              | Réalité (CORRIGÉ)                                            |
|----------------|------------------------------------------|-------------------------------------------------------------|
| `hmmlearn`     | « non installé, Oracle mort-né »         | **Installé (v0.3.3) dans `trading_env`**                     |
| Python utilisé | `/usr/bin/python3` (système, sans hmm)   | doit être `.../miniconda3/envs/trading_env/bin/python`      |
| État HMM       | gelé à `[1/3,1/3,1/3]`                    | **fonctionne** : probs varient, 3 régimes détectés          |
| Backtests      | « évalués hors-distribution »            | à **relancer** avec `trading_env` pour être valides         |

### Preuve empirique (probe relancé avec le bon Python)

`/home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python /tmp/probe_hmm_direct.py` :

```
hmmlearn IMPORT OK
HMM_AVAILABLE in module = True
probs std per state: [0.152, 0.399, 0.344]     ← VARIATION RÉELLE (≠ 0)
probs at t=0   : [0.333, 0.333, 0.333]  (prior, avant MIN_OBS=60)
probs at t=59  : [0, 1, 0]              (régime détecté avec certitude)
probs at t=199 : [0, 1, 0]
GLOBALLY FROZEN at uniform? False
unique row count: 3                     ← 3 états distincts
```

vs le run précédent (mauvais Python) : `hmmlearn IMPORT FAILED`,
`HMM_AVAILABLE=False`, std≈0, 1 seule valeur. **La seule différence est le
binaire Python.**

## 1. Leçon : TOUJOURS utiliser `trading_env`

Tout script ADAN0 (entraînement, backtest, probe, paper trading) DOIT être
lancé avec :

```
/home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python
```

Le Python système (`/usr/bin/python3`) n'a pas `hmmlearn` ni les autres
dépendances → désactive silencieusement l'Oracle (try/except ImportError) →
fausses conclusions.

## 2. Mécanisme des « 144 micro-trades à +0.052% » — RÉSOLU

Ces 144 micro-trades ne viennent **PAS** de l'environnement RL réel. Preuves :

1. **Protection anti-micro-trade dans l'env d'entraînement/backtest**
   (`multi_asset_chunked_env.py:7155`) :
   ```python
   if unrealized_pnl_pct < 0.0015:  # 0.15% threshold
       discrete_action = 0   # AGENT_CLOSE REJETÉ — profit trop petit vs frais
   ```
   ⇒ le modèle ne PEUT PAS clôturer manuellement un trade < 0.15 %. Un trade à
   +0.052 % est **impossible** par AGENT_CLOSE dans cet env.

2. **Frais d'entraînement gonflés exprès** (`:1145`) :
   `# OPTIMIZATION FOR 0.80% FEES (4× real 0.10% Binance fee)` — pour forcer la
   patience et rendre les micro-trades non rentables pendant l'apprentissage.

3. **L'entraînement ne prenait PAS de micro-positions** : dans le log
   d'entraînement (`logs/training/sandbox_training_20260617_225718.log`), les
   clôtures de fin d'entraînement ont `pnl=1.83`, `pnl=2.23` (magnitude élevée),
   pas 0.052 %.

Les 144 micro-trades proviennent donc de **`paper_trading_monitor.py`** — un
simulateur SÉPARÉ avec SL/TP fixes hardcodés (0.02/0.03) et sans la protection
ci-dessus. **`paper_trading_monitor.py` sert au MONITORING, pas à lancer le
paper trading.** Le vrai paper trading temps réel est **`run_bot.py`**.

## 3. Hypothèse causale (proposée par l'utilisateur, cohérente)

Si un backtest est lancé avec le mauvais Python (HMM gelé en « sideways » 1/3),
le modèle applique sur tout le test sa stratégie apprise pour le régime
« sideways » (micro-scalping de sécurité) au lieu de laisser courir les gains en
tendance. + un éventuel défaut de warmup d'indicateurs au step 0 aggraverait le
phénomène. ⇒ il faut **relancer tous les backtests avec `trading_env`** pour
trancher l'utilisabilité des modèles.

## 4. Actions (en cours)

1. ✅ Confirmé hmmlearn OK dans `trading_env`.
2. ⏳ Relancer le backtest 450k/500k (val + test) avec `trading_env` (HMM actif).
3. ⏳ Vérifier le warmup d'indicateurs au démarrage du backtest.
4. ⏳ Lancer le vrai paper trading via `run_bot.py` (Binance spot testnet).
5. Décision finale : si les modèles prennent des micro-positions AVEC HMM actif
   ET warmup correct → 450k/500k à jeter → correctifs + ré-entraînement 500k sur
   2 profils (workers sur 4). Sinon → conserver.
