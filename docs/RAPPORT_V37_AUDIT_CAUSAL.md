# RAPPORT COURANT — Run v37 BTC 500k — Audit causal (2026-09-03)

> Document de référence unique pour l'interprétation du run en cours.
> Remplace tous les anciens SESSION_*/V-audit/ANALYSIS_* (purgés).

## Contexte

- Run : `launch_asset_run.py --asset BTCUSDT_BINANCE --steps 500000` (PID 78472), lancé après le fix
  `exit_authority` (commit 85c3d98) qui a restauré SELL→SELL et ventilé les causes de rejet.
- Mission : audit causal POLICY RAW → ROUTING → GATES → EXÉCUTION avant toute décision
  budgétaire ou relance. Critère d'arrêt : à 100k steps, si peak_capital < 21 USDT → STOP.

## Méthode d'audit

Script : `logs/analysis/audit_causal_v37.py` sur `logs/v37_500k/btc_500k.log`.

Sources (le run n'écrit pas de trace JSONL — `ADAN_PIPELINE_TRACE_PATH` non défini) :
- `[TARGET_WEIGHT]` (1 step / 50) : brut policy `Raw=a0`, seuil `Thr`, action routée.
- `[ACTION_DIFF]` (1 step / 50) : `Requested` (post-routing) vs `Executed` + compteurs cumulés.
- `[POSITION FERMÉE]` : raisons de clôture et PnL.

`Requested` dans ACTION_DIFF est **post-routing** (pas le brut policy). Le brut est dans TARGET_WEIGHT.

## Résultats (fenêtre ~22k steps, 441 échantillons joints)

### [1] Distribution brute de la policy
| Catégorie | n | % |
|---|---|---|
| raw_wait (policy neutre, \|a0\|≤thr) | 7 | 1.6 % |
| raw BUY routé BUY | 1 | 0.2 % |
| raw SELL routé SELL | 4 | 0.9 % |
| raw BUY → HOLD (routing) | 0 | 0.0 % |
| **raw SELL → HOLD (routing)** | **429** | **97.3 %** |

### [2] Transformations par les gates (échantillon)
SELL→HOLD : 2 ; BUY→HOLD : 1. Quasi nul.

### [4] Deltas de compteurs sur la fenêtre
Tous à **0** : deadband, routing, budget_insufficient, close_gap, quota, break_even,
hold_min, portfolio_reject, trade_executed. Les compteurs affichés dans le log sont des
**cumuls figés depuis le début du run** (ex. `routing_reject=45080`, `trade_executed=7364`
au step ~22150) — ils n'augmentent plus.

### [6] Fermetures (log complet)
| Raison | n | PnL total | PnL moyen |
|---|---|---|---|
| AGENT_CLOSE | 3613 | $-207.58 | $-0.0575 |
| DRAWDOWN_KILL_FORCE_CLOSE | 16 | $-0.72 | $-0.0450 |
| stop_loss | 12 | $-1.92 | $-0.1600 |

## Verdict causal

**CONFIRMÉ — dérive comportementale de la POLICY, pas un blocage de l'environnement.**

- La policy émet des intentions fortes sur 98.4 % des décisions échantillonnées.
- 98.8 % de ces intentions sont des **SELL émis alors que le portefeuille est FLAT** →
  neutralisés par le routing (`sell_while_flat`), ce qui est la règle physique voulue
  (pas de vente sans position). Ce n'est PAS une transformation de décisions légitimes.
- Le budget n'est **plus** un facteur : `budget_insufficient=0`, budget revenu à 1.000/1.00,
  gates de close tous à 0 (exit_authority actif).
- Après le DRAWDOWN_KILL (equity 12.30, -40 %), la policy est FLAT et demande SELL en boucle
  depuis >15k steps sans jamais re-router vers BUY → **0 trade** sur la fenêtre observée.
- Historiquement dans ce run : 3613 AGENT_CLOSE à PnL moyen $-0.0575 → la policy
  ouvrait/fermait à perte (frais $0.04-0.07 par round-trip sur tailles ~$10-18).

**Infirmé** : « collapse HOLD par blocage budget » (le compteur budget_reject élevé de v34
était un agrégat historique, pas un mécanisme actif).

**Non résolu / à investiguer** : pourquoi la policy préfère SELL-while-flat à BUY
(pénalité `sell_while_flat` trop faible ? asymétrie de reward ? std PPO 0.369 figé ?).
Hypothèse probable : après drawdown, la valeur prédite de BUY < HOLD < SELL dans cet état,
et la pénalité d'intention invalide ne suffit pas à déplacer le mode de la politique.

## Décision

- **Aucun levier budgétaire appliqué** (ADAN_CLOSE_COST etc. suspendus) — conforme au verdict :
  on ne modifie pas l'environnement sur la base d'un compteur agrégé.
- Le run se poursuit jusqu'au critère : **100k steps & peak_capital < 21 USDT → STOP**,
  puis correction de la cause policy-side (reward/pénalité d'intention invalide), pas env-side.

## État du code (branch genspark_ai_developer)

- 85c3d98 (poussé) : exit_authority + ventilation des gates + tests v20/v29 alignés.
- Non commités au moment de l'audit : fix ordre de trace t2 (closes marché tracés avant opens
  différés), slot [27] = decision_budget_norm observable, purge vx/vy.
