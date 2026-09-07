# Diagnostic — Run v30_500k (BTCUSDT_BINANCE)

> Identifiant run (convention ETAT_DU_CODE.md) :
> - Répertoire de log : `logs/v30_500k/run.log`
> - Répertoire de checkpoint : `checkpoints/v30_500k/`
> - Commit au lancement : branche `genspark_ai_developer` (tip 65b05d3 à l'analyse)
> - Commande : `scripts/launch_asset_run.py --asset BTCUSDT_BINANCE --steps 500000`
> - Process : pid 169220, lancé le 2026-09-07 03:37, **vivant** à l'analyse (23:39)

## 1. État du run — CONFIRMÉ

| Métrique | Valeur | Source |
|---|---|---|
| Steps complétés | ~452 000 / 500 000 (~90 %) | `logs/rewards/worker_0_rewards_20260907_033810.jsonl` (452 426 lignes) |
| Checkpoints écrits | toutes les 10k steps jusqu'à 450 000 | `checkpoints/v30_500k/ppo_adan0_BTCUSDT_*_steps.zip` |
| Tracebacks | 0 | run.log |
| Episode courant | 0, profil `scalper`, worker 0 | rewards JSONL |

## 2. Comportement de la policy — CONFIRMÉ : collapse en hold

Distribution des actions sur les 452 426 steps du fichier rewards actif :

| Action | Occurrences | Part |
|---|---|---|
| hold | 447 789 | **99,0 %** |
| buy | 2 323 | 0,51 % |
| sell | 2 305 | 0,51 % |
| stop_loss | 9 | 0,002 % |

- `steps_since_last_trade` atteint **20 184** (seuil d'alerte configuré : 1 440 — dépassé ×14).
- Toutes les sorties récentes : `sell_score=1.0` mais `size_pct=-1.0` → veto size, action rabattue sur `hold`.
- Equity figée à **19,59** sur toute la fin du run (min historique 12,28 / max 20,52).

## 3. Économie — CONFIRMÉ : le critère > 21 $ n'est JAMAIS atteint (tier Micro)

| Métrique (run complet) | Valeur |
|---|---|
| Capital initial | 20,50 $ |
| Equity min / max / dernier | 12,28 / **20,52** / 19,59 |
| Trades clôturés | 2 324 |
| Taux de gain net | **1,51 %** (35 / 2 324) |
| PnL brut total | −12,61 $ |
| PnL net total | **−132,48 $** |
| Frais totaux | 119,87 $ (0,0516 $/trade) |
| Taux de frais implicite | 0,40 % — conforme à `round_trip_fees: 0.004` (pas de fuite comptable) |
| Notional moyen | 12,90 $ |
| Motifs de clôture | AGENT_CLOSE 2 305 · DRAWDOWN_KILL 10 · stop_loss 9 |

Note : les equity ~104 $ trouvées dans certains fichiers de metrics appartiennent à un autre tier (capital 100 $) — elles ne prouvent rien pour le tier Micro.

## 4. Classification des causes

- **CONFIRMÉ** — Plomberie saine : pas de traceback, checkpoints réguliers, frais conformes à la config, invariants de reward OK (`invariant_ok: true` partout).
- **CONFIRMÉ** — La policy a appris à ne presque plus trader (hold 99 %) ET les rares trades gagnent 1,5 % du temps avec RR moyen 0,318 (avg win +0,0185 / avg loss −0,0582).
- **CONFIRMÉ** — Géométrie SL/TP défavorable documentée (`sl_hi=0,0235` / `tp_lo=0,0135`, RR<1) : exige 60-74 % de win rate pour l'espérance positive ; le modèle est à 1,5 %.
- **PROBABLE** — Le collapse hold est une réponse rationnelle de la policy à une fonction de reward où trader détruit de l'equity (frais 0,4 % round-trip sur notional 12,9 $ = 0,05 $ par aller-retour, contre espérance de gain par trade négative).
- **INFIRMÉ** — « Le run est vert parce qu'il tourne » : il tourne, mais n'apprend plus rien d'utile sur la fin (equity figée, zéro trade sur les 20k derniers steps).
- **NON RÉSOLU** — Le run produira-t-il un checkpoint final exploitable à 500k ? Réponse à la fin du run (~48k steps restants).

## 5. Décision

1. **Laisser le run se terminer** (90 % fait, ~2-3 h restantes) — ne pas tuer un run à 90 % pour un collapse déjà documenté.
2. **Ne pas lancer de nouveau run 500k** sur cette config : le prochain levier est la géométrie SL/TP (viser RR>1 ou recalibrer sur la volatilité mesurée de BTC), pas plus de steps.
3. Avant tout nouveau run : appliquer la convention de nommage d'ETAT_DU_CODE.md (log dir + checkpoint dir + commit), et ajouter une sonde win-rate glissante dans le monitoring (le monitoring actuel détecte les crashes, pas le collapse économique).

*Diagnostic généré le 2026-09-07 à partir des logs réels du run — chaque chiffre est reproductible par les commandes grep/python de la session.*
