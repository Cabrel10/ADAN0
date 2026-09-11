# V36-A — Décision GO/NO-GO (mesure à 82 %, 40 960 / 50 000 steps)

Run: PID 2996698, log `logs/v36/v36a_ablation_20260823_141251.log`,
jsonl `logs/rewards/worker_0_rewards_20260823_141325.jsonl` (41 309 steps).

## Métriques d'APPRENTISSAGE — excellentes ✅
| métrique | V36-A | V35 (baseline) |
|---|---|---|
| EV (moy 5) | **+0.575** | ~ -4.5 (soutenu) |
| EV (dernier) | +0.184 | négatif |
| approx_kl (moy 5) | 0.0108 (stable) | — |
| value_loss | 0.01–0.04 (convergé) | — |
| policy std | 0.392 | 1.06 |

La fonction de valeur apprend enfin (EV franchement positif, KL stable, value_loss
convergé). L'ablation est bien active : `symmetry_penalty = 0 %`,
`future_contrib = 0 %` dans le breakdown.

## GO/NO-GO FINANCIER (le vrai critère du mandat) — échec ❌
| métrique | V36-A | seuil GO |
|---|---|---|
| PnL réalisé cumulé | **-35.46** | > 0 |
| profit factor | **0.213** | > 1 |
| win rate | 17.7 % | > ~40 % |
| reward moy GAGNANTS | **-0.0134** | > 0 |
| reward moy PERDANTS | -0.367 | — |
| ratio L/W | 4.64 | < 1.5 |
| BUY / SELL / HOLD (moy) | 96 / 4820 / 204 | équilibré |

## Cause identifiée
Part d'amplitude du reward PPO :
- **drawdown_penalty 59.1 %** ← domine désormais
- pnl_reward 22.2 %
- closure_bonus 18.5 %
- symmetry 0 %, future_contrib 0 % (ablation OK)

L'ablation A a bien retiré symmetry + FA, mais **drawdown_penalty a pris toute la
place (59 %)** et continue de noyer le signal financier (22 %). Le modèle
apprend bien une fonction de valeur, mais **la politique optimise l'évitement du
drawdown, pas le PnL** → biais SELL massif, PnL négatif.

## DÉCISION : NO-GO pour le run 500k
Lancer 500k maintenant reproduirait l'erreur que le mandat interdit
(« bon EV mais économie non rentable »). Les critères PnL/PF/winrate/reward-gagnants
ne sont pas remplis.

## Prochaine étape contrôlée : V36-A2
Changer **une seule variable** vs A : réduire `drawdown_penalty_factor` par tier
(≈ /4, ex. 2.0→0.5, 1.5→0.4, 1.0→0.25, 0.5→0.1) pour rendre le signal financier
majoritaire, puis re-mesurer le même GO/NO-GO. Objectif : pnl_reward ≥ 50 %,
drawdown ≤ 25 %, reward gagnants > 0.
