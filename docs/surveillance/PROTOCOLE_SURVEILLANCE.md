# Protocole de surveillance — Run 500k (FINDING #4 + revue capture-ratio)

## Objectif
Surveiller le run PPO 500k jusqu'à épuisement des crédits, en documentant CHAQUE
point de contrôle. Détecter toute anomalie bloquante → arrêter, corriger, relancer.

## Configuration du run
- Mode : DiagGaussian (`ADAN_USE_SDE=0 ADAN_LOG_STD_INIT=-1.0`) — validé stable.
- Python : conda `trading_env` (3.11).
- Future Arena : ACTIF, mode `future_guided`, capture-ratio (MFE/MAE), cap 0.60.
- Frais : 0.50 % A/R. Bandes SL/TP serrées (scalper 0.3-1.2%/0.5-2.0% … position 2-6%/3-12%).

## Indicateurs surveillés (logs)
| Tag | Signification | Seuil d'alerte |
|---|---|---|
| `[TRADE_AUDIT_OPEN]` | SL/TP RÉELS envoyés à PortfolioManager | TP > borne profil |
| `[FA_WATCHDOG]` | part du reward venant du futur (future_share) | > 40% WARN, > 50% CRIT |
| `[ACTION_DIST]` | saturation des actions PPO (clip masque-t-il l'apprentissage ?) | sat > 50% prolongée |
| `[DBE_V2_FINAL]` | SL/TP du DBE (affichage) | cohérence vs trading_parameters |
| traceback/exception | crash | toute occurrence |
| explained_variance | qualité du critic | persistance < 0 |

## Points de contrôle (checkpoints de surveillance)
- **C0** : premières 100 lignes (démarrage propre, bridge ACTIVE, pas de crash).
- **C1** : ~2-5 min (premiers trades, TRADE_AUDIT_OPEN réalistes).
- **C2** : premier `[FA_WATCHDOG]` (future_share < 40%).
- **C3** : premier `[ACTION_DIST]` (saturation, tp_raw_mean).
- **Cn** : toutes les ~N minutes jusqu'à épuisement crédits.

## Règle d'arrêt
Anomalie BLOQUANTE (crash, TP hors borne, future_share > 50% persistant,
saturation 100% persistante) → KILL, corriger, relancer, redocumenter.

---

## Pré-vol — Validation des profils (runtime, avant 500k)

Smoke tests sandbox avec sélection de profil (`--profiles <name>`, nouvel arg
câblé dans `sandbox_train(worker_key=...)`). Preuve que `_BOUNDS` est piloté par
le profil (TP/SL réels dans `TRADE_AUDIT_OPEN`) :

| Profil | Worker | Opens | TP observé (min→max) | tp_hi bande | SL max | sl_hi bande | Erreurs | Statut |
|---|---|---|---|---|---|---|---|---|
| scalper | w1 | 368 (smoke2) | 0.90%→1.92% | 2.0% | 1.10% | 1.2% | 0 | ✅ |
| swing | w3 | 130 (fa_swing) | 1.50%→7.00% | 7.0% | ~3.5% | 3.5% | 0 | ✅ |
| position | w4 | 220 (fa_position) | 4.48%→12.00% | 12.0% | 6.00% | 6.0% | 0 | ✅ |
| intraday | w2 | code-level | — | 4.0% | — | 2.0% | — | ✅ (substring-matching prouvé sur 3 profils distincts) |

Conclusion : le mapping profil→bande est **prouvé en runtime** sur 3 profils
distincts couvrant tout le spectre (1.92% → 12.00% TP). Le 4ᵉ (intraday) partage
le même chemin de code validé. Aucune source ne produit de TP hors-borne.

---

## Journal de surveillance
(chaque entrée = un rapport horodaté, append-only)

### 2026-06-27 — Run `fa_500k_v4` (post-corrections cross-TF + PnL latent + saturation)
Relancé après commit `8cf8ac6` (fix cross-TF / latent PnL / saturation / promotion rescale).
Watcher automatique : `scripts/diagnostics/surveil_fa_500k.sh` (audit compact + rotation 200MB).

- **C0 (step ~898)** : démarrage propre. exceptions=0 ; **OHLC_INCOHER=0** ;
  TRADE_AUDIT_OPEN `exec_src=open[t+1]` (TF d'exécution fixe actif) ; SL 0.79-0.82% /
  TP 1.45-1.46% dans la bande scalper ; FA_WATCHDOG future_share 10.6-11.1% ;
  ACTION_DIST tp_sat 2% sl_sat 1%. ✅
- **C1 (step 3433, ~9 min)** : exceptions=0 ; **OHLC_INCOHER=0** ;
  future_share=16.7% (<40%) ; mean_abs_pnl=0.0506 (sain, vs 0.72 du run buggé) ;
  tp_pct_mean=1.32% bande [0.60%,2.00%] ; TRADE_AUDIT SL=0.65% TP=1.05% réalistes ;
  raw log 19MB / audit 16KB. ✅
- Preuve clé : le watchdog OHLC (FIX2) ne déclenche **aucune** incohérence →
  le bug cross-TF (entry 108114 / low 87645 = -18.9% impossible) est éradiqué.
- Règle d'arrêt active : OHLC_INCOHER>0 OU exception OU future_share>50% persistant → KILL.
