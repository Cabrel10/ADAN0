# Run Health V25 — v25_20260806T025139Z_smokecfg

## Verdict

V25 était numériquement fini mais fonctionnellement invalide. Le critic breaker a arrêté le run à **41 985** pas après dix rollouts consécutifs avec `explained_variance < -0.2`. Le mean directionnel final était **-1.6169**, avec **100 % SELL** échantillonné puis transformé en HOLD lorsque le portefeuille était flat.

Le checkpoint final ne doit pas être repris et aucun run 1M n'est autorisé.

## Preuves du chemin causal

- **Forced trade inactif** : la phase active le désactive explicitement ; 1 440 pas ne déclenche qu'un warning.
- **Signe SL correct** : `sl_raw` est converti en distance positive bornée ; une valeur positive élargit le stop.
- **Routing intentionnel et neutre** : FLAT+SELL et LONG+BUY sont des régions no-op V12. Aucune pénalité de routing n'est active ou ajoutée.
- **Correction V16 absente de V25** : ni reward mark-to-market ni ancre L2 actor-level n'étaient actifs.

```text
crédit réalisé peu dense + grandes régions no-op state-conditioned
  -> dérive du mean actor vers une borne
  -> policy déterministe non exécutable lorsque flat
  -> collapse du signal return/advantage utile
  -> explained variance durablement négative
  -> arrêt par critic breaker
```

## Correction retenue pour le nouveau 500k

```text
DiagGaussian:            ADAN_USE_SDE=0, ADAN_LOG_STD_INIT=-1.0
PPO inchangé:            ent_coef=0.05, n_epochs=4, n_steps=512
Crédit dense:            ADAN_MTM_REWARD=1
Anti-saturation actor:   ADAN_L2_ANCHOR_LAMBDA=0.05
Isolation:               ADAN_AUX_LOSS_COEF=0.0
Sécurité:                critic breaker + collapse breaker actifs
Observabilité:           diagnostic 512, reward telemetry 100
Récupération:            checkpoints uniques tous les 10 000 pas
Scalers:                 persistés, sans refit ni sauvegarde de fin
```

gSDE seul, une pénalité de routing, une modification forced-trade, une inversion SL et un tuning PPO arbitraire ont été rejetés.

## Résumé V25

- EV moyenne **-0.3188**, moyenne des dix dernières **-1.3067**, minimum **-3.2598**.
- Clip fraction moyenne des dix dernières : **0.4698**.
- Opens/closes : **461/461**, aucune position logique non fermée.
- Win rate **16.92 %**, profit factor **0.1652**.
- Les cinq checkpoints donnent `NO_TRADES` au protocole déterministe identique de 1 000 pas.
- Tous les checkpoints se chargent et leurs paramètres sont finis.
- VecNormalize est resté désactivé et les pickles de scalers n'ont pas changé.

## Décision

Un **nouveau 500k corrigé est autorisé directement** après validation technique ciblée. V25 n'est pas repris, le dernier checkpoint n'est pas supposé meilleur et 1M reste interdit.
