# Trace : d'où viennent TP=4 % / SL=2 % ?

**Objectif (verbatim utilisateur) :** « retracer où ces valeurs ont réapparu … Le but
n'est pas encore de corriger. On doit déterminer qui écrit la valeur finale. »

## Méthode

1. `grep -RInE "take_profit_pct|stop_loss_pct|..." config src scripts`
2. `grep -RInE "ADAN_FREE_SLTP|FREE_SLTP|take_profit_pct|stop_loss_pct" .`
3. Lecture du mapping free_sltp dans `multi_asset_chunked_env.py` (L8931-9030).
4. Confrontation avec le **log d'entraînement A2 réel** (ce qui est appliqué, pas ce
   qui est écrit dans la config).

## Ce que la config contient

| Emplacement | Clé | Valeur |
|---|---|---|
| `config_v36a2.yaml` L1567/1568 (défaut global) | `stop_loss_pct` / `take_profit_pct` | 0.02 / 0.04 |
| `config_v36a2.yaml` L749/752 (hard_constraints) | SL max/min · TP max/min | 0.06/0.003 · 0.12/0.005 |
| `config_v36a2.yaml` L1639+ (profil scalper) | trading_config SL/TP | 0.015 / 0.03 |
| `config.yaml` L1752/1753 (défaut global) | SL / TP | 0.02 / 0.04 |

Donc oui, **la valeur 0.04 existe bien encore** dans les configs (défaut global).
Mais ce n'est PAS elle qui pilote l'entraînement.

## Ce qui est RÉELLEMENT appliqué (preuve par le log)

Sous `ADAN_FREE_SLTP=1` (mis par `_run_v36_bg.sh`) :

- Le code de l'env (`multi_asset_chunked_env.py` L8931-9030) **ignore** `take_profit_pct`
  de la config et remappe linéairement l'action brute du réseau dans une **bande fixe** :
  ```
  sl_lo, sl_hi = 0.003, 0.060      # SL ∈ [0.3 %, 6 %]
  tp_lo, tp_hi = 0.003, 0.120      # TP ∈ [0.3 %, 12 %]
  _round_trip  = max(2*comm, 0.005)
  tp_lo        = max(tp_lo, _round_trip*1.2)   # → ~0.60 %
  normalized_tp = (tp_raw + 1) / 2
  tp_pct        = clip(tp_lo + normalized_tp*(tp_hi - tp_lo))
  ```
- Le log ACTION_DIST du run rapporte (preuve vérifiée, `v36a_ablation_20260823_141251.log`
  L352959+, n=49600–50000) : **`tp_pct_mean ≈ 6.28–6.30 %`, `band[0.60 %, 12.00 %]`**,
  avec **`tp_raw_mean ≈ -0.002` (sortie réseau quasi-neutre)**. Autrement dit, même sans
  aucune intention du réseau, le milieu de bande donne déjà ~6 % de TP.
- La contrainte R/R ≥ 1.5 et le SL scalaire basé ATR sont **sautés** quand FREE_SLTP=1.

## Verdict

| Question | Réponse |
|---|---|
| La config 0.04 est-elle la source de la valeur finale ? | **NON** — bypassée par FREE_SLTP. |
| Une ancienne config a-t-elle « ressuscité » silencieusement ? | **NON** — la valeur appliquée (6.59 %) ne vient d'aucune config ; elle sort de la bande fixe + sortie neutre du réseau. |
| Qui écrit la valeur finale ? | **L'env**, via le mapping FREE_SLTP, borné par une **bande en % FIXE (0.6 %–12 %)**. |
| Est-ce le vrai problème ? | **OUI, et c'est pire** : le TP moyen effectif ~6.6 % = **~46× l'ATR médian (0.143 %)**. La bande n'est pas ancrée à la volatilité. |

**Conclusion :** ce n'est pas un config-fantôme à supprimer. Le vrai verrou est que
la bande SL/TP est **exprimée en pourcentages fixes**, pas en **unités d'ATR**. Cela
valide directement le **verrou #2** (« TP/SL en ATR ») et l'orientation
« exprimer tout en unités de volatilité ».

> Statut : **trace terminée, aucune correction appliquée** (conformément à
> « Le but n'est pas encore de corriger »).
