# État du code — source de vérité

> Ce document décrit **ce que le code fait**, pas ce qu'on espérait qu'il fasse.
> Chaque affirmation renvoie à un fichier + ligne, ou à une mesure dans
> `logs/validation/*.json`. Ce qui n'est pas vérifié est marqué comme tel.
>
> Il existe parce que les autres documents du dépôt ont accumulé des étiquettes
> de run ne correspondant plus au code.

---

## 0. Nommage des runs — convention obligatoire

**Les étiquettes « V25 / V29 / V30 / V35 » sont abandonnées.** Elles
désignaient des choses différentes selon les documents, et le même label a
servi pour des configurations distinctes. Toute conclusion attachée à un tel
label est inexploitable.

Un run s'identifie désormais par **trois** éléments, tous vérifiables :

| Élément | Où |
|---|---|
| répertoire de log | `logs/<run>/run.log` |
| répertoire de checkpoint | `checkpoints/<run>/` |
| commit qui l'a produit | `git log` au moment du lancement |

Une mesure sans ces trois éléments n'est pas citable dans une décision.

---

## 1. Univers d'entraînement

`scripts/launch_asset_run.py` L57 contraint durement :

```python
ap.add_argument("--asset", required=True,
                choices=["BTCUSDT_BINANCE", "DOGEUSDT_BINANCE"])
```

`derive_config()` réécrit ensuite **toutes** les clés de sélection d'actif :
`data.assets`, `environment.assets`, et `assets` de chaque worker.

**Conséquence à retenir :** `config/config.yaml` contient
`data.assets: [BTCUSDT, XRPUSDT]`, mais ce n'est **pas** ce qui tourne. Un
script de diagnostic qui lit la config sans passer par le launcher mesure un
autre univers que l'entraînement. Ce défaut a réellement faussé une série
entière de sondes, qui lisaient un échantillon de 7 991 lignes (27,7 jours) au
lieu du dataset réel.

Résolution des chemins : `data_loader.py` L256-273 →
`data/processed/indicators/<split>/<ASSET>/5m.parquet`.

---

## 2. Chaîne de décision

```
policy (PPO/SB3, MultiInputActorCriticPolicy, Box(-1,1,5))
   │  a0 continu
   ▼
route_action_by_state(a0, in_position, slot_available, threshold, sell_threshold)
   │  action_routing.py L51
   │  FLAT + quota libre → BUY si a0 > +thr, sinon HOLD
   │  FLAT sans quota    → HOLD
   │  LONG               → SELL si a0 < -sthr, sinon HOLD
   ▼
gates économiques (fee gate EV, deadband, cooldown, daily_limit, drawdown)
   ▼
exécution ou HOLD, avec la raison compteurisée
```

**Propriété structurelle, pas un bug :** le router ne peut **jamais** renvoyer
SELL quand la position est plate, ni BUY quand le quota est saturé. Une sonde
qui échantillonne `a0 ~ U(-1,1)` sans lire l'état demande donc massivement des
actions impossibles et attribue à tort le HOLD résultant à l'environnement.

Seuil réel : `config.environment.action_thresholds` L649 → `5m: 0.05`,
`1h: 0.08`, `4h: 0.10`. Le seuil est un **local** dans `step()` (L3765), pas un
attribut de l'env.

---

## 3. Fee gate EV

`action_routing.py` L184-220 :

```
p_min_required = (SL + round_trip_fees) / (SL + TP)
bloque si p_hmm <= p_min_required
```

`p_hmm = clip(context_vector[3], 0.01, 0.99)`.

**Mesuré** (`logs/validation/fee_gate_measured_*.json`) : `p_hmm` est
**bimodal** — p50 = 0,01 et p90 = 0,99. Le gate bloque le BUY précisément
quand la postérieure de régime est baissière.

| actif | block_rate | bloqués / acceptés |
|---|---|---|
| `BTCUSDT_BINANCE` | 0,827 | 153 / 32 |
| `DOGEUSDT_BINANCE` | 0,887 | 180 / 23 |

**Ce n'est pas un défaut.** Une politique aléatoire demande BUY sans consulter
le régime ; le refuser est économiquement correct. Un taux de blocage élevé
sous politique aléatoire ne dit rien sur la qualité du code.

---

## 4. Plumbing HMM (corrigé)

`dynamic_behavior_engine.py`, marqueur `ADAN0_HMM_READONLY_CONSUMERS`.

Deux appelants partageaient le buffer de fit glissant. Le consommateur ne
fournissait pas de features → `get_regime_probabilities` retombait sur
`(0.0, 0.0, 0.5, 1.0)` → `_update_hmm` l'ajoutait au buffer. Le cache de
déduplication ne pouvait pas court-circuiter, car son test comparait un
`observation_id` valant `None`.

Comme `predict_proba(X)[-1]` décrit la **dernière ligne ajoutée**, la
postérieure décrivait parfois l'observation fictive.

Correctif : `_is_producer = (observation_id is not None) or (log_ret != 0.0)`.
Un non-producteur reçoit `self._hmm_probs.copy()` et n'ingère rien. Le `return`
précède `_update_hmm`, ce qui bloque **les deux** conséquences : ingestion dans
le buffer **et** écrasement de `_hmm_probs`.

Ce second point est le plus important : `_hmm_probs` alimente
`context_vector[3,5]`, donc **l'observation envoyée à la policy**. Tout
entraînement antérieur à ce correctif a appris sur un signal de timing faux
une fois sur deux.

| mesure | avant | après |
|---|---|---|
| lignes synthétiques / 500 | 250 | **0** |
| points distincts | 251 / 500 | **301 / 301** |
| postérieures sur synthétique | 49,92 % | **0 %** |
| obs moteur / step | ~2 | **~1** |

---

## 5. Plafond Future Arena (corrigé)

`reward_service.py`, marqueur `ADAN0_FUTURE_SHARE_CAP`.

`max_future_contrib` bornait une **magnitude** (±0,60). Le watchdog mesure une
**part** : `|future| / (|future| + |pnl|)`. Avec `mean_abs_future = 0,0207` —
trente fois sous le plafond — le clamp ne s'est **jamais déclenché**, alors que
la part valait déjà **65,9 % dès la première fenêtre de 200 steps**, avant toute
saturation de la policy.

Borner une magnitude ne borne pas une proportion. Le paramètre documenté
« le PnL reste roi » était structurellement inopérant.

Nouveau `max_future_share` (défaut 0,40) :

```
part = f/(f+p) <= s   <=>   f <= p * s/(1-s)
```

Le signe est préservé (`math.copysign`) : seule l'amplitude change. À `p = 0`
le terme futur tombe à 0 — voulu, car sans PnL réalisé il n'y a rien à shaper,
et c'est exactement le régime où la part explosait.

**Mesuré** : part 65,9 % → **7,3 %**, avec `mean_abs_pnl` préservé
(0,0107 → 0,0096).

**Prudence sur l'interprétation.** Le déséquilibre était antérieur au collapse,
donc c'est une cause plausible et un défaut réel. Cela ne prouve pas qu'il en
était la cause unique.

---

## 6. Sémantique terminated / truncated

`multi_asset_chunked_env.py` L4648-4668, marqueur
`ADAN0_TRUNCATION_SEMANTICS`. Frontières de fenêtre → `truncated` ; morts
économiques (DRAWDOWN_KILL, faillite) → `terminated`. Conforme
SB3/Gymnasium : la value function n'est bootstrappée que dans le premier cas.

---

## 7. Énergie de décision

`config.yaml` L1616, lu **à la source** :

```yaml
decision_budget:
  enabled: true
  max: 1.0
  cost_buy: 0.15
  cost_close: 0.30
  recharge_hold: 0.02
```

**Vérifié dans le code :** une action structurellement invalide (SELL à plat)
ne consomme **pas** d'énergie. Le débit est dans la branche
`trade_executed_this_step` (L9790 close / L10107 buy), et la recharge
s'applique dès que `not trade_executed_this_step`. Donc SELL-à-plat **recharge**
la jauge. Pas de double punition.

**Angle mort connu :** sous `exit_authority=True`,
`resolve_agent_close_gate` renvoie `(False, "exit_authority")` avant de tester
`budget_insufficient` / `close_gap_active` / `daily_close_quota`. Ces compteurs
sont donc **structurellement condamnés à rester à 0**, qu'il y ait eu pénurie
ou non. Leur valeur nulle ne prouve rien.

---

## 8. Autorité des gates

| Outil | Statut | Raison |
|---|---|---|
| `scripts/validation/policy_aware_execution_test.py` | **autoritaire** | échantillonne parmi les actions légales selon l'état |
| `scripts/validation/financial_stability_check.py` gates B/C | **non-autoritaire** | échantillonne `U(-1,1)` sans lire le portefeuille |
| `financial_stability_check.py` gates A/D/E | lisibles | non affectés par le défaut de harnais |

Le harnais aléatoire fabriquait un plancher de HOLD de 0,440 : 182 SELL
demandés à plat + 38 BUY au quota, sur 500 steps. Ce plancher plus le fee gate
(0,306) plus le deadband (0,078) dépassait le seuil de 0,80 **avant** toute
discussion sur la qualité du signal.

**Mesuré avec le test state-aware** : `sell_while_flat` 182 → **0**,
`buy_while_long` 38 → **0**, divergence non attribuée **0,000**, HOLD sur
décisions légales **0,779**.

Toutes les divergences restantes ont une cause nommée :
`fee_gate 185 + cooldown_hold_min 20 + daily_limit 7 = 212`, `unattributed = 0`.

---

## 9. Ce qui n'est pas résolu

- **Saturation de la policy.** Un run a atteint 481 792 steps avec 0 traceback
  et a échoué : `a0` à ±1,000 sur 100 % des dernières décisions, equity figée
  sur une valeur unique, `explained_variance` négative sur 690 / 938 updates.
  `std` **montait** (0,368 → 0,373) pendant la saturation : l'effondrement est
  piloté par la **moyenne** de la gaussienne, pas par sa variance. Augmenter
  `ent_coef` ne corrigerait donc pas ce mode d'échec.
- **`explained_variance` sur la durée.** Positive au démarrage (+0,334 de
  moyenne sur 100 updates), elle retombe ensuite. Le signe au premier update ne
  prouve rien.
- **Usage du signal de timing par la policy.** Toutes les mesures post-correctif
  utilisent une politique **aléatoire**. Le nettoyage garantit que le signal est
  fiable, pas qu'un cerveau ait appris à s'en servir. `corr(a0, p_hmm)` reste à
  mesurer : `info` ne porte pas `context_vector`, il faut passer par le DBE.
- **Faisabilité / timing dans l'observation.** Les raisons de rejet sont dans
  `info`, que SB3 n'injecte pas dans l'observation. Le modèle voit l'état
  résultant, pas la cause. `sell_while_flat` a un reward exactement neutre
  (0,0), donc SELL-routé-en-HOLD et HOLD direct sont **indiscernables du point
  de vue du gradient**.
- ~~`portfolio_manager.py` L1560 `self.max_drawdown_pct` jamais assigné~~
  **RÉSOLU** (`379df76`) : slot[24] lit désormais `_pending_max_dd_frac`
  (autorité unique), fallback config `risk_management.max_drawdown_pct` au
  cold start. Les slots [24] et [29] encodaient des budgets contradictoires
  (25 % vs 40 %) dans le même vecteur d'état 32-dim.
- ~~`requirements.txt` épingle sklearn 1.6.1, l'environnement a 1.9.0.~~
  **RÉSOLU** (`bec88e9`) : pin aligné sur 1.9.0 = runtime mesuré.
  `models/exog_oracle.pkl` ne désérialise pas sous 1.9.0 mais est
  non-critique (DBE retombe sur des priors uniformes).
- Aucun test de régression ne garde les découvertes fee_gate / HMM.

---

## 10. Pièges opérationnels vérifiés

| Piège | Détail |
|---|---|
| `.gitignore` L29 = `logs/` | `git add -f` obligatoire pour `logs/validation/*.json` |
| logs binaires | `tr -d '\000'` avant tout `grep` |
| `ent_coef` réel | `config.yaml` bloc `[sandbox]` (0,03), **pas** le bloc racine (0,05). Lire la mauvaise clé mène à un faux diagnostic |
| niveau `CRITICAL` | utilisé pour du cycle de vie normal → grepper `CRITICAL` produit de faux tracebacks |
| `PYTHONPATH=src` | requis pour pytest |
| `tests/test_intervention3.py` | appelle `sys.exit` à l'import, casse la collecte du répertoire entier |
| `gh` CLI | non installé sur ce VPS |
| interpréteur | chemin absolu `~/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python` requis |
| runs longs | `nohup ... &` : les tool-calls expirent à 120 s, le process détaché survit |
