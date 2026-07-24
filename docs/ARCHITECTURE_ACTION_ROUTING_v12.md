# ADAN0 v12 — State-Conditioned Action Routing (la Machine à États)

> **Référence intersession.** Ce document est la source de vérité sur la façon
> dont l'axe `action[0]` (direction) est décodé en décision de trade. Toute
> évolution de la logique d'action DOIT mettre à jour ce fichier et les 3
> moteurs listés en §6.

---

## 1. Le problème résolu (pourquoi v12)

Les runs V10 (@70k) et V11 (@78k) ont **collapsé de façon identique** :
`pct_buy → 0.97`, `a0_mean` dérive monotone, `illegal_ratio` 25% → 68%,
entropy figée (~-0.556). Malgré les fixes successifs (signe V9, warmup C1,
gSDE-off), la pathologie s'est reproduite. **Ce n'était donc PAS un bug de
reward-shaping ni de signe.**

Cause racine **structurelle** : l'espace d'action `action[0]` était décodé de
façon **symétrique** (SELL si `a0 < -thr`, BUY si `a0 > +thr`, sinon HOLD),
**indépendamment de l'état du portefeuille**. Or un portefeuille SPOT n'a que
deux états par actif :

| État | Actions qui ont un sens | Action structurellement absurde |
|------|-------------------------|---------------------------------|
| **FLAT** (pas de position) | OPEN (BUY) ou HOLD | SELL (rien à vendre) |
| **LONG** (position ouverte) | CLOSE (SELL) ou HOLD | BUY (déjà exposé) |

Avec le décodage symétrique, l'agent passait la majorité de son expérience à
émettre des actions **illégales** (SELL-while-flat, BUY-while-open). Ces
échantillons illégaux polluaient le gradient de politique : le neurone SELL
finissait associé à la douleur (pénalités stériles), et une fois en position
l'agent était « terrorisé » de vendre → HOLD jusqu'à la mort (SL / max-duration)
→ collapse vers always-BUY.

**Diagnostic de l'Architecte : « Collapse par Asymétrie d'État ».**

---

## 2. La décision : routage dans l'env, pas MaskablePPO

`MaskablePPO` (sb3-contrib) ne gère que les espaces **discrets**
(`Discrete`/`MultiDiscrete`). Notre action_space est
`Box(-1, 1, (n_assets × 5,))` **continu**, et l'algo est `WorldModelPPO`
(PPO custom + auxiliary forward-prediction loss). Basculer sur MaskablePPO
imposerait de :

- installer `sb3-contrib` (absent),
- **abandonner `WorldModelPPO`** (perte de l'aux-loss),
- **convertir Box → MultiDiscrete**, ce qui **casserait les 4 autres dims**
  (Size, Timeframe, SL, TP) qui sont pilotées par le **Future Arena / Oracle**
  et doivent rester continues et intactes.

→ **Rejeté.** On applique le masquage **DANS l'environnement**, sur le seul
`action[0]`, sans toucher l'algo ni le reste de l'action_space. C'est du **hard
routing** : l'action illégale n'est jamais exécutée ET n'engendre **aucune
pénalité** — la moitié « inutile » de l'axe est simplement réinterprétée en
HOLD neutre. L'agent apprend une vraie machine à états, pas à éviter une
pénalité.

---

## 3. La spécification du routage (source unique : `action_routing.py`)

Fonction : `adan_trading_bot.environment.action_routing.route_action_by_state`

```
route_action_by_state(a0, in_position, slot_available, threshold) -> {0:HOLD, 1:BUY, 2:SELL}

  LONG (in_position=True):
      a0 < -threshold  -> SELL (CLOSE)
      sinon            -> HOLD_POS      (même a0 = +1.0 → HOLD, zéro pénalité)

  FLAT (in_position=False):
      slot_available=False -> HOLD      (NOOP : slot au-delà du quota du palier)
      a0 > +threshold      -> BUY (OPEN)
      sinon                -> HOLD       (même a0 = -1.0 → HOLD, zéro pénalité)
```

Propriétés garanties :

- Le routeur **ne peut jamais** retourner SELL en état FLAT, ni BUY en état LONG.
- Les branches de pénalité `sell_no_position`, `anti_spam_hold` (BUY-while-open)
  et `CASH_FLOOR_B` (min_notional_self_caused) deviennent donc **structurellement
  inatteignables**. Elles ont été **neutralisées** (HOLD neutre, zéro pénalité,
  zéro gradient) dans `multi_asset_chunked_env._execute_trades`. Réintroduire une
  pénalité sur ces branches ré-injecterait la pollution de gradient que v12
  élimine.
- Seul `action[0]` est routé. **Dims 1-4 (Size, Timeframe, SL, TP) INTACTES** —
  pilotées par le Future Arena (l'Oracle qui enseigne *où* placer TP/SL).

### Tableau de la machine à états (palier Micro, 1 actif = binaire pur)

| État | `a0` | Décision | Reward direction |
|------|------|----------|------------------|
| FLAT | > +thr | OPEN (BUY) | signal marché normal |
| FLAT | ≤ +thr (incl. -1.0) | HOLD | neutre (0) |
| LONG | < -thr | CLOSE (SELL) | signal marché normal |
| LONG | ≥ -thr (incl. +1.0) | HOLD_POS | neutre (0) |
| FLAT + slot > quota | tout | HOLD (NOOP) | neutre (0) |

---

## 4. Passage à l'échelle : paliers 1 → 5 (multi-asset)

Le routage s'applique **par slot d'actif**, dans la boucle
`for i, asset in enumerate(self.assets)`. Le quota de slots vient du palier
(capital tier verrouillé en début d'épisode, `self._locked_tier`) :

| Palier | `max_concurrent_positions` | Slots actifs |
|--------|----------------------------|--------------|
| Micro (0) | 1 | 1 |
| Small | 2 | 2 |
| Medium | 3 | 3 |
| High | 4 | 4 |
| Enterprise | 5 | 5 |

`slot_available = (n_open < max_concurrent_positions)`. Un slot FLAT au-delà du
quota → NOOP forcé. Un slot LONG peut **toujours** fermer (indépendant du
quota). Chaque actif a sa propre entrée `positions[asset]` dans
`portfolio_manager` (dict par actif, déjà existant), donc « gérer une position »
cible naturellement le bon actif — aucun nouveau mécanisme à créer.

**Config actuelle : `assets: [BTCUSDT]` seul.** Le palier Micro (1 slot, 1 actif)
= machine binaire pure. Les slots multiples ne s'activent qu'en ajoutant des
actifs à la liste ET en atteignant un palier supérieur. Le design est prêt pour
ce futur (multi-asset / multi-timeframe) **sans refonte**.

---

## 5. Migration scalper → intraday

Le profil est sélectionné **au lancement** via `--profiles intraday`, qui résout
le `worker_key` par nom de profil vers `w2` (profile: intraday). Aucun changement
de `config.yaml` requis (les 4 profils y sont déjà définis explicitement).

Les bornes SL/TP s'activent **automatiquement** via `_BOUNDS[profile]` dans
`_execute_trades` (dims 3-4, module Future Arena intact) :

| Profil | SL range | TP range | max_position_steps |
|--------|----------|----------|--------------------|
| scalper | 0.3 – 1.2 % | 0.5 – 2.0 % | 20 |
| **intraday** | **0.5 – 2.0 %** | **0.8 – 4.0 %** | **100** |
| swing | 1.0 – 3.5 % | 1.5 – 7.0 % | 200 |
| position | 2.0 – 6.0 % | 3.0 – 12.0 % | 500 |

Garde-fous préservés : **fee gate `TP_min ≥ 0.6 %` (= 1.2× frais A/R)** et
**R:R ≥ 1.5**. **Les frais restent à 0.5 % (`round_trip_fees = 0.005`) — INTACTS.**
Le Future Arena continue d'enseigner *où* placer TP/SL ; le Risk Manager agit en
**scaler** (borne la plage), pas en écraseur.

---

## 6. Ordre des moteurs à synchroniser (à chaque changement de logique d'action)

Un **seul** endroit définit la logique : `action_routing.route_action_by_state`.
Les moteurs l'importent — **zéro divergence possible** :

1. **`src/adan_trading_bot/environment/multi_asset_chunked_env.py`**
   (`_execute_trades`) — training. ✅ branché + pénalités mortes neutralisées.
2. **`scripts/monitoring/paper_trading_monitor.py`**
   (`interpret_target_weight_action`) — paper trading. ✅ branché + testé.
3. **`src/adan_trading_bot/trading/action_translator.py`**
   (`_parse_continuous_action`) — live trading. ✅ branché + compile OK.
   ⚠️ NB : ce module a un **import cassé pré-existant**
   (`PositionSizingMethod` absent de `position_sizer.py`), indépendant de v12,
   à corriger lors de la phase « autres moteurs ».

Le module `action_routing.py` embarque une suite d'assertions unitaires (voir
tests inline) validant les 9 cas FLAT/LONG/NOQUOTA.

---

## 7. Invariants à NE JAMAIS remettre en question

- **Frais = 0.5 %** (`commission: 0.0025`, `round_trip_fees: 0.005`). Intacts.
- **Dims 1-4 (Size, Timeframe, SL, TP)** pilotées par le Future Arena. Ne pas toucher.
- **VecNormalize désactivé volontairement.** Ne pas réactiver.
- Pas de `sb3-contrib` / `MaskablePPO`.
- `obs_schema_v2 = 28 dims`, exposure [70,90], max_position_size_pct 90,
  min_order_value_usdt 11.0, capital 20.5.

---

## 8. Critères de validation du run V12 (Mission 5)

Smoke 300 steps `--profiles intraday`, puis dans la télémétrie :

- `illegal_ratio ≈ 0` (plus d'action illégale de direction ; seul le dépassement
  de quota compte désormais, impossible à 1 actif / palier Micro).
- `pct_buy` démarre proche de 50 %, **pas de dérive monotone** vers 0.97.
- Dims 1-4 (SL/TP) produisent des valeurs non nulles dans les bornes intraday.
- Zéro erreur de shape / import.

Run long 500k lancé seulement si smoke vert ET ce document committé.
