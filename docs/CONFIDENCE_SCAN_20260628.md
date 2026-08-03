# 🔬 RADIOGRAPHIE PAR CHECKPOINT — Vérification factuelle (28 juin 2026)

> Objectif : prouver dans les **faits** (logs + backtests, pas hypothèses) si le PPO
> apprend, où est le « pic d'intelligence », et préparer un paper trading conforme.

---

## 1. FAITS ÉTABLIS DANS LE CODE (pas d'hypothèse)

### 1.1 Frais réels = 0.50 % A/R (et non 0.80 %)
- `config/config.yaml:658` → `commission: 0.0025`, **appliqué par côté**.
- `portfolio_manager.py:126-129` → `fee_pct = env_config["commission"] = 0.0025`.
- `portfolio_manager.py:790` `entry_fee = cost × 0.0025` ; `:946` `exit_fee = exit_value × 0.0025`.
- **⇒ frais aller-retour réels = 0.0025 × 2 = 0.50 %.** ✅ Ton analyse est correcte.
- Le `0.004` vu à `multi_asset_chunked_env.py:7814` n'est qu'un **fallback** jamais
  atteint (l'attribut `commission` est toujours défini). La « Radiographie » à 0.80 %
  a donc été produite avec un **ancien run/config** ; le run actuel est bien à 0.50 %.

### 1.2 Profil du run actuel = SCALPER (confirmé dans les logs)
- `logs/training/fa_500k_prod_*.log` → **12 053** lignes `profile=scalper`.
- Bandes SL/TP scalper (`multi_asset_chunked_env.py:1306` ET `execution_engine.py:118`,
  **identiques**) : **SL 0.3–1.2 % / TP 0.5–2.0 %**, avec fee-gate `TP_min ≥ 1.2 × 0.5 % = 0.6 %`.
- **⇒ Pour le paper trading il FAUT `--profile scalper`** (le défaut du bot est `intraday` =
  NON conforme → divergence SL/TP garantie).

### 1.3 Reward shaping vs PnL — le `-0.30` à PnL plat
- `[TIER_REWARD]` ne loggue que **5** des **11** composants de `raw_reward`
  (`multi_asset_chunked_env.py:6659-6672`). Les 6 cachés : `pnl_base_reward`,
  `future_contrib`, `latent_pnl_contrib`, `symmetry_penalty`, `action_entropy_penalty`,
  `saturation_penalty`. Le `Final=-0.30` à « PnL=+0.00 % » vient surtout du
  **PnL latent négatif** (capital qui s'érode 20.50→18.99), pas d'un bug de shaping.
- **Watchdog Future-Arena = OK** : `future_share=32 %` (cible <40 %),
  `mean_abs_future=0.0042` vs `mean_abs_pnl=0.0090`. **Le futur ne domine PAS le PnL.** ✅
  ⇒ Les zones 🟢🟡🔴 contribuent sans écraser le signal économique (design respecté).

### 1.4 Actions illégales (tueur de gradient) — confirmé
Dernier `ACTION_DIFF` du run : `sell_no_position=1359`, `min_notional=1313`,
`cooldown_wait=50`. ⇒ l'agent demande **massivement** des ventes sans position et des
ordres sous le notional minimum. Cela pollue effectivement le gradient.

### 1.5 Raisons de fermeture (full log) — cohérent avec la « Radiographie »
| Raison | Count | % |
|---|---|---|
| SL_HIT | 1508 | **59 %** |
| TP_HIT | 602 | 24 % |
| MAX_DURATION | 416 | 16 % |
| AGENT_CLOSE | 18 | 0.7 % |

`TP/SL = 602/1508 = 0.40` ⇒ **WR ≈ 28.5 %**. La Radiographie reflète bien CE run
(seule la valeur des frais 0.80 % y était périmée).

---

## 2. COURBE D'APPRENTISSAGE PAR CHECKPOINT (backtest split TEST, fixed-capital)

| Checkpoint | Trades | WR | Profit Factor | Expectancy/trade | Return | best_trade | worst_trade | Verdict |
|---|---|---|---|---|---|---|---|---|
| 40k  | 0  | —     | —     | —        | —       | —      | —      | NO_TRADES |
| 100k | 40 | 27.5 %| 0.462 | -0.188 % | -0.75 % | +1.28 %| -1.20 %| NO_EDGE |
| 160k | 23 | 34.8 %| 0.373 | -0.159 % | -0.37 % | +0.90 %| -1.20 %| NO_EDGE |
| 200k | 34 | **47.1 %**| 0.431 | -0.167 % | -0.57 % | +0.57 %| -1.20 %| NO_EDGE |
| 240k | 34 | 44.1 %| 0.416 | -0.170 % | -0.58 % | +0.57 %| -1.20 %| NO_EDGE |

### Lecture
1. **Le PPO apprend une chose** : le Win Rate monte nettement **27.5 % → 47 %**.
2. **Mais ce n'est PAS rentable** : `profit_factor < 0.5` partout, expectancy toujours
   négative, return toujours négatif. Verdict **NO_EDGE** à tous les checkpoints.
3. **La cause mécanique est limpide** :
   - `worst_trade = -1.20 %` **constant** = le SL max scalper (×levier) → toutes les pertes
     tapent le plafond du SL.
   - `best_trade` **décroît** 1.28 % → 0.57 % → l'agent ferme ses gains **de plus en plus tôt**
     pour gonfler le WR.
   - Résultat : **gros SL fixe (-1.2 %), petits TP qui rétrécissent** ⇒ R/R réalisé < 1
     ⇒ même un WR de 47 % ne couvre pas l'asymétrie + les 0.5 % de frais.

### Le « pic d'intelligence »
- **WR** : pic à **200k** (47.1 %).
- **Return / expectancy** (le seul critère qui compte) : « le moins pire » = **160k**
  (return -0.37 %, expectancy -0.159 %), mais aucun n'a d'edge.
- **⇒ Il n'existe pas (encore) de checkpoint rentable.** Le modèle optimise le WR au
  détriment du R/R : c'est de l'**over-fitting au reward** (WR↑) sans valeur économique.

---

## 3. DIAGNOSTIC FINAL

Les deux analyses (structurelle « il s'est corrigé » et mathématique « il perd ») sont
**toutes deux vraies** :
- ✅ Le comportement s'est assaini (AGENT_CLOSE 0.7 %, SL/TP réalistes, tenue de position).
- ❌ La traduction économique est négative car la **structure SL/TP est perdante** :
  SL plafonné à 1.2 % vs TP qui rétrécit sous 0.6 %, le tout grevé de 0.5 % de frais.

**Le blocage n'est pas « le PPO n'apprend pas » — c'est « il apprend à maximiser un WR
qui ne peut pas être rentable avec ces bandes SL/TP + ces frais ».**

---

## 4. RECOMMANDATIONS (mesurées, pas devinées)

1. **SL adapté à la volatilité (ATR), pas fixe** : un SL 0.6 % avec ATR > 0.6 % se fait
   sortir par le bruit. → tester `--stochastic-sltp` (calibrateur ATR×régime déjà codé).
2. **Élargir la bande TP** ou **abaisser les frais simulés** pour rétablir un R/R ≥ 1.5
   atteignable. (Le fee-gate force TP_min ≥ 0.6 %, mais l'agent choisit le bas de bande.)
3. **Réduire les actions illégales** (`sell_no_position`, `min_notional`) via un
   masque d'action / pénalité ciblée → nettoie le gradient.
4. **Simulation SL « what-if »** (sans réentraîner) : rejouer les trades gagnants pour
   compter combien auraient survécu à SL=1.0/1.2/1.5 % → quantifier le gain potentiel.

---

## 5. PAPER TRADING — conformité vérifiée

| Élément | Entraînement | Bot live (`run_bot.py`) | Conforme ? |
|---|---|---|---|
| Features (21/TF, 5m/1h/4h) | `config.yaml:260-336` | `live_state_builder.TRAIN_COLUMNS` | ✅ identique |
| Normalisation | pré-normalisé StateBuilder (pas de VecNormalize) | géré (clip [-5,5] + StateBuilder) | ✅ |
| Action space | Box(5,) dir/size/tf/sl/tp | décodé idem | ✅ |
| SL/TP bounds | scalper 0.3-1.2 % / 0.5-2.0 % | `execution_engine._PROFILE_BOUNDS` | ✅ **si `--profile scalper`** |
| Frais | 0.50 % A/R | simulés 0.50 % | ✅ |
| Tiers / exposition | `config.yaml capital_tiers` | chargés depuis config | ✅ |
| Slippage | 2 bps | 2 bps directionnel | ✅ |

⚠️ **Lancer impérativement avec `--profile scalper`** (sinon bandes intraday = NON conforme).
⚠️ Le modèle actuel **n'a pas d'edge** → le paper trading sert à **valider la chaîne
technique** (conformité, exécution), pas à espérer un profit.
