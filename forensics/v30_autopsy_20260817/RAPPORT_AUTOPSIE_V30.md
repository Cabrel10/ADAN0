# RAPPORT D'AUTOPSIE V30 — ADAN0
**Date** : 2026-08-17 | **Run** : V30 PPO 500k (complété 02:08, cum=500 224) | **Freeze** : forensics/v30_autopsy_20260817/ (git 229c485, SHA256)

---

## VERDICT FINAL : **NO-GO**

Le checkpoint V30 500k est une policy **morte** : `direction = +1.0000` constante, `a0_std = 0.0000`, verrouillée depuis le step ~27 502 (5.5% du run). Inutilisable en production, paper ou live. Backtest 500k : **NO_EDGE** (332 trades, WR=42.8%, expectancy=−0.0395%/trade, PF=0.7743, return=−1.31%). Backtest 450k : identique (mêmes métriques à 0.01% près — preuve que tout ce qui suit ~28k n'a rien appris).

---

## CHAÎNE CAUSALE COMPLÈTE (step → conséquence portfolio)

```
[ARCHITECTURE REWARD]
  capacity_reward: FLAT=−1.5 constant | OPEN(60-90%)=+2.0   (env:6525-6542)
  → swing reward de 3.5 entre flat et open, INDÉPENDANT de l'observation
  → sans levier, capacity_usage 60-90% ⇔ "position ouverte à 76.6%"
  → DONC: "être en position" = +2.0, "être flat" = −1.5
        ↓
[STEP 0 → 10k]  policy naïve explore (a0_mean=0.01→0.17, std=0.37)
  découvre: BUY→position ouverte→+2.0 > HOLD→flat→−1.5
        ↓
[STEP 10k → 23k]  gradient PPO pousse a0 vers +1 (a0_mean: 0.17→0.91)
  dir>0%: 70%→98% | KL encore normal (0.001-0.011) | EV sain (0.2-0.7)
        ↓
[STEP 23 552]  PREMIÈRE RUPTURE MESURABLE = crash EV critic (−1.96)
  a0 atteint 0.98 → variance d'advantage → 0 → le critic ne peut plus
  prédire la valeur d'une policy quasi-constante
        ↓
[STEP 24k → 27.5k]  derniers ajustements (a0: 0.98→1.0000)
  KL monte (0.008→0.045) | clip monte (0.35→0.58)
        ↓
[STEP ~27 502]  VERROUILLAGE TOTAL
  a0 = +1.0000 EXACT, a0_std = 0.0000, dir>0% = 100.0%, dir<−0.33% = 0.0%
  la policy ne répond plus à AUCUNE observation
        ↓
[STEP 28k → 500k]  PPO POST-MORTEM (472k steps = 94.5% du run GASPILLÉS)
  KL explose: 0.195→0.68 (jusqu'à 4.5× target_kl=0.15)
  clip_fraction: 0.77→0.91 (mean run = 0.81)
  833 KL early-stops sur 976 updates (85%)
  PPO pousse des updates massives sur un gradient nul
  → realized_pnl cumulatif = −399.91 | 7465 opens/closes | equity finale $20.50
        ↓
[CONSÉQUENCE PORTFOLIO]  saignée lente confirmée (pas choc ponctuel)
  7 443 cycles ouverture→fermeture
  delta moyen = −0.0268 | seulement 29.2% positifs | durée médiane = 7 steps
  cycle: BUY(+2.0 capacity) → petite perte → force_close/stop_loss → réouverture immédiate
  PV flat: std=2.02 (n=119k) | PV open: std=1.89 (n=47k)
  → le PV bouge AUSSI à flat (empiètement des fenêtres flat entre clôture et
    réouverture immédiate) — pas de bug comptable, mais exposition quasi permanente
```

---

## CAUSES

### 🔴 CAUSE RACINE #1 — `capacity_reward` attracteur BUY binaire (CONFIRMÉ)
**Fichier** : `src/adan_trading_bot/environment/multi_asset_chunked_env.py:6525-6542`
**Config** : `config.yaml:1284-1288` — `capacity_weight: 0.1` (réactivé S15+ ; S15 l'avait désactivé "killing exploration" — la réactivation a recréé l'attracteur)

```python
if 0.6 <= capacity_usage <= 0.9:  reward += 2.0      # OPEN
elif capacity_usage < 0.3:        reward -= (0.3 - capacity_usage) * 5  # FLAT → −1.5
```

**Preuve forensique** (166 836 échantillons, capacity_pv_analysis.txt) :
| État | capacity_reward | n | % |
|------|----------------|---|---|
| FLAT | **−1.5000 constant** | 117 542 | 70.5% |
| OPEN | **+1.9997 ≈ +2.0** | 47 294 | 28.3% |

**Mécanisme** : l'agent apprend que `BUY → position ouverte → +2.0` bat `HOLD → flat → −1.5`. Le PnL réel (pnl_reward mean ≈ −0.002, ~0.1% du signal) ne pèse rien face à ce swing de 3.5. La policy converge vers "toujours BUY" pour capturer le bonus capacité, pas pour gagner de l'argent. reward ≈ invalid+capacity dans chaque bucket (rew=−0.1151 vs inval=−0.1161 à 0k ; rew=−0.1244 vs inval=−0.1239 à 58k).

**Confiance** : HAUTE (95%) — mesure directe 166k échantillons + code source + config.

### 🔴 CAUSE RACINE #2 — Garde-fou `ADAN_ANCHOR_LAMBDA` documenté mais JAMAIS activé (CONFIRMÉ)
**Fichier** : `src/adan_trading_bot/environment/multi_asset_chunked_env.py:514-530`
**Statut V30** : `ADAN_ANCHOR_LAMBDA=0.0` (défaut OFF) — 0 occurrence `[ANCHOR]` dans v30_500k.log (1 seul match "anchor" = bannière)

Le code contient explicitement la contre-mesure (2026-07-06, docs/EMPIRICAL_BEHAVIOR_ANALYSIS.md) :
> *"on no-op steps the reward is INDEPENDENT of a0 magnitude (grad ≈ 0). a0=+0.25 and a0=+1.0 both get ~+0.0007. So a0_mean drifts freely under noise → directional collapse (pct_buy→1.0). This adds a SYMMETRIC quadratic pull toward a0=0."*

C'est **exactement** le collapse observé. Le garde-fou existait, était documenté, jamais exporté dans l'environnement V30. Même classe d'erreur que `ADAN_CRITIC_BREAKER` (garde-fou présent dans le code, absent du run).

**Confiance** : HAUTE (90%) — grep du log + code commenté + variable d'env absente.

### 🟠 CONTRIBUTING #1 — `behavior_invalid_penalty` asymétrique (CONFIRMÉ)
`config.yaml:1415` : `sell_while_flat: −0.28`. Reward effectif ≈ inval à ~99% dans tous les buckets. Punit l'intention SELL-à-plat même quand l'action exécutée est HOLD (hold_score=0.94), jamais BUY-à-plat → biais directionnel acheteur supplémentaire. Ironie documentée dans le code (env:8362) : le fix DIAGNOSTIC-V4 visait à *symétriser* ce penalty pour éviter la saturation BUY — la version V28 additive a réintroduit l'asymétrie.

### 🟠 CONTRIBUTING #2 — Dataset pathologiquement court (CONFIRMÉ)
V30 log : **1 seul chunk**, BTCUSDT seul. 5m=7 991 lignes (~28 jours), 1h=912 (~38j), 4h=521 (~87j). 500k steps sur ~1 mois de données = répétition massive → exploitation du reward au lieu d'apprentissage de régimes. Un seul régime de marché vu.

### 🟠 CONTRIBUTING #3 — PPO sans frein sur policy morte (CONFIRMÉ)
`target_kl=0.15` n'a pas empêché KL de monter à 0.68. 833 early-stops (85%) mais l'entraînement a continué 472k steps après la mort de la policy. Aucun collapse breaker sur `a0_std→0` ou `|a0_mean|→1`. std figée à 0.368→0.369 (plancher), entropy figée à −2.10 — la gaussienne d'exploration ne bouge plus non plus.

### 🟠 CONTRIBUTING #4 — Sizing paper découplé + context_vector dégradé (CONFIRMÉ, mineur)
`decode_action` (execution_engine.py:286-307) : `size_pct` vient de `exp_min+(exp_max−exp_min)×HMM_confidence` (LINEAR_EXPO, env:6908), PAS de la tête size du modèle (size=−1.0 saturée mais ignorée). Paper : `run_bot.py:413` passe `context_vector=None` → LiveStateBuilder émet uniform 1/17 → confidence clip [0.01,0.99] → cv[3]≈0.0588→0.0588... → size=76.6% constant. Explique le "76.60%" figé, PAS la saturation directionnelle. L'env training construit le vrai context_vector via DBE (env:6040-6055 : HMM posteriors [3-5] + oracle [14-16]) — asymétrie réelle mais non causale pour le collapse.

---

## FAUX COUPABLES (testés et réfutés)

| Hypothèse | Verdict | Preuve |
|-----------|---------|--------|
| **H1: Parité obs train/paper cause la saturation** | **RÉFUTÉ** | Saturation reproduite à l'identique sur 450k ET 500k dès tick 1, checkpoints différents. La policy émet +1.0 constant **en entraînement aussi** (reward JSONL, dès 28k). L'asymétrie context_vector/scalers existe mais ne cause pas le collapse — le modèle est déjà mort. |
| **"KL trop haut = cause"** | **RÉFUTÉ** | KL normal (0.001-0.011) pendant toute la dérive (0-23k). KL n'explose (0.195→0.68) qu'APRÈS le verrouillage a0 (~27.5k). KL = SYMPTÔME du PPO poussant sur gradient nul. |
| **"EV crash @23.5k = cause première"** | **RÉFUTÉ** | Première rupture *mesurable* dans les métriques PPO, mais la dérive a0 commence dès step 0 (0.011→0.91 sur 23k steps). EV crash = conséquence de a0→0.98. |
| **"Saturation = dérive de checkpoint"** | **RÉFUTÉ** | 450k et 500k saturent identiquement dès tick 1. Structurel, pas checkpoint-spécifique. |
| **Backtest "332 trades variés = modèle sain"** | **RÉFUTÉ** | Le backtest utilise le même env (gates, force_close, stop_loss) qui transforment a0=+1.0 constant en séquences BUY/hold/force_close variées. La variété vient de l'env, pas du modèle. |
| **Backtest "53 789 RESETs = anomalie"** | **RÉFUTÉ** | Artefact de comptage (tqdm `\r` + logs CRITICAL dupliqués). Réalité : 4 épisodes réels pour 10k steps (≈2500 steps/épisode, cohérent avec le chunk test). Backtest sain sur ce point. |

---

## TESTS CONTREFACTUELS + ATTENTION HEADS (chaîne causale bouclée)

### Test contrefactuel (counterfactual_obs_eps.txt) — POLICY = FONCTION CONSTANTE
```
a0=+1.000000 pour TOUTES les entrées :
  obs réelles (idx 500/4000/7000)     → a0=+1.0
  obs marché TOUTES à zéro            → a0=+1.0  (Δ=0.000000)
  obs ALÉATOIRES N(0,1)               → a0=+1.0  (Δ=0.000000)
  perturbation ±0.5σ chaque bloc      → Δa0=0.000000 (5m/1h/4h/context/portfolio)
```
**Conclusion** : le réseau ne lit plus AUCUNE observation. Le +1.0 vient d'un biais interne saturé du réseau (pre-tanh logit profondément positif). ACTION DECODER audit CLOS : +1.0 = policy, pas decoder. Paper-replay parity clos par transitivité (action constante ⇒ replay==paper forcément).

### Audit attention heads (attention_head_audit.txt) — ENCODEUR VIVANT, TÊTE MORTE
```
Architecture: features_extractor.cross_attention = HierarchicalCrossAttention
             (MultiheadAttention + LayerNorm×3 + FFN GELU/Dropout)
latent (256-dim) std across 5 obs différentes: mean=0.149 max=0.585
latent pairwise diff (obs500 vs obs7000): max|Δ|=1.61
*** ENCODER ALIVE: latents vary — collapse is in policy head only ***
```
**Conclusion** : l'encodeur/attention discrimine correctement les états de marché. Seule la tête de policy est morte (biais saturé). Le collapse est LOCALISÉ à la couche de sortie, pas à l'encodeur — le capacity_reward a saturé la tête sans détruire la représentation (bonne nouvelle pour la réutilisabilité de l'encodeur).

### CHAÎNE CAUSALE FINALE (bouclée)
```
obs (variées) → encodeur/attention (SAIN, latents variés) → latent 256d (discriminant)
  → TÊTE POLICY (MORTE: biais saturé → a0=+1.0 constant)
  → decoder (BUY quand flat, hold figé quand open)
  → reward dominé par capacity (+2.0 open vs −1.5 flat) → PPO renforce → NO_EDGE
```

---

## HIÉRARCHIE DES CRITÈRES DE SUCCÈS (EV ≠ ligne d'arrivée)
Correction méthodologique majeure intégrée : l'EV positif n'est que le **feu vert n°1**, pas le succès. ADAN doit satisfaire SIMULTANÉMENT :
```
EV>0 → PnL>0 → WR>60% → PF≥seuil → DD acceptable → palier franchi → OOS robuste → > Buy&Hold
```
**V30 échoue à TOUS les niveaux** (pas seulement EV). Le capacity_reward explique le COLLAPSE comportemental, mais ne démontre PAS qu'ADAN corrigé atteindrait l'objectif global — c'est la question ouverte pour V31.

---

## VÉRIFICATION LIGNE↔STEP (correction méthodologique intégrée)
- Ligne 1 : step=1, episode=0 | Ligne 27 502 : step=3682, episode=0 (step = compteur par épisode, reset périodique)
- Ligne 500 224 : step=4, episode=0 → les lignes SONT 1:1 avec les steps globaux worker_0 (500 224 lignes = 500 224 steps = total_timesteps du run)
- `last_unsaturated_line=27 502` (a0=0.803 à cette ligne) → verrouillage entre 27 502 et 28 000 steps globaux ✓
- Transitions flat→open mesurées = 7 443 ≈ 7 465 opens du résumé terminal → `positions.count` fiable ✓

---

## CORRECTION MINIMALE PROPOSÉE (post-autopsie)

**Ne PAS relancer V31 = V30 + tweak.** Corriger les 2 causes racines + collapse breaker :

1. **Neutraliser l'attracteur capacité** : `capacity_weight: 0.0` (ou refonte : bonus proportionnel au PnL positif réalisé, jamais binaire flat/open). **Critique.**
2. **Activer l'ancre** : `ADAN_ANCHOR_LAMBDA=0.01` (deadzone 0.30) — le garde-fou documenté existe, il suffit de l'exporter. **Critique.**
3. **Collapse breaker précoce** : tuer le run si `|a0_mean|>0.85` ET `a0_std<0.05` pendant 5 updates consécutives. Détecte à ~25k au lieu de gaspiller 500k.
4. **Symétriser/neutraliser le penalty invalide** : selon C6 (env:8348) "penalizing rejected actions violates Bellman" — traiter BUY-à-plat comme SELL-à-plat, ou 0 pour les deux.
5. **Dataset** : multi-régimes, multi-actifs, multi-chunks avant tout 500k. Walk-forward temporel (jamais shuffle). Objectif V31-50k : BUY/SELL/HOLD tous > 5%, pas le profit.

**Test de réfutation** : run diagnostic 50k avec corrections 1+2+3. Si `a0_std>0.05` et `dir<−0.33% > 5%` à 50k → corrections efficaces. Si re-saturation → cause plus profonde (auditer tête dir du réseau + init).

---

## AUTO-CRITIQUE MÉTHODOLOGIQUE (erreur de hiérarchisation)
`capacity_reward=−1.5` était visible dès le dump du schéma reward (step 1) mais classé "anomalie secondaire" au lieu d'hypothèse causale prioritaire. Trois biais : (1) capture par la chronologie KL/EV — recherche d'un *événement* de rupture alors que c'était une *pente* continue dès step 0 ; (2) lecture de la constante sans lire immédiatement `calculate_capacity_based_reward()` (l.6525) ; (3) temps alloué à la piste parité obs alors que la contre-preuve (saturation tick 1 sur 2 checkpoints) la réfutait déjà. Leçon : toute composante de reward CONSTANTE et DOMINANTE doit être auditée en premier — c'est un attracteur potentiel par construction.

---

## MÉTHODOLOGIE (doctrine respectée)
- ✅ Forensic freeze SHA256 avant toute modification (git 229c485)
- ✅ Timeline : 976 updates PPO + 833 early-stops + reward JSONL 500 224 lignes
- ✅ Rupture detection affinée : 50k → 2k → step exact (27 502)
- ✅ Vérification ligne↔step
- ✅ Hypothèses testées, pas argumentées (H1 parité RÉFUTÉE par contre-preuve)
- ✅ OBSERVATION→TEST→RÉSULTAT→CONCLUSION→CONFIANCE pour chaque cause
- ✅ Aucune modification de code production pendant l'autopsie (R6)

**Artefacts** : `timeline/ppo_updates.json` (976), `timeline/early_stops.json` (833), `timeline/collapse_fine_2k.txt`, `timeline/reward_buckets_50k.txt`, `timeline/reward_component_stats.txt`, `timeline/capacity_pv_analysis.txt`, `backtest/backtest_v30_450k_test.json`, `logs/validation/backtest_v30_500k_test.json`
