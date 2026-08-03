# EMPIRICAL BEHAVIOR ANALYSIS — data-driven calibration of the Behavior Layer

> Instruction du propriétaire : *"pour les ajustements essaie de voir exactement
> après analyse mes meilleurs optimaux, ne te laisse pas influencer et base-toi
> sur des analyses pas juste sur les propositions."*
>
> Ce document confronte les **seuils PROPOSÉS** (par les critiques IA + le
> propriétaire) aux **valeurs MESURÉES** dans les données réelles d'ADAN0.
> Aucun seuil n'est retenu sans preuve chiffrée. Verdict par ligne.

Sources mesurées :
- `logs/training/diag_manifesto_500k_COLLAPSED_242k.csv` (242 lignes, trajectoire 1k→242k)
- `logs/training/reward_components_manifesto_500k.csv` (n=80, termes de reward par pas)
- `logs/paper/450k_20260623_092905/trades_paper_*.csv` (27 fichiers, **614 trades clôturés réels** du modèle 450k)
- `config/config.yaml` (fees, penalties, min_holding)

---

## 1. La trajectoire du collapse — MESURÉE (pas supposée)

Le collapse n'est **pas** une rupture soudaine. C'est une **dérive linéaire monotone**
du centre de gravité de la politique, à `a0_std` CONSTANT.

| step | a0_mean | a0_std | pct_buy | pct_sell | entropy | req_SELL |
|-----:|--------:|-------:|--------:|---------:|--------:|---------:|
| 1000 | -0.007 | 0.136 | 0.460 | 0.482 | -0.581 | 0.264 |
| 5000 | +0.042 | 0.133 | 0.601 | 0.346 | -0.581 | 0.209 |
| 10000| +0.143 | 0.136 | 0.833 | 0.133 | -0.579 | 0.090 |
| 11000| +0.157 | 0.130 | **0.869** | 0.103 | -0.579 | 0.073 |  ← franchit 0.85
| 15000| +0.256 | 0.137 | 0.971 | 0.024 | -0.578 | 0.014 |
| 20000| +0.443 | 0.144 | **1.000** | 0.000 | -0.574 | 0.000 |  ← saturation
| 242k | +2.358 | 0.142 | 1.000 | 0.000 | -0.535 | 0.000 |

**Faits chiffrés (non négociables) :**
1. **a0_std reste plat ~0.135 tout du long** → ce N'EST PAS un collapse bimodal
   (l'ancien V3 explosait à 13.48). C'est une dérive **directionnelle** du `mean`.
   → Le verdict web ne détectait que le mode bimodal ; corrigé (commit e2fd718).
2. **L'entropie bouge à peine** (-0.581 → -0.535 sur 242k) → la politique ne
   devient pas "pointue/confiante", elle **déplace son centre**. Toute mesure
   de santé basée UNIQUEMENT sur l'entropie est aveugle à ce collapse.
3. **Fenêtre saine = 1k–11k UNIQUEMENT.** L'intervention doit mordre AVANT 11k.
   Passé 20k, pct_buy=1.0 est absorbant (aucun gradient de sortie).

---

## 2. Les termes de reward qui atteignent RÉELLEMENT l'agent (n=80)

mean|.| sur l'échantillon collapsé :

| terme | mean | mean\|.\| | nonzero | verdict |
|-------|-----:|--------:|:-------:|---------|
| pnl_base | -0.00484 | 0.00484 | 7/80 | fire seulement aux clôtures (rares) |
| latent_pnl | **+0.00018** | 0.00031 | 65/80 | **POSITIF + minuscule** → récompense le hold |
| symmetry_penalty | -0.00022 | 0.00022 | 6/80 | négligeable |
| future_contrib | 0 | 0 | 0/80 | **MORT** |
| promotion_bonus | 0 | 0 | 0/80 | **MORT** |
| demotion_penalty | 0 | 0 | 0/80 | **MORT** |
| closure_bonus | 0 | 0 | 0/80 | **MORT** |
| drawdown_penalty | 0 | 0 | 0/80 | **MORT** |
| action_entropy_penalty | 0 | 0 | 0/80 | **MORT** |
| saturation_penalty | 0 | 0 | 0/80 | **MORT** |
| holding_cost | 0 | 0 | 0/80 | **MORT** |
| smart_flat | 0 | 0 | 0/80 | **MORT** |
| time_decay | 0 | 0 | 0/80 | **MORT** |

**Fait chiffré :** en régime de hold, **2 signaux seulement** atteignent l'agent :
`pnl_base` (rare) et `latent_pnl` (constant, positif, +0.00018). Les **11 autres
termes sont à zéro absolu**. Le "vocabulaire" de reward est vide pendant le hold.
→ Confirme : ajouter un 14ᵉ terme par-pas est vain tant que les 11 existants
sont déconnectés. Le problème n'est pas le nombre de termes, c'est leur silence.

**config.yaml a DÉJÀ un vocabulaire (mais débranché) :** `overstay_penalty:-1.0`,
`missed_penalty:-0.5`, `early_exit_bonus`, `take_profit_bonus`, `stop_loss_penalty`,
`duration_penalty_weight` (L.1453-1469). Tous mesurés à 0 dans le reward réel.

---

## 3. Les trades RÉELS (614 cycles, modèle 450k) vs les seuils PROPOSÉS

| grandeur | PROPOSÉ | MESURÉ (614 trades réels) | verdict |
|----------|---------|---------------------------|---------|
| durée d'un trade | [20, 300] steps | **médiane 1.01 bar, 96% < 1.5 bar**, max 30 | ❌ proposé irréaliste : le modèle churn à 1 step |
| R-multiple cible | R ≥ 1.5 | **mean R = -0.018**, p75 = +0.006, max +0.044 | ❌ jamais observé ; R≥1.5 est fictif ici |
| capture ratio | 66% | non atteignable (trades 1-bar) | ❌ non mesurable sur ces données |
| winrate | — | **25.4%** (156/614) | modèle 450k = **perdant net** |
| exit reason | mix SL/TP/AGENT | **100% AGENT_CLOSE**, 0 SL, 0 TP | le 450k s'auto-coupe toujours |

**Conclusion majeure (data-driven) :** les receipts paper du modèle 450k sont
ceux d'un **modèle perdant hyper-churn** (25% winrate, R négatif, 1-bar). Les
utiliser comme "trader de référence" **enseignerait à perdre**. Les seuils
proposés (durée [20,300], R≥1.5, capture 66%) ne correspondent à **aucune donnée
réelle** d'ADAN0 — ils viennent d'un idéal théorique, pas d'un optimum observé.

---

## 4. La distribution NATURELLE de l'agent (ancre data-driven)

Fenêtre pré-dérive (steps 1000-4000), AVANT que le gradient ne corrompe :

| mesure (routé, `req_*`) | valeur | | mesure (tête brute, `a0_*`) | valeur |
|-------------------------|-------:|-|-----------------------------|-------:|
| req_HOLD_pct | **0.679** | | a0_pct_buy | 0.508 |
| req_SELL_pct | **0.267** | | a0_pct_sell | 0.434 |
| req_BUY_pct | **0.055** | | a0_pct_hold_band | 0.058 |
| | | | a0_std | 0.140 |
| | | | a0_mean | +0.012 |

**Distinction critique que les propositions ignorent :** il y a DEUX
distributions. La tête d'action brute (`a0_*`) est ~équilibrée buy/sell. Mais le
**comportement EXÉCUTÉ** (`req_*`, après routing/masking) est **HOLD 68% / SELL
27% / BUY 5%**. Le routing convertit les BUY-while-long en HOLD (no-op).

→ La cible "disciplinée" proposée `pct_buy ∈ [0.15,0.40] ET pct_sell ∈ [0.15,0.40]`
est basée sur la mauvaise variable. Sur le comportement EXÉCUTÉ, la seule
distribution que l'agent produit naturellement quand il est sain est
**~68/27/5**, pas symétrique. Un régularisateur de discipline doit viser CETTE
ancre mesurée (ou son voisinage), pas un idéal symétrique jamais observé.

---

## 5. Ce que l'analyse impose au design de la Behavior Layer

Décisions **fondées sur les mesures** ci-dessus (pas sur les propositions) :

1. **Anchor de discipline = distribution EXÉCUTÉE mesurée (`req_*`), pas la tête
   brute, pas un idéal symétrique.** Cible de référence provisoire = voisinage de
   HOLD∈[0.55,0.85], SELL∈[0.10,0.40], BUY∈[0.02,0.25] (englobe l'observé sain
   68/27/5 avec marge). À re-calibrer sur le PREMIER run sain qu'on obtiendra.

2. **La métrique de collapse doit être `a0_mean` + one-sided `pct_buy/sell`, PAS
   `a0_std` ni l'entropie** (mesuré : tous deux plats pendant le collapse).
   → déjà appliqué au dashboard (Mode-2). À reproduire dans le diag CSV training.

3. **Fenêtre d'action = 1k–11k.** Tout signal comportemental doit être calculé
   sur fenêtre glissante COURTE et actif tôt, sinon il arrive après l'absorption.

4. **Ne PAS importer les seuils R≥1.5 / durée[20,300] / capture 66%** : aucun
   support empirique. Les garder comme *métriques de monitoring* (calculées,
   loggées) mais **jamais comme reward** tant qu'un run sain n'a pas fourni leurs
   distributions réelles. (Cohérent avec le consensus "monitoring ≠ reward".)

5. **Rebrancher/mesurer d'abord les 11 termes morts** avant d'en créer un 12ᵉ.
   Le vrai problème mesuré = silence des termes existants, pas leur absence.

6. **Le "trader de référence" ne peut PAS venir des receipts 450k** (perdant).
   Il doit être soit (a) synthétique/analytique (oracle de règles), soit (b)
   ré-estimé sur le premier run non-collapsé. Décision reportée à après le
   prochain run, faute de données saines.

---

## 6. Verdict global de l'analyse

- Le collapse manifesto est **directionnel** (mean drift), pas bimodal, pas
  entropique. Mesuré, reproductible, onset ~11-20k.
- Le reward réel pendant le hold = **2 signaux** (pnl_base rare + latent positif
  minuscule) ; **11 termes morts**. Vocabulaire vide, pas insuffisant en nombre.
- **Aucune donnée réelle d'ADAN0 ne supporte les seuils proposés** (R, durée,
  capture, symétrie 15-40). La seule ancre empirique disponible est la
  distribution exécutée pré-dérive **HOLD 68% / SELL 27% / BUY 5%**.
- Prochaine étape légitime **data-driven** : instrumenter (monitoring) les
  métriques comportementales SANS les mettre en reward, lancer un run, mesurer
  leurs vraies distributions, PUIS calibrer. On ne fige aucun seuil à l'aveugle.
