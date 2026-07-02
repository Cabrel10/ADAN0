# AUDIT V10 — Autopsie complète du run 500k (collapse à 70k)

**Date de reprise** : 2026-07-02, ~7h après le lancement V10.
**Statut du run** : ARRÊTÉ par le disjoncteur à **70 000 steps** (sur 500k visés).
Durée : 9272 s (~2h35). 52 trades exécutés. Checkpoint final : `ppo_adan0_sandbox_70000steps.zip`.

---

## 1. Résumé exécutif

Le run V10 (PPO standard + fix V9 + fix C1 + gSDE off) a **collapsé vers 100 % BUY**, plus TÔT
que le v8 historique (70k vs 128k). Le disjoncteur `DiagnosticCollapseCallback` a correctement
détecté et stoppé (`pct_buy=0.975 streak=2 -> BREAKER`).

**MAIS l'autopsie renverse l'hypothèse de travail précédente sur deux points majeurs :**

1. **Le critic APPREND.** `explained_variance` n'est PAS coincée à 0 — elle MONTE clairement de
   ~0 (bruyant au début) à un plateau stable **~0.30 sur le dernier tiers** (pics à 0.46). C'est
   le **Verdict A** de la grille d'audit : le signal C3a est exploité par le critic.

2. **Le fix V9/C1 fonctionne mécaniquement mais est HORS-SUJET.** `sterile_pen` est toujours
   NÉGATIF (max magnitude -0.055) et escalade correctement. Mais il ne freine rien car il n'est
   pas le moteur du collapse. Le collapse démarre dès **step ~4000**, quand `illegal_ratio` n'est
   encore que 0.25 — donc la majorité des BUY à ce stade sont des BUY LÉGAUX récompensés, pas des
   BUY illégaux pénalisés. Le warmup C1 (15k) est non pertinent : le biais est déjà à pct_buy=0.68
   à 10k.

**Le vrai moteur** : la policy converge vers « toujours demander BUY / rester en position » parce
que le critic a correctement appris que **rester en position a de la valeur** (via
`latent_pnl_contrib` + la structure du reward), et la policy maximise cette valeur. Ce n'est PAS
un bug de signe de pénalité — c'est un **problème de fonction de récompense qui récompense
structurellement le maintien de position long**, sur des données 5m pourtant BAISSIÈRES.

---

## 2. Chronologie (diag CSV, toutes les 2000 steps)

| Step | a0_mean | a0_std | pct_buy | pct_sell | illegal | entropy | Phase |
|---|---|---|---|---|---|---|---|
| 2000  | 0.003 | 0.134 | 0.471 | 0.459 | 0.217 | -0.581 | sain (équilibré) |
| 4000  | 0.025 | 0.136 | 0.556 | 0.379 | 0.256 | -0.581 | **dérive commence** |
| 10000 | 0.073 | 0.132 | 0.677 | 0.272 | 0.282 | -0.579 | biais BUY installé |
| 20000 | 0.127 | 0.136 | 0.804 | 0.166 | 0.415 | -0.577 | collapse en cours |
| 30000 | 0.169 | 0.132 | 0.885 | 0.088 | 0.520 | -0.571 | avancé |
| 50000 | 0.218 | 0.136 | 0.944 | 0.041 | 0.648 | -0.554 | quasi total |
| 70000 | 0.285 | 0.143 | 0.975 | 0.021 | 0.786 | -0.536 | **BREAKER** |

**Signatures :**
- **a0_std PARFAITEMENT STABLE ~0.13-0.14** sur tout le run → le fix gSDE-off a marché, le bruit
  d'exploration n'a PAS divergé. C'est le MEAN (a0_mean) qui dérive, pas la variance.
- Dérive **monotone et graduelle** de a0_mean (0.003→0.285) et pct_buy (0.47→0.975). Pas de saut
  brutal : glissement continu = la policy suit un gradient de reward cohérent, pas une instabilité.
- `illegal_ratio` monte 0.22→0.79 : conséquence (agent sur-exposé demande des BUY bloqués), pas
  cause (le biais précède la montée de illegal_ratio).
- `entropy` (diag) bouge à peine (-0.58→-0.54). SB3 `entropy_loss` reste PLAT à 2.68 → **illusion
  de l'espace continu** : σ reste large (~0.13) donc SB3 croit l'entropie haute, mais le MEAN a
  tellement glissé vers BUY que 97 % des échantillons décodent BUY malgré σ.

## 3. Collapse : type, quand, pourquoi

- **Type** : BUY collapse (distribution + action). PAS un entropy collapse (σ stable), PAS un
  critic collapse (EV monte), PAS une divergence numérique (value_loss borné 0.02-1.06).
- **Quand** : démarre ~4k, irréversible ~30k, total ~70k.
- **Pourquoi** : gradient de policy cohérent vers BUY. Le critic valorise le maintien de position ;
  la policy exploite. `anti_spam_hold` déclenché **2824 fois**, CASH_FLOOR_B **582 fois**, mais
  seulement **52 trades exécutés** → l'agent entre une position tôt puis SPAMME BUY (bloqué) au
  lieu de trader. Il ne scalpe plus, il « HODL » une position unique.
- **Gravité** : maximale (policy inutilisable), mais **récupérable** car la cause est identifiée
  et c'est un problème de reward-shaping, pas d'architecture cassée.

## 4. Le critic APPREND (Verdict A) — preuve chiffrée

`explained_variance` par update PPO (135 updates sur 70k steps) :
- Updates 1-33 (0-~17k) : très bruyant, -0.70 à +0.40, moyenne ~0.05.
- Updates 34-90 (~17k-47k) : se stabilise 0.15-0.35.
- **Updates 95-135 (~50k-70k) : 0.20-0.46, moyenne ~0.30, stable.**

`value_loss` borné et non-divergent (0.02-1.06, pics ponctuels). Le critic est numériquement sain
et prédit de mieux en mieux la valeur. **CONCLUSION : signal existe (C3a) ET critic l'exploite
(EV→0.30). La chaîne CNN→critic N'EST PAS aveugle.** → C3b (shuffle CNN) devient INUTILE : on a
déjà la preuve positive que le critic apprend.

## 5. Le paradoxe central (le vrai enseignement de V10)

**Le critic apprend bien ET la policy collapse en même temps.** Ce n'est pas contradictoire :
le critic apprend correctement un paysage de reward où **BUY/hold domine**. Sur les données 5m
(BAISSIÈRES -24 %), récompenser le maintien de position long via `latent_pnl_contrib` +
composantes pro-position pousse le critic à valoriser BUY, et la policy à ne faire que ça — en
PERDANT de l'argent (portefeuille 20.0 → 16.3, **-18 %**).

→ Le problème n'est NI le collapse-de-signe (V9), NI le warmup (C1), NI l'extraction (CNN OK),
NI le critic (apprend). **Le problème est que la fonction de récompense elle-même a un optimum
en « rester long », désaligné avec l'objectif « scalper profitable sur marché baissier ».**

## 6. Données : incohérence de périodes entre timeframes (à investiguer, hors scope collapse)

- 5m : -24.3 % (baissier) | 1h : +469.9 % | 4h : +255.6 %. **Les 3 TF ne couvrent PAS la même
  période.** Le 5m (TF de décision du scalper) est récent/baissier ; 1h/4h sont un long bull
  historique. Le CNN reçoit donc des contextes multi-TF potentiellement incohérents
  temporellement. À vérifier séparément — ce n'est pas la cause directe du collapse mais ça
  pollue le signal.

---

## 7. Confondeurs (Phase 2.1) — écartés

| Confondeur | Constat | Verdict |
|---|---|---|
| clip_range_vf | à vérifier mais EV monte quand même | non bloquant |
| gae_lambda / gamma | value_loss borné, EV monte | non responsable |
| taille réseau value | EV atteint 0.30 → capacité suffisante | non responsable |
| échelle du reward | symlog appliqué (final = sign·log1p(|raw|)) | compresse déjà |
| extraction CNN | EV monte → CNN fournit un signal exploitable | **CNN OK** |

Aucun confondeur d'hyperparamètre n'explique le collapse. Le critic apprend (EV↑). Donc **C3b
(shuffle CNN) est inutile** : la preuve positive d'extraction existe déjà.

## 8. Analyse du moteur pro-BUY (reward-shaping)

`raw_reward = pnl_base + promotion_bonus + demotion_penalty + closure_bonus + drawdown_penalty
+ symmetry_penalty + action_entropy_penalty + future_contrib + latent_pnl_contrib + saturation_penalty`.

- `promotion_bonus` : sur tier de capital, ne fire pas sur run perdant → pas le moteur.
- `future_contrib` : seulement sur trades FERMÉS (52 en 70k) + plafonné → pas le moteur par-step.
- `latent_pnl_contrib` : par-step, asymétrique gain 0.10 / perte 0.15 (anti-long en théorie).
- **Moteur le plus probable** : `steps_open_pct` monte 0.66→0.85 dans le diag. **Être EN POSITION
  génère un flux de petits signaux par-step (latent_pnl sur les 50.4 % de bougies positives),
  tandis qu'être FLAT ne génère RIEN** (survival_bonus et patience_bonus ont été retirés par les
  devs — le flat est devenu du « temps mort » sans reward). Le gradient pousse donc vers « rester
  en position », et la seule action pour y entrer/y rester est BUY. Le collapse BUY est la
  conséquence logique d'un reward qui n'a **aucun signal positif pour l'état FLAT**.

## 9. DÉCISION ARCHITECTURE (Phase 8)

**Option retenue : #2 — Reprendre un run PPO standard MODIFIÉ (correction reward-shaping ciblée),
PAS WorldModelPPO, PAS DSpark.**

Justification technique :
- **PAS WorldModelPPO** : il force l'extraction d'un signal temporel. Or l'extraction FONCTIONNE
  déjà (EV→0.30). WorldModelPPO résoudrait un problème qu'on n'a pas. Une variable à la fois.
- **PAS DSpark** : calibrer la confiance d'une policy qui collapse n'a aucun sens tant que la
  policy elle-même a un optimum désaligné.
- **Le vrai fix** = rééquilibrer le reward pour que l'état FLAT ne soit pas « mort ». Le collapse
  n'est pas une instabilité, c'est une CONVERGENCE vers un optimum réel mais indésirable. Il faut
  changer l'optimum, pas ajouter de l'architecture.

## 10. Prochaines actions (par priorité)

1. **[P0] Neutraliser le fix C1/warmup comme piste morte** — documenté : ce n'était pas le moteur.
   Ne PAS le réactiver comme « solution », il reste en place (inoffensif) mais hors-sujet.
2. **[P0] Corriger l'asymétrie flat/position dans le reward** — UNE variable : soit réintroduire
   un signal minimal pour l'état flat quand rester flat est correct (ex: petit bonus si le marché
   baisse et qu'on est flat), soit réduire/annuler `latent_pnl_contrib` positif (garder seulement
   la pénalité de perte latente). À trancher avec l'utilisateur (touche au reward → justification
   isolée obligatoire). **NE PAS toucher aux frais.**
3. **[P1] Investiguer l'incohérence de périodes 5m vs 1h/4h** — le CNN voit des TF de périodes
   différentes. Potentiellement un bug de pipeline data qui fausse tout.
4. **[P1] Ajouter la télémétrie des composantes de reward** dans le log (par-step, échantillonné)
   pour PROUVER quel terme domine, au lieu de l'inférer. Indispensable avant le prochain 500k.
5. **[P2] disk_guard : bug de self-match du pattern** (voir commit dédié) — resserrer le pgrep.
6. **[P2] monitor_v10 : bug de sortie prématurée** — il est mort à 00:46 (5 min après lancement)
   car lancé avant que le training ait le pattern ; à rendre robuste (attendre l'apparition du
   process avant d'armer la boucle de sortie).

**Le run V10 n'est PAS un échec** : il a produit la preuve la plus importante de tout l'audit —
le critic APPREND (EV→0.30), donc le problème n'a jamais été l'extraction ni un bug de signe,
mais l'ALIGNEMENT de la fonction de récompense. C'est un diagnostic actionnable.
