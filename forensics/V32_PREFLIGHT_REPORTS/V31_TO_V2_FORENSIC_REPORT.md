# RAPPORT 7 — V31_TO_V2_FORENSIC_REPORT (LE PLUS IMPORTANT)

**Objet** : comparer V30 et V31 **ligne par ligne** sur la fonction de reward et
les forces agissant sur μ, pour établir **ce que V30 avait et que V31 a perdu**
(et inversement), afin que V32 ne perde AUCUNE des deux protections.

**Source** : `RAPPORT_AUTOPSIE_V30.md` (+ADDENDUM RC1 révisé),
`RAPPORT_COLLAPSE_V31_500K.md`, config.yaml, feature_extractors.py, env
`_calculate_reward`. Tous les chiffres sont des mesures figées (SHA256).

---

## 0. Résumé exécutif (le fait central)

- **V30 = always-BUY** (a0=+1.0000, std=0, verrou ~step 27502). Cause **RC1
  révisée** : pénalité d'intention invalide V28 **asymétrique** (FLAT+SELL et
  OPEN+BUY punis ~0.23 de plus) → gradient pousse a0→+1. Backtest NO_EDGE
  (WR 42.8 %, PF 0.7743).
- **V31 = always-SELL** (μ→-8.58, σ→108). Cause : on a **neutralisé la pénalité
  asymétrique** (sell_while_flat/buy_while_open : -0.28 → **0.0**) MAIS on a
  activé **gSDE (σ non bornée)** et **retiré la force de rappel sur μ** →
  μ dérive librement vers -8.58.

**V30 avait une force qui bornait μ (par accident : la pénalité asymétrique
tirait a0 vers +1, donc μ restait fini côté +). V31 a retiré cette force SANS la
remplacer par une force de rappel LÉGITIME (ancre L2) → μ a fui côté −.**
Le collapse a simplement **changé de direction**, pas de nature.

---

## 1. Comparaison ligne par ligne — fonction de reward

| Terme reward | V30 | V31 | Effet du changement |
|--------------|-----|-----|---------------------|
| `behavior_penalty.sell_while_flat` (config 1426) | **-0.28** | **0.0** | supprime le driver always-BUY (bon) MAIS retire une force qui bornait a0 |
| `behavior_penalty.buy_while_open` (config 1434) | **-0.28** | **0.0** | idem, côté OPEN |
| `buy_flat_opening` (config 1440) | 0.0 | 0.0 | inchangé (BUY ouvrant = valide) |
| `capacity_reward` (env télémétrie) | FLAT=-1.5/OPEN=+2.0 mais **β=0 dans raw** (RÉFUTÉ, addendum V30) | idem, télémétrie morte | **aucun** (jamais dans le gradient, prouvé régression 40k, résidu 1e-6) |
| `action_anchor_penalty` (env 7382) | plafonné 0.02, dead-zone 0.30 | idem | inchangé, borné, non déterminant |
| `final_reward = symlog(raw)` (env 7456) | oui | oui | inchangé (écrase les pénalités, RAPPORT 3 §6) |
| Somme `raw_reward` (env 7400) | 15 termes | 15 termes identiques | **structure de reward quasi identique** |

**Conclusion reward** : la fonction de reward V30→V31 n'a changé QUE sur les deux
coefficients `behavior_penalties` (-0.28 → 0.0). Le reste est identique. **Donc
le collapse V31 ne vient PAS d'un nouveau terme de reward** — il vient de la
**suppression d'une force + ajout d'instabilité côté POLITIQUE (loss actor)**.

---

## 2. Comparaison ligne par ligne — forces sur μ (loss actor)

| Force sur μ | V30 | V31 | V32 requis |
|-------------|-----|-----|-----------|
| Pénalité asymétrique (borne a0 côté +) | **ACTIVE** (-0.28) | supprimée | ne PAS restaurer (biaise) |
| Ancre L2 `λ·(μ²).mean()` (feat_ext 1330) | `ANCHOR_LAMBDA=0` **jamais activé** (RC2 CONTRIBUTING) | **absente** au lancement V31 | **ADAN_L2_ANCHOR_LAMBDA=0.05 OBLIGATOIRE** |
| Distribution | DiagGaussian (`use_sde=0`) | **gSDE (`use_sde=1`)** → σ non bornée | **ADAN_USE_SDE=0 OBLIGATOIRE** |
| Clamp `log_std` [-5,+2] (ppo_safety) | effectif (DiagGaussian) | **inopérant** (gSDE module σ hors clamp) | effectif si use_sde=0 |
| SatGuard | présent | présent mais **contre-productif** (14 bumps ↑ σ) | conservé (redevient utile si DiagGaussian) |
| ent_coef floor | 0.02 | 0.02 (récompense σ élevé) | conservé mais surveillé |

**LE FAIT LE PLUS IMPORTANT** : V30 n'avait **aucune ancre L2 active** (RC2 :
`ANCHOR_LAMBDA=0`), mais μ restait borné **par accident** grâce à la pénalité
asymétrique qui tirait a0 vers +1 (μ fini positif). V31 a retiré cet effet de
bord SANS activer l'ancre L2 ET en ajoutant gSDE → **plus aucune force ne borne
μ** → dérive libre vers -8.58.

---

## 3. Ce que V30 AVAIT et que V31 a PERDU (la question centrale)

| # | V30 possédait | V31 l'a perdu | Conséquence |
|---|---------------|---------------|-------------|
| P1 | une **force nette qui bornait a0** (pénalité asymétrique, même si biaisée) | supprimée sans remplacement | μ libre de fuir |
| P2 | **DiagGaussian** (σ bornée par clamp) | remplacé par gSDE (σ explose) | σ 108 |
| P3 | un collapse **côté +** (a0=+1, au moins des BUY exécutés, 332 trades) | bascule côté − (SELL stérile, 0.32 % exec) | exécution quasi nulle |

Et **ce que V31 a bien fait** (à conserver) :
- neutraliser la pénalité asymétrique (fin du biais always-BUY) — **correct**,
  justifié par C6 (rejet = pas de transition s→s′ = reward 0.0).

**Donc V32 doit combiner** : la neutralisation V31 (pas de biais) **+** une force
de rappel LÉGITIME que ni V30 ni V31 n'avaient réellement : **ancre L2 active
(λ=0.05) + DiagGaussian**. C'est exactement le paquet de corrections démontré au
bas du rapport V31 (validé gate V16-final, tag v16-final, checkpoints 300-320k).

---

## 4. Calibration de l'ancre : leçon d'ABSORPTION_QUANTIFIED

L'ancre L2 λ=0.05 en V31 (activée tardivement en analyse) a stabilisé μ dans
**[-9.2, -7.7]** — STABLE mais au mauvais endroit. Or l'absorption survient déjà
à **μ≈-1.145** (E[nB]≈1), pas à -8. Donc :

- Un clamp/ancre qui borne μ à ±3 est **INUTILE** (E[nB]=0 dès μ=-3, prouvé).
- Pour garder la diversité, il faut **|μ| ≲ 0.7-0.8** (E[nB]≫1).
- **Implication V32** : λ=0.05 seul ne garantit pas |μ|≲0.8 ; il faut soit un λ
  suffisant pour cibler μ≈0, soit un mécanisme complémentaire (plancher
  d'exploration / reset de μ au franchissement d'un seuil de diversité `s4`).
  → c'est le rôle du radar live L4 (RAPPORT 5) + du hook (RAPPORT 6 §6).

---

## 5. Table de vérité V30 / V31 / V32 (cible)

| Paramètre | V30 | V31 | **V32 (cible)** |
|-----------|-----|-----|-----------------|
| sell_while_flat / buy_while_open | -0.28 | 0.0 | **0.0** (garder V31) |
| ADAN_L2_ANCHOR_LAMBDA | 0 | 0 (au lancement) | **0.05 (min), viser |μ|≲0.8** |
| ADAN_USE_SDE | 0 | **1** | **0** (DiagGaussian) |
| Clamp log_std [-5,+2] | effectif | inopérant | **effectif** |
| SatGuard | on | on (nuisible) | **on (utile en DiagGaussian)** |
| Radar live L4 (s4) | absent | absent | **présent (early-stop diversité)** |
| Résultat | always-BUY | always-SELL | **diversité maintenue (cible)** |

---

## 6. Verdict et pré-conditions V32 (issues des 7 rapports)

1. **La fonction de reward n'est PAS le problème** : V30 et V31 la partagent à 2
   coefficients près. Le problème est **la force sur μ + la stabilité de σ**
   (loss actor), pas le shaping.
2. **V32 doit** : garder la neutralisation V31 (pas de biais) + activer l'ancre
   L2 + DiagGaussian + radar L4 live borné sur la diversité.
3. **Le RAL, s'il est codé, ne doit JAMAIS être la protection anti-absorption** :
   il agit sur le reward (aval du symlog, RAPPORT 6 R4) et ne rouvre pas la
   boucle (RAPPORT 4/6). Son rôle légitime = modulation QUALITÉ (PF/diversité)
   **tant que s4 est sain**, et **déclenchement du levier loss-actor** (ancre/σ)
   quand s4 menace — jamais une pénalité reward de plus.
4. **Test déterministe obligatoire AVANT V32 500k** : les 6 combinaisons
   état×action doivent donner un reward sans biais directionnel (comme exigé par
   l'addendum V30), ET μ doit rester borné |μ|≲0.8 sur un mini-run.

---

## 7. Cohérence inter-rapports (validation croisée)

| Rapport | Fait clé | Cohérent avec R7 ? |
|---------|----------|--------------------|
| R1 EV_INVENTORY | ev_norm non effectif (β via RewardCalculator) | ✅ reward pas le driver |
| R2 EV_DEPENDENCY | BUY gaté / SELL non gaté = arête causale des 2 collapses | ✅ |
| R3 PENALTY_AUDIT | seule behavior_penalty bouge a0 ; symlog écrase | ✅ |
| R4 ERROR_LEARNING | P1 ok, mais absorption = perte de diversité (nB=0) | ✅ problème = diversité |
| R5 RADAR_V2 | s4 = min(P)·N_batch, par update | ✅ détecte R1/R7 tôt |
| R6 PIPELINE | R1 fatale = diversité ; levier = loss actor | ✅ |
| **R7** | reward identique V30/V31 ; diff = force sur μ + σ | ✅ **synthèse cohérente** |

**Les 7 rapports convergent** : la cause n'est pas le reward mais la
**diversité d'échantillonnage / la force sur μ**. Le RAL doit être subordonné au
radar L4 et ne jamais se substituer au levier loss-actor. **GO conditionnel pour
coder le RAL selon cette contrainte, puis V32 500k après test déterministe.**
