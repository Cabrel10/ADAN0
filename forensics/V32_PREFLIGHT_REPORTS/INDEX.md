# V32 PREFLIGHT REPORTS — INDEX & VALIDATION DE COHÉRENCE

Les 7 rapports préalables au lancement de V32 (RÈGLE #1 : aucun code
d'entraînement / RAL / modif `_calculate_reward` avant que les 7 rapports soient
produits, relus, validés). Tous factuels (RÈGLE #4 : code lu, valeurs extraites,
logs V30/V31), aucune supposition.

| # | Fichier | Objet | Fait central |
|---|---------|-------|--------------|
| 1 | `EV_INVENTORY.md` | inventaire des EV | 1 seule EV agit sur le contrôle (`resolve_ev_fee_gate`) ; `ev_norm` non effectif |
| 2 | `EV_DEPENDENCY_GRAPH.md` | dépendances entre EV | BUY gaté / SELL non gaté = arête causale des 2 collapses |
| 3 | `PENALTY_AUDIT.md` | audit des pénalités | seule `behavior_penalty` bouge a0 ; symlog écrase les grosses pénalités |
| 4 | `ERROR_LEARNING_AUDIT.md` | P1/P2/P3 par erreur | P1 ok (+42pts) MAIS absorption = perte de diversité (nB=nH=0), erreur persiste (P(rép)=12.6 % vs 1.1 %) |
| 5 | `LEARNING_RADAR_V2_SPEC.md` | radar LIVE | s4 = min(P_BUY,P_HOLD,P_SELL)·N_batch, calculé par update PPO |
| 6 | `REWARD_PENALTY_PIPELINE.md` | pipeline + ruptures | R1 fatale = diversité (μ≈-1.145) ; levier correct = loss actor, PAS reward |
| 7 | `V31_TO_V2_FORENSIC_REPORT.md` | **V30 vs V31 ligne par ligne** | reward identique à 2 coefs près ; diff réelle = force sur μ + σ (gSDE) |

---

## Conclusion de cohérence (validée)

**Les 7 rapports convergent sans contradiction** sur une cause unique :

> Le collapse (V30 always-BUY, V31 always-SELL) n'est PAS causé par la fonction
> de reward (identique à 2 coefficients près entre V30 et V31), mais par une
> **rupture de diversité d'échantillonnage** : μ dérive → `P(action minoritaire)
> < 1/N_batch` (dès μ≈-1.145) → `nB=nH=0` dans le buffer PPO → `adv=NaN` → plus
> de contre-exemple → correction impossible (état absorbant).

**Conséquence directe pour le RAL** (contrainte de conception non négociable) :
1. Moduler des récompenses/pénalités NE rouvre PAS la boucle (aval du symlog,
   n'agit pas sur μ) → un RAL purement reward reproduirait l'échec de l'ancre L2.
2. Le seul levier qui restaure la diversité est côté **loss actor** : ancre L2
   active (λ≥0.05, cible |μ|≲0.8) + DiagGaussian (`use_sde=0`) + clamp log_std
   effectif + radar L4 live avec early-stop sur `s4`.
3. Le RAL est LÉGITIME uniquement s'il : (a) lit `s4` live ; (b) tant que `s4`
   sain, module la qualité (PF/diversité) ; (c) dès que `s4` menace, déclenche le
   levier loss-actor — jamais une pénalité reward de plus.

## Table de vérité V32 (cible, issue R7 §5)

| Paramètre | V30 | V31 | **V32** |
|-----------|-----|-----|---------|
| sell_while_flat / buy_while_open | -0.28 | 0.0 | **0.0** |
| ADAN_L2_ANCHOR_LAMBDA | 0 | 0 | **0.05 (min)** |
| ADAN_USE_SDE | 0 | 1 | **0** |
| Radar live L4 | absent | absent | **présent** |
| Résultat | always-BUY | always-SELL | **diversité maintenue** |

## GO / NO-GO

**GO CONDITIONNEL** pour coder le RAL selon la contrainte ci-dessus, PUIS test
déterministe 6 combinaisons état×action (reward sans biais directionnel + μ borné
|μ|≲0.8 sur mini-run), PUIS V32 500k. Tout écart → NO-GO documenté.

## Livrable causal (preuve de bout en bout)

**`CAUSAL_LEARNING_PIPELINE.md`** — sonde déterministe SANS PPO (`scripts/diagnostics/probe_env_deterministic.py`),
3 campagnes (advisory/gated × sl-tp saturant/non), 3 splits, 5 séquences. Résultats prouvés :
- **Ambiguïté nB/nH/nS résolue par code** : réponse (A) = a0 brut pré-routing (pas exécution).
- **saturation_penalty domine** le reward quand SL/TP saturent (-2.87/40 pas) et est décorrélée du trade ;
  retirée → reward quasi nul (hold/sell = +0.00000) ⇒ **pas de biais structurel**, seul un biais **dataset** subsiste.
- **fee-gate = seule EV de CONTRÔLE** : bloque 40/40 BUY (mode gated).
- **symlog** rend les pénalités sous-linéaires/sous-quadratiques ; **action_anchor plafonne à 0.02** (négligeable).
- **behavior_penalty punit des SELL légitimes** ; **min_hold ignore des SELL légitimes**.

**Verdict : V2/V32 = NO-GO** tant que les corrections §10 ne sont pas validées **par sonde**, une variable à la fois.
