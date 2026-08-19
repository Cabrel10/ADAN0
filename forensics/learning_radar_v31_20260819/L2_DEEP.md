# L2 APPROFONDI — Apprentissage d'erreurs (transitions + réaction aux pénalités)

Données : ACTION_DIFF dédupliqué (8821 lignes, double-logging logger+print éliminé).
Read-only, artefacts V31 figés.

## Matrices de transition des actions REQUESTED

### H1 (début du run)
```
BUY  -> BUY 26.2% | HOLD 61.0% | SELL 12.8%  (n=164)
HOLD -> BUY  2.6% | HOLD 95.9% | SELL  1.5%  (n=4150)
SELL -> BUY 13.7% | HOLD 73.7% | SELL 12.6%  (n=95)
```
→ Politique exploratoire : les 3 actions transitent, HOLD attracteur mais non exclusif.

### H2 (fin du run)
```
HOLD -> BUY 0.0% | HOLD 100.0% | SELL 0.0%  (n=4410)
```
→ **Seul HOLD→HOLD survit dans ACTION_DIFF**. Combinaison avec ANCHOR_DEBUG (nS=64..2048,
nB=nH=0 dans le buffer PPO) : la tête policy propose SELL saturé, le pipeline route SELL-flat
vers HOLD, et c'est ce HOLD exécuté qui est loggé. Les deux vues sont cohérentes :
policy=SELL saturé → routing→HOLD → ACTION_DIFF(HOLD,HOLD).

⚠️ Rappel méthodologique (validé) : ACTION_DIFF = vue pipeline env ; ANCHOR_DEBUG = contenu
réel du rollout PPO. Ce n'est PAS la même population — la jointure des deux est précisément
ce qui prouve le mécanisme d'absorption (policy sature SELL, l'env neutralise en HOLD, PPO
ne voit que des SELL dans son buffer).

## Réaction aux pénalités d'invalidité (inv_penalty < 0)

| Contexte | change d'action | répète | n |
|----------|----------------|--------|---|
| Après pénalité | **44.7%** | 55.3% | 320 |
| Après neutre | 2.7% | 97.3% | 8500 |
| **Delta** | **+42.0 pts** | | |

### Verdict : [CONFIRME] L2.réaction_aux_pénalités
L'agent apprend localement de ses erreurs d'invalidité : une pénalité multiplie par ~16
la probabilité de changer d'action au step suivant. Le signal de pénalité ATTEINT la politique.

### Verdict : [CONFIRME] L2.mais_apprentissage_absorbé
Cet apprentissage local est rendu inopérant à l'échelle du run : la saturation tanh (μ≈-8)
réduit la diversité d'échantillonnage à zéro, donc aucune action alternative ne peut être
sélectionnée même quand la pénalité dit « change ». Le gradient de correction existe mais
ne peut plus s'exprimer dans l'action échantillonnée.

## Chaîne causale consolidée (CONFIRMEE bout-en-bout)

```
marché baissier tôt (78400→66400, -0.46% mesuré sur closes)
  → SELL statistiquement avantageux tôt
  → gradient pousse μ vers négatif
  → tanh sature (μ < -3) : diversité BUY/HOLD → 0 dans le sampling
  → buffer PPO : nB=0, nH=0 → adv_BUY=adv_HOLD=NaN
  → plus AUCUN contre-exemple : la correction comparative devient impossible
  → seul l'anchor L2 (λ=0.05, ≈2.6) borne μ en [-9.2, -7.7] — équilibre loin de 0
  → état absorbant stable mais stérile : 95.5% routing_reject, 0.32% exec
```

## Ce que le radar réfute / confirme (récapitulatif)

| Sujet | Statut | Évidence |
|-------|--------|----------|
| L1 flux conséquences | CONFIRME (partiel) | 219 closes dédup avec PnL/fees/hold/reason par trade |
| L2 réaction pénalités | CONFIRME | +42pts de changement après pénalité vs neutre |
| L2 évitement SELL stérile | REFUTE | P(SELL|stérile t-1)=12.6% >> baseline 1.1% — la répétition persiste |
| L2 réduction taux SL | PROBABLE | SL% H1=60.8% → H2=41.5% |
| L3 adaptation vol→fréquence | PROBABLE | corr=0.831 ; fréquence/win chute 36→3 avec la vol et le collapse |
| L4 collapse absorbant | CONFIRME | share_SELL=1.0 dès upd≈368, advBUY_nan=100%, a0=-8.07 |
| L4 spam stérile | CONFIRME | fenêtre 9 : 406k policy → 32 exec (0.01%) |
| L5 performance | CONFIRME (négative) | WR 23.5%→21.8%, PF moyen 0.29, Sharpe indicatif -8.81 |

## Scores radar consolidés (0-100)

```
L1 conséquences  : 15.8   (flux existe mais exploitation faible)
L2 erreurs       : 40.9   (réaction locale OUI, évitement global NON)
L3 environnement : 83.1   (corrélation vol/fréquence forte — à confirmer hors collapse)
L4 cohérence     :  2.1   (diversité nulle, état absorbant)
L5 performance   : 33.3   (WR 22%, PF 0.29, Sharpe -8.8)
```

Note méthodologique : L3=83 est un artefact partiel du collapse (la fréquence chute quand
la politique meurt, pas seulement quand la vol change). À re-mesurer sur un run sain.
