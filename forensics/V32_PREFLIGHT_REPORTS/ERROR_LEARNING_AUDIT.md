# RAPPORT 4 — ERROR_LEARNING_AUDIT

**Objet** : pour chaque type d'erreur de la politique, mesurer les trois
niveaux de traitement :
- **P1** — pénalité immédiate (le signal négatif atteint-il la politique ?)
- **P2** — correction comportementale (l'agent change-t-il d'action après ?)
- **P3** — persistance de l'erreur (l'erreur disparaît-elle sur le run ?)

**Source** : `forensics/learning_radar_v31_20260819/L2_DEEP.md` (matrices de
transition, réaction aux pénalités — 8821 lignes ACTION_DIFF dédupliquées),
`RADAR.md`, `ABSORPTION_QUANTIFIED.md`, + code d'injection env 8655-8694.
Chiffres réels du run V31 500k. Aucune estimation.

---

## 1. Typologie des erreurs (états observés)

| Erreur | Définition | Détectée où |
|--------|------------|-------------|
| **E-inv-SELL-flat** | intent SELL alors que FLAT (rejeté → HOLD) | routing `routing_reject/sell_while_flat`, env 8656 |
| **E-inv-BUY-open** | intent BUY alors que position OUVERTE (rejeté → HOLD) | routing `routing_reject/buy_while_long`, env 8656 |
| **E-deadband** | `|a0| ≤ thr` → aucune action franchie | `deadband_reject`, env 8653 |
| **E-collapse** | politique saturée (μ≈-8) → un seul signe échantillonné | ANCHOR_DEBUG (nB=nH=0), L2_DEEP |

---

## 2. P1 — la pénalité immédiate atteint-elle la politique ? → **OUI (CONFIRMÉ)**

Mesure L2_DEEP « Réaction aux pénalités d'invalidité » :

| Contexte | change d'action au step suivant | répète | n |
|----------|-------------------------------|--------|---|
| **Après pénalité** | **44.7 %** | 55.3 % | 320 |
| Après neutre | 2.7 % | 97.3 % | 8500 |
| **Delta** | **+42.0 pts** | | |

**Verdict P1 = CONFIRMÉ** : une pénalité multiplie par ~16 la probabilité de
changer d'action au step suivant. Le signal de pénalité **atteint** la politique
(le gradient de correction existe localement). *(Note : ceci a été mesuré quand
la pénalité valait encore ≠0 ; en V31 finale sell_while_flat/buy_while_open=0.0,
donc P1 n'est plus déclenché par ces deux erreurs — voir §5.)*

---

## 3. P2 — la correction comportementale se produit-elle ? → **LOCALEMENT OUI, GLOBALEMENT NON**

- **Localement** : le +42 pts (§2) EST une correction comportementale locale.
- **Globalement (échelle run)** : REFUTÉ. Matrices de transition L2_DEEP :

```
H1 (début) HOLD -> BUY 2.6% | HOLD 95.9% | SELL 1.5%   (n=4150)  ← 3 actions vivantes
H2 (fin)   HOLD -> BUY 0.0% | HOLD 100.0%| SELL 0.0%    (n=4410)  ← plus qu'une seule
```

En fin de run, **seul HOLD→HOLD survit** dans ACTION_DIFF (vue pipeline), tandis
que le rollout PPO ne contient que des SELL saturés (nB=nH=0). La correction
locale ne peut plus **s'exprimer** car l'action alternative n'est plus jamais
échantillonnée.

**Verdict P2 = correction présente mais ABSORBÉE** (L2_DEEP : « le gradient de
correction existe mais ne peut plus s'exprimer dans l'action échantillonnée »).

---

## 4. P3 — l'erreur persiste-t-elle ? → **OUI (persistance CONFIRMÉE)**

Mesure L2_DEEP « évitement SELL stérile » : **REFUTÉ** que l'agent évite l'erreur.

```
P(SELL | stérile au step t-1) = 12.6 %
baseline P(SELL)               =  1.1 %
```

→ Après une erreur SELL-stérile, la probabilité de la RÉPÉTER (12.6 %) est
**~11× la baseline** (1.1 %). L'erreur ne s'auto-corrige pas : elle **persiste
et se renforce**. C'est la signature de l'état absorbant.

Corroboration L4 (RADAR.md) : `share_SELL=1.0 dès upd≈368`, `advBUY_nan=100 %`,
`a0=-8.07`, exécution effective 0.32 % (95.5 % routing_reject).

---

## 5. Fait clé : le paradoxe P1↔P3 (pourquoi la pénalité échoue)

| Niveau | Résultat | Chiffre |
|--------|----------|---------|
| P1 (signal reçu) | ✅ OUI | +42 pts |
| P2 (correction locale) | ✅ OUI (mais absorbée) | 44.7 % vs 2.7 % |
| P2 (correction globale) | ❌ NON | H2 : 100 % HOLD→HOLD |
| P3 (erreur éliminée) | ❌ NON, persiste | P(rép)=12.6 % vs 1.1 % |

**Le signal d'erreur est correctement reçu et déclenche une réaction locale,
mais devient inopérant à l'échelle du run** parce que la saturation tanh (μ≈-8)
réduit la diversité d'échantillonnage à zéro. Formellement (ABSORPTION_QUANTIFIED)
`P(BUY)/échantillon = 1 − Φ((atanh(thr) − μ)/σ)` devient < 1/2048 pour μ≲-1.1 →
aucun BUY/HOLD n'entre dans le buffer PPO → `adv_BUY = adv_HOLD = NaN` → plus
aucun contre-exemple → la correction comparative devient mathématiquement
impossible.

---

## 6. Chaîne causale bout-en-bout (CONFIRMÉE, L2_DEEP)

```
marché baissier tôt (78400→66400, -0.46% closes)
  → SELL statistiquement avantageux tôt
  → gradient pousse μ vers négatif
  → tanh sature (μ < -3) : diversité BUY/HOLD → 0 dans le sampling
  → buffer PPO : nB=0, nH=0 → adv_BUY=adv_HOLD=NaN
  → plus AUCUN contre-exemple → correction comparative impossible
  → anchor L2 (λ=0.05) borne μ en [-9.2, -7.7] mais loin de 0
  → état absorbant stable mais stérile : 95.5% routing_reject, 0.32% exec
```

---

## 7. Synthèse — implications pour le RAL

1. **Le problème n'est PAS que l'agent n'apprend pas de ses erreurs** (P1/P2 local
   confirmés). Le problème est **la disparition du contre-exemple** dans le
   buffer (nB=nH=0). C'est un problème de **diversité d'échantillonnage**, pas de
   magnitude de pénalité.
2. **Corollaire décisif pour le RAL** : moduler des pénalités/récompenses NE
   restaure PAS la diversité d'échantillonnage. Ce que le RAL (ou le fix V32)
   doit garantir, c'est que **P(BUY) et P(HOLD) restent > 1/N_batch** (μ borné
   près de 0, σ contrôlée) — condition L4/diversité, pas condition reward.
3. Le futur radar live (RAPPORT 5) doit exposer **P(BUY)/P(HOLD)/P(SELL) par
   échantillon** et **nB/nH/nS dans le buffer** comme signaux L4 de danger
   d'absorption (détection AVANT que adv devienne NaN).
4. Détection d'absorption opérationnelle : `s4 < θ_danger` sur K épisodes ⇔
   `min(P(BUY),P(HOLD),P(SELL)) < 1/N_batch` persistant.
