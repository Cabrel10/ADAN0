# RAPPORT — Sonde MIROIR SPOT PUR (achat ET vente, jamais short)

- Script : `scripts/diagnostics/diag_spot_mirror_exit.py` (commit 0d7caea)
- Log : `logs/spot_mirror_run.log`
- JSON : `logs/validation/spot_mirror_20260916_171847.json`
- Date : 2026-09-16 · Univers : `_BINANCE` uniquement (garde-fou `_asset_guard`) · Durée : ~15 s

---

## 1. Pourquoi cette sonde existait

Trois faits déjà mesurés rendaient ce test obligatoire avant toute autre dépense :

1. **`fee_sensitivity`** : `EVbrut == BE_cost` partout ; meilleur budget de frais OOS = +0,041 %
   (DOGE B2 test). Il n'y a quasi pas d'edge brut long à protéger — la piste « baisser les
   frais » est structurellement morte pour l'entrée longue seule.
2. **`mtf_confluence_v2`** : monotonie **négative** (BTC b1=−0,0039, p=0,0005 ; DOGE
   b1=−0,0059, p=0,011) — plus la confluence MTF haussiere est forte, plus l'EV baisse.
3. **Angle mort** : toutes les sondes (R1→R4, MFE/MAE, LOW-MAE, fee_sensitivity) étaient
   **long-only**. Le constat N6 « l'EV suit le signe du B&H » était presque une tautologie.

Contrainte d'univers décidée : **TRADING SPOT PUR** — achat et vente de l'actif détenu,
**jamais de short à découvert**. Un signal baissier ne s'encaisse donc pas en vendant à
découvert : il vaut comme **signal de VENTE** (valeur = −E[r_fwd]) ou de non-achat. La
stratégie spot complète est le cycle ACHETER (signal haussier) → VENDRE (signal baissier) →
FLAT. C'est exactement ce cycle qui n'avait jamais été mesuré hors PPO.

## 2. Protocole pré-enregistré (avant exécution)

- **Q1 — valeur du signal de vente** : cross_down MACD 5m + MTF baissier (miroir exact de
  MTF_ALIGNE), stride 40, EV_vente = −E[r_fwd] à **H=288 (24h) fixé a priori** ;
  2 actifs × 3 splits = **6 tests**. Contrôles UP_ALIGNE et BLIND + statistique N6-proof
  (BLIND − DOWN > 0).
- **Q2 — stratégie spot complète** : machine à états séquentielle, entrée
  cross_up+MTF_ALIGNE à close[t], sortie au premier de {cross_down 5m, invalidation
  ema_100_ratio_4h < 1, SL 3×ATR (C2 seulement), timeout 2016 barres} ; variantes
  C1 (sans SL) / C2 (SL 3×ATR) ; **8 cellules** (2 variantes × 2 actifs × {val,test})
  jugées à **0,10 % RT** par translation exacte du PnL brut.
- **α = 0,05/14 = 0,003571** (Bonferroni sur l'espace total rapporté, même discipline que
  le 0,05/12 de fee_sensitivity). GO exige : EV>0 ET IC95_bas>0 ET p<α ET **n≥200** ET
  **val ET test** (Q1 exige en plus le signe cohérent sur train).

## 3. Résultats Q1 — le signal de VENTE ne vaut rien

EV_vente (24h) du bucket DOWN+BAISSIER, n largement suffisants (694–3776) :

| Actif | train | val | test | Verdict |
|---|---|---|---|---|
| BTC | −0,053 % (p=0,68) | −0,214 % (p=0,27) | +0,072 % (p=0,61, IC95bas=−0,19 %) | **NO-GO** |
| DOGE | −0,337 % (p=0,13) | +0,043 % (p=0,94) | +0,004 % (p=0,98) | **NO-GO** |

Deux faits plus forts encore que le NO-GO :

- **Le signal baissier ne prédit même pas la baisse.** Sur 4 des 6 cellules, le rendement
  forward après cross_down+MTF baissier est **positif** (BTC val : +0,214 % !). Le miroir
  exact du signal d'achat n'a aucune valeur prédictive de vente — symétrie absente.
- **Il ne bat jamais l'aveugle.** DELTA(BLIND − DOWN) a un IC95 qui contient 0 sur les
  6 cellules (ex. BTC train : +0,128 %, IC95=[−0,13 %, +0,39 %]). Ce n'est pas un artefact
  de régime : même rapporté au B&H du split, le signal DOWN n'apporte rien.

## 4. Résultats Q2 — la stratégie spot complète est morte par le turnover

| Variante | Actif | n (val/test) | EV **brut** /trade | BE cost | EV @ 0,10 % RT | hold médiane |
|---|---|---|---|---|---|---|
| C1 | BTC | 2249 / 1830 | +0,006 % / +0,009 % | idem | −0,094 % / −0,091 % (p=0,000) | **9 barres (~45 min)** |
| C1 | DOGE | 1350 / 871 | +0,052 % / +0,003 % | idem | −0,049 % / −0,097 % | **9 barres** |
| C2 | BTC | 2249 / 1830 | +0,006 % / +0,010 % | idem | −0,094 % / −0,090 % (p=0,000) | 9 barres |
| C2 | DOGE | 1350 / 871 | +0,053 % / +0,010 % | idem | −0,047 % / −0,090 % | 8 barres |

**8 cellules sur 8 FAIL**, y compris à coût nul en termes de signe robuste : le meilleur
EV brut observé (+0,053 %, DOGE val) est ~2× sous le seuil de déblocage de 0,10 % RT et
meurt en test (+0,010 %). Mécanisme : **cross_down déclenche la vente en ~45 minutes
médianes** → 9840 allers-retours sur le train BTC → chaque trade ne dégage que ~0,01 %
brut. Le cycle achat-vente spot complet est un scalper hyper-actif dont l'edge brut par
trade est 2 à 10× sous le coût maker parfait. Le SL 3×ATR (C2) ne change rien (le SL ne
touche presque jamais avant le cross_down).

## 5. Verdict pré-enregistré

> **NO-GO global.** Ni Q1 (0/2 actifs) ni Q2 (0/8 cellules) ne passent le protocole
> (α=0,003571, val ET test, n≥200). **L'architecture OHLCV/5m/spot est close DANS LES
> DEUX DIRECTIONS**, long comme court-terme-vente. L'audit β/α devient inutile : avec
> EVbrut ≈ +0,01 %/trade et un β nécessairement positif en long-only, il ne ferait que
> documenter un zéro déjà mesuré deux fois.

Conséquences sur le débat « trois analyses » :
- Analyse 1 (régime-conditionnel) : **infirmée** — même en libérant la sortie (vente sur
  signal miroir au lieu de boîtes TP/SL), l'edge brut reste ~0.
- Analyse 2 (méthodologie β/α) : correcte comme garde-fou, mais **court-circuitée** par
  BE_cost et maintenant par cette sonde.
- Analyse 3 (fee_sensitivity) : **confirmée et renforcée** — ce n'est pas « les frais
  mangent l'edge », il n'y a pas d'edge brut dans aucune des deux directions.

## 6. Ce qui reste (classé honnêtement)

1. **Microstructure** (Δ-OI, funding, taker imbalance, liquidations, basis spot/perp) :
   seul espace d'information réellement non testé. Coût : download API + plusieurs jours.
   Rien ne garantit le résultat — mais c'est le seul endroit où une réponse positive est
   encore possible.
2. **Horizon multi-jour** : change l'univers de données ; les frais relatifs tombent 20-50×.
   Compatible spot pur, mais c'est un autre problème que le 5m.
3. **Arrêt propre de la branche OHLCV/5m/spot.** Les NO-GO s'empilent désormais dans un
   ordre qui ne laisse plus de porte de sortie à cette architecture :
   R3 (aucune cellule cohérente) → MFE/MAE (les runners existent mais ne sont pas
   détectés) → LOW-MAE (N3 échoué, AUC<0,60) → fee_sensitivity (pas d'edge brut) →
   **spot_mirror (pas d'edge dans l'autre direction non plus)**.

**Non-options** : pas de R7+, pas de relance PPO, pas de reward shaping, pas de
desserrage du seuil Bonferroni a posteriori.
