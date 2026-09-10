# ADAN — RAPPORT MFE/EXCURSION : « le TP fixe détruit-il les runners ? » (2026-09-10)

Document de verdict. Source de preuve : `logs/validation/mfe_excursion_20260910_193118.json`
(généré 19:31 UTC, run terminé 21:31). Commit du diagnostic : `540c15e`.
Univers : **launcher strict** (`BTCUSDT_BINANCE`, `DOGEUSDT_BINANCE`), garanti par
`scripts/diagnostics/_asset_guard.py` — les chemins + rows sont imprimés dans
`logs/mfe_excursion_run.log` pour chaque split (ex. BTC/train = 662 643 barres 5m, 2017→2023).

## Question testée

Le brief affirme : les MFE à longue échéance sont grands (p99 7j ~ 8-12 %), donc un
TP fixe court (+1,2 %) « coupe les runners » et détruit l'espérance. Trois expériences
pré-enregistrées :

- **EXP1** — distributions MFE/MAE à 1h / 4h / 24h / 48h / 7j, conditions ALL /
  MTF_ALIGNE / MTF_CONTRADICTOIRE.
- **EXP2** — sortie structurelle (SL ATR + invalidation 4h) vs TP fixe :
  - B1 = TP +1,2 % / SL −0,6 % / timeout 24h (la boîte du brief),
  - B2 = SL 0,5×ATR + sortie sur invalidation 4h,
  - B3 = SL 3×ATR + sortie sur invalidation 4h.
- **EXP3** — délai d'entrée 0/2/5/10/20/40 barres (l'edge survit-il à une entrée tardive ?).

Paramètres : fees_rt = 0,004, slippage_rt = 0,001 (coût RT = 0,005), stride = 40 barres.
Seuil de débat : P(MFE24h > 3× coûts) = P(> 1,2 % net).

## EXP1 — Les attentes du brief sont DÉPASSÉES (les runners existent)

Train (mesures vs attentes du brief) :

| Actif | p50 1h (attendu ~0,2 %) | p90 4h (~1,5 %) | p95 24h (~3 %) | p99 7j (~8-12 %) |
|---|---|---|---|---|
| BTCUSDT_BINANCE | **+0,293 %** | **+2,431 %** | **+8,679 %** | **+38,015 %** |
| DOGEUSDT_BINANCE | **+0,397 %** | **+3,610 %** | **+14,356 %** | **+201,846 %** |

Le p99 à 7 jours est **3 à 20× au-dessus** de la borne haute du brief. La matière
première (excursions favorables longues) existe bien dans les données.

P(MFE24h > 3× coûts) — train : BTC aligné 56,9 % / contradictoire 53,7 % ;
DOGE aligné 68,7 % / contradictoire 59,1 %. L'avantage « MTF aligné » est réel mais
modeste (+3 à +10 pts selon l'actif), pas un séparateur binaire.

Concentration (part d'énergie MFE 7j) : BTC top5 % = 20,4 % / top1 % = 5,6 % ;
DOGE top5 % = 51,6 % / top1 % = 31,0 %. DOGE est beaucoup plus « runner-driven ».

## EXP2 — Sortie structurelle vs TP fixe (EV net par trade, bootstrap CI95)

| Actif / split | B1 TP fixe | B2 SL 0,5 ATR | B3 SL 3 ATR + inval. 4h |
|---|---|---|---|
| BTC train | **−0,486 %** (p=0,000) | −0,404 % (p=0,000) | **+0,225 %** (p=0,067) |
| BTC val | −0,484 % (p=0,000) | — | −0,063 % (p=0,748) |
| BTC test | −0,446 % (p=0,000) | — | **−0,648 %** (p=0,000) |
| DOGE train | **−0,566 %** (p=0,000) | +0,329 % (p=0,374) | **+1,910 %** (p=0,004) |
| DOGE val | −0,457 % (p=0,000) | — | +1,064 % (p=0,171) |
| DOGE test | −0,546 % (p=0,000) | — | −0,561 % (p=0,085) |

Lecture :

1. **B1 (TP fixe) est négatif partout, toujours significatif** (p=0,000 sur les 6
   cellules, WR 30-36 %, 60-69 % des sorties sur SL). La boîte actuelle perd
   ~0,45-0,57 % par trade sur l'univers réel. Le constat du brief est CONFIRMÉ.
2. **B3 bat B1 sur train** (+0,23 % BTC, +1,91 % DOGE) — la direction du brief
   (sortie structurelle, laisser courir) est correcte sur l'échantillon d'entraînement.
3. **Mais B3 ne se réplique pas hors échantillon** : BTC val −0,06 % (n.s.),
   BTC test −0,65 % (significativement négatif), DOGE test −0,56 % (n.s.).
   Seul DOGE/val reste positif et non significatif (+1,06 %, p=0,171).
   L'edge « runners » est donc **non robuste au walk-forward** : il vit sur
   2017-2023 (bull structurel) et s'éteint sur les splits récents.

Mécanique B3 (train) : WR 7-10 % seulement, mais p99 = +25 % (BTC) / +104 % (DOGE),
holding médian 42-48 barres 5m, skew 4,1-10,7. C'est un profil de capture de queue —
exactement ce que le brief décrit — mais la queue s'est raréfiée hors train.

## EXP3 — Délai d'entrée : pas d'edge de timing

EV net B1 à 4h quasi invariant au délai (BTC : −0,523 % à d0 → −0,500 % à d40 ;
DOGE : −0,539 % → −0,515 %). MFE 4h médian stable (~0,62-0,65 % BTC, ~0,85-0,88 %
DOGE). **Le signal d'entrée n'a pas de fenêtre critique** : entrer 40 barres plus tard
ne détruit rien, ce qui confirme aussi qu'il n'y a pas d'edge d'entrée court-terme à
protéger — le problème est du côté sortie/coûts, pas du côté timing.

## Verdict

| Sous-hypothèse du brief | Statut |
|---|---|
| Les MFE longs existent et dépassent 8-12 % (p99 7j) | **CONFIRMÉ** (dépassé 3-20×) |
| Le TP fixe +1,2 % a une EV nette négative | **CONFIRMÉ** (6/6 cellules, p=0,000) |
| La sortie structurelle restaure une EV positive | **INFIRMÉ hors train** (train oui, val/test non) |
| Le délai d'entrée explique la perte | **INFIRMÉ** (EV plate en fonction du délai) |

**NO-GO maintenu.** La correction « sortie structurelle » est nécessaire (B1 perd à
coup sûr) mais **pas suffisante** : l'edge train de B3 ne survit pas aux splits
val/test. Modifier le reward PPO sur cette seule base reviendrait à entraîner sur une
anomalie 2017-2023. Conséquences inchangées : reward gelé, fee gate intact, 500k bloqué.

Voies restantes (ordre de leverage mesuré) :
1. **Coûts** : 0,5 % RT mangent 2-4× le MFE 4h médian (0,62 % BTC). Toute réduction
   de `fees_rt`/`slippage_rt` déplace directement l'EV de toutes les variantes.
2. **Horizon/conditionnement** : l'écart MTF_ALIGNE vs CONTRADICTOIRE existe
   (+3-10 pts de P(>3×coûts)) mais est trop faible seul ; à combiner avec un filtre
   de régime (volatilité) plutôt qu'avec la direction seule.
3. **Prior des entrées** : puisque le timing ne coûte rien (EXP3), un prior
   sélectif n'a pas de coût d'opportunité de délai — mais il doit être validé
   walk-forward, ce que B3 a échoué à faire.

## Reproductibilité

```
cd /home/ubuntu/webapp/MORNINGSTAR/ADAN0
python3 scripts/diagnostics/diag_mfe_excursion.py   # ~2h, écrit logs/validation/mfe_excursion_<ts>.json
```

Le diagnostic charge uniquement `*_BINANCE` (garde `_asset_guard.py`) et imprime
`[DATASET] asset=… path=… rows=…` pour chaque timeframe avant toute mesure.
