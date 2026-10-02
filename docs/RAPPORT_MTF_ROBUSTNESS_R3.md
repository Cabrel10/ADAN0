# ADAN — MTF Round 3 : verdict de robustesse (2026-09-10)

## Protocole
- Filtre de consistance : cellule retenue seulement si EV_net > 0 sur train ET val ET test (n >= 30 partout).
- Block bootstrap (2000 réplications, blocs de 5 trades) sur la meilleure cellule de chaque (asset, règle).
- Règles testées : A2_trend_cross, B2_pullback_cross, D2_strict_cross ; H in {40, 100, 288} ; grille ATR k_sl x k_tp ; frais round-trip 0.40 %.

## Résultat : NÉGATIF
Aucune cellule (asset x règle x k_sl x k_tp x H) n'est simultanément :
1. positive sur les 3 splits (train, val, test),
2. significative au bootstrap (IC95 bas > 0).

Assets : BTCUSDT, BTCUSDT_BINANCE, DOGEUSDT, DOGEUSDT_BINANCE — toutes « aucune cellule cohérente ».

## Diagnostic (pourquoi)
- Le biais prédictif MTF est réel (rounds 1-2 : filtered > blind de façon reproductible) mais de l'ordre de quelques bps, très inférieur à la frontière de frais 0.40 %.
- Le cadre règles-discrètes + argmax ponctuel sur grille ne sépare pas signal et bruit : la cellule gagnante change d'un split à l'autre.
- La métrique P(TP) seule est inutilisable ; seule EV(SL,TP,H | état) avec stabilité spatiale compte.

## Conséquence (ordre utilisateur : négatif -> trouver pourquoi -> corriger)
Correction = passage à la méthodologie v2 (probe `diag_mtf_confluence_v2.py`) :
confluence continue c(t)=tanh(Σ w_k z_k) à poids d'information mutuelle (train only),
Edge Ratio (MAE winners vs losers), MFE conditionné à la survie au SL*,
EV normalisée par temps de détention, IC de Wilson (n>=200/cellule),
block bootstrap L=2H B=2000, correction FDR Benjamini-Hochberg,
stabilité spatiale (plateau 3x3, pas argmax), validation croisée BTC<->DOGE,
walk-forward >= 3 fenêtres multi-régimes.

## Garde-fou
Tant que la chaîne confluence -> G_H biaisée -> EV>0 stable n'est pas démontrée par v2 :
AUCUNE modification du reward PPO. Si v2 est négatif aussi : intégration en PRIOR
de policy (masque d'actions / reward shaping par qualité de chemin), pas en stratégie autonome.

Rapport JSON : logs/validation/mtf_robustness_20260910_092159.json
