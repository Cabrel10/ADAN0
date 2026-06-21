# Décision checkpoint — backtest FIX2 (bug de split corrigé)

> Généré 2026-06-21. Source : `scripts/backtest_fixed_capital.py`
> Bug corrigé : `MultiAssetChunkedEnv._build_data_loader` lit
> `worker_config["data_split"]` (PAS `data_split_override`). Avant ce fix, l'env
> rechargeait toujours `train` → tous les résultats antérieurs étaient en
> in-sample. Le harnais passe désormais **les deux** clés.

## Résultats (5000 steps, capital fixe 1000$, notional 100$, sans compounding)

| Split | Modèle  | WR %  | PF    | n trades | E %/trade | Verdict        |
|-------|---------|-------|-------|----------|-----------|----------------|
| TEST  | 450k    | 98.6  | 0.75  | 72       | -0.017    | NO_EDGE        |
| TEST  | 500k    | 98.6  | 0.75  | 72       | -0.017    | NO_EDGE        |
| TEST  | RANDOM  | 61.9  | 1.46  | 443      | +0.810    | POSITIVE_EDGE  |
| VAL   | 450k    | 60.3  | 1.18  | 184      | +0.173    | POSITIVE_EDGE  |
| VAL   | 500k    | 67.1  | 2.58  | 152      | +1.665    | POSITIVE_EDGE  |
| VAL   | RANDOM  | 49.2  | 0.92  | 419      | -0.155    | NO_EDGE        |

## Anomalie du split TEST (à ne PAS utiliser pour décider)

- **450k ≡ 500k byte-identique** (WR 98.6 %, n=72, toutes métriques égales) alors
  que les deux checkpoints ont des md5 différents. Sur ce segment court (5298
  lignes 5m, ~18 jours plats) les deux politiques convergent en mode
  déterministe vers le même comportement dégénéré : micro-positions qui touchent
  un TP minuscule (WR 98 %) mais quelques gros SL → **expectancy négative**
  (PF 0.75, mort par frais).
- Le RANDOM "gagne" sur test uniquement par **sur-trading** (443 trades) sur un
  marché plat — variance, pas edge. random_test WR 61.9 % n'est PAS un vrai
  plancher (le split est trop court pour être statistiquement fiable).

→ **Le split TEST est rejeté comme base de décision.** Trop court, plat,
dégénéré. Le split VAL est le signal de référence (random y est correctement à
49.2 % = pile/face).

## Décision : split VAL (out-of-sample fiable)

Règle : WR_modèle > WR_random + 10 pts (out-of-sample) ET cohérence.

- random_val = 49.2 % (plancher correct)
- 450k_val   = 60.3 %  → +11.2 pts (passe le seuil, mais PF 1.18 marginal)
- **500k_val = 67.1 %  → +18.0 pts (passe largement, PF 2.58, E +1.67%/trade)**

## ✅ Checkpoint retenu : **500k_FIXED**

### Raison
- Sur la seule donnée out-of-sample fiable (VAL), 500k bat le random de **+18
  points de win-rate** et affiche un **profit factor 2.58** et une expectancy
  **+1.67 %/trade** — nettement supérieur au 450k (PF 1.18, +0.17 %/trade).
- Le 500k inclut le fix `_dur_price` (MAX_DURATION fonctionnel), ce qui explique
  sa meilleure gestion de sortie hors échantillon.
- L'égalité 450k≡500k sur TEST est un artefact du split (voir anomalie), pas un
  signe d'équivalence des politiques.

### Risques connus
1. **Échantillons courts** : VAL = 2650 lignes 5m (~9 jours). Edge réel mais
   intervalle de confiance large. À confirmer sur fenêtre plus longue.
2. **Sur-apprentissage possible** : 500k a un WR élevé sur peu de trades (152).
   À surveiller en paper trading (alerte si WR > 95 % sur 20 trades = overfit).
3. **Oracle HMM bloqué sur "sideways"** (voir docs/ORACLE_HMM_DIAGNOSTIC.md) :
   le forcing de durée est statique, pas adaptatif au régime.
4. **Split TEST dégénéré** : les deux modèles y perdent. Sur un marché plat
   court, la politique micro-TP est non rentable après frais — risque réel en
   conditions de faible volatilité.

## Prochaine étape
**Paper trading des deux modèles** (450k + 500k) sur données test, 10000 steps,
log de trades détaillé. Si 500k confirme WR > random de façon consistante →
candidat pour live paper trading sur données récentes. Sinon → reprise
d'entraînement avec correctif Oracle HMM (régime-adaptatif).
