# FORENSIC — Policy Collapse du run v8 500k (obs_schema_v2 / 28d)

**Date:** 2026-07-01
**Run:** `train_v8_500k.log` / `diagnostic_collapse_v8_500k.csv`
**Verdict:** COLLAPSE TOTAL confirmé. Point d'inflexion **124k-128k timesteps**.

## 1. Preuve (extraite du CSV réel)

| timesteps | a0_mean | a0_std | pct_buy | pct_sell | histo (bin 10 = extrême BUY) | phase |
|-----------|---------|--------|---------|----------|------------------------------|-------|
| 2k–122k   | -0.06..+0.10 | 0.66–0.94 | 45–55% | 45–54% | étalé (multi-modal) | ✅ SAIN |
| 124k      | 0.143   | 0.92   | 53%     | 46%      | 298\|...\|448 | ⚠️ début dérive |
| 128k      | 0.342   | 1.21   | 61%     | 38%      | 329\|...\|677 | 🔴 BASCULE |
| 134k      | 1.19    | 2.27   | 70%     | 30%      | 390\|...\|1126 | 🔴 emballement |
| 148k      | 22.98   | 12.2   | 96.6%   | 3.4%     | 60\|...\|1910 | 🔴 quasi-mort |
| 170k      | 99.7    | 33.7   | 100%    | 0%       | 0\|...\|1996 | 💀 collapse |
| 218k      | 476.1   | 95.0   | 100%    | 0%       | 0\|0\|...\|2000 | 💀 mort total |

Signes textbook:
- `a0_mean` diverge vers +∞ (476 !) = logits/sortie continue non bornés.
- histogramme `0|0|0|0|0|0|0|0|0|2000` = politique 100% déterministe.
- `steps_open_pct ≈ 0.88` constant = agent toujours en position (max_concurrent=1).
- `illegal_ratio ≈ 0.978` = 97.8% des intents rejetés.
- `policy_entropy` remonte de -0.58 vers -0.51 (la distribution se resserre autour d'un point).

## 2. Cause racine (prouvée dans le code)

`multi_asset_chunked_env.py` :

**L.7602-7611** — quand l'agent demande BUY alors qu'une position est ouverte
et que l'exposition est déjà proche de la cible :
```python
if is_open and discrete_action == 1:
    ...
    if exposure_diff < 0.10:
        discrete_action = 0  # Override to HOLD   <-- SILENCIEUX
```
**L.7619-7629** — ZERO pénalité sur cet override (commentaire explicite :
« NO escalating penalty. Just telemetry »). Le pari : la Capability Vector (ACM)
enseignera la légalité. **Ce pari a échoué (CSV le prouve).**

### La chaîne d'auto-renforcement (diagnostic utilisateur, confirmé)
```
Agent en position (max_concurrent=1)
   -> demande BUY (illégal, can_open=0)
   -> capability force HOLD (L.7611)   [ZERO pénalité, L.7619]
   -> position tenue, marché monte -> reward positif
   -> PPO stocke l'action ÉCHANTILLONNÉE (BUY brut) dans le rollout buffer
   -> l'avantage du reward marché est crédité à BUY
   -> P(BUY) augmente -> moins d'exploration SELL/HOLD
   -> boucle auto-renforcée -> 100% BUY
```

Pourquoi les garde-fous existants n'ont pas suffi :
- `target_kl=0.03` : ne se déclenche pas — le collapse est porté par un gradient
  de reward LÉGITIME, pas par une grosse mise à jour. Chaque update reste petit.
- Sterile penalties V4/V5/V6 : ne couvrent PAS le cas BUY-while-open (mis à 0
  par design). Le `min_notional` Cas B ne pénalise que 0.002 — négligeable.
- `use_sde=True` (gSDE) : le bruit d'exploration scale avec la moyenne ; quand
  `a0_mean` grandit, l'exploration relative s'effondre -> accélère le collapse.

## 3. Plan de correction (3 leviers, aucun ne touche les frais 0.5%)

1. **Symétrie du signal (cause racine)** : appliquer une pénalité réelle et
   bornée sur l'override BUY-while-open (L.7611), au lieu de zéro. L'ACM reste
   dans l'observation (enseigne la légalité) MAIS le reward ne doit plus créditer
   l'action illégale. Objectif : neutraliser l'avantage marché mal attribué.
2. **Garde anti-divergence (DSpark-like)** : pénalité de sur-confiance sur la
   sortie continue — borne l'explosion de `a0_mean`. `loss += λ·relu(|mean|-cap)`.
3. **Disjoncteur de collapse (callback)** : auto-stop si `pct_buy>0.97` OU
   `a0_mean>5` OU `entropy` remonte anormalement — évite de gaspiller 90k steps
   après la mort.

## 4. Point de reprise
Checkpoints SAINS conservés : 25k, 50k, 75k, 100k, 100k_SWEETSPOT, 125k.
Reprise possible ≤ 125k (jamais 150k+ = post-collapse).
Recommandé : relance FROM SCRATCH avec les 3 correctifs (pour prouver que la
correction empêche le collapse dès le départ, pas juste le retarde).
