# Audit de divergence Training ↔ Paper ↔ Live

Date : 2026-06-26. Périmètre : SPOT BTC/USDT, leverage=1.
Source de vérité = `multi_asset_chunked_env.py` (env d'entraînement, ce que le PPO a réellement appris).
Cibles auditées = `execution_engine.py` (décodage live/paper), `live_state_builder.py` (observation).

---

## Tableau de synchronisation

| Élément | TRAINING (env, vérité) | LIVE (execution_engine) | Verdict |
|---|---|---|---|
| **Mapping action** | Box(5): [dir, size, tf, sl_raw, tp_raw] | idem (mêmes indices) | ✅ identique |
| **Décodage SL/TP** | `(raw+1)/2` → clip(lo,hi) | `(raw+1)/2` → clip(lo,hi) | ✅ formule identique |
| **fee gate tp_lo** | `max(tp_lo, 0.006)` | `max(tp_lo, 0.006)` | ✅ identique |
| **R/R floor** | `tp ≥ sl×1.5` | `tp ≥ sl×1.5` | ✅ identique |
| **_BOUNDS scalper** | sl(0.003,0.012) tp(0.005,0.020) | **sl(0.020,0.030) tp(0.040,0.060)** | ❌ **DIVERGENT** |
| **_BOUNDS intraday** | sl(0.005,0.020) tp(0.008,0.040) | **sl(0.040,0.060) tp(0.080,0.120)** | ❌ **DIVERGENT** |
| **_BOUNDS swing** | sl(0.010,0.035) tp(0.015,0.070) | **sl(0.070,0.100) tp(0.140,0.200)** | ❌ **DIVERGENT** |
| **_BOUNDS position** | sl(0.020,0.060) tp(0.030,0.120) | **sl(0.150,0.200) tp(0.300,0.400)** | ❌ **DIVERGENT** |
| **ATR scalper floor** | `max(0.006, 3×ATR)` (env:7535) | absent du chemin model | ⚠️ manquant |
| **Commission** | 0.25%/côté (0.50% A/R) | 0.10%/côté (0.20% A/R) | ⚠️ réel≠train (voir note) |
| **Slippage** | 2 bps (0.02%) | `SLIPPAGE_BPS=2.0` (ligne 39/636) | ✅ identique (déjà OK) |
| **Sizing** | tier exposure × confidence (HMM) | tier exposure × confidence (HMM) | ✅ même logique |
| **Cooldown** | post-SELL / post-BUY (config) | `_last_trade_time` rate-limit | ⚠️ logiques différentes |
| **Timeout / max_hold** | force_after par TF (freq gate) | non répliqué | ⚠️ manquant |
| **Funding** | N/A (SPOT lev=1) | N/A | ✅ non applicable |
| **Liquidation** | futures only (non utilisé en SPOT) | kill-switch capital floor | ✅ cohérent (SPOT) |
| **stochastic_sltp override** | n'existe PAS dans l'env | `_compute_stochastic_sltp` (ATR×regime) | ⚠️ logique live-only |

---

## Divergences BLOQUANTES (à corriger avant tout paper)

### D1 — `_PROFILE_BOUNDS` (execution_engine) = ANCIENNES bornes 8-40%
C'est **l'intrigue confirmée**. Le commentaire dit *"MUST stay IDENTICAL to the training env"*
mais les valeurs sont restées aux bornes pré-FINDING#4. Conséquence : un `tp_raw=0`
(milieu de bande) produit en training un TP scalper ≈ 1.25%, mais en live ≈ 5%.
**Le modèle exécuterait des TP 4x trop larges → comportement totalement divergent.**
→ **Correction obligatoire** : recopier exactement la table `_BOUNDS` de l'env.

### D2 — `tp_lo = max(tp_lo, 0.006)` appliqué sur des bornes fausses
Tant que D1 n'est pas corrigé, le fee gate ne mord jamais (bornes déjà ≥4%).
Se résout automatiquement avec D1.

---

## Divergences NON-BLOQUANTES (à documenter / décider)

### N1 — Commission 0.10% (live) vs 0.25% (train)
Le **réel Binance SPOT taker = 0.10%** : la valeur live est correcte pour le réel.
Le modèle a été entraîné PLUS pénalisé (0.25%) → en réel il est *avantagé* (frais plus
bas que ceux qu'il a appris à craindre). Biais **favorable**, donc acceptable, mais à
mentionner dans le rapport de paper. NE PAS aligner le live sur 0.25% (ce serait fausser le réel).

### N2 — ATR scalper floor manquant en live
L'env relève le SL scalper à `max(0.006, 3×ATR)`. Le live ne le fait que via
`stochastic_sltp`. → Ajouter le même floor dans le chemin model du live pour cohérence.

### N3 — Slippage ✅ DÉJÀ CONFORME
Vérification : `execution_engine.py:39` `SLIPPAGE_BPS = 2.0` appliqué au fill BUY
(ligne 636) et SELL (735) — identique au training (2 bps). Aucune action requise.

### N4 — stochastic_sltp est une logique live-only
Elle remplace la décision SL/TP du modèle par un calculateur ATR×regime. Ce n'est PAS
ce que le modèle a appris. Pour un paper qui valide LE MODÈLE, il faut
`stochastic_sltp=False` (utiliser la décision réelle du modèle). À garder désactivé.

### N5 — Cooldown/timeout non strictement répliqués
Mineur pour un premier paper (le modèle décide quand sortir via SL/TP). À aligner avant Live.

---

## Plan de correction (ordre)
1. **D1** : aligner `_PROFILE_BOUNDS` du live sur `_BOUNDS` de l'env (CRITIQUE). ✅ FAIT
2. **N2** : ajouter ATR scalper floor au chemin model live. ✅ FAIT
3. **N3** : slippage déjà conforme (2 bps). ✅ rien à faire
4. Garder `stochastic_sltp=False` pour le paper de validation du modèle (N4). → à fixer au lancement
5. Documenter N1 (frais) dans le rapport paper. → fait dans ce doc
6. N5 (cooldown/timeout) : aligner avant le Live (pas bloquant pour paper). → reste à faire avant Live

## État post-correction (2026-06-26)
- ✅ D1 corrigé : `_PROFILE_BOUNDS` = bandes serrées de l'env (scalper tp 0.5-2.0% … position tp 3-12%).
- ✅ N2 corrigé : ATR scalper floor `max(0.006, 3×ATR)` ajouté au chemin model live.
- ✅ N3 : slippage 2 bps déjà présent.
- ⏳ N5 (cooldown/timeout) : à répliquer avant passage Live.
- Chaîne de décodage action désormais STRICTEMENT identique training↔live pour SL/TP.
