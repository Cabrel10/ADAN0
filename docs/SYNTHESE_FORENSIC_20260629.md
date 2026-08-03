# ADAN0 — Synthèse Forensic & Décision (Carte Blanche)
**Date:** 2026-06-29 · **Branche:** feat/future-arena-v2 · **Run:** FA_500k_prod (terminé 500k)

> Méthode imposée: **mesurer, jamais supposer**. Chaque affirmation ci‑dessous
> est étiquetée **[FAIT]** (prouvé par un script + données) ou **[HYPOTHÈSE]**
> (à confirmer). Les scripts sont dans `scripts/research/` et `scripts/backtest/`.

---

## 0. TL;DR — la décision

**Le "signal particulier" que tu sentais est réel et il est double, mais ce
n'est PAS un bug de la policy. C'est une INADÉQUATION ENVIRONNEMENT/MARCHÉ:**

1. **[FAIT] Le marché de test est 51% zones RED.** Sur 5241 bougies 5m, l'excursion
   favorable médiane (MFE) sur 36 bougies = **0,273%**, contre des frais
   aller‑retour de **0,5%**. La médiane du marché est *sous* les frais.
2. **[FAIT] L'agent entre au hasard par rapport aux zones** (mix 15,9/33,1/53,7
   GREEN/ORANGE/RED ≈ baseline marché 15,9/33,1/51,0). Il n'a **aucun edge
   d'entrée**, mais n'est pas pire que l'aléatoire.
3. **[FAIT] La sortie est passive** (barrière temporelle ~26 bougies), pas
   optimale (pic de MFE). C'est là qu'il perd ~0,29%/trade vs un oracle.

**Décision recommandée (carte blanche):** NE PAS relancer un entraînement à
l'identique. Recalibrer l'environnement AVANT toute relance — voir §7.

---

## 1. (c) Les zones sont‑elles calculables sans lookahead ? — **[FAIT]**
Script: `scripts/research/zone_lookahead_audit.py` → `logs/validation/research/zone_audit.json`

- **Statique:** `compute_mfe_mae` lit `df.iloc[idx+1:end]` → la zone est **ex‑post**
  (utilise des bougies APRÈS l'entrée). Elle ne peut PAS être calculée à l'entrée.
- **Runtime (per‑key diff):** en corrompant les bougies FUTURES, les fenêtres
  d'observation **5m/1h/4h sont byte‑identiques** (`max_abs_diff = 0.0`).
  → **AUCUN lookahead dans le prix que voit la policy.**
- `context_vector` change de 0,69 **uniquement** par la statefulness du filtre
  HMM entre deux appels; ses *entrées* par pas (`_get_current_market_data_for_hmm`)
  lisent `iloc[safe_idx]`/`[safe_idx-1]` (passé/présent) — vérifié dans le code.

**Verdict:** `EX_POST_REWARD_ONLY`. Les zones sont une **étiquette de reward
légitime** (le reward a le droit d'utiliser le hindsight). MAIS: **l'agent ne
peut jamais "voir" la zone à l'entrée** → inutile d'espérer un classifieur
"évite le RED" appris directement. Le shaping n'enseigne qu'**indirectement**.
> ⚠️ Ta mise en garde était juste: un `if zone==RED: reward-=X` reste OK côté
> reward, mais ne crée AUCUNE capacité de décision à l'entrée. À ne pas confondre.

---

## 2. Les zones ont‑elles un pouvoir prédictif ? — **[FAIT]**
Script: `scripts/backtest/forensic_trades.py` (6 checkpoints, 46–68 trades chacun)

| Ckpt | GREEN exp | ORANGE exp | RED exp | n trades |
|------|-----------|-----------|---------|----------|
| 100k | +0,48 | +0,09 | −0,58 | 68 |
| 200k | +0,44 | +0,13 | −0,49 | 57 |
| 300k | +0,73 | +0,13 | −0,45 | 54 |
| 400k | +0,37 | +0,11 | −0,48 | 46 |
| 430k | +0,74 | +0,12 | −0,52 | 54 |
| 450k | +0,13 | +0,05 | −0,50 | 47 |

**GREEN > ORANGE > RED à TOUS les checkpoints.** Le système de zones **n'est
pas cassé**; il porte une vraie information. (Confirme ton point, sur 326 trades
et plus seulement 17.)

---

## 3. Le SL est‑il "trop serré" / tapé par le bruit ? — **[FAIT, RÉFUTÉ]**
Script: `forensic_trades.py` (bloc `sl_tp_atr`) + `scripts/diagnostics/analyze_trades.py`

- `SL%/ATR%` = **10,5 à 16,3** à tous les checkpoints. **0%** des SL sont sous
  0,5×ATR. Le SL (1,2%) vaut ~11× l'ATR 5m (≈0,11%, vérifié vs (H‑L)/C brut).
- Durée moyenne avant SL_HIT = **26,6 bougies** (seulement 3,4% en ≤3 bougies).
- SL_HIT (26,6) ≈ TP_HIT (27,3) → **sortie temporelle, pas décisionnelle.**

**Le bruit ne tape pas le stop.** L'agent reste bloqué ~2h dans une tendance
adverse puis se fait couper. Problème de **timing d'entrée + gestion de sortie**,
pas de largeur de SL.

---

## 4. (b) Distribution winners/losers + capture ratio — **[FAIT]**
Script: `scripts/research/winner_distribution.py`

| ckpt | WR | avgWin | avgLoss | R/R réalisé | capture ratio |
|------|-----|--------|---------|-------------|---------------|
| 200k | 47,4% | +0,28 | −0,56 | 0,50 | 0,92 |
| 300k | 40,7% | +0,36 | −0,52 | 0,69 | 0,99 |
| 430k | 48,1% | +0,41 | −0,63 | 0,64 | 1,03 |
| 450k | 38,3% | +0,31 | −0,51 | 0,60 | 1,29 |

- **Capture ratio ≈ 0,8–1,3** → l'agent encaisse ~100% du MFE *qui se matérialise
  avant sa sortie*. Il ne "coupe pas ses gagnants trop tôt" comme on le croyait.
- **Le vrai problème = R/R réalisé 0,5–0,64**: les pertes (−0,5/−0,6, jusqu'au
  SL −1,2%) dépassent les gains (+0,3/+0,4). Pourquoi des gains si petits ?
  → §5: le MFE disponible est minuscule.

> ⚠️ Ton point sur "TP=2,5% arbitraire" est validé: le bon TP se déduit de la
> distribution de MFE, pas d'un chiffre. Or **0% des trades atteignent 2,0% de
> MFE** → fixer TP=2,0% est déjà irréaliste (voir §5).

---

## 5. CAUSE RACINE — MFE du marché vs frais — **[FAIT]**
Script: `scripts/research/market_mfe_baseline.py` + `fee_horizon_sensitivity.py`

**Baseline marché (5241 bougies test, horizon 36, long):**
- MFE médian = **0,273%** · MFE p90 = 0,785% · MAE médian = 0,350%
- % bougies MFE ≥ 0,6% (break‑even frais) = **17,9%**
- % bougies MFE ≥ 2,0% (cible TP) = **1,0%**
- Zones marché: GREEN 15,9% / ORANGE 33,1% / RED **51,0%**

**Entrées agent (430k):** GREEN 14,8 / ORANGE 31,5 / RED 53,7 — **≈ baseline**.
→ **[FAIT] L'agent n'a aucun edge d'entrée: il entre comme le marché moyen.**

**Borne oracle (entrée GREEN parfaite + sortie au pic MFE) − frais:**
| Horizon | GREEN% | MFE médian GREEN | oracle GREEN @0,5% | oracle TOUTES @0,5% |
|---------|--------|------------------|---------------------|----------------------|
| H36 (3h) | 15,9% | ~1,4% | **+0,498%** | +0,157% |
| H72 (6h) | ~18% | — | +0,592% | — |
| H144 (12h)| ~20% | — | +0,789% | — |
| H288 (24h)| 22,5% | 1,81% | **+1,159%** | +0,355% |

**Conclusions [FAIT]:**
- Le marché **EST** gagnable **SI** on sait viser le GREEN — mais la zone est
  ex‑post (impossible à l'entrée).
- À frais 0,5%, l'horizon scalper 5m/36 est le **régime le plus dur**. Allonger
  l'horizon multiplie le MFE disponible (H288 GREEN médian 1,8%).
- L'écart oracle‑toutes (+0,16%) vs agent réel (−0,13%) ≈ **0,29%/trade** est la
  **perte de timing de SORTIE** (barrière temporelle au lieu du pic MFE).

---

## 6. (a) Matrice de confusion action × état — *(en cours, 10k steps)*
Script: `scripts/research/confusion_matrix.py` → `confusion_{ckpt}.json`
*(Résultats injectés ici dès la fin du run; smoke 5k indiquait illegal_ratio≈0,95/step.)*

Hypothèses à trancher (tes objections, légitimes):
- **Hyp A** (oscillation près du seuil) → mesurée par `action0_pct_near_threshold_band`.
- **Hyp B** (spam illégal moins puni que HOLD+time‑decay) → comparée via breakdown FLAT/OPEN.
- **Hyp "câblage croisé"** → confirmée seulement si `SELL_illegal_pct` (FLAT) et
  `BUY_illegal_pct` (OPEN) dominent réellement sur 10k steps.

---

## 7. Décision & plan (carte blanche)

**Ne pas relancer à l'identique.** Trois leviers, par ordre d'impact prouvé:

1. **[Priorité 1 — frais]** Les frais d'entraînement 0,5% A/R dépassent le MFE
   médian (0,27%). Ils ont été mis ×2 *intentionnellement* pour durcir, mais ils
   rendent l'horizon 5m mathématiquement perdant. **Tester frais réalistes
   (0,1% Binance VIP / 0,2%)** OU **allonger l'horizon** (H72–H144) pour que le
   MFE disponible dépasse les frais.
2. **[Priorité 2 — sortie]** L'agent perd ~0,29%/trade en sortie passive.
   Introduire un **trailing/MFE‑aware exit** (sortir quand MFE se retourne) plutôt
   qu'attendre une barrière à 26 bougies.
3. **[Priorité 3 — actions illégales]** ~0,95 action illégale/step pollue le
   gradient. À confirmer par la matrice (§6) avant de durcir le masque d'action
   (action‑masking strict FLAT→{HOLD,BUY}, OPEN→{HOLD,SELL}).

**Checkpoint de reprise:** aucun n'a d'edge (PF<1 partout). Si reprise, partir
de **430k** (meilleur PF 0,53, meilleur R/R réalisé 0,64) — mais **seulement
après recalibration de l'environnement** (sinon même plafond).

**Ce que je NE ferai pas** (pièges écartés par les faits):
- ❌ Pénaliser RED directement à l'entrée (zone ex‑post, créerait une attente
  irréaliste / pas de lookahead exploitable).
- ❌ Fixer TP=2,5% (0% des trades atteignent même 2,0% de MFE).
- ❌ Resserrer le SL (déjà 11× ATR; le bruit ne le tape pas).
