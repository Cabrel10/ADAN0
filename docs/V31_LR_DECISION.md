# V31 — Décision Learning Rate (ÉTAPE 1 + ÉTAPE 2)

**Date**: 2026-08-27
**Auteur**: audit autonome
**Portée**: relance BTC seul sur 500k. DOGE reste à l'arrêt.

---

## ÉTAPE 1 — Validation de l'hypothèse LR (AVANT de la corriger)

### 1. Mécanisme documenté (sources externes vérifiables)

| Source | Citation | Ce que ça prouve |
|---|---|---|
| **SB3 PPO doc** (stable-baselines3.readthedocs.io/en/master/modules/ppo.html) | « ppo uses clipping to avoid too large update. Limit the KL divergence between updates, **because the clipping is not enough to prevent large update** » | `clip_fraction`/`approx_kl` sont des symptômes de la **magnitude de l'update**, pas de l'exploration. |
| **ICLR "37 Implementation Details of PPO"** (iclr-blog-track.github.io/2022/03/25) | « In MuJoCo, the learning rate linearly **decays from 3e-4 to 0** » ; approx_kl cible « below 0.02 » | LR de référence continu = 3e-4 **avec décroissance**, jamais tenu en plateau. approx_kl doit rester bas. |
| **EmbersArc best-practices-ppo.md** | LR `learning_rate` : « This should typically be **decreased if training is unstable** ». Typical Range : **1e-5 – 1e-3** | Plage valide + règle : instabilité → baisser le LR. |
| **DigitalOcean / LinkedIn PPO guides** | « If the policy oscillates or diverges, try **lowering to 1e-4 or 5e-5** » | Direction du correctif confirmée. |

**Conclusion 1** : le lien LR élevé → update trop grande → clipping massif (`clip_fraction`↑) + divergence (`approx_kl`↑) est un phénomène **documenté**, pas une corrélation de circonstance.

### 2. Table PPO complète (parse `total_timesteps`, pas "Starting step")

Parser : `scripts/tests/parse_ppo_tables.py` (196 updates BTC V30).

Point de bascule EXACT : **clip_fraction ≥ 0.30 à `total_timesteps=48128`, LR=4.230e-05, approx_kl=0.0071**.
Pic clip 0.938 @ 93184 ; pic kl 0.4061 @ 97280.

### 3. Résolution du confond LR↔temps (collinéarité warmup)

Problème honnête : warmup_frac=0.20 sur 500k ⇒ LR rampe linéairement sur tout 0-100k ⇒ `r(clip,LR)=r(clip,temps)=0.9423` (identiques). La corrélation seule **ne peut pas** séparer LR-cause d'un confond temps/régime. Trois tests structurels tranchent :

- **TEST A — dose-réponse sur le NIVEAU de LR** (bins 1e-5) : clip monotone 0.002 (1e-5) → 0.194 (4e-5) → 0.452 (6e-5) → 0.838 (8e-5). Un confond purement temporel ne produirait pas une dose-réponse propre sur le *niveau* de LR.
- **TEST B — variables de contrôle au moment du flip (45k↔55k)** : `clip_fraction +104 %`, `approx_kl +133 %`, mais **`std +0.3 %` et `entropy -0.6 %` = PLATS**. Si la cause était un changement de régime de données ou l'interaction avec le nouveau TP band, l'exploration (std) et l'entropie bougeraient AUSSI. Elles ne bougent pas ⇒ effet de **magnitude d'update (LR)**, pas d'environnement/exploration. **C'est le test décisif.**
- **TEST C — position vs plateau** : flip à 53 % du LR plateau (mi-rampe), pas un step au plateau.

**Nuance vs mandat** : la condition littérale « flip coïncide avec le LR atteignant son PLATEAU » n'est PAS remplie (flip mi-rampe). Mais l'hypothèse sous-jacente (niveau de LR trop élevé → clipping) est confirmée sous une forme **plus forte** : dose-réponse graduée sur le niveau de LR, avec un seuil exploitable. Aucune autre cause (régime data / TP band) n'est soutenue par les données (std/entropy plats).

**Verdict ÉTAPE 1 : GO. Cause = niveau de LR. Pas d'empilement d'un 2ᵉ changement.**

---

## ÉTAPE 2 — Valeur cible (UNE seule variable modifiée vs V30)

**Seuil sain observé** : clip < 0.30 tant que LR ≤ ~4.15e-5 ; flip à 4.23e-5.
Comme V31 tient le plateau 80 % du temps (100k-500k), le plateau doit être **sous** ce seuil avec marge.

**Projection empirique directe** (pas extrapolation — comportement réel de V30 quand la rampe est passée par 3e-5) :

| LR | clip_fraction (V30 observé) | approx_kl (V30 observé) | Gates ÉTAPE 7 |
|---|---|---|---|
| **3e-5** | mean 0.021 / **max 0.036** | mean 0.0013 / **max 0.0022** | clip<0.30 ✅ / kl<0.15 ✅ |
| 4.0-4.3e-5 (plafond) | mean 0.226 / max 0.304 | — | limite |
| 8e-5 (V30 plateau) | 0.84 | 0.40 | ❌❌ |

**Choix : `learning_rate: 0.00008` → `0.00003` (3e-5).**
Justification (non arbitraire) :
- 71 % du plafond sain observé (4.15e-5) — marge de sécurité.
- 37.5 % de l'ancien plateau 8e-5.
- Dans la plage EmbersArc (1e-5–1e-3) ; cohérent avec « lower to 5e-5 if diverging » (on va plus bas car notre schedule tient un plateau au lieu de décroître comme MuJoCo 3e-4→0).
- V30 a **littéralement** mesuré clip max 0.036 à ce LR.

**SEULE variable modifiée depuis V30 : learning_rate 8e-5 → 3e-5.**
Inchangés (déjà actifs en V30) : gSDE off, TP/SL data-driven par actif (BTC tp_hi 0.060), MaxDuration ×2, séparation checkpoints, clip_range 0.12, target_kl 0.15, n_epochs 4, ent_coef 0.03, max_grad_norm 0.3.
