# RAPPORT COLLAPSE V31-500k — 2026-08-18

## Verdict

| Item | Valeur |
|---|---|
| Run | `v31_500k_20260817_2141` (driver 400696, workers 401115/401116) |
| Durée | 8.7 h (23:42 → 08:26 UTC) |
| Steps PBT rapportés | 40k (w0) / 30k (w1) — ~185 tables PPO cumulées |
| Statut | **COLLAPSE CONFIRMÉ — STOP exécuté** (kill 06:26 UTC) |

## Signature mesurée (metrics.json, extraction complète)

| Métrique | w0 (401115) | w1 (401116) | Seuil DIAG (collapse) |
|---|---|---|---|
| a0_mean_raw (μ pré-tanh) | −0.018 → **−8.58** | −0.057 → **−14.8** | −0.22 / −0.36 |
| a0_std_raw (σ effectif batch) | 0.774 → **108.0** | 0.841 → **85.3** | std→0 (DIAG) |
| approx_kl max | **433.4** | **58.3** | 0.58 / 0.96 |
| entropy_loss final | −20.3 | −20.6 | −9.25 / −10.1 |
| SatGuard bumps | 14 total, bump #6 au plafond | idem | 0 en DIAG |
| Saturation finale | **96-98% TOUTES têtes** (direction/size/tf/sl/tp) | idem | 62-72% direction |
| PV | random walk 14.9–20.3, aucun apprentissage | idem | pv 20.5→14.77 |

## Mécanisme racine (CONFIRMÉ par le code)

**Ce n'est PAS le collapse DIAG (std→0, tanh-lock par effondrement). C'est l'inverse : explosion de la Gaussienne pré-tanh.**

1. `a0_mean_raw`/`a0_std_raw` = moyenne et std de la distribution pré-tanh μ(s), σ(s)
   (feature_extractors.py l.1307-1322 : `mu, std = self._distribution_mu_std(...)`).
2. **μ diverge librement** : aucune force de rappel. Le ratio d'importance π/π_old ≈ 1
   quand deux Gaussiennes sont toutes deux « loin » (docstring l.1150-1158) → le clipping
   PPO ne mord jamais → μ fuit (mean −14.8).
3. **σ explose avec gSDE** : le clamp `PpoStdSafetyCallback`/`SatGuard` borne le
   *paramètre* `log_std` à [−5, +2] (ppo_safety.py l.11-31), mais avec
   `use_sde=1` (gSDE state-dependent), le σ effectif = exp(log_std_param) ×
   modulation(features) — **non borné** par le clamp. Mesuré : σ effectif 85-155.
4. **SatGuard a aggravé le collapse** : conçu contre l'effondrement DIAG (relève le
   plancher log_std +0.5/bump), il a poussé 14 bumps jusqu'au plafond +2.0,
   *alimentant* l'explosion. Outil adapté au mauvais régime de collapse.
5. **ent_coef floor 0.02 constant** : le bonus d'entropie récompense σ élevé
   (−ent_coef × H, H croît avec log σ). Sans clamp effectif sur σ gSDE, contribution
   au régime d'explosion. Non démontré comme cause première, mais aggravant probable.

## Corrections DÉMONTRÉES (appliquées au relancement)

| # | Correction | Preuve |
|---|---|---|
| 1 | `ADAN_L2_ANCHOR_LAMBDA=0.05` — ancre L2 `λ·(μ²).mean()` directement dans la loss acteur, au-dessus de la chaîne avantage/GAE | Validée gate **V16-final** (docs/V16_FINAL_GATE_REPORT.md l.157, tag v16-final, checkpoints 300k/310k/320k retenus). Mécanisme V15 : la seule force qui s'applique sur μ quand le ratio PPO ≈ 1 |
| 2 | `ADAN_USE_SDE=0` — DiagGaussian au lieu de gSDE | Le code lui-même : train_parallel_agents.py l.1343 « ADAN_USE_SDE=0 selects the stable DiagGaussian path recommended for the production 500k run ». Avec DiagGaussian, σ = exp(log_std paramètre) ∈ [exp(−5), exp(2)] — le clamp redevient **effectif** |
| 3 | SatGuard conservé (redevient utile : avec DiagGaussian le clamp ±2.0 borne réellement σ) + ent_coef floor 0.02 conservé (PBT explore [0.02, 0.1]) | inchangé |

## Artefacts

- `full_log_copy.txt` (32.9 MB) — log complet du run tué
- `metrics.json` — séries PPO par worker (89/91 updates)
- `satguard_first_events.txt` / `satguard_last_events.txt` — activité SatGuard
- Checkpoints PBT : 40k (w0) / 30k (w1) conservés dans training_output/v31_500k_20260817_2141

## Classification

- **CONFIRMÉ** : collapse par explosion μ+σ pré-tanh (pas le régime DIAG)
- **CONFIRMÉ** : ancre L2 inactive par défaut (env var non exportée au lancement V31)
- **CONFIRMÉ** : clamp log_std inefficace sous gSDE
- **PROBABLE** : ent_coef constant 0.02+ aggravant (non isolé expérimentalement)
- **RÉFUTÉ** : l'hypothèse « SatGuard empêche tout collapse » (il en a empiré un)
