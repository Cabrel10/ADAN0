# HANDOFF — Investigation complète du collapse ADAN0 (V8→V12)

> Document autosuffisant. Une personne/IA reprenant sur un nouvel environnement doit
> pouvoir comprendre l'état complet SANS accès à l'historique de conversation.
> Dernière mise à jour : 2026-07-04 (post-120h, VPS en fin de vie).

---

## 0. TL;DR (à lire en premier)

- **Le collapse "always-BUY" n'est PAS causé par l'espace d'action ni par un bug de
  signe de reward.** Il est causé par une **asymétrie structurelle du reward** :
  **l'état FLAT est puni en moyenne (`final_reward` ≈ -0.161), l'état LONG est neutre
  (≈ -0.007)**. Le moteur principal de cette punition du FLAT est **`future_contrib`
  (module Future Arena / Oracle) ≈ -0.112** : rester flat pendant que les candles
  futures montent est pénalisé → l'agent apprend à toujours être en position → toujours
  BUY.
- **V12 (routage d'action conditionnel par état) a RE-COLLAPSÉ**, plus tôt que V10/V11
  (~40k steps vs 70-78k). Preuve : `a0_mean` -0.005→+0.302 (dérive monotone),
  `pct_buy` 45%→98%, `pct_sell` 49%→1.6%. `illegal_ratio` resté bas (8-11%) → les
  actions ne sont PAS illégales, l'agent choisit délibérément BUY. **Ceci invalide
  l'hypothèse de la refonte d'espace d'action** et pointe définitivement vers le reward.
- **Prochain travail = fix reward** (rééquilibrer `future_contrib`/pénalité du FLAT),
  testé par ablation causale courte AVANT tout run long.

---

## 1. Chronologie des runs

| Run | Profil | Step collapse / fin | Nature | Fix tenté | Résultat |
|-----|--------|--------------------|--------|-----------|----------|
| V8 / SWEETSPOT | scalper | ckpt @100k (étiquette "sweetspot" NON vérifiée) | — | — | Baseline douteuse (VecNorm identity, cf §5) |
| V10 | scalper | ~70k | BUY runaway (pct_buy→0.97) | signe pénalité V9 | Collapse |
| V11 | scalper | ~78k | BUY runaway identique | telemetry reward + warmup C1 | Collapse identique |
| **V12** | **intraday** | **~40k** | **BUY runaway, PLUS RAPIDE** | **routage conditionnel par état** | **RE-COLLAPSE (cf §3)** |

---

## 2. L'erreur méthodologique "97.67%" (NE PAS RÉPÉTER — 3e fois)

- **Affirmation fausse** : *"le reward est corrélé au portfolio value à 97.67%"*.
- **Origine** : un tableau "quelle composante domine" sommait `abs()` de TOUTES les
  colonnes du CSV, dont `portfolio` (échelle ~13-18) mélangée aux vraies composantes de
  reward (échelle ~0.001-0.05). `portfolio` "dominait" à 97.67% par pure magnitude
  brute — **artefact d'échelle, PAS une corrélation.**
- **Vraie corrélation recalculée** (scipy, 2026-07-04) :
  `final_reward` vs `portfolio` → **Pearson r=+0.010 (p=0.93), Spearman r=+0.134
  (p=0.26)** — statistiquement **non significatif**. Il n'y a PAS de corrélation.
- **Limite de l'échantillon** : `reward_components_v12.csv` = 74 lignes mais seulement
  **7 steps uniques** (doublons multi-worker, écriture concurrente non triée) → aucune
  conclusion statistique forte n'est possible sur ce fichier. Toute lecture est
  QUALITATIVE.

---

## 3. Preuve du re-collapse V12 (analyse de tendance, pas un coup d'œil)

Source : `logs/training/diag_v12_500k.csv` (20 points, 2k→40k steps).

| Métrique | @2k | @20k | @40k | Pente/step | Verdict |
|----------|-----|------|------|-----------|---------|
| a0_mean | -0.005 | +0.137 | **+0.302** | +7.8e-6 | Dérive monotone +0.306 |
| pct_buy | 45.4% | 81.6% | **97.9%** | +1.3e-5 | Runaway BUY |
| pct_sell | 48.9% | 14.5% | **1.6%** | -1.2e-5 | Collapse SELL |
| steps_open | 64% | 83% | **86%** | +4.4e-6 | Position quasi permanente |
| illegal_ratio | 8.6% | 6.0% | 10.9% | ~0 | **Bas = actions PAS illégales** |
| a0_std | 0.134 | 0.132 | 0.141 | ~0 | Stable (gSDE OK) |

Segments (tiers) : a0_mean 0.040 → 0.134 → 0.252 (aggravation régulière).

**Conclusion** : collapse de MÊME nature que V10/V11 (BUY runaway) mais **plus rapide**.
Le routage conditionnel a supprimé le bruit de gradient (illegal_ratio bas) ce qui a
**accéléré** la convergence vers le collapse au lieu de l'empêcher. → Le moteur est
ailleurs : le reward.

---

## 4. Cause racine (preuve causale qualitative)

`reward_components_v12.csv`, moyennes par état :

| État | final_reward moyen | future_contrib | closure_bonus | pnl_base |
|------|-------------------|----------------|---------------|----------|
| **flat** (n=22) | **-0.161** | **-0.112** | -0.064 | -0.036 |
| **long** (n=52) | **-0.007** | ~0 | ~0 | ~0 |

- **Être FLAT est puni ; être LONG est neutre.** L'agent minimise la punition en
  restant en position → BUY systématique. Le routage v12 n'y change rien car le biais
  est dans le signal de reward, pas dans le décodage.
- **`future_contrib` (Future Arena / Oracle) = principal coupable** : il pénalise le
  fait d'être flat quand les candles futures montent ("tu aurais dû être en position").
  Exemples : HOLD en flat @step500 → future_contrib -0.40, final_reward -0.57.
- Répartition des composantes réelles (abs-sum, cols reward seulement) :
  `future_contrib` 47%, `closure_bonus` 27%, `pnl_base` 25%, `symmetry_penalty` 1%,
  `latent_pnl` 0.1%.

---

## 5. Fixes déjà en place (tous corrects, aucun n'a résolu le collapse)

- **V9** : correction du signe de pénalité. Correct, insuffisant.
- **C1** : warmup par-raison de `min_notional_self_caused`. Correct, insuffisant.
- **v12 (commit 4ef2945)** : routage conditionnel d'action `route_action_by_state`
  (FLAT→BUY/HOLD, LONG→SELL/HOLD, NOOP si quota tier dépassé), source unique
  `src/adan_trading_bot/environment/action_routing.py`, câblé env + paper + live.
  Neutralisation des pénalités sell_no_position/anti_spam_hold/CASH_FLOOR_B. Correct
  architecturalement (illegal_ratio effondré 67-78%→8-15%) mais **n'a pas empêché le
  collapse** car le biais reward demeure.
- **fix(live) commit fee26f2** : `PositionSizingMethod` ré-exporté depuis
  `position_sizer.py` → import runtime de `action_translator` OK → live trading
  débloqué.

**→ Le prochain fix DOIT porter sur le reward** (§4), pas sur l'espace d'action.

---

## 6. Bugs ops connus et corrigés

- **disk_guard** pattern-matching : `pgrep -f` incluait le shell parent → faux positif.
  Corrigé (`grep -vx` sur `$$`/`$PPID`). v12 : `guard_target_alive` = training OU paper.
- **monitor premature-exit** : `grep -icE` renvoie exit 1 si count=0, cassait `&&`.
- **paper trading `load_model()` clobber** : `run()` réécrasait le modèle chargé via
  `--model`. Corrigé (commit 72b74cd).
- **import `PositionSizingMethod`** : enum absent de position_sizer.py → ImportError
  runtime. Corrigé (commit fee26f2).

---

## 7. État disque et sa cause

- Sur ce VPS, **Docker ne prend que 4K** (`/var/lib/docker` quasi vide, pas la cause).
- Vrais consommateurs : fichiers de benchmark fio dans `/home/ubuntu`
  (`randrw.*` 4×1G + `testfile` 1G = 5G, **PURGÉS** le 2026-07-04), `logs/rewards/`
  (jsonl write-only, jamais relus — purge sûre), `logs/adan_trading_bot.json.*` (rotés).
- `sudo` sans mot de passe **indisponible** sur ce VPS (confirmé lors de l'audit).
- Après purge : 13G libre (92%).

---

## 8. Environnements & invariants

- **Python** : `/home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python`
  (3.11, SB3 2.8.0).
- **Repo** : `/home/ubuntu/webapp/MORNINGSTAR/ADAN0`, branche `feat/diagnostic-v4`,
  remote `github.com/Cabrel10/ADAN0`, PR #9 (→ feat/future-arena-v2).
- **INVARIANTS BINDING** : frais 0.5% (`commission 0.0025`, `round_trip_fees 0.005`)
  INTACTS. Dims 1-4 de l'action (Size/TF/SL/TP, module Future Arena) INTACTES.
  VecNormalize DÉSACTIVÉ volontairement (OOM). Pas de sb3-contrib/MaskablePPO
  (action_space continu Box, algo WorldModelPPO custom).
- **Éditer `multi_asset_chunked_env.py`** (9300+ lignes) : via heredoc python
  `src.replace(old,new,1)` + `assert count==1`, PAS l'outil Edit.
- **Lancement background** : `setsid/nohup ... &` dans une commande avec sleep/echo →
  timeout 120s (artefact shell), le process se lance quand même — vérifier avec un
  `ps aux` séparé.

---

## 9. Prochaines étapes recommandées (par priorité, VPS en fin de vie)

1. **Sécuriser** (fait) : git poussé, checkpoints inventoriés, ce handoff.
2. **Ablation causale reward** (runs courts 5-8k, 1 variable à la fois) :
   - `future_contrib=0` en état flat → la dérive `a0_mean` disparaît-elle ? (test direct
     de §4).
   - holding cost léger vs closure bonus (cf docs/ABLATION_RESULTS_v13.md).
   - critère GO : `pct_buy` ET `pct_sell` restent dans [35%,65%] (éviter le collapse
     inverse "flat forever").
3. Calcul de magnitude AVANT implémentation finale (discipline C1).
4. Ne PAS viser un 500k (ne finira pas avant expiration). Run intermédiaire 100-150k max.

---

## 10. Fichiers clés

- `src/adan_trading_bot/environment/action_routing.py` — routeur partagé (source unique).
- `src/adan_trading_bot/environment/multi_asset_chunked_env.py` — env (routage @_execute_trades,
  reward @ final_reward ; `future_contrib` calculé ici).
- `src/adan_trading_bot/trading/action_translator.py` + `position_sizer.py` — pipeline live.
- `docs/ARCHITECTURE_ACTION_ROUTING_v12.md` — spec du routage.
- `logs/training/diag_v12_500k.csv` — diag du collapse V12 (preuve §3).
- `logs/training/reward_components_v12.csv` — composantes reward (preuve §4, échantillon faible).
- `checkpoints/*.zip` — voir `docs/CHECKPOINTS_INVENTORY.md`.
