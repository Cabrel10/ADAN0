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

---

## 11. CORRECTIONS V13 (2026-07-04) — révision méthodologique (analyse user)

> Cette section **corrige** les conclusions des §3–§9 ci-dessus. En cas de conflit,
> **la §11 fait foi.**

### 11.1 — "flat est puni" : confusion état/étape (CORRECTION MAJEURE)
La lecture "future_contrib = −0.112 punit l'état flat" est **fausse**. Dans
`reward_components_v12.csv`, `position_state="flat"` est enregistré **au moment de la
fermeture** d'un trade (le SELL vient de s'exécuter, `is_open→False`). Donc ce −0.112
punit la **qualité de la sortie** (TP/SL sous-optimal vs candles futures), **pas** le
fait d'être flat. Le moteur n'est pas "fuir l'état flat" mais **"éviter de fermer une
position mal placée"**.

### 11.2 — future_contrib : disculpé AVEC RÉSERVE (conclusion précédente prématurée)
L'ablation `future_contrib=0` n'a de points qu'à @2k/@4k (10k max) — **phase
d'initialisation** où baseline et ablation sont encore équilibrés. La dérive V12
n'apparaît qu'après @10k. **On ne peut donc PAS conclure** que future_contrib n'est
pas le moteur sur cette base. Il faut @8k–@10k minimum, idéalement le run 500k complet.
future_contrib est **partiellement disculpé, non blanchi**.

### 11.3 — 7 méthodes math = théâtre statistique à cette échelle
PCA/LDA/SVD/moyennes h·g/t-test sur 5–6 points ne prouvent rien de plus qu'un coup
d'œil sur la trajectoire (PC1 explique 70–90% par artefact géométrique ; rang SVD≈1
automatique ; t-test n=5 sous-puissant → "non significatif" = "on ne peut conclure",
PAS "pentes identiques"). **Règle : <30 points → comparaison de trajectoires + régression
linéaire simple avec IC 95% honnête. PAS de PCA/LDA/SVD.** `scripts/analysis/collapse_math_analysis.py`
est conservé pour l'historique mais NE DOIT PAS être réutilisé à cette échelle.

### 11.4 — holding_cost h=0.001 : mal calibré, CONFIRMÉ PAR MESURE
Run complet récupéré (5 points @2k–10k) : pct_buy 0.485→0.623, drift **NON cassé**.
Cause : h=0.001 comparé à `closure_bonus` (0.5, événement rare) au lieu des composantes
per-step. **Mesure (docs/CALIBRATION_AUDIT.md §2, n=52 steps long)** : symmetry_penalty
|mean|=0.00404, somme per-step 0.00426 (std 0.00256). **h=0.001 est ~4× sous le bruit
d'une seule pénalité active** → noyé. h à re-dériver ∈ [0.004, 0.012], test en bracket.

### 11.5 — Le VRAI moteur : asymétrie de VARIANCE (nouvelle hypothèse principale)
Les données prouvent que l'asymétrie réelle est entre **HOLD-flat (reward=0, variance
nulle)** et **BUY-flat (reward≠0, variance non nulle sur trades gagnants)**. PPO, face à
une action toujours neutre et une action parfois positive, **converge vers la seconde
même si son espérance est négative**. Le fix n'est PAS d'ablater Future Arena mais
d'ajouter un **signal POSITIF pour le HOLD intelligent** : récompenser les steps flat où
le marché AURAIT baissé si on avait acheté.
→ **Implémenté : `ADAN_SMART_FLAT`** (hook anti-oracle, lookahead chunk-futur).
Calibré par mesure (smoke test sur 5m réel) : k=0.05 → mean-active 0.0275 ≈ somme
per-step 0.0043 ; actif ~13.9% des steps flat. Bracket à tester : k ∈ {0.02,0.05,0.10,0.20}.

### 11.6 — Routage : nécessaire mais pas suffisant (accélération ≠ invalidation)
Supprimer les gradients illégaux (routage) rend l'apprentissage plus propre → accélère
AUSSI la convergence vers le minimum dégénéré si le reward y pousse encore. Le routage
est une **condition nécessaire non suffisante** ; dire qu'il "invalide l'hypothèse" est
excessif.

### 11.7 — Couverture 1h/4h dégénérée (réserve sur TOUT "signal exploité")
Mesuré : **1h ne couvre que 14.6%** de la fenêtre 5m (fin @2025-08-15 vs 5m @2026-05-12,
~9 mois manquants) ; 4h couvre 70.6%. Pendant ~85% du run, le canal 1h est figé (ffill).
**Toute conclusion sur "le modèle exploite un signal temporel" doit porter cette réserve.**
Motive le test C3b (shuffle temporel).

### 11.8 — Bug "doublons" reward_components_v12.csv : RÉSOLU (pas un bug de données)
74 lignes / 7 steps uniques, **worker 0 seul, 0 doublon exact** → 74 évaluations reward
RÉELLEMENT distinctes. C'est un artefact de **libellé** (step diag grossier), pas une
corruption. CSV **fiable pour la mesure de magnitude**. Fix libellé = priorité basse.

### 11.9 — Collapse-breaker rendu OPT-IN
Le "tueur de script" (DiagnosticCollapseCallback, return False @pct_buy≥0.97×2) tuerait
un 500k vers ~40–70k, détruisant la plage visuelle voulue. Désormais **OPT-IN via
`ADAN_COLLAPSE_BREAKER=1`** (défaut OFF = télémétrie seule, le run va au bout).

### 11.10 — Run : 500k, PAS 10k
~5h entre sessions → les runs 10k ne servent à rien. Lancer **500k** (breaker OFF,
DIAG_EVERY=500) pour donner à la prochaine session une plage visuelle large, **tout en
analysant pendant l'entraînement**.

---

## 12 — V13.1 SYNTHÈSE : cause confirmée = calibration du coût de portage asymétrique

Session T+3h (2026-07-05). Investigation à une variable près, chaque verdict chiffré.

### 12.1 — Causes ÉLIMINÉES par mesure (pas par intuition)
| hypothèse | statut | preuve |
|---|---|---|
| Espace d'action (BUY/SELL/HOLD, FSM) | disculpé (sessions préc.) | FSM déjà actif en training, collapse persiste |
| `latent_pnl` "purge du PnL latent" | **disculpé** | contribution 0.0-0.6% sur config V13 (n=18,20), 3-4 nonzero |
| `time_decay` symétrique (levier 6-juin) | **disculpé comme FIX** | isolé -0.001: pct_buy@10k=0.90, slope +6e-05 (3.3× pire que holding) |

### 12.2 — Cause CONFIRMÉE
**Mauvaise calibration du coût de portage ASYMÉTRIQUE (`holding_cost`).** Le collapse BUY
vient de l'asymétrie de variance (HOLD-flat reward=0 variance nulle vs position
reward≠0). Le seul levier qui l'attaque directement est un coût **qui ne frappe QUE la
position** (asymétrique). Bracket isolé (std=-2.0, intraday) :

| holding_cost | comportement @10-15k | pct_buy slope [2000,10000] |
|---|---|---|
| 0.006 | dérive BUY lente | +1.8e-05 |
| **0.012** | **testé en run long (équilibre visé)** | — |
| 0.020 | **sur-correction SELL** (pct_buy→0.05, pct_sell→0.94, a0_mean→-0.24) | inversé |

Le fait que 0.02 **inverse** le runaway (BUY→SELL) prouve que le modèle n'est PAS cassé :
il est parfaitement contrôlable, il manquait juste le bon point d'équilibre de portage.

### 12.3 — Facteur confondant MAJEUR découvert : `ADAN_LOG_STD_INIT`
Les lanceurs `launch_500k_v5`/`launch_1M_v13` forçaient `ADAN_LOG_STD_INIT=-1.0`
(std0≈0.37), vs défaut code **-2.0** (std0≈0.135). La std 2.7× plus large accélère la
dérive de a0_mean → une partie du "collapse" observé était de l'**exploration excessive**,
pas seulement du reward. **Tous les runs de validation doivent utiliser -2.0** (défaut).

### 12.4 — Run long en cours (validation horizon complet)
`launch_long_hc012.sh` : **500k steps, holding_cost=0.012, intraday, std=-2.0**,
time_decay/smart_flat OFF, breaker OFF (capture crash complet si collapse @~70k),
diag EVERY=2000, ckpt par step /50k. Objectif : voir si l'équilibre 0.012 tient
au-delà de l'horizon de collapse historique (~70k).

### 12.5 — Prochaines étapes (ordre, si 0.012 tient)
1. Si dérive résiduelle : bracket fin {0.010, 0.012, 0.014}, critère |pct_buy-pct_sell|<0.1 sur 15-20k.
2. Réintégrer `future_contrib` puis `smart_flat` UN PAR UN, vérifier que l'équilibre tient.
3. **Backtest déterministe** (`scripts/backtest/deterministic_backtest.py`) du checkpoint
   final : vérifier qu'il est RENTABLE, pas seulement stable. (Jamais fait — étape logique suivante.)
4. Curriculum via MASQUE (pas via changement d'espace d'action = MDP non-stationnaire que
   PPO déteste) si besoin : verrouiller SELL/tailles au début puis déverrouiller.

## 13. SESSION 2026-07-05 — Vision clarifiée + cause racine MESURÉE

### Critique méthodologique acceptée
FIX A+C avaient été lancés ensemble (confondant). Corrigé : FIX D lancé isolé
(FIX A=information pure ON, FIX C=gates legacy OFF, FIX D=seuil SELL asymétrique).

### Angle mort #1 MESURÉ (plus de raisonnement-sur-code)
diag archfix (PID 256463, logs réels):
  a0_pct_sell @2k = 0.471 (VEUT vendre 47%) mais req_SELL = 0.081 (8% routés).
  Borne haute si a0<0 en LONG -> SELL = 0.316. Manque 23.5 pts.
  => l'intention de sortie (a0 négatif faible -0.10..0) meurt dans la ZONE MORTE
     |a0|<=threshold(0.10) et est routée HOLD. req_SELL est capté AVANT les 3
     gardes -> la perte est dans le ROUTING, PAS les gardes budget/gap/barrier.
     L'agent apprend "SELL ne se déclenche presque jamais" -> arrête -> BUY runaway.

### FIX D (correction de conception, ciblée sur la mesure)
Seuil asymétrique: ENTRÉE=engagement (buy thr 0.10), SORTIE=protection (sell 0.02).
Backward compatible (sell_threshold=None = legacy), unit-testé.

### Résultat SELFIX (FIX A+D, FIX C off) — premières fenêtres
  step  pct_buy pct_sell reqSELL   (vs archfix reqSELL~0.08 au même horizon)
  1000  0.453   0.486    0.261
  2000  0.479   0.464    0.292
  3000  0.511   0.432    0.272
=> req_SELL x3.5 vs archfix. L'agent ferme ENFIN ses positions. Léger BUY-lean
   naissant (a0_mean 0->+0.015) mais SELL soutenu (pas d'effondrement vers 0).
   Angle mort #2 (sur-trading/frais): illegal_ratio ~0.29-0.30 à surveiller.
   PROCHAINE ÉTAPE: horizon long (>50k) pour voir si l'équilibre tient ou dérive.

## 14. SESSION 2026-07-05 (suite) — AUTOPSIE SELFIX @92k = COLLAPSE (verdict A)

VERDICT: collapse INCHANGE dans sa nature (retarde de ~5k au mieux, voire pire).
Run vivant a 92k, mais pct_buy=1.000 et reqSELL=0.000 depuis 22k.

Trajectoire (mesuree):
  phase 1 (1k-8k): reqSELL=0.220 -> FIX D MARCHE (vs archfix 0.08). L'agent VEND.
  onset:  reqSELL<0.10 des 9k ; pct_buy>=0.99 des 16k ; pct_buy=1.000 des 20k.
  phase 2 (>=25k): reqSELL=0.000, pct_sell=0.000, pct_buy=1.000.
  a0_mean: -0.004@1k -> +0.507@25k -> +1.229@50k -> +2.437@90k.
  pente a0_mean = +2.76e-05/step, DIVERGE lineairement, AUCUN plateau.

PREUVE que la cause est le GRADIENT, pas le routing:
  a0_mean diverge sans limite. Meme avec sell_threshold=-0.02, PLUS AUCUN a0 ne
  descend sous -0.02 car TOUTE la distribution a migre vers le positif. FIX D
  (routing) est CONTOURNE par la derive de la policy. Le reward pousse
  activement a0 -> +inf. BUY est inconditionnellement plus payant que SELL/HOLD.

Angle mort #2 (over-trading) REFUTE:
  AGENT_CLOSE=42 vs SL/TP auto=7654 (0.5%). Pas de churn -> paralysie de sortie.
  Portfolio: 20.46 -> 16.13@50k -> 13.98@150k -> 18.11 fin (perte ~30% au creux).

CE QUI EST ACQUIS:
  FIX D cree une fenetre saine de ~8k steps (1ere fois que l'agent vend autant).
  Preuve que rendre la sortie facile AIDE transitoirement. Mais insuffisant seul:
  la cause racine est dans le REWARD (BUY paye tjrs plus). PROCHAINE CIBLE = reward.

## 15. SESSION 2026-07-05 (autopsie confirmatoire @94k) — GRADIENT MESURÉ, ZÉRO PATCH

RÈGLE RESPECTÉE : aucune modification de code. Observation pure. On mesure, on
comprend, on décide ensuite.

### 15.1 Recon
  PID 269201 ALIVE (ETIME cumul CPU 762min, 310% CPU, 1.86 GB RAM).
  diag_selfix_500k.csv = 94 fenêtres (@94k steps). Disk 65G libre (56%).
  Commit autopsie 2c36523 présent en local — PAS ENCORE poussé (origin @ eb2f328).

### 15.2 Trajectoire (94 fenêtres, OLS + IC95%)
  a0_mean     : -0.0044@1k -> +2.5483@94k. pente globale +2.77e-05±3.7e-07/step.
                pente 20 DERNIÈRES fenêtres = +3.75e-05±1.7e-06/step => S'ACCÉLÈRE.
                AUCUN plateau. Divergence monotone non bornée.
  pct_buy     : 0.453@1k -> 1.000@24k (et reste 1.000). pente[10k+]=+2.9e-07.
  pct_sell    : 0.486@1k -> 0.000@24k. 
  req_SELL    : 0.261@1k (FIX D marche) -> <0.10@9k -> 0.000@23k.
  policy_entropy: -0.581 -> -0.476 (monte vers 0 = exploration s'effondre lentement).
  ONSET: reqSELL<0.10@9k | pct_buy>=0.99@16k | pct_buy=1.000@24k | pct_sell=0@24k.
  TRANSITOIRE FIX D: reqSELL>=0.20 uniquement fenêtres 1k-5k (durée ~5k steps).

### 15.3 Angle mort #2 (over-trading) — RÉFUTÉ DÉFINITIVEMENT
  AGENT_CLOSE (fermetures volontaires) = 42.
  SL/TP auto (pre-captured)            = 7840.  Ratio agent/auto = 0.54%.
  DECISION_BUDGET blocks               = 0 (pas de churn).
  Portfolio: oscille bande [12.4 ; 20.5], fin 20.08, delta net -0.38 sur 189k épisodes.
  => PAS d'over-trading. C'est une PARALYSIE DE SORTIE, pas un churn.
     L'excès inverse redouté (frais 0.5% érodant le capital) N'EXISTE PAS ici.

### 15.4 LE GRADIENT DOMINANT — MESURÉ (reward_components_selfix_500k.csv, n=30)
  Agrégation par (état, action) sur les composantes de reward RÉELLES :

    état LONG + BUY  (a0>0, routé no-op HOLD) : n=29  raw_mean = -0.0038  pnl_base=0.0000
    état LONG + HOLD (a0<0, réalise position) : n= 1  raw_mean = -0.3041  pnl_base=-0.3004

  RAPPORT = 0.3041 / 0.0038 ≈ 80×.

  Sortir un a0 POSITIF quand LONG => route_action_by_state ne peut PAS renvoyer
  BUY en position -> tombe dans la branche morte L.7934-7939
  ("discrete_action = 0  # Override to HOLD (neutral, no penalty)") -> pnl_base=0,
  seule la micro symmetry_penalty s'applique (~-0.004). GRATUIT.
  Sortir un a0 NÉGATIF => SELL/réalisation -> pnl_base=-0.30 RÉALISÉ -> PUNI 80×.

  CONCLUSION GRADIENT : le reward enseigne littéralement "a0 positif = sûr/gratuit,
  a0 négatif = risque d'être puni". PPO maximise donc a0 -> +inf (disposition effect :
  "ne jamais réaliser la perte"). FIX D (routing) est CONTOURNÉ car TOUTE la
  distribution a0 migre au-dessus de +0.02 ; plus aucun échantillon ne franchit le
  seuil SELL. Le problème n'est PAS le routage, c'est la STRUCTURE du reward.

### 15.5 VERDICT (grille des 4 catégories)
  A - collapse inchangé  <=== VERDICT
  B - collapse retardé   (retardé ~5k au mieux vs archfix, mais pas supprimé)
  C - comportement nouveau  NON
  D - début d'apprentissage réel  NON

  Q1 pct_sell remonte ?      NON (0.486 -> 0.000).
  Q2 collapse retardé/supprimé ? RETARDÉ (~5k), pas supprimé.
  Q3 budget utilisé ?         N/A (0 blocks, mode silencieux ; sorties inexistantes).
  Q4 ventes réussissent ?     NON (42 AGENT_CLOSE en 94k steps).
  Q5 cycles BUY/SELL/BUY ?    NON — BUY BUY BUY (a0 sature, position figée LONG).

### 15.6 DÉCISION (règle absolue respectée)
  Verdict = A => on identifie le gradient dominant (fait : asymétrie reward 80×).
  ❌ PAS de nouveau patch. ❌ PAS de nouveau holding_cost. ❌ PAS de nouveau reward
  appliqué dans cette session. Le prochain axe de conception (à décider par l'humain)
  vise la SYMÉTRIE DU REWARD : rendre "rester LONG" (a0 no-op) NON gratuit, ou
  créditer explicitement le coût d'opportunité de la position non réalisée, de sorte
  que a0 positif ne soit plus un puits de gradient sans fond. Aucune action tant que
  la décision de design n'est pas validée.

## 16. SESSION 2026-07-06 — MANIFESTO run 6h = COLLAPSE (Cas B), 1/6 tests

**Steps atteints :** 242k (run tue apres verdict). Débit ~1000 steps/7min wall.

### 16.1 Les 3 verifications techniques (protocole)
- V1.1 every=1 : OK. L.6728 lit self.latent_pnl_every_n (surchargé=1 par env),
  L.6741 `_held % _every`. Pas de bug de gating. Applique CHAQUE pas.
- V1.2 vrai cout de vente : cout_vente(u,w=0.5)=u*0.5. steps_pour_egaler ~0.83
  => la calib "conservateur" etait deja AGRESSIVE (vendre rationnel des le 1er pas
  d'une perte). Mais ININFLUENT (voir 16.3).
- V1.3 sur-trading : AGENT_CLOSE=46 vs SL/TP=20124 sur 242k (0.23%). PAS de churn.
  Portfolio 20.5 -> 13.01 (recul, mais par erosion SL/TP en collapse, pas par churn).

### 16.2 Tests S1-S6 (binaires)
  S1 req_SELL>0.10  : 0.000  FAIL
  S2 slope buy<5e-6 : +4e-22 PASS (FAUX POSITIF: pct_buy deja sature=1.0)
  S3 collapse >50k  : @17000 FAIL (identique selfix @16k)
  S4 a0_mean<+1.0   : +2.392 FAIL
  S5 pct_sell>0.05  : 0.000  FAIL
  S6 pct_buy<0.85   : 1.000  FAIL
  => 1/6 (S2 faux positif) => COLLAPSE. Onset @17k, IDENTIQUE a selfix.

### 16.3 POURQUOI le latent lineaire a echoue (MESURE, telemetrie n=80)
  Latent ACTIF (65/80 non-nul) MAIS magnitude ridicule:
    latent_pnl mean=+0.00018 min=-0.00027 max=+0.00127
    long+BUY (no-op, n=65): latent MOYEN = +0.000205 (POSITIF!)
    long+HOLD (realise, n=1): raw=-0.247
  DEUX causes cumulees:
   (1) Les positions sont majoritairement en LEGER GAIN latent (les SL/TP dims 1-4
       coupent les pertes AVANT accumulation) => le latent est POSITIF => il
       RECOMPENSE le maintien au lieu de le punir.
   (2) Une position ne saigne jamais assez LONGTEMPS pour que le latent negatif
       cumule atteigne -0.30 : SL/TP ferment en quelques pas.
  => Le "battement de coeur" ne bat jamais dans le rouge assez fort/longtemps.

### 16.4 CONCLUSION STRATEGIQUE (elimination d'une classe entiere de fixes)
  Tout fix "reward PAR PAS en position" (holding_cost, time_decay, latent log,
  latent lineaire) est STRUCTURELLEMENT VAIN tant que:
   - les SL/TP (Oracle, dims 1-4, INTOUCHABLES) gerent le risque a la place de
     l'agent => rester LONG est objectivement neutre/rentable a court terme;
   - "rester LONG" = sortir un a0 positif = TOUJOURS SUR (le no-op est absorbe).
  L'agent n'apprend pas le disposition effect: il apprend la VERITE de son
  environnement — dans un monde ou un oracle coupe tes pertes, ne jamais vendre
  soi-meme est optimal. Le collapse est RATIONNEL etant donne l'architecture.

  => Le vrai manque, comme dit par l'utilisateur et les critiques: on n'a JAMAIS
     defini "bon trade / mauvais trade / comportement sain / deviation". Le
     prochain chantier n'est PAS un nouveau terme de reward par-pas, mais une
     COUCHE DE COMPORTEMENT (Trader Constitution / Behavior Reward) qui juge le
     CYCLE de trade complet vs un trader de reference, PAS l'action a l'instant t.

**Verdict :** FAIL (1/6). Latent lineaire elimine. **Prochaine etape :**
definir formellement le comportement (module TradeBehaviorAnalyzer + deviation vs
oracle), UNE variable a la fois, avec test S1-S6 prealable — pas un patch de plus.
