# Diagnostic du gel à 12417 steps — conclusion fondée sur preuves

> Méthode imposée : tester H1/H2/H3 avec preuves mesurables, **ne PAS** supposer
> qu'une cause est vraie à 100 %. Ce document liste les preuves, pas des théories.

## Faits établis (mesurés)

| # | Fait | Preuve |
|---|---|---|
| F1 | Run `fa_500k_v4` gelé à **step 12417** | `logs/training/fa_500k_v4.log`, max `[STEP`=12417 |
| F2 | Gel **sans exception ni traceback** | `grep -icE "Traceback\|Exception\|Error:"` = **0** |
| F3 | Dernière sortie = **table d'update PPO complète** (n_updates=1580, loss, std, value_loss) puis **plus rien** | `tail -6 fa_500k_v4.log` |
| F4 | Run `fa_500k_v5_omp1` (avec `OMP_NUM_THREADS=1`) a atteint **40000 steps** sans gel | checkpoints 10k(20:08)/20k(20:24)/30k(20:40)/40k(20:55) |
| F5 | Cadence de checkpoint **parfaitement stable** : 16/16/15 min par 10k | timestamps ci-dessus |
| F6 | Taille d'un checkpoint = **7,4 Mo** (pas 100 Mo, pas 600 Mo) | `ls -la checkpoints/*.zip` |
| F7 | Disque **77 % util., 35 Go libres** ; inodes **8 %** | `df -h /home`, `df -i /home` |
| F8 | `checkpoint_frequency`, `SHOULD_CHECKPOINT`, `checkpoint_at_end` : **absents** | grep code = vide |
| F9 | `_save_checkpoint_on_promotion` : appelé **seulement si** `checkpoint_on_promotion` (absent de config.yaml) **et** `self.model` (jamais défini sur l'env en sandbox) | env L2519-2543 |
| F10 | `_save_adaptation_state` (AdaptiveDBE) **non importé** par l'env sandbox | grep `adaptive_dbe` dans env = vide |
| F11 | 0 fichier `model_*_promo*`, 0 `adaptive_dbe_state_*.json` produit | `find` = 0 |

## Évaluation des hypothèses (révisée par les preuves)

### H1 — saturation I/O par save_checkpoint  →  **RÉFUTÉE** (preuves F5–F11)
- Checkpoint = 7,4 Mo toutes les ~16 min = **0,008 Mo/s** : négligeable.
- Cadence **constante** (pas de ralentissement cumulatif) → pas de saturation progressive.
- Disque/inodes loin de la saturation.
- Les chemins de sauvegarde "lourds" (promotion, adaptation) **ne s'exécutent jamais**
  en sandbox (F9–F11). L'hypothèse "600 Mo toutes les 12k" est **fausse**.
- Probabilité réévaluée : **~5 %**.

### H2 — watcher Bash destructif  →  **PLAUSIBLE mais NON PROUVÉ ; neutralisé**
- Le watcher `surveil_fa_500k.sh` faisait `tail` + troncature (`tail -n 5000 > tmp; mv`)
  sur le log pendant que Python écrivait : **risque réel** de désync de file
  descriptor / blocage sur `flush()`.
- MAIS je n'ai **pas** la corrélation temporelle exacte entre une troncature et le
  gel original de 8 h (logs de ce run non conservés). Je ne peux donc **pas** l'affirmer.
- Action conservatoire : **watcher Bash supprimé du dépôt** ; remplacé par un
  watchdog Python **lecture seule** (jamais de troncature). Risque éliminé par design.
- Probabilité réévaluée : **~25 %** (cause possible du run 8 h, pas du gel v4 à 12417
  car ce run-là n'avait pas de watcher actif au moment du gel).

### H3 — deadlock OpenMP/PyTorch  →  **SOUTENUE par les preuves** (F2, F3, F4)
- Signature : arrêt **propre, sans erreur**, **juste après une update PPO** (F2+F3) =
  signature classique d'un **deadlock de pool de threads** (pas un crash, pas une boucle).
- **Test différentiel décisif (F4)** :
  - v4 **sans** limite de threads → **gel à 12417**.
  - omp1 **avec** `OMP_NUM_THREADS=1` → **40000 steps, aucun gel**.
- Le seul facteur changé entre les deux est la limitation des threads.
- Probabilité réévaluée : **~60 %** (cause la plus soutenue par les données).

## Conclusion (sans prétention de certitude absolue)

> La preuve la plus forte pointe vers **H3 (contention/deadlock OpenMP)** comme cause
> du gel à 12417 : signature « arrêt propre post-update PPO » + test différentiel
> (OMP=1 passe 40000, sans limite gèle à 12417). **H1 est réfutée** par les mesures.
> **H2 reste un risque réel** (désync FD) que l'on **neutralise par conception** en
> supprimant toute rotation destructive externe.

**Correctifs retenus (du plus sûr au plus risqué)** :
1. (H3) `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_NUM_THREADS=1` + `torch.set_num_threads(1)`
   + `set_num_interop_threads(1)` — **appliqué**, validé par F4.
2. (H2) Suppression du watcher Bash destructif ; watchdog Python lecture seule —
   **appliqué**.
3. (perf/log) Réduction de la verbosité per-step (STEP/REWARD/RISK_PARAMS) —
   **en cours** (le mode SILENT n'a pas fonctionné, 44325 lignes INFO subsistent ;
   correctif déterministe INFO→DEBUG à appliquer).
4. (test) n_epochs 20→10 surchargeable — **appliqué** (réduit la fenêtre de backward).

## Critère de validation avant 500k
Run par paliers **20k → 50k → 100k** sans : gel, step figé, exception ; CPU/RAM
stables ; logs qui progressent ; checkpoints fonctionnels. **Seulement ensuite** : 500k.
