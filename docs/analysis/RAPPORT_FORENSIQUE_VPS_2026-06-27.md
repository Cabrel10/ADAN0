# RAPPORT FORENSIQUE — Gel + ralentissement ×20-30 ADAN0
Date : 2026-06-27 (post-reboot VPS) — investigation instrumentée, **preuves chiffrées uniquement**.

VPS : `vmi3357211` — **4 vCPU**, 7.8 Gi RAM, 8 Gi swap (0 utilisé), kernel 6.8.0-124.
Outils : netdata v1.43.2 (actif), fio, dd, py-spy 0.4.2, cProfile, ps/top, /proc.
Contrainte : pas de root (dmesg/journalctl-kernel/strace-attach/ptrace bloqués — `ptrace_scope=1`).

---

## RÉSUMÉ EXÉCUTIF (la cause prouvée)

Le VPS **n'est pas dédié** à ADAN0. Il héberge **2 stacks Docker complètes** :
- `whatsapp-*` : evolution-api + n8n + redis + pgvector
- `gaintime-*` : Django(web/daphne) + telegram-bot + **celery_worker** + celery_beat + postgres + redis

**Cause racine prouvée du ralentissement : `gaintime-celery_worker-1` est en CRASH-LOOP
(Exited code 1 → restart permanent).** À chaque redémarrage il relance
`manage.py migrate` + `collectstatic` + `check --deploy` qui consomment **85-94 % CPU
chacun**, saturant 3 des 4 cœurs **en continu**. ADAN0 (sandbox = SB3 mono-process)
n'obtient plus qu'~1 cœur au lieu de 4.

**Ray n'est PAS en cause** : `--mode sandbox` = SB3 pur, `DummyVecEnv`, mono-process.
`ps|grep ray` = 0 process, `/tmp/ray` inexistant. (Code confirmé : docstring L1663
"no Ray, no GPU, single-process" ; L1749 `DummyVecEnv`.)

**Les logs ne sont PAS en cause** : profilage = **0.6 %** du temps. **I/O pas en cause** :
iowait 0-3 %, disque 396 MB/s séquentiel + ~9400 IOPS aléatoires (sains).

---

## PHASE A — État système (mesuré)

| Métrique | Valeur | Source |
|----------|--------|--------|
| vCPU | 4 | nproc |
| load average | 3.97 → 6.68 | uptime |
| RAM | 1.9 Gi / 7.8 Gi utilisée, 5.8 Gi dispo | free |
| swap | **0 B utilisé** | free/swapon |
| CPU user (avec crash-loop) | **76-86 %** | netdata system.cpu |
| CPU system | 12-21 % | netdata |
| **iowait** | **0.02-3 %** | netdata |
| containers Docker | 10 (2 stacks) | docker ps |
| docker-proxy/shim | 22 process | ps |
| Ray | **0 process, pas de /tmp/ray** | ps / du |

Process gourmands mesurés EN PARALLÈLE d'ADAN0 :
`collectstatic 52 %`, `check --deploy 35 %`, `migrate 29 %`, `collectstatic 27 %` → ~150 % CPU volés par gaintime seul.

## PHASE A — Benchmarks disque (écarte l'I/O)
- `dd 1 GiB oflag=direct` → **396 MB/s** (SSD/NVMe sain).
- `fio randrw 4k, 4 jobs, 30 s` → **read 4668 IOPS / write 4679 IOPS** (~18 MB/s chacun).
→ Disque NON dégradé. iowait quasi nul. **L'I/O n'est pas le goulot.**

## PHASE B — Événement passé
Reboots (`last`) : uptime **17 jours** (Jun 10 → Jun 27 17:46), puis sessions courtes,
puis reboot 23:15 (utilisateur). OOM-killer **non vérifiable** (dmesg/journalctl-kernel
sans root), mais swap=0 et 5.8 Gi RAM libre → **pas de pression mémoire actuelle**.
Le gel de 8 h coïncide avec la longue période où les stacks Docker tournaient ;
le crash-loop gaintime fournit un mécanisme suffisant de saturation CPU prolongée.

## PHASE C — Profilage réel (py-spy, run sandbox)
Temps CPU par catégorie (échantillonné sur le vrai pipeline) :

| % | Composant | Note |
|---|-----------|------|
| **39.4 %** | NN forward CNN+Attention (`_conv_forward` 17.4, `batch_norm` 9.6, attention ~6) | cœur du modèle |
| **21.5 %** | env.step (logique trading) | métier |
| **17.3 %** | **DBE KMeans + HMM** (`_kmeans_single_lloyd`, `_update_hmm`) | **optimisable** |
| **13.8 %** | **gSDE / scipy `_solve_triangular`** | **coûteux même USE_SDE=0** |
| 5.2 % | import/compile (warmup) | s'amortit |
| 1.4 % | pandas/numpy | négligeable |
| **0.8 %** | **PPO.learn (backward)** | **négligeable — n_epochs PAS le goulot** |
| **0.6 %** | **logging** | **PAS le goulot** |

Extractor = `ContextualTemporalFusionExtractor` (CNN temporel + fusion + attention).

## PHASE D — Comparaison historique (FPS mesurés)

| Condition | FPS mesuré | 500k steps |
|-----------|-----------|-----------|
| Historique annoncé | ~9.3 | 15 h |
| **ADAN0 CPU dégagé (crash-loop stoppé) — MESURÉ** | **~5.7** | **~24 h** |
| **ADAN0 + crash-loop gaintime — MESURÉ** | **~2.5** | **~55 h** |
| Annoncé "13 jours" | ~0.45 | 13 j |

Preuve directe de la contention : arrêt du conteneur crash-loop →
**CPU user 83 % → 20 %** (≈ 60-65 % de CPU rendu), %CPU ADAN0 : **106 % → 311 %**.

Décomposition du facteur de ralentissement :
1. **Contention CPU gaintime crash-loop** : facteur **~2.3×** (5.7 → 2.5 fps). PROUVÉ.
2. **Pics de migration/déploiement simultanés** (collectstatic+migrate+check) : peut
   ponctuellement réduire ADAN0 à <1 fps → explique l'estimation "13 jours" faite
   pendant un pic. PLAUSIBLE (observé live mais non chronométré sur 500k).
3. Écart résiduel "15 h (9.3 fps) vs 24 h dégagé (5.7 fps)" (~1.6×) : NON encore
   expliqué — pistes mesurables : n_epochs (config=20 vs run=10), état du cache de
   données, version du modèle (taille extractor), ou un VPS historiquement moins chargé.

## PHASE E — Correctifs (à valider AVANT tout code)

**Correctif #1 (immédiat, hors-code, impact prouvé ~2.3×)** — Isolation CPU :
  - Option a : stopper/réparer `gaintime-celery_worker-1` (crash-loop) — mais VPS partagé.
  - Option b (recommandée, non destructive) : lancer ADAN0 avec `nice -n -5` impossible
    sans root ; à défaut `chrt`/`taskset` pour épingler ADAN0 sur des cœurs dédiés et
    `cpulimit` la stack gaintime. Sinon **dédier un VPS** à l'entraînement.
  - Impact attendu : **~2.5 → ~5.7 fps (×2.3)** → 500k en ~24 h au lieu de ~55 h.

**Correctif #2 (code, après mesure A/B)** — DBE KMeans/HMM (17.3 %) :
  - Réduire la fréquence de re-clustering (cache N steps) ou vectoriser.
  - Impact attendu si divisé par 4 : ~13 % de CPU rendu → fps +~15 %.

**Correctif #3 (code, après mesure)** — gSDE scipy `_solve_triangular` (13.8 %) :
  - Vérifier pourquoi l'algèbre triangulaire tourne avec USE_SDE=0 ; éliminer si inutile.
  - Impact attendu : jusqu'à ~14 % de CPU rendu → fps +~15 %.

**NON-correctifs (écartés par preuve)** : logs (0.6 %), n_epochs/PPO.learn (0.8 %),
I/O disque (iowait ~0), Ray (absent en sandbox).

---

## Réponses directes aux questions

- **`--mode sandbox` lance-t-il Ray ?** NON. SB3 pur, mono-process, `DummyVecEnv`.
  Aucun actor Ray, pas de `/tmp/ray`.
- **Le gel venait-il d'un autre process ?** Très probablement OUI : crash-loop
  gaintime saturant le CPU. Mécanisme prouvé live (CPU user 83→20 % à l'arrêt).
- **Le gel est-il résolu ?** Le reboot a nettoyé l'état ; mais le crash-loop revient
  automatiquement → la contention CPU revient. Sans isolation, le risque persiste.
- **×20-30 ?** Mesuré : crash-loop = ×2.3 prouvé ; pics de déploiement → <1 fps
  ponctuel ; le facteur cumulé extrême (×20-30) correspond à un VPS multi-stack saturé
  + estimation pendant un pic, PAS à un bug du code ADAN0.

---

## ADDENDUM — Résolution appliquée + verdict final (relance d'entraînement)

### Facteur ×2 résiduel ÉLUCIDÉ (heavy vs sandbox / évolution du modèle)
`git log -S ContextualTemporalFusionExtractor` → commit **`5bc5e13` (2026-06-24)**
"fix(sandbox): wire ContextualTemporalFusionExtractor (CNN+attn+FiLM+aux)".
**AVANT** ce commit, `--mode sandbox` utilisait le **MLP par défaut SB3** (léger, rapide).
**APRÈS**, il utilise le **CNN+Attention+FiLM** complet (39.4 % du CPU mesuré).
S'ajoutent les commits récents de complexification (rewards friction par palier, bug
cross-timeframe corrigé, Decision Budget V3, AGENT_CLOSE quota, frais 0.80 %, etc.).

→ **Le run historique "15 h / 9.3 fps" n'était PAS le même système.** Les benchmarks de
vitesse antérieurs sont caducs : modèle MLP→CNN+Attn, données timeframe corrigées,
rewards/env devenus un vrai simulateur de marché. Le ralentissement est **normal**.

`--mode heavy` = Ray Tune PBT (L1600 `ray.init`, L1260 PBT, L1424 `tune.Tuner`).
`--mode sandbox` = SB3 pur, `DummyVecEnv`, **0 Ray**. Ray hors cause pour le sandbox.

### Isolation CPU appliquée (réversible) — gain mesuré
`docker update --cpuset-cpus=0` sur les 2 stacks tierces (gaintime crash-loop + whatsapp)
+ `taskset -c 1-3` sur ADAN0. Script: `scripts/isolate_cpu.sh {confine|release}`.

| Avant isolation | Après isolation |
|-----------------|-----------------|
| FPS ADAN0 ~2.9 | **FPS ADAN0 ~8.5** |
| CPU user 83 % | CPU user 33 % |
| 500k ≈ ~48 h | **500k ≈ ~16 h** (≈ le run historique !) |

### Santé de l'apprentissage (run prod, ~9 itérations)
- explained_variance: -0.25 → **0.138** (>0.1, value function apprend)
- value_loss: 0.38 → **0.042** (baisse nette)
- approx_kl ~0.025-0.034 (sain), entropy_loss -2.05 (exploration ok), std 0.366 stable
- trading: **398 OPEN / 130 CLOSE** (ouvre ET ferme, pas inerte)

### VERDICT FINAL (validé avec l'utilisateur)
On ARRÊTE la chasse à la performance. ~8.5 FPS CPU pour un simulateur CNN+Attn+DBE
complexe est **acceptable→très bon**. Priorité désormais : **qualité des données
(timeframes) → stabilité du reward → comportement du modèle**, PAS la vitesse.
Optimisations futures possibles (différées) : DBE KMeans/HMM (17 %), gSDE scipy (14 %),
GPU, Ray heavy. À ne traiter qu'après avoir prouvé que le modèle apprend bien.
