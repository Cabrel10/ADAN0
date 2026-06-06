# 🔴 DIAGNOSTIC COMPLET - State of the Union

## Les 3 Vrais Problèmes (Confirmés)

### 1️⃣ **RESTORE RAY EST CASSÉ** (CRITIQUE)
**Symptôme:** Ray dit "✅ Successfully resumed" mais crée une **nouvelle session** (workers `a4a97_*`) au lieu de continuer l'ancienne (`b7791_*`). Tu perds les 80k steps.

**Cause Root:** Le script cherche un fichier fixé `experiment_state-2026-06-04_16-06-16.json` (ligne 1057):
```python
if (restore_path / "experiment_state-2026-06-04_16-06-16.json").exists():
```
Ce chemin est codé en dur → cassé à chaque nouvelle session.

**Fix Requis:** Chercher le fichier `experiment_state-*.json` le plus récent au lieu d'un chemin fixé.

---

### 2️⃣ **BACKTEST EST BUGGÉ** (Données Non Fiables)
**Symptôme:** Workers 1 et 3 ont exactement les mêmes résultats (7 trades, +2.41%, $0.4948 PnL).

**Cause Root:** Le backtest a chargé **le même checkpoint Worker 1** pour les deux tests:
```json
"vecnorm_used": "/mnt/new_data/adan_logs/checkpoints/adan_pbt_training/
ADAN_PBT_Worker_b7791_00001_1_ent_coef=0.0185,gamma=0.9956,
learning_rate=0.0001,worker_idx=1_2026-06-03_17-55-21/checkpoint_000000/vecnormalize.pkl"
```

Tu as seulement **2 workers** (0 et 1) dans la session `b7791` du 3 juin. Il n'existe pas de Worker 3.
L'argument `--ckpt` pour Workers 1 et 3 pointait vers le même chemin.

**Fix Requis:** 
1. Corriger le script pour charger le bon checkpoint par worker_idx
2. Vérifier que Worker 0 a aussi un checkpoint (il n'en a pas actuellement)

---

### 3️⃣ **RAY GCS CRASH MALGRÉ TIMEOUT AUGMENTÉ** (Infrastructure)
**Symptôme:** Process meurt après ~20-25 min: `Failed to connect to GCS`

**Cause Root:** Augmenter le timeout à 600s n'a pas suffi. Causes probables:
- Fuite mémoire dans les workers (RAM occupée passe de 3% à 8%+ rapidement)
- Trop de workers (4) pour le hardware disponible (8 CPU, 15GB RAM, avec overhead Ray + Pandas + PyTorch)
- Conflit `ray.init()` + environnement conda

**Fix Requis:**
1. Réduire à 2 workers au lieu de 4 (mode `--mode light` au lieu de `heavy`)
2. Augmenter encore `RAY_gcs_rpc_server_reconnect_timeout_s` (1200s)
3. Augmenter `RAY_memory = 500_000_000` (500MB per worker)

---

## État Actuel Confirmé

| Aspect | Trouvé | État |
|--------|--------|------|
| **Sessions Ray** | `b7791_*` (2 workers: 0, 1) du 3 juin | ✅ Intact |
| **Checkpoints** | Worker 1: OUI (7.4MB model.zip) | ✅ OK |
| | Worker 0: NON (juste params.json) | ❌ Manquant |
| **Hyperparams Découverts** | Worker 1: gamma=0.9956, ent=0.0185, lr=1e-4 | ✅ Excellent |
| **Performance Worker 1** | Mean Sharpe: 3.02 à 80k steps | ✅ Bon |
| **Backtest Worker 1** | +2.41% sur 500 steps test | ✅ Profitable |
| **Backtest Worker 3** | Identique à Worker 1 (BUG) | ❌ Faux |
| **Resume Session** | Crée nouveaux workers au lieu de restorer | ❌ Cassé |
| **Ray Uptime** | ~20-25 min avant GCS crash | ❌ Instable |

---

## Ce Qu'IL FAUT FAIRE (EN ORDRE DE PRIORITÉ)

### PRIORITÉ 1: Corriger le Restore Ray
**Action:** Modifier `train_parallel_agents.py` ligne ~1057

Remplacer:
```python
if (restore_path / "experiment_state-2026-06-04_16-06-16.json").exists():
```

Par:
```python
# Chercher le fichier experiment_state-*.json le plus récent
import glob
exp_states = sorted(glob.glob(str(restore_path / "experiment_state-*.json")))
if exp_states:
```

**Objectif:** Permettre à Ray de trouver et restaurer automatiquement la session précédente sans chemin fixé.

**Impact:** Récupérer les 80k steps perdus et les hyperparams optimaux (gamma=0.9956).

---

### PRIORITÉ 2: Corriger le Backtest
**Action:** Revoir `deterministic_backtest.py`

Problèmes à corriger:
1. La fonction `find_latest_checkpoint()` cherche le vieux chemin `checkpoints/ppo_adan0_sandbox_*.zip`
2. Pas de logique pour charger des checkpoints différents par worker_idx
3. Besoin de supporter `/mnt/new_data/adan_logs/checkpoints/adan_pbt_training/ADAN_PBT_Worker_b7791_*`

**Objectif:** Pouvoir faire:
```bash
python scripts/deterministic_backtest.py \
  --worker 0 \
  --ckpt-dir /mnt/new_data/adan_logs/checkpoints/adan_pbt_training/
```

Et charger automatiquement le bon checkpoint par worker_idx.

**Impact:** Obtenir des résultats backtest valides pour comparer les workers.

---

### PRIORITÉ 3: Stabiliser Ray
**Action:** Modifier `train_parallel_agents.py` ligne ~136-142 (Ray env vars)

```python
# Avant
os.environ["RAY_gcs_rpc_server_reconnect_timeout_s"] = "600"

# Après
os.environ["RAY_gcs_rpc_server_reconnect_timeout_s"] = "1200"
os.environ["RAY_memory"] = str(500_000_000)  # 500MB per worker
os.environ["RAY_object_store_memory"] = str(4_000_000_000)  # 4GB object store
```

**Additionnel:** Réduire le nombre de workers:
```bash
python scripts/train_parallel_agents.py --mode light --steps 500000
# --mode light = 2 workers au lieu de 4
```

**Objectif:** Empêcher les crashes GCS après 20-25 min.

**Impact:** Entraînement peut tourner 20+ heures sans interruption.

---

### PRIORITÉ 4: Valider les Fixes
Une fois les 3 priorités traitées:

1. Lancer l'entraînement avec restore:
```bash
python scripts/train_parallel_agents.py \
  --mode light \
  --steps 100000 \
  --checkpoint-dir /mnt/new_data/adan_logs/checkpoints/adan_pbt_training/adan_pbt_training \
  --resume
```

2. Laisser tourner 30 min sans crash
3. Arrêter proprement
4. Backtest tous les workers:
```bash
python scripts/deterministic_backtest.py --worker 0 --ckpt-dir ...
python scripts/deterministic_backtest.py --worker 1 --ckpt-dir ...
```

5. Comparer les résultats (doivent être différents cette fois!)

---

## Code Changes Required

### File 1: `scripts/train_parallel_agents.py`

**Change 1** (Line ~1057): Fix experiment_state.json hardcoded path
```python
# BEFORE
if (restore_path / "experiment_state-2026-06-04_16-06-16.json").exists():

# AFTER
import glob as _glob
exp_states = sorted(_glob.glob(str(restore_path / "experiment_state-*.json")))
if exp_states:
    # Use the most recent one
    exp_state_file = exp_states[-1]
```

**Change 2** (Line ~136-142): Increase Ray timeouts and memory
```python
# Add/modify
os.environ["RAY_gcs_rpc_server_reconnect_timeout_s"] = "1200"
os.environ["RAY_memory"] = str(500_000_000)
os.environ["RAY_object_store_memory"] = str(4_000_000_000)
os.environ["RAY_task_retry_delay_ms"] = "5000"
```

### File 2: `scripts/deterministic_backtest.py`

**Change 1**: Support Ray PBT checkpoint structure
```python
# ADD function
def find_pbt_checkpoint(worker_idx: int, ckpt_dir: str) -> str | None:
    """Find checkpoint for specific worker in PBT structure."""
    import glob as _glob
    pattern = f"{ckpt_dir}/ADAN_PBT_Worker_*_worker_idx={worker_idx}_*/checkpoint_*/model.zip"
    ckpts = sorted(_glob.glob(pattern), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None
```

**Change 2**: Update main() to accept worker_idx
```python
parser.add_argument("--worker", type=int, default=None)
parser.add_argument("--ckpt-dir", type=str, default=None)

if args.worker is not None and args.ckpt_dir:
    ckpt = find_pbt_checkpoint(args.worker, args.ckpt_dir)
```

---

## Timeline Estimé

| Step | Time | What | Status |
|------|------|------|--------|
| NOW | 0h | Arrêter tous les processes | ✅ DONE |
| FIX 1 | 30 min | Corriger restore Ray | ⏳ À faire |
| FIX 2 | 30 min | Corriger backtest | ⏳ À faire |
| FIX 3 | 15 min | Augmenter timeouts Ray | ⏳ À faire |
| VALIDATE | 1h | Lancer test 30 min | ⏳ À faire |
| BACKTEST | 30 min | Tester tous les workers | ⏳ À faire |
| **TOTAL** | **~2-3h** | **Tous les bugs fixes** | **→ GO!** |

---

## Important Notes

1. **NE TOUCHE PAS** à `config.yaml` ou `agent.yaml` — Ray gère les hyperparams via checkpoint
2. **Les hyperparams optimaux sont stockés** dans le checkpoint Worker 1 (gamma=0.9956, ent=0.0185) — une fois le restore fixé, ils seront restaurés automatiquement
3. **Worker 0 n'a pas de model.zip** — c'est OK, juste les params.json. À relancer après les fixes.
4. **Il n'y a que 2 workers** (0 et 1) dans la session b7791 du 3 juin — c'est normal, tu en avais lancé 4 mais 2 ont crashé

---

Ready? On implémente les fixes maintenant.
