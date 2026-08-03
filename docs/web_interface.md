# ADAN0 Terminal — Web Interface / Mission Control

> **Statut** : Plan validé. Phase 1 (MVP Training Monitor) en construction.
> **But** : centre de contrôle web pour suivre ADAN0 en temps réel
> (entraînement, backtest, paper, live) sans SSH sur le VPS.
> **Contrainte projet (rappel)** : les frais restent intacts
> (`commission: 0.0025`, `round_trip_fees: 0.005`). Le site est **read-only**
> par défaut sur la logique de trading ; il observe, il ne modifie pas le
> reward shaping ni les frais.

---

## 0. Principes directeurs

1. **Construire sur l'existant** : le site lit les artefacts RÉELS déjà
   produits par `scripts/train_parallel_agents.py` (logs, CSV de télémétrie,
   checkpoints). Aucun nouveau format inventé tant que l'existant suffit.
2. **Faits, pas hypothèses** : chaque widget affiche une donnée mesurée
   (timestep, a0_std, illegal_ratio, PnL réel d'un backtest), jamais une
   estimation.
3. **Découplage** : le backend n'importe PAS le code lourd de l'env
   (`multi_asset_chunked_env.py`, 430 KB). Il parse fichiers + appelle des
   scripts CLI existants en sous-processus. Le training tourne dans son propre
   process ; le site ne le bloque jamais.
4. **Léger d'abord** : VPS mémoire-contraint (~0.9 Gi libres pendant un run).
   Le MVP n'ajoute ni PostgreSQL ni Redis ni Ray API tant que SQLite + lecture
   de fichiers suffisent. PostgreSQL/Redis = phase ultérieure si besoin réel.

---

## 1. Sources de données réelles (FAITS)

Le site s'appuie sur ces fichiers/chemins existants :

| Source | Chemin réel | Contenu |
|--------|-------------|---------|
| Log d'entraînement | `logs/training/train_v4_500k.log` | stdout SB3 : `total_timesteps`, losses, ckpt |
| Télémétrie collapse | `logs/training/diagnostic_collapse_v4.csv` | CSV écrit tous `ADAN_DIAG_EVERY` steps |
| Checkpoints | `checkpoints/ppo_adan0_sandbox_checkpoint_<N>_steps.zip` | modèles SB3 sauvés tous `ADAN_CKPT_FREQ` |
| VecNormalize | `checkpoints/..._vecnorm.pkl` | normalisation associée au ckpt |
| Config | `config/config.yaml` | hyperparams, frais, reward shaping |
| Backtests | `logs/training/*.log`, sorties `scripts/backtest/*` | résultats forensic / equity |

### 1.1 Schéma du CSV de télémétrie (colonnes réelles)

```
timesteps, a0_mean, a0_std, a0_pct_buy, a0_pct_sell, a0_pct_hold_band,
req_HOLD_pct, req_BUY_pct, req_SELL_pct, steps_flat_pct, steps_open_pct,
illegal_ratio, policy_entropy, a0_histo
```

`a0_histo` = 10 buckets `|`-séparés (histogramme de l'action0 sur la fenêtre).

Ces colonnes pilotent directement les graphiques de l'onglet **Training**
(détection de collapse : `a0_std` ↑, `req_HOLD_pct` ↓, `illegal_ratio` ↑).

---

## 2. Architecture

### 2.1 MVP (Phase 1) — léger, ce qu'on construit maintenant

```
Browser (React + TS + Vite + Tailwind)
        │  HTTP (REST) + WebSocket
        ▼
FastAPI (uvicorn, 1 worker)
        │
        ├── LogTailService     → parse train_*.log (regex SB3)
        ├── TelemetryService   → lit diagnostic_collapse_v4.csv (incrémental)
        ├── CheckpointService  → liste checkpoints/*.zip + métadonnées
        ├── ConfigService      → lit config.yaml (read-only)
        └── ProcessService     → ps/PID du run + CPU/MEM + free -h
        │
        ▼
SQLite (cache léger : runs détectés, snapshots métriques)
fichiers du repo (source de vérité)
```

> Pas de PostgreSQL / Redis / Ray au MVP. Ils sont prévus en **Phase 3**
> uniquement si le besoin est mesuré (multi-runs concurrents, historique lourd).

### 2.2 Cible (Phases 2-3) — la vision « Trading Operating System »

```
Browser ─ React/TS/Vite/Tailwind/Shadcn ─ TanStack Query ─ Zustand
   │
   ▼  REST + WebSocket + SSE
FastAPI Gateway
   ├── Training Service     (logs, télémétrie, lancement/arrêt de runs)
   ├── Backtest Service     (wrap scripts/backtest/*)
   ├── Paper Trading Service
   ├── Live Trading Service
   ├── Data Service         (parquet OHLCV / indicateurs)
   ├── Model Registry       (checkpoints versionnés)
   ├── System Service       (CPU/RAM/GPU/Docker)
   └── Alert Service        (telegram/discord/email/web push)
   ▼
PostgreSQL (runs/trades/models) · TimescaleDB (séries temps réel) ·
Redis (streaming/WS) · MinIO ou FS (checkpoints/exports)
```

---

## 3. Stack technique

**Backend** : Python 3.11 (conda `trading_env`), FastAPI, uvicorn, pydantic,
SQLAlchemy (SQLite au MVP → PostgreSQL plus tard), `watchfiles` pour le tail.

**Frontend** : React 18 + TypeScript + Vite, TailwindCSS, composants type
Shadcn, TanStack Query (fetch/cache), Zustand (état global léger),
graphiques : Recharts/ECharts au MVP ; TradingView Lightweight Charts pour le
chandelier/replay en Phase 2.

**Temps réel** : WebSocket (FastAPI) pour pousser les nouvelles lignes de log
et de télémétrie ; fallback polling TanStack Query (refetchInterval).

**Déploiement** : `uvicorn` + `vite build` servi en statique. Docker Compose
fourni en Phase 3 (frontend, backend, postgres, redis, nginx).

**Thème** : Bloomberg + Cyberpunk.
`bg #09090B`, `panels #18181B`, vert `#22C55E`, rouge `#EF4444`,
orange `#F97316`, bleu `#3B82F6`, violet `#8B5CF6`.

---

## 4. Onglets / navigation

```
ADAN0 Terminal
├── Dashboard   (Mission Control : vue globale santé)
├── Training    ★ priorité MVP — suivi du run en temps réel
├── Backtest    (equity, trades, MAE/MFE, confusion matrix)
├── Paper       (positions/PnL simulé)
├── Live        (exchange, capital, risque, alertes)
├── Research    (campagnes scripts/research/*)
├── Models      (checkpoint explorer + registry)
├── Agents      (multi-agents : tâches/statut/logs)
└── System      (CPU/RAM/swap/process/Docker)
```

### 4.1 Dashboard (Mission Control)
Cartes : timestep courant / 500k, vitesse (steps/min), état du process
(PID, CPU, MEM), RAM/swap du VPS, dernier checkpoint, indicateur collapse
(feu vert/orange/rouge selon `a0_std` & `illegal_ratio`), nb de trades du
dernier rollout.

### 4.2 Training ★ (MVP)
- **Liste des runs détectés** (à partir des logs/checkpoints).
- **Courbes temps réel** : `total_timesteps`, `policy_entropy`, `a0_std`,
  `req_HOLD/BUY/SELL_pct`, `illegal_ratio`, `steps_flat/open_pct`.
- **Histogramme action0** (`a0_histo`, 10 buckets) → visualisation directe du
  collapse bimodal ±1.
- **Indicateur collapse** : règle de décision (FAITS) —
  collapse si `a0_std` croît en tendance ET `req_HOLD_pct` → 0 ET
  `illegal_ratio` → 1.
- **Console live** : tail des dernières lignes de `train_v4_500k.log`
  (filtrable : ERROR / checkpoint / STERILE).
- **Détection d'erreur** : bannière rouge si une ligne `Error in step` ou
  `NameError` apparaît (régression de type bug `_base`).

### 4.3 Backtest / Paper / Live / Research / Models / Agents / System
Spécifiés ici pour les phases suivantes (cf. §6 roadmap). Chaque onglet
réutilise des scripts existants (`scripts/backtest/forensic_trades.py`,
`scripts/research/*`) exécutés en sous-process et dont la sortie est parsée.

---

## 5. API backend (MVP)

| Méthode | Route | Réponse |
|---------|-------|---------|
| GET | `/api/health` | `{status:"ok"}` |
| GET | `/api/runs` | runs détectés (logs + checkpoints) |
| GET | `/api/training/status` | PID, CPU, MEM, timestep, vitesse, état |
| GET | `/api/training/telemetry?since=N` | lignes CSV depuis le step N |
| GET | `/api/training/log?tail=200` | dernières lignes du log |
| GET | `/api/training/collapse` | verdict collapse (FAITS) + raisons |
| GET | `/api/checkpoints` | liste des `.zip` + step + taille + mtime |
| GET | `/api/config` | extrait sûr du config.yaml (frais, reward, sandbox) |
| GET | `/api/system` | `free -h`, charge CPU, swap |
| WS  | `/ws/training` | push télémétrie + log en continu |

Sécurité MVP : bind `0.0.0.0`, exposé via GetServiceUrl ; lecture seule ;
aucune route ne modifie config.yaml ni les frais.

---

## 6. Roadmap par phases

- **Phase 1 (MVP, en cours)** : backend FastAPI + onglet Training live + Dashboard.
  Lit le run 500k V4 en cours. Recharts/ECharts. SQLite optionnel.
- **Phase 2** : Backtest Studio (equity/trades/MAE-MFE/confusion), Checkpoint
  Explorer, Market Replay (TradingView Lightweight Charts), comparateur de runs.
- **Phase 3** : Paper/Live centers, Alert center (telegram/discord), Model
  Registry, System/Docker, PostgreSQL+TimescaleDB+Redis, Docker Compose,
  multi-agents.

---

## 7. Arborescence du code (dans le repo)

```
web/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + routes + WS
│   │   ├── services/
│   │   │   ├── log_service.py
│   │   │   ├── telemetry_service.py
│   │   │   ├── checkpoint_service.py
│   │   │   ├── config_service.py
│   │   │   └── system_service.py
│   │   └── settings.py          # chemins repo (logs/, checkpoints/, config/)
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.js
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── api/client.ts
        ├── store.ts
        ├── components/  (cards, charts, console, badges)
        └── tabs/  (Dashboard, Training, ...)
```

---

## 8. Critères d'acceptation MVP

1. Ouvrir l'URL → voir le **timestep courant** du run 500k qui **augmente**.
2. Voir les courbes `a0_std`, `req_HOLD_pct`, `illegal_ratio` se mettre à jour.
3. Voir l'histogramme action0 et l'indicateur collapse (vert au début).
4. Voir la console log live et une bannière rouge si une erreur survient.
5. Aucune modification d'aucun fichier de config (frais intacts garantis).
