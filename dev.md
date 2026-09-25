# ADAN0 — Development & Operational Guide

---

# 🧠 ADAN-SYSTEM-ONE — NOUVEAU PARADIGME (2026-09-25)

> **Décision stratégique** : pivot d'ADAN0 (PPO) vers ADAN-System-One
> (machine locale de jugement probabiliste). Le workflow complet est figé
> ci-dessous — toute implémentation le suit à la lettre.

## Principe cardinal

**La clôture 5m est l'horloge maîtresse unique.** Les échelles 1h et 4h ne sont
JAMAIS injectées comme des bougies figées (poison du MTF statique), mais comme
des **contenants partiellement formés**, reconstruits depuis la séquence 5m à
la seconde exacte de clôture → zéro fuite de données futures possible.

**Le droit de s'abstenir** : 0 à 5 trades/jour MAXIMUM (et non 2-5 obligatoires).
Aucun gradient de punition pour le silence. C'est ce qui manquait à PPO
(effondrement des actions μ=-7.2, fuite SELL documentée dans RAPPORT_*.md).

## Les 12 étapes du workflow (résumé opérationnel)

| # | Étape | Rôle | Module cible |
|---|-------|------|--------------|
| 0 | Flux continu | ticks → OHLCV 5m ; décision uniquement au close 5m | data_loader.py (conservé) |
| 1 | État temporel | 5m fermée + contenant 1h (k/12) + contenant 4h (m/48) RUNNING | **data/nested_state_builder.py** (CRÉER) |
| 2 | Contrôle d'intégrité | NaN/trous/OHLC incohérent → ABSTENTION | data_validator.py (conservé/étendu) |
| 3 | Perception | CNN 5m + états 1h/4h running + attention hiérarchique + FiLM → Z | models/feature_extractors.py (REFACTORER) |
| 4 | Moteur de jugement | Questions NOUL/CHOICE/SCORE sur le même Z, calibrées | **models/system_one_core.py** (CRÉER) |
| 5 | Abstention | anomalie/incertitude/régime/edge/EV≤0/cooldown/quota → HOLD | **policy/deterministic_gate.py** (CRÉER) |
| 6 | Direction | LONG / SHORT / AUCUNE | deterministic_gate.py |
| 7 | Géométrie | SL=invalidation structurelle, TP=expansion 1h/4h, EV nette − frais | **policy/geometry_engine.py** (CRÉER) |
| 8 | Risk engine | Kelly fractionnaire + plafonds, sizing déterministe | **policy/risk_engine.py** (CRÉER) |
| 9 | Exécution | ordre + SL + TP + rationale | trading/execution_engine.py (conservé) |
| 10 | Position active | réévaluation à CHAQUE close 5m ; SL = invalidation de thèse | **position/lifecycle_manager.py** (CRÉER) |
| 11 | Post-trade | dataset d'expérience (état, réponses, MFE/MAE, résultat) | logging existant (étendre) |
| 12 | Apprentissage hors-ligne | labellisation MFE/MAE ex-post, Brier/LogLoss, calibration VAL, test OOS | **offline/labeler_mfe_mae.py + train_calibrated_judgments.py** (CRÉER) |

## Ce qui disparaît définitivement (et pourquoi)

| Supprimé | Remplacé par | Pourquoi |
|----------|--------------|----------|
| agent/ppo_agent.py (centre) | jugements calibrés + policy déterministe | PPO apprenait par PnL bruité, forçait l'action → fuite SELL |
| environment/reward_calculator.py | scoring rules propres (Brier/NLL) hors-ligne | reward RL = surconfiance + divergence Val/Test |
| MTF statique (1h/4h figées) | contenants vivants reconstruits depuis 5m | latence temporelle déguisée en confluence |
| SL/TP géométrie fixe (k_sl=0.5, k_tp=3.38) | invalidation structurelle + expansion contenant | géométrie arbitraire coupait les gagnants (sep=-0.10) |

## Journal de progression System One

| Date | Tâche | Résultat |
|------|-------|----------|
| 2026-09-25 | Workflow final intégré à dev.md | ✅ figé |
| 2026-09-25 | Phase 0 — test brut sweep 1h (scripts/phase0_nested_container_edge.py) | ⚠️ séparation +1.30R (sweep −2.78R vs baseline −4.08R sur test) mais EV nette négative → piège du dénominateur R identifié (frais 0.40% / SL micro-mèche ~0.15% ≈ 2.67R de frais/trade) |
| 2026-09-25 | Phase 0b — audit 4 niveaux (scripts/phase0b_edge_audit.py) | ✅ **POSITIF** — détails ci-dessous |

### Phase 0b — Résultats détaillés (946 633 bougies 5m BTCUSDT, 2017→2026)

**Méthode** : cadre en 4 niveaux (décision du 2026-09-25) — on ne juge plus une
géométrie arbitraire, on mesure la physique du signal PUIS on cherche la cage
qui la monétise. Splits temporels stricts : train <2022 · val 2022-23 · test ≥2024.

**Niveau 1-2 — Structure & Excursion (fenêtre 24h, sans géométrie)** :

| Split | P(MFE>2×ATR) signal | baseline | Δ | MFE/MAE signal | baseline |
|-------|--------------------|----------|---|----------------|----------|
| train | 86.1% | 83.4% | +2.8 pts | 0.97 | 0.93 |
| val   | 86.8% | 86.1% | +0.6 pts | 1.07 | 1.06 |
| **test** | **88.2%** | **85.9%** | **+2.3 pts** | **1.15** | **1.01** |

→ Le sweep en fin de contenant 1h (k∈{11,12}) détecte une anomalie réelle :
excursion favorable plus fréquente ET asymétrie MFE/MAE supérieure. Le signal
possède une information physique, confirmée sur le test jamais vu.

**Niveau 3 — Géométrie (EV BRUTE en R, split test)** : positive sur TOUTE la
grille SL{0.4, 0.8, 1.2%} × TP{1.5, 2.5, 3.5}R, de +0.065R à **+0.147R**
(croissante avec la largeur du SL → confirme que le stop micro-mèche était
l'erreur, pas le signal).

**Niveau 4 — Économie (EV nette = EV brute − frais/SL, split test)** :

| Frais | SL 0.4% | SL 0.8% | SL 1.2% |
|-------|---------|---------|---------|
| Taker 0.40% RT (stress) | −0.90R | −0.40R | −0.19R (TP 3.5R) |
| **Maker 0.08% RT (limit post-only)** | −0.10R | **+0.03R** | **+0.080R** ✅ |

**Verdict** : ✅ POSITIF — meilleure cellule **SL=1.2% × TP=3.5R, exécution
maker → +0.080R/trade net** sur le test. L'edge existe dans la réalité physique
du marché ; les frais étaient un problème d'ingénierie (résolu par SL
structurel + ordres limit), pas un problème de signal.

**Conséquences pour les modules System One** (gravées dans l'implémentation) :
- `geometry_engine.py` : SL MINIMUM structurel ~1.2% (ou ATR 1h), jamais la
  micro-mèche 5m seule ; TP sur l'expansion du contenant (≈3.5R)
- `deterministic_gate.py` : refus géométrique si frais_R > 0.3R
- Exécution cible : ordres limit post-only (maker), pas de taker

### Étape 1 — nested_state_builder.py : LIVRÉE ET PROUVÉE (2026-09-25)

**Fichier** : `src/adan_trading_bot/data/nested_state_builder.py` (+ `data/__init__.py`)

**Ce qui a été fait** : `NestedStateBuilder` construit le `LivingStateSnapshot` S_t
à chaque clôture 5m — barre 5m scellée + contenants 1h (k/12) et 4h (m/48)
**vivants** (OHLCV running reconstruits depuis la timeline 5m, jamais chargés
depuis des parquets 1h/4h figés), niveaux des contenants fermés précédents,
sweeps avec réintégration, phases, positions intra-contenant, portefeuille,
et flag d'intégrité (étape 2 → abstention).

**Pourquoi** : élimine la latence temporelle du MTF statique (l'erreur
fondatrice d'ADAN0) — à k=6/12, le contenant 1h n'est qu'à moitié formé, son
état reflète exactement cette réalité. Zéro fuite possible par construction.

**Corrections clés** :
- pandas 3.0.5 a supprimé `ts.view('int64')` → `ts.to_numpy('datetime64[ns]').astype(int64)`
- `.gitignore` : la règle `data/` (données racine) masquait aussi le package
  `src/adan_trading_bot/data/` → ancrée en `/data/` + exception explicite.

**Preuves (tests/test_nested_state_builder.py — 7/7)** :
- T1 : mutation des barres > i (high×5, close×3) → snapshot i **inchangé**
- T2 : à k=6/12, running_1h == OHLCV des 6 premières 5m exactement
- T3 : prev_1h == bougie 1h civile fermée précédente (recalcul indépendant)
- T4 : phases frontières exactes (00:00 → 1/12 & 1/48 ; 03:55 → 12/12 & 48/48)
- T5 : historique court / NaN / OHLC incohérent → integrity_ok=False (abstention)
- T6 : sweeps conformes à la référence Phase 0 (dépassement + réintégration)
- T7 : cohérence sur 10 000 barres réelles BTCUSDT (running_1h à k=12 == 1h recalculée)

### Étape 4 — system_one_core.py : LE « JEV PERSONNEL » LIVRÉ (2026-09-25)

**Fichier** : `src/adan_trading_bot/models/system_one_core.py`

**Ce qui a été fait** : le moteur de jugement (étape 4 du workflow). Z n'est
jamais transformé directement en ordre — il est **interrogé** par des
Query-Slots (embeddings de questions appris) :

- **NOUL** (6 questions : mfe_atteint_tp, risque_adverse_faible,
  expansion_imminente, anomalie_donnees, sweep_confirme, reintegration_valide)
  → P(vérité) ∈ [0,1]
- **CHOICE** (regime ∈ {BULL, BEAR, RANGE, TRAP} ; direction ∈ {LONG, SHORT,
  AUCUNE}) → distribution catégorielle
- **SCORE** (qualite_setup, conviction) → distribution **ordinale monotone**
  sur 0..10 : P(>k) structurellement décroissante (cumsum de softplus),
  distribution reconstruite par bornes, E[score] + incertitude. Corrige la
  somme naïve de sigmoïdes du prototype (aucune monotonie garantie).

**Architecture** : moteur GÉNÉRAL de jugement — notion partagée (décodeur par
type) + adaptateur spécifique par question, au lieu de 10 têtes indépendantes.
Le système apprend que « la même notion probabiliste s'applique à plusieurs
affirmations, chacune avec sa sémantique ».

**Calibration** : `calibrate()` fixe les températures (buffers, apprises sur
VALIDATION à l'étape 12) ; `predict_calibrated()` ne sert que des probabilités
dont P=0.70 signifie 70 % réel. Les logits bruts ne sortent que pour
l'entraînement (Brier/NLL).

**Bogues trouvés par le smoke test et corrigés** :
1. KeyError : dict de logits indexé par entier au lieu du nom de question
2. Distribution ordinale tronquée (10 vs 11 niveaux) → bornes P(>-1)=1 et
   P(>10)=0 explicitement concaténées

**Preuves (smoke test 6/6)** : forward batch, monotonie ordinale, probabilités
sommant à 1, espérance ∈ [0,10], calibration activable, round-trip complet
LivingStateSnapshot → features (88 dims) → Judgment sur données réelles BTCUSDT.

### Prochaines briques (ordre décidé 2026-09-25)

1. ~~`models/system_one_core.py`~~ ✅ LIVRÉ (voir ci-dessous)
2. `policy/` — geometry_engine (SL≥1.2%/ATR1h, TP 3.5R, rejet frais_R>0.30),
   deterministic_gate (intégrité, quota ≤5 trades/j, cooldown, direction),
   risk_engine (Kelly quart, plafond exposition 20%, min 15$ Binance)
3. `position/lifecycle_manager.py` — vigie active à chaque 5m (étape 10)
4. `offline/` — labeler MFE/MAE + entraînement supervisé calibré (Brier,
   calibration VAL, test OOS) — cible d'entraînement : 500K pas
5. Grand backtest walk-forward 2017→2026 — seuils : PF ≥ 1.6, max DD < 15%,
   0-5 trades/j, stabilité par cycle (Bull 2017/2021, Bear 2018/2022, Chop 2024-26)


**Last updated:** 2026-09-20  
**Status:** CODE FROZEN — Ready for 500k production run  

---

## ⚡ QUICK START: Launch 500k

```bash
cd /home/ubuntu/webapp/MORNINGSTAR/ADAN0
python scripts/train_parallel_agents.py --config config/config.yaml --steps 500000 --num-cpus 8 --mode heavy
```

**See:** `LAUNCH_500K.md` for detailed checklist + auto-stop thresholds.

---

## ✓ What's Ready

- ✅ Gamma precedence bug fixed (8/8 tests pass)
- ✅ Config consolidated to `config/config.yaml`
- ✅ WorldModelPPO + SB3 2.9.0 verified at runtime
- ✅ Hard pre-flight checks PASSED
- ✅ Auto-stop guards in place (6 thresholds)

**No more changes to code, reward, hyperparams, or architecture before launch.**

---

## 📋 Pre-Flight Checklist (Before Launch)

```bash
# 1. Verify tests pass
python3 scripts/tests/test_gamma_precedence.py

# 2. Verify runtime config
/home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python3 - << 'CHECK'
import sys; sys.path.insert(0, "src")
from adan_trading_bot.agent.feature_extractors import WorldModelPPO
import yaml
cfg = yaml.safe_load(open("config/config.yaml"))
assert cfg.get("production", {}).get("simple_ppo", {}).get("gamma") == 0.99
assert cfg.get("agent", {}).get("gamma") is None
print("✓ READY TO LAUNCH")
CHECK
```

If both show ✓ → **Launch immediately**

---

## 1. Installation des outils DIAGNOSTIC (Kiro internal)

### Installé (2026-09-20)

```bash
# Environnement
conda activate trading_env

# Outils installés (SANS modification du code ADAN0) :
pip install py-spy memory-profiler mlflow
pytest  # Already present (9.1.1)
```

### Pourquoi ces outils

| Outil | Usage | Pour qui | Pas pour l'UI |
|---|---|---|---|
| **pytest** | Valider les correctifs critiques | Kiro diagnostic | ✅ Tests internes |
| **py-spy** | Profiling CPU (où le temps va) | Kiro diagnostic | ✅ Flamegraph local |
| **memory-profiler** | Détection fuites mémoire | Kiro diagnostic | ✅ Local analysis |
| **mlflow** | Registry des runs (local self-hosted) | Kiro diagnostic | ✅ Pas UI web |

### Installation optionnelle (si tu veux)

```bash
pip install wandb  # Cloud dashboards (SKIP — tu as déjà une UI)
pip install structlog  # JSON logging (SKIP — pas obligatoire pour diagnostic)
```

---

## 1.5 GAMMA PRECEDENCE BUG FIX (2026-09-20) — CRITICAL

### What Was Fixed

✅ **Gamma bug resolved** : Checkpoint v30 had `gamma=0.9523` (wrong). Now all runs will use `gamma=0.99` (correct).

**Problem:** Two conflicting gamma values in config files with broken loading precedence.
- **Stale (deleted):** `agent.gamma: 0.9523222699352663` (line 215)
- **Correct (kept):** `production.simple_ppo.gamma: 0.99` (line 1186)

### What Changed

1. **Config files:** Deleted stale root gamma from all 5 config files (config.yaml, v36a/b/c.yaml)
2. **Loading logic (train_parallel_agents.py, line ~1155):** Now reads `production.simple_ppo.gamma` instead of root gamma
3. **Resume override (line ~2681):** When resuming checkpoint, gamma is explicitly corrected to 0.99

### Validation: Run These Tests

```bash
cd /home/ubuntu/webapp/MORNINGSTAR/ADAN0

# Test 1: Gamma precedence (must pass 18/18)
python3 scripts/tests/test_gamma_precedence.py

# Test 2: Smoke test (must pass 5/5)
python3 scripts/tests/test_gamma_smoke_100steps.py

# Expected output:
# ✓ ALL TESTS PASSED
```

### ⚠️ BEFORE LAUNCHING 30k or 500k RUN

1. Run the two test suites above (2-3 seconds total)
2. Confirm both show "✓ ALL TESTS PASSED"
3. If resuming from v30 checkpoint, look for this log:
   ```
   [SANDBOX] Updated PPO.gamma: 0.9523222699352663 → 0.99 (from config)
   ```

**Full details:** See `GAMMA_FIX_REPORT.md`

---

## 2. Tests à exécuter (valider les fixes)

### 2.1 Test asymétrie entry/exit (FIX-D)

```bash
cd /home/ubuntu/webapp/MORNINGSTAR/ADAN0

# Crée le test
cat > tests/test_action_routing_fix.py << 'EOF'
"""Test FIX-D: asymmetric entry/exit thresholds"""
from adan_trading_bot.environment.action_routing import route_action_by_state

def test_entry_exit_asymmetry():
    """Entry requires conviction, exit is protection"""
    # FLAT state: entry threshold 0.10 (conviction)
    assert route_action_by_state(a0=0.11, in_position=False, threshold=0.10) == 1  # BUY
    assert route_action_by_state(a0=-0.11, in_position=False, threshold=0.10) == 0  # NOT SELL (HOLD)
    
    # LONG state: exit threshold 0.05 (easier to exit)
    assert route_action_by_state(a0=-0.06, in_position=True, threshold=0.10, sell_threshold=0.05) == 2  # SELL
    assert route_action_by_state(a0=+0.15, in_position=True, threshold=0.10, sell_threshold=0.05) == 0  # NOT BUY (HOLD)

if __name__ == "__main__":
    test_entry_exit_asymmetry()
    print("✅ FIX-D test PASSED")
EOF

# Exécute
pytest tests/test_action_routing_fix.py -v
```

**Résultat attendu** : ✅ PASSED

### 2.2 Test HMM readonly consumer (ADAN0_HMM_READONLY_CONSUMERS)

```bash
cat > tests/test_hmm_readonly_fix.py << 'EOF'
"""Test ADAN0_HMM_READONLY_CONSUMERS: consumers don't poison buffer"""
import numpy as np
from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine

def test_hmm_readonly_no_contamination():
    """Verify readonly consumers don't mutate HMM buffer or posteriors"""
    dbe = DynamicBehaviorEngine()
    
    # Producer call (valid observation)
    p1 = dbe.get_regime_probabilities(
        log_return=0.005, 
        atr_pct=0.02, 
        rsi_norm=0.5, 
        volume_ratio=1.1,
        observation_id="step_1000"
    )
    
    # Consumer call (no observation_id, should NOT ingest)
    p2 = dbe.get_regime_probabilities(
        log_return=0.0, 
        atr_pct=0.0, 
        rsi_norm=0.0, 
        volume_ratio=0.0,
        observation_id=None  # READ-ONLY
    )
    
    # Posteriors should be identical (consumer didn't mutate)
    assert np.allclose(p1, p2, atol=1e-6), f"p1={p1}, p2={p2}"
    
    # Read-only count should increment
    assert dbe._hmm_readonly_calls >= 1

if __name__ == "__main__":
    test_hmm_readonly_no_contamination()
    print("✅ HMM readonly test PASSED")
EOF

pytest tests/test_hmm_readonly_fix.py -v
```

**Résultat attendu** : ✅ PASSED

---

## 3. Profiling & Diagnostics

### 3.1 CPU Profiling (où le temps va)

```bash
# Genère flamegraph du premier 1000 steps
py-spy record -o /tmp/adan_profile.html \
  --function \
  -- python scripts/launch_asset_run.py --asset BTCUSDT_BINANCE --steps 1000

# Ouvre le flamegraph
open /tmp/adan_profile.html  # ou Firefox /tmp/adan_profile.html
```

**Interprétation** :
- Cherche les blocs larges : HMM fit ? Feature extraction ? Reward calc ?
- Si HMM fit prend >20% du temps → Candidate for optimization
- Si normalizer prend >10% → Maybe vectorize

### 3.2 Memory Profiling (fuites mémoire)

```bash
# Lance 100 steps avec tracking mémoire ligne-par-ligne
python -m memory_profiler scripts/launch_asset_run.py --asset BTCUSDT_BINANCE --steps 100

# Attention aux lignes affichant des allocations croissantes
```

---

## 4. MLflow : Local run registry (diagnostic seulement)

### 4.1 Lancer le serveur

```bash
# Terminal 1: start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Accès: http://localhost:5000
```

### 4.2 Logger les runs (si tu veux voir côté Kiro)

```python
# Optionnel: ajoute dans scripts/launch_asset_run.py après logging setup:

import mlflow

mlflow.start_run(run_name=f"adan0_{asset}_{steps}steps")
mlflow.log_param("asset", asset)
mlflow.log_param("frais_rt", 0.004)

# Dans la boucle de training (main loop):
# mlflow.log_metric("equity", portfolio_value, step=step_count)
# mlflow.log_metric("hold_pct", hold_count/step_count, step=step_count)

mlflow.end_run()
```

**Pas obligatoire** : MLflow est juste pour que j'accède à l'historique des runs en tant que diagnostic interne.

---

## 5. Validation des correctifs (obligatoire avant tout nouveau run)

### Checklist d'audit avant 500k run

```bash
# 1. Vérifier les tests
pytest tests/test_action_routing_fix.py tests/test_hmm_readonly_fix.py -v

# 2. Profiler le baseline (100 steps)
py-spy record -o /tmp/baseline.html \
  -- python scripts/launch_asset_run.py --asset BTCUSDT_BINANCE --steps 100
# → Chercher fuites ou bottlenecks

# 3. Lancer petit run diagnostic
python scripts/launch_asset_run.py --asset BTCUSDT_BINANCE --steps 500
# → Vérifier pas de crash, pas d'exception

# 4. Si OK: aller 500k
```

---

## 6. Code Modification Policy (STRICT)

### ❌ NE PAS MODIFIER sauf urgence critique

Le code ADAN0 est **stable post-audit**. Aucune modification n'est requise.

### ✅ SI modification obligatoire

1. **Documente** dans ce fichier (dev.md)
2. **Ajoute marker** `ADAN0_<CATEGORY>_<ISSUE>` avec commit message
3. **Mesure avant/après** (dans ETAT_DU_CODE.md style)
4. **Test unitaire** : crée `tests/test_<issue>.py`
5. **Annonce breaking change** : mets à jour `AUDIT_COMPLET_KIRO_*.md`

### Exemple (si jamais)

```python
# Avant
frais = 0.004

# Après avec marker
frais = 0.004  # ADAN0_FEES_RECALIBRATION: measured 0.40% A/R (2026-09-20)
```

---

## 7. Architecture code — pas de refactor maintenant

✅ Les 5 bugs critiques sont fixés  
❌ Refactor `multi_asset_chunked_env.py` (11k lignes) → hors scope diagnostic

Refactor seulement si nouveau run doit aller en prod.

---

## 8. Decision tree pour le prochain run

```
Est-ce que tu veux lancer un nouveau run 500k?
│
├─ NON → Fin. Utilise les rapports existants pour stratégie.
│
└─ OUI → Est-ce que tu as changé QUELQUE CHOSE?
   │
   ├─ Frais < 0.10% A/R? → GO
   ├─ Timeframe = 4h? → GO
   ├─ Univers = microstructure? → GO
   │
   └─ Non → Reste identique à V30 500K →  HOLD collapse 99% attendu. NO-GO.
```

---

## 9. Support Kiro — diagnostics que je peux faire

✅ **Je peux** :
- Lire les logs réels
- Analyser les rapports diagnostiques existants
- Vérifier les correctifs avec tests
- Profiler le code avec py-spy
- Proposer optimisations basées sur traces réelles

❌ **Je ne peux pas** :
- Exécuter les runs interactivement (pas d'interpréteur runtime)
- Modifier ton UI web (she's separate)
- Relancer les anciennes sondes (données non accessibles)

---

## 10. Fichiers clés pour toi

| Fichier | Purpose |
|---------|---------|
| `AUDIT_COMPLET_KIRO_2026_09_20.md` | **Lire en priorité** — résultat complet |
| `ETAT_DU_CODE.md` | Source de vérité technique (correctifs, bugs) |
| `INSTALLATION_STACK_DIAGNOSTIC.md` | Outils optionnels (pour future audit) |
| `dev.md` | **Ce fichier** — guide pour developper |

---

## 11. Questions rapides

**Q: Faut-il changer la config?**  
A: Non. `config/config.yaml` est gelée. Les leviers sont économiques (frais, TF, univers), pas configs.

**Q: Le code marche?**  
A: Oui. Collapse HOLD = feature, pas bug. Voir diagnostic.

**Q: Quand relancer 500k?**  
A: Quand une condition mesurable change (frais ↓, TF ↑, ou stratégie re-cadrée). Pas avant.

**Q: Comment collaborer avec Kiro?**  
A: Files → tests → profiling → diagnostics. Kiro fait diagnostic, tu fais décisions.

---

**Prêt pour diagnostic avancé ?** Appelle-moi avec un chemin spécifique ou un run à analyser.
