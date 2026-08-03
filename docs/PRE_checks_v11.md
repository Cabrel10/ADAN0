# PRE-checks V11 — vérifications gratuites AVANT de toucher au reward

> Discipline demandée par l'utilisateur : **preuve avant conclusion**, ordre gratuit→coûteux.
> Ces deux vérifications ont été faites sur les logs/données EXISTANTS, sans relancer de run.

---

## PRE-0 — Signe de `anti_spam_hold` (2824 occ ?) — **RÉSOLU / corrige une erreur de session précédente**

**Hypothèse à écarter (soulevée par l'utilisateur) :** `anti_spam_hold` aurait tiré 2824×
(5× plus que `CASH_FLOOR_B`) et pourrait cacher un problème de signe/magnitude non audité.

**Méthode :** grep sur `logs/training/train_v10_500k_20260702_004114.log` (59 Mo, run V10).

**Preuves :**
```
valeur MAX de 'anti_spam_hold' dans les rejections  = 0   (constante, tout le run)
valeur finale                                        = 0
sterile_pen  : count=582  min=-0.05550  max=-0.00050  all_negative=TRUE
contexte des 582 sterile_pen : 582 = "CASH_FLOOR" / "V9 anti-runaway"  (100 %)
aucune valeur sterile_pen positive (grep 'sterile_pen=[0-9]' → vide)
```

**Verdict :**
1. **`anti_spam_hold` n'a JAMAIS tiré dans V10** (compteur = 0 en permanence). Le "2824"
   de la session précédente était une **erreur de lecture** (autre run, ou confusion avec
   un compteur de rejets cumulés `rejections={...}` qui n'applique aucune pénalité reward).
2. La **seule** branche de pénalité stérile active est **CASH_FLOOR_B (582×)**, toutes
   négatives (max −0.055), signe correct, aucun sign-bug résiduel.
3. → **Aucun signal caché plus gros que CASH_FLOOR_B.** La conclusion "fix V9/C1 hors-sujet"
   tient, et pour une raison plus forte : la seule pénalité active est **négligeable**
   (max −0.055) face au flux `latent_pnl_contrib` (cap **+0.10** tous les 3 steps).

---

## PRE-1 — Alignement temporel des timeframes — **RÉSOLU : pas de bug de lookahead, mais BUG DE COUVERTURE DE DONNÉES (P0)**

**Hypothèse (utilisateur) :** l'observation aligne-t-elle les 3 TF par **index** (bug) ou par
**timestamp** (correct) ? Si par index, la cross-attention apprendrait sur des contextes qui
n'ont jamais coexisté → nuancerait la lecture optimiste d'`explained_variance`.

**Méthode :** lecture de `_build_observation` (env L.5616) → `StateBuilder.build_observation`
(state_builder L.1215) → `ChunkedDataLoader.load_chunk` (data_loader L.480) →
`_align_master_clock` (L.645). Puis exécution réelle du loader.

**Chaîne de preuves :**

1. `_build_observation` passe **un seul scalaire** `current_idx = step_in_chunk` pour les 3 TF,
   et `StateBuilder` slice `df.iloc[start:end]` → **slicing positionnel**. *À première vue* c'est
   le bug… MAIS :
2. `load_chunk` appelle `_align_master_clock` (L.554) AVANT de servir les données. Cette fonction :
   - reindexe 1h/4h sur l'**index DatetimeIndex 5m** via `reindex(master_idx, method="ffill")` (L.708) ;
   - applique un **`shift(1)` anti-lookahead** sur les TF hautes (L.689) → à 10h05 l'agent voit
     la clôture 1h de **09h00**, pas 10h00 → filtration causale garantie ;
   - l'index parquet est un **vrai `DatetimeIndex`** (vérifié à l'exécution).
3. **Après alignement, les 3 TF ont exactement la même longueur (18544) et le même index** :
   ```
   LOADED BTCUSDT/5m: 18544 rows [2025-06-29 06:35 .. 2026-05-12 02:55]
   LOADED BTCUSDT/1h: 18544 rows [2025-06-29 06:35 .. 2026-05-12 02:55]
   LOADED BTCUSDT/4h: 18544 rows [2025-06-29 06:35 .. 2026-05-12 02:55]  | ALL EQUAL: True
   ```
   → **Le slicing positionnel devient correct** parce que toutes les séries partagent le
   même index temporel. **PAS de bug d'incohérence temporelle / lookahead.**

**MAIS — le vrai problème, mesuré :** les parquets **bruts** ne se recouvrent que sur **46 jours** :
```
raw 5m : 18544 rows [2025-06-29 .. 2026-05-12]
raw 1h :  5483 rows [2022-07-14 .. 2025-08-15]   ← se termine 9 mois AVANT le 5m
raw 4h :  1685 rows [2022-10-14 .. 2026-02-08]
5m ∩ 1h ∩ 4h (chevauchement réel) = 46 jours = 2800 barres 5m (15 % du 5m)
```
Comme `_align_master_clock` **ffill** au lieu de trim, le canal 1h est **gelé (constant)** sur
`2025-08-15 → 2026-05-12` ≈ **270 jours = 15 744 barres = 85 % de l'entraînement**. Le 4h est
gelé sur ~30 % de fin. Donc :

- ✅ **Pas de bug de lookahead** ni d'incohérence temporelle (l'alignement est correct et causal).
- ⚠️ **BUG DE COUVERTURE (P0) :** sur **85 %** des steps, le canal 1h fourni au CNN est une
  **constante** (dernière clôture réelle ffillée). La cross-attention/FiLM tourne sur des
  entrées 1h dégénérées la majorité du temps.

**Impact sur le verdict "le critic apprend (EV≈0.30)" :**
- Le verdict **tient toujours** — mais l'EV≈0.30 est atteint **essentiellement via le flux 5m**
  (seul canal avec des données fraîches sur tout le run), les canaux 1h/4h étant constants la
  plupart du temps. L'architecture multi-TF est donc **sous-alimentée, pas cassée.**
- Cela **nuance l'optimisme** : on ne peut pas conclure que la fusion multi-TF « fonctionne »,
  seulement que le pipeline 5m suffit à un critic partiellement prédictif.

**Action P0 recommandée (données — NE PAS toucher sans validation) :** régénérer/étendre les
parquets 1h et 4h pour couvrir la même plage que le 5m (jusqu'à 2026-05-12). C'est un problème
**de données**, hors périmètre "reward" — à traiter séparément.

---

## Conséquence sur l'ordre des Phases

Ces deux PRE-checks **ne contredisent pas** la conclusion V10 (le critic apprend, collapse =
mésalignement du reward), mais ajoutent une **cause de confusion data** (couverture 1h/4h) qui
devra être notée dans la matrice de responsabilité (Phase 9) et qui **milite pour instrumenter
avant de patcher le reward** (Phase 2), comme demandé.
