# reward_map_v11.md — Inventaire complet du reward (Phase 1)

> Source de vérité : `src/adan_trading_bot/environment/multi_asset_chunked_env.py`,
> `_calculate_reward` (L.6394) → assemblage `raw_reward` (L.6691) → `final_reward` (L.6707).
> Lecture intégrale du code, aucune inférence. Config : `config/config.yaml §reward_shaping`.

## Formule finale

```python
raw_reward = (
    pnl_base_reward        # (1)
  + promotion_bonus        # (2)
  + demotion_penalty       # (3)
  + closure_bonus          # (4)
  + drawdown_penalty       # (5)
  + symmetry_penalty       # (6)
  + action_entropy_penalty # (7)
  + future_contrib         # (8)
  + latent_pnl_contrib     # (9)
  + saturation_penalty     # (10)
)                          # survival_bonus (A6) & patience_bonus (A4) = 0.0 (RETIRÉS)
final_reward = sign(raw_reward) * log1p(|raw_reward|)   # symlog
```
Note : `sterile_pen` / CASH_FLOOR_B (§V9) est ajouté séparément via `self._step_invalid_penalty`,
en dehors de `raw_reward` (branche de gate d'action), négatif, max −0.055 (582× en V10).

## Tableau des composantes

| # | Composante | Signe | Bornes | Fréquence (déclencheur) | Dépend de l'état | Ligne |
|---|---|---|---|---|---|---|
| 1 | `pnl_base_reward` = `pnl_pct*0.5` | ± | ∝ PnL réalisé | **uniquement à la clôture** d'un trade (realized_pnl≠0) | **FLAT=0, HOLD=0**, seulement au CLOSE | 6556 |
| 2 | `promotion_bonus` | + | +0.5 → +4.0 (doublant/tier) | rare (changement de tier capital) | tier ↑ | 6478 |
| 3 | `demotion_penalty` | − | −0.5 → −4.0 | rare (tier ↓) | tier ↓ | 6482 |
| 4 | `closure_bonus` | ± | +0.5 (agent-close gagnant) / −0.2 (MaxDuration) | à la clôture (receipt) | CLOSE seulement | 6490 |
| 5 | `drawdown_penalty` = `-50*dd²*factor` | − | 0 → ~−2 (à −20% DD) | par step si `drawdown < −1%` | equity en baisse (flat OU position) | 6521 |
| 6 | `symmetry_penalty` | − | 0 → −0.15/position (cap) | **par step tant qu'une position est ouverte** | **POSITION uniquement** | 6602 |
| 7 | `action_entropy_penalty` | − | 0 → −0.03*(rate−0.5) | par step si switch-rate>0.5 | historique d'actions (switch spam) | 6617 |
| 8 | `future_contrib` | ± | plafonné (<40% du PnL, watchdog) | **rare** (à la clôture, MFE/MAE ex-post) | CLOSE seulement (52× en 70k) | 6628 |
| 9 | **`latent_pnl_contrib`** | **± (gain + / loss −)** | +0.10 gain / −0.15 loss (cap 0.30) | **toutes les 3 steps tant que position ouverte** | **POSITION uniquement** | 6639–6667 |
| 10 | `saturation_penalty` | − | 0 → −0.20 (cap) | par step si SL/TP saturent (>50% fenêtre) | SL/TP spam (position) | 6687 |
| — | `survival_bonus` (A6) | 0 | 0 | **RETIRÉ** (récompensait l'inaction) | — | 6557 |
| — | `patience_bonus` (A4) | 0 | 0 | **RETIRÉ** (récompensait l'attente) | — | 6544 |
| — | `sterile_pen`/CASH_FLOOR_B (V9) | − | max −0.055 | gate d'action (cash deficit self-caused) | action illégale | 7735+ |

## Détail des formules sensibles

**(9) latent_pnl_contrib** (le suspect principal) :
```python
_u = (current_price - entry_price) / entry_price   # PnL latent fractionnaire (SPOT long)
if _u >= 0:  latent += min(0.30, 0.10 * log1p(_u*10)/10)   # GAIN → reward POSITIF
else:        latent -= min(0.30, 0.15 * log1p(|_u|*10)/10)  # LOSS → penalty
# appliqué UNIQUEMENT si held>0 et held % 3 == 0, POUR CHAQUE position ouverte
```
- fuel/step réaliste (+1% latent) = `0.10*log1p(0.1)/10 ≈ 0.000953` toutes les 3 steps.
- **asymétrique** : poids perte (0.15) > poids gain (0.10) — MAIS la moyenne des bougies 5m
  étant positive (50.4% up), l'espérance par step en position est **> 0**.

**(1) pnl_base_reward** :
```python
pnl_pct = realized_pnl * 100 / max(initial_capital, 1)   # % sur capital
pnl_base_reward = pnl_pct * 0.5
```
Zéro sauf à la clôture. C'est le seul « vrai » signal économique, mais il est **épisodique**.

## 🔴 Constat structurel (fondation de l'hypothèse mésalignement)

**Décomposition du reward par état, tirée du code (pas inférée) :**

| État | Composantes pouvant être non-nulles | Signe atteignable |
|---|---|---|
| **FLAT** (cash, aucune position) | `drawdown_penalty` (≈0 si cash préservé), `action_entropy_penalty` (≤0) | **≤ 0 seulement** — AUCUN chemin vers un reward positif |
| **EN POSITION (hold)** | `latent_pnl_contrib` (+/−, **espérance >0** sur bougies up), `symmetry_penalty` (≤0), `saturation_penalty` (≤0), `drawdown_penalty` (≤0) | **peut être > 0** via latent_pnl |
| **CLOSE** | `pnl_base_reward` (±), `closure_bonus` (±), `future_contrib` (±), `promotion/demotion` | ± (épisodique) |

**Conséquence :** un agent flat ne peut, au mieux, qu'obtenir **reward = 0** (et souvent < 0 via
drawdown/entropy) ; un agent en position touche un **flux positif** `latent_pnl_contrib` toutes les
3 steps sur les bougies montantes. Le gradient de politique pousse donc mécaniquement vers
**"toujours en position / toujours BUY"** — un optimum réel mais indésirable. `survival_bonus` et
`patience_bonus`, qui donnaient jadis un signal à l'état flat, ont été **retirés** (L.6544, 6557).

**Ce constat reste une HYPOTHÈSE FORTE (fondée sur le code) — pas encore une preuve chiffrée.**
La Phase 2 (instrumentation `reward_components.csv`) doit **mesurer** la contribution réelle de
chaque terme par état pour confirmer que `latent_pnl_contrib` est bien le moteur dominant, avant
tout patch du reward (discipline C1 : calcul noir-sur-blanc avant action).
