# Diagnostic — Collapse "SELL-sans-position" + Reward arithmétique vs difficulté géométrique

Date : 2026-06-27 · Run analysé : `fa_500k_v4` (gelé à step 12417, tué).

## 0. État du run (FACTUEL)
- Log figé depuis **7 h** au step **12417/500000** ; `fps=6` (vs centaines attendues) ;
  `iterations=80`, `time_elapsed=6395 s`. → **deadlock / run inutilisable**, tué.
- Portfolio figé à 20.50 ; `last_step=-` (aucun trade dans la fenêtre récente).
- **OHLC_INCOHER=0** sur toute la durée → le fix cross-TF (commit 8cf8ac6) **tient**.
  Le problème actuel n'est PAS le bug cross-TF, c'est un collapse de la politique.

## 1. Compteurs décisifs (Requested vs Executed)
> ⚠️ Le raw log a été tronqué par la rotation du watcher (erreur de méthode, cf. §4).
> Les compteurs ci-dessous portent sur la fenêtre survivante + l'audit append-only,
> mais le signal est sans ambiguïté car `action[0]=-1` est constant.

| Transition | Compte | Lecture |
|---|---|---|
| SELL → HOLD | 100 % des décisions | le modèle demande SELL en permanence |
| BUY → HOLD | 0 | le modèle ne tente JAMAIS de BUY |
| BUY → BUY | 0 | aucune ouverture |
| SELL → SELL | 0 | aucune fermeture (rien à fermer) |
| TRADE_AUDIT_OPEN / CLOSE | 0 (fenêtre récente) | inaction totale |

`action[0] = -1.0` constant ; `ACTION_DIST tp_raw=-0.400 sl_raw=+0.287` → politique
**déterministe collapsée** sur un coin de l'espace d'action.

## 2. Cause primaire PROUVÉE par le code — SELL-sans-position = point fixe GRATUIT
`multi_asset_chunked_env.py`, branches de décision :
- L7759 : `if discrete_action == 2 and is_open:`  → SELL **avec** position → ferme.
- L7943 : `elif discrete_action == 1 and not is_open:` → BUY **sans** position → ouvre.
- **MANQUANT** : aucune branche `discrete_action == 2 and not is_open`.

Conséquence : quand le modèle émet SELL alors qu'aucune position n'est ouverte,
**aucune** branche ne s'exécute, **aucune** pénalité n'est ajoutée
(`inv_penalty=0.00000`). HOLD perpétuel **gratuit** → équilibre stable sans gradient
de sortie → PPO converge vers `action[0]=-1`.

C'est la définition même d'un reward-hack : le modèle a trouvé l'action à coût nul.

## 3. Hypothèses écartées / nuancées
- **Gates bloquent les BUY** : ÉCARTÉ. Les gates (RISK_GATE, EV_GATE, cooldown…
  L7960-8074) sont dans la branche BUY, jamais atteinte car le modèle ne demande
  jamais BUY. Ils ne sont donc pas la cause.
- **Exploration morte** : CONFIRMÉ comme *conséquence*, pas cause racine. La politique
  s'est figée APRÈS avoir trouvé l'équilibre gratuit. Ne pas toucher `ent_coef` /
  `log_std` seuls : on corrige d'abord la cause (le coût nul), puis on réévalue.

## 4. Reward ARITHMÉTIQUE vs difficulté GÉOMÉTRIQUE (point central utilisateur)
`config.yaml capital_tier_rewards` (FACTUEL) :

| Palier | stagnation_penalty / step | drawdown_factor |
|---|---|---|
| Micro  | -0.00025          | 2.0 |
| Small  | -0.000125 (÷2)    | 1.5 |
| Medium | -0.0000625 (÷2)   | 1.0 |
| High   | -0.000025 (÷2)    | 0.5 |

- La pénalité de stagnation **décroît géométriquement (÷2 / palier)** → plus on monte,
  plus l'inaction est *gratuite*. C'est l'INVERSE de ce qu'il faut.
- `invalid_trade_penalty_weight: 0.005` est **plat (arithmétique)**, indépendant du palier.
- Or la difficulté de progression est **géométrique** : passer Micro(11→30) puis
  Small(30→100) puis Medium(100→300) exige un PnL relatif croissant. Récompense plate
  + difficulté géométrique ⇒ il devient rationnel de NE RIEN FAIRE aux paliers hauts.

→ La structure de reward **encourage la lâcheté de plus en plus** à mesure qu'on progresse.

## 5. Plan de correction (UNE cause à la fois, prouvée avant/après)

### FIX A (cause primaire) — coût de l'inaction stérile
Ajouter la branche manquante `discrete_action == 2 and not is_open` : SELL sans
position = action invalide → pénalité via `_step_invalid_penalty`, reliée au pipeline
principal (L8203 `realized_pnl += self._step_invalid_penalty`). Magnitude : à calibrer
de façon **géométrique par palier** (cf. FIX B), PAS un ×5 arithmétique arbitraire.

### FIX B (point géométrique) — pénalité géométrique CROISSANTE par palier
Remplacer la décroissance ÷2 par une **croissance** géométrique de la friction
d'inaction : `pen_tier = base * r^k` (k = index palier, r > 1), avec **cap** pour
rester borné (jamais exponentiel non maîtrisé — cohérent avec la directive
"log, pas exp, seuil/cap"). Le but : neutraliser le gain géométrique de la lâcheté.

### Vérif (avant relance 500k)
Test 300 steps : (1) `Requested=SELL Executed=HOLD` ne doit plus être gratuit
(`inv_penalty<0` quand SELL-sans-pos), (2) `action[0]` ne doit plus rester collé à -1,
(3) au moins quelques BUY tentés, (4) OHLC_INCOHER=0 maintenu, (5) pas de deadlock
(fps raisonnable, steps qui avancent).

## 6. Erreurs de méthode reconnues
- **Rotation du log** = mauvaise idée : elle a détruit la preuve historique. À la place,
  réduire la VERBOSITÉ du logger (ne pas logger chaque step en INFO), pas tronquer.
- **Ne pas réactiver une pénalité au hasard** : on ne touche qu'à la cause prouvée,
  une variable à la fois, avec mesure avant/après.
