# AUDIT D'ARCHITECTURE v14 — "Le modèle comme adversaire" (2026-07-05)

> Objectif : arrêter de patcher des paramètres (holding_cost) et faire un audit de
> CONCEPTION. Répondre à : **quels modules participent réellement à l'apprentissage ?**
> Décision utilisateur : "c'est la conception qui est le défi… on patch chacune des
> fuites et on arrive où il ne trouve plus de fuite évidente."

## 1. Tableau Module × Rôle réel

| Module | Existe | Utilisé PPO | Influence REWARD | Influence ACTION | Dans OBS | Verdict |
|---|---|---|---|---|---|---|
| PnL réalisé (`pnl_base_reward ×0.5`) | oui | oui | **fort** | — | via [2] | Sain — MAIS ne se déclenche qu'à la fermeture |
| Tuteur (`RewardCalculator`) | oui (instancié) | **NON — jamais appelé** | non | non | non | **MORT** — tourne dans le vide |
| Future Arena (`future_contrib`) | oui | oui | oui (plafonné) | dims 1-4 | via [14-16] | Actif ; ne récompense que trades FERMÉS → n'aggrave pas SELL directement |
| decision_budget (énergie/"mana") | oui | partiel | pénalité soft si bloqué | **bloque CLOSE** | **NON (invisible)** | **DÉSYNCHRONISÉ** — POMDP |
| Tiers de capital (promotion/demotion) | oui | oui | gros bonus | — | via [0-1] | **JAMAIS ATTEINT** (collapse < tier 2 = 30$) |
| holding_cost | oui | opt-in | oui | — | non | Retarde le collapse, ne le corrige pas (PROUVÉ) |
| smart_flat | oui | opt-in | oui | — | non | Non mesuré isolément |
| time_decay | oui | opt-in OFF | oui | — | non | Mauvais levier (PROUVÉ 3.3× pire) |
| Pénalité quadratique (drawdown `-50×dd²`) | oui | oui | oui | — | via [4] | Active |
| symmetry_penalty (RR/ATR) | oui | oui | latent (λ=0.02) | — | non | Faible |
| latent_pnl | oui | oui | 0-0.6% | — | non | Négligeable (PROUVÉ) |
| CLOSE barrier (1.5× frais) | oui | oui | pénalité | **bloque CLOSE** | non | **Sur-restrictif** |
| Fonctions polaires | **non** | — | — | — | — | N'existent pas (`_last_trade_step` vestigial) |
| cash_floor / mana / fatigue | **non** | — | — | — | — | N'existent plus dans portfolio_manager |

## 2. Diagnostic racine — 3 maladies de CONCEPTION

**Maladie 1 — Pas de concept de "trade".** Le reward est per-step. Le seul signal
économique fort (`pnl_base_reward`) n'arrive qu'à la fermeture d'une position.

**Maladie 2 — La SORTIE est structurellement sabotée (LA fuite inverse).** CLOSE est
bloqué par TROIS gardes simultanées : `decision_budget < 0.30`, `gap < 12 steps`,
`pnl < 1.5×frais`. Défauts : cost_close=0.30, recharge=0.02 → après ~3 CLOSE il faut
**15 HOLD** pour refermer, + gap 12 + barrière. L'obs annonce `can_close=1` mais l'env
refuse silencieusement → PPO apprend **"SELL ne marche jamais"** → collapse BUY.
On a tellement puni le sur-trading qu'on a rendu la fermeture impossible.

**Maladie 3 — Les modules pilotes sont invisibles ou morts.** Le tuteur ne parle
jamais à la policy (jamais appelé). L'énergie (`decision_budget`) n'est PAS dans
l'obs → contrainte cachée → **POMDP** → l'agent ne peut pas apprendre à la gérer
(cf. ACNO-MDP / hidden-state POMDP, littérature RL). Les tiers ne sont jamais atteints.

## 3. Corrections appliquées (design, une famille de variables à la fois)

- **FIX A (POMDP) — énergie OBSERVABLE.** L'env pousse `_close_energy_ready` =
  `budget_ready × gap_ready` (∈[0,1]) vers le portfolio ; le slot [21] `can_close`
  devient `has_position × close_readiness` (au lieu d'un booléen). La contrainte
  cachée devient observable → POMDP → MDP. Activable via `ADAN_ENERGY_OBS` (défaut ON).
  Ne change PAS le schéma 28-dims (enrichit la sémantique d'un slot, comme [9]).

- **FIX C — débloquer la SORTIE.** Env vars pour desserrer les 3 gardes :
  `ADAN_CLOSE_MIN_GAP` (12→6), `ADAN_CLOSE_COST` (0.30→0.20),
  `ADAN_CLOSE_RECHARGE` (0.02→0.04), `ADAN_CLOSE_MAX_PER_DAY` (7→12). Défauts
  conservent le comportement legacy (rien ne change si non défini).

- **holding_cost = 0.0** au run archfix : prouvé inefficace (ne retarde que le
  collapse : hc=0.012 collapse @110k, hc=0.016 collapse @28k).

### Tuteur — décision assumée
Le tuteur (RewardCalculator) reste DÉCONNECTÉ pour ce run. Le brancher comme 2ᵉ
fonction de reward ajouterait un objectif contradictoire de plus (la maladie
"multi-objectif non maîtrisé"). La discipline "une variable à la fois" impose de
d'abord corriger les 2 causes prouvées (POMDP énergie + sortie étranglée) et de
mesurer, avant d'introduire une imitation-loss du tuteur.

## 4. Contraintes respectées
FRAIS 0.5% INTACTS (commission 0.0025, round_trip 0.005). Dims 1-4 (Future Arena)
INTACTES. Pas de VecNormalize. Pas de MaskablePPO/sb3-contrib. obs_schema 28 dims
INCHANGÉ. std_init défaut -2.0.

## 5. Run de validation
`scripts/launch_archfix_500k.sh` — 500k steps, 1 worker intraday, FIX A + FIX C,
breaker OFF (capture crash complet), diag every 2000. Questions à répondre au run :
passe-t-il 70k ? pct_sell remonte-t-il (l'agent réapprend-il à vendre) ? apparition
de PHASES (BUY → HOLD → SELL intelligents) = vrai apprentissage vs nouvel exploit ?
