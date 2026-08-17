# RAPPORT FORENSIQUE — Collapse V31-DIAG-50k
**Date** : 2026-08-17 ~20:05 UTC
**Run** : V31-DIAG 50k (lancé 17:53 UTC, arrêté à step 5609 après 4h04)
**Branche décision table utilisateur** : ❌ « Dérive a0, asymétrie, KL explosif → Arrêter, diagnostiquer, corriger »

---

## 1. VERDICT : collapse PPO confirmé sur les DEUX workers

| Signal | worker_0 (scalper) | worker_1 (intraday) | Seuil alarme | Statut |
|---|---|---|---|---|
| approx_kl (dernier update) | **0.576** | **0.959** | > 0.5 persistant | ❌ les deux |
| entropy_loss | −9.25 | **−10.1** | collapse | ❌ |
| explained_variance | 0.074 | **−0.543** | < 0 = value morte | ❌ w1 |
| std (policy) | 0.136 | 0.136 | collapsé (< 0.2 init) | ❌ |
| a0_mean (premier 1k → dernier 500) | +0.000 → **−0.223** | −0.071 → **−0.359** | dérive | ❌ |
| lock \|a0\|≥0.999 (derniers 500) | 56.6% | 63.4% | lock boundary | ❌ |
| Buffer PPO w1 | nS=6900 / nB=2540 / nH=560 | | asymétrie SELL | ❌ |
| Portfolio | 20.50 → **14.77** (−28%) | 20.50 → **16.20** (−21%) | | ❌ |
| Win rate | 16.5% | 24.7% | | ❌ |
| mean PnL/trade | −0.059 | −0.055 | bleed fees | ❌ |

**Les 3 critères de la branche STOP sont remplis sur w1, et w0 bascule aussi (KL 0.576 > 0.5).**

## 2. MÉCANISME : différent de V30

- **V30** = dérive a0 **forcée par le reward** (pénalité asymétrique sell_while_flat/buy_while_open −0.28).
- **V31** = collapse **indépendant du reward** : corr(a0, toutes composantes reward) ≈ 0 (max −0.06). La pénalité invalide est à **0.00000** (le fix config sell_while_flat=0.0/buy_while_open=0.0 fonctionne parfaitement — symétrie vérifiée live).

**Mécanisme V31** : tanh saturation + entropy collapse → gradient-free float + boundary lock. Le policy std s'effondre à 0.136, les actions se figent aux bornes ±1 (tanh saturé), les gradients → 0, KL explose à chaque update car le policy ne peut plus explorer.

## 3. ROOT CAUSE (probable → à confirmer par web doc)

**ent_coef trop faible** pour le landscape reward Micro-tier dominé par les fees :
- worker_0 : ent_coef = 0.0143
- worker_1 : ent_coef = 0.0131

Avec un reward quasi-nul ou négatif constant (fees −0.0656/trade, WR ~17-25%), le signal d'avantage est faible → le policy gradient domine → entropy s'effondre → tanh sature → lock. PBT ne peut PAS sauver ça : les deux workers ont des ent_coef quasi identiques (0.0143 vs 0.0131), l'espace de recherche ne contient pas le remède.

**Remédiation documentée** (PPO continuous control) : entropy regularization plus forte (ent_coef ↑), ou bornage log_std, ou entropy bonus adaptatif. C'est un fix **hyperparamètre**, PAS un changement de reward — conforme à « On ne corrige que ce qui est démontré ».

## 4. CE QUI FONCTIONNE (acquis V31 validés live)

✅ Fix reward asymétrie : invalid_pen mean = **0.00000** sur les 1000 derniers steps des deux workers. SELL-intent advantage +0.0025 (vs +0.22 en V30, réduction 90×). asymFLAT/asymOPEN ~0.000-0.0009.
✅ Provenance reward streams Ray résolue (working_dirs).
✅ Monitor 6-gates opérationnel (a326b12).
✅ Contamination tests quarantinée (4 streams isolés).
✅ `_last_discrete_action` semantics résolu.
✅ −0.01 constant = SL_TP_CLAMPED (symétrique, bénin).

## 5. DÉCOMPOSITION invalid_ratio (dernier step logué, w1, step 5600)

```
rejections = {
  'fee_gate': 915,        # 16.3% — frais > gain espéré (structurel Micro tier)
  'hysteresis': 1106,     # 19.8% — anti-oscillation
  'cooldown_wait': 127,   # 2.3%
  'cooldown_hold_min': 17,
  'pm_rejected': 48,
  'risk_gate': 0, 'min_notional': 0, 'sell_no_position': 0, ...
}
pipeline = {policy: 5600, deadband_reject: 275, routing_reject: 3027,
            budget_reject: 1106, barrier_reject: 915, portfolio_reject: 48,
            trade_executed: 155}  # 2.8% d'exécution
```
invalid_ratio 94-95% ≈ **structurel** (fee_gate + hysteresis + routing), PAS un bug V31.

## 6. DÉCISION & PROCHAINE ÉTAPE

**STOP exécuté** (kill TERM gracieux, 20:05 UTC). Prochaine étape : fix ent_coef (hyperparamètre PBT search space), puis UN SEUL run contrôlé. **Pas de V32 à l'aveugle** — le fix doit être validé par web doc + gate déterministe avant relance.

---
*Artefacts : metrics.json, pre_kill_snapshot.txt (94 lignes PPO tables)*
