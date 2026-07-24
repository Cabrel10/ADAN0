# Inventaire des checkpoints ADAN0

> Règle : ne JAMAIS se fier au nom de fichier ni à "le plus récent". Verdict croisé
> avec les diag CSV (pct_buy/pct_sell/a0_mean au step du checkpoint).
> MAJ 2026-07-04.

Aucun `*.pkl` VecNormalize présent → cohérent avec VecNormalize DÉSACTIVÉ. Les
checkpoints n'ont donc PAS de mismatch de stats de normalisation (contrairement au
witness SWEETSPOT V8, cf HANDOFF §5).

| Checkpoint | Run | Step | Date | a0_mean | pct_buy | pct_sell | Verdict |
|-----------|-----|------|------|---------|---------|----------|---------|
| ppo_adan0_sandbox_512steps.zip | V12 | 512 | Jul 3 04:23 | ~-0.005 | ~45% | ~49% | **SAIN** (avant dérive, équilibré) |
| ppo_adan0_sandbox_checkpoint_25000_steps.zip | V12 | 25000 | Jul 3 05:41 | +0.185 | ~90% | ~7.5% | **COLLAPSÉ** (BUY runaway installé) |
| ppo_adan0_sandbox_40000steps.zip | V12 | 40000 | Jul 3 06:26 | +0.302 | 97.9% | 1.6% | **COLLAPSÉ** (pire point) |
| ppo_adan0_sandbox_2000_steps.zip | V10/V11 | 2000 | Jul 2 10:23 | équilibré | ~50% | ~50% | Sain (mais scalper, obsolète) |
| ppo_adan0_sandbox_2048steps.zip | V10/V11 | 2048 | Jul 2 10:23 | équilibré | ~50% | ~50% | Sain (scalper) |
| ppo_adan0_sandbox_checkpoint_50000_steps.zip | V11 | 50000 | Jul 2 15:18 | dérive partielle | monte | baisse | Suspect (mi-collapse) |
| ppo_adan0_sandbox_checkpoint_75000_steps.zip | V11 | 75000 | Jul 2 16:30 | ~collapse | ~0.9+ | bas | Collapsé |
| ppo_adan0_sandbox_78000steps.zip | V11 | 78000 | Jul 2 16:38 | collapse | 0.97 | bas | Collapsé |
| ppo_adan0_sandbox_70000steps.zip | V10 | 70000 | Jul 2 03:16 | collapse | 0.97 | bas | Collapsé |
| ppo_adan0_sandbox_checkpoint_100000_steps_SWEETSPOT.zip | V8 | 100000 | Jul 1 20:30 | ? | ? | ? | **NE PAS PROMOUVOIR** (étiquette non vérifiée, VecNorm identity, cf HANDOFF §5) |

## Conclusion

- **Aucun checkpoint sain "post-fix" exploitable** : le seul point V12 avant dérive est
  @512 steps (trop peu entraîné pour être utile). Tous les checkpoints > ~10k steps de
  chaque run sont contaminés par le BUY runaway.
- **La cause étant dans le reward (HANDOFF §4), aucun re-training ne produira un
  checkpoint sain tant que le reward n'est pas corrigé.** Inutile de promouvoir un
  checkpoint existant en paper trading : ils collapseront tous vers always-BUY à
  l'inférence.
- Recommandation : conserver `512steps` (témoin "avant dérive") + les diag CSV comme
  preuve ; ne rien promouvoir en live avant fix reward + nouveau run validé.
