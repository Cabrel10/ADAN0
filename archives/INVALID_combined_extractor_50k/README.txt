INVALIDE — NE PAS UTILISER POUR REPRENDRE UN ENTRAÎNEMENT.

Ce checkpoint (ppo_adan0_sandbox_50176steps.zip, 50k steps, run V2 du 2026-06-24
20:14) a été entraîné avec le mauvais feature extractor : SB3 CombinedExtractor
(simple flatten, 0 paramètre) au lieu de ContextualTemporalFusionExtractor.

Conséquence : le CNN, la cross-attention, la modulation FiLM (mémoire/contexte)
et la tête auxiliaire forward_predictor N'ONT JAMAIS été exécutés. Ce n'est PAS
l'architecture cible — c'est un PPO MLP nu sur observations aplaties.

=> Tout verdict μ(size)/Cas A tiré de ce run est NON VALIDE (mauvaise archi).
=> Bug corrigé dans scripts/train_parallel_agents.py (sandbox_train wire
   désormais ContextualTemporalFusionExtractor, identique au mode heavy).
=> Preuve : scripts/audit_execution.py.
