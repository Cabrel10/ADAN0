#!/usr/bin/env python3
"""V29 PATCH 4: rendre le sizing a l'agent (size_raw), HMM ne sert plus qu'a l'EV gate.

Strategie: match sur des ancres ASCII uniquement (le heredoc bash corrompt
les caracteres accentues). On localise la ligne d'ecrasement LINEAR_EXPO et
on remplace le bloc par une version ou HMM est lecture-seule (EV gate).
"""
from pathlib import Path

path = Path("src/adan_trading_bot/environment/multi_asset_chunked_env.py")
src = path.read_text(encoding="utf-8")

# --- Verrou d'idempotence -------------------------------------------------
if "V29 PATCH 4" in src:
    print("PATCH 4 deja applique - skip")
    raise SystemExit(0)

# --- Ancres ASCII ----------------------------------------------------------
anchor_start = "            # ---- Target-Weight Sizing ----"
anchor_end = "                notional_usd = max(min_order_value, capital * target_exposure_pct)"

i_start = src.find(anchor_start)
if i_start < 0:
    raise SystemExit("ERREUR: ancre start introuvable - abort")
i_end = src.find(anchor_end, i_start)
if i_end < 0:
    raise SystemExit("ERREUR: ancre end introuvable - abort")
i_end += len(anchor_end)

old_block = src[i_start:i_end]
# Securite: le bloc doit contenir la double assignation (anomalie C).
if old_block.count("target_exposure_pct =") < 2:
    raise SystemExit(
        f"ERREUR: bloc sans double assignation target_exposure_pct "
        f"({old_block.count('target_exposure_pct =')}) - abort"
    )

new_block = (
    "            # ---- Target-Weight Sizing ----\n"
    "            # V29 PATCH 4 (2026-08-12): le sizing est RENDU A L'AGENT.\n"
    "            # Avant : cette valeur etait ecrasee plus bas par LINEAR_EXPO x\n"
    "            # bull_prob_HMM (anomalie C du rapport PHASE 2) -> le canal size\n"
    "            # de la politique n'avait aucun gradient utile.\n"
    "            # Desormais target_exposure_pct reste pilote par size_raw et\n"
    "            # HMM ne sert plus qu'a l'EV gate (p_hmm).\n"
    "            normalized_size = (size_raw + 1.0) / 2.0  # 0..1\n"
    "            normalized_size = max(0.0, min(1.0, normalized_size))\n"
    "            target_exposure_pct = min_exp + normalized_size * (max_exp - min_exp)\n"
    "            # ==============================================================\n"
    "            # HMM confidence : lecture SEULE pour l'EV gate (p_hmm).\n"
    "            # N'ecrase PLUS le sizing (reste strictement dans les bornes\n"
    "            # du palier par construction ci-dessus).\n"
    "            # ==============================================================\n"
    "            p_hmm = 0.5  # used later by EV gate\n"
    "            try:\n"
    "                # Confiance HMM (bull probability) - advisory EV gate only\n"
    "                obs = getattr(self, '_last_observation', None)\n"
    "                if obs is not None and isinstance(obs, dict):\n"
    "                    ctx = obs.get('context_vector')\n"
    "                    if ctx is not None and hasattr(ctx, '__len__') and len(ctx) >= 6:\n"
    "                        bull_prob = float(ctx[3])\n"
    "                        p_hmm = max(0.01, min(0.99, bull_prob))  # save for EV gate\n"
    "                # Exposition garantie dans [min_exp, max_exp] (bornees palier)\n"
    "                target_exposure_pct = max(min_exp, min(max_exp, target_exposure_pct))\n"
    "                # Montant a investir (plancher min_order)\n"
    "                notional_usd = max(min_order_value, capital * target_exposure_pct)"
)

src = src[:i_start] + new_block + src[i_end:]
path.write_text(src, encoding="utf-8")

# Validation syntaxique immediate.
import ast
ast.parse(path.read_text(encoding="utf-8"))
print("PATCH 4 applique OK - syntax OK")
