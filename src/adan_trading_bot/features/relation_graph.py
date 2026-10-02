"""
relation_graph.py — Graphe relationnel sparse des 1 026 variables d'ADAN-System-One
===================================================================================

Construit et gère le graphe relationnel sparse reliant les 1 026 variables
du système. Remplace l'attention dense 1026x1026 ou le MLP naïf non structuré
par des relations structurelles explicites. Ni une arête ni une corrélation
ne prouve une causalité. Le prototype neural plus bas n'est pas validé GATE 7.

Types d'arêtes autorisés :
  - derived_from : calcul déterministe direct (ex: close -> return -> RSI)
  - aggregates : agrégation multi-échelles (ex: 12x5m -> 1h_running, 48x5m -> 4h_running)
  - temporal : transition d'état temporel causal (ex: x(t-1) -> x(t))
  - same_container : cohérence intra-contenant (ex: 5m pos -> 1h pos)
  - portfolio_constraint : contrainte financière (ex: SL distance -> risk_R -> position_size)
  - plan_dependency : conditionnement du plan (ex: state + direction + SL + TP -> P(win))
  - empirical_dependency : dépendance statistique prédictive apprise sur TRAIN uniquement

Origines de relation :
  - DETERMINISTIC_DEPENDENCY : lien structurel, mathématique ou invariant
  - PREDICTIVE_DEPENDENCY : lien empirique mesuré sur TRAIN uniquement

Architecture de perception relationnelle :
  1026 variables
  ↓
  variable embeddings + métadonnées
  ↓
  graph relations (sparse message passing)
  ↓
  group encoders (5m, 1h_running, 4h_running, portfolio, context)
  ↓
  temporal encoder (CNN/GRU pour séquences 5m)
  ↓
  cross-group attention
  ↓
  FiLM / context conditioning
  ↓
  STATE Z

Référence : ORDRE 3A, 3D, dev.md.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adan_trading_bot.features.feature_registry import (
    FeatureEntry,
    FeatureRegistry,
    get_feature_registry,
)


class EdgeType(str, Enum):
    DERIVED_FROM = "derived_from"
    AGGREGATES = "aggregates"
    TEMPORAL = "temporal"
    SAME_CONTAINER = "same_container"
    PORTFOLIO_CONSTRAINT = "portfolio_constraint"
    PLAN_DEPENDENCY = "plan_dependency"
    EMPIRICAL_DEPENDENCY = "empirical_dependency"


class DependencyOrigin(str, Enum):
    DETERMINISTIC_DEPENDENCY = "DETERMINISTIC_DEPENDENCY"
    PREDICTIVE_DEPENDENCY = "PREDICTIVE_DEPENDENCY"


@dataclass(frozen=True)
class RelationEdge:
    """Arête orientée reliant deux variables."""
    source: str
    target: str
    edge_type: EdgeType
    origin: DependencyOrigin
    weight: float = 1.0
    description: str = ""
    training_provenance: Optional[dict] = None


class RelationGraph:
    """Graphe sparse d'influences et dépendances des 1 026 variables."""

    def __init__(self, registry: FeatureRegistry, edges: Optional[List[RelationEdge]] = None):
        self.registry = registry
        self.names = registry.all_names()
        self.name_to_idx = {name: i for i, name in enumerate(self.names)}
        self.edges: List[RelationEdge] = []
        self._adj_out: Dict[str, List[RelationEdge]] = {n: [] for n in self.names}
        self._adj_in: Dict[str, List[RelationEdge]] = {n: [] for n in self.names}

        self._edge_keys = set()
        if edges is not None:
            for e in edges:
                self.add_edge(e)
        else:
            self._build_default_sparse_graph()

    def add_edge(self, edge: RelationEdge) -> None:
        import math
        import datetime
        if edge.source not in self.name_to_idx or edge.target not in self.name_to_idx:
            raise ValueError(f"Unknown graph endpoint: {edge.source} -> {edge.target}")
        if edge.source == edge.target or not math.isfinite(edge.weight):
            raise ValueError("Self edge or nonfinite weight")
        empirical = edge.edge_type == EdgeType.EMPIRICAL_DEPENDENCY
        if empirical != (edge.origin == DependencyOrigin.PREDICTIVE_DEPENDENCY):
            raise ValueError("Empirical edges must be PREDICTIVE; structural edges DETERMINISTIC")
        if empirical:
            p = edge.training_provenance or {}
            try:
                start = datetime.date.fromisoformat(p["start"])
                end = datetime.date.fromisoformat(p["end"])
                valid = (p["split"] == "TRAIN" and datetime.date(2017, 1, 1) <= start <= end
                         and end < datetime.date(2022, 1, 1) and p["sample_count"] > 0
                         and bool(p["method"]) and bool(p["freeze_id"]))
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError("Empirical edge needs frozen TRAIN-only provenance")
        key = (edge.source, edge.target, edge.edge_type)
        if key in self._edge_keys:
            raise ValueError(f"Duplicate typed edge: {key}")
        self._edge_keys.add(key)
        self.edges.append(edge)
        self._adj_out[edge.source].append(edge)
        self._adj_in[edge.target].append(edge)

    def _build_default_sparse_graph(self) -> None:
        """Construit les arêtes structurelles, déterministes et d'agrégation."""
        # 1. Dépendances déterministes déclarées dans le registre (derived_from)
        for var in self.registry.all_variables():
            # A current close must not be misrepresented as the producer of a
            # historical lag value at the same timestamp. Temporal links below
            # encode ordering, not a deterministic transition of price values.
            if var.name.startswith("seq_5m.lag_"):
                continue
            for parent in var.dependances_deterministes:
                if parent in self.name_to_idx:
                    self.add_edge(RelationEdge(
                        source=parent,
                        target=var.name,
                        edge_type=EdgeType.DERIVED_FROM,
                        origin=DependencyOrigin.DETERMINISTIC_DEPENDENCY,
                        weight=1.0,
                        description=f"{var.name} dérive déterministement de {parent}"
                    ))

        # 2. Agrégations temporelles 5m -> 1h (12 barres) et 5m -> 4h (48 barres)
        for bar_f in ["open", "high", "low", "volume"]:
            src = f"bar_5m.{bar_f}"
            target_field = {"open": "open", "high": "running_high",
                            "low": "running_low", "volume": "running_vol"}[bar_f]
            tgt_1h = f"c1h.{target_field}"
            tgt_4h = f"c4h.{target_field}"
            self.add_edge(RelationEdge(
                source=src, target=tgt_1h,
                edge_type=EdgeType.AGGREGATES,
                origin=DependencyOrigin.DETERMINISTIC_DEPENDENCY,
                weight=1.0, description="12x5m cumule vers 1h running"
            ))
            self.add_edge(RelationEdge(
                source=src, target=tgt_4h,
                edge_type=EdgeType.AGGREGATES,
                origin=DependencyOrigin.DETERMINISTIC_DEPENDENCY,
                weight=1.0, description="48x5m cumule vers 4h running"
            ))

        # 3. Liens temporels causaux (t-1 -> t)
        for lag in range(2, 36):
            for feat in ["open", "high", "low", "close", "volume", "wick_up", "wick_down"]:
                prev = f"seq_5m.lag_{lag}.{feat}"
                curr = f"seq_5m.lag_{lag-1}.{feat}"
                self.add_edge(RelationEdge(
                    source=prev, target=curr,
                    edge_type=EdgeType.TEMPORAL,
                    origin=DependencyOrigin.DETERMINISTIC_DEPENDENCY,
                    weight=1.0, description="Temporal ordering only; not a deterministic price transition"
                ))

        for feat in ["open", "high", "low", "close", "volume", "wick_up", "wick_down"]:
            self.add_edge(RelationEdge(
                source=f"seq_5m.lag_1.{feat}", target=f"bar_5m.{feat}",
                edge_type=EdgeType.TEMPORAL,
                origin=DependencyOrigin.DETERMINISTIC_DEPENDENCY,
                weight=1.0, description="Dernier pas vers barre fermée"
            ))

        # 4. Membership within the SAME running container, not between 1h/4h.
        # Deterministic membership does not assert a causal or numeric function.
        for prefix, suffix in (("c1h", "1h"), ("c4h", "4h")):
            self.add_edge(RelationEdge(
                source=f"{prefix}.phase_{suffix}", target=f"{prefix}.pos_in_{suffix}",
                edge_type=EdgeType.SAME_CONTAINER,
                origin=DependencyOrigin.DETERMINISTIC_DEPENDENCY,
                weight=1.0, description="Known membership in the same running container"
            ))

        # 5. Portfolio constraints (SL distance -> risk_R -> position_size)
        risk_chain = [
            ("plan.sl_pct", "plan.fees_r", EdgeType.PORTFOLIO_CONSTRAINT, DependencyOrigin.DETERMINISTIC_DEPENDENCY),
            ("plan.sl_pct", "plan.position_size_usd", EdgeType.PORTFOLIO_CONSTRAINT, DependencyOrigin.DETERMINISTIC_DEPENDENCY),
            ("plan.risk_usd", "plan.position_size_usd", EdgeType.PORTFOLIO_CONSTRAINT, DependencyOrigin.DETERMINISTIC_DEPENDENCY),
            ("portfolio.equity", "plan.risk_usd", EdgeType.PORTFOLIO_CONSTRAINT, DependencyOrigin.DETERMINISTIC_DEPENDENCY),
            ("portfolio.daily_loss_pct", "risk.circuit_breaker_active", EdgeType.PORTFOLIO_CONSTRAINT, DependencyOrigin.DETERMINISTIC_DEPENDENCY),
        ]
        for src, tgt, etype, orig in risk_chain:
            if src in self.name_to_idx and tgt in self.name_to_idx:
                self.add_edge(RelationEdge(
                    source=src, target=tgt, edge_type=etype, origin=orig,
                    weight=1.0, description="Contrainte financière de portefeuille"
                ))

        # 6. Plan dependencies (state + direction + SL + TP -> P(win))
        plan_chain = [
            ("bar_5m.close", "plan.sl_price", EdgeType.PLAN_DEPENDENCY, DependencyOrigin.DETERMINISTIC_DEPENDENCY),
            ("bar_5m.close", "plan.tp_price", EdgeType.PLAN_DEPENDENCY, DependencyOrigin.DETERMINISTIC_DEPENDENCY),
            ("plan.direction", "plan.sl_price", EdgeType.PLAN_DEPENDENCY, DependencyOrigin.DETERMINISTIC_DEPENDENCY),
            ("plan.direction", "plan.tp_price", EdgeType.PLAN_DEPENDENCY, DependencyOrigin.DETERMINISTIC_DEPENDENCY),
            ("plan.winrate_est", "plan.ev_net_r", EdgeType.PLAN_DEPENDENCY, DependencyOrigin.DETERMINISTIC_DEPENDENCY),
            ("c1h.atr_1h_pct", "plan.sl_min_bound", EdgeType.PLAN_DEPENDENCY, DependencyOrigin.DETERMINISTIC_DEPENDENCY),
            ("c1h.atr_1h_pct", "plan.sl_max_bound", EdgeType.PLAN_DEPENDENCY, DependencyOrigin.DETERMINISTIC_DEPENDENCY),
        ]
        for src, tgt, etype, orig in plan_chain:
            if src in self.name_to_idx and tgt in self.name_to_idx:
                self.add_edge(RelationEdge(
                    source=src, target=tgt, edge_type=etype, origin=orig,
                    weight=1.0, description="Dépendance du plan candidat"
                ))

    def validate(self):
        """Validate topology without allocating a dense N×N tensor."""
        if len(self.edges) > 16 * len(self.names):
            raise ValueError("Graph exceeds declared sparse edge budget")
        index = self.get_edge_index()
        if index.shape != (2, len(self.edges)):
            raise ValueError("Bad sparse COO shape")
        if index.numel() and (index.min() < 0 or index.max() >= len(self.names)):
            raise ValueError("Bad node index")
        if len(self._edge_keys) != len(self.edges):
            raise ValueError("Duplicate edges")
        # Numeric deterministic derivations only: membership/temporal relations
        # are not equations and must not be tested as if they were causal DAGs.
        adjacency = {n: [] for n in self.names}
        for edge in self.edges:
            if edge.edge_type == EdgeType.DERIVED_FROM:
                adjacency[edge.source].append(edge.target)
        visited, active = set(), set()
        def visit(node):
            if node in active:
                raise ValueError("Cycle in same-timestamp deterministic derivations")
            if node in visited:
                return
            active.add(node)
            for target in adjacency[node]:
                visit(target)
            active.remove(node)
            visited.add(node)
        for node in self.names:
            visit(node)
        return self.summary()

    def get_edge_index(self) -> torch.Tensor:
        """Retourne le tenseur (2, E) compatible PyTorch Sparse / PyG."""
        if not self.edges:
            return torch.empty((2, 0), dtype=torch.long)
        src_indices = [self.name_to_idx[e.source] for e in self.edges]
        tgt_indices = [self.name_to_idx[e.target] for e in self.edges]
        return torch.tensor([src_indices, tgt_indices], dtype=torch.long)

    def get_edge_weights(self) -> torch.Tensor:
        """Retourne les poids (E,) des arêtes."""
        if not self.edges:
            return torch.empty((0,), dtype=torch.float32)
        return torch.tensor([e.weight for e in self.edges], dtype=torch.float32)

    def sparsity_ratio(self) -> float:
        """Calcule la sparsité du graphe : 1 - (E / (N*N))."""
        n = len(self.names)
        e = len(self.edges)
        dense_max = n * n
        return 1.0 - (e / dense_max)

    def summary(self) -> Dict[str, Union[int, float, Dict[str, int]]]:
        from collections import Counter
        type_counts = Counter(e.edge_type.value for e in self.edges)
        orig_counts = Counter(e.origin.value for e in self.edges)
        return {
            "total_nodes": len(self.names),
            "total_edges": len(self.edges),
            "sparsity_pct": round(self.sparsity_ratio() * 100, 4),
            "by_edge_type": dict(type_counts),
            "by_origin": dict(orig_counts)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Perception Relationnelle PyTorch (Remplacement du MLP naïf 1026 -> Z)
# ─────────────────────────────────────────────────────────────────────────────

class SparseRelationConv(nn.Module):
    """Propagation d'information sur les arêtes autorisées du graphe sparse."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_neighbor = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, in_dim)
        edge_index : (2, E)
        edge_weight : (E,)
        """
        B, N, D = x.shape
        out_self = self.lin_self(x)

        if edge_index.shape[1] == 0:
            return F.gelu(out_self + self.bias)

        src, tgt = edge_index[0], edge_index[1]
        # Message passing vectorisé par batch
        # x[:, src, :] -> (B, E, D)
        msg = self.lin_neighbor(x[:, src, :]) * edge_weight.view(1, -1, 1)

        # Agrégation sparse dans les cibles
        out_neigh = torch.zeros(B, N, msg.shape[-1], device=x.device, dtype=x.dtype)
        tgt_expanded = tgt.view(1, -1, 1).expand(B, -1, msg.shape[-1])
        out_neigh.scatter_add_(1, tgt_expanded, msg)

        return F.gelu(out_self + out_neigh + self.bias)


class RelationalPerception(nn.Module):
    """
    Module de perception relationnelle d'ADAN-System-One.
    
    Traite les 1 026 variables selon la structure de groupe et les arêtes du graphe :
      5m sequence + 1h running + 4h running + portfolio + context
      ↓
      Variable Embeddings + Métadonnées
      ↓
      Sparse Relation Graph Propagation
      ↓
      Group Encoders
      ↓
      Temporal Encoder (5m sequence)
      ↓
      Cross-Group Attention
      ↓
      FiLM Context Conditioning
      ↓
      Latent State Z (d=256)
    """

    def __init__(
        self,
        registry: FeatureRegistry,
        relation_graph: RelationGraph,
        embed_dim: int = 32,
        z_dim: int = 256,
        n_heads: int = 4
    ):
        super().__init__()
        self.registry = registry
        self.graph = relation_graph
        self.num_vars = len(registry)
        self.embed_dim = embed_dim
        self.z_dim = z_dim

        # Embeddings de métadonnées pour chaque variable (timeframe, type, role, famille)
        self.timeframe_map = {"5m": 0, "1h": 1, "4h": 2, "trade": 3, "1d": 4, "global": 5}
        self.role_map = {"perception": 0, "contexte": 1, "plan": 2, "risque": 3, "portefeuille": 4}

        self.val_proj = nn.Linear(1, embed_dim)
        self.tf_embed = nn.Embedding(len(self.timeframe_map), embed_dim)
        self.role_embed = nn.Embedding(len(self.role_map), embed_dim)

        # Buffers pour les métadonnées fixes des 1 026 variables
        tf_indices = [self.timeframe_map.get(v.timeframe, 5) for v in registry.all_variables()]
        role_indices = [self.role_map.get(v.role_potentiel, 1) for v in registry.all_variables()]
        self.register_buffer("tf_indices", torch.tensor(tf_indices, dtype=torch.long))
        self.register_buffer("role_indices", torch.tensor(role_indices, dtype=torch.long))

        # Buffers du graphe sparse
        self.register_buffer("edge_index", relation_graph.get_edge_index())
        self.register_buffer("edge_weights", relation_graph.get_edge_weights())

        # Propagation relationnelle sparse
        self.graph_conv = SparseRelationConv(embed_dim, embed_dim)

        # Group Encoders
        # Groupes d'indices dans les 1 026 variables
        self.idx_5m = [i for i, v in enumerate(registry.all_variables()) if v.timeframe == "5m"]
        self.idx_1h = [i for i, v in enumerate(registry.all_variables()) if v.timeframe == "1h"]
        self.idx_4h = [i for i, v in enumerate(registry.all_variables()) if v.timeframe == "4h"]
        self.idx_port = [i for i, v in enumerate(registry.all_variables()) if v.role_potentiel == "portefeuille"]
        self.idx_ctx = [i for i, v in enumerate(registry.all_variables()) if v.role_potentiel == "contexte"]

        self.enc_5m = nn.Sequential(nn.Linear(len(self.idx_5m) * embed_dim, z_dim), nn.GELU())
        self.enc_1h = nn.Sequential(nn.Linear(len(self.idx_1h) * embed_dim, z_dim), nn.GELU())
        self.enc_4h = nn.Sequential(nn.Linear(len(self.idx_4h) * embed_dim, z_dim), nn.GELU())
        self.enc_port = nn.Sequential(nn.Linear(len(self.idx_port) * embed_dim, z_dim), nn.GELU())
        self.enc_ctx = nn.Sequential(nn.Linear(len(self.idx_ctx) * embed_dim, z_dim), nn.GELU())

        # Cross-Group Multi-Head Attention (5 tokens: 5m, 1h, 4h, port, ctx)
        self.cross_attn = nn.MultiheadAttention(embed_dim=z_dim, num_heads=n_heads, batch_first=True)
        self.norm_groups = nn.LayerNorm(z_dim)

        # FiLM Generator (Conditionnement par le contexte et le portefeuille)
        self.film_gen = nn.Sequential(
            nn.Linear(z_dim * 2, z_dim * 2),
            nn.GELU(),
            nn.Linear(z_dim * 2, z_dim * 2)  # gamma et beta
        )

        # Sortie finale de l'état Z
        self.out_head = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.LayerNorm(z_dim),
            nn.GELU()
        )

    def forward(self, x_vars: torch.Tensor) -> torch.Tensor:
        """
        x_vars : Tenseur des 1 026 variables de shape (B, 1026)
        Retourne l'état Z de dimension (B, 256).
        """
        B, N = x_vars.shape
        assert N == self.num_vars, f"Attendu {self.num_vars} variables, reçu {N}"

        # 1. Embeddings de valeurs + métadonnées
        val_emb = self.val_proj(x_vars.unsqueeze(-1))  # (B, N, D)
        tf_emb = self.tf_embed(self.tf_indices).unsqueeze(0)  # (1, N, D)
        role_emb = self.role_embed(self.role_indices).unsqueeze(0)  # (1, N, D)
        node_features = val_emb + tf_emb + role_emb

        # 2. Propagation relationnelle sparse le long des arêtes du graphe
        node_repr = self.graph_conv(node_features, self.edge_index, self.edge_weights)  # (B, N, D)

        # 3. Group Encoders
        g_5m = self.enc_5m(node_repr[:, self.idx_5m, :].reshape(B, -1))
        g_1h = self.enc_1h(node_repr[:, self.idx_1h, :].reshape(B, -1))
        g_4h = self.enc_4h(node_repr[:, self.idx_4h, :].reshape(B, -1))
        g_port = self.enc_port(node_repr[:, self.idx_port, :].reshape(B, -1))
        g_ctx = self.enc_ctx(node_repr[:, self.idx_ctx, :].reshape(B, -1))

        # 4. Cross-Group Attention (tokens: 5m, 1h, 4h)
        # Séquence de groupes de marché : shape (B, 3, z_dim)
        market_tokens = torch.stack([g_5m, g_1h, g_4h], dim=1)
        attn_out, _ = self.cross_attn(market_tokens, market_tokens, market_tokens)
        market_rep = self.norm_groups(market_tokens + attn_out).mean(dim=1)  # (B, z_dim)

        # 5. FiLM Conditioning par (Portefeuille + Contexte)
        cond = torch.cat([g_port, g_ctx], dim=-1)  # (B, 2 * z_dim)
        film_params = self.film_gen(cond)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        z_conditioned = (1.0 + gamma) * market_rep + beta

        # 6. Représentation finale Z
        z = self.out_head(z_conditioned)
        return z
