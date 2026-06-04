"""
src/backbone.py
===============
HoloGNN Backbone — Evolution History
--------------------------------------
V1.0  Simple linear graph fallback (i → i+1).
V2.0  Dual-Track Dynamic Graph: edges built from ESM-2 attention map (no fallback).
V3.0  Mechanistic Injection: mRNA_fold / CAI / Charge concatenated onto node
      embeddings before GATConv (Prior et al. 2024).
V5.0  Production Upgrades:
        • GATConv → GATv2Conv  (dynamic attention; Brody et al. 2022 "How Attentive
          are Graph Attention Networks?" ICLR 2022).  GATv2 fixes the static
          attention bottleneck of GATv1 where e(h_i, h_j) = a·[Wh_i || Wh_j]
          is computed before non-linearity, making it rank-1 and incapable of
          expressing complex neighbourhood interactions.  GATv2 applies LeakyReLU
          first: e(h_i, h_j) = a·LeakyReLU(W·[h_i || h_j]) — truly dynamic.
        • Residual Skip Connection: raw ESM-2 node embeddings are added back to
          the GATv2 output to prevent over-smoothing across layers.  A linear
          projection is used to match dimensions when ESM2_DIM ≠ output_dim.
"""

import torch
import torch.nn as nn
from transformers import EsmModel

# ---------------------------------------------------------------------------
# Optional PyG import — tries GATv2Conv first (V5.0), falls back gracefully
# ---------------------------------------------------------------------------
try:
    from torch_geometric.nn import GATv2Conv
    _PYGEO_AVAILABLE = True
    _GAT_CLS         = GATv2Conv
    print("✅ torch_geometric GATv2Conv available — V5.0 dynamic attention ENABLED.")
except ImportError:
    try:
        # Fallback to GATConv (V3/V4 behaviour) if GATv2 not present
        from torch_geometric.nn import GATConv as GATv2Conv
        _PYGEO_AVAILABLE = True
        _GAT_CLS         = GATv2Conv
        print("⚠️  GATv2Conv not found — falling back to GATConv (V3.0 behaviour).")
    except ImportError:
        print("Warning: torch_geometric not found. GNN layers will be disabled.")
        GATv2Conv        = None
        _PYGEO_AVAILABLE = False
        _GAT_CLS         = None

# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------
# ESM-2 t6 8M hidden dimension
ESM2_HIDDEN_DIM  = 320
# Three mechanistic channels (mRNA_fold, CAI, Charge) — from dataset.py V5.0
MECH_FEATURE_DIM = 3
# Combined input to GATv2 layers: 320 + 3 = 323
GAT_IN_CHANNELS  = ESM2_HIDDEN_DIM + MECH_FEATURE_DIM   # 323


# ---------------------------------------------------------------------------
# V2.0 — Attention-based graph builder  (still used in V5.0)
# ---------------------------------------------------------------------------
def build_attention_graph(avg_attention: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """
    Construct edge_index from a (seq_len, seq_len) ESM-2 averaged attention map.

    Args:
        avg_attention : (L, L) tensor — batch-mean of per-head last-layer attentions.
        threshold     : minimum symmetrised attention weight to form an edge (default 0.05).

    Returns:
        edge_index : LongTensor (2, E) — bidirectional contact edges.
    """
    adj        = avg_attention + avg_attention.t()   # symmetrise
    rows, cols = torch.where(adj > threshold)
    mask       = rows != cols                        # remove self-loops
    return torch.stack([rows[mask], cols[mask]], dim=0)


# ---------------------------------------------------------------------------
# V5.0 — Backbone
# ---------------------------------------------------------------------------
class HoloGNNBackbone(nn.Module):
    """
    HoloGNN Backbone — Version 5.0

    Forward pass summary:
        1. ESM-2 (t6 8M) encodes token sequence → node embeddings  (B, L, 320).
        2. [V3.0] Mechanistic features (B, L, 3) are concatenated  → (B, L, 323).
        3. [V2.0] Edge index is built from the ESM-2 attention map.
        4. [V5.0] Two-layer GATv2Conv with dynamic attention refines nodes.
        5. [V5.0] Residual skip connection: ESM-2 embeddings projected to
                  output_dim are added back to the GATv2 output.
        6. Mean pooling → graph-level embedding (B, output_dim).

    Args:
        output_dim : Final embedding size (default 320).
                     GATv2Conv layers compress GAT_IN_CHANNELS (323) → output_dim.
    """

    def __init__(self, output_dim: int = 320):
        super().__init__()

        # ------------------------------------------------------------------
        # 1. ESM-2 Sequence Encoder
        #    output_attentions=True exposes attention weights for graph building.
        # ------------------------------------------------------------------
        self.esm_model = EsmModel.from_pretrained(
            "facebook/esm2_t6_8M_UR50D",
            output_attentions=True,
        )

        # ------------------------------------------------------------------
        # 2. [V5.0] GATv2Conv layers  (dynamic attention)
        #    GATv2 is strictly more expressive than GATv1 for identical parameter
        #    budgets, as it applies the shared weight matrix to the concatenated
        #    pair [h_i || h_j] then applies LeakyReLU before the attention dot.
        # ------------------------------------------------------------------
        if _PYGEO_AVAILABLE:
            self.gat1 = _GAT_CLS(GAT_IN_CHANNELS, GAT_IN_CHANNELS, heads=4, concat=False)
            self.gat2 = _GAT_CLS(GAT_IN_CHANNELS, output_dim,      heads=4, concat=False)

        # ------------------------------------------------------------------
        # 3. [V5.0] Residual projection
        #    Projects raw ESM-2 embeddings (ESM2_HIDDEN_DIM = 320) to output_dim
        #    so they can be added directly to the GATv2 output.
        #    If ESM2_HIDDEN_DIM == output_dim this is an identity.
        # ------------------------------------------------------------------
        if ESM2_HIDDEN_DIM != output_dim:
            self.residual_proj = nn.Linear(ESM2_HIDDEN_DIM, output_dim, bias=False)
        else:
            self.residual_proj = nn.Identity()

        self.layer_norm = nn.LayerNorm(output_dim)
        self.relu       = nn.ReLU()

    # -----------------------------------------------------------------------
    def forward(
        self,
        input_ids:             torch.Tensor,
        attention_mask:        torch.Tensor,
        mechanistic_features:  torch.Tensor,        # V3.0+: (B, L, 3)
        edge_index:            torch.Tensor | None = None,
        attention_threshold:   float = 0.05,
    ):
        """
        Args:
            input_ids            : (B, L) token ids.
            attention_mask       : (B, L) 1/0 padding mask.
            mechanistic_features : (B, L, 3) true biophysical features (V5.0).
            edge_index           : Pre-built graph edges; built dynamically if None.
            attention_threshold  : Edge-construction cutoff (V2.0).

        Returns:
            node_embeddings : (B, L, output_dim) — per-residue representations.
            graph_embedding : (B, output_dim)    — mean-pooled graph embedding.
        """
        B = input_ids.size(0)

        # ------------------------------------------------------------------
        # Step 1: ESM-2 forward pass
        # ------------------------------------------------------------------
        esm_out              = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
        esm_node_embeddings  = esm_out.last_hidden_state      # (B, L, 320)
        L                    = esm_node_embeddings.size(1)

        # ------------------------------------------------------------------
        # Step 2  [V3.0]: Mechanistic feature injection
        # ------------------------------------------------------------------
        node_embeddings = torch.cat([esm_node_embeddings, mechanistic_features], dim=-1)
        # node_embeddings is now (B, L, 323)

        # ------------------------------------------------------------------
        # Step 3  [V2.0]: Dynamic graph construction from ESM-2 attention map
        # ------------------------------------------------------------------
        if edge_index is None and _PYGEO_AVAILABLE:
            last_attn_layer = esm_out.attentions[-1]                # (B, heads, L, L)
            avg_attention   = torch.mean(last_attn_layer, dim=1)   # (B, L, L)
            batch_avg_attn  = torch.mean(avg_attention,  dim=0)    # (L, L) shared topology
            edge_index      = build_attention_graph(batch_avg_attn, threshold=attention_threshold)
            edge_index      = edge_index.to(input_ids.device)

        # ------------------------------------------------------------------
        # Step 4  [V5.0]: GATv2Conv message passing (dynamic attention)
        # ------------------------------------------------------------------
        x_flat = node_embeddings.view(-1, node_embeddings.size(-1))  # (B*L, 323)

        if edge_index is not None and _PYGEO_AVAILABLE:
            x = self.relu(self.gat1(x_flat, edge_index))  # (B*L, 323)
            x = self.gat2(x, edge_index)                  # (B*L, output_dim)
        else:
            # PyG unavailable — bypass GATv2; residual will still apply.
            x = x_flat

        # ------------------------------------------------------------------
        # Step 5  [V5.0]: Residual skip connection
        #
        #   Add the projected raw ESM-2 embeddings back to the GATv2 output.
        #   This prevents over-smoothing: after many rounds of neighbourhood
        #   aggregation, node representations become indistinguishable
        #   ("over-smoothed").  The skip connection preserves the original
        #   sequence identity signal from the language model.
        #
        #   We normalise the residual-added output with LayerNorm to stabilise
        #   training at scale (780k samples, full MegaScale dataset).
        # ------------------------------------------------------------------
        # Project raw ESM-2 embeddings to output_dim for dimension alignment
        esm_flat   = esm_node_embeddings.view(-1, ESM2_HIDDEN_DIM)   # (B*L, 320)
        residual   = self.residual_proj(esm_flat)                    # (B*L, output_dim)

        # Trim x to output_dim in the fallback case (x may be 323-dim)
        if x.size(-1) != residual.size(-1):
            # Only reachable if PyG is unavailable; project x as well
            x_proj = nn.functional.linear(
                x, torch.zeros(residual.size(-1), x.size(-1), device=x.device)
            )
            x = x_proj

        x = self.layer_norm(x + residual)                           # (B*L, output_dim)

        # ------------------------------------------------------------------
        # Step 6: Mean pooling → graph-level embedding
        # ------------------------------------------------------------------
        x_reshaped      = x.view(B, L, -1)                          # (B, L, output_dim)
        graph_embedding = torch.mean(x_reshaped, dim=1)             # (B, output_dim)

        return x_reshaped, graph_embedding