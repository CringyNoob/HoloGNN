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

# Future-work modules (§8.2) — optional, off by default.
from src.sequence_mixers import CrossAttentionFusion, SelectiveSSM
# V6 modules.
from src.pooling import AttentionPooling, masked_mean
from src.utils.graph_builder import build_batched_attention_graph

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

    def __init__(self, output_dim: int = 320,
                 fusion_mode: str = "concat",
                 use_ssm: bool = False,
                 pool: str = "attention",
                 graph_mode: str = "per_sample",
                 mech_feature_dim: int = MECH_FEATURE_DIM,
                 top_k: int = 8,
                 freeze_esm: bool = False,
                 freeze_esm_layers: int = 0):
        """
        Args:
            output_dim        : final embedding size (default 320).
            fusion_mode       : "concat" (V5) or "cross_attention" (§8.2-i).
            use_ssm           : insert a Selective-SSM / Mamba mixer (§8.2-ii).
            pool              : [V6] "attention" (mask-aware learned pooling,
                                default) or "mean" (padding-aware mean; "mean"
                                also reproduces V5 when no padding is present).
            graph_mode        : [V6] "per_sample" (each protein its own graph
                                with attention-weighted edges + backbone edges,
                                default) or "shared" (V5 batch-mean topology).
            mech_feature_dim  : number of mechanistic channels (3 = V5 default;
                                6 enables the expanded protein-only descriptors).
            top_k             : [V6] neighbours per node in per-sample graphs.
            freeze_esm        : freeze the entire ESM-2 encoder.
            freeze_esm_layers : freeze ESM-2 embeddings + the first N encoder
                                layers (ignored if freeze_esm is True).
        """
        super().__init__()

        if fusion_mode not in ("concat", "cross_attention"):
            raise ValueError(f"fusion_mode must be 'concat' or 'cross_attention', got {fusion_mode!r}")
        if pool not in ("attention", "mean"):
            raise ValueError(f"pool must be 'attention' or 'mean', got {pool!r}")
        if graph_mode not in ("per_sample", "shared"):
            raise ValueError(f"graph_mode must be 'per_sample' or 'shared', got {graph_mode!r}")

        self.fusion_mode      = fusion_mode
        self.use_ssm          = use_ssm
        self.pool             = pool
        self.graph_mode       = graph_mode
        self.mech_feature_dim = mech_feature_dim
        self.top_k            = top_k
        gat_in                = ESM2_HIDDEN_DIM + mech_feature_dim   # 323 (V5) or 326

        # ------------------------------------------------------------------
        # 1. ESM-2 Sequence Encoder
        # ------------------------------------------------------------------
        self.esm_model = EsmModel.from_pretrained(
            "facebook/esm2_t6_8M_UR50D",
            output_attentions=True,
        )
        self._apply_esm_freezing(freeze_esm, freeze_esm_layers)

        # ------------------------------------------------------------------
        # 1b. [§8.2] Optional future-work mixers (OFF by default)
        # ------------------------------------------------------------------
        self.ssm = SelectiveSSM(dim=ESM2_HIDDEN_DIM) if use_ssm else None
        self.fusion = (CrossAttentionFusion(esm_dim=ESM2_HIDDEN_DIM,
                                            mech_dim=mech_feature_dim)
                       if fusion_mode == "cross_attention" else None)

        # ------------------------------------------------------------------
        # 2. GATv2Conv layers (edge_dim=1 → attention weight as an edge feature;
        #    self-loops are supplied by the graph builders, so add_self_loops=False).
        # ------------------------------------------------------------------
        if _PYGEO_AVAILABLE:
            self.gat1 = _GAT_CLS(gat_in, gat_in,     heads=4, concat=False,
                                 edge_dim=1, add_self_loops=False)
            self.gat2 = _GAT_CLS(gat_in, output_dim, heads=4, concat=False,
                                 edge_dim=1, add_self_loops=False)

        # ------------------------------------------------------------------
        # 3. Residual projection (raw ESM-2 → output_dim)
        # ------------------------------------------------------------------
        if ESM2_HIDDEN_DIM != output_dim:
            self.residual_proj = nn.Linear(ESM2_HIDDEN_DIM, output_dim, bias=False)
        else:
            self.residual_proj = nn.Identity()

        # Used only when torch_geometric is unavailable (GAT bypass).
        self.fallback_proj = nn.Linear(gat_in, output_dim)

        # [V6] Mask-aware pooling head.
        self.pool_layer = AttentionPooling(output_dim) if pool == "attention" else None

        self.layer_norm = nn.LayerNorm(output_dim)
        self.relu       = nn.ReLU()

    # -----------------------------------------------------------------------
    def _apply_esm_freezing(self, freeze_esm: bool, freeze_esm_layers: int) -> None:
        """Freeze the whole ESM-2 encoder, or its embeddings + first N layers."""
        if freeze_esm:
            for p in self.esm_model.parameters():
                p.requires_grad = False
            return
        if freeze_esm_layers and freeze_esm_layers > 0:
            for p in self.esm_model.embeddings.parameters():
                p.requires_grad = False
            layers = self.esm_model.encoder.layer
            for layer in layers[:min(freeze_esm_layers, len(layers))]:
                for p in layer.parameters():
                    p.requires_grad = False

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
        # Step 1b  [§8.2-ii]: Optional Selective-SSM / Mamba sequence mixing
        # ------------------------------------------------------------------
        if self.ssm is not None:
            esm_node_embeddings = self.ssm(esm_node_embeddings)   # (B, L, 320)

        # ------------------------------------------------------------------
        # Step 2  [V3.0 / §8.2-i]: Mechanistic feature injection
        #   "concat"          → simple geometric stacking (original V5).
        #   "cross_attention" → cross-modal attention fusion, still → (B, L, 323).
        # ------------------------------------------------------------------
        if self.fusion is not None:
            key_padding_mask = (attention_mask == 0)              # True at padding
            node_embeddings  = self.fusion(esm_node_embeddings, mechanistic_features,
                                           key_padding_mask=key_padding_mask)
        else:
            node_embeddings = torch.cat([esm_node_embeddings, mechanistic_features], dim=-1)
        # node_embeddings is now (B, L, 323)

        # ------------------------------------------------------------------
        # Step 3  [V6]: Dynamic graph construction from the ESM-2 attention map.
        #   "per_sample" → each protein gets its own attention-weighted graph
        #                  (top-k + backbone edges); the weights become edge_attr.
        #   "shared"     → V5 batch-mean topology, replicated per sample with
        #                  unit edge weights (kept for backward comparison).
        # ------------------------------------------------------------------
        device   = input_ids.device
        edge_attr = None
        if edge_index is None and _PYGEO_AVAILABLE:
            last_attn_layer = esm_out.attentions[-1]                # (B, heads, L, L)
            avg_attention   = torch.mean(last_attn_layer, dim=1)   # (B, L, L)

            if self.graph_mode == "per_sample":
                edge_index, edge_attr = build_batched_attention_graph(
                    avg_attention, attention_mask, k=self.top_k, add_backbone=True
                )
            else:  # "shared" (V5)
                shared     = torch.mean(avg_attention, dim=0)      # (L, L)
                base_ei    = build_attention_graph(shared, threshold=attention_threshold)
                diag       = torch.arange(L, device=device)        # self-loops
                base_src   = torch.cat([base_ei[0], diag])
                base_dst   = torch.cat([base_ei[1], diag])
                srcs, dsts = [], []
                for b in range(B):
                    srcs.append(base_src + b * L)
                    dsts.append(base_dst + b * L)
                edge_index = torch.stack([torch.cat(srcs), torch.cat(dsts)], dim=0).long()
                edge_attr  = torch.ones(edge_index.size(1), 1, device=device)
            edge_index = edge_index.to(device)
            edge_attr  = edge_attr.to(device)

        # A caller-supplied edge_index arrives without weights → use unit weights.
        if edge_index is not None and edge_attr is None and _PYGEO_AVAILABLE:
            edge_attr = torch.ones(edge_index.size(1), 1, device=device)

        # ------------------------------------------------------------------
        # Step 4: GATv2Conv message passing (attention weights as edge features)
        # ------------------------------------------------------------------
        x_flat = node_embeddings.view(-1, node_embeddings.size(-1))  # (B*L, gat_in)

        if edge_index is not None and edge_index.numel() > 0 and _PYGEO_AVAILABLE:
            x = self.relu(self.gat1(x_flat, edge_index, edge_attr))  # (B*L, gat_in)
            x = self.gat2(x, edge_index, edge_attr)                  # (B*L, output_dim)
        else:
            # PyG unavailable (or empty graph) — bypass GATv2 and project so the
            # node features are preserved (not zeroed) before the residual add.
            x = self.fallback_proj(x_flat)                           # (B*L, output_dim)

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

        # x is always output_dim here (GATv2 output, or fallback_proj output).
        x = self.layer_norm(x + residual)                           # (B*L, output_dim)

        # ------------------------------------------------------------------
        # Step 6  [V6]: Mask-aware pooling → graph-level embedding.
        #   Padding tokens never contribute (attention pool or padding-aware mean).
        # ------------------------------------------------------------------
        x_reshaped = x.view(B, L, -1)                               # (B, L, output_dim)
        if self.pool_layer is not None:
            graph_embedding = self.pool_layer(x_reshaped, attention_mask)
        else:
            graph_embedding = masked_mean(x_reshaped, attention_mask)

        return x_reshaped, graph_embedding