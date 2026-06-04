"""
src/full_model.py
=================
HoloGNN Full Model — Evolution History
----------------------------------------
V1.0  Single-sequence forward pass; 'idr' task called siamese_head.mlp directly.
V4.0  True Siamese Pass: 'idr' task accepts a (data_wt, data_mt) tuple, passes
      each through the backbone independently, and feeds (z_wt, z_mt) into
      SiameseStabilityHead.  This unlocks the AntisymmetricLoss in loss.py.
"""

import torch
import torch.nn as nn
from src.backbone import HoloGNNBackbone
from src.heads import (
    ProteomicsHead,
    SiameseStabilityHead,
    EnsembleIDRHead,
    ThreeStateStabilityHead,
    MFIHead,
    StabilityRegressionHead,
)


class HoloGNN(nn.Module):
    """
    HoloGNN — Multi-task Protein Structure & Function Predictor
    Version 4.0

    Supported tasks
    ---------------
    'proteomics'
        Single-sequence retention-time regression (MassIVE-KB pre-training).
        Input : data        — single DataBatch
        Output: scalar prediction per sample

    'idr'  [V4.0 upgrade]
        Siamese ΔΔG stability regression (MegaScale / FireProtDB).
        Input : data        — tuple (data_wt, data_mt)
                              Each is a DataBatch with .input_ids, .mask,
                              .edge_index, .mechanistic_features
        Output: (dG_wt_to_mt, dG_mt_to_wt)
                Both are scalar predictions used by AntisymmetricLoss.

    default
        Returns the raw graph embedding for downstream fine-tuning.
    """

    def __init__(self, fusion_mode: str = "concat", use_ssm: bool = False,
                 num_species: int = 0):
        """
        Args:
            fusion_mode : "concat" (default) or "cross_attention" (§8.2-i).
            use_ssm     : enable the Selective-SSM / Mamba mixer (§8.2-ii).
            num_species : >0 enables multi-species conditioning on the
                          proteomics head (§8.2 multi-species proteomics).
        All default to the original V5 behaviour.
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Backbone  (V3.0+)
        # output_dim=320 is the GATConv output dimension.
        # The backbone internally handles the 323-dim GATConv input
        # (ESM-2 320 + 3 mechanistic features).
        # ------------------------------------------------------------------
        self.backbone = HoloGNNBackbone(
            output_dim=320, fusion_mode=fusion_mode, use_ssm=use_ssm
        )

        # ------------------------------------------------------------------
        # Task heads — all receive a 320-dim graph embedding.
        # ------------------------------------------------------------------
        self.proteomics_head = ProteomicsHead(input_dim=320, num_species=num_species)
        self.siamese_head    = SiameseStabilityHead(input_dim=320)
        self.idr_head        = EnsembleIDRHead(input_dim=320)

        # §8.2 future-work heads
        self.three_state_head = ThreeStateStabilityHead(input_dim=320)
        self.mfi_head         = MFIHead(input_dim=320)
        self.stability_head   = StabilityRegressionHead(input_dim=320)

    # -----------------------------------------------------------------------
    def _encode(self, data) -> torch.Tensor:
        """
        Shared helper: run a single DataBatch through the backbone and return
        the graph-level embedding z of shape (B, 320).

        Args:
            data : object with attributes
                     .input_ids             — (B, L)
                     .mask                  — (B, L)
                     .mechanistic_features  — (B, L, 3)  [V3.0]
                     .edge_index            — (2, E) or None [V2.0]
        """
        _, graph_emb = self.backbone(
            input_ids            = data.input_ids,
            attention_mask       = data.mask,
            mechanistic_features = data.mechanistic_features,
            edge_index           = data.edge_index,
        )
        return graph_emb   # (B, 320)

    # -----------------------------------------------------------------------
    def forward(self, data, task: str = "proteomics"):
        """
        Args:
            data : DataBatch  —or—  tuple(DataBatch, DataBatch) for the paired
                   tasks ('idr', 'three_state').
            task : 'proteomics' | 'stability' | 'mfi' | 'idr' | 'three_state'.

        Returns:
            'proteomics'  → Tensor (B, 1)   retention-time / expression
            'stability'   → Tensor (B, 1)   single-sequence absolute ΔG
            'mfi'         → Tensor (B, 1)   mean fluorescence intensity
            'idr'         → tuple( dG_wt_to_mt (B,1), dG_mt_to_wt (B,1) )
            'three_state' → Tensor (B, 3)   class logits (destab/neutral/stab)
            default       → Tensor (B, 320) raw embedding
        """

        # ------------------------------------------------------------------
        # Task: proteomics  (single-sequence regression; optional species id)
        # ------------------------------------------------------------------
        if task == "proteomics":
            z = self._encode(data)
            return self.proteomics_head(z, getattr(data, "species_id", None))

        # ------------------------------------------------------------------
        # Task: stability  (single-sequence absolute ΔG regression — MegaScale)
        # ------------------------------------------------------------------
        if task == "stability":
            z = self._encode(data)
            return self.stability_head(z)

        # ------------------------------------------------------------------
        # Task: mfi  (§8.2-iv — auxiliary mean-fluorescence-intensity regression)
        # ------------------------------------------------------------------
        if task == "mfi":
            z = self._encode(data)
            return self.mfi_head(z)

        # ------------------------------------------------------------------
        # Task: three_state  (§8.2-iii — destabilising / neutral / stabilising)
        #   data is a (data_wt, data_mt) tuple, like 'idr'.
        # ------------------------------------------------------------------
        if task == "three_state":
            data_wt, data_mt = data
            z_wt = self._encode(data_wt)
            z_mt = self._encode(data_mt)
            return self.three_state_head(z_wt, z_mt)   # (B, 3) logits

        # ------------------------------------------------------------------
        # Task: idr  [V4.0 — True Siamese Pass]
        #
        # data must be a (data_wt, data_mt) tuple.
        # Each is passed through the shared backbone independently.
        # The resulting embeddings (z_wt, z_mt) are fed into SiameseStabilityHead
        # in BOTH directions.  This exposes the antisymmetry constraint that
        # AntisymmetricLoss enforces during training.
        #
        # Biological motivation:
        #   ΔΔG(WT→MT) should equal −ΔΔG(MT→WT).  The Siamese head computes
        #   the difference vector (z_mt − z_wt), so calling it in both orders
        #   gives predictions that AntisymmetricLoss can penalise for asymmetry.
        # ------------------------------------------------------------------
        elif task == "idr":
            data_wt, data_mt = data   # unpack tuple

            # Independent backbone passes — each sees its own attention graph
            z_wt = self._encode(data_wt)   # (B, 320)
            z_mt = self._encode(data_mt)   # (B, 320)

            # Forward direction:  WT → MT   (predicts ΔΔG)
            dG_wt_to_mt = self.siamese_head(z_wt, z_mt)   # (B, 1)

            # Reverse direction: MT → WT   (should be −dG_wt_to_mt)
            dG_mt_to_wt = self.siamese_head(z_mt, z_wt)   # (B, 1)

            return dG_wt_to_mt, dG_mt_to_wt

        # ------------------------------------------------------------------
        # Default: return raw graph embedding
        # ------------------------------------------------------------------
        return self._encode(data)